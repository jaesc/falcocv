import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from PyQt6.QtCore import QTimer, Qt

from falcocv.core.vision_image import VisionImage
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.style import SUCCESS, WARNING, disabled_css
from falcocv.ui.tool_config_helpers import (
    configure_params_splitter,
    connect_execution_time,
    create_execution_footer,
    save_params_splitter,
    wrap_params_scroll,
)
from falcocv.ui.widgets.roi_editor import RoiEditorWidget
from falcocv.ui.widgets.viewer import ImageViewer


class ShapeMatchConfig(QWidget):
    def __init__(self, tool):
        super().__init__()
        self.tool = tool
        self._picking_origin = False
        self.tool._active_roi_key = None

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.content_layout = QSplitter(Qt.Orientation.Horizontal)

        self.left_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.left_panel)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)

        self.train_roi_editor = RoiEditorWidget("Training ROI", param_key="train_roi")
        self.search_roi_editor = RoiEditorWidget("Search ROI", param_key="search_roi")
        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_search_tab(), "Search")
        self.tabs.addTab(self._build_training_tab(), "Training")
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.controls_layout.addWidget(self.tabs)

        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)

        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.record_selector = QComboBox()
        for key in ("input image", "train image", "output image"):
            self.record_selector.addItem(key, key)
        self.record_selector.currentIndexChanged.connect(self.on_record_changed)
        self.viewer_layout.addWidget(self.record_selector)
        self.viewer = ImageViewer()
        self.viewer_layout.addWidget(self.viewer)
        self.content_layout.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "ShapeMatch"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        self.footer, self.label_time = create_execution_footer(self)
        connect_execution_time(self.tool, self.label_time)
        self.main_layout.addWidget(self.footer)

        def train_image_source():
            if self.tool._static_train_image is None:
                return None
            fixture = self.tool._static_train_fixture
            return VisionImage(
                self.tool._static_train_image,
                x=fixture.get("x", 0.0),
                y=fixture.get("y", 0.0),
                angle=fixture.get("angle", 0.0),
            )

        self.train_roi_editor.set_image_getter(train_image_source)
        self.search_roi_editor.set_image_getter(
            lambda: self.tool.inputs.get("input image")
        )
        self.train_roi_editor.set_tool(self.tool, self.viewer)
        self.search_roi_editor.set_tool(self.tool, self.viewer)
        self.train_roi_editor.btn_edit.toggled.connect(self._on_train_edit_toggled)
        self.search_roi_editor.btn_edit.toggled.connect(self._on_search_edit_toggled)
        self.template_viewer.mouse_clicked.connect(self._on_template_clicked)

        self._exec_timer = QTimer()
        self._exec_timer.setSingleShot(True)
        self._exec_timer.setInterval(250)
        self._exec_timer.timeout.connect(self.tool.mark_dirty)

        self.tool.data_changed.connect(self.refresh_ui)
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))
        self.refresh_ui()

    # ------------------------------------------------------------------
    # Tabs
    # ------------------------------------------------------------------
    def _build_training_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.train_roi_editor)

        capture = QPushButton("Capture Train Image")
        capture.clicked.connect(self._on_capture)
        layout.addWidget(capture)

        train = QPushButton("Train ShapeMatch")
        train.clicked.connect(self._on_train)
        layout.addWidget(train)

        self.label_state = QLabel("Not trained")
        self.label_state.setWordWrap(True)
        layout.addWidget(self.label_state)

        origin_group = QGroupBox("Tool Origin Offset")
        origin_layout = QVBoxLayout(origin_group)
        origin_row = QHBoxLayout()
        self.spin_ox = QDoubleSpinBox()
        self.spin_ox.setRange(-10000.0, 10000.0)
        self.spin_ox.setDecimals(2)
        self.spin_ox.setValue(float(self.tool.params.get("origin_x", 0.0)))
        self.spin_oy = QDoubleSpinBox()
        self.spin_oy.setRange(-10000.0, 10000.0)
        self.spin_oy.setDecimals(2)
        self.spin_oy.setValue(float(self.tool.params.get("origin_y", 0.0)))
        self.spin_ox.valueChanged.connect(self._on_origin_changed)
        self.spin_oy.valueChanged.connect(self._on_origin_changed)
        origin_row.addWidget(QLabel("X:"))
        origin_row.addWidget(self.spin_ox)
        origin_row.addWidget(QLabel("Y:"))
        origin_row.addWidget(self.spin_oy)
        origin_layout.addLayout(origin_row)

        self.btn_pick_origin = QPushButton("Pick Origin Point (Click on Template)")
        self.btn_pick_origin.setCheckable(True)
        self.btn_pick_origin.clicked.connect(self._on_pick_origin_clicked)
        origin_layout.addWidget(self.btn_pick_origin)
        layout.addWidget(origin_group)

        preview_group = QGroupBox("Shape Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.template_viewer = ImageViewer()
        self.template_viewer.setMinimumSize(250, 250)
        preview_layout.addWidget(self.template_viewer)
        layout.addWidget(preview_group)
        layout.addStretch()
        return panel

    def _build_search_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.search_roi_editor)

        confidence_group = QGroupBox("Sensitivity")
        confidence_layout = QVBoxLayout(confidence_group)
        self.spin_score = QDoubleSpinBox()
        self.spin_score.setRange(0.05, 1.0)
        self.spin_score.setSingleStep(0.05)
        self.spin_score.setValue(float(self.tool.params.get("min_score", 0.45)))
        self.spin_score.valueChanged.connect(lambda value: self._set_param("min_score", value))
        confidence_layout.addWidget(QLabel("Confidence"))
        confidence_layout.addWidget(self.spin_score)
        layout.addWidget(confidence_group)

        angle_group = QGroupBox("Angle Range")
        angle_layout = QHBoxLayout(angle_group)
        self.spin_angle_start = QDoubleSpinBox()
        self.spin_angle_start.setRange(-180.0, 180.0)
        self.spin_angle_start.setDecimals(2)
        self.spin_angle_start.setValue(float(self.tool.params.get("angle_start", -30.0)))
        self.spin_angle_start.valueChanged.connect(
            lambda value: self._set_param("angle_start", value)
        )
        self.spin_angle_end = QDoubleSpinBox()
        self.spin_angle_end.setRange(-180.0, 180.0)
        self.spin_angle_end.setDecimals(2)
        self.spin_angle_end.setValue(float(self.tool.params.get("angle_end", 30.0)))
        self.spin_angle_end.valueChanged.connect(
            lambda value: self._set_param("angle_end", value)
        )
        angle_layout.addWidget(QLabel("Min:"))
        angle_layout.addWidget(self.spin_angle_start)
        angle_layout.addWidget(QLabel("Max:"))
        angle_layout.addWidget(self.spin_angle_end)
        layout.addWidget(angle_group)

        scale_group = QGroupBox("Scale Range")
        scale_layout = QHBoxLayout(scale_group)
        self.spin_scale_start = QDoubleSpinBox()
        self.spin_scale_start.setRange(0.05, 5.0)
        self.spin_scale_start.setDecimals(3)
        self.spin_scale_start.setSingleStep(0.01)
        self.spin_scale_start.setValue(float(self.tool.params.get("scale_start", 0.9)))
        self.spin_scale_start.valueChanged.connect(
            lambda value: self._set_param("scale_start", value)
        )
        self.spin_scale_end = QDoubleSpinBox()
        self.spin_scale_end.setRange(0.05, 5.0)
        self.spin_scale_end.setDecimals(3)
        self.spin_scale_end.setSingleStep(0.01)
        self.spin_scale_end.setValue(float(self.tool.params.get("scale_end", 1.1)))
        self.spin_scale_end.valueChanged.connect(
            lambda value: self._set_param("scale_end", value)
        )
        scale_layout.addWidget(QLabel("Min:"))
        scale_layout.addWidget(self.spin_scale_start)
        scale_layout.addWidget(QLabel("Max:"))
        scale_layout.addWidget(self.spin_scale_end)
        layout.addWidget(scale_group)

        results_group = QGroupBox("Expected Results")
        results_layout = QVBoxLayout(results_group)
        self.spin_max_matches = QSpinBox()
        self.spin_max_matches.setRange(0, 100)
        self.spin_max_matches.setSpecialValueText("All")
        self.spin_max_matches.setValue(int(self.tool.params.get("max_matches", 1)))
        self.spin_max_matches.valueChanged.connect(
            lambda value: self._set_param("max_matches", value)
        )
        results_layout.addWidget(QLabel("Number of matches (0 = All)"))
        results_layout.addWidget(self.spin_max_matches)
        self.chk_fixture = QCheckBox("Act as Fixture (Anchor)")
        self.chk_fixture.setChecked(bool(self.tool.params.get("act_as_fixture", False)))
        self.chk_fixture.toggled.connect(
            lambda checked: self._set_param("act_as_fixture", checked)
        )
        results_layout.addWidget(self.chk_fixture)
        layout.addWidget(results_group)

        visibility_group = QGroupBox("Visibility")
        visibility_layout = QVBoxLayout(visibility_group)
        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(bool(self.tool.params.get("show_overlays", True)))
        self.chk_overlays.toggled.connect(lambda checked: self._on_overlays_toggled(checked))
        visibility_layout.addWidget(self.chk_overlays)
        for key, label in (
            ("show_search_roi", "Show Search Region"),
            ("show_found_region", "Show Found Region"),
            ("show_found_origin", "Show Found Origin"),
            ("show_labels", "Show Labels"),
            ("copy_overlays_to_input", "Copy overlays to input image"),
        ):
            check = QCheckBox(label)
            check.setChecked(bool(self.tool.params.get(key, True)))
            check.toggled.connect(
                lambda checked, k=key: self._set_param(k, checked)
            )
            visibility_layout.addWidget(check)
        layout.addWidget(visibility_group)
        layout.addStretch()
        return panel

    # ------------------------------------------------------------------
    # Actions
    # ------------------------------------------------------------------
    def _set_param(self, key, value):
        self.tool.params[key] = value
        self._exec_timer.start()

    def _on_origin_changed(self):
        self.tool.params["origin_x"] = float(self.spin_ox.value())
        self.tool.params["origin_y"] = float(self.spin_oy.value())
        self._exec_timer.start()
        self.refresh_ui()

    def _on_overlays_toggled(self, checked):
        self.tool.params["show_overlays"] = bool(checked)
        self._exec_timer.start()
        self.refresh_ui()

    def _on_capture(self):
        self._stop_all_editing()
        if self.tool.inputs.get("input image") is None:
            self.label_state.setText("No input image connected.")
            return
        self.tool.save_train_image()
        self.tool.mark_dirty()
        self.record_selector.setCurrentText("train image")
        self.refresh_ui()

    def _on_train(self):
        self._stop_all_editing()
        try:
            roi = list(self.tool.params.get("train_roi", [0, 0, 0, 0, 0]))
            self.tool.train_from_roi(*roi)
            self.tool.mark_dirty()
            self.refresh_ui(auto_fit=True)
        except Exception as exc:
            self.label_state.setText(f"Training failed: {exc}")
            self.label_state.setStyleSheet(f"color: {WARNING}; font-weight: bold;")

    def _on_pick_origin_clicked(self, checked):
        self._picking_origin = bool(checked)
        if checked:
            self.template_viewer.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.template_viewer.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_template_clicked(self, x, y):
        if not self._picking_origin or not self.tool.is_trained:
            return
        width, height = self.tool._template_size
        origin_x = x - width / 2.0
        origin_y = y - height / 2.0
        self.spin_ox.blockSignals(True)
        self.spin_oy.blockSignals(True)
        self.spin_ox.setValue(origin_x)
        self.spin_oy.setValue(origin_y)
        self.spin_ox.blockSignals(False)
        self.spin_oy.blockSignals(False)
        self.tool.params["origin_x"] = float(origin_x)
        self.tool.params["origin_y"] = float(origin_y)
        self._exec_timer.start()
        self.btn_pick_origin.setChecked(False)
        self._on_pick_origin_clicked(False)
        self.refresh_ui()

    def _on_tab_changed(self, _index):
        self._stop_all_editing()
        self.refresh_ui()

    def on_record_changed(self, _index=None):
        self.refresh_ui()

    def _stop_all_editing(self):
        self.train_roi_editor.btn_edit.setChecked(False)
        self.search_roi_editor.btn_edit.setChecked(False)
        self.viewer.set_roi_editing(False)

    def _on_train_edit_toggled(self, checked):
        if checked:
            self.search_roi_editor.btn_edit.setChecked(False)
        self.viewer.set_drawing_new_roi(checked)

    def _on_search_edit_toggled(self, checked):
        if checked:
            self.train_roi_editor.btn_edit.setChecked(False)
        self.viewer.set_drawing_new_roi(checked)

    # ------------------------------------------------------------------
    # Refresh
    # ------------------------------------------------------------------
    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if key is None:
            return

        if key == "output image":
            image_data = self.tool.outputs.get("output image")
        elif key == "train image":
            image_data = self.tool._static_train_image
        else:
            image_data = self.tool.inputs.get("input image")
        image = image_data.data if isinstance(image_data, VisionImage) else image_data

        self.viewer.set_image(
            image_data,
            auto_fit=auto_fit,
            render_overlays_flag=self.chk_overlays.isChecked(),
        )

        roi_key = None
        if isinstance(image, np.ndarray):
            if key == "train image":
                roi_key = "train_roi"
            elif key == "input image":
                roi_key = "search_roi"

        self.tool._active_roi_key = roi_key
        if roi_key is not None:
            roi_data = list(self.tool.params.get(roi_key, [0, 0, 0, 0, 0]))
            if len(roi_data) == 4:
                roi_data = roi_data + [0.0]
            if roi_data[2] > 0 and roi_data[3] > 0:
                if key == "train image":
                    fixture = self.tool._static_train_fixture
                    v_img = VisionImage(
                        image,
                        x=fixture.get("x", 0.0),
                        y=fixture.get("y", 0.0),
                        angle=fixture.get("angle", 0.0),
                    )
                else:
                    input_img = self.tool.inputs.get("input image")
                    v_img = VisionImage.from_array(input_img) if input_img is not None else None
                absolute_roi = v_img.fixture_to_image(roi_data) if v_img is not None else roi_data
                self.viewer.set_roi(absolute_roi)
            else:
                self.viewer.set_roi(None)
        else:
            self.viewer.set_roi(None)

        self.train_roi_editor.update_from_params()
        self.search_roi_editor.update_from_params()

        disabled = disabled_css()
        is_train = key == "train image"
        self.train_roi_editor.setEnabled(is_train)
        self.train_roi_editor.setStyleSheet(disabled if not is_train else "")
        if not is_train:
            self.train_roi_editor.btn_edit.setChecked(False)

        is_search = key == "input image"
        self.search_roi_editor.setEnabled(is_search)
        self.search_roi_editor.setStyleSheet(disabled if not is_search else "")
        if not is_search:
            self.search_roi_editor.btn_edit.setChecked(False)

        if self.tool.is_trained:
            self.label_state.setText(
                f"Trained: {self.tool._model_x.size} features, "
                f"{self.tool._template_size[0]}x{self.tool._template_size[1]} px"
            )
            self.label_state.setStyleSheet(f"color: {SUCCESS}; font-weight: bold;")
            self._update_template_preview()
        else:
            self.label_state.setText("Not trained")
            self.label_state.setStyleSheet("")
            self.template_viewer.set_image(None)

    def _update_template_preview(self):
        if self.tool._template_color is None:
            self.template_viewer.set_image(None)
            return
        template = self.tool._template_color.copy()
        height, width = template.shape[:2]
        edges = self.tool._template_edges
        if edges is not None:
            template[edges > 0] = (255, 255, 0)

        count = min(int(self.tool._model_x.size), 200)
        for index in range(count):
            px = int(round(float(self.tool._model_x[index]) + width / 2.0))
            py = int(round(float(self.tool._model_y[index]) + height / 2.0))
            if not (0 <= px < width and 0 <= py < height):
                continue
            theta = float(self.tool._model_theta[index])
            ex = int(round(px + 4.0 * np.cos(theta)))
            ey = int(round(py + 4.0 * np.sin(theta)))
            cv2.line(template, (px, py), (ex, ey), (0, 255, 255), 1)
            cv2.circle(template, (px, py), 1, (0, 255, 0), -1)

        preview = VisionImage.from_array(template)
        layer = preview.get_overlay_layer("default", source_tool=self.tool.name)
        origin_x = width / 2.0 + float(self.tool.params.get("origin_x", 0.0))
        origin_y = height / 2.0 + float(self.tool.params.get("origin_y", 0.0))
        layer.add_axes(
            (int(round(origin_x)), int(round(origin_y))),
            angle=0,
            length=30,
            color_x=(0, 0, 255),
            color_y=(0, 255, 0),
            thickness=2,
        )
        layer.add_point(
            (int(round(origin_x)), int(round(origin_y))),
            radius=4,
            color=(255, 255, 255),
        )
        self.template_viewer.set_image(preview, auto_fit=True, render_overlays_flag=True)

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "ShapeMatch")
