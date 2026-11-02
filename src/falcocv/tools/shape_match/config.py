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
    add_min_results_group,
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
        self._syncing_training = False
        self._preview_scale_x = 1.0
        self._preview_scale_y = 1.0
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

        features_group = QGroupBox("Feature Extraction")
        features_layout = QVBoxLayout(features_group)

        edge_row = QHBoxLayout()
        edge_row.addWidget(QLabel("Canny low:"))
        self.spin_edge_low = QSpinBox()
        self.spin_edge_low.setRange(0, 254)
        self.spin_edge_low.setValue(int(self.tool.params.get("edge_low", 50)))
        edge_row.addWidget(self.spin_edge_low)
        edge_row.addWidget(QLabel("high:"))
        self.spin_edge_high = QSpinBox()
        self.spin_edge_high.setRange(1, 255)
        self.spin_edge_high.setValue(int(self.tool.params.get("edge_high", 150)))
        edge_row.addWidget(self.spin_edge_high)
        features_layout.addLayout(edge_row)

        gradient_row = QHBoxLayout()
        gradient_row.addWidget(QLabel("Min gradient:"))
        self.spin_min_gradient = QDoubleSpinBox()
        self.spin_min_gradient.setRange(0.0, 5000.0)
        self.spin_min_gradient.setDecimals(1)
        self.spin_min_gradient.setSingleStep(5.0)
        self.spin_min_gradient.setValue(
            float(self.tool.params.get("min_gradient", 0.0))
        )
        gradient_row.addWidget(self.spin_min_gradient)
        gradient_row.addStretch()
        features_layout.addLayout(gradient_row)

        features_row = QHBoxLayout()
        features_row.addWidget(QLabel("Max features:"))
        self.spin_max_features = QSpinBox()
        self.spin_max_features.setRange(20, 5000)
        self.spin_max_features.setSingleStep(50)
        self.spin_max_features.setValue(
            int(self.tool.params.get("max_features", 800))
        )
        features_row.addWidget(self.spin_max_features)
        features_row.addStretch()
        features_layout.addLayout(features_row)

        self.label_features = QLabel("")
        self.label_features.setWordWrap(True)
        features_layout.addWidget(self.label_features)

        self.btn_edit_features = QPushButton("Edit Features...")
        self.btn_edit_features.clicked.connect(self._on_edit_features)
        features_layout.addWidget(self.btn_edit_features)
        layout.addWidget(features_group)

        mask_group = QGroupBox("Training Mask")
        mask_layout = QVBoxLayout(mask_group)
        self.label_mask = QLabel("Mask: none")
        self.label_mask.setWordWrap(True)
        mask_layout.addWidget(self.label_mask)
        mask_row = QHBoxLayout()
        self.btn_edit_mask = QPushButton("Edit Mask...")
        self.btn_edit_mask.clicked.connect(self._on_edit_mask)
        self.btn_clear_mask = QPushButton("Clear Mask")
        self.btn_clear_mask.clicked.connect(self._on_clear_mask)
        mask_row.addWidget(self.btn_edit_mask)
        mask_row.addWidget(self.btn_clear_mask)
        mask_row.addStretch()
        mask_layout.addLayout(mask_row)
        layout.addWidget(mask_group)

        for widget in (
            self.spin_edge_low,
            self.spin_edge_high,
            self.spin_min_gradient,
            self.spin_max_features,
        ):
            widget.valueChanged.connect(self._on_training_param_changed)

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
        self.template_viewer.setFixedSize(250, 250)
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
        self.spin_angle_start.setToolTip(
            "Positive angles rotate clockwise in the image frame, "
            "like ROI/fixture angles."
        )
        self.spin_angle_start.setValue(float(self.tool.params.get("angle_start", -30.0)))
        self.spin_angle_start.valueChanged.connect(
            lambda value: self._set_param("angle_start", value)
        )
        self.spin_angle_end = QDoubleSpinBox()
        self.spin_angle_end.setRange(-180.0, 180.0)
        self.spin_angle_end.setDecimals(2)
        self.spin_angle_end.setToolTip(
            "Positive angles rotate clockwise in the image frame, "
            "like ROI/fixture angles."
        )
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

        pre_gate_group = QGroupBox("Pre-gate scorer")
        pre_gate_layout = QVBoxLayout(pre_gate_group)
        self.combo_pre_gate = QComboBox()
        self.combo_pre_gate.addItem("Grayscale NCC (stable)", "ncc")
        self.combo_pre_gate.addItem(
            "Gradient alignment (prototype)", "alignment"
        )
        index = self.combo_pre_gate.findData(
            str(self.tool.params.get("pre_gate_scorer", "ncc"))
        )
        self.combo_pre_gate.setCurrentIndex(index if index >= 0 else 0)
        self.combo_pre_gate.setToolTip(
            "Candidate ranking before the full NCC scan. "
            "Grayscale NCC uses appearance; gradient alignment uses only "
            "scene/model gradients. Alignment is a prototype pending "
            "real-job validation."
        )
        self.combo_pre_gate.currentIndexChanged.connect(
            self._on_pre_gate_changed
        )
        pre_gate_layout.addWidget(QLabel("Candidate ranking"))
        pre_gate_layout.addWidget(self.combo_pre_gate)
        layout.addWidget(pre_gate_group)

        mode_group = QGroupBox("Matching mode")
        mode_layout = QVBoxLayout(mode_group)
        self.combo_match_mode = QComboBox()
        self.combo_match_mode.addItem("Hybrid (NCC/ECC)", "hybrid")
        self.combo_match_mode.addItem(
            "Geometry only (prototype)", "geometry"
        )
        mode_index = self.combo_match_mode.findData(
            str(self.tool.params.get("match_mode", "geometry"))
        )
        self.combo_match_mode.setCurrentIndex(
            mode_index if mode_index >= 0 else 0
        )
        self.combo_match_mode.setToolTip(
            "Hybrid uses grayscale NCC/ECC verification. "
            "Geometry only uses edge distance + gradient alignment + ICP; "
            "experimental and under validation."
        )
        self.combo_match_mode.currentIndexChanged.connect(
            self._on_match_mode_changed
        )
        mode_layout.addWidget(QLabel("Verification"))
        mode_layout.addWidget(self.combo_match_mode)
        layout.addWidget(mode_group)

        results_group = QGroupBox("Expected Results")
        results_layout = QVBoxLayout(results_group)
        self.spin_max_matches = QSpinBox()
        self.spin_max_matches.setRange(0, 100)
        self.spin_max_matches.setSpecialValueText("All")
        self.spin_max_matches.setValue(int(self.tool.params.get("max_matches", 1)))
        self.spin_max_matches.setToolTip(
            "Maximum number of matches to return (0 = all). "
            "This does not control the minimum-result error policy."
        )
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

        self.spin_min_results = add_min_results_group(
            layout, self.tool, "Minimum matches:", 100, "matches"
        )

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

    def _on_pre_gate_changed(self, _index=None):
        value = self.combo_pre_gate.currentData()
        if value is not None:
            self._set_param("pre_gate_scorer", str(value))

    def _sync_pre_gate_control(self):
        value = str(self.tool.params.get("pre_gate_scorer", "ncc"))
        index = self.combo_pre_gate.findData(value)
        if index >= 0 and index != self.combo_pre_gate.currentIndex():
            self.combo_pre_gate.blockSignals(True)
            self.combo_pre_gate.setCurrentIndex(index)
            self.combo_pre_gate.blockSignals(False)
        self.combo_pre_gate.setEnabled(
            str(self.tool.params.get("match_mode", "geometry")) != "geometry"
        )

    def _on_match_mode_changed(self, _index=None):
        value = self.combo_match_mode.currentData()
        if value is not None:
            self._set_param("match_mode", str(value))
            self._sync_pre_gate_control()

    def _sync_match_mode_control(self):
        value = str(self.tool.params.get("match_mode", "geometry"))
        index = self.combo_match_mode.findData(value)
        if index >= 0 and index != self.combo_match_mode.currentIndex():
            self.combo_match_mode.blockSignals(True)
            self.combo_match_mode.setCurrentIndex(index)
            self.combo_match_mode.blockSignals(False)

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
            self.tool.refresh_stale_state()
            self.label_state.setText(f"Training failed: {exc}")
            self.label_state.setStyleSheet(f"color: {WARNING}; font-weight: bold;")
            self.label_features.setText("")
            self._update_mask_status()

    def _on_training_param_changed(self):
        if self._syncing_training:
            return
        self.tool.params["edge_low"] = int(self.spin_edge_low.value())
        self.tool.params["edge_high"] = int(self.spin_edge_high.value())
        self.tool.params["min_gradient"] = float(self.spin_min_gradient.value())
        self.tool.params["max_features"] = int(self.spin_max_features.value())
        self.tool.mark_dirty()

    def _sync_training_controls(self):
        self._syncing_training = True
        try:
            self.spin_edge_low.setValue(int(self.tool.params.get("edge_low", 50)))
            self.spin_edge_high.setValue(int(self.tool.params.get("edge_high", 150)))
            self.spin_min_gradient.setValue(
                float(self.tool.params.get("min_gradient", 0.0))
            )
            self.spin_max_features.setValue(
                int(self.tool.params.get("max_features", 800))
            )
        finally:
            self._syncing_training = False

    def _on_edit_mask(self):
        if self.tool._static_train_image is None:
            self.label_state.setText("Capture a training image before editing the mask.")
            self.label_state.setStyleSheet(f"color: {WARNING}; font-weight: bold;")
            return
        try:
            crop = self.tool.training_crop()
            mask = self.tool.get_train_mask()
            from .mask_editor import ShapeMatchMaskDialog

            dialog = ShapeMatchMaskDialog(crop, mask, self)
            if dialog.exec():
                self.tool.set_train_mask(dialog.mask)
                self.tool.mark_dirty()
                self.refresh_ui()
        except Exception as exc:
            self.label_state.setText(f"Mask failed: {exc}")
            self.label_state.setStyleSheet(f"color: {WARNING}; font-weight: bold;")

    def _on_clear_mask(self):
        try:
            self.tool.clear_train_mask()
            self.tool.mark_dirty()
            self.refresh_ui()
        except Exception as exc:
            self.label_state.setText(f"Mask failed: {exc}")
            self.label_state.setStyleSheet(f"color: {WARNING}; font-weight: bold;")

    def _on_edit_features(self):
        if not self.tool.is_trained or self.tool.is_model_stale:
            self.label_state.setText(
                "Retrain ShapeMatch before editing features."
            )
            self.label_state.setStyleSheet(f"color: {WARNING}; font-weight: bold;")
            return
        try:
            from .feature_editor import ShapeMatchFeatureEditorDialog

            dialog = ShapeMatchFeatureEditorDialog(self.tool, self)
            if dialog.exec():
                self.tool.mark_dirty()
                self.refresh_ui()
        except Exception as exc:
            self.label_state.setText(f"Feature editor failed: {exc}")
            self.label_state.setStyleSheet(f"color: {WARNING}; font-weight: bold;")

    def _update_mask_status(self):
        has_image = self.tool._static_train_image is not None
        has_mask = self.tool.has_train_mask
        self.btn_edit_mask.setEnabled(has_image)
        self.btn_clear_mask.setEnabled(has_image and has_mask)
        if not has_mask:
            self.label_mask.setText("Mask: none")
        else:
            ignored = self.tool.train_mask_ignored_fraction() * 100.0
            self.label_mask.setText(f"Mask: {ignored:.1f}% ignored")

    def _on_pick_origin_clicked(self, checked):
        self._picking_origin = bool(checked)
        if checked:
            self.template_viewer.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.template_viewer.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_template_clicked(self, x, y):
        if not self._picking_origin or not self.tool.is_trained:
            return
        x = float(x) * self._preview_scale_x
        y = float(y) * self._preview_scale_y
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

        self._sync_training_controls()
        self._sync_match_mode_control()
        self._sync_pre_gate_control()
        self.tool.refresh_stale_state()
        self._update_mask_status()
        self.btn_edit_features.setEnabled(
            self.tool.is_trained and not self.tool.is_model_stale
        )

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

        if self.tool._model_load_error:
            self.label_state.setText(self.tool._model_load_error)
            self.label_state.setStyleSheet(f"color: {WARNING}; font-weight: bold;")
            self.label_features.setText("")
            self.template_viewer.set_image(None)
        elif self.tool.is_model_stale:
            self.label_state.setText(
                "STALE — training parameters changed. Retrain ShapeMatch."
            )
            self.label_state.setStyleSheet(f"color: {WARNING}; font-weight: bold;")
            self._update_feature_status()
            self._update_template_preview()
        elif self.tool.is_trained:
            suffix = " (manual)" if self.tool.is_manual_edit else ""
            self.label_state.setText(
                f"Trained: {self.tool._model_x.size} features, "
                f"{self.tool._template_size[0]}x{self.tool._template_size[1]} px"
                f"{suffix}"
            )
            self.label_state.setStyleSheet(f"color: {SUCCESS}; font-weight: bold;")
            self._update_feature_status()
            self._update_template_preview()
        else:
            self.label_state.setText("Not trained")
            self.label_state.setStyleSheet("")
            self.label_features.setText("")
            self.template_viewer.set_image(None)

    def _update_feature_status(self):
        count = int(self.tool._model_x.size)
        low = int(getattr(self.tool, "_model_edge_low", 50))
        high = int(getattr(self.tool, "_model_edge_high", 150))
        canny = int(getattr(self.tool, "_model_canny_count", 0))
        text = f"{count} features · Canny {low}/{high} · {canny} edges"
        if getattr(self.tool, "_model_canny_fallback", False):
            text += " · fallback: top 10 % Sobel"
        self.label_features.setText(text)

    def _update_template_preview(self):
        source = self.tool._template_color
        if source is None:
            self.template_viewer.set_image(None)
            return

        height, width = source.shape[:2]
        scale = min(250.0 / width, 250.0 / height, 2.0)
        thumb_w = max(1, int(round(width * scale)))
        thumb_h = max(1, int(round(height * scale)))
        interpolation = (
            cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        )
        thumb = cv2.resize(
            source, (thumb_w, thumb_h), interpolation=interpolation
        )
        thumb = (thumb.astype(np.float32) * 0.60).astype(np.uint8)

        self._preview_scale_x = width / float(thumb_w)
        self._preview_scale_y = height / float(thumb_h)
        sx = thumb_w / float(width)
        sy = thumb_h / float(height)

        count = int(self.tool._model_x.size)
        for index in range(count):
            px = int(round(
                (float(self.tool._model_x[index]) + width / 2.0) * sx
            ))
            py = int(round(
                (float(self.tool._model_y[index]) + height / 2.0) * sy
            ))
            if 0 <= px < thumb_w and 0 <= py < thumb_h:
                cv2.circle(thumb, (px, py), 1, (0, 255, 255), -1)

        if self.tool.is_model_stale:
            cv2.putText(
                thumb,
                "STALE - RETRAIN",
                (4, 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        preview = VisionImage.from_array(thumb)
        layer = preview.get_overlay_layer("default", source_tool=self.tool.name)
        origin_x = (
            width / 2.0 + float(self.tool.params.get("origin_x", 0.0))
        ) * sx
        origin_y = (
            height / 2.0 + float(self.tool.params.get("origin_y", 0.0))
        ) * sy
        layer.add_axes(
            (int(round(origin_x)), int(round(origin_y))),
            angle=0,
            length=20,
            color_x=(0, 0, 255),
            color_y=(0, 255, 0),
            thickness=2,
        )
        layer.add_point(
            (int(round(origin_x)), int(round(origin_y))),
            radius=3,
            color=(255, 255, 255),
        )
        self.template_viewer.set_image(
            preview, auto_fit=True, render_overlays_flag=True
        )

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "ShapeMatch")
