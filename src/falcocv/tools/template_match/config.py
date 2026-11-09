import cv2
import numpy as np
import math
import base64
from falcocv.core.tool_base import BaseTool, ToolResult, ToolResults
from falcocv.core.roi_utils import extract_roi
from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QGridLayout,
    QLabel,
    QDoubleSpinBox,
    QSizePolicy,
    QSpinBox,
    QPushButton,
    QGroupBox,
    QTabWidget,
    QCheckBox,
    QSplitter,
)
from PyQt6.QtCore import QRectF, QTimer, Qt, QThread, pyqtSignal as Signal
from falcocv.core.config import ConfigManager
from falcocv.ui.style import ALT_BG, BORDER, SUCCESS, WARNING, disabled_css
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    add_fixture_group,
    add_record_header,
    make_record_selector,
    add_min_results_group,
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
    connect_execution_time,
)
from falcocv.ui.widgets.viewer import ImageViewer


# --- Background cache builder -------------------------------------------
from .logic import _CacheWorker, _transform, TemplateMatchResult, TemplateMatch
class _CacheWorker(QThread):
    """Builds the template cache in a background thread."""
    finished = Signal(dict, dict)  # (cache, cache_params)

    def __init__(self, template_gray, params, template_size, existing_cache=None,
                 angle_offset=0.0):
        super().__init__()
        self._template_gray = template_gray
        self._params = params
        self._template_size = template_size
        self._existing_cache = existing_cache or {}
        self._angle_offset = float(angle_offset)

    def run(self):
        try:
            levels  = int(self._params["pyramid_levels"])
            angle_step = max(0.1, float(self._params["angle_step"]))
            angles  = list(np.arange(
                float(self._params["angle_start"]) - self._angle_offset,
                float(self._params["angle_end"]) - self._angle_offset + 0.001,
                angle_step,
            ))
            scales  = np.arange(self._params["scale_start"],
                                self._params["scale_end"] + 0.001,
                                max(0.01, self._params["scale_step"])).tolist()

            # Compute expected key set, rounding floats for stability
            expected = set()
            for s in scales:
                for a in angles:
                    for l in range(levels):
                        expected.add((round(s, 6), round(a, 6), l))

            # Start with existing cache, remove stale keys
            cache = dict(self._existing_cache)
            stale = [k for k in cache if k not in expected]
            for k in stale:
                del cache[k]

            # Only compute missing keys
            for s in scales:
                for a in angles:
                    key_0 = (round(s, 6), round(a, 6), 0)
                    if key_0 not in cache:
                        # Public clockwise angle; OpenCV rotates the other way.
                        t_base = _transform(self._template_gray, -a, s)
                        cache[key_0] = t_base
                    else:
                        t_base = cache[key_0]
                    curr = t_base
                    for l in range(1, levels):
                        key = (round(s, 6), round(a, 6), l)
                        if key not in cache:
                            curr = cv2.pyrDown(curr)
                            cache[key] = curr
                        else:
                            curr = cache[key]

            self._cache = cache

            cache_params = {
                "angle_start":   self._params["angle_start"],
                "angle_end":     self._params["angle_end"],
                "angle_step":    self._params["angle_step"],
                "scale_start":   self._params["scale_start"],
                "scale_end":     self._params["scale_end"],
                "scale_step":    self._params["scale_step"],
                "pyramid_levels": self._params["pyramid_levels"],
                "template_size": self._template_size,
                "angle_offset":  self._angle_offset,
            }
            self._cache_params = cache_params
            self.finished.emit(cache, cache_params)
        except Exception:
            self._cache = {}
            self._cache_params = {}
            self.finished.emit({}, {})


def _transform(template, angle, scale):
    """Rotate + scale a template image, auto-expanding the canvas.

    ``angle`` is an OpenCV angle (positive = counter-clockwise).  Callers that
    hold a public/FalcoCV angle (positive = clockwise) must negate it.
    """
    if angle == 0 and scale == 1.0:
        return template
    h, w = template.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    cos_v, sin_v = abs(M[0, 0]), abs(M[0, 1])
    new_w = int((h * sin_v) + (w * cos_v))
    new_h = int((h * cos_v) + (w * sin_v))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(template, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
# -------------------------------------------------------------------------


class TemplateMatchResult(ToolResult):
    def __init__(self, id, x, y, angle, scale, score):
        super().__init__(id=id, x=x, y=y, angle=angle, scale=scale, score=score)


class TemplateMatchConfig(QWidget):
    def __init__(self, tool):
        super().__init__()
        self.tool = tool
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self._picking_origin = False
        self.tool._active_roi_key = None

        self.content_layout = QSplitter(Qt.Orientation.Horizontal)

        self.left_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.left_panel)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        from falcocv.ui.widgets.roi_editor import RoiEditorWidget
        self.train_roi_editor = RoiEditorWidget("Training ROI", param_key="train_roi")
        self.search_roi_editor = RoiEditorWidget("Search ROI", param_key="search_roi")
        self.tabs = QTabWidget(); self.tabs.addTab(self._build_search_tab(), "Search"); self.tabs.addTab(self._build_training_tab(), "Training")
        self.tabs.currentChanged.connect(self._on_tab_changed); self.controls_layout.addWidget(self.tabs)
        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)

        # Right side: view selector (anchored to viewer width) + viewer
        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.record_selector = make_record_selector()
        [self.record_selector.addItem(k, k) for k in ["input image", "train image", "output image"]]
        self.record_selector.currentIndexChanged.connect(self.on_record_changed)
        add_record_header(self.viewer_layout, self.record_selector, tool=self.tool)
        self.viewer = ImageViewer(); self.viewer_layout.addWidget(self.viewer)
        self.content_layout.addWidget(self.right_panel)

        # The old layout forced a wide left panel and saved that width.  Reset
        # it once so the compact panel opens at its natural size.
        config = ConfigManager()
        revisions = config.data.setdefault("layout_revisions", {})
        if not revisions.get("template_match_compact_panel"):
            config.set_tool_splitter("TemplateMatch", [])
            revisions["template_match_compact_panel"] = True
            config.save()

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "TemplateMatch"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        connect_execution_time(self.tool, self.label_time)
        self.main_layout.addWidget(self.footer)

        self.train_roi_editor.set_tool(self.tool, self.viewer); self.train_roi_editor.btn_edit.toggled.connect(self._on_train_edit_toggled)
        self.search_roi_editor.set_tool(self.tool, self.viewer); self.search_roi_editor.btn_edit.toggled.connect(self._on_search_edit_toggled)

        self.tool.data_changed.connect(self.refresh_ui)
        self._exec_timer = QTimer(); self._exec_timer.setSingleShot(True); self._exec_timer.setInterval(250); self._exec_timer.timeout.connect(self.tool.mark_dirty)

        # Connect template click for origin picking
        self.template_viewer.mouse_clicked.connect(self._on_template_clicked)

        self.refresh_ui()

    # ------------------------------------------------------------------
    # Layout helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _field(label, widget):
        """Compact field: caption above the control."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(QLabel(label))
        layout.addWidget(widget)
        return container

    @classmethod
    def _field_grid(cls, entries, columns=2):
        """Two-column grid of compact fields; keeps the panel narrow."""
        container = QWidget()
        grid = QGridLayout(container)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(4)
        for index, (label, widget) in enumerate(entries):
            grid.addWidget(
                cls._field(label, widget), index // columns, index % columns
            )
        return container

    @staticmethod
    def _compact_spin(spin, minimum_width=90):
        """Let the layout ignore the spin's wide size hint."""
        spin.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
        )
        spin.setMinimumWidth(minimum_width)

    def _build_training_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.train_roi_editor)

        btn_capture = QPushButton("Capture Train Image")
        btn_capture.clicked.connect(self._on_capture)
        layout.addWidget(btn_capture)
        btn_train = QPushButton("Train Pattern")
        btn_train.clicked.connect(self._on_train_pattern)
        layout.addWidget(btn_train)
        self.label_state = QLabel("Not trained")
        self.label_state.setWordWrap(True)
        layout.addWidget(self.label_state)

        orig_group = QGroupBox("Tool Origin Offset")
        orig_layout = QVBoxLayout(orig_group)
        self.spin_ox = QDoubleSpinBox()
        self.spin_ox.setRange(-500.0, 500.0)
        self.spin_ox.setMaximumWidth(110)
        self.spin_ox.setValue(self.tool.params["origin_x"])
        self._compact_spin(self.spin_ox)
        self.spin_oy = QDoubleSpinBox()
        self.spin_oy.setRange(-500.0, 500.0)
        self.spin_oy.setMaximumWidth(110)
        self.spin_oy.setValue(self.tool.params["origin_y"])
        self._compact_spin(self.spin_oy)
        self.spin_ox.valueChanged.connect(lambda v: self._set_param("origin_x", v))
        self.spin_oy.valueChanged.connect(lambda v: self._set_param("origin_y", v))
        orig_layout.addWidget(
            self._field_grid((("X", self.spin_ox), ("Y", self.spin_oy)))
        )

        self.btn_pick_origin = QPushButton("Pick Origin")
        self.btn_pick_origin.setToolTip(
            "Click a point on the pattern preview to set the origin offset."
        )
        self.btn_pick_origin.setCheckable(True)
        self.btn_pick_origin.clicked.connect(self._on_pick_origin_clicked)
        orig_layout.addWidget(self.btn_pick_origin)
        layout.addWidget(orig_group)

        preview_box = QGroupBox("Pattern Preview")
        preview_layout = QVBoxLayout(preview_box)
        self.template_viewer = ImageViewer()
        self.template_viewer.setFixedSize(180, 180)
        preview_layout.addWidget(self.template_viewer)
        layout.addWidget(preview_box)
        layout.addStretch()
        return panel

    def _build_search_tab(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.addWidget(self.search_roi_editor)

        confidence_group = QGroupBox("Sensitivity")
        confidence_layout = QVBoxLayout(confidence_group)
        self.spin_score = QDoubleSpinBox()
        self.spin_score.setRange(0.1, 1.0)
        self.spin_score.setMaximumWidth(110)
        self.spin_score.setValue(self.tool.params["score_threshold"])
        self.spin_score.valueChanged.connect(
            lambda v: self._set_param("score_threshold", v)
        )
        self._compact_spin(self.spin_score)
        confidence_layout.addWidget(self._field("Confidence", self.spin_score))
        layout.addWidget(confidence_group)

        angle_group = QGroupBox("Angle Range")
        angle_layout = QVBoxLayout(angle_group)
        self.spin_a0 = QSpinBox()
        self.spin_a0.setRange(-180, 180)
        self.spin_a0.setValue(self.tool.params["angle_start"])
        self.spin_a0.setToolTip(
            "Positive angles rotate clockwise in the image frame, "
            "like ROI/fixture angles."
        )
        self.spin_a0.valueChanged.connect(
            lambda v: self._set_param("angle_start", v)
        )
        self._compact_spin(self.spin_a0)
        self.spin_a1 = QSpinBox()
        self.spin_a1.setRange(-180, 180)
        self.spin_a1.setValue(self.tool.params["angle_end"])
        self.spin_a1.setToolTip(
            "Positive angles rotate clockwise in the image frame, "
            "like ROI/fixture angles."
        )
        self.spin_a1.valueChanged.connect(
            lambda v: self._set_param("angle_end", v)
        )
        self._compact_spin(self.spin_a1)
        self.spin_as = QDoubleSpinBox()
        self.spin_as.setRange(0.1, 1.0)
        self.spin_as.setSingleStep(0.1)
        self.spin_as.setDecimals(1)
        self.spin_as.setValue(self.tool.params["angle_step"])
        self.spin_as.valueChanged.connect(
            lambda v: self._set_param("angle_step", v)
        )
        self._compact_spin(self.spin_as)
        angle_layout.addWidget(
            self._field_grid(
                (
                    ("Min", self.spin_a0),
                    ("Max", self.spin_a1),
                    ("Step", self.spin_as),
                )
            )
        )
        layout.addWidget(angle_group)

        scale_group = QGroupBox("Scale Range")
        scale_layout = QVBoxLayout(scale_group)
        self.spin_s0 = QDoubleSpinBox()
        self.spin_s0.setRange(0.1, 5.0)
        self.spin_s0.setValue(self.tool.params["scale_start"])
        self.spin_s0.valueChanged.connect(
            lambda v: self._set_param("scale_start", v)
        )
        self._compact_spin(self.spin_s0)
        self.spin_s1 = QDoubleSpinBox()
        self.spin_s1.setRange(0.1, 5.0)
        self.spin_s1.setValue(self.tool.params["scale_end"])
        self.spin_s1.valueChanged.connect(
            lambda v: self._set_param("scale_end", v)
        )
        self._compact_spin(self.spin_s1)
        self.spin_ss = QDoubleSpinBox()
        self.spin_ss.setRange(0.01, 1.0)
        self.spin_ss.setValue(self.tool.params["scale_step"])
        self.spin_ss.valueChanged.connect(
            lambda v: self._set_param("scale_step", v)
        )
        self._compact_spin(self.spin_ss)
        scale_layout.addWidget(
            self._field_grid(
                (
                    ("Min", self.spin_s0),
                    ("Max", self.spin_s1),
                    ("Step", self.spin_ss),
                )
            )
        )
        layout.addWidget(scale_group)

        performance_group = QGroupBox("Performance")
        performance_layout = QVBoxLayout(performance_group)
        self.spin_pyr = QSpinBox()
        self.spin_pyr.setRange(1, 5)
        self.spin_pyr.setValue(self.tool.params["pyramid_levels"])
        self.spin_pyr.valueChanged.connect(
            lambda v: self._set_param("pyramid_levels", v)
        )
        self._compact_spin(self.spin_pyr)
        performance_layout.addWidget(self._field("Levels", self.spin_pyr))
        layout.addWidget(performance_group)

        results_group = QGroupBox("Expected Results")
        results_layout = QVBoxLayout(results_group)
        self.spin_max = QSpinBox()
        self.spin_max.setRange(0, 100)
        self.spin_max.setValue(self.tool.params["max_matches"])
        self.spin_max.setSpecialValueText("All")
        self.spin_max.setToolTip(
            "Maximum number of matches to return (0 = all). "
            "This does not control the minimum-result error policy."
        )
        self.spin_max.valueChanged.connect(
            lambda v: self._set_param("max_matches", v)
        )
        self._compact_spin(self.spin_max)
        results_layout.addWidget(self._field("Matches", self.spin_max))
        layout.addWidget(results_group)

        self.chk_fixture, self.fixture_axes = add_fixture_group(
            layout,
            self.tool,
            on_change=lambda key, value: self._set_param(key, value),
        )

        self.spin_min_results = add_min_results_group(
            layout, self.tool, "Min matches:", 100, "matches"
        )

        visibility_group = QGroupBox("Visibility")
        visibility_layout = QVBoxLayout(visibility_group)
        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(self.tool.params.get("show_overlays", True))
        self.chk_overlays.toggled.connect(
            lambda v: self._set_param("show_overlays", v)
        )
        visibility_layout.addWidget(self.chk_overlays)
        self.chk_show_search = QCheckBox("Search ROI")
        self.chk_show_search.setToolTip("Show the search region.")
        self.chk_show_search.setChecked(
            self.tool.params.get("show_search_roi", True)
        )
        self.chk_show_search.toggled.connect(
            lambda v: self._set_param("show_search_roi", v)
        )
        visibility_layout.addWidget(self.chk_show_search)
        self.chk_show_region = QCheckBox("Found ROI")
        self.chk_show_region.setToolTip("Show the found match region.")
        self.chk_show_region.setChecked(
            self.tool.params.get("show_found_region", True)
        )
        self.chk_show_region.toggled.connect(
            lambda v: self._set_param("show_found_region", v)
        )
        visibility_layout.addWidget(self.chk_show_region)
        self.chk_show_origin = QCheckBox("Origin")
        self.chk_show_origin.setToolTip("Show the found match origin.")
        self.chk_show_origin.setChecked(
            self.tool.params.get("show_found_origin", True)
        )
        self.chk_show_origin.toggled.connect(
            lambda v: self._set_param("show_found_origin", v)
        )
        visibility_layout.addWidget(self.chk_show_origin)
        self.chk_labels = QCheckBox("Labels")
        self.chk_labels.setChecked(self.tool.params.get("show_labels", True))
        self.chk_labels.toggled.connect(
            lambda v: self._set_param("show_labels", v)
        )
        visibility_layout.addWidget(self.chk_labels)
        self.chk_copy_in = QCheckBox("Copy overlays")
        self.chk_copy_in.setToolTip(
            "Copy this tool's overlays to the input image so downstream "
            "tools can draw on them."
        )
        self.chk_copy_in.setChecked(
            self.tool.params.get("copy_overlays_to_input", True)
        )
        self.chk_copy_in.toggled.connect(
            lambda v: self._set_param("copy_overlays_to_input", v)
        )
        visibility_layout.addWidget(self.chk_copy_in)
        layout.addWidget(visibility_group)
        layout.addStretch()
        return panel

    def _set_param(self, key, value): self.tool.params[key] = value; self._exec_timer.start()
    def _on_capture(self): self._stop_all_editing(); self.tool.save_train_image(); self.record_selector.setCurrentText("train image"); self.refresh_ui()
    def _on_train_pattern(self):
        self._stop_all_editing()
        try:
            roi = self.tool.params.get("train_roi", [0, 0, 0, 0, 0])
            self.tool.train_from_roi(*roi)
            self.refresh_ui()
            self.tool.mark_dirty()
        except Exception as e:
            print(f"Train error: {e}")

    def _on_pick_origin_clicked(self, checked):
        self._picking_origin = checked
        if checked:
            self.template_viewer.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.template_viewer.setCursor(Qt.CursorShape.ArrowCursor)

    def _on_template_clicked(self, x, y):
        if self._picking_origin and self.tool.is_trained:
            tw, th = self.tool._template_size
            ox = x - tw/2
            oy = y - th/2
            self.spin_ox.setValue(ox)
            self.spin_oy.setValue(oy)
            self.btn_pick_origin.setChecked(False)
            self._on_pick_origin_clicked(False)
            self._set_param("origin_x", ox)
            self._set_param("origin_y", oy)

    def _on_tab_changed(self, idx):
        self._stop_all_editing()
        self.refresh_ui()
    def on_record_changed(self): self.refresh_ui()
    def _stop_all_editing(self): self.train_roi_editor.btn_edit.setChecked(False); self.search_roi_editor.btn_edit.setChecked(False); self.viewer.set_roi_editing(False)
    def _on_train_edit_toggled(self, checked):
        if checked: self.search_roi_editor.btn_edit.setChecked(False)
        self.viewer.set_drawing_new_roi(checked)
    def _on_search_edit_toggled(self, checked):
        if checked: self.train_roi_editor.btn_edit.setChecked(False)
        self.viewer.set_drawing_new_roi(checked)

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if key is None: return
        img_data = self.tool.outputs.get("output image") if key == "output image" else (self.tool._static_train_image if key == "train image" else self.tool.inputs.get(key))

        from falcocv.core.vision_image import VisionImage
        img = img_data.data if isinstance(img_data, VisionImage) else img_data

        # Pass overlay rendering flag like blobs tool
        self.viewer.set_image(img_data, auto_fit=auto_fit,
                              render_overlays_flag=self.chk_overlays.isChecked() if hasattr(self, "chk_overlays") else True)

        # ROI visibility based on selected image (independent of tab)
        if isinstance(img, np.ndarray):
            if key == "train image":
                rk = "train_roi"
                show = True
            elif key == "input image":
                rk = "search_roi"
                show = True
            else:
                rk = None
                show = False

            if show and rk:
                self.tool._active_roi_key = rk
                roi_data = self.tool.params.get(rk, [0, 0, 0, 0, 0])
                if len(roi_data) == 4:
                    roi_data = list(roi_data) + [0.0]
                if roi_data[2] > 0 and roi_data[3] > 0:
                    input_img = self.tool.inputs.get("input image")
                    v_img = VisionImage.from_array(input_img) if input_img is not None else None
                    if v_img is not None:
                        abs_roi = v_img.fixture_to_image(roi_data)
                    else:
                        abs_roi = roi_data
                    self.viewer.set_roi(abs_roi)
                else:
                    self.viewer.set_roi(None)
            else:
                self.viewer.set_roi(None)
        else:
            self.viewer.set_roi(None)

        # Enable/disable ROI editors based on selected image
        disabled = disabled_css()
        if hasattr(self, "train_roi_editor"):
            is_train = (key == "train image")
            self.train_roi_editor.setEnabled(is_train)
            self.train_roi_editor.setStyleSheet(disabled if not is_train else "")
            if not is_train:
                self.train_roi_editor.btn_edit.setChecked(False)
        if hasattr(self, "search_roi_editor"):
            is_search = (key == "input image")
            self.search_roi_editor.setEnabled(is_search)
            self.search_roi_editor.setStyleSheet(disabled if not is_search else "")
            if not is_search:
                self.search_roi_editor.btn_edit.setChecked(False)

        if self.isVisible() and hasattr(self, "template_viewer") and self.template_viewer is not None:
            if self.tool.is_trained:
                from falcocv.core.vision_image import VisionImage
                tpl = self.tool._template_color.copy()
                th, tw = tpl.shape[:2]
                cx, cy = tw // 2, th // 2
                tpl_color = cv2.cvtColor(tpl, cv2.COLOR_GRAY2BGR) if tpl.ndim == 2 else tpl.copy()

                # Gradient magnitude overlay (cyan = regions of high contrast
                # that contribute more to normalized cross-correlation)
                gray = cv2.cvtColor(tpl_color, cv2.COLOR_BGR2GRAY)
                gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
                gy = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
                mag = cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32))
                mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, high_contrast = cv2.threshold(mag, 64, 255, cv2.THRESH_BINARY)
                tpl_color[high_contrast > 0] = [255, 255, 0]

                v_tpl = VisionImage.from_array(tpl_color)
                v_tpl.get_overlay_layer("default", source_tool=self.tool.name)

                ox, oy = self.tool.params["origin_x"], self.tool.params["origin_y"]
                ocx, ocy = int(cx + ox), int(cy + oy)

                v_tpl.get_overlay_layer("default").add_axes(
                    (ocx, ocy), angle=0, length=30,
                    color_x=(0, 0, 255), color_y=(0, 255, 0), thickness=2
                )
                v_tpl.get_overlay_layer("default").add_point(
                    (ocx, ocy), radius=5, color=(255, 255, 255)
                )

                self.template_viewer.set_image(v_tpl, auto_fit=True, render_overlays_flag=True)
                self.label_state.setText("Trained")
                self.label_state.setStyleSheet(f"color: {SUCCESS}; font-weight: bold;")
            else:
                self.template_viewer.set_image(None); self.label_state.setText("Not trained"); self.label_state.setStyleSheet("")

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "TemplateMatch")

