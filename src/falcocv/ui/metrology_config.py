"""Shared PyQt6 config panel for the metrology fit tools (LineFit / CircleFit)."""

import numpy as np
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGroupBox,
    QLabel,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from falcocv.core.vision_image import VisionImage
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.style import disabled_css
from falcocv.ui.tool_config_helpers import (
    add_min_results_group,
    add_record_header,
    configure_params_splitter,
    create_execution_footer,
    make_record_selector,
    save_params_splitter,
    wrap_params_scroll,
)
from falcocv.ui.widgets.viewer import ImageViewer


class MetrologyFitConfigBase(QWidget):
    """ROI + caliper-array + robust-fit controls with a preview viewer.

    Subclasses set ``TOOL_KEY`` (splitter persistence), ``FIT_LABEL``,
    ``RESULT_NOUN`` and ``IS_CIRCLE``; the parameter names are shared with
    ``_fit_support``.
    """

    TOOL_KEY = "MetrologyFit"
    FIT_LABEL = "Show Fitted Shape"
    RESULT_NOUN = "result"
    MAX_RESULTS = 10
    IS_CIRCLE = False

    def __init__(self, tool):
        super().__init__()
        self.tool = tool

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self.content_layout = QSplitter(Qt.Orientation.Horizontal)
        self.left_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.left_panel)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)

        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)

        self._build_controls()
        self._build_viewer()

        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)
        self.content_layout.addWidget(self.right_panel)
        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, self.TOOL_KEY
        )
        self.main_layout.addWidget(self.content_layout, 1)

        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))

    # ── Controls ──────────────────────────────────────────────────────

    def _build_controls(self):
        from falcocv.ui.widgets.roi_editor import RoiEditorWidget

        self.roi_editor = RoiEditorWidget()
        self.controls_layout.addWidget(self.roi_editor)

        source_group = QGroupBox("Point Source")
        source_layout = QVBoxLayout(source_group)
        self.combo_source = QComboBox()
        self.combo_source.addItem("Caliper array (ROI)", "calipers")
        self.combo_source.addItem("Overlays / contour points", "existing")
        index = self.combo_source.findData(
            str(self.tool.params.get("point_source", "calipers"))
        )
        self.combo_source.setCurrentIndex(max(0, index))
        self.combo_source.setToolTip(
            "Caliper array: the ROI is scanned and every caliper contributes one "
            "subpixel edge point. Overlays / contour points: fit points "
            "produced by another tool (POINT overlays or contour/polygon "
            "vertices)."
        )
        self.combo_source.currentIndexChanged.connect(self._on_source_changed)
        source_layout.addWidget(self.combo_source)

        self.chk_points = QCheckBox("Overlay points")
        self.chk_points.setChecked(bool(self.tool.params.get("use_overlay_points", True)))
        self.chk_points.toggled.connect(
            lambda value: self._set_param("use_overlay_points", value)
        )
        source_layout.addWidget(self.chk_points)

        self.chk_contours = QCheckBox("Contour / polygon vertices")
        self.chk_contours.setChecked(bool(self.tool.params.get("use_contours", False)))
        self.chk_contours.toggled.connect(
            lambda value: self._set_param("use_contours", value)
        )
        source_layout.addWidget(self.chk_contours)

        hint = QLabel(
            "Caliper mode needs only the ROI. Points mode reads the visible "
            "POINT overlays and contour/polygon vertices."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color: #9a9a9a; font-size: 11px;")
        source_layout.addWidget(hint)
        self.controls_layout.addWidget(source_group)

        caliper_group = QGroupBox("Caliper Array")
        caliper_group_layout = QVBoxLayout(caliper_group)
        self.caliper_box = QWidget()
        caliper_body = QVBoxLayout(self.caliper_box)
        caliper_body.setContentsMargins(0, 0, 0, 0)

        def add_spin(label, spin):
            caliper_body.addWidget(QLabel(label))
            caliper_body.addWidget(spin)
            return spin

        self.spin_calipers = QSpinBox()
        self.spin_calipers.setRange(3 if self.IS_CIRCLE else 2, 360)
        self.spin_calipers.setValue(
            int(self.tool.params.get("caliper_count", 16 if self.IS_CIRCLE else 9))
        )
        self.spin_calipers.setToolTip(
            "Number of calipers distributed on the ROI. Each caliper returns at "
            "most one edge point."
        )
        self.spin_calipers.valueChanged.connect(
            lambda value: self._set_param("caliper_count", int(value))
        )
        add_spin("Calipers:", self.spin_calipers)

        if self.IS_CIRCLE:
            self.spin_radius = QDoubleSpinBox()
            self.spin_radius.setRange(0.0, 10000.0)
            self.spin_radius.setDecimals(1)
            self.spin_radius.setSingleStep(2.0)
            self.spin_radius.setSpecialValueText("Auto")
            self.spin_radius.setValue(
                float(self.tool.params.get("caliper_radius", 0.0))
            )
            self.spin_radius.setToolTip(
                "Radius of the caliper ring around the ROI centre. Auto uses "
                "half of the ROI's short side."
            )
            self.spin_radius.valueChanged.connect(
                lambda value: self._set_param("caliper_radius", float(value))
            )
            add_spin("Ring radius (px, 0 = auto):", self.spin_radius)

            self.combo_direction = QComboBox()
            self.combo_direction.addItem("Inwards (towards centre)", "inward")
            self.combo_direction.addItem("Outwards (away from centre)", "outward")
            direction = str(self.tool.params.get("caliper_direction", "inward"))
            self.combo_direction.setCurrentIndex(
                max(0, self.combo_direction.findData(direction))
            )
            self.combo_direction.setToolTip(
                "Radial scan direction from the caliper ring. Inwards searches "
                "between the ring and the centre; outwards searches past the "
                "ring. The edge polarity follows this direction."
            )
            self.combo_direction.currentIndexChanged.connect(
                lambda _index: self._set_param(
                    "caliper_direction", self.combo_direction.currentData()
                )
            )
            add_spin("Search direction:", self.combo_direction)

        self.spin_length = QDoubleSpinBox()
        self.spin_length.setRange(0.0, 10000.0)
        self.spin_length.setDecimals(1)
        self.spin_length.setSingleStep(2.0)
        self.spin_length.setSpecialValueText("Auto")
        self.spin_length.setValue(
            float(self.tool.params.get("caliper_length", 0.0))
        )
        self.spin_length.setToolTip(
            "Line: scan length across the ROI height (auto = full height). "
            "Circle: radial band starting at the ring and scanned in the "
            "selected direction (auto = 35% of the ROI short side)."
        )
        self.spin_length.valueChanged.connect(
            lambda value: self._set_param("caliper_length", float(value))
        )
        add_spin("Caliper length (px, 0 = auto):", self.spin_length)

        self.spin_width = QDoubleSpinBox()
        self.spin_width.setRange(1.0, 1000.0)
        self.spin_width.setDecimals(1)
        self.spin_width.setSingleStep(1.0)
        self.spin_width.setValue(float(self.tool.params.get("caliper_width", 5.0)))
        self.spin_width.setToolTip(
            "Caliper width averaged into the 1D profile: along the ROI for a "
            "line, along the tangent for a circle."
        )
        self.spin_width.valueChanged.connect(
            lambda value: self._set_param("caliper_width", float(value))
        )
        add_spin("Caliper width (px):", self.spin_width)

        self.combo_polarity = QComboBox()
        for label in ("Any", "Dark→Light", "Light→Dark"):
            self.combo_polarity.addItem(label, label)
        polarity = str(self.tool.params.get("caliper_edge_polarity", "Any"))
        self.combo_polarity.setCurrentIndex(
            max(0, self.combo_polarity.findData(polarity))
        )
        self.combo_polarity.setToolTip(
            "Edge polarity along the scan direction. The line scans towards the "
            "ROI's local bottom; the circle scans inwards or outwards from the "
            "caliper ring."
        )
        self.combo_polarity.currentIndexChanged.connect(
            lambda _index: self._set_param(
                "caliper_edge_polarity", self.combo_polarity.currentData()
            )
        )
        add_spin("Edge polarity:", self.combo_polarity)

        self.combo_selector = QComboBox()
        for label in ("Best", "First", "Last"):
            self.combo_selector.addItem(label, label)
        selector = str(self.tool.params.get("caliper_selector", "Best"))
        self.combo_selector.setCurrentIndex(
            max(0, self.combo_selector.findData(selector))
        )
        self.combo_selector.setToolTip(
            "Which edge of each caliper profile is used: strongest peak, or the "
            "first/last peak along the scan direction."
        )
        self.combo_selector.currentIndexChanged.connect(
            lambda _index: self._set_param(
                "caliper_selector", self.combo_selector.currentData()
            )
        )
        add_spin("Edge selector:", self.combo_selector)

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.5, 255.0)
        self.spin_threshold.setDecimals(1)
        self.spin_threshold.setSingleStep(1.0)
        self.spin_threshold.setValue(
            float(self.tool.params.get("caliper_threshold", 5.0))
        )
        self.spin_threshold.setToolTip(
            "Minimum gradient of the 1D profile for a caliper to report an edge."
        )
        self.spin_threshold.valueChanged.connect(
            lambda value: self._set_param("caliper_threshold", float(value))
        )
        add_spin("Edge threshold:", self.spin_threshold)

        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.0, 10.0)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setValue(float(self.tool.params.get("caliper_sigma", 1.0)))
        self.spin_sigma.setToolTip(
            "Gaussian smoothing of the 1D profile before the derivative."
        )
        self.spin_sigma.valueChanged.connect(
            lambda value: self._set_param("caliper_sigma", float(value))
        )
        add_spin("Smoothing sigma:", self.spin_sigma)

        self.chk_subpixel = QCheckBox("Subpixel edge position")
        self.chk_subpixel.setChecked(
            bool(self.tool.params.get("caliper_subpixel", True))
        )
        self.chk_subpixel.toggled.connect(
            lambda value: self._set_param("caliper_subpixel", bool(value))
        )
        caliper_body.addWidget(self.chk_subpixel)

        caliper_group_layout.addWidget(self.caliper_box)
        self.controls_layout.addWidget(caliper_group)

        fit_group = QGroupBox("Fit")
        fit_layout = QVBoxLayout(fit_group)
        self.chk_robust = QCheckBox("Robust (RANSAC)")
        self.chk_robust.setChecked(bool(self.tool.params.get("robust", True)))
        self.chk_robust.toggled.connect(lambda value: self._set_param("robust", value))
        fit_layout.addWidget(self.chk_robust)

        self.spin_residual = QDoubleSpinBox()
        self.spin_residual.setRange(0.0, 500.0)
        self.spin_residual.setDecimals(2)
        self.spin_residual.setSingleStep(0.5)
        self.spin_residual.setSpecialValueText("Auto")
        self.spin_residual.setValue(
            float(self.tool.params.get("max_residual_px", 0.0))
        )
        self.spin_residual.setToolTip(
            "Maximum allowed residual in pixels. Auto uses 2.5 x the median "
            "residual of the least-squares fit."
        )
        self.spin_residual.valueChanged.connect(
            lambda value: self._set_param("max_residual_px", float(value))
        )
        fit_layout.addWidget(QLabel("Max residual (px):"))
        fit_layout.addWidget(self.spin_residual)

        self.spin_iterations = QSpinBox()
        self.spin_iterations.setRange(10, 5000)
        self.spin_iterations.setSingleStep(50)
        self.spin_iterations.setValue(int(self.tool.params.get("iterations", 200)))
        self.spin_iterations.valueChanged.connect(
            lambda value: self._set_param("iterations", int(value))
        )
        fit_layout.addWidget(QLabel("RANSAC iterations:"))
        fit_layout.addWidget(self.spin_iterations)

        self.spin_outliers = QDoubleSpinBox()
        self.spin_outliers.setRange(0.0, 50.0)
        self.spin_outliers.setDecimals(0)
        self.spin_outliers.setSingleStep(5.0)
        self.spin_outliers.setSuffix(" %")
        self.spin_outliers.setValue(
            float(self.tool.params.get("max_outlier_percent", 20.0))
        )
        self.spin_outliers.setToolTip(
            "Maximum percentage of points the fit may ignore. Points are only "
            "discarded when they are outside the robust residual cutoff, and "
            "if more than this are outside it the tool reports an error "
            "instead of publishing a biased fit. 0 disables the limit."
        )
        self.spin_outliers.valueChanged.connect(
            lambda value: self._set_param("max_outlier_percent", float(value))
        )
        self.spin_outliers.setEnabled(self.chk_robust.isChecked())
        self.chk_robust.toggled.connect(self.spin_outliers.setEnabled)
        fit_layout.addWidget(QLabel("Max outliers (%):"))
        fit_layout.addWidget(self.spin_outliers)
        self.controls_layout.addWidget(fit_group)

        add_min_results_group(
            self.controls_layout,
            self.tool,
            "Minimum results:",
            self.MAX_RESULTS,
            self.RESULT_NOUN,
        )

        vis_group = QGroupBox("Visibility")
        vis_layout = QVBoxLayout(vis_group)
        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(True)
        self.chk_overlays.toggled.connect(self.refresh_ui)
        vis_layout.addWidget(self.chk_overlays)

        for key, text, default in (
            ("show_search_roi", "Show Search Region", True),
            ("show_calipers", "Show Calipers", True),
            ("show_fit_line", self.FIT_LABEL, True),
            ("show_center", "Show Center", True),
            ("show_inliers", "Show Inliers", True),
            ("show_outliers", "Show Outliers", False),
            ("show_labels", "Show Labels", True),
            ("copy_overlays_to_input", "Copy overlays to input image", True),
        ):
            if key not in self.tool.params:
                continue
            box = QCheckBox(text)
            box.setChecked(bool(self.tool.params.get(key, default)))
            box.toggled.connect(
                lambda value, k=key: self._set_param(k, bool(value))
            )
            vis_layout.addWidget(box)
        self.controls_layout.addWidget(vis_group)
        self.controls_layout.addStretch()

        self._update_source_state()

    def _build_viewer(self):
        self.record_selector = make_record_selector()
        self.record_selector.addItem("input image", "input image")
        self.record_selector.addItem("output image", "output image")
        self.record_selector.currentIndexChanged.connect(self._on_record_changed)
        add_record_header(self.viewer_layout, self.record_selector, tool=self.tool)

        self.viewer = ImageViewer()
        self.viewer_layout.addWidget(self.viewer)
        self.roi_editor.set_tool(self.tool, self.viewer)

    # ── Wiring ────────────────────────────────────────────────────────

    def _set_param(self, key, value):
        self.tool.params[key] = value
        self.tool.mark_dirty()

    def _on_source_changed(self):
        self._set_param("point_source", self.combo_source.currentData())
        self._update_source_state()

    def _update_source_state(self):
        calipers = self.combo_source.currentData() == "calipers"
        self.caliper_box.setEnabled(calipers)
        self.chk_points.setEnabled(not calipers)
        self.chk_contours.setEnabled(not calipers)

    def _on_record_changed(self):
        if self.record_selector.currentData() != "input image":
            self.roi_editor.btn_edit.setChecked(False)
            self.viewer.set_roi_editing(False)
        self.refresh_ui()

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if not key:
            return
        is_input = key == "input image"
        image = self.tool.inputs.get("input image") if is_input else self.tool.outputs.get(key)

        if isinstance(image, np.ndarray):
            self.viewer.set_image(image, auto_fit=auto_fit)
            self.viewer.set_roi(None)
        elif isinstance(image, VisionImage):
            self.viewer.set_image(
                image,
                auto_fit=auto_fit,
                render_overlays_flag=self.chk_overlays.isChecked(),
            )
            if is_input:
                roi = self.tool.params.get("roi", [0, 0, 0, 0, 0])
                if len(roi) >= 4 and roi[2] > 0 and roi[3] > 0:
                    self.viewer.set_roi(image.fixture_to_image(list(roi)))
                else:
                    self.viewer.set_roi(None)
            else:
                self.viewer.set_roi(None)
        else:
            self.viewer.set_image(None)

        self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")

        self.roi_editor.setEnabled(is_input)
        self.roi_editor.setStyleSheet(disabled_css() if not is_input else "")
        if not is_input:
            self.roi_editor.btn_edit.setChecked(False)
            self.viewer.set_roi_editing(False)

    def on_config_closed(self):
        save_params_splitter(self.content_layout, self.TOOL_KEY)
