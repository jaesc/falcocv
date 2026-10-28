import numpy as np
from falcocv.core.vision_image import VisionImage
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter,
                             QComboBox, QCheckBox, QGroupBox, QSpinBox, QDoubleSpinBox)
from PyQt6.QtCore import Qt, QTimer
from falcocv.ui.style import disabled_css
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)
from falcocv.ui.widgets.viewer import ImageViewer

from .logic import BarcodeDecoder


class BarcodeDecoderConfig(QWidget):
    """Configuration widget for BarcodeDecoder — Python backend."""

    ARUCO_DICTS = ["DICT_4X4_50", "DICT_4X4_100", "DICT_4X4_250", "DICT_4X4_1000",
                   "DICT_5X5_50", "DICT_5X5_100", "DICT_5X5_250", "DICT_5X5_1000",
                   "DICT_6X6_50", "DICT_6X6_100", "DICT_6X6_250", "DICT_6X6_1000",
                   "DICT_7X7_50", "DICT_7X7_100", "DICT_7X7_250", "DICT_7X7_1000",
                   "DICT_APRILTAG_16h5", "DICT_APRILTAG_25h9", "DICT_APRILTAG_36h10", "DICT_APRILTAG_36h11"]

    def __init__(self, tool: BarcodeDecoder):
        super().__init__()
        self.tool = tool
        self._params_widgets = {}
        self._suppress_execute = True

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.content_layout = QSplitter(Qt.Orientation.Horizontal)

        # LEFT: Controls
        self.left_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.left_panel)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)

        from falcocv.ui.widgets.roi_editor import RoiEditorWidget
        self.roi_editor = RoiEditorWidget()
        self.controls_layout.addWidget(self.roi_editor)

        # Code Type
        cfg_group = QGroupBox("Code Type")
        cfg_lyt = QVBoxLayout(cfg_group)
        self.combo_code_type = QComboBox()
        self.combo_code_type.addItems(["Auto", "QR Code", "Barcode 1D", "DataMatrix", "ArUco"])
        self.combo_code_type.setCurrentText(self.tool.params["code_type"])
        self.combo_code_type.currentTextChanged.connect(self._on_code_type_changed)
        cfg_lyt.addWidget(QLabel("Detection Mode:"))
        cfg_lyt.addWidget(self.combo_code_type)
        self.controls_layout.addWidget(cfg_group)

        self._build_general_panel()
        self._build_qr_panel()
        self._build_barcode_panel()
        self._build_dmtx_panel()
        self._build_aruco_panel()
        self._build_auto_panel()
        self._build_preprocessing_panel()

        # Fixture toggle
        self.chk_fixture = QCheckBox("Act as fixture (use first code as image origin)")
        self.chk_fixture.setChecked(self.tool.params.get("act_as_fixture", False))
        self.chk_fixture.toggled.connect(
            lambda c: self.tool.params.__setitem__("act_as_fixture", c) or self.tool.mark_dirty()
        )
        self.controls_layout.addWidget(self.chk_fixture)

        # Visibility checkboxes
        vis_group = QGroupBox("Visibility")
        vis_lyt = QVBoxLayout(vis_group)
        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(True)
        self.chk_overlays.toggled.connect(self._on_overlays_toggled)
        vis_lyt.addWidget(self.chk_overlays)
        self.chk_search_roi = QCheckBox("Show Search Region")
        self.chk_search_roi.setChecked(self.tool.params.get("show_search_roi", True))
        self.chk_search_roi.toggled.connect(lambda v: self._set_top_param("show_search_roi", v))
        vis_lyt.addWidget(self.chk_search_roi)
        self.chk_found_region = QCheckBox("Show Found Region")
        self.chk_found_region.setChecked(self.tool.params.get("show_found_region", True))
        self.chk_found_region.toggled.connect(lambda v: self._set_top_param("show_found_region", v))
        vis_lyt.addWidget(self.chk_found_region)
        self.chk_labels = QCheckBox("Show Labels")
        self.chk_labels.setChecked(self.tool.params.get("show_labels", True))
        self.chk_labels.toggled.connect(lambda v: self._set_top_param("show_labels", v))
        vis_lyt.addWidget(self.chk_labels)
        self.chk_found_origin = QCheckBox("Show Found Origin")
        self.chk_found_origin.setChecked(self.tool.params.get("show_found_origin", True))
        self.chk_found_origin.toggled.connect(lambda v: self._set_top_param("show_found_origin", v))
        vis_lyt.addWidget(self.chk_found_origin)
        self.chk_copy_in = QCheckBox("Copy overlays to input image")
        self.chk_copy_in.setChecked(self.tool.params.get("copy_overlays_to_input", True))
        self.chk_copy_in.toggled.connect(lambda v: self._set_top_param("copy_overlays_to_input", v))
        vis_lyt.addWidget(self.chk_copy_in)
        self.controls_layout.addWidget(vis_group)

        self.controls_layout.addStretch()
        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)

        # RIGHT: Viewer
        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.record_selector = QComboBox()
        self.record_selector.addItem("input image", "input image")
        self.record_selector.addItem("output image", "output image")
        self.record_selector.currentIndexChanged.connect(self.on_record_changed)
        self.viewer_layout.addWidget(self.record_selector)

        self.viewer = ImageViewer()
        if hasattr(self, "roi_editor"):
            self.roi_editor.set_tool(self.tool, self.viewer)
        self.viewer_layout.addWidget(self.viewer)

        self.content_layout.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "BarcodeDecoder"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)
        self._update_visibility()
        self._suppress_execute = False
        self.tool.mark_dirty()
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))

    # ── Panel builders ───────────────────────────────────────────

    def _build_general_panel(self):
        g = QGroupBox("General Parameters")
        l = QVBoxLayout(g)
        s = self._add_spin(l, "Max Count (0 = unlimited)", 0, 100, 1,
                           self.tool.params.get("max_count", 1),
                           lambda v: self._set_top_param("max_count", v))
        self._params_widgets["max_count"] = ("max_count", s)
        self.controls_layout.addWidget(g)

    def _build_qr_panel(self):
        g = self.qr_params_group = QGroupBox("QR Parameters")
        l = QVBoxLayout(g)
        s = self._add_dspin(l, "Eps X", 0.01, 2.0, 0.01,
                            self.tool.params["qr_params"]["eps_x"],
                            lambda v: self._set_sub("qr_params", "eps_x", v))
        self._params_widgets["qr_eps_x"] = ("qr_params", "eps_x", s)
        s = self._add_dspin(l, "Eps Y", 0.01, 2.0, 0.01,
                            self.tool.params["qr_params"]["eps_y"],
                            lambda v: self._set_sub("qr_params", "eps_y", v))
        self._params_widgets["qr_eps_y"] = ("qr_params", "eps_y", s)
        self.controls_layout.addWidget(g)

    def _build_barcode_panel(self):
        g = self.barcode_params_group = QGroupBox("1D Barcode Parameters")
        l = QVBoxLayout(g)
        info = QLabel(
            "1D codes are decoded with pyzbar + zxing-cpp.\n"
            "Supported: EAN-13/8, UPC-A/E, CODE128, CODE39, CODE93,\n"
            "ITF, Codabar, DataBar and DataBar Stacked."
        )
        info.setWordWrap(True)
        l.addWidget(info)
        self.controls_layout.addWidget(g)

    def _build_dmtx_panel(self):
        g = self.dmtx_params_group = QGroupBox("DataMatrix Parameters")
        l = QVBoxLayout(g)
        l.addWidget(QLabel("Decode params:"))
        s = self._add_spin(l, "Shrink", 1, 8, 1,
                           self.tool.params["dmtx_params"]["shrink"],
                           lambda v: self._set_sub("dmtx_params", "shrink", v))
        self._params_widgets["dm_shrink"] = ("dmtx_params", "shrink", s)
        s = self._add_spin(l, "Timeout (ms)", 100, 5000, 100,
                           self.tool.params["dmtx_params"]["timeout_ms"],
                           lambda v: self._set_sub("dmtx_params", "timeout_ms", v))
        self._params_widgets["dm_timeout"] = ("dmtx_params", "timeout_ms", s)
        chk = self._add_chk(l, "Enable sharpening (unsharp mask)",
                            self.tool.params["dmtx_params"]["enable_sharpening"],
                            lambda v: self._set_sub("dmtx_params", "enable_sharpening", v))
        self._params_widgets["dm_sharpen"] = ("dmtx_params", "enable_sharpening", chk)
        self.controls_layout.addWidget(g)

    def _build_aruco_panel(self):
        g = self.aruco_params_group = QGroupBox("ArUco Parameters")
        l = QVBoxLayout(g)
        cb = QComboBox()
        cb.addItems(self.ARUCO_DICTS)
        cb.setCurrentText(self.tool.params["aruco_params"]["dict_name"])
        cb.currentTextChanged.connect(lambda v: self._set_sub("aruco_params", "dict_name", v))
        self._params_widgets["aruco_dict"] = ("aruco_params", "dict_name", cb)
        row = QHBoxLayout()
        row.addWidget(QLabel("Dictionary:"))
        row.addWidget(cb)
        l.addLayout(row)
        cb2 = QComboBox()
        cb2.addItems(["NONE", "SUBPIX", "CONTOUR", "APRILTAG"])
        cb2.setCurrentIndex(self.tool.params["aruco_params"]["cornerRefinementMethod"])
        cb2.currentIndexChanged.connect(lambda v: self._set_sub("aruco_params", "cornerRefinementMethod", v))
        self._params_widgets["aruco_cr"] = ("aruco_params", "cornerRefinementMethod", cb2)
        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Corner Refinement:"))
        row2.addWidget(cb2)
        l.addLayout(row2)
        chk = self._add_chk(l, "Detect Inverted Markers",
                            self.tool.params["aruco_params"]["detectInvertedMarker"],
                            lambda v: self._set_sub("aruco_params", "detectInvertedMarker", v))
        self._params_widgets["aruco_inv"] = ("aruco_params", "detectInvertedMarker", chk)
        self.controls_layout.addWidget(g)

    def _build_auto_panel(self):
        g = self.auto_params_group = QGroupBox("Auto Mode Parameters")
        l = QVBoxLayout(g)
        l.addWidget(QLabel("ArUco Dictionary for Auto mode:"))
        cb = QComboBox()
        cb.addItems(self.ARUCO_DICTS)
        cb.setCurrentText(self.tool.params["aruco_params"]["dict_name"])
        cb.currentTextChanged.connect(lambda v: self._set_sub("aruco_params", "dict_name", v))
        self._params_widgets["auto_aruco"] = ("aruco_params", "dict_name", cb)
        l.addWidget(cb)
        l.addWidget(QLabel("Downsample Threshold in Auto:"))
        s = self._add_spin(l, "", 128, 4096, 128,
                           self.tool.params["barcode_params"]["downsampling_threshold"],
                           lambda v: self._set_sub("barcode_params", "downsampling_threshold", v))
        self._params_widgets["auto_ds"] = ("barcode_params", "downsampling_threshold", s)
        self.controls_layout.addWidget(g)

    def _build_preprocessing_panel(self):
        g = QGroupBox("Preprocessing")
        l = QVBoxLayout(g)
        cb = QComboBox()
        cb.addItems(["no_clahe_then_fallback", "always_clahe", "always_no_clahe"])
        cb.setCurrentText(self.tool.params["preprocessing"]["strategy"])
        cb.currentTextChanged.connect(lambda v: self._set_sub("preprocessing", "strategy", v))
        self._params_widgets["pre_strat"] = ("preprocessing", "strategy", cb)
        row = QHBoxLayout()
        row.addWidget(QLabel("Strategy:"))
        row.addWidget(cb)
        l.addLayout(row)
        s = self._add_dspin(l, "CLAHE Clip Limit", 0.5, 10.0, 0.5,
                            self.tool.params["preprocessing"]["clahe_clip_limit"],
                            lambda v: self._set_sub("preprocessing", "clahe_clip_limit", v))
        self._params_widgets["pre_clip"] = ("preprocessing", "clahe_clip_limit", s)
        s = self._add_dspin(l, "CLAHE Tile %", 0.5, 20.0, 0.5,
                            self.tool.params["preprocessing"]["clahe_tile_pct"],
                            lambda v: self._set_sub("preprocessing", "clahe_tile_pct", v))
        self._params_widgets["pre_tile"] = ("preprocessing", "clahe_tile_pct", s)
        self.controls_layout.addWidget(g)

    # ── Helper widget factories ───────────────────────────────────

    def _add_spin(self, layout, label, min_v, max_v, step, default, callback):
        row = QHBoxLayout()
        if label:
            row.addWidget(QLabel(label))
        s = QSpinBox()
        s.setRange(min_v, max_v)
        s.setSingleStep(step)
        s.setValue(default)
        s.valueChanged.connect(callback)
        row.addWidget(s)
        row.addStretch()
        layout.addLayout(row)
        return s

    def _add_dspin(self, layout, label, min_v, max_v, step, default, callback):
        row = QHBoxLayout()
        if label:
            row.addWidget(QLabel(label))
        s = QDoubleSpinBox()
        s.setRange(min_v, max_v)
        s.setSingleStep(step)
        s.setDecimals(3)
        s.setValue(default)
        s.valueChanged.connect(callback)
        row.addWidget(s)
        row.addStretch()
        layout.addLayout(row)
        return s

    def _add_chk(self, layout, label, default, callback):
        cb = QCheckBox(label)
        cb.setChecked(default)
        cb.toggled.connect(callback)
        layout.addWidget(cb)
        return cb

    # ── Param setters ────────────────────────────────────────────

    def _set_sub(self, sub_key, key, value):
        self.tool.params[sub_key][key] = value
        self.tool.mark_dirty()

    def _set_top_param(self, key, value):
        self.tool.params[key] = value
        self.tool.mark_dirty()

    # ── Visibility & refresh ──────────────────────────────────────

    def _update_visibility(self):
        ct = self.combo_code_type.currentText()
        self.qr_params_group.setVisible(ct in ("QR Code", "Auto"))
        self.barcode_params_group.setVisible(ct == "Barcode 1D")
        self.dmtx_params_group.setVisible(ct in ("DataMatrix", "Auto"))
        self.aruco_params_group.setVisible(ct == "ArUco")
        self.auto_params_group.setVisible(ct == "Auto")

    def _on_code_type_changed(self, v):
        self.tool.params["code_type"] = v
        self._update_visibility()
        self.tool.mark_dirty()

    def _on_overlays_toggled(self, v):
        self.refresh_ui()

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "BarcodeDecoder")

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if not key:
            return
        is_input = (key == "input image")
        img = self.tool.inputs.get("input image") if is_input else self.tool.outputs.get(key)

        if isinstance(img, np.ndarray):
            self.viewer.set_image(img, auto_fit=auto_fit)
            self.viewer.set_roi(None)
        elif isinstance(img, VisionImage):
            self.viewer.set_image(img, auto_fit=auto_fit,
                                  render_overlays_flag=self.chk_overlays.isChecked())
            if is_input:
                roi_data = self.tool.params["roi"]
                if roi_data[2] > 0 and roi_data[3] > 0:
                    img_input = self.tool.inputs.get("input image")
                    if isinstance(img_input, VisionImage):
                        abs_roi = img_input.fixture_to_image(roi_data)
                    else:
                        abs_roi = roi_data
                    self.viewer.set_roi(abs_roi)
                else:
                    self.viewer.set_roi(None)
            else:
                self.viewer.set_roi(None)
        else:
            self.viewer.set_image(None)

        if hasattr(self, "label_time"):
            self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")

        disabled = disabled_css()
        if hasattr(self, "roi_editor"):
            self.roi_editor.setEnabled(is_input)
            self.roi_editor.setStyleSheet(disabled if not is_input else "")
            if not is_input:
                self.roi_editor.btn_edit.setChecked(False)
                self.viewer.set_roi_editing(False)

    def on_record_changed(self):
        if self.record_selector.currentData() != "input image":
            if hasattr(self, "roi_editor"):
                self.roi_editor.btn_edit.setChecked(False)
            self.viewer.set_roi_editing(False)
        self.refresh_ui()
