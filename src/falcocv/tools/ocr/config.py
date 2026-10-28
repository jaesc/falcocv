"""OCR tool using PP-OCRv5 (ONNX Runtime) — pure recognition."""

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter,
    QComboBox, QCheckBox, QGroupBox, QLineEdit,
    QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, QTimer

from falcocv.core.tool_base import BaseTool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi
from falcocv.core.ocr.industrial_ocr_engine import (
    PPOCRv5Engine, OCRResult, preload_engine,
)
from falcocv.ui.widgets.viewer import ImageViewer
from falcocv.ui.style import ALT_BG, BORDER, SUCCESS, CAUTION, WHITE
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)


from .logic import OCR
class OCRConfigWidget(QWidget):
    def __init__(self, tool: "OCR"):
        super().__init__()
        self.tool = tool
        self._suppress_execute = True

        self.setWindowTitle("OCR Config")
        self.setMinimumSize(600, 400)

        self._debounce_timer = QTimer()
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(200)
        self._debounce_timer.timeout.connect(self.tool.mark_dirty)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.content_layout = QSplitter(Qt.Orientation.Horizontal)
        self.content_layout.setChildrenCollapsible(False)

        self.left_panel = QWidget()
        left_outer = QVBoxLayout(self.left_panel)
        left_outer.setContentsMargins(8, 8, 8, 8)

        self.viewer = ImageViewer()
        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.record_selector = QComboBox()
        self.record_selector.addItem("input image", "input image")
        self.record_selector.addItem("output image", "output image")
        self.record_selector.currentIndexChanged.connect(self.refresh_ui)
        self.viewer_layout.addWidget(self.record_selector)
        self.viewer_layout.addWidget(self.viewer)

        # ── ROI ──
        from falcocv.ui.widgets.roi_editor import RoiEditorWidget
        self.roi_editor = RoiEditorWidget(param_key="roi")
        left_outer.addWidget(self.roi_editor)

        # ── Mode ──
        mode_group = QGroupBox("OCR Mode")
        mode_lyt = QVBoxLayout(mode_group)
        self.cmb_mode = QComboBox()
        self.cmb_mode.addItems(["Read", "Verify", "Locate"])
        self.cmb_mode.setCurrentText(self.tool.params.get("mode", "Read"))
        self.cmb_mode.currentTextChanged.connect(self._on_mode_changed)
        mode_lyt.addWidget(self.cmb_mode)
        left_outer.addWidget(mode_group)

        # ── Model type ──
        model_group = QGroupBox("Model")
        model_lyt = QVBoxLayout(model_group)
        self.cmb_model = QComboBox()
        self.cmb_model.addItems(["Server", "Mobile"])
        self.cmb_model.setCurrentText(
            "Server" if self.tool.params.get("model_type", "server") == "server" else "Mobile"
        )
        self.cmb_model.currentTextChanged.connect(self._on_model_changed)
        model_lyt.addWidget(self.cmb_model)
        left_outer.addWidget(model_group)

        # ── Verify settings ──
        verify_group = QGroupBox("Verify Settings")
        verify_lyt = QVBoxLayout(verify_group)
        self.txt_expected = QLineEdit()
        self.txt_expected.setPlaceholderText("Enter expected text...")
        self.txt_expected.setText(self.tool.params.get("expected_text", ""))
        self.txt_expected.textChanged.connect(
            lambda v: self.tool.params.__setitem__("expected_text", v) or self._debounce_timer.start()
        )
        verify_lyt.addWidget(QLabel("Expected text:"))
        verify_lyt.addWidget(self.txt_expected)
        left_outer.addWidget(verify_group)

        # ── Recognition params ──
        rec_group = QGroupBox("Recognition Parameters")
        rec_lyt = QVBoxLayout(rec_group)
        self.chk_split = QCheckBox("Multi-line (auto-split by projection)")
        self.chk_split.setChecked(self.tool.params.get("split_lines", False))
        self.chk_split.toggled.connect(lambda c: self._set_param("split_lines", c))
        rec_lyt.addWidget(self.chk_split)
        s = QDoubleSpinBox()
        s.setRange(0.0, 1.0)
        s.setSingleStep(0.05)
        s.setValue(self.tool.params.get("min_region_confidence", 0.5))
        s.valueChanged.connect(lambda v: self._set_param("min_region_confidence", v))
        h = QHBoxLayout(); h.addWidget(QLabel("Min confidence:")); h.addWidget(s)
        rec_lyt.addLayout(h)
        left_outer.addWidget(rec_group)

        # ── Visibility ──
        vis_group = QGroupBox("Visibility")
        vis_lyt = QVBoxLayout(vis_group)
        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(True)
        self.chk_overlays.toggled.connect(self.refresh_ui)
        vis_lyt.addWidget(self.chk_overlays)
        self.chk_search_roi = QCheckBox("Show Search Region")
        self.chk_search_roi.setChecked(self.tool.params.get("show_search_roi", True))
        self.chk_search_roi.toggled.connect(lambda c: self.tool.params.__setitem__("show_search_roi", c) or self.refresh_ui())
        vis_lyt.addWidget(self.chk_search_roi)
        self.chk_regions = QCheckBox("Show Text Regions")
        self.chk_regions.setChecked(self.tool.params.get("show_regions", True))
        self.chk_regions.toggled.connect(lambda c: self.tool.params.__setitem__("show_regions", c) or self.refresh_ui())
        vis_lyt.addWidget(self.chk_regions)
        self.chk_labels = QCheckBox("Show Labels")
        self.chk_labels.setChecked(self.tool.params.get("show_labels", True))
        self.chk_labels.toggled.connect(lambda c: self.tool.params.__setitem__("show_labels", c) or self.refresh_ui())
        vis_lyt.addWidget(self.chk_labels)
        self.chk_found_origin = QCheckBox("Show Found Origin")
        self.chk_found_origin.setChecked(self.tool.params.get("show_found_origin", True))
        self.chk_found_origin.toggled.connect(lambda c: self.tool.params.__setitem__("show_found_origin", c) or self.refresh_ui())
        vis_lyt.addWidget(self.chk_found_origin)
        self.chk_copy = QCheckBox("Copy overlays to input image")
        self.chk_copy.setChecked(self.tool.params.get("copy_overlays_to_input", True))
        self.chk_copy.toggled.connect(lambda c: self.tool.params.__setitem__("copy_overlays_to_input", c) or self._set_param("copy_overlays_to_input", c))
        vis_lyt.addWidget(self.chk_copy)
        left_outer.addWidget(vis_group)
        left_outer.addStretch()

        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)
        self.content_layout.addWidget(self.right_panel)
        self.main_layout.addWidget(self.content_layout, 1)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "OCR"
        )

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        self.roi_editor.set_tool(self.tool, self.viewer)
        self.tool.data_changed.connect(self.refresh_ui)
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))
        self._suppress_execute = False

    def _set_param(self, key, value):
        self.tool.params[key] = value
        self._debounce_timer.start()

    def _on_mode_changed(self, text: str):
        self.tool.params["mode"] = text
        self._debounce_timer.start()

    def _on_model_changed(self, text: str):
        model_type = "server" if text == "Server" else "mobile"
        if model_type != self.tool.params.get("model_type"):
            self.tool._engine = None
        self.tool.params["model_type"] = model_type
        self._debounce_timer.start()

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        is_input = key == "input image"

        if is_input:
            img = self.tool.inputs.get("input image")
            if isinstance(img, VisionImage):
                self.viewer.set_image(img, auto_fit=auto_fit,
                                      render_overlays_flag=self.chk_overlays.isChecked())
                roi_data = self.tool.params.get("roi", [0, 0, 0, 0, 0])
                if roi_data[2] > 0 and roi_data[3] > 0:
                    if isinstance(img, VisionImage):
                        abs_roi = img.fixture_to_image(roi_data)
                    else:
                        abs_roi = roi_data
                    self.viewer.set_roi(abs_roi)
                else:
                    self.viewer.set_roi(None)
        else:
            out = self.tool.outputs.get(key)
            if isinstance(out, VisionImage):
                self.viewer.set_image(out, auto_fit=auto_fit,
                                      render_overlays_flag=self.chk_overlays.isChecked())
            else:
                self.viewer.set_image(None)
            self.viewer.set_roi(None)

        if hasattr(self, "label_time"):
            self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "OCR")
