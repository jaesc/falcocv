import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool, ToolResult, ToolResults
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter,
                               QGroupBox, QPushButton, QComboBox, QSpinBox,
                               QCheckBox, QFrame)
from PyQt6.QtCore import Qt, QRectF, QTimer, QPointF
from PyQt6.QtGui import (QPainter, QPixmap, QPen, QBrush, QColor, QFont,
                         QPolygonF, QImage)
from falcocv.ui.style import TEXT, TEXT_MUTED, HIST_BG, HIST_BAR, HIST_BAR_BORDER, HIST_STAT, HIST_LABEL, TOOLTIP_BG, disabled_css
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)
from falcocv.ui.widgets.viewer import ImageViewer


from .logic import HistogramResult, HistogramTool
class HistogramResult(ToolResult):
    def __init__(self, mean, std_dev, min_val, max_val, median_val, bins):
        super().__init__(
            mean=float(mean), std_dev=float(std_dev),
            min_val=int(min_val), max_val=int(max_val),
            median_val=float(median_val), bins=int(bins)
        )


class HistogramConfig(QWidget):
    def __init__(self, tool: HistogramTool):
        super().__init__()
        self.tool = tool

        # Main Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        # Content Layout
        self.content_layout = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT SIDE: CONTROLS ---
        self.left_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.left_panel)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)

        # --- ROI ---
        from falcocv.ui.widgets.roi_editor import RoiEditorWidget
        self.roi_editor = RoiEditorWidget()
        self.controls_layout.addWidget(self.roi_editor)

        # --- Histogram Settings ---
        settings_group = QGroupBox("Histogram Settings")
        settings_lyt = QVBoxLayout(settings_group)

        # BinMode
        bin_mode_lyt = QHBoxLayout()
        bin_mode_lyt.addWidget(QLabel("BinMode:"))
        self.combo_bin_mode = QComboBox()
        self.combo_bin_mode.addItems(["Auto", "Fixed"])
        self.combo_bin_mode.setCurrentText(self.tool.params["bin_mode"])
        self.combo_bin_mode.currentTextChanged.connect(self._on_bin_mode_changed)
        bin_mode_lyt.addWidget(self.combo_bin_mode)
        bin_mode_lyt.addStretch()
        settings_lyt.addLayout(bin_mode_lyt)

        # NumBins
        num_bins_lyt = QHBoxLayout()
        num_bins_lyt.addWidget(QLabel("NumBins:"))
        self.spin_num_bins = QSpinBox()
        self.spin_num_bins.setRange(1, 65536)
        self.spin_num_bins.setValue(self.tool.params["num_bins"])
        self.spin_num_bins.valueChanged.connect(self._on_num_bins_changed)
        num_bins_lyt.addWidget(self.spin_num_bins)
        num_bins_lyt.addStretch()
        settings_lyt.addLayout(num_bins_lyt)

        # Disable NumBins if Auto
        self._update_num_bins_enabled()

        # Separator line
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        line.setFrameShadow(QFrame.Shadow.Sunken)
        settings_lyt.addWidget(line)

        # Enable output image checkbox
        self.chk_enable_output = QCheckBox("Enable Output Image")
        self.chk_enable_output.setChecked(self.tool.params.get("enable_output_image", True))
        self.chk_enable_output.toggled.connect(self._on_enable_output_toggled)
        settings_lyt.addWidget(self.chk_enable_output)

        # Output resolution
        res_lyt = QHBoxLayout()
        res_lyt.addWidget(QLabel("Width:"))
        self.spin_out_w = QSpinBox()
        self.spin_out_w.setRange(200, 4000)
        self.spin_out_w.setValue(self.tool.params.get("output_width", 800))
        self.spin_out_w.valueChanged.connect(lambda v: self._on_res_changed("output_width", v))
        res_lyt.addWidget(self.spin_out_w)
        res_lyt.addWidget(QLabel("Height:"))
        self.spin_out_h = QSpinBox()
        self.spin_out_h.setRange(200, 4000)
        self.spin_out_h.setValue(self.tool.params.get("output_height", 600))
        self.spin_out_h.valueChanged.connect(lambda v: self._on_res_changed("output_height", v))
        res_lyt.addWidget(self.spin_out_h)
        settings_lyt.addLayout(res_lyt)

        settings_group.setStyleSheet(f"QGroupBox {{ color: {TEXT}; }} QGroupBox::title {{ subcontrol-position: top left; }}")
        self.controls_layout.addWidget(settings_group)

        self.controls_layout.addStretch()
        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)

        # --- RIGHT SIDE: VIEWER ---
        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)

        header_lyt = QHBoxLayout()
        self.record_selector = QComboBox()
        self.record_selector.addItem("Input Image", "input image")
        self.record_selector.addItem("Histogram Image", "output image")
        self.record_selector.setCurrentIndex(0)
        self.record_selector.currentIndexChanged.connect(self.on_record_changed)
        header_lyt.addWidget(self.record_selector)
        self.viewer_layout.addLayout(header_lyt)

        self.viewer = ImageViewer()
        if hasattr(self, "roi_editor"):
            self.roi_editor.set_tool(self.tool, self.viewer)
        self.viewer_layout.addWidget(self.viewer)

        self.content_layout.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "HistogramTool"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))

        # Initial state
        QTimer.singleShot(50, self.on_record_changed)
        QTimer.singleShot(500, self.on_record_changed)

    def _on_bin_mode_changed(self, text: str):
        self.tool.params["bin_mode"] = text
        self._update_num_bins_enabled()
        self.tool.mark_dirty()

    def _update_num_bins_enabled(self):
        enabled = self.tool.params["bin_mode"] == "Fixed"
        self.spin_num_bins.setEnabled(enabled)

    def _on_num_bins_changed(self, value: int):
        self.tool.params["num_bins"] = value
        self.tool.mark_dirty()

    def _on_enable_output_toggled(self, checked):
        self.tool.params["enable_output_image"] = checked
        self.spin_out_w.setEnabled(checked)
        self.spin_out_h.setEnabled(checked)
        self.tool.mark_dirty()

    def _on_res_changed(self, key, value):
        self.tool.params[key] = value
        self.tool.mark_dirty()

    def _set_param(self, key, value):
        self.tool.params[key] = value
        self.tool.mark_dirty()

    def on_record_changed(self):
        if self.record_selector.currentData() != "input image":
            if hasattr(self, "roi_editor"):
                self.roi_editor.btn_edit.setChecked(False)
            self.viewer.set_roi_editing(False)
        self.refresh_ui()

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if not key: return

        is_input = (key == "input image")
        img = self.tool.inputs.get("input image") if is_input else self.tool.outputs.get(key)

        # Security Check: Only images go to the viewer
        if isinstance(img, np.ndarray):
            self.viewer.set_image(img, auto_fit=auto_fit)
            self.viewer.set_roi(None)
        elif isinstance(img, VisionImage):
            self.viewer.set_image(img, auto_fit=auto_fit)
            # ROI Logic: only show on input image
            if is_input:
                roi_data = self.tool.params["roi"]
                if roi_data[2] > 0 and roi_data[3] > 0:
                    # Convert fixture-relative ROI to absolute image coordinates for display
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

        # Update common ROI widget
        if hasattr(self, "roi_editor"):
            self.roi_editor.update_from_params()

    def showEvent(self, event):
        super().showEvent(event)
        # Ensure image is refreshed and fitted once the window is actually visible
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "HistogramTool")
