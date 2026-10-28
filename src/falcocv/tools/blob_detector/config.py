import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool, ToolResult, ToolResults
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi, create_roi_only_output
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter,
                             QSlider, QComboBox, QCheckBox, QGroupBox, QSpinBox, QPushButton)
from PyQt6.QtCore import Qt, QRectF, QTimer
from falcocv.ui.style import disabled_css
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)
from falcocv.ui.widgets.viewer import ImageViewer


from .logic import BlobDetector, BlobResult
class BlobResult(ToolResult):
    """Specific result for a Blob detection."""
    def __init__(self, id, x, y, area):
        super().__init__(id=id, x=x, y=y, area=area)


class BlobDetectorConfig(QWidget):
    def __init__(self, tool: BlobDetector):
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

        # --- MODE & POLARITY ---
        cfg_group = QGroupBox("Core Config")
        cfg_lyt = QVBoxLayout(cfg_group)
        self.combo_out = QComboBox()
        self.combo_out.addItems(["Full Image", "ROI Only"])
        self.combo_out.setCurrentText(self.tool.params["output_mode"])
        self.combo_out.currentTextChanged.connect(self._on_out_mode_changed)
        cfg_lyt.addWidget(QLabel("Output Strategy:"))
        cfg_lyt.addWidget(self.combo_out)
        
        self.combo_pol = QComboBox()
        self.combo_pol.addItems(["Bright Blobs", "Dark Blobs"])
        self.combo_pol.setCurrentText(self.tool.params["polarity"])
        self.combo_pol.currentTextChanged.connect(self._on_pol_changed)
        cfg_lyt.addWidget(QLabel("Target Type:"))
        cfg_lyt.addWidget(self.combo_pol)
        self.controls_layout.addWidget(cfg_group)

        # --- SEGMENTATION ---
        thresh_group = QGroupBox("Segmentation")
        t_lyt = QVBoxLayout(thresh_group)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Fixed", "Adaptive"])
        self.combo_mode.setCurrentText(self.tool.params["threshold_mode"])
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        t_lyt.addWidget(QLabel("Mode:"))
        t_lyt.addWidget(self.combo_mode)
        
        self.fixed_widget = QWidget()
        f_lyt = QVBoxLayout(self.fixed_widget)
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 255)
        self.slider.setValue(self.tool.params["threshold_value"])
        self.slider.valueChanged.connect(self._on_thresh_changed)
        self.label_thresh = QLabel(f"Threshold: {self.tool.params['threshold_value']}")
        f_lyt.addWidget(self.label_thresh)
        f_lyt.addWidget(self.slider)
        t_lyt.addWidget(self.fixed_widget)

        self.adaptive_widget = QWidget()
        a_lyt = QVBoxLayout(self.adaptive_widget)
        self.spin_bs = QSpinBox()
        self.spin_bs.setRange(3, 99)
        self.spin_bs.setSingleStep(2)
        self.spin_bs.setValue(self.tool.params["adaptive_block_size"])
        self.spin_bs.valueChanged.connect(self._on_bs_changed)
        a_lyt.addWidget(QLabel("Block Size:"))
        a_lyt.addWidget(self.spin_bs)
        t_lyt.addWidget(self.adaptive_widget)
        self.controls_layout.addWidget(thresh_group)

        # --- FILTER RESULTS ---
        f_group = QGroupBox("Filter Results")
        f_lyt = QVBoxLayout(f_group)
        self.spin_min = QSpinBox()
        self.spin_min.setRange(0, 10000000)
        self.spin_min.setValue(int(self.tool.params["area_min"]))
        self.spin_min.valueChanged.connect(self._on_min_changed)
        f_lyt.addWidget(QLabel("Min Area:"))
        f_lyt.addWidget(self.spin_min)

        self.spin_max = QSpinBox()
        self.spin_max.setRange(0, 10000000)
        self.spin_max.setValue(int(self.tool.params["area_max"]))
        self.spin_max.valueChanged.connect(self._on_max_changed)
        f_lyt.addWidget(QLabel("Max Area:"))
        f_lyt.addWidget(self.spin_max)

        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["None", "Area Ascending", "Area Descending", "X Ascending", "Y Ascending"])
        self.combo_sort.setCurrentText(self.tool.params["sort_mode"])
        self.combo_sort.currentTextChanged.connect(self._on_sort_changed)
        f_lyt.addWidget(QLabel("Sort By:"))
        f_lyt.addWidget(self.combo_sort)
        self.controls_layout.addWidget(f_group)

        # --- VISIBILITY ---
        vis_group = QGroupBox("Visibility")
        vis_lyt = QVBoxLayout(vis_group)
        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(True)
        self.chk_overlays.toggled.connect(self._on_overlays_toggled)
        vis_lyt.addWidget(self.chk_overlays)
        self.chk_search_roi = QCheckBox("Show Search Region")
        self.chk_search_roi.setChecked(self.tool.params.get("show_search_roi", True))
        self.chk_search_roi.toggled.connect(lambda v: self._set_param("show_search_roi", v))
        vis_lyt.addWidget(self.chk_search_roi)
        self.chk_contours = QCheckBox("Show Boundaries")
        self.chk_contours.setChecked(self.tool.params["show_contours"])
        self.chk_contours.toggled.connect(lambda v: self._set_param("show_contours", v))
        vis_lyt.addWidget(self.chk_contours)
        self.chk_centroids = QCheckBox("Show Centroids")
        self.chk_centroids.setChecked(self.tool.params["show_centroids"])
        self.chk_centroids.toggled.connect(lambda v: self._set_param("show_centroids", v))
        vis_lyt.addWidget(self.chk_centroids)
        self.chk_labels = QCheckBox("Show Labels")
        self.chk_labels.setChecked(self.tool.params["show_labels"])
        self.chk_labels.toggled.connect(lambda v: self._set_param("show_labels", v))
        vis_lyt.addWidget(self.chk_labels)
        self.chk_copy_in = QCheckBox("Copy overlays to input image")
        self.chk_copy_in.setChecked(self.tool.params.get("copy_overlays_to_input", True))
        self.chk_copy_in.toggled.connect(lambda v: self._set_param("copy_overlays_to_input", v))
        vis_lyt.addWidget(self.chk_copy_in)
        self.controls_layout.addWidget(vis_group)

        self.controls_layout.addStretch()
        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)

        # --- RIGHT SIDE: VIEWER ---
        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.record_selector = QComboBox()
        self.record_selector.addItem("input image", "input image")
        self.record_selector.addItem("binary mask", "binary mask")
        self.record_selector.addItem("output image", "output image")
        self.record_selector.currentIndexChanged.connect(self.on_record_changed)
        self.viewer_layout.addWidget(self.record_selector)

        self.viewer = ImageViewer()
        if hasattr(self, "roi_editor"):
            self.roi_editor.set_tool(self.tool, self.viewer)
        self.viewer_layout.addWidget(self.viewer)

        self.content_layout.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "BlobDetector"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)
        
        self.tool.data_changed.connect(self.refresh_ui)
        self._update_visibility()
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if not key: return

        is_input = (key == "input image")
        img = self.tool.inputs.get("input image") if is_input else self.tool.outputs.get(key)

        # Security Check: Only images go to the viewer
        from falcocv.core.vision_image import VisionImage
        if isinstance(img, np.ndarray):
            self.viewer.set_image(img, auto_fit=auto_fit)
            self.viewer.set_roi(None)
        elif isinstance(img, VisionImage):
            self.viewer.set_image(img, auto_fit=auto_fit, render_overlays_flag=self.chk_overlays.isChecked())
            # ROI Logic: only show on input image
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

        # Enable/disable ROI editor based on selected image (like pattern matcher)
        disabled = disabled_css()
        if hasattr(self, "roi_editor"):
            self.roi_editor.setEnabled(is_input)
            self.roi_editor.setStyleSheet(disabled if not is_input else "")
            if not is_input:
                self.roi_editor.btn_edit.setChecked(False)
                self.viewer.set_roi_editing(False)

    def on_record_changed(self):
        if self.record_selector.currentData() != "input image": 
            if hasattr(self, "roi_editor"): self.roi_editor.btn_edit.setChecked(False)
            self.viewer.set_roi_editing(False)
        self.refresh_ui()

    def _on_thresh_changed(self, v): 
        self.tool.params["threshold_value"] = v
        self.label_thresh.setText(f"Threshold: {v}")
        self.tool.mark_dirty()

    def _on_mode_changed(self, v): 
        self.tool.params["threshold_mode"] = v
        self._update_visibility()
        self.tool.mark_dirty()

    def _on_bs_changed(self, v): self.tool.params["adaptive_block_size"] = v; self.tool.mark_dirty()
    def _on_pol_changed(self, v): self.tool.params["polarity"] = v; self.tool.mark_dirty()
    def _on_min_changed(self, v): self.tool.params["area_min"] = v; self.tool.mark_dirty()
    def _on_max_changed(self, v): self.tool.params["area_max"] = v; self.tool.mark_dirty()
    def _on_sort_changed(self, v): self.tool.params["sort_mode"] = v; self.tool.mark_dirty()
    def _on_out_mode_changed(self, v): self.tool.params["output_mode"] = v; self.tool.mark_dirty()
    def _on_overlays_toggled(self, v): self.refresh_ui()
    def _set_param(self, k, v): self.tool.params[k] = v; self.tool.mark_dirty()
    def _update_visibility(self):
        is_fixed = self.combo_mode.currentText() == "Fixed"
        self.fixed_widget.setVisible(is_fixed)
        self.adaptive_widget.setVisible(not is_fixed)

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "BlobDetector")
        self.tool.data_changed.disconnect(self.refresh_ui)
