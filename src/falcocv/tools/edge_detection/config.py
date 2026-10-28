"""
Edge Detection Tool — Affine 1D caliper for industrial subpixel edge detection.

Algorithm: samples a rotated ROI band directly from the source image using
cv2.remap with bilinear interpolation, projects into a 1D intensity profile
by averaging along the longitudinal axis, applies a derivative-of-gaussian,
and detects peaks by polarity. Subpixel refinement via parabolic interpolation.

Reconstruction: P = center + v * edge_position

Inspired by Cognex VisionPro caliper tools.
"""

import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool, ToolResult, ToolResults
from falcocv.core.vision_image import VisionImage
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel, QSplitter,
                             QComboBox, QCheckBox, QGroupBox, QSpinBox,
                             QDoubleSpinBox, QPushButton)
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


from .logic import EdgeDetection, EdgePairResult, SingleEdgeResult
class EdgePairResult(ToolResult):
    def __init__(self, id, x1, y1, x2, y2, width, score1, score2):
        super().__init__(id=id, x1=x1, y1=y1, x2=x2, y2=y2,
                         width=width, score1=score1, score2=score2)


class SingleEdgeResult(ToolResult):
    def __init__(self, id, x, y, score, polarity):
        super().__init__(id=id, x=x, y=y, score=score, polarity=polarity)


class EdgeDetectionConfig(QWidget):
    def __init__(self, tool: EdgeDetection):
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
            self.content_layout, self.left_scroll, self.right_panel, "EdgeDetection"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))
        self._update_visibility()

    def _build_controls(self):
        from falcocv.ui.widgets.roi_editor import RoiEditorWidget

        self.roi_editor = RoiEditorWidget()
        self.controls_layout.addWidget(self.roi_editor)

        mode_group = QGroupBox("Detection Mode")
        mode_lyt = QVBoxLayout(mode_group)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Single Edge", "Edge Pair"])
        self.combo_mode.setCurrentText(self.tool.params["detect_mode"])
        self.combo_mode.currentTextChanged.connect(self._on_mode_changed)
        mode_lyt.addWidget(QLabel("Mode:"))
        mode_lyt.addWidget(self.combo_mode)
        self.controls_layout.addWidget(mode_group)

        polarity_group = QGroupBox("Edge Polarity")
        p_lyt = QVBoxLayout(polarity_group)

        self.lbl_e1 = QLabel("Edge 1 Polarity:")
        self.combo_e1 = QComboBox()
        self.combo_e1.addItems(["Dark\u2192Light", "Light\u2192Dark", "Any"])
        self.combo_e1.setCurrentText(self.tool.params["edge1_polarity"])
        self.combo_e1.currentTextChanged.connect(lambda v: self._set_param("edge1_polarity", v))
        p_lyt.addWidget(self.lbl_e1)
        p_lyt.addWidget(self.combo_e1)

        self.lbl_e2 = QLabel("Edge 2 Polarity:")
        self.combo_e2 = QComboBox()
        self.combo_e2.addItems(["Dark\u2192Light", "Light\u2192Dark", "Any"])
        self.combo_e2.setCurrentText(self.tool.params["edge2_polarity"])
        self.combo_e2.currentTextChanged.connect(lambda v: self._set_param("edge2_polarity", v))
        p_lyt.addWidget(self.lbl_e2)
        p_lyt.addWidget(self.combo_e2)

        self.lbl_single = QLabel("Single Edge Polarity:")
        self.combo_single = QComboBox()
        self.combo_single.addItems(["Dark\u2192Light", "Light\u2192Dark", "Any"])
        self.combo_single.setCurrentText(self.tool.params["single_polarity"])
        self.combo_single.currentTextChanged.connect(lambda v: self._set_param("single_polarity", v))
        p_lyt.addWidget(self.lbl_single)
        p_lyt.addWidget(self.combo_single)

        self.controls_layout.addWidget(polarity_group)

        sel_group = QGroupBox("Edge Selector")
        sel_lyt = QVBoxLayout(sel_group)
        self.combo_selector = QComboBox()
        self.combo_selector.addItems(["First", "Last", "Strongest"])
        self.combo_selector.setCurrentText(self.tool.params["edge_selector"])
        self.combo_selector.currentTextChanged.connect(lambda v: self._set_param("edge_selector", v))
        sel_lyt.addWidget(QLabel("Select:"))
        sel_lyt.addWidget(self.combo_selector)
        self.lbl_selector = sel_group
        self.controls_layout.addWidget(sel_group)

        thresh_group = QGroupBox("Edge Threshold")
        t_lyt = QVBoxLayout(thresh_group)
        self.spin_thresh = QSpinBox()
        self.spin_thresh.setRange(1, 255)
        self.spin_thresh.setValue(self.tool.params["edge_threshold"])
        self.spin_thresh.valueChanged.connect(lambda v: self._set_param("edge_threshold", v))
        t_lyt.addWidget(QLabel("Threshold:"))
        t_lyt.addWidget(self.spin_thresh)

        self.lbl_sigma = QLabel("Gaussian Sigma:")
        self.spin_sigma = QDoubleSpinBox()
        self.spin_sigma.setRange(0.1, 5.0)
        self.spin_sigma.setSingleStep(0.1)
        self.spin_sigma.setDecimals(1)
        self.spin_sigma.setValue(self.tool.params.get("gaussian_sigma", 1.0))
        self.spin_sigma.valueChanged.connect(lambda v: self._set_param("gaussian_sigma", v))
        t_lyt.addWidget(self.lbl_sigma)
        t_lyt.addWidget(self.spin_sigma)

        self.chk_subpixel = QCheckBox("Sub-pixel Interpolation")
        self.chk_subpixel.setChecked(self.tool.params["subpixel"])
        self.chk_subpixel.toggled.connect(lambda v: self._set_param("subpixel", v))
        t_lyt.addWidget(self.chk_subpixel)
        self.controls_layout.addWidget(thresh_group)

        self.width_group = QGroupBox("Edge Pair Width Filter")
        w_lyt = QVBoxLayout(self.width_group)
        self.spin_min_w = QSpinBox()
        self.spin_min_w.setRange(1, 9999)
        self.spin_min_w.setValue(self.tool.params["min_edge_width"])
        self.spin_min_w.valueChanged.connect(lambda v: self._set_param("min_edge_width", v))
        w_lyt.addWidget(QLabel("Min Width (px):"))
        w_lyt.addWidget(self.spin_min_w)

        self.spin_max_w = QSpinBox()
        self.spin_max_w.setRange(1, 9999)
        self.spin_max_w.setValue(self.tool.params["max_edge_width"])
        self.spin_max_w.valueChanged.connect(lambda v: self._set_param("max_edge_width", v))
        w_lyt.addWidget(QLabel("Max Width (px):"))
        w_lyt.addWidget(self.spin_max_w)
        self.controls_layout.addWidget(self.width_group)

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

        self.chk_arrow = QCheckBox("Show Direction Arrow")
        self.chk_arrow.setChecked(self.tool.params.get("show_direction_arrow", True))
        self.chk_arrow.toggled.connect(lambda v: (self._set_param("show_direction_arrow", v), self.viewer.set_roi_direction_visible(v)))
        vis_lyt.addWidget(self.chk_arrow)

        self.chk_lines = QCheckBox("Show Edge Lines")
        self.chk_lines.setChecked(self.tool.params.get("show_edge_lines", True))
        self.chk_lines.toggled.connect(lambda v: self._set_param("show_edge_lines", v))
        vis_lyt.addWidget(self.chk_lines)

        self.chk_points = QCheckBox("Show Edge Points")
        self.chk_points.setChecked(self.tool.params.get("show_edge_points", True))
        self.chk_points.toggled.connect(lambda v: self._set_param("show_edge_points", v))
        vis_lyt.addWidget(self.chk_points)

        self.chk_labels = QCheckBox("Show Labels")
        self.chk_labels.setChecked(self.tool.params.get("show_labels", True))
        self.chk_labels.toggled.connect(lambda v: self._set_param("show_labels", v))
        vis_lyt.addWidget(self.chk_labels)

        self.chk_width = QCheckBox("Show Width Line (Pair)")
        self.chk_width.setChecked(self.tool.params.get("show_width_line", True))
        self.chk_width.toggled.connect(lambda v: self._set_param("show_width_line", v))
        vis_lyt.addWidget(self.chk_width)

        self.chk_copy_in = QCheckBox("Copy overlays to input image")
        self.chk_copy_in.setChecked(self.tool.params.get("copy_overlays_to_input", True))
        self.chk_copy_in.toggled.connect(lambda v: self._set_param("copy_overlays_to_input", v))
        vis_lyt.addWidget(self.chk_copy_in)

        self.controls_layout.addWidget(vis_group)
        self.controls_layout.addStretch()

    def _build_viewer(self):
        self.record_selector = QComboBox()
        self.record_selector.addItem("input image", "input image")
        self.record_selector.addItem("output image", "output image")
        self.record_selector.currentIndexChanged.connect(self._on_record_changed)
        self.viewer_layout.addWidget(self.record_selector)

        self.viewer = ImageViewer()
        self.roi_editor.set_tool(self.tool, self.viewer)
        self.viewer.set_roi_direction_visible(True)
        self.destroyed.connect(lambda _: self.viewer.set_roi_direction_visible(False))
        self.viewer_layout.addWidget(self.viewer)

        self.roi_editor.set_tool(self.tool, self.viewer)
        self.viewer.set_roi_direction_visible(True)
        self.destroyed.connect(lambda _: self.viewer.set_roi_direction_visible(False))

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if not key:
            return

        is_input = (key == "input image")
        img = self.tool.inputs.get("input image") if is_input else self.tool.outputs.get(key)

        from falcocv.core.vision_image import VisionImage
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

        if hasattr(self, "roi_editor"):
            disabled = disabled_css()
            self.roi_editor.setEnabled(is_input)
            self.roi_editor.setStyleSheet(disabled if not is_input else "")
            if not is_input:
                self.roi_editor.btn_edit.setChecked(False)
                self.viewer.set_roi_editing(False)

    def _on_record_changed(self):
        if self.record_selector.currentData() != "input image":
            if hasattr(self, "roi_editor"):
                self.roi_editor.btn_edit.setChecked(False)
            self.viewer.set_roi_editing(False)
        self.refresh_ui()

    def _on_mode_changed(self, v):
        self.tool.params["detect_mode"] = v
        self._update_visibility()
        self.tool.mark_dirty()

    def _set_param(self, k, v):
        self.tool.params[k] = v
        self.tool.mark_dirty()

    def _on_overlays_toggled(self, v):
        self.refresh_ui()

    def _update_visibility(self):
        is_pair = self.combo_mode.currentText() == "Edge Pair"
        self.lbl_e1.setVisible(is_pair)
        self.combo_e1.setVisible(is_pair)
        self.lbl_e2.setVisible(is_pair)
        self.combo_e2.setVisible(is_pair)
        self.lbl_single.setVisible(not is_pair)
        self.combo_single.setVisible(not is_pair)
        self.lbl_selector.setVisible(not is_pair)
        self.combo_selector.setVisible(not is_pair)
        self.width_group.setVisible(is_pair)

    def set_tool(self, tool, viewer):
        self.roi_editor.set_tool(tool, self.viewer)

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "EdgeDetection")
