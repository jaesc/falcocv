

from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QStackedWidget, 
                             QSplitter, QComboBox, QLabel, QSpinBox, QLineEdit,
                             QCheckBox, QMenu, QApplication)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer, QPointF, QPoint, QRect
from PyQt6.QtGui import QIcon, QPainter, QPixmap, QPen, QBrush, QPolygonF, QColor
from falcocv.ui.widgets.viewer import ImageViewer
try:
    from falcocv.ui.widgets.viewer3d import Viewer3D
except ImportError:
    Viewer3D = None
from falcocv.ui.widgets.workflow_tree import WorkflowTree
from falcocv.ui.widgets.toolbox import Toolbox
from falcocv.ui.style import ALT_BG, ARROW_COLOR, BORDER, FOOTER_BG, HIGHLIGHT, PANEL_BG, SUCCESS, TEXT, TEXT_MUTED, WARNING
from falcocv.core.workflow import Workflow
from falcocv.core.tool_registry import get_registration
from falcocv.core.workflow_controller import WorkflowController, ControllerState
from falcocv.tools.image_loader import ImageLoader
from falcocv.tools.depth_range_loader import DepthRangeLoader
import sys
import numpy as np
from falcocv.core.depth_viz import should_offer_depth_controls, map_depth_to_rgb
from falcocv.core.pointcloud_utils import axis_aligned_bounding_box

class WorkflowEditor(QWidget):
    _global_show_overlays = True
    workflow_changed = pyqtSignal()

    def __init__(self, workflow=None, parent=None):
        super().__init__(parent)
        self.controller = WorkflowController(workflow or Workflow())
        self.workflow = self.controller.workflow
        self._running_workflow = False
        self.controller.set_scheduler(lambda fn: QTimer.singleShot(0, fn))
        from falcocv.core.config import ConfigManager
        WorkflowEditor._global_show_overlays = ConfigManager().get_show_overlays()
        self.controller.execution_finished.connect(self._on_execution_finished)
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Left Panel: Tree + Toolbox
        tree_container = QWidget()
        tree_layout = QVBoxLayout(tree_container)
        
        self.tree = WorkflowTree()
        self.tree.tool_selected.connect(self.on_tool_selected)
        self.tree.order_changed.connect(self.on_reorder)
        self.tree.tool_deleted.connect(self.on_deletion)
        self.tree.tool_dropped.connect(self.handle_add_tool)
        self.tree.tool_pasted.connect(self.on_tool_pasted)

        tree_header = QHBoxLayout()
        tree_header.setContentsMargins(0, 0, 0, 0)
        tree_header.setSpacing(2)

        def _arrow_icon(up=True):
            wi, hi = 16, 11
            pm = QPixmap(wi, hi)
            pm.fill(Qt.GlobalColor.transparent)
            p = QPainter(pm)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(QPen(ARROW_COLOR, 1))
            p.setBrush(QBrush(ARROW_COLOR))
            if up:
                poly = QPolygonF([QPointF(wi/2, 1), QPointF(2, hi-2), QPointF(wi-2, hi-2)])
            else:
                poly = QPolygonF([QPointF(wi/2, hi-2), QPointF(2, 2), QPointF(wi-2, 2)])
            p.drawPolygon(poly)
            p.end()
            return QIcon(pm)

        btn_collapse = QPushButton()
        btn_collapse.setIcon(_arrow_icon(up=True))
        btn_collapse.setFixedSize(18, 20)
        btn_collapse.setToolTip("Collapse All")
        btn_collapse.clicked.connect(self.tree.collapseAll)
        tree_header.addWidget(btn_collapse)
        btn_expand = QPushButton()
        btn_expand.setIcon(_arrow_icon(up=False))
        btn_expand.setFixedSize(18, 20)
        btn_expand.setToolTip("Expand All")
        btn_expand.clicked.connect(self.tree.expandAll)
        tree_header.addWidget(btn_expand)
        tree_header.addStretch()
        tree_layout.addLayout(tree_header)

        tree_layout.addWidget(self.tree)
        
        from falcocv.ui.widgets.toolbox import Toolbox
        Toolbox.set_active_editor(self)
        
        self.splitter.addWidget(tree_container)
        
        # Right Panel: Viewer
        viewer_container = QWidget()
        viewer_layout = QVBoxLayout(viewer_container)
        
        header = QHBoxLayout()
        
        header = QHBoxLayout()
        header.setSpacing(5)
        header.setContentsMargins(5, 5, 5, 5)
        
        self.btn_refresh = QPushButton("REFRESH")
        self.btn_refresh.setAutoDefault(False)
        self.btn_refresh.setStyleSheet(f"font-weight: bold; color: {SUCCESS};")
        self.btn_refresh.setToolTip("Run once without advancing")
        self.btn_refresh.clicked.connect(lambda: self.run_workflow(advance=False))
        header.addWidget(self.btn_refresh)

        self.btn_step = QPushButton("RUN")
        self.btn_step.setStyleSheet(f"font-weight: bold; color: {SUCCESS};")
        self.btn_step.clicked.connect(lambda: self.run_workflow(advance=True))
        header.addWidget(self.btn_step)
        
        self.btn_loop = QPushButton("RUN LOOP")
        self.btn_loop.setStyleSheet(
            f"QPushButton {{ font-weight: bold; color: {SUCCESS}; }} "
            f"QPushButton:checked {{ background-color: {PANEL_BG}; border: 1px solid {HIGHLIGHT}; color: {SUCCESS}; }}"
        )
        self.btn_loop.setCheckable(True)
        self.btn_loop.clicked.connect(self.toggle_loop)
        header.addWidget(self.btn_loop)
        
        # Editable counter: [1/20]
        self.counter_widget = QWidget()
        from PyQt6.QtWidgets import QSizePolicy
        self.counter_widget.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        counter_layout = QHBoxLayout(self.counter_widget)
        counter_layout.setContentsMargins(5, 0, 5, 0)
        counter_layout.setSpacing(0)
        
        label_open = QLabel("[")
        label_open.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: bold;")
        counter_layout.addWidget(label_open)
        
        self.edit_index = QLineEdit()
        self.edit_index.setFixedWidth(35)
        self.edit_index.setStyleSheet(f"background: transparent; border: none; color: {TEXT}; font-weight: bold;")
        self.edit_index.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.edit_index.returnPressed.connect(self.on_index_edited)
        counter_layout.addWidget(self.edit_index)
        
        self.label_total_frames = QLabel("/-]")
        self.label_total_frames.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: bold;")
        counter_layout.addWidget(self.label_total_frames)
        header.addWidget(self.counter_widget)
        self.counter_widget.hide()
        
        self.record_selector = QComboBox()
        self.record_selector.setMinimumWidth(300)
        self.record_selector.currentIndexChanged.connect(self.update_viewer_image)
        header.addWidget(self.record_selector)

        self.chk_show3d = QCheckBox("Show 3D")
        self.chk_show3d.setVisible(False)
        self.chk_show3d.toggled.connect(self.update_viewer_image)
        header.addWidget(self.chk_show3d)

        header.addStretch()
        
        viewer_layout.addLayout(header)

        self.depth_bar = None
        
        self.viewer_stack = QStackedWidget()
        self.viewer_2d = ImageViewer()
        self.viewer_2d.set_roi_editing(False)
        self.viewer_stack.addWidget(self.viewer_2d)
        if Viewer3D is not None:
            self.viewer_3d = Viewer3D()
            self.viewer_stack.addWidget(self.viewer_3d)
        else:
            self.viewer_3d = None
        self.viewer = self.viewer_2d
        viewer_layout.addWidget(self.viewer_stack)
        
        self.splitter.addWidget(viewer_container)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 1)
        
        # Apply splitter proportions from config
        from falcocv.core.config import ConfigManager
        sizes = ConfigManager().get_editor_splitter()
        self.splitter.setSizes(sizes)
        
        layout.setSpacing(0)
        layout.addWidget(self.splitter, 1)
        
        # Footer Bar
        footer_widget = QWidget()
        footer_widget.setFixedHeight(20)
        footer_widget.setStyleSheet(f"background-color: {FOOTER_BG}; border-top: 1px solid {BORDER};")
        footer_layout = QHBoxLayout(footer_widget)
        footer_layout.setContentsMargins(5, 0, 5, 0)
        footer_layout.setSpacing(0)

        self.label_total_time = QLabel("0.0 ms")
        self.label_total_time.setStyleSheet(f"color: {SUCCESS}; font-weight: bold; font-size: 11px;")
        footer_layout.addWidget(self.label_total_time)

        # Error Label: uses all available central space
        self.label_error = QLabel("")
        self.label_error.setStyleSheet(f"color: {WARNING}; font-weight: bold; font-size: 11px;")
        self.label_error.setToolTip("")
        self.label_error.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.label_error.customContextMenuRequested.connect(self._show_error_context_menu)
        footer_layout.addWidget(self.label_error, 1) # Stretch factor 1

        self.label_coords = QLabel("X,Y: -, -")
        self.label_coords.setFixedWidth(130)
        self.label_coords.setStyleSheet(f"color: {TEXT_MUTED}; font-family: Consolas, monospace; font-size: 11px;")
        footer_layout.addWidget(self.label_coords)

        self.label_value = QLabel("VALUE: -")
        self.label_value.setFixedWidth(140)
        self.label_value.setStyleSheet(f"color: {TEXT_MUTED}; font-family: Consolas, monospace; font-size: 11px;")
        footer_layout.addWidget(self.label_value)

        self.label_web = QLabel("Web: ○")
        self.label_web.setMinimumWidth(120)
        self.label_web.setStyleSheet(f"color: {TEXT_MUTED}; font-family: Consolas, monospace; font-size: 10px;")
        footer_layout.addWidget(self.label_web)

        layout.addWidget(footer_widget)


    # ── Tool Handlers ────────────────────────────────────────────

    def handle_add_tool(self, tool_type, index=None):
        was_looping = self.btn_loop.isChecked()
        if was_looping:
            self.controller.run_loop_stop()

        reg = get_registration(tool_type)
        if reg is None:
            return

        count = sum(1 for t in self.workflow.tools if type(t).__name__ == tool_type)
        tool = reg.cls(f"{reg.default_name}_{count}")
        tool.prefix = reg.prefix

        if index is not None and 0 <= index <= len(self.workflow.tools):
            self.workflow.tools.insert(index, tool)
        else:
            self.workflow.add_tool(tool)

        tool.set_controller(self.controller)
            
        self.refresh_ui()
        
        if was_looping:
            self.controller.run_loop_start()

    def on_tool_pasted(self, tool):
        tool.set_controller(self.controller)
        self.refresh_ui()

    def setup_tool_connections(self, tool):
        tool.set_controller(self.controller)
        try:
            tool.structure_changed.disconnect(self.refresh_ui)
        except Exception:
            # Not connected yet; ignore.
            pass
        tool.structure_changed.connect(self.refresh_ui)

    def on_reorder(self):
        was_looping = self.btn_loop.isChecked()
        if was_looping: self.controller.run_loop_stop()
        
        new_tools = []
        for i in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(i)
            tool = self.tree.get_tool_from_item(item)
            if tool:
                new_tools.append(tool)
        self.workflow.tools = new_tools
        self.refresh_ui()
        
        if was_looping: self.controller.run_loop_start()

    def on_deletion(self, tool):
        was_looping = self.btn_loop.isChecked()
        if was_looping: self.controller.run_loop_stop()
        
        self.workflow.remove_tool(tool)
        self.refresh_ui()
        self.controller.mark_dirty()
        
        if was_looping: self.controller.run_loop_start()

    def refresh_ui(self):
        """Total rebuild of the UI structure."""
        self.tree.set_workflow(self.workflow)
        self.update_record_list()
        self.update_image_counter()

    def refresh_values(self):
        """Lightweight refresh of values and status."""
        self.tree.refresh_status()
        self.update_record_list()
        self.update_image_counter()

    def set_workflow(self, workflow):
        self._last_records = None
        self.controller = WorkflowController(workflow or Workflow())
        self.controller.set_scheduler(lambda fn: QTimer.singleShot(0, fn))
        self.controller.execution_finished.connect(self._on_execution_finished)
        self.workflow = self.controller.workflow
        for tool in self.workflow.tools:
            tool.set_controller(self.controller)
            try:
                tool.structure_changed.disconnect(self.refresh_ui)
            except Exception:
                # Not connected yet; ignore.
                pass
            tool.structure_changed.connect(self.refresh_ui)

        self.refresh_ui()

    def update_record_list(self):
        # Avoid rebuilding the combo while its popup is open, otherwise the
        # current hover/selection can jump around during run-loop updates.
        if self.record_selector.view().isVisible():
            return

        from falcocv.tools.block import Block
        from falcocv.core.vision_image import VisionImage

        def is_image(val, name=None):
            if val is None:
                if not name:
                    return False
                lower = name.lower()
                return any(keyword in lower for keyword in ["image", "mask", "frame", "img", "rgb", "bgr", "gray", "grey", "depth", "range"])

            # Extract raw array for checks if it's a VisionImage
            raw = val.data if isinstance(val, VisionImage) else val
            if not isinstance(raw, np.ndarray):
                return False
            if raw.ndim == 2:
                return raw.dtype.kind in "uif"
            if raw.ndim == 3 and raw.shape[2] in (1, 3, 4):
                return raw.dtype.kind in "uif"
            return False

        # Build the new record list.
        records = []

        # 1. Inputs of the current level (if we are inside a block)
        if self.workflow.parent_tool:
            for in_name, val in sorted(self.workflow.parent_tool.inputs.items()):
                if is_image(val, in_name):
                    records.append(f"Inputs.{in_name}")

        # 2. Tools at this level
        for tool in self.workflow.tools:
            if isinstance(tool, Block):
                # For blocks, check outputs
                for out_name, val in sorted(tool.outputs.items()):
                    if is_image(val, out_name):
                        records.append(f"{tool.name}.Outputs.{out_name}")
            else:
                # Normal tools
                for out_name, val in sorted(tool.outputs.items()):
                    if is_image(val, out_name):
                        records.append(f"{tool.name}.{out_name}")

        # Rebuild the combo only when the record list actually changes.
        if getattr(self, "_last_records", None) == records:
            return
        self._last_records = records.copy()

        self.record_selector.blockSignals(True)
        self.record_selector.setUpdatesEnabled(False)
        current_sel = self.record_selector.currentText()
        self.record_selector.clear()

        for r in records:
            self.record_selector.addItem(r)

        idx = self.record_selector.findText(current_sel)
        if idx >= 0:
            self.record_selector.setCurrentIndex(idx)
        elif self.record_selector.count() > 0:
            self.record_selector.setCurrentIndex(0)

        self.record_selector.setUpdatesEnabled(True)
        self.record_selector.blockSignals(False)
        self.update_viewer_image()

    def find_record_by_path(self, workflow, path):
        from falcocv.tools.block import Block
        parts = path.split(".", 1)
        if len(parts) < 2: return None, None
        
        name, rest = parts
        
        # 1. Check if it's "Inputs" or "Outputs" of the CURRENT level (editing a block)
        if name == "Inputs" and workflow.parent_tool:
            return workflow.parent_tool.inputs.get(rest), workflow.parent_tool
        if name == "Outputs" and workflow.parent_tool:
            return workflow.parent_tool.outputs.get(rest), workflow.parent_tool
            
        # 2. Check tools at this level
        target_tool = next((t for t in workflow.tools if t.name == name), None)
        if target_tool:
            # Handle block's exposed ports with "Inputs." or "Outputs." prefix
            if isinstance(target_tool, Block):
                if rest.startswith("Inputs."):
                    port_name = rest.split(".", 1)[1]
                    return target_tool.inputs.get(port_name), target_tool
                if rest.startswith("Outputs."):
                    port_name = rest.split(".", 1)[1]
                    return target_tool.outputs.get(port_name), target_tool
            
            # Normal tool record or fallback for blocks
            if rest in target_tool.inputs:
                return target_tool.inputs[rest], target_tool
            return target_tool.get_output(rest), target_tool
                
        return None, None

    def update_viewer_image(self):
        selection = self.record_selector.currentText()
        if not selection:
            self.viewer.set_image(None)
            return
            
        # Recursive resolution of the hierarchical path
        data, target_tool = self.find_record_by_path(self.workflow, selection)
        
        # Display data
        from falcocv.core.vision_image import VisionImage
        
        if isinstance(data, (np.ndarray, VisionImage)):
            # If it's a VisionImage, extract the raw data for display/analysis
            raw_data = data.data if isinstance(data, VisionImage) else data
            
            # Check if we have 3D data to display
            has_3d = (isinstance(data, VisionImage) and hasattr(data, '_3d_data') and data._3d_data is not None)
            
            self.chk_show3d.setVisible(has_3d and self.viewer_3d is not None)
            show_3d_view = has_3d and self.viewer_3d is not None and self.chk_show3d.isChecked()

            if show_3d_view:
                self.viewer = self.viewer_3d
                self.viewer_stack.setCurrentWidget(self.viewer_3d)
                d3 = data._3d_data
                pts = d3["points"]
                colors = d3.get("colors")
                intensity = d3.get("intensity")
                if colors is None and intensity is not None and len(intensity) == len(pts):
                    colors = map_depth_to_rgb(intensity, "TURBO", None, None)
                from falcocv.core.config import ConfigManager
                bbox_center = d3.get("bbox_center")
                bbox_size = d3.get("bbox_size")
                self.viewer_3d.set_data(pts, colors, bbox_center=bbox_center, bbox_size=bbox_size)
                self.viewer_3d.set_point_size(ConfigManager().get_3d_point_size())
                self.viewer_3d.set_overlays_from_image(data)
                if bbox_center is not None and bbox_size is not None:
                    self.viewer_3d.set_roi(bbox_center, bbox_size)
                    self.viewer_3d.show_roi(True)
                elif d3["points"].shape[0] > 0:
                    center, size = axis_aligned_bounding_box(d3["points"])
                    self.viewer_3d.set_roi(center, size)
                    self.viewer_3d.show_roi(True)
            else:
                self.viewer = self.viewer_2d
                self.viewer_stack.setCurrentWidget(self.viewer_2d)

                port_name = self._port_name_from_record(selection)
                show_depth = should_offer_depth_controls(raw_data, port_name)
                if show_depth:
                    cmap = "TURBO"
                    vmin = None
                    vmax = None
                    if target_tool and hasattr(target_tool, 'params') and target_tool.params:
                        cmap = str(target_tool.params.get("_display_cmap", "TURBO"))
                        vmin = target_tool.params.get("_display_vmin")
                        vmax = target_tool.params.get("_display_vmax")
                    depth_opts = {
                        "use_colormap": True,
                        "colormap": cmap,
                        "vmin": vmin,
                        "vmax": vmax,
                    }
                    self.viewer.set_image(data if isinstance(data, VisionImage) else raw_data, auto_fit=True, depth_display=depth_opts, render_overlays_flag=WorkflowEditor._global_show_overlays)
                else:
                    self.viewer.set_image(data if isinstance(data, VisionImage) else raw_data, auto_fit=True, render_overlays_flag=WorkflowEditor._global_show_overlays)
            
            self.viewer.show()
            # ROI Sync logic (2D only)
            if hasattr(target_tool, 'params') and "roi" in target_tool.params:
                roi = target_tool.params["roi"]
                if len(roi) == 4:
                    x, y, w, h = roi
                    a = 0
                else:
                    x, y, w, h, a = roi
                    
                show_roi = False
                if data is not None and w > 0 and h > 0:
                    h_img, w_img = data.shape[:2]
                    if x > 0 or y > 0 or w < w_img or h < h_img or a != 0:
                        show_roi = True
                
                if ".input_image" in selection and show_roi:
                    self.viewer.set_roi([x, y, w, h, a])
                else:
                    self.viewer.set_roi(None)
            else:
                self.viewer.set_roi(None)
        else:
            self.viewer.set_image(None)
            self.viewer.set_roi(None)

    def on_tool_selected(self, tool):
        pass # Decoupled as requested

    def open_tool_config(self, item):
        tool = self.tree.get_tool_from_item(item)
        if tool:
            self.controller.run(advance=False) # Populate inputs first
            parent_rect = self.rect()
            viewer_pos = self.viewer.mapTo(self, QPoint(0, 0))
            y_offset = viewer_pos.y()
            tl = self.mapToGlobal(QPoint(0, y_offset))
            geom = QRect(tl.x(), tl.y(), parent_rect.width(), parent_rect.height() - y_offset-25)
            from falcocv.ui.widgets.tool_config_dialog import ToolConfigDialog
            ToolConfigDialog(tool, parent=self, geometry=geom).show()

    def run_workflow(self, advance=False):
        self.controller.run(advance=advance)

    def _on_execution_finished(self, result):
        self.refresh_values()
        self.update_viewer_image()
        time_ms = result.get("total_time_ms", 0)
        self.label_total_time.setText(f"{time_ms:.1f} ms")
        error = result.get("error")
        if error:
            self.label_error.setText(error.split("\n")[0])
            self.label_error.setToolTip(error)
            print(f"Workflow Error: {error}")
            self.label_total_time.setStyleSheet(f"color: {WARNING}; font-weight: bold; font-size: 11px;")
        else:
            self.label_error.setText("")
            self.label_error.setToolTip("")
            self.label_total_time.setStyleSheet(f"color: {SUCCESS}; font-weight: bold; font-size: 11px;")
        self.workflow_changed.emit()

    def _show_error_context_menu(self, pos):
        full = self.label_error.toolTip()
        if not full:
            return
        menu = QMenu(self)
        menu.addAction("Copy error", lambda: QApplication.clipboard().setText(full))
        menu.exec(self.label_error.mapToGlobal(pos))

    def toggle_loop(self, checked):
        if checked:
            self.btn_loop.setText("PAUSE LOOP")
            self.controller.run_loop_start()
        else:
            self.btn_loop.setText("RUN LOOP")
            self.controller.run_loop_stop()

    def _find_folder_loader(self):
        """Find the first folder-mode loader tool (ImageLoader or DepthRangeLoader)."""
        for t in self.workflow.tools:
            if isinstance(t, (ImageLoader, DepthRangeLoader)) and t.params.get("mode") == "Folder":
                return t
        return None

    def on_index_edited(self):
        loader = self._find_folder_loader()
        if loader:
            try:
                new_idx = int(self.edit_index.text()) - 1
                total = len(loader._image_files)
                if 0 <= new_idx < total:
                    loader.params["current_index"] = new_idx
                    self.controller.run(advance=False)
                else:
                    self.update_image_counter()
            except ValueError:
                self.update_image_counter()

    def update_image_counter(self):
        loader = self._find_folder_loader()
        if loader:
            self.counter_widget.show()
            total = len(loader._image_files)
            current = loader.params["current_index"] + 1 if total > 0 else 0
            self.edit_index.setText(str(current))
            self.label_total_frames.setText(f"/{total}]")
        else:
            self.counter_widget.hide()

    def set_web_status(self, text: str, color_hex: str):
        self.label_web.setText(text)
        self.label_web.setStyleSheet(f"color: {color_hex}; font-family: Consolas, monospace; font-size: 10px;")

    def _port_name_from_record(self, record) -> str:
        return ""
