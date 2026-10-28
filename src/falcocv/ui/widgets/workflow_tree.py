from PyQt6.QtWidgets import QTreeWidget, QTreeWidgetItem, QAbstractItemView, QMenu, QStyledItemDelegate
from PyQt6.QtCore import Qt, pyqtSignal, QMimeData, QTimer
from PyQt6.QtGui import QFont, QColor, QAction, QCursor
import numpy as np
from falcocv.ui.style import ALT_BG, DANGER, DARK_BG, HIGHLIGHT, INFO, PANEL_BG, SUCCESS, TEXT, TEXT_MUTED, HANDLE_FILL, TEXT_DIM, MENU_BG, WHITE
from falcocv.core.config import ConfigManager
from falcocv.core.tool_base import ToolResults, ToolResult
from falcocv.core.tool_registry import get_registration_by_class


def _format_array_shape(arr):
    """Return a compact shape string for any numpy ndarray dimensionality."""
    if arr.ndim == 0:
        return "[scalar]"
    if arr.ndim == 1:
        return f"[{arr.shape[0]}]"
    return "[{}]".format("x".join(str(s) for s in arr.shape))


class NameColumnDelegate(QStyledItemDelegate):
    def createEditor(self, parent, option, index):
        if index.column() == 0:
            return super().createEditor(parent, option, index)
        return None


class WorkflowTree(QTreeWidget):
    tool_selected = pyqtSignal(object)
    order_changed = pyqtSignal()
    tool_deleted = pyqtSignal(object)
    tool_dropped = pyqtSignal(str, int) # tool_type, index
    tool_pasted = pyqtSignal(object)
    connection_changed = pyqtSignal(object, str, object, str) # src, out, dst, in

    _clipboard = None # Static clipboard for tool data

    def __init__(self, parent=None):
        super().__init__(parent)
        self.config = ConfigManager()
        self.setColumnCount(2)
        self.setHeaderLabels(["Tool / Port", "Source / Info"])
        self.setHeaderHidden(False)
        self.header().setStretchLastSection(True)
        self.header().setSectionsMovable(False)
        self.header().setStyleSheet(f"QHeaderView::section {{ background-color: {ALT_BG}; color: {TEXT_MUTED}; border: none; padding: 4px; font-weight: bold; }}")
        self.setItemDelegate(NameColumnDelegate(self))
        self.setIndentation(25)
        self.setAnimated(True)
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop) # Allow external drops
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setDragDropOverwriteMode(False)
        self.setDropIndicatorShown(True)
        self.setDefaultDropAction(Qt.DropAction.MoveAction)
        self.setExpandsOnDoubleClick(False)
        # Double-click opens the tool properties. Renaming is explicit (F2),
        # otherwise Qt also starts an inline editor and commits it when the
        # properties dialog takes focus, rebuilding the tree mid-signal.
        self.setEditTriggers(QAbstractItemView.EditTrigger.EditKeyPressed)
        
        self.setColumnWidth(0, 180)
        
        self.invisibleRootItem().setFlags(Qt.ItemFlag.ItemIsDropEnabled | Qt.ItemFlag.ItemIsEnabled)
        
        self.setStyleSheet(f"""
            QTreeWidget {{
                background-color: {DARK_BG};
                border: none;
                color: {TEXT};
                font-family: 'Segoe UI', sans-serif;
                font-size: 13px;
                outline: none;
            }}
            QTreeWidget::item {{
                padding: 8px;
                border-bottom: 1px solid {ALT_BG};
            }}
            QTreeWidget::item:selected {{
                background-color: {PANEL_BG};
                color: {TEXT};
                border-left: 3px solid {HIGHLIGHT};
            }}
            QTreeWidget::item:hover {{
                background-color: {ALT_BG};
            }}
        """)
        
        self.itemClicked.connect(self.handle_click)
        self.itemDoubleClicked.connect(self.handle_double_click)
        self.itemChanged.connect(self.handle_rename)
        self.itemExpanded.connect(self.handle_item_expanded)
        self.model().rowsMoved.connect(lambda: self.order_changed.emit())
        
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self.show_context_menu)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        
        self._id_to_tool = {}
        self._workflow = None
        self._last_refresh_time = 0

    def mimeTypes(self):
        types = super().mimeTypes()
        types.append("application/x-falcocv-tool-uuid")
        return types

    def mimeData(self, items):
        mime = super().mimeData(items)
        if items:
            tool_id = items[0].data(0, Qt.ItemDataRole.UserRole) or ""
            if tool_id and tool_id in self._id_to_tool:
                tool = self._id_to_tool[tool_id]
                mime.setData("application/x-falcocv-tool-uuid", tool.uuid.encode("utf-8"))
        return mime

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Delete:
            item = self.currentItem()
            if item:
                tool = self.get_tool_from_item(item)
                if tool and self._workflow and tool != self._workflow.parent_tool:
                    self.tool_deleted.emit(tool)
        super().keyPressEvent(event)

    def dragEnterEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragEnterEvent(event)

    def dragMoveEvent(self, event):
        if event.mimeData().hasText():
            event.acceptProposedAction()
        else:
            super().dragMoveEvent(event)

    def dropEvent(self, event):
        # Cross-workflow tree drag (tool move between different WorkflowTree instances)
        if (event.source() is not self and
            event.mimeData().hasFormat("application/x-falcocv-tool-uuid")):
            tool_uuid = bytes(event.mimeData().data("application/x-falcocv-tool-uuid")).decode("utf-8")
            source_tree = event.source()
            source_wf = source_tree._workflow

            tool = next((t for t in source_wf.tools if t.uuid == tool_uuid), None)
            if tool and self._workflow is not None:
                source_wf.remove_tool(tool)

                index = self.indexAt(event.position().toPoint()).row()
                if 0 <= index <= len(self._workflow.tools):
                    self._workflow.tools.insert(index, tool)
                else:
                    self._workflow.add_tool(tool)

                source_tree.set_workflow(source_wf)
                self.set_workflow(self._workflow)

                source_tree.order_changed.emit()
                self.order_changed.emit()

            event.acceptProposedAction()
            return

        if event.mimeData().hasText() and event.source() is not self:
            # Drop from Toolbox
            tool_type = event.mimeData().text()
            index = self.indexAt(event.position().toPoint()).row()
            self.tool_dropped.emit(tool_type, index)
            event.acceptProposedAction()
        else:
            # Internal move
            super().dropEvent(event)
            # Signal reorder after internal drop
            self.order_changed.emit()

    def set_workflow(self, workflow):
        expanded = {}
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            tid = item.data(0, Qt.ItemDataRole.UserRole)
            expanded[tid] = item.isExpanded()
        self._workflow = workflow
        self.blockSignals(True)
        self.clear()
        self._id_to_tool = {}
        
        # Load colors from config
        c_t_in = QColor(self.config.get_color("tool_input"))
        c_t_out = QColor(self.config.get_color("tool_output"))
        c_b_in = QColor(self.config.get_color("block_input"))
        c_b_out = QColor(self.config.get_color("block_output"))

        if workflow.parent_tool:
            block_item = QTreeWidgetItem(self)
            block_item.setText(0, "BLOCK INPUTS")
            block_item.setFont(0, QFont("Segoe UI", 10, QFont.Weight.Bold))
            block_item.setForeground(0, HANDLE_FILL)
            block_item.setFlags(Qt.ItemFlag.ItemIsEnabled) # Not draggable, not droppable
            
            for in_name in workflow.parent_tool.inputs:
                in_item = QTreeWidgetItem(block_item)
                in_item.setText(0, f"{in_name} \u2192")
                in_item.setForeground(0, c_b_in)
                in_item.setData(0, Qt.ItemDataRole.UserRole, "output")
                in_item.setData(1, Qt.ItemDataRole.UserRole, in_name)
                in_item.setData(0, Qt.ItemDataRole.UserRole + 1, workflow.parent_tool)
                in_item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            block_item.setExpanded(True)

        for tool in workflow.tools:
            tool_id = str(id(tool))
            self._id_to_tool[tool_id] = tool
            
            tool_item = QTreeWidgetItem(self)
            display = f"{tool.prefix} {tool.name}" if tool.prefix else tool.name
            tool_item.setText(0, display)
            tool_item.setFont(0, QFont("Consolas", 9, QFont.Weight.Bold))
            
            tool_item.setData(0, Qt.ItemDataRole.UserRole, tool_id)
            # TOOLS are draggable but NOT droppable (we only drop on the root)
            tool_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled | 
                                Qt.ItemFlag.ItemIsEditable | Qt.ItemFlag.ItemIsDragEnabled)
            
            if not getattr(tool, 'enabled', True):
                tool_item.setForeground(0, QColor(TEXT_MUTED))
                tool_item.setText(1, "Disabled")
            
            # Direct child items for Inputs
            for in_name in tool.inputs:
                in_item = QTreeWidgetItem(tool_item)
                in_item.setText(0, f"\u2192 {in_name}")
                in_item.setForeground(0, c_t_in)
                in_item.setData(0, Qt.ItemDataRole.UserRole, "input")
                in_item.setData(1, Qt.ItemDataRole.UserRole, in_name)
                
                source_text = "[ Click to Link ]"
                source_color = QColor(TEXT_DIM)
                for src_t, out_t, dst_t, in_t in workflow.connections:
                    if dst_t == tool and in_t == in_name:
                        src_name = src_t.name if src_t != workflow.parent_tool else "System Inputs"
                        source_text = f"\u21aa {src_name}.{out_t}"
                        break
                
                in_item.setText(1, source_text)
                in_item.setForeground(1, source_color)
                in_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            
            # Direct child items for Outputs
            for out_name in tool.outputs:
                out_item = QTreeWidgetItem(tool_item)
                out_item.setText(0, f"{out_name} \u2192")
                out_item.setForeground(0, c_t_out)
                out_item.setData(0, Qt.ItemDataRole.UserRole, "output")
                out_item.setData(0, Qt.ItemDataRole.UserRole + 1, tool)
                out_item.setData(1, Qt.ItemDataRole.UserRole, out_name)
                
                # Show current value snippet if small
                val = tool.get_output(out_name)
                if isinstance(val, (int, float, str)):
                    out_item.setText(1, str(val))
                elif isinstance(val, np.ndarray):
                    out_item.setText(1, _format_array_shape(val))
                elif isinstance(val, ToolResults):
                    out_item.setText(1, f"({val.Count} results)")
                    out_item.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
                
                out_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            
            tool_item.setExpanded(True)

        if workflow.parent_tool:
            out_block_item = QTreeWidgetItem(self)
            out_block_item.setText(0, "BLOCK OUTPUTS")
            out_block_item.setFont(0, QFont("Segoe UI", 10, QFont.Weight.Bold))
            out_block_item.setForeground(0, HANDLE_FILL)
            out_block_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
            
            for out_name in workflow.parent_tool.outputs:
                out_item = QTreeWidgetItem(out_block_item)
                out_item.setText(0, f"\u2192 {out_name}")
                out_item.setForeground(0, c_b_out) # Use Block Output color
                out_item.setData(0, Qt.ItemDataRole.UserRole, "input")
                out_item.setData(1, Qt.ItemDataRole.UserRole, out_name)
                out_item.setData(0, Qt.ItemDataRole.UserRole + 1, workflow.parent_tool)
                
                # Show source
                source_text = "[ Select Source ]"
                for src_t, src_o, dst_t, dst_i in workflow.connections:
                    if dst_t == workflow.parent_tool and dst_i == out_name:
                        source_text = f"Linked: {src_t.name}.{src_o}"
                        break
                out_item.setText(1, source_text)
                out_item.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled)
            out_block_item.setExpanded(True)
            
        self.blockSignals(False)
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            tid = item.data(0, Qt.ItemDataRole.UserRole)
            if tid in expanded:
                item.setExpanded(expanded[tid])
        self.refresh_status()

    def refresh_status(self):
        """Update the status column for each tool item with execution time and error state."""
        import time
        now = time.perf_counter()
        if now - self._last_refresh_time < 0.1: # Limit to 10Hz
            return
        self._last_refresh_time = now
        
        self.blockSignals(True)
        try:
            COLOR_OK = QColor(SUCCESS)
            COLOR_ERR = QColor(DANGER)
            COLOR_VAL = QColor(TEXT_MUTED)
            COLOR_NONE = QColor(TEXT_DIM)
            
            def set_port_info(child, val):
                # Special handling for ToolResults (collections)
                if isinstance(val, ToolResults):
                    summary = f"({val.Count} results)"
                    child.setChildIndicatorPolicy(QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator)
                    current_text = child.text(1)
                    if "\u21aa" in current_text: # Link symbol
                        link_part = current_text.split("=")[0].strip()
                        child.setText(1, f"{link_part} = {summary}")
                    elif "Linked:" in current_text:
                        link_part = current_text.split("=")[0].strip()
                        child.setText(1, f"{link_part} = {summary}")
                    else:
                        child.setText(1, summary)
                    child.setForeground(1, COLOR_VAL)
                    
                    # Lazy Loading: Only update children if expanded
                    if not child.isExpanded():
                        return
                        
                    limit = 20
                    display_count = min(val.Count, limit)
                    existing_count = child.childCount()
                    expected_count = display_count + (1 if val.Count > limit else 0)

                    # Rebuild structure only if count changed
                    if existing_count != expected_count:
                        child.takeChildren()
                        for idx in range(display_count):
                            res = val[idx]
                            res_item = QTreeWidgetItem(child)
                            res_item.setText(0, f"Res [{idx}]")
                            res_item.setForeground(0, QColor(TEXT_MUTED))
                            res_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                            
                            if hasattr(res, 'area'):
                                res_item.setText(1, f"A:{int(res.area)}")
                            elif hasattr(res, 'fill_factor'):
                                res_item.setText(1, f"F:{res.fill_factor:.1f}%")
                            elif hasattr(res, 'format') and hasattr(res, 'text'):
                                txt = str(res.text)[:80]
                                res_item.setText(1, f"{res.format}: {txt}")

                            max_props = 20
                            props_shown = 0
                            for k, v in res.items():
                                if props_shown >= max_props: break
                                prop_item = QTreeWidgetItem(res_item)
                                prop_item.setText(0, k)
                                prop_item.setText(1, str(v) if not isinstance(v, float) else f"{v:.3f}")
                                prop_item.setForeground(0, QColor(TEXT_MUTED))
                                prop_item.setForeground(1, QColor(TEXT_MUTED))
                                prop_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                                props_shown += 1
                        
                        if val.Count > limit:
                            more_item = QTreeWidgetItem(child)
                            more_item.setText(0, f"... and {val.Count - limit} more")
                            more_item.setForeground(0, QColor(TEXT_DIM))
                            more_item.setFlags(Qt.ItemFlag.ItemIsEnabled)
                    else:
                        # Count unchanged: update values in-place so x, y, angle refresh each frame
                        for idx in range(display_count):
                            res_item = child.child(idx)
                            if res_item is None: continue
                            res = val[idx]
                            
                            if hasattr(res, 'area'):
                                res_item.setText(1, f"A:{int(res.area)}")
                            elif hasattr(res, 'fill_factor'):
                                res_item.setText(1, f"F:{res.fill_factor:.1f}%")
                            elif hasattr(res, 'format') and hasattr(res, 'text'):
                                txt = str(res.text)[:80]
                                res_item.setText(1, f"{res.format}: {txt}")

                            # Update each property value in-place
                            prop_list = list(res.items())
                            for pi in range(min(res_item.childCount(), len(prop_list))):
                                k, v = prop_list[pi]
                                prop_item = res_item.child(pi)
                                if prop_item:
                                    prop_item.setText(1, str(v) if not isinstance(v, float) else f"{v:.3f}")


                elif isinstance(val, np.ndarray):
                    label = _format_array_shape(val)
                    current_text = child.text(1)
                    if "\u21aa" in current_text:
                        link_part = current_text.split("=")[0].strip()
                        child.setText(1, f"{link_part} = {label}")
                    elif "Linked:" in current_text:
                        link_part = current_text.split("=")[0].strip()
                        child.setText(1, f"{link_part} = {label}")
                    else:
                        child.setText(1, label)
                    child.setForeground(1, COLOR_VAL)

                elif hasattr(val, 'data') and isinstance(val.data, np.ndarray):
                    label = _format_array_shape(val.data)
                    current_text = child.text(1)
                    if "\u21aa" in current_text:
                        link_part = current_text.split("=")[0].strip()
                        child.setText(1, f"{link_part} = {label}")
                    elif "Linked:" in current_text:
                        link_part = current_text.split("=")[0].strip()
                        child.setText(1, f"{link_part} = {label}")
                    else:
                        child.setText(1, label)
                    child.setForeground(1, COLOR_VAL)

                elif isinstance(val, (int, float, str, bool)):
                    if isinstance(val, float):
                        val_str = f"{val:.3f}"
                    elif isinstance(val, bool):
                        val_str = "True" if val else "False"
                    else:
                        val_str = str(val)
                        
                    current_text = child.text(1)
                    if "\u21aa" in current_text:
                        link_part = current_text.split("=")[0].strip()
                        child.setText(1, f"{link_part} = {val_str}")
                    elif "Linked:" in current_text:
                        link_part = current_text.split("=")[0].strip()
                        child.setText(1, f"{link_part} = {val_str}")
                    else:
                        child.setText(1, val_str)
                    child.setForeground(1, COLOR_VAL)
                elif val is None:
                    current_text = child.text(1)
                    if "\u21aa" in current_text:
                        link_part = current_text.split("=")[0].strip()
                        child.setText(1, link_part)
                    elif "Linked:" in current_text:
                        link_part = current_text.split("=")[0].strip()
                        child.setText(1, link_part)
                    else:
                        child.setText(1, "")
                    child.setForeground(1, COLOR_VAL)

            for i in range(self.topLevelItemCount()):
                item = self.topLevelItem(i)
                if not item: continue
                
                try:
                    _ = item.text(0)
                except RuntimeError: continue
                
                if item.text(0) == "BLOCK INPUTS":
                    if self._workflow and self._workflow.parent_tool:
                        for j in range(item.childCount()):
                            child = item.child(j)
                            name = child.data(1, Qt.ItemDataRole.UserRole)
                            set_port_info(child, self._workflow.parent_tool.inputs.get(name))
                    continue
                elif item.text(0) == "BLOCK OUTPUTS":
                    if self._workflow and self._workflow.parent_tool:
                        for j in range(item.childCount()):
                            child = item.child(j)
                            name = child.data(1, Qt.ItemDataRole.UserRole)
                            set_port_info(child, self._workflow.parent_tool.outputs.get(name))
                    continue

                tool = self.get_tool_from_item(item)
                if not tool: continue
                
                # 1. Update Tool Level Info
                if not getattr(tool, 'enabled', True):
                    item.setText(1, "Disabled")
                    item.setForeground(0, QColor(TEXT_MUTED))
                    item.setForeground(1, QColor(TEXT_MUTED))
                    for j in range(item.childCount()):
                        child = item.child(j)
                        if not child: continue
                        child.setForeground(0, QColor(TEXT_MUTED))
                        child.setForeground(1, QColor(TEXT_MUTED))
                    continue

                if tool.last_execution_time_ms > 0 or tool.last_error:
                    if tool.last_error:
                        item.setText(1, f"{tool.last_execution_time_ms:.1f} ms  \u26a0")
                        item.setForeground(1, COLOR_ERR)
                        item.setForeground(0, COLOR_ERR)
                    else:
                        item.setText(1, f"{tool.last_execution_time_ms:.1f} ms")
                        item.setForeground(1, COLOR_OK)
                        item.setForeground(0, QColor(TEXT))
                else:
                    item.setText(1, "")
                    item.setForeground(1, COLOR_NONE)
                    item.setForeground(0, QColor(TEXT))
                    
                # 2. Restore children foreground colors when re-enabled
                c_t_in = QColor(self.config.get_color("tool_input"))
                c_t_out = QColor(self.config.get_color("tool_output"))
                try:
                    for j in range(item.childCount()):
                        child = item.child(j)
                        if not child: continue
                        port_type = child.data(0, Qt.ItemDataRole.UserRole)
                        port_name = child.data(1, Qt.ItemDataRole.UserRole)
                        if port_type == "input":
                            child.setForeground(0, c_t_in)
                            set_port_info(child, tool.inputs.get(port_name))
                        elif port_type == "output":
                            child.setForeground(0, c_t_out)
                            set_port_info(child, tool.outputs.get(port_name))
                except RuntimeError: continue
        finally:
            self.blockSignals(False)

    def handle_item_expanded(self, item):
        # When the user expands a variable/result collection, refresh its children
        # so the tree shows the latest exposed values immediately.
        if not item:
            return
        self.refresh_status()

    def _unique_name(self, base_name: str, skip=None) -> str:
        existing = {t.name for t in self._workflow.tools if t is not skip}
        if base_name not in existing:
            return base_name
        i = 2
        while f"{base_name}_{i}" in existing:
            i += 1
        return f"{base_name}_{i}"

    def _restore_selection(self, tool_id):
        for i in range(self.topLevelItemCount()):
            item = self.topLevelItem(i)
            if item.data(0, Qt.ItemDataRole.UserRole) == tool_id:
                self.setCurrentItem(item)
                self.scrollToItem(item)
                break

    def handle_rename(self, item, column):
        if column != 0: return
        tool = self.get_tool_from_item(item)
        if not tool: return

        if not tool.prefix:
            reg = get_registration_by_class(type(tool))
            if reg:
                tool.prefix = reg.prefix

        prefix_full = f"{tool.prefix} " if tool.prefix else ""
        new_text = item.text(0)
        if prefix_full and new_text.startswith(prefix_full):
            user_part = new_text[len(prefix_full):].strip()
        else:
            user_part = new_text.strip()
        if not user_part:
            user_part = tool.name

        if tool.name != user_part:
            tool.name = self._unique_name(user_part, skip=tool)
            display = f"{tool.prefix} {tool.name}" if tool.prefix else tool.name
            # Rename in place. Rebuilding the whole tree here (set_workflow)
            # clears the item that Qt is currently editing and can abort the
            # process with a C++ use-after-free.
            self.blockSignals(True)
            try:
                item.setText(0, display)
            finally:
                self.blockSignals(False)
            self.order_changed.emit()

    def handle_double_click(self, item, column):
        tool = self.get_tool_from_item(item)
        if not tool or not hasattr(tool, "get_config_widget"):
            return
        from falcocv.ui.widgets.tool_config_dialog import open_tool_config
        try:
            open_tool_config(tool, parent=self.window())
        except Exception:
            # Never let a configuration error escape into Qt's event loop.
            import traceback
            traceback.print_exc()

    def handle_click(self, item, column):
        if column == 1 and item.data(0, Qt.ItemDataRole.UserRole) == "input":
            self.show_connection_menu(item)
            return

        tool = self.get_tool_from_item(item)
        if tool and tool != self._workflow.parent_tool:
            self.tool_selected.emit(tool)

    def show_connection_menu(self, item):
        dst_tool = item.data(0, Qt.ItemDataRole.UserRole + 1)
        if not dst_tool:
            dst_tool = self.get_tool_from_item(item.parent())
            
        in_name = item.data(1, Qt.ItemDataRole.UserRole)
        
        menu = QMenu(self)
        menu.setStyleSheet(f"background-color: {MENU_BG}; color: {WHITE};")
        
        none_action = menu.addAction("Disconnect")
        none_action.triggered.connect(lambda: self.apply_connection(None, None, dst_tool, in_name))
        menu.addSeparator()
        
        is_block_output = (dst_tool == self._workflow.parent_tool)
        
        if not is_block_output and self._workflow.parent_tool:
            for b_in_name in self._workflow.parent_tool.inputs:
                action = menu.addAction(f"Inputs.{b_in_name}")
                action.triggered.connect(lambda checked, s=self._workflow.parent_tool, o=b_in_name: 
                                         self.apply_connection(s, o, dst_tool, in_name))
            menu.addSeparator()

        for tool in self._workflow.tools:
            if tool == dst_tool: continue
            
            tool_menu = menu.addMenu(tool.name)
            tool_menu.setStyleSheet(f"background-color: {MENU_BG}; color: {WHITE};")
            for out_name in tool.outputs:
                action = tool_menu.addAction(out_name)
                action.triggered.connect(lambda checked, s=tool, o=out_name: 
                                         self.apply_connection(s, o, dst_tool, in_name))

        menu.exec(QCursor.pos())

    def apply_connection(self, src_tool, out_name, dst_tool, in_name):
        if src_tool is None:
            # Disconnect: just remove existing connection, don't add a new one
            self._workflow.remove_connection(dst_tool, in_name)
        else:
            self._workflow.connect(src_tool, out_name, dst_tool, in_name)
        self.set_workflow(self._workflow)
        self.connection_changed.emit(src_tool, out_name, dst_tool, in_name)

    def show_context_menu(self, pos):
        item = self.itemAt(pos)
        if not item: return
        
        tool = self.get_tool_from_item(item)
        if not tool or tool == self._workflow.parent_tool: return
        
        menu = QMenu(self)
        menu.setStyleSheet(f"background-color: {MENU_BG}; color: {WHITE};")
        
        enabled_action = QAction("Enabled", menu)
        enabled_action.setCheckable(True)
        enabled_action.setChecked(getattr(tool, 'enabled', True))
        enabled_action.triggered.connect(lambda checked, t=tool: self.set_tool_enabled(t, checked))
        menu.addAction(enabled_action)
        menu.addSeparator()

        copy_action = menu.addAction("Copy")
        copy_action.triggered.connect(lambda: self.copy_tool(tool))
        
        paste_action = menu.addAction("Paste")
        paste_action.setEnabled(WorkflowTree._clipboard is not None)
        paste_action.triggered.connect(lambda: self.paste_tool(self.indexAt(pos).row()))
        
        menu.addSeparator()
        
        save_action = menu.addAction(f"Save Tool As...")
        save_action.triggered.connect(lambda: self._save_tool_to_file(tool))

        menu.addSeparator()

        del_action = menu.addAction(f"Delete {tool.name}")
        del_action.triggered.connect(lambda: self.tool_deleted.emit(tool))

        menu.addSeparator()

        help_action = menu.addAction(f"Help: {tool.name}")
        help_action.triggered.connect(
            lambda: self._show_tool_help(tool)
        )

        menu.exec(self.viewport().mapToGlobal(pos))

    def _save_tool_to_file(self, tool):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        data = tool.serialize()
        data.pop("uuid", None)
        default_name = f"{tool.name}.cvtool"
        filepath, _ = QFileDialog.getSaveFileName(self, "Save Tool As", default_name, "FalcoCV Tool (*.cvtool)")
        if filepath:
            if not filepath.endswith(".cvtool"):
                filepath += ".cvtool"
            try:
                with open(filepath, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save tool: {e}")

    def _show_tool_help(self, tool):
        from falcocv.ui.widgets.help_dialog import ToolHelpDialog
        ToolHelpDialog.show_for_tool(tool, parent=self)

    def copy_tool(self, tool):
        WorkflowTree._clipboard = tool.serialize()
        # Remove UUID and Name from clipboard to ensure a fresh clone
        if "uuid" in WorkflowTree._clipboard:
            del WorkflowTree._clipboard["uuid"]
            
    def paste_tool(self, index=-1):
        if not WorkflowTree._clipboard or not self._workflow:
            return
            
        # Create a deep copy of the clipboard data
        import json
        data = json.loads(json.dumps(WorkflowTree._clipboard))
        
        original_name = data.get("name", "Tool")
        data["name"] = self._unique_name(original_name)
        
        # Create new tool instance
        new_tool = self._workflow.create_tool_from_data(data)
        if new_tool:
            # We don't add it here directly, we emit a signal so the editor handles it
            # actually, let's add it and signal the editor to refresh
            import uuid
            new_tool.uuid = str(uuid.uuid4())
            
            if index >= 0:
                self._workflow.tools.insert(index, new_tool)
            else:
                self._workflow.add_tool(new_tool)
            
            self.set_workflow(self._workflow)
            self.tool_pasted.emit(new_tool)
            self.order_changed.emit()

    def get_tool_from_item(self, item):
        if not item: return None
        tool_id = item.data(0, Qt.ItemDataRole.UserRole)
        return self._id_to_tool.get(tool_id)

    def set_tool_enabled(self, tool, enabled: bool):
        tool.enabled = enabled
        self.refresh_status()
        tool.mark_dirty()
