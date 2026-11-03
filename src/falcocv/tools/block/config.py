from falcocv.core.tool_base import BaseTool
from falcocv.core.workflow import Workflow
from PyQt6.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QTabWidget
from falcocv.ui.style import SUCCESS, WARNING, ALT_BG, BORDER
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.widgets.port_table import PortTable
from falcocv.ui.widgets.workflow_editor import WorkflowEditor

from .logic import Block
class BlockConfig(QWidget):
    def __init__(self, tool: Block):
        super().__init__()
        self.tool = tool

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        self.tabs = QTabWidget()
        self.main_layout.addWidget(self.tabs)
        
        from falcocv.ui.widgets.toolbox import Toolbox
        self._previous_active_editor = Toolbox._active_editor
        self.editor = WorkflowEditor(self.tool.workflow)
        self.tool.data_changed.connect(self.refresh_ui)
        self.tool.structure_changed.connect(self._on_structure_changed)
        self.tabs.addTab(self.editor, "Workflow")
        from falcocv.core.config import ConfigManager
        saved = ConfigManager().get_tool_splitter("Block")
        if saved:
            self.editor.splitter.setSizes(saved)
        else:
            min_w = self.editor.tree.minimumSizeHint().width()
            self.editor.splitter.setSizes([min_w, 9999])
        
        self.terminals = QWidget()
        self.content_layout = QHBoxLayout(self.terminals)

        self.input_table = PortTable(
            self.tool, "inputs", title="INPUTS (Block External Ports)"
        )
        self.content_layout.addWidget(self.input_table)

        self.output_table = PortTable(
            self.tool, "outputs", title="OUTPUTS (Block External Ports)"
        )
        self.content_layout.addWidget(self.output_table)

        self.tabs.addTab(self.terminals, "Terminals")

    @safe_refresh
    def refresh_ui(self):
        self.editor.tree.refresh_status(force=True)
        self.editor.update_viewer_image()

    def _on_structure_changed(self):
        if self.editor is not None:
            self.editor.refresh_ui()

    def on_config_closed(self):
        from falcocv.core.config import ConfigManager
        from falcocv.ui.widgets.toolbox import Toolbox
        wf = self.tool.workflow
        ConfigManager().set_tool_splitter("Block", self.editor.splitter.sizes())
        self.editor.update_image_counter()
        # The editor is hidden now; drop the callbacks it registered on the
        # tools so a later structure change cannot rebuild a hidden tree.
        self.editor.detach()
        Toolbox.set_active_editor(self._previous_active_editor)
        time_ms = wf.last_execution_time_ms
        if wf.last_error:
            self.editor.label_total_time.setText(f"{time_ms:.1f} ms  \u26a0")
            self.editor.label_total_time.setStyleSheet(f"color: {WARNING}; font-weight: bold; font-size: 11px;")
        else:
            self.editor.label_total_time.setText(f"{time_ms:.1f} ms")
            self.editor.label_total_time.setStyleSheet(f"color: {SUCCESS}; font-weight: bold; font-size: 11px;")
