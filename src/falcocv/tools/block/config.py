from falcocv.core.tool_base import BaseTool
from falcocv.core.workflow import Workflow
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QTabWidget, 
                             QPushButton, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QComboBox, QLabel)
from PyQt6.QtCore import pyqtSignal
from falcocv.ui.style import SUCCESS, WARNING, ALT_BG, BORDER
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.widgets.workflow_editor import WorkflowEditor

from .logic import Block
class PortTable(QWidget):
    ports_updated = pyqtSignal()

    def __init__(self, tool, port_type="inputs"):
        super().__init__()
        self.tool = tool
        self.port_type = port_type
        
        layout = QVBoxLayout(self)
        
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "Type"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.table)
        
        btn_layout = QHBoxLayout()
        btn_add = QPushButton("Add Port")
        btn_add.clicked.connect(self.add_port)
        btn_layout.addWidget(btn_add)
        
        btn_remove = QPushButton("Remove Port")
        btn_remove.clicked.connect(self.remove_port)
        btn_layout.addWidget(btn_remove)
        
        layout.addLayout(btn_layout)
        
        self.refresh_table()

    def refresh_table(self):
        self.table.setRowCount(0)
        target_dict = getattr(self.tool, self.port_type)
        metadata = self.tool.port_metadata[self.port_type]
        
        for name in sorted(target_dict.keys()):
            row = self.table.rowCount()
            self.table.insertRow(row)
            
            name_item = QTableWidgetItem(name)
            self.table.setItem(row, 0, name_item)
            
            type_combo = QComboBox()
            type_combo.addItems(["Image", "Number", "String", "Boolean", "Any"])
            type_combo.setCurrentText(metadata.get(name, "Any"))
            type_combo.currentTextChanged.connect(lambda text, n=name: self.update_port_type(n, text))
            self.table.setCellWidget(row, 1, type_combo)

    def add_port(self):
        name = f"Port_{self.table.rowCount()}"
        target_dict = getattr(self.tool, self.port_type)
        target_dict[name] = None
        self.tool.port_metadata[self.port_type][name] = "Any"
        self.refresh_table()
        self.ports_updated.emit()
        self.tool.data_changed.emit()
        self.tool.structure_changed.emit()

    def remove_port(self):
        row = self.table.currentRow()
        if row >= 0:
            name = self.table.item(row, 0).text()
            target_dict = getattr(self.tool, self.port_type)
            if name in target_dict:
                del target_dict[name]
                del self.tool.port_metadata[self.port_type][name]
            self.refresh_table()
            self.ports_updated.emit()
            self.tool.data_changed.emit()
            self.tool.structure_changed.emit()

    def update_port_type(self, name, port_type):
        self.tool.port_metadata[self.port_type][name] = port_type
        self.ports_updated.emit()


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
        
        from falcocv.core.config import ConfigManager
        config = ConfigManager()
        c_in = config.get_color("tool_input")
        c_out = config.get_color("tool_output")
        
        input_container = QWidget()
        input_vbox = QVBoxLayout(input_container)
        input_vbox.addWidget(QLabel(f"<b style='color: {c_in};'>INPUTS (Block External Ports)</b>"))
        self.input_table = PortTable(self.tool, "inputs")
        self.input_table.ports_updated.connect(self.editor.refresh_ui)
        input_vbox.addWidget(self.input_table)
        self.content_layout.addWidget(input_container)
        
        output_container = QWidget()
        output_vbox = QVBoxLayout(output_container)
        output_vbox.addWidget(QLabel(f"<b style='color: {c_out};'>OUTPUTS (Block External Ports)</b>"))
        self.output_table = PortTable(self.tool, "outputs")
        self.output_table.ports_updated.connect(self.editor.refresh_ui)
        output_vbox.addWidget(self.output_table)
        self.content_layout.addWidget(output_container)
        
        self.tabs.addTab(self.terminals, "Terminals")

    @safe_refresh
    def refresh_ui(self):
        wf = self.tool.workflow
        self.editor.tree.refresh_status()
        self.editor.update_viewer_image()

    def on_config_closed(self):
        from falcocv.core.config import ConfigManager
        from falcocv.ui.widgets.toolbox import Toolbox
        ConfigManager().set_tool_splitter("Block", self.editor.splitter.sizes())
        self.editor.update_image_counter()
        Toolbox.set_active_editor(self._previous_active_editor)
        time_ms = wf.last_execution_time_ms
        if wf.last_error:
            self.editor.label_total_time.setText(f"{time_ms:.1f} ms  \u26a0")
            self.editor.label_total_time.setStyleSheet(f"color: {WARNING}; font-weight: bold; font-size: 11px;")
        else:
            self.editor.label_total_time.setText(f"{time_ms:.1f} ms")
            self.editor.label_total_time.setStyleSheet(f"color: {SUCCESS}; font-weight: bold; font-size: 11px;")
