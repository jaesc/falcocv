"""Shared editor for a tool's input or output ports.

Both Block and Python Script use it: the name is editable and the kind is
chosen from a combo (Image / Plane / Any).  Renaming a port follows the
connection list so the wiring is not lost.
"""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from falcocv.core.port_types import (
    ANY,
    PORT_LABELS,
    add_port,
    normalize_kind,
    remove_port,
    rename_port,
    set_port_type,
    unique_port_name,
)


class PortTable(QWidget):
    """Edits the ``inputs`` or ``outputs`` of a tool with a kind per port."""

    ports_updated = pyqtSignal()

    def __init__(self, tool, direction="inputs", title=None, parent=None):
        super().__init__(parent)
        self.tool = tool
        self.direction = direction
        self._updating = False
        self._names: list[str] = []

        label = title or ("INPUTS" if direction == "inputs" else "OUTPUTS")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.title = QLabel()
        self.title.setTextFormat(Qt.TextFormat.RichText)
        self.title.setText(f"<b>{label}</b>")
        self._apply_title_color()
        layout.addWidget(self.title)

        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["Name", "Type"])
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.itemChanged.connect(self._on_name_changed)
        layout.addWidget(self.table)

        buttons = QHBoxLayout()
        self.btn_add = QPushButton("+ Add Port")
        self.btn_add.setAutoDefault(False)
        self.btn_add.clicked.connect(self.add_port)
        self.btn_remove = QPushButton("- Remove Port")
        self.btn_remove.setAutoDefault(False)
        self.btn_remove.clicked.connect(self.remove_selected)
        buttons.addWidget(self.btn_add)
        buttons.addWidget(self.btn_remove)
        layout.addLayout(buttons)

        self.refresh()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _apply_title_color(self):
        key = "tool_input" if self.direction == "inputs" else "tool_output"
        try:
            from falcocv.core.config import ConfigManager

            color = ConfigManager().get_color(key)
        except Exception:
            color = "#4da6ff"
        text = self.title.text()
        self.title.setText(f"<b style='color: {color};'>{text}</b>")

    def _port_kind(self, name):
        if self.direction == "inputs":
            return self.tool.get_input_port_type(name)
        return self.tool.get_output_port_type(name)

    def _workflow(self):
        controller = getattr(self.tool, "_controller", None)
        return getattr(controller, "workflow", None)

    def _emit_change(self):
        self.tool.data_changed.emit()
        self.tool.structure_changed.emit()
        self.ports_updated.emit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def refresh(self):
        self._updating = True
        try:
            self.table.setRowCount(0)
            target = (
                self.tool.inputs
                if self.direction == "inputs"
                else self.tool.outputs
            )
            self._names = list(target.keys())
            for name in self._names:
                row = self.table.rowCount()
                self.table.insertRow(row)
                self.table.setItem(row, 0, QTableWidgetItem(name))

                combo = QComboBox()
                for kind, label in PORT_LABELS.items():
                    combo.addItem(label, kind)
                index = combo.findData(normalize_kind(self._port_kind(name)))
                combo.setCurrentIndex(index if index >= 0 else 0)
                combo.currentIndexChanged.connect(
                    lambda _index, n=name, c=combo: self._on_type_changed(
                        n, c.currentData()
                    )
                )
                self.table.setCellWidget(row, 1, combo)
        finally:
            self._updating = False

    def add_port(self):
        name = unique_port_name(self.tool, self.direction)
        add_port(self.tool, self.direction, name, ANY)
        self.refresh()
        self._emit_change()

    def remove_selected(self):
        row = self.table.currentRow()
        if row < 0 or row >= len(self._names):
            return
        name = self._names[row]
        workflow = self._workflow()
        if workflow is not None:
            workflow.remove_port_connections(self.tool, self.direction, name)
        remove_port(self.tool, self.direction, name)
        self.refresh()
        self._emit_change()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _on_name_changed(self, item):
        if self._updating or item.column() != 0:
            return
        row = item.row()
        if row < 0 or row >= len(self._names):
            return
        old = self._names[row]
        final = rename_port(self.tool, self.direction, old, item.text())
        if final != old:
            workflow = self._workflow()
            if workflow is not None:
                workflow.rename_port(self.tool, self.direction, old, final)
            self.refresh()
            self._emit_change()

    def _on_type_changed(self, name, kind):
        if self._updating:
            return
        set_port_type(self.tool, self.direction, name, kind)
        self._emit_change()
