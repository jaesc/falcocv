"""Editor for Modbus TCP and OPC UA server runtimes."""

from __future__ import annotations

import copy

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from falcocv.comms.servers import (
    ServerConfigError,
    default_servers_config,
    parse_servers_config,
)
from falcocv.core.hmi import HMIProject

_SERVER_TYPES = (
    ("Modbus TCP Server", "modbus_tcp_server"),
    ("OPC UA Server", "opcua_server"),
)
_ACCESS = ("r", "rw")
_MAPPING_COLUMN_WIDTHS = (240, 260, 80)


def _server_type_label(server_type: str) -> str:
    for label, value in _SERVER_TYPES:
        if value == server_type:
            return label
    return server_type


class HMIServersWidget(QWidget):
    modified = pyqtSignal()

    def __init__(self, project: HMIProject):
        super().__init__()
        self.project = project
        self._draft_config = copy.deepcopy(
            getattr(project, "servers_config", None)
            or default_servers_config()
        )
        self._loading = False
        self._current_row = -1
        self._field_widgets: dict[str, QWidget] = {}
        self._build()
        self._load_servers()

    # ── construction ──────────────────────────────────────────────────────

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        helper = QLabel(
            "Servers expose existing Communications tags to external Modbus "
            "TCP or OPC UA clients. Mappings are explicit and read-only by "
            "default."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: #888;")
        layout.addWidget(helper)

        horizontal = QSplitter(Qt.Orientation.Horizontal)
        horizontal.setObjectName("hmi_servers.horizontal")
        self.horizontal_splitter = horizontal
        left = QGroupBox("Servers")
        left_layout = QVBoxLayout(left)
        self.server_list = QListWidget()
        self.server_list.currentRowChanged.connect(self._on_server_selected)
        left_layout.addWidget(self.server_list, 1)
        buttons = QHBoxLayout()
        add = QPushButton("+ Server")
        remove = QPushButton("- Server")
        add.clicked.connect(self._add_server)
        remove.clicked.connect(self._remove_server)
        buttons.addWidget(add)
        buttons.addWidget(remove)
        left_layout.addLayout(buttons)
        horizontal.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.server_group = QGroupBox("Server")
        self.form = QFormLayout(self.server_group)
        self.server_id = QLineEdit()
        self.server_id.setPlaceholderText("server id, e.g. modbus")
        self.form.addRow("ID:", self.server_id)
        self.server_type = QComboBox()
        for label, value in _SERVER_TYPES:
            self.server_type.addItem(label, value)
        self.server_type.currentIndexChanged.connect(self._on_type_changed)
        self.form.addRow("Type:", self.server_type)
        self.server_enabled = QCheckBox("Enabled")
        self.form.addRow("", self.server_enabled)
        self.fields_container = QWidget()
        self.fields = QFormLayout(self.fields_container)
        self.fields.setContentsMargins(0, 0, 0, 0)
        self.form.addRow("Configuration:", self.fields_container)

        self.server_scroll = QScrollArea()
        self.server_scroll.setObjectName("hmi_servers.device_scroll")
        self.server_scroll.setWidgetResizable(True)
        self.server_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.server_scroll.setWidget(self.server_group)

        mappings_group = QGroupBox("Mappings")
        mappings_layout = QVBoxLayout(mappings_group)
        self.mappings_table = QTableWidget(0, 3)
        self.mappings_table.setHorizontalHeaderLabels(
            ("Address / NodeId", "Tag", "Access")
        )
        mapping_header = self.mappings_table.horizontalHeader()
        mapping_header.setObjectName("hmi_servers.mappings")
        mapping_header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        mapping_header.setStretchLastSection(False)
        for column, width in enumerate(_MAPPING_COLUMN_WIDTHS):
            self.mappings_table.setColumnWidth(column, width)
        self.mappings_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        mappings_layout.addWidget(self.mappings_table, 1)
        mapping_buttons = QHBoxLayout()
        add_mapping = QPushButton("+ Mapping")
        remove_mapping = QPushButton("- Mapping")
        add_mapping.clicked.connect(self._add_mapping)
        remove_mapping.clicked.connect(self._remove_mapping)
        mapping_buttons.addWidget(add_mapping)
        mapping_buttons.addWidget(remove_mapping)
        mapping_buttons.addStretch()
        mappings_layout.addLayout(mapping_buttons)

        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical)
        self.vertical_splitter.setObjectName("hmi_servers.vertical")
        self.vertical_splitter.addWidget(self.server_scroll)
        self.vertical_splitter.addWidget(mappings_group)
        self.vertical_splitter.setStretchFactor(0, 0)
        self.vertical_splitter.setStretchFactor(1, 1)
        self.vertical_splitter.setSizes([330, 470])
        right_layout.addWidget(self.vertical_splitter, 1)

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #888;")
        right_layout.addWidget(self.status)
        horizontal.addWidget(right)
        horizontal.setSizes([220, 700])
        layout.addWidget(horizontal, 1)

        self.server_id.textChanged.connect(self._refresh_current_label)
        self.server_id.textChanged.connect(self._mark_dirty)
        self.server_enabled.toggled.connect(self._mark_dirty)
        self.mappings_table.cellChanged.connect(self._mark_dirty)

    def reset_view_state(self):
        self.horizontal_splitter.setSizes([220, 700])
        self.vertical_splitter.setSizes([330, 470])
        for column, width in enumerate(_MAPPING_COLUMN_WIDTHS):
            self.mappings_table.setColumnWidth(column, width)

    def _clear_fields(self):
        while self.fields.count():
            item = self.fields.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._field_widgets = {}

    def _build_fields(self, server_type: str, config: dict):
        self._clear_fields()
        if server_type == "modbus_tcp_server":
            values = (
                ("host", "Host", QLineEdit, config.get("host", "0.0.0.0")),
                ("port", "Port", QSpinBox, config.get("port", 502)),
                ("unit_id", "Unit ID", QSpinBox, config.get("unit_id", 1)),
            )
            for key, label, cls, value in values:
                widget = cls()
                if isinstance(widget, QSpinBox):
                    widget.setRange(0 if key == "unit_id" else 1, 65535)
                    widget.setValue(int(value))
                else:
                    widget.setText(str(value))
                widget.valueChanged.connect(self._mark_dirty) if isinstance(
                    widget, QSpinBox
                ) else widget.textChanged.connect(self._mark_dirty)
                self._field_widgets[key] = widget
                self.fields.addRow(label + ":", widget)
            order = QComboBox()
            order.addItem("Big endian words", "big")
            order.addItem("Little endian words", "little")
            idx = order.findData(config.get("word_order", "big"))
            if idx >= 0:
                order.setCurrentIndex(idx)
            order.currentIndexChanged.connect(self._mark_dirty)
            self._field_widgets["word_order"] = order
            self.fields.addRow("32-bit word order:", order)
        else:
            endpoint = QLineEdit(
                str(config.get("endpoint", "opc.tcp://0.0.0.0:4840/falcocv"))
            )
            namespace = QLineEdit(str(config.get("namespace", "urn:falcocv")))
            endpoint.textChanged.connect(self._mark_dirty)
            namespace.textChanged.connect(self._mark_dirty)
            self._field_widgets["endpoint"] = endpoint
            self._field_widgets["namespace"] = namespace
            self.fields.addRow("Endpoint:", endpoint)
            self.fields.addRow("Namespace:", namespace)

    def _load_servers(self, select: int = 0):
        self._loading = True
        try:
            self.server_list.clear()
            for server in self._servers():
                self.server_list.addItem(
                    QListWidgetItem(
                        f"{server.get('id', '?')}  "
                        f"[{_server_type_label(server.get('type', ''))}]"
                    )
                )
            if self.server_list.count():
                row = max(0, min(select, self.server_list.count() - 1))
                self.server_list.setCurrentRow(row)
                self._load_server(row)
            else:
                self._clear_server_form()
        finally:
            self._loading = False

    def _clear_server_form(self):
        self._current_row = -1
        self.server_id.clear()
        self.server_type.setCurrentIndex(0)
        self.server_enabled.setChecked(True)
        self._build_fields("modbus_tcp_server", {})
        self.mappings_table.setRowCount(0)

    def _servers(self) -> list[dict]:
        servers = self._draft_config.setdefault("servers", [])
        if not isinstance(servers, list):
            servers = []
            self._draft_config["servers"] = servers
        return servers

    def _load_server(self, row: int):
        servers = self._servers()
        if row < 0 or row >= len(servers):
            self._clear_server_form()
            return
        server = servers[row]
        self._current_row = row
        self._loading = True
        try:
            self.server_id.setText(str(server.get("id", "")))
            server_type = str(server.get("type", "modbus_tcp_server"))
            idx = self.server_type.findData(server_type)
            if idx >= 0:
                self.server_type.setCurrentIndex(idx)
            self.server_enabled.setChecked(bool(server.get("enabled", True)))
            self._build_fields(server_type, server)
            self._load_mappings(server)
        finally:
            self._loading = False

    def _load_mappings(self, server: dict):
        self._loading = True
        try:
            mappings = server.get("mappings", [])
            self.mappings_table.setRowCount(0)
            for mapping in mappings:
                row = self.mappings_table.rowCount()
                self.mappings_table.insertRow(row)
                self.mappings_table.setItem(
                    row,
                    0,
                    QTableWidgetItem(
                        str(
                            mapping.get(
                                "address", mapping.get("node_id", "")
                            )
                        )
                    ),
                )
                self.mappings_table.setItem(
                    row, 1, QTableWidgetItem(str(mapping.get("tag", "")))
                )
                access = QComboBox()
                access.addItems(_ACCESS)
                idx = access.findText(str(mapping.get("access", "r")))
                if idx >= 0:
                    access.setCurrentIndex(idx)
                access.currentIndexChanged.connect(self._mark_dirty)
                self.mappings_table.setCellWidget(row, 2, access)
        finally:
            self._loading = False

    def _refresh_current_label(self, _text: str = ""):
        row = self._current_row
        if 0 <= row < self.server_list.count():
            item = self.server_list.item(row)
            if item is not None:
                item.setText(
                    f"{self.server_id.text().strip() or '?'}  "
                    f"[{_server_type_label(self.server_type.currentData())}]"
                )

    def _on_server_selected(self, row: int):
        if self._loading:
            return
        if row != self._current_row and not self._capture_current():
            self.server_list.blockSignals(True)
            self.server_list.setCurrentRow(self._current_row)
            self.server_list.blockSignals(False)
            return
        self._load_server(row)

    def _on_type_changed(self, _index: int):
        if self._loading:
            return
        server_type = self.server_type.currentData()
        old_type = ""
        if 0 <= self._current_row < len(self._servers()):
            old_type = str(
                self._servers()[self._current_row].get("type", "")
            )
        if old_type and old_type != server_type:
            answer = QMessageBox.question(
                self,
                "Change server type",
                "Changing the server type clears its mappings. Continue?",
                QMessageBox.StandardButton.Yes
                | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                idx = self.server_type.findData(old_type)
                self.server_type.blockSignals(True)
                self.server_type.setCurrentIndex(idx)
                self.server_type.blockSignals(False)
                return
            self.mappings_table.setRowCount(0)
        self._build_fields(server_type, {})
        self._refresh_current_label()
        self._mark_dirty()

    def _capture_current(self) -> bool:
        row = self._current_row
        if row < 0 or row >= len(self._servers()):
            return True
        server_type = self.server_type.currentData()
        server = {
            "id": self.server_id.text().strip(),
            "type": server_type,
            "enabled": self.server_enabled.isChecked(),
        }
        for key, widget in self._field_widgets.items():
            if isinstance(widget, QSpinBox):
                server[key] = widget.value()
            elif isinstance(widget, QComboBox):
                server[key] = widget.currentData()
            elif isinstance(widget, QLineEdit):
                server[key] = widget.text()
        mappings = []
        for row_index in range(self.mappings_table.rowCount()):
            address = self.mappings_table.item(row_index, 0)
            tag = self.mappings_table.item(row_index, 1)
            access = self.mappings_table.cellWidget(row_index, 2)
            mappings.append(
                {
                    "address": address.text().strip() if address else "",
                    "tag": tag.text().strip() if tag else "",
                    "access": access.currentText() if access else "r",
                }
            )
        server["mappings"] = mappings
        self._servers()[row] = server
        return True

    def _add_server(self):
        if not self._capture_current():
            return
        existing = {str(item.get("id", "")) for item in self._servers()}
        server_id = "modbus"
        index = 1
        while server_id in existing:
            index += 1
            server_id = f"modbus{index}"
        self._servers().append(
            {
                "id": server_id,
                "type": "modbus_tcp_server",
                "enabled": True,
                "host": "0.0.0.0",
                "port": 502,
                "unit_id": 1,
                "word_order": "big",
                "mappings": [],
            }
        )
        self._load_servers(select=len(self._servers()) - 1)
        self.modified.emit()

    def _remove_server(self):
        if not self._capture_current():
            return
        row = self.server_list.currentRow()
        if row < 0:
            return
        answer = QMessageBox.question(
            self,
            "Remove server",
            f"Remove server '{self.server_id.text()}'?",
            QMessageBox.StandardButton.Yes
            | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        del self._servers()[row]
        self._load_servers(select=max(0, row - 1))
        self.modified.emit()

    def _add_mapping(self):
        if self._current_row < 0:
            return
        row = self.mappings_table.rowCount()
        self.mappings_table.insertRow(row)
        server_type = self.server_type.currentData()
        if server_type == "modbus_tcp_server":
            address = "coil:0"
            tag = self._first_tag()
        else:
            tag = self._first_tag()
            address = f"ns=2;s={tag.split('.')[-1] if tag else 'Tag'}"
        self.mappings_table.setItem(row, 0, QTableWidgetItem(address))
        self.mappings_table.setItem(row, 1, QTableWidgetItem(tag))
        access = QComboBox()
        access.addItems(_ACCESS)
        access.currentIndexChanged.connect(self._mark_dirty)
        self.mappings_table.setCellWidget(row, 2, access)
        self.modified.emit()

    def _remove_mapping(self):
        row = self.mappings_table.currentRow()
        if row >= 0:
            self.mappings_table.removeRow(row)
            self.modified.emit()

    def _first_tag(self) -> str:
        for device in self.project.comms_config.get("devices", []):
            device_id = device.get("id", "")
            for tag in device.get("tags", []):
                name = tag.get("name", "")
                if device_id and name:
                    return f"{device_id}.{name}"
        return ""

    def _mark_dirty(self, *_args):
        if not self._loading:
            self.modified.emit()

    def flush_to_project(self) -> bool:
        if self._current_row >= 0 and not self._capture_current():
            return False
        try:
            parse_servers_config(self._draft_config)
        except ServerConfigError as exc:
            self.status.setText(str(exc))
            self.status.setStyleSheet("color: #f14c4c;")
            return False
        self.project.servers_config = copy.deepcopy(self._draft_config)
        self.status.setText("")
        self.status.setStyleSheet("color: #888;")
        return True

    def validate(self) -> tuple[bool, str]:
        try:
            parse_servers_config(self._draft_config)
        except ServerConfigError as exc:
            return False, str(exc)
        return True, ""
