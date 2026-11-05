"""Communications editor tab for the HMI Designer.

The editor changes only a draft of the serialisable ``comms`` configuration.
It never opens devices; deploying the project starts the runtime in the worker.
"""

from __future__ import annotations

import copy
import json

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from falcocv.comms import CommsConfigError, parse_comms_config
from falcocv.comms.drivers import available_drivers, get_driver_schema
from falcocv.comms.drivers.schema import DriverSchema, field_keys
from falcocv.core.hmi import HMIProject

_TAG_COLUMNS = ("Name", "Address", "Type", "Access", "Poll ms", "Default", "Description")


def _bool_text(value) -> str:
    return "true" if value is True else "false"


def _parse_default(text: str, tag_type: str):
    text = text.strip()
    if not text:
        return None
    if tag_type == "bool":
        lowered = text.lower()
        if lowered in ("true", "1", "yes"):
            return True
        if lowered in ("false", "0", "no"):
            return False
        raise ValueError("bool default must be true/false")
    if tag_type == "int":
        return int(text, 10)
    if tag_type == "float":
        return float(text)
    return text


class HMICommsWidget(QWidget):
    """Editor for devices and tags in ``HMIProject.comms_config``."""

    modified = pyqtSignal()

    def __init__(self, project: HMIProject, parent=None):
        super().__init__(parent)
        self.project = project
        self._draft_config = copy.deepcopy(project.comms_config)
        self._loading = False
        self._current_row = -1
        self._driver_schema = get_driver_schema("dummy")
        self._config_widgets: dict[str, QWidget] = {}
        self._build()
        self._load_devices()

    # ── construction ──────────────────────────────────────────────────────

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        helper = QLabel(
            "Configure devices and tags. Changes stay in the project draft until "
            "Save; the worker opens these devices only when the project is deployed."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: #888;")
        layout.addWidget(helper)

        horizontal = QSplitter(Qt.Orientation.Horizontal)
        layout.addWidget(horizontal, 1)

        left = QGroupBox("Devices")
        left_layout = QVBoxLayout(left)
        self.device_list = QListWidget()
        self.device_list.currentRowChanged.connect(self._on_device_selected)
        left_layout.addWidget(self.device_list, 1)
        device_buttons = QHBoxLayout()
        add_device = QPushButton("+ Device")
        remove_device = QPushButton("- Device")
        add_device.clicked.connect(self._add_device)
        remove_device.clicked.connect(self._remove_device)
        device_buttons.addWidget(add_device)
        device_buttons.addWidget(remove_device)
        left_layout.addLayout(device_buttons)
        horizontal.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        vertical = QSplitter(Qt.Orientation.Vertical)
        right_layout.addWidget(vertical, 1)

        device_group = QGroupBox("Device")
        device_form = QFormLayout(device_group)
        self.device_id = QLineEdit()
        self.device_id.setPlaceholderText("device id, e.g. plc")
        device_form.addRow("ID:", self.device_id)

        self.driver_combo = QComboBox()
        self.driver_combo.addItems(available_drivers())
        self.driver_combo.currentTextChanged.connect(self._on_driver_changed)
        device_form.addRow("Driver:", self.driver_combo)

        self.enabled_check = QCheckBox("Enabled")
        self.enabled_check.setChecked(True)
        device_form.addRow("", self.enabled_check)

        self.config_container = QWidget()
        self.config_layout = QFormLayout(self.config_container)
        self.config_layout.setContentsMargins(0, 0, 0, 0)
        device_form.addRow("Configuration:", self.config_container)

        self.advanced_check = QCheckBox("Advanced config (JSON)")
        self.advanced_check.setChecked(False)
        self.advanced_edit = QPlainTextEdit()
        self.advanced_edit.setLineWrapMode(
            QPlainTextEdit.LineWrapMode.NoWrap
        )
        self.advanced_edit.setPlaceholderText(
            'Optional driver-specific keys, for example {"simulate": {...}}'
        )
        self.advanced_edit.setMinimumHeight(180)
        self.advanced_edit.textChanged.connect(self._mark_dirty)
        self.advanced_check.toggled.connect(self.advanced_edit.setVisible)
        self.advanced_edit.hide()
        device_form.addRow(self.advanced_check)
        device_form.addRow(self.advanced_edit)

        self.device_id.textChanged.connect(self._mark_dirty)
        self.enabled_check.toggled.connect(self._mark_dirty)
        vertical.addWidget(device_group)

        tags_group = QGroupBox("Tags")
        tags_layout = QVBoxLayout(tags_group)
        self.tags_table = QTableWidget(0, len(_TAG_COLUMNS))
        self.tags_table.setHorizontalHeaderLabels(_TAG_COLUMNS)
        self.tags_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.tags_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.tags_table.cellChanged.connect(self._on_cell_changed)
        tags_layout.addWidget(self.tags_table, 1)

        tag_buttons = QHBoxLayout()
        add_tag = QPushButton("+ Tag")
        remove_tag = QPushButton("- Tag")
        add_tag.clicked.connect(self._add_tag)
        remove_tag.clicked.connect(self._remove_tag)
        tag_buttons.addWidget(add_tag)
        tag_buttons.addWidget(remove_tag)
        tag_buttons.addStretch()
        tags_layout.addLayout(tag_buttons)
        vertical.addWidget(tags_group)
        vertical.setSizes([330, 470])

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll.setWidget(right)
        horizontal.addWidget(scroll)
        horizontal.setSizes([220, 700])

        self.status = QLabel("")
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color: #888;")
        layout.addWidget(self.status)

    # ── device list / form ────────────────────────────────────────────────

    def _devices(self) -> list[dict]:
        devices = self._draft_config.setdefault("devices", [])
        if not isinstance(devices, list):
            devices = []
            self._draft_config["devices"] = devices
        return devices

    def _mark_dirty(self, *_args):
        if not self._loading:
            self.modified.emit()

    def _load_devices(self, select: int = 0):
        self._loading = True
        try:
            self.device_list.clear()
            for device in self._devices():
                label = f"{device.get('id', '?')}  [{device.get('driver', '?')}]"
                self.device_list.addItem(QListWidgetItem(label))
            if self.device_list.count():
                row = max(0, min(select, self.device_list.count() - 1))
                self.device_list.setCurrentRow(row)
                self._load_device(row)
            else:
                self._clear_device_form()
        finally:
            self._loading = False

    def _clear_device_form(self):
        self._current_row = -1
        self.device_id.clear()
        self.enabled_check.setChecked(True)
        self._rebuild_config_fields(self._driver_schema, {})
        self.advanced_edit.clear()
        self.advanced_check.setChecked(False)
        self.advanced_edit.hide()
        self.tags_table.setRowCount(0)

    def _load_device(self, row: int):
        devices = self._devices()
        if row < 0 or row >= len(devices):
            self._clear_device_form()
            return
        device = devices[row]
        self._current_row = row
        self._loading = True
        try:
            self.device_id.setText(str(device.get("id", "")))
            driver = str(device.get("driver", "dummy"))
            idx = self.driver_combo.findText(driver)
            if idx < 0:
                self.driver_combo.addItem(driver)
                idx = self.driver_combo.count() - 1
            self.driver_combo.setCurrentIndex(idx)
            self.enabled_check.setChecked(bool(device.get("enabled", True)))
            config = device.get("config", {})
            if not isinstance(config, dict):
                config = {}
            schema = get_driver_schema(driver)
            self._rebuild_config_fields(schema, config)
            known = field_keys(schema)
            extra = {k: v for k, v in config.items() if k not in known}
            self.advanced_edit.setPlainText(
                json.dumps(extra, indent=2) if extra else ""
            )
            self.advanced_check.setChecked(bool(extra))
            self.advanced_edit.setVisible(bool(extra))
            self._load_tags(device)
        finally:
            self._loading = False

    def _load_tags(self, device: dict):
        tags = device.get("tags", [])
        self._loading = True
        try:
            self.tags_table.setRowCount(0)
            for tag in tags:
                row = self.tags_table.rowCount()
                self.tags_table.insertRow(row)
                values = [
                    tag.get("name", ""),
                    tag.get("address", ""),
                    tag.get("type", "bool"),
                    tag.get("access", "rw"),
                    str(tag.get("poll_ms", 0)),
                    "" if tag.get("default") is None else str(tag.get("default")),
                    tag.get("description", ""),
                ]
                for col, value in enumerate(values):
                    self.tags_table.setItem(row, col, QTableWidgetItem(str(value)))
        finally:
            self._loading = False

    def _on_device_selected(self, row: int):
        if self._loading:
            return
        if row != self._current_row and not self._capture_current():
            previous = self._current_row
            self._loading = True
            try:
                self.device_list.setCurrentRow(previous)
            finally:
                self._loading = False
            return
        self._loading = True
        try:
            self._load_device(row)
        finally:
            self._loading = False

    def _capture_current(self) -> bool:
        """Copy the current form and tag table into the in-memory draft."""

        if self._current_row < 0:
            return True
        return self._apply_tags(reload=False) and self._apply_device(reload=False)

    # ── driver configuration form ────────────────────────────────────────

    def _clear_config_layout(self):
        while self.config_layout.count():
            item = self.config_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _set_config_value(self, widget: QWidget, value):
        if isinstance(widget, QLineEdit):
            widget.setText("" if value is None else str(value))
        elif isinstance(widget, QSpinBox):
            widget.setValue(int(value) if value is not None else 0)
        elif isinstance(widget, QDoubleSpinBox):
            widget.setValue(float(value) if value is not None else 0.0)
        elif isinstance(widget, QCheckBox):
            widget.setChecked(bool(value))
        elif isinstance(widget, QComboBox):
            idx = widget.findData(value)
            if idx >= 0:
                widget.setCurrentIndex(idx)

    def _config_value(self, field, widget: QWidget):
        if isinstance(widget, QLineEdit):
            value = widget.text()
            if field.kind == "str" and not value.strip() and not field.required:
                return None
            return value
        if isinstance(widget, QSpinBox):
            return int(widget.value())
        if isinstance(widget, QDoubleSpinBox):
            return float(widget.value())
        if isinstance(widget, QCheckBox):
            return bool(widget.isChecked())
        if isinstance(widget, QComboBox):
            return widget.currentData()
        return None

    def _rebuild_config_fields(self, schema: DriverSchema, config: dict):
        self._driver_schema = schema
        self._config_widgets = {}
        self._clear_config_layout()
        for field in schema.fields:
            if field.kind == "bool":
                widget = QCheckBox()
            elif field.kind == "int":
                widget = QSpinBox()
                widget.setRange(
                    int(field.minimum) if field.minimum is not None else -2_147_483_648,
                    int(field.maximum) if field.maximum is not None else 2_147_483_647,
                )
            elif field.kind == "float":
                widget = QDoubleSpinBox()
                widget.setDecimals(3)
                widget.setRange(
                    float(field.minimum) if field.minimum is not None else -1e18,
                    float(field.maximum) if field.maximum is not None else 1e18,
                )
            elif field.kind == "enum":
                widget = QComboBox()
                for label, value in field.choices:
                    widget.addItem(label, value)
            else:
                widget = QLineEdit()
                widget.setPlaceholderText(field.placeholder)
                if field.secret:
                    widget.setEchoMode(QLineEdit.EchoMode.Password)
            if field.help:
                widget.setToolTip(field.help)
            self._set_config_value(widget, config.get(field.key, field.default))
            self._config_widgets[field.key] = widget
            self.config_layout.addRow(field.label + ":", widget)
            self._connect_config_widget(widget)

    def _connect_config_widget(self, widget: QWidget):
        if isinstance(widget, QLineEdit):
            widget.textChanged.connect(self._mark_dirty)
        elif isinstance(widget, QSpinBox):
            widget.valueChanged.connect(self._mark_dirty)
        elif isinstance(widget, QDoubleSpinBox):
            widget.valueChanged.connect(self._mark_dirty)
        elif isinstance(widget, QCheckBox):
            widget.toggled.connect(self._mark_dirty)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._mark_dirty)

    def _on_driver_changed(self, driver: str):
        if self._loading:
            return
        old_config = {}
        try:
            old_config = self._collect_config()
        except Exception:
            pass
        self._loading = True
        try:
            self._rebuild_config_fields(get_driver_schema(driver), old_config)
            has_advanced = bool(self.advanced_edit.toPlainText().strip())
            self.advanced_check.setChecked(has_advanced)
            self.advanced_edit.setVisible(has_advanced)
        finally:
            self._loading = False
        self._mark_dirty()

    def _collect_config(self) -> dict:
        schema = self._driver_schema
        known = field_keys(schema)
        config: dict = {}
        for field in schema.fields:
            widget = self._config_widgets.get(field.key)
            if widget is None:
                continue
            value = self._config_value(field, widget)
            if value is not None:
                config[field.key] = value

        advanced_text = self.advanced_edit.toPlainText().strip()
        if advanced_text:
            try:
                extra = json.loads(advanced_text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Advanced config is not valid JSON: {exc}") from exc
            if not isinstance(extra, dict):
                raise ValueError("Advanced config must be a JSON object")
            for key, value in extra.items():
                if key not in known:
                    config[key] = value
        return config

    def _apply_device(self, reload: bool = True) -> bool:
        devices = self._devices()
        if self._current_row < 0 or self._current_row >= len(devices):
            return False
        device = copy.deepcopy(devices[self._current_row])
        device_id = self.device_id.text().strip()
        if not device_id:
            self._show_error("Device ID is required.")
            return False
        device["id"] = device_id
        device["driver"] = self.driver_combo.currentText()
        device["enabled"] = self.enabled_check.isChecked()
        try:
            device["config"] = self._collect_config()
        except ValueError as exc:
            self._show_error(str(exc))
            return False
        candidate = copy.deepcopy(self._draft_config)
        candidate["devices"][self._current_row] = device
        if not self._validate_candidate(candidate):
            return False
        self._draft_config = candidate
        if reload:
            self._load_devices(select=self._current_row)
        return True

    # ── tags ──────────────────────────────────────────────────────────────

    def _on_cell_changed(self, *_args):
        if not self._loading:
            self.modified.emit()

    def _add_tag(self):
        if self._current_row < 0:
            self._show_error("Select or create a device first.")
            return
        row = self.tags_table.rowCount()
        self.tags_table.insertRow(row)
        name = f"tag{row + 1}"
        values = [name, name, "bool", "rw", "0", "false", ""]
        self._loading = True
        try:
            for col, value in enumerate(values):
                self.tags_table.setItem(row, col, QTableWidgetItem(value))
        finally:
            self._loading = False
        self.tags_table.setCurrentCell(row, 0)
        self.modified.emit()

    def _remove_tag(self):
        row = self.tags_table.currentRow()
        if row >= 0:
            self.tags_table.removeRow(row)
            self.modified.emit()

    def _tags_from_table(self) -> list[dict]:
        tags = []
        for row in range(self.tags_table.rowCount()):
            cells = [
                self.tags_table.item(row, col).text()
                if self.tags_table.item(row, col) else ""
                for col in range(len(_TAG_COLUMNS))
            ]
            name, address, tag_type, access, poll_text, default_text, desc = cells
            poll_text = poll_text.strip() or "0"
            try:
                poll_ms = int(poll_text, 10)
            except ValueError as exc:
                raise ValueError(f"Row {row + 1}: poll ms must be an integer") from exc
            if poll_ms < 0:
                raise ValueError(f"Row {row + 1}: poll ms cannot be negative")
            try:
                default = _parse_default(default_text, tag_type)
            except ValueError as exc:
                raise ValueError(f"Row {row + 1}: {exc}") from exc
            tag = {
                "name": name.strip(),
                "address": address.strip() or name.strip(),
                "type": tag_type.strip() or "bool",
                "access": access.strip() or "rw",
                "poll_ms": poll_ms,
                "description": desc,
            }
            if default is not None:
                tag["default"] = default
            tags.append(tag)
        return tags

    def _apply_tags(self, reload: bool = True) -> bool:
        devices = self._devices()
        if self._current_row < 0 or self._current_row >= len(devices):
            return False
        try:
            tags = self._tags_from_table()
        except ValueError as exc:
            self._show_error(str(exc))
            return False
        candidate = copy.deepcopy(self._draft_config)
        device = copy.deepcopy(candidate["devices"][self._current_row])
        device["tags"] = tags
        candidate["devices"][self._current_row] = device
        if not self._validate_candidate(candidate):
            return False
        self._draft_config = candidate
        if reload:
            self._load_devices(select=self._current_row)
        return True

    # ── validation / persistence ──────────────────────────────────────────

    def _validate_candidate(self, candidate: dict) -> bool:
        try:
            parse_comms_config(candidate)
        except CommsConfigError as exc:
            self._show_error(str(exc))
            return False
        return True

    def flush_to_project(self) -> bool:
        """Commit the validated draft into the working project.

        The working project is still in RAM; only ``HMIProject.save`` writes
        the ``.hmi`` file.
        """

        if self._current_row >= 0 and not self._capture_current():
            return False
        if not self._validate_candidate(self._draft_config):
            return False
        self.project.comms_config = copy.deepcopy(self._draft_config)
        if not self._devices():
            self.status.setText("No devices configured.")
            self.status.setStyleSheet("color: #888;")
        return True

    def validate(self) -> tuple[bool, str]:
        try:
            parse_comms_config(self._draft_config)
        except CommsConfigError as exc:
            return False, str(exc)
        return True, ""

    # ── actions ───────────────────────────────────────────────────────────

    def _add_device(self):
        if not self._capture_current():
            return
        devices = self._devices()
        index = 1
        existing = {str(d.get("id", "")) for d in devices}
        device_id = "device"
        while device_id in existing:
            index += 1
            device_id = f"device_{index}"
        devices.append(
            {
                "id": device_id,
                "driver": "dummy",
                "enabled": True,
                "config": {},
                "tags": [],
            }
        )
        self._load_devices(select=len(devices) - 1)
        self.modified.emit()

    def _remove_device(self):
        if not self._capture_current():
            return
        row = self.device_list.currentRow()
        if row < 0:
            return
        answer = QMessageBox.question(
            self,
            "Remove device",
            f"Remove device '{self.device_list.currentItem().text()}'?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        del self._devices()[row]
        self._load_devices(select=max(0, row - 1))
        self.modified.emit()

    def _show_error(self, message: str):
        self.status.setText(message)
        self.status.setStyleSheet("color: #f14c4c;")
