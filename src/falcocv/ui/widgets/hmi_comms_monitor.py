"""Live signal monitor for the HMI Designer communications tab.

The monitor is a local bench tool: it creates a temporary ``CommsManager`` from
the project configuration and never talks to a deployed worker.  Reads/writes
run in a QThread so a slow PLC cannot block the Qt event loop.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any

from PyQt6.QtCore import QObject, QThread, QTimer, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from falcocv.comms import (
    CommsConfigError,
    CommsMonitorSession,
    TagDef,
    TagValue,
    parse_comms_config,
)
from falcocv.core.config import ConfigManager
from falcocv.core.hmi import HMIProject

_COLUMNS = ("Tag", "Value", "Quality", "Updated", "Access", "Device")


class _MonitorWorker(QObject):
    refreshed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    write_done = pyqtSignal(str, bool, str)

    def __init__(self, session: CommsMonitorSession):
        super().__init__()
        self.session = session

    @pyqtSlot()
    def refresh(self):
        try:
            self.refreshed.emit(self.session.refresh())
        except Exception as exc:
            self.failed.emit(str(exc))

    @pyqtSlot(str, object)
    def write(self, tag_id: str, value: Any):
        try:
            self.session.write(tag_id, value)
        except Exception as exc:
            self.write_done.emit(tag_id, False, str(exc))
            return
        self.write_done.emit(tag_id, True, "")


class _LiveMonitorWorker(QObject):
    """Poll the deployed worker through the HMI HTTP API."""

    refreshed = pyqtSignal(dict)
    failed = pyqtSignal(str)
    write_done = pyqtSignal(str, bool, str)

    def __init__(self, base_url: str, project_id: str):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.project_id = project_id

    def _http_error(self, exc: urllib.error.HTTPError) -> str:
        message = f"HTTP {exc.code}: {exc.reason}"
        try:
            body = exc.read().decode("utf-8")
            data = json.loads(body)
            if isinstance(data, dict):
                message = data.get("error") or data.get("detail") or message
            else:
                message = f"{message}: {data!r}"
        except Exception:
            pass
        return f"{message} (project '{self.project_id}')"

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}/{path.lstrip('/')}"
        sep = "&" if "?" in url else "?"
        url = f"{url}{sep}project_id={self.project_id}"
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise RuntimeError(self._http_error(exc)) from exc
        except Exception as exc:
            raise RuntimeError(
                f"Cannot reach live worker at {self.base_url}: {exc}"
            ) from exc
        if not isinstance(data, dict):
            raise RuntimeError(
                f"Unexpected response from live worker: {type(data).__name__}"
            )
        if data.get("error"):
            raise RuntimeError(str(data["error"]))
        return data

    @pyqtSlot()
    def refresh(self):
        try:
            self.refreshed.emit(self._get("/api/hmi/comms/snapshot"))
        except Exception as exc:
            self.failed.emit(str(exc))

    @pyqtSlot(str, object)
    def write(self, tag_id: str, value: Any):
        try:
            url = f"{self.base_url}/api/hmi/comms/write"
            body = json.dumps(
                {
                    "project_id": self.project_id,
                    "tag": tag_id,
                    "value": value,
                }
            ).encode("utf-8")
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            try:
                with urllib.request.urlopen(req, timeout=5) as response:
                    data = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                raise RuntimeError(self._http_error(exc)) from exc
        except Exception as exc:
            self.write_done.emit(tag_id, False, str(exc))
            return
        if not isinstance(data, dict):
            self.write_done.emit(
                tag_id,
                False,
                f"Unexpected response from live worker: {type(data).__name__}",
            )
            return
        if not data.get("ok"):
            self.write_done.emit(
                tag_id, False, data.get("error", "Write failed")
            )
            return
        self.write_done.emit(tag_id, True, "")


class HMICommsMonitorWidget(QWidget):
    """Read/write monitor for the tags defined in the project."""

    refresh_requested = pyqtSignal()
    write_requested = pyqtSignal(str, object)

    def __init__(self, project: HMIProject, parent=None):
        super().__init__(parent)
        self.project = project
        self._session: CommsMonitorSession | None = None
        self._thread: QThread | None = None
        self._worker: _MonitorWorker | _LiveMonitorWorker | None = None
        self._mode = "local"
        self._snapshot = {}
        self._tags = {}
        self._selected_tag_id = ""
        self._refresh_inflight = False
        self._write_inflight = False
        self._connected = False
        self._connection_state = "disconnected"
        self._live_base = ""
        self._editor_dirty = False

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._request_refresh)

        self._build()
        self._set_connection_state("disconnected")
        self._set_status("Not connected.")

    # ── construction ──────────────────────────────────────────────────────

    def _build(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        top = QHBoxLayout()
        top.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Local test", "local")
        self.mode_combo.addItem("Live worker", "live")
        self.mode_combo.setMinimumWidth(140)
        self.mode_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        top.addWidget(self.mode_combo)
        self.state_label = QLabel("● DISCONNECTED")
        self.state_label.setMinimumWidth(135)
        self.state_label.setStyleSheet("color: #888; font-weight: bold;")
        top.addWidget(self.state_label)
        self.connect_btn = QPushButton("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setEnabled(False)
        self.connect_btn.clicked.connect(self.connect_monitor)
        self.disconnect_btn.clicked.connect(self.disconnect_monitor)
        self.refresh_btn.clicked.connect(self._request_refresh)
        top.addWidget(self.connect_btn)
        top.addWidget(self.disconnect_btn)
        top.addWidget(self.refresh_btn)

        top.addWidget(QLabel("Refresh:"))
        self.interval_combo = QComboBox()
        for label, ms in (("250 ms", 250), ("500 ms", 500),
                          ("1 s", 1000), ("2 s", 2000)):
            self.interval_combo.addItem(label, ms)
        self.interval_combo.setCurrentIndex(1)
        self.interval_combo.setMinimumWidth(120)
        self.interval_combo.setSizeAdjustPolicy(
            QComboBox.SizeAdjustPolicy.AdjustToContents
        )
        self.interval_combo.currentIndexChanged.connect(self._update_timer)
        top.addWidget(self.interval_combo)

        self.auto_check = QCheckBox("Auto refresh")
        self.auto_check.setChecked(True)
        self.auto_check.toggled.connect(self._update_timer)
        top.addWidget(self.auto_check)

        self.live_url_edit = QLineEdit()
        host = ConfigManager().data.get("web_host") or "localhost"
        port = ConfigManager().get_web_port()
        self.live_url_edit.setText(f"http://{host}:{port}")
        self.live_url_edit.setPlaceholderText("http://server:8000")
        self.live_url_edit.hide()
        top.addWidget(self.live_url_edit, 1)

        top.addWidget(QLabel("Filter:"))
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("tag or device")
        self.filter_edit.textChanged.connect(self._populate_table)
        top.addWidget(self.filter_edit, 1)
        layout.addLayout(top)

        helper = QLabel(
            "Local test session: this monitor has its own dummy/PLC connection "
            "and is separate from a deployed HMI worker."
        )
        helper.setWordWrap(True)
        helper.setStyleSheet("color: #888;")
        layout.addWidget(helper)

        edit_group = QGroupBox("Selected tag")
        edit_layout = QFormLayout(edit_group)
        self.selected_label = QLabel("—")
        edit_layout.addRow("Tag:", self.selected_label)

        value_row = QHBoxLayout()
        self.bool_combo = QComboBox()
        self.bool_combo.addItems(["False", "True"])
        self.bool_combo.activated.connect(self._mark_editor_dirty)
        self.bool_combo.hide()
        self.value_edit = QLineEdit()
        self.value_edit.setPlaceholderText("value")
        self.value_edit.textEdited.connect(self._mark_editor_dirty)
        value_row.addWidget(self.bool_combo)
        value_row.addWidget(self.value_edit, 1)
        self.write_btn = QPushButton("Write")
        self.write_btn.setEnabled(False)
        self.write_btn.clicked.connect(self._write_selected)
        value_row.addWidget(self.write_btn)
        self.toggle_btn = QPushButton("Toggle")
        self.toggle_btn.setEnabled(False)
        self.toggle_btn.hide()
        self.toggle_btn.clicked.connect(self._toggle_selected)
        value_row.addWidget(self.toggle_btn)
        edit_layout.addRow("New value:", value_row)

        self.quality_label = QLabel("—")
        edit_layout.addRow("Quality:", self.quality_label)
        layout.addWidget(edit_group)

        self.table = QTableWidget(0, len(_COLUMNS))
        self.table.setHorizontalHeaderLabels(_COLUMNS)
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        layout.addWidget(self.table, 1)

        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def _on_mode_changed(self, _index: int):
        self.live_url_edit.setVisible(
            self.mode_combo.currentData() == "live"
        )
        if self._connected:
            self.disconnect_monitor()

    def connect_monitor(self):
        self.disconnect_monitor()
        if self.mode_combo.currentData() == "live":
            self._connect_live()
        else:
            self._connect_local()

    def _connect_local(self):
        self._mode = "local"
        self._set_connection_state("connecting", "Starting local monitor...")
        try:
            config = parse_comms_config(self.project.comms_config)
        except CommsConfigError as exc:
            self._set_connection_state("error", str(exc))
            self._set_status(str(exc), error=True)
            return
        if not config.devices:
            message = "No devices configured."
            self._set_connection_state("error", message)
            self._set_status(message, error=True)
            return

        session = CommsMonitorSession(config)
        try:
            session.start()
        except Exception as exc:
            message = f"Could not start monitor: {exc}"
            self._set_connection_state("error", message)
            self._set_status(message, error=True)
            return

        self._session = session
        self._tags = {tag.id: tag for tag in session.tags}
        self._thread = QThread(self)
        self._worker = _MonitorWorker(session)
        self._worker.moveToThread(self._thread)
        self.refresh_requested.connect(
            self._worker.refresh,
            Qt.ConnectionType.QueuedConnection,
        )
        self.write_requested.connect(
            self._worker.write,
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker.refreshed.connect(self._on_refreshed)
        self._worker.failed.connect(self._on_refresh_failed)
        self._worker.write_done.connect(self._on_write_done)
        self._thread.start()

        self._connected = True
        self._set_connection_state("connected", "Local monitor")
        self._populate_table()
        self._update_timer()
        self._set_status("Local monitor connected.")
        self._request_refresh()

    def _connect_live(self):
        self._mode = "live"
        base_url = self.live_url_edit.text().strip().rstrip("/")
        if not base_url.startswith(("http://", "https://")):
            message = "Live server URL must start with http:// or https://"
            self._set_connection_state("error", message)
            self._set_status(message, error=True)
            return

        self._live_base = base_url
        self._set_connection_state(
            "connecting",
            f"Connecting to {base_url} for project '{self.project.project_id}'",
        )
        self._tags = {}
        self._snapshot = {}
        self._thread = QThread(self)
        self._worker = _LiveMonitorWorker(base_url, self.project.project_id)
        self._worker.moveToThread(self._thread)
        self.refresh_requested.connect(
            self._worker.refresh,
            Qt.ConnectionType.QueuedConnection,
        )
        self.write_requested.connect(
            self._worker.write,
            Qt.ConnectionType.QueuedConnection,
        )
        self._worker.refreshed.connect(self._on_live_refreshed)
        self._worker.failed.connect(self._on_refresh_failed)
        self._worker.write_done.connect(self._on_write_done)
        self._thread.start()

        # The transport is up, but the worker is not considered connected
        # until the first successful snapshot.
        self._connected = True
        self._populate_table()
        self._update_timer()
        self._set_status(
            f"Connecting to {base_url} · project: {self.project.project_id}"
        )
        self._request_refresh()

    def disconnect_monitor(self):
        self._timer.stop()
        self._connected = False
        self._refresh_inflight = False
        self._write_inflight = False
        self._editor_dirty = False

        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(2000)
            self._thread = None
            self._worker = None
        if self._session is not None:
            self._session.stop()
            self._session = None

        self._set_connection_state("disconnected")
        self.write_btn.setEnabled(False)
        self.toggle_btn.setEnabled(False)
        self._snapshot = {}
        self._populate_table()
        self._set_status("Not connected.")

    def shutdown(self):
        self.disconnect_monitor()

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)

    def _update_timer(self):
        if not self._connected:
            self._timer.stop()
            return
        if self.auto_check.isChecked():
            self._timer.start(int(self.interval_combo.currentData()))
        else:
            self._timer.stop()

    # ── reads ─────────────────────────────────────────────────────────────

    def _request_refresh(self):
        if not self._connected or self._refresh_inflight or self._worker is None:
            return
        self._refresh_inflight = True
        self.refresh_requested.emit()

    def _on_refreshed(self, snapshot: dict):
        self._refresh_inflight = False
        self._snapshot = snapshot
        self._populate_table()
        if not self._editor_dirty:
            self._update_selected_value()
        statuses = self._session.statuses() if self._session else []
        connected = sum(1 for status in statuses if status.connected)
        self._set_connection_state("connected", "Local monitor")
        self._set_status(
            f"Local monitor: {len(snapshot)} tag(s), "
            f"{connected}/{len(statuses)} device(s) connected."
        )

    def _on_live_refreshed(self, payload: dict):
        self._refresh_inflight = False
        if payload.get("error"):
            self._on_refresh_failed(str(payload["error"]))
            return
        tags = {}
        for item in payload.get("tags", []):
            try:
                tag = TagDef(
                    device=item["device"],
                    name=item["name"],
                    address=item.get("address", item["name"]),
                    type=item.get("type", "bool"),
                    access=item.get("access", "rw"),
                    poll_ms=int(item.get("poll_ms", 0)),
                    default=item.get("default"),
                    description=item.get("description", ""),
                )
            except Exception:
                continue
            tags[tag.id] = tag
        values = {}
        for tag_id, item in payload.get("values", {}).items():
            values[tag_id] = TagValue(
                value=item.get("value"),
                quality=item.get("quality", "bad"),
                timestamp=float(item.get("timestamp", 0.0) or 0.0),
                error=item.get("error", ""),
            )
        self._tags = tags
        self._snapshot = values
        self._populate_table()
        if not self._editor_dirty:
            self._update_selected_value()
        statuses = payload.get("statuses", [])
        connected = sum(1 for status in statuses if status.get("connected"))
        warning = payload.get("warning") or ""
        self._set_connection_state("connected", self._live_base)
        message = (
            f"Live: {self._live_base} · project: {self.project.project_id} · "
            f"{len(values)} tag(s), {connected}/{len(statuses)} device(s) connected."
        )
        if warning:
            message += f" {warning}"
        self._set_status(message, error=bool(warning))

    def _on_refresh_failed(self, message: str):
        self._refresh_inflight = False
        self._connected = False
        self._timer.stop()
        self._set_connection_state("error", message)
        self._set_status(message, error=True)

    # ── table / selection ─────────────────────────────────────────────────

    def _filtered_tags(self):
        text = self.filter_edit.text().strip().lower()
        tags = sorted(self._tags.values(), key=lambda tag: tag.id)
        if not text:
            return tags
        return [
            tag for tag in tags
            if text in tag.id.lower() or text in tag.device.lower()
        ]

    def _populate_table(self):
        tags = self._filtered_tags()
        current = self._selected_tag_id
        self.table.blockSignals(True)
        try:
            self.table.setRowCount(len(tags))
            for row, tag in enumerate(tags):
                value = self._snapshot.get(tag.id)
                value_text = "—"
                quality = "—"
                updated = "—"
                if value is not None:
                    value_text = self._format_value(value.value)
                    quality = value.quality
                    updated = time.strftime(
                        "%H:%M:%S", time.localtime(value.timestamp)
                    )
                cells = (
                    tag.id,
                    value_text,
                    quality,
                    updated,
                    tag.access,
                    tag.device,
                )
                for col, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setData(Qt.ItemDataRole.UserRole, tag.id)
                    self.table.setItem(row, col, item)
                if tag.id == current:
                    self.table.selectRow(row)
        finally:
            self.table.blockSignals(False)
        if current:
            for row in range(self.table.rowCount()):
                item = self.table.item(row, 0)
                if item and item.data(Qt.ItemDataRole.UserRole) == current:
                    self.table.selectRow(row)
                    break

    def _on_selection_changed(self):
        row = self.table.currentRow()
        if row < 0:
            self._selected_tag_id = ""
            self.selected_label.setText("—")
            self.write_btn.setEnabled(False)
            self.toggle_btn.setEnabled(False)
            return
        item = self.table.item(row, 0)
        if item is None:
            return
        new_id = item.data(Qt.ItemDataRole.UserRole) or ""
        if new_id != self._selected_tag_id:
            self._selected_tag_id = new_id
            self._editor_dirty = False
        tag = self._tags.get(self._selected_tag_id)
        if tag is None:
            return
        self.selected_label.setText(tag.id)
        if not self._editor_dirty:
            self._update_selected_value()
        writable = (
            self._connection_state == "connected" and tag.access != "r"
        )
        self.write_btn.setEnabled(writable)
        self.toggle_btn.setVisible(tag.type == "bool")
        self.toggle_btn.setEnabled(writable and tag.type == "bool")

    def _mark_editor_dirty(self, *_args):
        self._editor_dirty = True

    def _update_selected_value(self):
        tag = self._tags.get(self._selected_tag_id)
        if tag is None:
            return
        tag_value = self._snapshot.get(tag.id)
        self.quality_label.setText(
            "—" if tag_value is None else tag_value.quality
        )
        if tag.type == "bool":
            self.bool_combo.show()
            self.value_edit.hide()
            self.bool_combo.setCurrentIndex(
                1 if tag_value is not None and bool(tag_value.value) else 0
            )
        else:
            self.bool_combo.hide()
            self.value_edit.show()
            self.value_edit.setText(
                "" if tag_value is None else self._format_value(tag_value.value)
            )

    # ── writes ────────────────────────────────────────────────────────────

    def _write_selected(self):
        tag = self._tags.get(self._selected_tag_id)
        if tag is None or self._worker is None:
            return
        try:
            value = self._read_editor_value(tag.type)
        except ValueError as exc:
            self._set_status(str(exc), error=True)
            return
        self._send_write(tag, value)

    def _toggle_selected(self):
        tag = self._tags.get(self._selected_tag_id)
        if tag is None or tag.type != "bool" or self._worker is None:
            return
        tag_value = self._snapshot.get(tag.id)
        current = bool(tag_value.value) if tag_value is not None else False
        self._send_write(tag, not current)

    def _send_write(self, tag, value):
        self._write_inflight = True
        self.write_btn.setEnabled(False)
        self.toggle_btn.setEnabled(False)
        self._set_status(f"Writing {tag.id}...")
        self.write_requested.emit(tag.id, value)

    def _on_write_done(self, tag_id: str, ok: bool, message: str):
        self._write_inflight = False
        self._editor_dirty = False
        tag = self._tags.get(tag_id)
        writable = (
            self._connection_state == "connected"
            and tag is not None
            and tag.access != "r"
        )
        self.write_btn.setEnabled(writable)
        self.toggle_btn.setEnabled(writable and tag is not None and tag.type == "bool")
        if ok:
            self._set_status(f"Wrote {tag_id}.")
            self._request_refresh()
        else:
            self._set_status(message, error=True)

    def _read_editor_value(self, tag_type: str):
        if tag_type == "bool":
            return self.bool_combo.currentIndex() == 1
        text = self.value_edit.text()
        try:
            if tag_type == "int":
                return int(text, 10)
            if tag_type == "float":
                return float(text)
        except ValueError as exc:
            raise ValueError(f"Invalid {tag_type} value: {text!r}") from exc
        return text

    # ── helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _format_value(value) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value)

    def _set_connection_state(self, state: str, detail: str = ""):
        self._connection_state = state
        colors = {
            "disconnected": "#888888",
            "connecting": "#ffa726",
            "connected": "#4ec9b0",
            "error": "#f14c4c",
        }
        labels = {
            "disconnected": "● DISCONNECTED",
            "connecting": "● CONNECTING...",
            "connected": "● CONNECTED",
            "error": "● ERROR",
        }
        self.state_label.setText(labels.get(state, state.upper()))
        self.state_label.setStyleSheet(
            f"color: {colors.get(state, '#888888')}; font-weight: bold;"
        )
        self.state_label.setToolTip(detail)
        self.connect_btn.setEnabled(state in ("disconnected", "error"))
        self.disconnect_btn.setEnabled(state in ("connecting", "connected"))
        self.refresh_btn.setEnabled(state == "connected")
        if state != "connected":
            self.write_btn.setEnabled(False)
            self.toggle_btn.setEnabled(False)

    def _set_status(self, message: str, error: bool = False):
        self.status_label.setText(message)
        self.status_label.setStyleSheet(
            "color: #f14c4c;" if error else "color: #888;"
        )
