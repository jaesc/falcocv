import asyncio
import json
import os
import shutil
import threading
import zipfile
from urllib.parse import urljoin

from PyQt6.QtCore import Qt, QRectF, QTimer, pyqtSignal, QObject, QFileInfo
from PyQt6.QtGui import QKeySequence, QShortcut, QAction
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QSplitter, QWidget,
    QLabel, QSpinBox, QPushButton, QComboBox, QGroupBox,
    QFormLayout, QLineEdit, QFileDialog, QMessageBox,
    QListWidget, QListWidgetItem, QTabWidget, QTextEdit,
    QDoubleSpinBox, QDialogButtonBox, QApplication, QMenuBar,
    QTreeWidget, QTreeWidgetItem, QInputDialog, QMenu,
    QAbstractItemView, QStyle, QFileIconProvider, QRadioButton,
    QColorDialog, QFrame,
)

from falcocv.core.hmi import HMIProject, HMISlot, WIDGET_TYPES, WIDGET_DEFAULT_SIZES
from falcocv.core.config import ConfigManager
from falcocv.ui.widgets.hmi_scene import HMIWidgetScene, HMIWidgetView, GRID
from falcocv.ui.widgets.hmi_preview import HMIPreviewWidget
from falcocv.ui.widgets.script_editor import ScriptEditor, FindReplaceBar
from falcocv.ui.widgets.debug_panels import DebugToolbar, VariableInspector, WatchPanel
from falcocv.ui.style import SUCCESS, WARNING, INFO, DANGER


class _HMIWSClient(QObject):
    message_received = pyqtSignal(dict)
    connected = pyqtSignal()
    disconnected = pyqtSignal()

    def __init__(self, url):
        super().__init__()
        self._url = url
        self._ws = None
        self._running = False
        self._send_queue = []
        self._send_lock = threading.Lock()

    def start(self):
        self._running = True
        t = threading.Thread(target=self._run, daemon=True)
        t.start()

    def stop(self):
        self._running = False
        if self._ws:
            async def _close():
                await self._ws.close()
            try:
                asyncio.run_coroutine_threadsafe(_close(), self._loop)
            except Exception:
                pass

    def send(self, data: dict):
        with self._send_lock:
            self._send_queue.append(data)

    def _run(self):
        import websockets
        async def _client():
            self._loop = asyncio.get_running_loop()
            while self._running:
                try:
                    async with websockets.connect(self._url) as ws:
                        self._ws = ws
                        self.connected.emit()
                        while self._running:
                            # Process sends
                            with self._send_lock:
                                q = list(self._send_queue)
                                self._send_queue.clear()
                            for msg in q:
                                await ws.send(json.dumps(msg))
                            # Receive (with timeout so sends don't block)
                            try:
                                raw = await asyncio.wait_for(ws.recv(), timeout=0.5)
                                data = json.loads(raw)
                                self.message_received.emit(data)
                            except asyncio.TimeoutError:
                                continue
                except Exception as e:
                    if self._running:
                        self.disconnected.emit()
                        await asyncio.sleep(3)
                finally:
                    self._ws = None
        try:
            asyncio.run(_client())
        except Exception:
            pass


class _WidgetEditPopup(QDialog):
    def __init__(self, data: dict, project: HMIProject, parent=None):
        super().__init__(parent)
        self.data = data
        self.project = project
        self.setWindowFlags(Qt.WindowType.Dialog | Qt.WindowType.WindowCloseButtonHint)
        self.setModal(True)
        self._build()
        self._load()

    def _build(self):
        pass

    def _load(self):
        pass

    def _save(self):
        pass

    def _slot_names(self):
        return [s.name for s in self.project.slots]

    def _get_outputs_for_slot(self, slot_name: str) -> list[str]:
        for s in self.project.slots:
            if s.name == slot_name and s.job_path and os.path.isfile(s.job_path):
                try:
                    with zipfile.ZipFile(s.job_path, "r") as zf:
                        data = json.loads(zf.read("workflow.json"))
                    tools = data.get("tools", [])
                    result = []
                    for t in tools:
                        tname = t.get("name", t.get("type", "?"))
                        for o in t.get("ports", {}).get("outputs", []):
                            result.append(f"{tname} > {o}")
                    if result:
                        return result
                except Exception:
                    pass
                break
        return ["output image"]

    def _on_slot_changed(self, slot_name: str):
        pass


class ImageViewerPopup(_WidgetEditPopup):
    def _build(self):
        self.setWindowTitle("ImageViewer Properties")
        layout = QFormLayout(self)

        self._slot_cb = QComboBox()
        self._slot_cb.addItems(self._slot_names())
        self._slot_cb.currentTextChanged.connect(self._on_slot_changed)
        layout.addRow("Slot:", self._slot_cb)

        self._output_cb = QComboBox()
        layout.addRow("Output:", self._output_cb)

        self._label_edit = QLineEdit()
        layout.addRow("Label:", self._label_edit)

        self._btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._btns.accepted.connect(self._save_and_accept)
        self._btns.rejected.connect(self.reject)
        layout.addRow(self._btns)

    def _load(self):
        s = self.data.get("slot", "")
        idx = self._slot_cb.findText(s)
        if idx >= 0:
            self._slot_cb.setCurrentIndex(idx)
        self._on_slot_changed(self._slot_cb.currentText())
        o = self.data.get("output", "")
        oidx = self._output_cb.findText(o)
        if oidx < 0:
            for i in range(self._output_cb.count()):
                if self._output_cb.itemText(i).endswith(f" > {o}"):
                    oidx = i
                    break
        if oidx >= 0:
            self._output_cb.setCurrentIndex(oidx)
        self._label_edit.setText(self.data.get("label", ""))

    def _on_slot_changed(self, name: str):
        self._output_cb.clear()
        self._output_cb.addItems(self._get_outputs_for_slot(name))

    def _save_and_accept(self):
        self.data["slot"] = self._slot_cb.currentText()
        self.data["output"] = self._output_cb.currentText()
        self.data["label"] = self._label_edit.text()
        self.accept()


class StatusIndicatorPopup(_WidgetEditPopup):
    def _build(self):
        self.setWindowTitle("StatusIndicator Properties")
        layout = QFormLayout(self)
        self._si_colors = {}

        self._slot_cb = QComboBox()
        self._slot_cb.addItems(self._slot_names())
        layout.addRow("Slot:", self._slot_cb)

        self._label_edit = QLineEdit()
        layout.addRow("Label:", self._label_edit)

        for key, label, default in [("color_pass", "Pass", "#2e7d32"),
                                      ("color_fail", "Fail", "#c62828"),
                                      ("color_pending", "Pending", "#c1c100"),
                                      ("color_idle", "Idle", "#888888")]:
            swatch = QLabel()
            swatch.setFixedSize(32, 24)
            self._si_colors[key] = self.data.get(key, default)
            swatch.setStyleSheet(f"background-color: {self._si_colors[key]}; border: 1px solid #888; border-radius: 2px;")
            setattr(self, f"_si_swatch_{key}", swatch)
            pick = QPushButton("Pick...")
            pick.clicked.connect(lambda checked, k=key: self._pick_si_color(k))
            h = QHBoxLayout()
            h.addWidget(swatch)
            h.addWidget(pick)
            h.addWidget(QLabel(label))
            layout.addRow("Color:", h)

        self._btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._btns.accepted.connect(self._save_and_accept)
        self._btns.rejected.connect(self.reject)
        layout.addRow(self._btns)

    def _pick_si_color(self, key):
        from PyQt6.QtGui import QColor
        current = QColor(self._si_colors.get(key, "#888888"))
        c = QColorDialog.getColor(current, self, f"Pick {key.replace('color_', '').title()} Color")
        if c.isValid():
            self._si_colors[key] = c.name()
            swatch = getattr(self, f"_si_swatch_{key}")
            swatch.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #888; border-radius: 2px;")

    def _load(self):
        s = self.data.get("slot", "")
        idx = self._slot_cb.findText(s)
        if idx >= 0:
            self._slot_cb.setCurrentIndex(idx)
        self._label_edit.setText(self.data.get("label", ""))
        defaults = {"color_pass": "#2e7d32", "color_fail": "#c62828",
                     "color_pending": "#c1c100", "color_idle": "#888888"}
        for key, default in defaults.items():
            hex_val = self.data.get(key, default)
            self._si_colors[key] = hex_val
            swatch = getattr(self, f"_si_swatch_{key}")
            swatch.setStyleSheet(f"background-color: {hex_val}; border: 1px solid #888; border-radius: 2px;")

    def _save_and_accept(self):
        self.data["slot"] = self._slot_cb.currentText()
        self.data["label"] = self._label_edit.text()
        for key, hex_val in self._si_colors.items():
            self.data[key] = hex_val
        self.accept()


class MeasureLabelPopup(_WidgetEditPopup):
    def _build(self):
        self.setWindowTitle("MeasureLabel Properties")
        layout = QVBoxLayout(self)

        # ── Data origin ──
        g1 = QGroupBox("Data origin")
        f1 = QFormLayout(g1)
        self._slot_cb = QComboBox()
        self._slot_cb.addItems(self._slot_names())
        self._slot_cb.currentTextChanged.connect(self._on_slot_changed)
        f1.addRow("Slot:", self._slot_cb)
        self._output_cb = QComboBox()
        f1.addRow("Output:", self._output_cb)
        self._field_edit = QLineEdit("[0].value")
        f1.addRow("Field:", self._field_edit)
        layout.addWidget(g1)

        # ── Visual configuration ──
        g2 = QGroupBox("Visual configuration")
        f2 = QFormLayout(g2)
        self._label_edit = QLineEdit()
        f2.addRow("Label:", self._label_edit)
        self._fs_spin = QSpinBox()
        self._fs_spin.setRange(8, 72)
        self._fs_spin.setValue(self.data.get("font_size", 14))
        f2.addRow("Font Size:", self._fs_spin)
        self._ha_cb = QComboBox()
        self._ha_cb.addItems(["left", "center", "right"])
        f2.addRow("H Align:", self._ha_cb)
        self._ll_cb = QComboBox()
        self._ll_cb.addItems(["stacked", "inline"])
        f2.addRow("Label pos:", self._ll_cb)
        layout.addWidget(g2)

        # ── Data evaluation ──
        g3 = QGroupBox("Data evaluation")
        self._eval_form = QFormLayout(g3)
        self._vt_cb = QComboBox()
        self._vt_cb.addItems(["numeric", "text"])
        self._vt_cb.currentTextChanged.connect(self._on_value_type_changed)
        self._eval_form.addRow("Value Type:", self._vt_cb)
        self._cond_cb = QComboBox()
        self._cond_cb.currentTextChanged.connect(self._update_field_visibility)
        self._eval_form.addRow("Condition:", self._cond_cb)
        self._cv_edit = QLineEdit()
        self._eval_form.addRow("Compare Value:", self._cv_edit)
        self._regex_edit = QLineEdit()
        self._eval_form.addRow("Regex:", self._regex_edit)
        self._min_spin = QDoubleSpinBox()
        self._min_spin.setRange(-99999, 99999)
        self._eval_form.addRow("Min:", self._min_spin)
        self._max_spin = QDoubleSpinBox()
        self._max_spin.setRange(-99999, 99999)
        self._max_spin.setValue(100)
        self._eval_form.addRow("Max:", self._max_spin)
        layout.addWidget(g3)

        layout.addStretch()
        self._btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._btns.accepted.connect(self._save_and_accept)
        self._btns.rejected.connect(self.reject)
        layout.addWidget(self._btns)

    def _set_row_visible(self, widget, visible):
        widget.setVisible(visible)
        if hasattr(self, '_eval_form') and self._eval_form:
            lbl = self._eval_form.labelForField(widget)
            if lbl:
                lbl.setVisible(visible)

    def _on_value_type_changed(self, vt: str):
        self._cond_cb.blockSignals(True)
        self._cond_cb.clear()
        if vt == "numeric":
            self._cond_cb.addItems(["between", ">", ">=", "<", "<=", "=", "!="])
        else:
            self._cond_cb.addItems(["equals", "!equals", "contains", "!contains", "regex"])
        self._cond_cb.blockSignals(False)
        self._update_field_visibility()

    def _update_field_visibility(self):
        vt = self._vt_cb.currentText()
        cond = self._cond_cb.currentText()
        is_between = vt == "numeric" and cond == "between"
        is_regex = vt == "text" and cond == "regex"
        show_cv = not is_between and not is_regex
        self._set_row_visible(self._cv_edit, show_cv)
        self._set_row_visible(self._regex_edit, is_regex)
        self._set_row_visible(self._min_spin, is_between)
        self._set_row_visible(self._max_spin, is_between)

    def _load(self):
        self._vt_cb.blockSignals(True)
        self._cond_cb.blockSignals(True)
        s = self.data.get("slot", "")
        idx = self._slot_cb.findText(s)
        if idx >= 0:
            self._slot_cb.setCurrentIndex(idx)
        self._on_slot_changed(self._slot_cb.currentText())
        o = self.data.get("output", "")
        oidx = self._output_cb.findText(o)
        if oidx < 0:
            for i in range(self._output_cb.count()):
                if self._output_cb.itemText(i).endswith(f" > {o}"):
                    oidx = i
                    break
        if oidx >= 0:
            self._output_cb.setCurrentIndex(oidx)
        self._field_edit.setText(self.data.get("field", "[0].value"))
        self._label_edit.setText(self.data.get("label", ""))
        self._fs_spin.setValue(self.data.get("font_size", 14))
        self._ha_cb.setCurrentText(self.data.get("halign", "center"))
        self._ll_cb.setCurrentText(self.data.get("label_layout", "stacked"))
        vt = self.data.get("value_type", "numeric")
        self._vt_cb.setCurrentText(vt)
        self._on_value_type_changed(vt)
        cond = self.data.get("condition", "between")
        self._cond_cb.setCurrentText(cond)
        self._cv_edit.setText(self.data.get("compare_value", ""))
        self._regex_edit.setText(self.data.get("regex", ""))
        self._min_spin.setValue(self.data.get("min", 0))
        self._max_spin.setValue(self.data.get("max", 100))
        self._vt_cb.blockSignals(False)
        self._cond_cb.blockSignals(False)
        self._update_field_visibility()

    def _on_slot_changed(self, name: str):
        self._output_cb.clear()
        self._output_cb.addItems(self._get_outputs_for_slot(name))

    def _save_and_accept(self):
        self.data["slot"] = self._slot_cb.currentText()
        self.data["output"] = self._output_cb.currentText()
        self.data["field"] = self._field_edit.text()
        self.data["label"] = self._label_edit.text()
        self.data["value_type"] = self._vt_cb.currentText()
        self.data["condition"] = self._cond_cb.currentText()
        self.data["compare_value"] = self._cv_edit.text()
        self.data["regex"] = self._regex_edit.text()
        self.data["min"] = self._min_spin.value()
        self.data["max"] = self._max_spin.value()
        self.data["font_size"] = self._fs_spin.value()
        self.data["halign"] = self._ha_cb.currentText()
        self.data["label_layout"] = self._ll_cb.currentText()
        self.accept()


class LedIndicatorPopup(_WidgetEditPopup):
    def _build(self):
        self.setWindowTitle("LedIndicator Properties")
        layout = QFormLayout(self)
        self._led_colors = {}

        self._slot_cb = QComboBox()
        self._slot_cb.addItems(self._slot_names())
        layout.addRow("Slot:", self._slot_cb)

        self._label_edit = QLineEdit()
        layout.addRow("Label:", self._label_edit)

        for key, label, default in [("color_pass", "Pass", "#2e7d32"),
                                      ("color_fail", "Fail", "#c62828"),
                                      ("color_pending", "Pending", "#c1c100"),
                                      ("color_idle", "Idle", "#888888")]:
            swatch = QLabel()
            swatch.setFixedSize(32, 24)
            self._led_colors[key] = self.data.get(key, default)
            swatch.setStyleSheet(f"background-color: {self._led_colors[key]}; border: 1px solid #888; border-radius: 2px;")
            setattr(self, f"_swatch_{key}", swatch)
            pick = QPushButton("Pick...")
            pick.clicked.connect(lambda checked, k=key: self._pick_led_color(k))
            h = QHBoxLayout()
            h.addWidget(swatch)
            h.addWidget(pick)
            h.addWidget(QLabel(label))
            layout.addRow("Color:", h)

        self._btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._btns.accepted.connect(self._save_and_accept)
        self._btns.rejected.connect(self.reject)
        layout.addRow(self._btns)

    def _pick_led_color(self, key):
        from PyQt6.QtGui import QColor
        current = QColor(self._led_colors.get(key, "#888888"))
        c = QColorDialog.getColor(current, self, f"Pick {key.replace('color_', '').title()} Color")
        if c.isValid():
            self._led_colors[key] = c.name()
            swatch = getattr(self, f"_swatch_{key}")
            swatch.setStyleSheet(f"background-color: {c.name()}; border: 1px solid #888; border-radius: 2px;")

    def _load(self):
        s = self.data.get("slot", "")
        idx = self._slot_cb.findText(s)
        if idx >= 0:
            self._slot_cb.setCurrentIndex(idx)
        self._label_edit.setText(self.data.get("label", ""))
        defaults = {"color_pass": "#2e7d32", "color_fail": "#c62828",
                     "color_pending": "#c1c100", "color_idle": "#888888"}
        for key, default in defaults.items():
            hex_val = self.data.get(key, default)
            self._led_colors[key] = hex_val
            swatch = getattr(self, f"_swatch_{key}")
            swatch.setStyleSheet(f"background-color: {hex_val}; border: 1px solid #888; border-radius: 2px;")

    def _save_and_accept(self):
        self.data["slot"] = self._slot_cb.currentText()
        self.data["label"] = self._label_edit.text()
        for key, hex_val in self._led_colors.items():
            self.data[key] = hex_val
        self.accept()


class TextLabelPopup(_WidgetEditPopup):
    def _build(self):
        self.setWindowTitle("TextLabel Properties")
        layout = QFormLayout(self)

        self._text_edit = QLineEdit(self.data.get("label", ""))
        layout.addRow("Text:", self._text_edit)

        self._bg_color = list(self.data.get("bg_color", [45, 45, 45]))
        self._bg_btn = QPushButton()
        self._bg_btn.setFixedSize(32, 24)
        self._update_color_btn(self._bg_btn, self._bg_color)
        self._bg_btn.clicked.connect(lambda: self._pick_color(self._bg_btn, self._bg_color))
        hbg = QHBoxLayout()
        hbg.addWidget(self._bg_btn)
        hbg.addWidget(QLabel("Background"))
        layout.addRow("Bg Color:", hbg)

        self._tc_color = list(self.data.get("text_color", [204, 204, 204]))
        self._tc_btn = QPushButton()
        self._tc_btn.setFixedSize(32, 24)
        self._update_color_btn(self._tc_btn, self._tc_color)
        self._tc_btn.clicked.connect(lambda: self._pick_color(self._tc_btn, self._tc_color))
        htc = QHBoxLayout()
        htc.addWidget(self._tc_btn)
        htc.addWidget(QLabel("Text"))
        layout.addRow("Text Color:", htc)

        self._fs_spin = QSpinBox()
        self._fs_spin.setRange(8, 72)
        self._fs_spin.setValue(self.data.get("font_size", 14))
        layout.addRow("Font Size:", self._fs_spin)

        self._ha_cb = QComboBox()
        self._ha_cb.addItems(["Left", "Center", "Right"])
        self._ha_cb.setCurrentText(self.data.get("halign", "center").capitalize())
        layout.addRow("H Align:", self._ha_cb)

        self._va_cb = QComboBox()
        self._va_cb.addItems(["Top", "Center", "Bottom"])
        self._va_cb.setCurrentText(self.data.get("valign", "center").capitalize())
        layout.addRow("V Align:", self._va_cb)

        self._btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._btns.accepted.connect(self._save_and_accept)
        self._btns.rejected.connect(self.reject)
        layout.addRow(self._btns)

    def _update_color_btn(self, btn, color):
        btn.setStyleSheet(f"background-color: rgb({color[0]},{color[1]},{color[2]}); border: 1px solid #888;")

    def _pick_color(self, btn, storage):
        from PyQt6.QtGui import QColor
        c = QColorDialog.getColor(QColor(*storage), self, "Pick Color")
        if c.isValid():
            storage[0] = c.red()
            storage[1] = c.green()
            storage[2] = c.blue()
            self._update_color_btn(btn, storage)

    def _load(self):
        self._text_edit.setText(self.data.get("label", ""))
        bg = self.data.get("bg_color", [45, 45, 45])
        self._bg_color[:] = bg
        self._update_color_btn(self._bg_btn, self._bg_color)
        tc = self.data.get("text_color", [204, 204, 204])
        self._tc_color[:] = tc
        self._update_color_btn(self._tc_btn, self._tc_color)
        self._fs_spin.setValue(self.data.get("font_size", 14))
        self._ha_cb.setCurrentText(self.data.get("halign", "center").capitalize())
        self._va_cb.setCurrentText(self.data.get("valign", "center").capitalize())

    def _save_and_accept(self):
        self.data["label"] = self._text_edit.text()
        self.data["bg_color"] = list(self._bg_color)
        self.data["text_color"] = list(self._tc_color)
        self.data["font_size"] = self._fs_spin.value()
        self.data["halign"] = self._ha_cb.currentText().lower()
        self.data["valign"] = self._va_cb.currentText().lower()
        self.accept()


class LogPanelPopup(_WidgetEditPopup):
    def _build(self):
        self.setWindowTitle("LogPanel Properties")
        layout = QFormLayout(self)
        self._fs_spin = QSpinBox()
        self._fs_spin.setRange(8, 72)
        self._fs_spin.setValue(self.data.get("font_size", 14))
        layout.addRow("Font Size:", self._fs_spin)
        self._btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self._btns.accepted.connect(self._save_and_accept)
        self._btns.rejected.connect(self.reject)
        layout.addRow(self._btns)

    def _load(self):
        self._fs_spin.setValue(self.data.get("font_size", 14))

    def _save_and_accept(self):
        self.data["font_size"] = self._fs_spin.value()
        self.accept()


POPUPS = {
    "ImageViewer": ImageViewerPopup,
    "StatusIndicator": StatusIndicatorPopup,
    "LedIndicator": LedIndicatorPopup,
    "MeasureLabel": MeasureLabelPopup,
    "TextLabel": TextLabelPopup,
    "LogPanel": LogPanelPopup,
}


class HMIDesignerDialog(QDialog):
    def __init__(self, project: HMIProject | None = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("HMI Designer")
        self.project = project
        self._current_path: str | None = None
        self._modified = False
        self._loading = False
        self._saving = False
        self.setMinimumSize(1100, 700)
        self.setWindowFlags(
            self.windowFlags()
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )

        self._file_tree: QTreeWidget | None = None
        self._editor_tabs: QTabWidget | None = None
        self._editor_by_path: dict[str, ScriptEditor] = {}
        self._tab_path: list[str] = []

        self._tabs: QTabWidget | None = None
        self._scene: HMIWidgetScene | None = None
        self._view: HMIWidgetView | None = None
        self._layout_splitter: QSplitter | None = None
        self._script_splitter: QSplitter | None = None
        self._debug_splitter: QSplitter | None = None
        self.script_edit: ScriptEditor | None = None
        self._find_bar: FindReplaceBar | None = None
        self._debug_toolbar: DebugToolbar | None = None
        self._var_inspector: VariableInspector | None = None
        self._watch_panel: WatchPanel | None = None
        self._debug_container: QWidget | None = None
        self.rows_spin: QSpinBox | None = None
        self.cols_spin: QSpinBox | None = None
        self._ws_client: _HMIWSClient | None = None
        self._welcome_label: QLabel | None = None

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ── Menu bar ──
        menu_bar = QMenuBar(self)

        file_menu = menu_bar.addMenu("File")
        file_menu.addAction("New...", self._new_project)
        file_menu.addAction("New script project", self._new_script_project)
        file_menu.addAction("New multi-file project", self._new_multi_project)
        file_menu.addSeparator()
        file_menu.addAction("Open HMI project...", self._open_project)
        file_menu.addAction("Close project", self._close_project_action)
        file_menu.addSeparator()
        file_menu.addAction("Save", self._save_project)
        file_menu.addAction("Save as...", self._save_project_as)
        file_menu.addSeparator()
        file_menu.addAction("Exit", self.close)

        settings_menu = menu_bar.addMenu("Settings")
        settings_menu.addAction("Project ID...", self._on_project_id_settings)
        settings_menu.addSeparator()
        settings_menu.addAction("Grid size...", self._on_grid_settings)
        self._show_grid_action = QAction("Show grid", self, checkable=True)
        self._show_grid_action.setChecked(True)
        self._show_grid_action.toggled.connect(self._on_show_grid_toggled)
        settings_menu.addAction(self._show_grid_action)
        settings_menu.addAction("Server connection...", self._on_server_settings)

        help_menu = menu_bar.addMenu("Help")
        help_menu.addAction("HMI Reference...", self._on_hmi_help)
        help_menu.addAction("HMI Scripting...", self._on_hmi_scripting_help)

        main_layout.addWidget(menu_bar)

        # ── Hidden state widgets (no layout, just signals) ──
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(10, 100)
        self.rows_spin.setValue(56)
        self.rows_spin.valueChanged.connect(self._on_grid_size_changed)
        self.rows_spin.valueChanged.connect(self._update_footer)

        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(10, 100)
        self.cols_spin.setValue(100)
        self.cols_spin.valueChanged.connect(self._on_grid_size_changed)
        self.cols_spin.valueChanged.connect(self._update_footer)

        host_val = ConfigManager().data.get("web_host") or "localhost"
        self._server_host = QLineEdit(host_val)
        self._server_host.textChanged.connect(
            lambda t: ConfigManager().data.__setitem__("web_host", t))
        self._server_host.textChanged.connect(self._update_footer)

        self._server_port = QLineEdit(str(ConfigManager().get_web_port()))
        self._server_port.textChanged.connect(
            lambda t: ConfigManager().set_web_port(int(t)) if t.isdigit() else None)
        self._server_port.textChanged.connect(self._update_footer)

        # ── Welcome / project area ──
        self._welcome_label = QLabel("No project loaded.\nClick File > New or File > Open to begin.")
        self._welcome_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._welcome_label.setStyleSheet("color: #666; font: 16px sans-serif;")

        if project:
            self._build_from_project()
            self._welcome_label.hide()
        main_layout.addWidget(self._welcome_label, 1)

        # ── Footer ──
        footer_layout = QHBoxLayout()
        self._footer_sources = QLabel()
        self._footer_sources.setStyleSheet("color: #888; font: 11px sans-serif;")
        footer_layout.addWidget(self._footer_sources)
        footer_layout.addStretch()
        self._footer_label = QLabel()
        self._footer_label.setStyleSheet("color: #888; font: 11px sans-serif;")
        footer_layout.addWidget(self._footer_label)
        main_layout.addLayout(footer_layout)
        self._update_footer()

        self._ws_client = None
        self._last_debug_sizes = None
        QTimer.singleShot(100, self._start_ws)

    # ── Menu actions ──────────────────────────────────────────

    def _new_script_project(self):
        if not self._close_project():
            return
        self.project = HMIProject()
        self._current_path = None
        self._modified = False
        self._build_from_project()

    def _new_multi_project(self):
        dlg = _NewProjectDialog(self)
        dlg._rb_multi.setChecked(True)
        dlg._on_type_changed()
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if not self._close_project():
            return
        self.project = HMIProject()
        self._current_path = None
        self._modified = False
        self._set_source_dir(dlg.source_dir)
        self.project.entry_point = dlg.entry_point
        self._build_from_project()

    def _close_project_action(self):
        if not self.project:
            return
        self._close_project()
        self._welcome_label.show()

    def _on_grid_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Grid Settings")
        layout = QFormLayout(dlg)
        rs = QSpinBox()
        rs.setRange(10, 100)
        rs.setValue(self.rows_spin.value())
        layout.addRow("Rows:", rs)
        cs = QSpinBox()
        cs.setRange(10, 100)
        cs.setValue(self.cols_spin.value())
        layout.addRow("Cols:", cs)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self.rows_spin.setValue(rs.value())
            self.cols_spin.setValue(cs.value())

    def _on_hmi_help(self):
        from falcocv.ui.widgets.help_dialog import ToolHelpDialog
        ToolHelpDialog(parent=self, preselected_key="HMI Project").exec()

    def _on_hmi_scripting_help(self):
        from falcocv.ui.widgets.help_dialog import ToolHelpDialog
        ToolHelpDialog(parent=self, preselected_key="HMI Scripting").exec()

    def _on_server_settings(self):
        dlg = QDialog(self)
        dlg.setWindowTitle("Server Connection")
        layout = QFormLayout(dlg)
        he = QLineEdit(self._server_host.text())
        layout.addRow("Host:", he)
        pe = QLineEdit(str(ConfigManager().get_web_port()))
        layout.addRow("Port:", pe)
        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addRow(btns)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._server_host.setText(he.text())
            self._server_port.setText(pe.text())

    def _on_project_id_settings(self):
        if not self.project:
            return
        name, ok = QInputDialog.getText(self, "Project ID",
            "Identifier (letters, numbers, _, -):",
            text=self.project.project_id)
        if ok and name and name != self.project.project_id:
            try:
                self.project.set_project_id(name)
            except ValueError as e:
                QMessageBox.warning(self, "Invalid ID", str(e))
                return
            self._update_footer()
            self._mark_modified()

    def _update_footer(self):
        host = self._server_host.text() or "localhost"
        port = self._server_port.text() or str(ConfigManager().get_web_port())
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        self._footer_label.setText(
            f"Grid: {rows}\u00d7{cols}    |    Server: {host}:{port}")
        if self.project:
            parts = [f"Project: {self.project.project_id}"]
            if self.project.source_dir:
                parts.append(f"Sources: {os.path.abspath(self.project.source_dir)}")
            else:
                parts.append("Sources: (single)")
            if self.btn_stop.isEnabled():
                parts.append(f"URL: http://{host}:{port}/?id={self.project.project_id}")
            self._footer_sources.setText("    |    ".join(parts))
        else:
            self._footer_sources.setText("")

    # ── Panels ─────────────────────────────────────────────────

    def _build_from_project(self):
        self._welcome_label.hide()
        if self._tabs:
            self._tabs.deleteLater()

        main_layout = self.layout()

        self.rows_spin.blockSignals(True)
        self.rows_spin.setValue(self.project.layout.get("rows", 56))
        self.rows_spin.blockSignals(False)
        self.cols_spin.blockSignals(True)
        self.cols_spin.setValue(self.project.layout.get("cols", 100))
        self.cols_spin.blockSignals(False)

        self._tabs = QTabWidget()

        layout_tab = QWidget()
        lt = QHBoxLayout(layout_tab)
        lt.setContentsMargins(0, 0, 0, 0)

        self._layout_splitter = QSplitter(Qt.Orientation.Horizontal)
        left_panel = self._build_left_panel()
        self._layout_splitter.addWidget(left_panel)

        self._scene = HMIWidgetScene(self.project)
        self._view = HMIWidgetView(self._scene)
        self._scene.widget_double_clicked.connect(self._on_widget_double_clicked)
        self._scene.project_modified.connect(self._mark_modified)
        self._layout_splitter.addWidget(self._view)
        self._layout_splitter.setSizes([220, 700])
        self._layout_splitter.setStretchFactor(0, 0)
        self._layout_splitter.setStretchFactor(1, 1)

        script_tab = self._build_script_tab()
        self._tabs.addTab(script_tab, "Script")

        lt.addWidget(self._layout_splitter)
        self._tabs.addTab(layout_tab, "HMI Layout")

        self._preview = HMIPreviewWidget(self.project)
        self._tabs.addTab(self._preview, "Preview")

        main_layout.insertWidget(main_layout.count() - 1, self._tabs, 1)

        # ── Deploy buttons (tab corner) ──
        self.btn_deploy = QPushButton("\u25b6 Deploy")
        self.btn_deploy.clicked.connect(self._on_deploy)
        self.btn_deploy.setStyleSheet("color: #4ec9b0; font-weight: bold;")

        self.btn_stop = QPushButton("\u25a0 Stop")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("color: #f14c4c; font-weight: bold;")

        self._deploy_status = QLabel("Idle")
        self._deploy_status.setFixedWidth(130)

        self.btn_check = QPushButton("Check")
        self.btn_check.clicked.connect(self._on_check)
        self.btn_check.setStyleSheet("color: #569cd6;")

        corner = QWidget()
        cl = QHBoxLayout(corner)
        cl.setContentsMargins(0, 0, 8, 0)
        cl.setSpacing(4)
        cl.addWidget(self.btn_check)
        cl.addWidget(self.btn_deploy)
        cl.addWidget(self.btn_stop)
        cl.addWidget(self._deploy_status)
        cl.addStretch()
        self._tabs.setCornerWidget(corner, Qt.Corner.TopRightCorner)

        if self.script_edit:
            self.script_edit.textChanged.connect(self._mark_modified)

        self._loading = True
        self._rebuild_slot_list()
        self._rebuild_widget_list()
        if self._scene:
            self._scene.rebuild_from_project()
        self._loading = False
        self.update_title()
        QTimer.singleShot(100, self._start_ws)
        self._update_footer()

    def _build_left_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Slots
        slots_g = QGroupBox("Job Slots")
        sl = QVBoxLayout(slots_g)
        st = QHBoxLayout()
        self.slot_list = QListWidget()
        self.slot_list.currentRowChanged.connect(self._on_slot_selected)
        st.addWidget(self.slot_list)
        sb = QVBoxLayout()
        for lbl, fn in [("+", self._add_slot), ("-", self._del_slot)]:
            b = QPushButton(lbl)
            b.setFixedWidth(28)
            b.clicked.connect(fn)
            sb.addWidget(b)
        sb.addStretch()
        st.addLayout(sb)
        sl.addLayout(st)

        self.slot_name_edit = QLineEdit()
        self.slot_name_edit.setPlaceholderText("Slot name")
        self.slot_name_edit.textChanged.connect(self._on_slot_name_changed)
        sl.addWidget(self.slot_name_edit)

        sp = QHBoxLayout()
        self.slot_path_edit = QLineEdit()
        self.slot_path_edit.setPlaceholderText("Path to .cvj")
        self.slot_path_edit.textChanged.connect(self._on_slot_path_changed)
        sp.addWidget(self.slot_path_edit)
        b2 = QPushButton("...")
        b2.setFixedWidth(30)
        b2.clicked.connect(self._browse_job)
        sp.addWidget(b2)
        sl.addLayout(sp)

        layout.addWidget(slots_g)

        # Widgets
        wg = QGroupBox("Widgets")
        wl = QVBoxLayout(wg)
        wt = QHBoxLayout()
        self.widget_list = QListWidget()
        self.widget_list.currentRowChanged.connect(self._on_widget_selected)
        self.widget_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.widget_list.customContextMenuRequested.connect(self._on_widget_list_context_menu)
        wt.addWidget(self.widget_list)
        wb = QVBoxLayout()
        for lbl, fn in [("+", self._add_widget_from_list), ("-", self._del_widget)]:
            b = QPushButton(lbl)
            b.setFixedWidth(28)
            b.clicked.connect(fn)
            wb.addWidget(b)
        wb.addStretch()
        wt.addLayout(wb)
        wl.addLayout(wt)
        layout.addWidget(wg)

        return panel

    def _build_script_tab(self):
        if self.project.source_dir and os.path.isdir(self.project.source_dir):
            return self._build_script_tab_multi()
        return self._build_script_tab_single()

    def _build_script_tab_single(self):
        script_tab = QWidget()
        sl = QVBoxLayout(script_tab)
        sl.setContentsMargins(0, 0, 0, 0)

        # Vertical splitter: editor | debug
        self._script_splitter = QSplitter(Qt.Orientation.Vertical)
        self._script_splitter.setHandleWidth(6)

        # ── Editor ────────────────────────────────────────────
        editor_container = QWidget()
        editor_lyt = QVBoxLayout(editor_container)
        editor_lyt.setContentsMargins(0, 0, 0, 0)
        self.script_edit = ScriptEditor()
        self.script_edit.setText(self.project.script)
        self.script_edit.textChanged.connect(self._mark_modified)
        editor_lyt.addWidget(self.script_edit, 1)
        self._find_bar = FindReplaceBar(self.script_edit)
        self.script_edit.set_find_bar(self._find_bar)
        editor_lyt.addWidget(self._find_bar)
        self._script_splitter.addWidget(editor_container)

        # ── Debug container (VarInspector | WatchPanel) ───────
        debug_container = QWidget()
        debug_lyt = QVBoxLayout(debug_container)
        debug_lyt.setContentsMargins(0, 0, 0, 0)
        debug_lyt.setSpacing(0)

        self._debug_toolbar = DebugToolbar()
        self._debug_toolbar.continue_requested.connect(self._on_debug_continue)
        self._debug_toolbar.step_over_requested.connect(self._on_debug_step_over)
        self._debug_toolbar.step_into_requested.connect(self._on_debug_step_into)
        self._debug_toolbar.step_out_requested.connect(self._on_debug_step_out)
        self._debug_toolbar.stop_requested.connect(self._on_debug_stop)
        debug_lyt.addWidget(self._debug_toolbar)

        self._debug_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._debug_splitter.setHandleWidth(6)
        self._var_inspector = VariableInspector()
        self._var_inspector.show()
        self._debug_splitter.addWidget(self._var_inspector)
        self._watch_panel = WatchPanel()
        self._watch_panel.show()
        self._debug_splitter.addWidget(self._watch_panel)
        self._debug_splitter.setSizes([350, 350])
        debug_lyt.addWidget(self._debug_splitter, 1)

        self._debug_container = debug_container
        self._script_splitter.addWidget(debug_container)
        self._script_splitter.setSizes([500, 300])
        self._script_splitter.splitterMoved.connect(self._on_debug_splitter_moved)

        sl.addWidget(self._script_splitter, 1)

        # Find/Replace shortcuts
        sc_find = QShortcut(QKeySequence("Ctrl+F"), self.script_edit)
        sc_find.activated.connect(self._on_script_find)
        sc_find_replace = QShortcut(QKeySequence("Ctrl+H"), self.script_edit)
        sc_find_replace.activated.connect(self._on_script_find_replace)

        # Show DebugToolbar when breakpoints exist + send hot BPs
        self.script_edit.breakpoints_changed.connect(self._update_debug_visibility)
        self.script_edit.breakpoints_changed.connect(self._on_breakpoints_changed)
        QTimer.singleShot(0, self._update_debug_visibility)

        # Debug keyboard shortcuts
        sc = QShortcut(QKeySequence("F5"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_debug_continue)
        self._sc_continue = sc
        sc = QShortcut(QKeySequence("F10"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_debug_step_over)
        self._sc_step_over = sc
        sc = QShortcut(QKeySequence("F11"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_debug_step_into)
        self._sc_step_into = sc
        sc = QShortcut(QKeySequence("Shift+F11"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_debug_step_out)
        self._sc_step_out = sc

        sc = QShortcut(QKeySequence("F2"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_f2_rename)
        self._sc_f2 = sc

        return script_tab

    # ── Deploy / Check / Stop ─────────────────────────────────

    def _deploy_url(self):
        host = self._server_host.text() or "localhost"
        port = int(self._server_port.text()) if self._server_port.text().isdigit() else ConfigManager().get_web_port()
        return f"http://{host}:{port}"

    def _ws_url(self):
        host = self._server_host.text() or "localhost"
        port = int(self._server_port.text()) if self._server_port.text().isdigit() else ConfigManager().get_web_port()
        return f"ws://{host}:{port}/api/ws?project_id={self.project.project_id}"

    def _check_compile(self) -> tuple[bool, str]:
        errors = []
        if self._editor_by_path:
            for abspath, editor in self._editor_by_path.items():
                try:
                    content = self._tab_to_spaces(editor.text())
                    compile(content, abspath, "exec")
                except SyntaxError as e:
                    errors.append(f"  File \"{os.path.basename(abspath)}\", line {e.lineno}\n    {e.msg}")
        else:
            script = self.project.script if self.project else ""
            ed = self._get_active_editor()
            if ed:
                script = self._tab_to_spaces(ed.text())
            try:
                compile(script, "<hmi_script>", "exec")
            except SyntaxError as e:
                errors.append(f"  File \"<hmi_script>\", line {e.lineno}\n    {e.msg}")
        if errors:
            return False, "Syntax errors found:\n" + "\n".join(errors)
        return True, ""

    def _show_compile_errors(self, text: str):
        dlg = QDialog(self)
        dlg.setWindowTitle("Compilation Error")
        dlg.setMinimumSize(500, 300)
        layout = QVBoxLayout(dlg)
        edit = QTextEdit()
        edit.setReadOnly(True)
        edit.setPlainText(text)
        layout.addWidget(edit, 1)
        hb = QHBoxLayout()
        copy_btn = QPushButton("Copy")
        copy_btn.clicked.connect(lambda: QApplication.clipboard().setText(edit.toPlainText()))
        hb.addWidget(copy_btn)
        hb.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(dlg.accept)
        hb.addWidget(close_btn)
        layout.addLayout(hb)
        dlg.exec()

    def _on_check(self):
        ok, msg = self._check_compile()
        if ok:
            QMessageBox.information(self, "Check", "No syntax errors found.")
        else:
            self._show_compile_errors(msg)

    def _stop_sync(self) -> bool:
        """Send stop request to server. Returns True if server acknowledged."""
        import urllib.request
        body = json.dumps({"project_id": self.project.project_id}).encode("utf-8")
        url = urljoin(self._deploy_url().rstrip("/") + "/", "api/hmi/stop")
        try:
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=5):
                return True
        except Exception:
            return False

    def _on_deploy(self):
        self._flush_to_project()
        if self.btn_stop.isEnabled():
            self._set_status("Stopping...", WARNING)
            self.btn_deploy.setEnabled(False)
            self.btn_stop.setEnabled(False)
            self._stop_sync()

        self._set_status("Deploying...", INFO)
        self.btn_deploy.setEnabled(False)
        self.btn_stop.setEnabled(False)

        ok, errors = self._check_compile()
        if not ok:
            self._show_compile_errors(errors)
            self.btn_deploy.setEnabled(True)
            self._set_status("Compile error", WARNING)
            return

        parent = self.parent()
        if parent and hasattr(parent, '_cleanup_workflow'):
            parent._cleanup_workflow()

        self._var_inspector.clear_variables()
        self._watch_panel.clear_values()
        ed = self._get_active_editor()
        if ed:
            ed.clear_debug_line()
        self._debug_container.hide()

        all_bps = self._all_breakpoints()
        bps = list(all_bps.get("<hmi_script>", set()))
        if not bps:
            for v in all_bps.values():
                bps.extend(v)

        payload: dict[str, object] = {
            "script": self.project.script,
            "layout": self.project.layout,
            "slots": [
                {"name": s.name, "job_path": s.job_path}
                for s in self.project.slots
            ],
        }

        if self.project.source_dir and os.path.isdir(self.project.source_dir):
            payload["files"] = self.project.get_source_files()
            payload["entry_point"] = self.project.entry_point

        payload["bp_by_file"] = {k: list(v) for k, v in all_bps.items()}
        payload["project_id"] = self.project.project_id

        import urllib.request
        url = urljoin(self._deploy_url().rstrip("/") + "/", "api/hmi/load")
        try:
            body = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=body, method="POST")
            req.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
        except Exception as e:
            import traceback
            self._show_error("Deploy Error",
                f"URL: {url}\n\nError: {e}\n\n{traceback.format_exc()}")
            self.btn_deploy.setEnabled(True)
            self._set_status("Deploy failed", WARNING)
            return

        if not result.get("ok"):
            self._show_error("Deploy Error", f"Server error: {result.get('error', 'unknown')}")
            self.btn_deploy.setEnabled(True)
            self._set_status("Deploy failed", WARNING)
            return

        if bps:
            self._debug_container.show()
            self._debug_toolbar.show()
            self._debug_toolbar.set_state("Deployed, waiting...", INFO)

        self.btn_deploy.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self._set_status("Deployed", SUCCESS)

    def _on_stop(self):
        self._set_status("Stopping...", WARNING)
        self.btn_deploy.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self._stop_sync()
        self.btn_deploy.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("Stopped", WARNING)
        ed = self._get_active_editor()
        if ed:
            ed.clear_debug_line()
        self._debug_container.hide()

    def _show_loading(self, text="Loading..."):
        self._loading_dlg = QDialog(self)
        self._loading_dlg.setWindowTitle("Please wait")
        self._loading_dlg.setWindowModality(Qt.WindowModality.ApplicationModal)
        self._loading_dlg.setFixedSize(300, 80)
        layout = QVBoxLayout(self._loading_dlg)
        layout.addStretch()
        lbl = QLabel(text)
        lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(lbl)
        layout.addStretch()
        self._loading_dlg.show()
        QApplication.processEvents()

    def _hide_loading(self):
        dlg = getattr(self, '_loading_dlg', None)
        if dlg:
            dlg.close()
            self._loading_dlg = None

    def _show_error(self, title: str, text: str):
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumSize(600, 300)
        layout = QVBoxLayout(dlg)
        te = QTextEdit()
        te.setReadOnly(True)
        te.setPlainText(text)
        layout.addWidget(te)
        btn = QPushButton("Close")
        btn.clicked.connect(dlg.accept)
        layout.addWidget(btn)
        dlg.exec()

    def _set_status(self, text, color):
        self._deploy_status.setText(text)
        self._deploy_status.setStyleSheet(f"color: {color}; font-weight: bold;")

    def _on_script_find(self):
        ed = self._get_active_editor()
        sel = ""
        if ed and ed.hasSelectedText():
            sel = ed.selectedText()
        self._find_bar.show(text=sel)

    def _on_script_find_replace(self):
        self._on_script_find()
        self._find_bar.input_replace.setFocus()

    # ── WS message handler ──────────────────────────────────

    def _on_ws_message(self, data: dict):
        msg_type = data.get("type", "")
        if msg_type == "hmi_debug_paused":
            line = data.get("line", 0)
            locals_dict = data.get("locals", {})
            fn = data.get("file", "<hmi_script>")
            self._on_debugger_paused_ui(line, locals_dict, fn)
        elif msg_type == "hmi_debug_resumed":
            self._on_debugger_resumed_ui()
        elif msg_type == "hmi_stopped":
            self._on_stopped_ui()
        elif msg_type == "hmi_loaded":
            self._set_status("Deployed", SUCCESS)
            self._update_footer()

    def _on_stopped_ui(self):
        self.btn_deploy.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._set_status("Stopped", WARNING)
        ed = self._get_active_editor()
        if ed:
            ed.clear_debug_line()
        self._debug_container.hide()

    # ── Debug callbacks ──────────────────────────────────────

    def _send_debug_cmd(self, cmd: str):
        if self._ws_client:
            self._ws_client.send({"cmd": cmd})

    def _on_debug_continue(self):
        self._send_debug_cmd("hmi_debug_continue")

    def _on_debug_step_over(self):
        self._send_debug_cmd("hmi_debug_step_over")

    def _on_debug_step_into(self):
        self._send_debug_cmd("hmi_debug_step_into")

    def _on_debug_step_out(self):
        self._send_debug_cmd("hmi_debug_step_out")

    def _on_debug_stop(self):
        self._send_debug_cmd("hmi_debug_stop")

    def _on_debugger_paused_ui(self, line, locals_dict, filename="<hmi_script>"):
        editor = self._get_active_editor()
        if self._editor_by_path:
            for abspath, ed in self._editor_by_path.items():
                ed.clear_debug_line()
                if os.path.basename(abspath) == filename or abspath.endswith(filename.replace("/", os.sep)):
                    self._open_file(abspath)
                    editor = ed
        editor.set_debug_line(line)
        self._debug_container.show()
        if locals_dict and locals_dict != getattr(self, '_last_debug_locals', None):
            self._last_debug_locals = locals_dict
            self._var_inspector.update_variables(locals_dict)
            self._watch_panel.update_values(locals_dict)
        self._debug_toolbar.set_state(f"Paused at line {line}", WARNING)

    def _on_debugger_resumed_ui(self):
        ed = self._get_active_editor()
        if ed:
            ed.clear_debug_line()

    def _on_breakpoints_changed(self, bps):
        if self._ws_client:
            all_bps = self._all_breakpoints()
            if all_bps.get("<hmi_script>"):
                self._ws_client.send({"cmd": "hmi_set_breakpoints", "bps": list(all_bps["<hmi_script>"])})
            elif any(k != "<hmi_script>" for k in all_bps):
                self._ws_client.send({"cmd": "hmi_set_breakpoints", "bps": {k: list(v) for k, v in all_bps.items()}})
            else:
                self._ws_client.send({"cmd": "hmi_set_breakpoints", "bps": []})

    def _start_ws(self):
        if not hasattr(self, '_server_host') or self._server_host is None:
            return
        try:
            client = _HMIWSClient(self._ws_url())
            client.message_received.connect(self._on_ws_message)
            client.start()
            self._ws_client = client
        except Exception as e:
            pass

    def _on_debug_splitter_moved(self, pos, index):
        splitter = getattr(self, '_debug_vsplitter', None) or self._script_splitter
        self._last_debug_sizes = list(splitter.sizes())

    def _update_debug_visibility(self):
        if self._editor_by_path:
            has_bps = any(e.get_breakpoints() for e in self._editor_by_path.values())
        else:
            ed = self._get_active_editor()
            has_bps = len(ed.get_breakpoints()) > 0 if ed else False
        splitter = getattr(self, '_debug_vsplitter', None) or self._script_splitter
        saved = splitter.sizes()
        if has_bps:
            was_hidden = not self._debug_container.isVisible()
            self._debug_container.show()
            self._debug_toolbar.show()
            if was_hidden and self._last_debug_sizes:
                saved = list(self._last_debug_sizes)
            elif was_hidden and len(saved) == 2 and saved[1] < 100:
                saved[1] = 250
        else:
            if self._debug_container.isVisible():
                self._last_debug_sizes = splitter.sizes()
            self._debug_container.hide()
        QTimer.singleShot(0, lambda s=saved: splitter.setSizes(s))

    # ── Modified tracking ─────────────────────────────────────

    def update_title(self):
        name = "Untitled"
        if self._current_path:
            name = os.path.basename(self._current_path)
        suffix = " *" if self._modified else ""
        self.setWindowTitle(f"HMI Designer — {name}{suffix}")

    def _mark_modified(self):
        if self._loading or self._saving:
            return
        self._modified = True
        self.update_title()

    def _confirm_save_if_modified(self) -> bool:
        if not self._modified:
            return True
        btn = QMessageBox.question(
            self, "Unsaved Changes",
            "The HMI project has unsaved changes.\nDo you want to save before closing?",
            QMessageBox.StandardButton.Save
            | QMessageBox.StandardButton.Discard
            | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )
        if btn == QMessageBox.StandardButton.Save:
            self._save_project()
            return not self._modified
        elif btn == QMessageBox.StandardButton.Discard:
            return True
        return False

    def closeEvent(self, event):
        if self._close_project():
            if self._ws_client:
                self._ws_client._running = False
            event.accept()
        else:
            event.ignore()

    # ── Project IO ─────────────────────────────────────────────

    def _close_project(self) -> bool:
        """Close current project, prompting to save if modified. Returns True if closed."""
        if self.project and self._modified:
            name = os.path.basename(self._current_path) if self._current_path else "current project"
            btn = QMessageBox.question(self, "Unsaved Changes",
                f"Save changes to {name}?",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel)
            if btn == QMessageBox.StandardButton.Cancel:
                return False
            if btn == QMessageBox.StandardButton.Save:
                self._save_project()

        for attr in ('_sc_continue', '_sc_step_over', '_sc_step_into', '_sc_step_out', '_sc_f2'):
            sc = getattr(self, attr, None)
            if sc is not None:
                sc.deleteLater()
                setattr(self, attr, None)

        if self._tabs:
            self._tabs.deleteLater()
            self._tabs = None

        if self._editor_tabs:
            while self._editor_tabs.count():
                self._editor_tabs.removeTab(0)
            self._editor_tabs = None
        self._editor_by_path.clear()
        self._tab_path.clear()
        self._file_tree = None
        self.script_edit = None
        self._scene = None
        self._view = None
        self._preview = None
        self._layout_splitter = None
        self._script_splitter = None
        self._debug_container = None
        self._debug_vsplitter = None
        self._debug_toolbar = None
        self._debug_splitter = None
        self._var_inspector = None
        self._watch_panel = None

        self.project = None
        self._current_path = None
        self._modified = False
        self._loading = False

        self.setWindowTitle("HMI Designer")
        return True

    def _new_project(self):
        dlg = _NewProjectDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        if not self._close_project():
            return

        self.project = HMIProject()
        self._current_path = None
        self._modified = False

        if dlg.project_type == "multi":
            self._set_source_dir(dlg.source_dir)
            self.project.entry_point = dlg.entry_point

        self._build_from_project()

    def _open_project(self):
        if not self._close_project():
            return

        path, _ = QFileDialog.getOpenFileName(self, "Open HMI Project", "",
                                              "HMI Files (*.hmi);;All Files (*)")
        if path:
            self._show_loading("Loading project...")
            try:
                self.project = HMIProject.load(path)
                self._current_path = path
                self._modified = False
                self._build_from_project()
            finally:
                self._hide_loading()

    def _save_project(self):
        if not self.project:
            return
        if self._current_path:
            self._flush_to_project()
            self.project.save(self._current_path)
            self._modified = False
            self.update_title()
        else:
            self._save_project_as()

    def _save_project_as(self):
        if not self.project:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save HMI Project As", "",
                                              "HMI Files (*.hmi);;All Files (*)")
        if path:
            self._flush_to_project()
            self.project.save(path)
            self._current_path = path
            self._modified = False
            self.update_title()

    def _full_reload(self):
        self._loading = True
        self.rows_spin.blockSignals(True)
        self.cols_spin.blockSignals(True)
        self.rows_spin.setValue(self.project.layout.get("rows", 56))
        self.cols_spin.setValue(self.project.layout.get("cols", 100))
        self.rows_spin.blockSignals(False)
        self.cols_spin.blockSignals(False)
        if self._editor_by_path:
            if self._tabs and self.project.source_dir:
                self._editor_tabs.blockSignals(True)
                while self._editor_tabs.count() > 0:
                    self._editor_tabs.removeTab(0)
                self._editor_tabs.blockSignals(False)
                self._editor_by_path.clear()
                self._tab_path.clear()
                self._rebuild_file_tree()
                entry_path = os.path.abspath(os.path.join(self.project.source_dir, self.project.entry_point))
                if os.path.isfile(entry_path):
                    self._open_file(entry_path)
            else:
                self._editor_by_path.clear()
                self._tab_path.clear()
        else:
            self.script_edit.setText(self.project.script)
        self._rebuild_slot_list()
        self._rebuild_widget_list()
        self._scene.project = self.project
        srect = QRectF(0, 0, self.project.layout.get("cols", 100) * 24,
                       self.project.layout.get("rows", 56) * 24)
        self._scene.setSceneRect(srect)
        self._scene.rebuild_from_project()
        self._view.fitInView(srect, Qt.AspectRatioMode.KeepAspectRatio)
        self._loading = False
        self.update_title()

    def _flush_to_project(self):
        self._saving = True
        self.project.layout["rows"] = self.rows_spin.value()
        self.project.layout["cols"] = self.cols_spin.value()
        if self._editor_by_path:
            self._write_all_sources()
        else:
            self.project.script = self._tab_to_spaces(self.script_edit.text())
        self._scene.sync_all()
        self._saving = False

    def get_project(self) -> HMIProject:
        self._flush_to_project()
        return self.project

    # ── Slots ──────────────────────────────────────────────────

    def _add_slot(self):
        slot = self.project.add_slot(name=f"Slot {len(self.project.slots) + 1}")
        self._rebuild_slot_list()
        self.slot_list.setCurrentRow(len(self.project.slots) - 1)
        self._mark_modified()

    def _del_slot(self):
        row = self.slot_list.currentRow()
        if row >= 0:
            self.project.remove_slot(row)
            self._rebuild_slot_list()
            self._mark_modified()

    def _rebuild_slot_list(self):
        self.slot_list.clear()
        for s in self.project.slots:
            label = f"{s.name}  [{os.path.basename(s.job_path) if s.job_path else 'no job'}]"
            self.slot_list.addItem(QListWidgetItem(label))

    def _on_slot_selected(self, row):
        self.slot_name_edit.blockSignals(True)
        self.slot_path_edit.blockSignals(True)
        if 0 <= row < len(self.project.slots):
            s = self.project.slots[row]
            self.slot_name_edit.setText(s.name)
            self.slot_path_edit.setText(s.job_path)
        else:
            self.slot_name_edit.clear()
            self.slot_path_edit.clear()
        self.slot_name_edit.blockSignals(False)
        self.slot_path_edit.blockSignals(False)

    def _on_slot_name_changed(self, text):
        row = self.slot_list.currentRow()
        if 0 <= row < len(self.project.slots):
            self.project.slots[row].name = text
            self._rebuild_slot_list()
            self.slot_list.setCurrentRow(row)
            self._mark_modified()

    def _on_slot_path_changed(self, text):
        row = self.slot_list.currentRow()
        if 0 <= row < len(self.project.slots):
            self.project.slots[row].job_path = text
            self._rebuild_slot_list()
            self.slot_list.setCurrentRow(row)
            self._mark_modified()

    def _browse_job(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Job", "",
                                              "FalcoCV Jobs (*.cvj);;All Files (*)")
        if path:
            self.slot_path_edit.setText(path)
            row = self.slot_list.currentRow()
            if 0 <= row < len(self.project.slots):
                self.project.slots[row].job_path = path
                self._rebuild_slot_list()
                self.slot_list.setCurrentRow(row)
                self._mark_modified()

    # ── Widgets ────────────────────────────────────────────────

    def _add_widget_from_list(self):
        dlg = _AddWidgetDialog(WIDGET_TYPES, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            wt = dlg.selected_type
            default_name = self.project._default_widget_name(wt)
            name, ok = QInputDialog.getText(self, "Add Widget", f"Name for {wt}:",
                                            text=default_name)
            if not ok or not name.strip():
                return
            name = name.strip()
            if self.project._widget_name_exists(name):
                QMessageBox.warning(self, "Duplicate Name", f"Widget name '{name}' already exists.")
                return
            self._add_widget_to_scene(wt, name)

    def _add_widget_to_scene(self, widget_type: str, name: str | None = None):
        data = self.project.add_widget(widget_type, x=2, y=2, name=name)
        self._scene.add_widget(data)
        self._rebuild_widget_list()
        self._view.viewport().update()
        self._mark_modified()

    def _del_widget(self):
        sel = self._scene.selected_widget_id()
        if sel:
            self.project.remove_widget(sel)
            self._scene.remove_widget(sel)
            self._rebuild_widget_list()
            self._mark_modified()

    def _rebuild_widget_list(self):
        self.widget_list.clear()
        for w in self.project.layout.get("widgets", []):
            label = f"{w.get('name', w['type'])}  ({w['type']})"
            self.widget_list.addItem(QListWidgetItem(label))
        if self._preview:
            self._preview.update()

    def _on_widget_selected(self, row):
        widgets = self.project.layout.get("widgets", [])
        if 0 <= row < len(widgets):
            wid = widgets[row]["id"]
            # select in scene
            self._scene.clearSelection()
            item = self._scene._items.get(wid)
            if item:
                item.setSelected(True)

    def _on_widget_double_clicked(self, widget_id: str):
        for w in self.project.layout.get("widgets", []):
            if w["id"] == widget_id:
                self._open_properties_popup(w)
                return

    def _open_properties_popup(self, data: dict):
        wt = data.get("type", "ImageViewer")
        cls = POPUPS.get(wt)
        if cls is None:
            return
        dlg = cls(data, self.project, self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            self._rebuild_widget_list()
            self._scene.update()
            if self._preview:
                self._preview.update()
            self._mark_modified()

    def _on_f2_rename(self):
        wid = self._scene.selected_widget_id() if self._scene else None
        if not wid:
            row = self.widget_list.currentRow()
            if row < 0:
                return
            widgets = self.project.layout.get("widgets", []) if self.project else []
            if row < len(widgets):
                wid = widgets[row]["id"]
        if wid:
            self._on_widget_rename(wid)

    def _on_widget_list_context_menu(self, pos):
        item = self.widget_list.itemAt(pos)
        if not item:
            return
        row = self.widget_list.row(item)
        widgets = self.project.layout.get("widgets", [])
        if row < 0 or row >= len(widgets):
            return
        wid = widgets[row]["id"]
        menu = QMenu(self)
        rename_action = menu.addAction("Rename")
        action = menu.exec(self.widget_list.mapToGlobal(pos))
        if action == rename_action:
            self._on_widget_rename(wid)

    def _on_widget_rename(self, widget_id: str):
        for w in self.project.layout.get("widgets", []):
            if w["id"] == widget_id:
                old_name = w.get("name", "")
                new_name, ok = QInputDialog.getText(self, "Rename Widget", "New name:",
                                                    text=old_name)
                if not ok or not new_name.strip():
                    return
                new_name = new_name.strip()
                if new_name == old_name:
                    return
                try:
                    self.project.rename_widget(widget_id, new_name)
                except ValueError as e:
                    QMessageBox.warning(self, "Error", str(e))
                    return
                self._rebuild_widget_list()
                self._scene.update()
                if self._preview:
                    self._preview.update()
                self._mark_modified()
                return

    # ── Grid ───────────────────────────────────────────────────

    def _on_grid_size_changed(self):
        rows = self.rows_spin.value()
        cols = self.cols_spin.value()
        self.project.layout["rows"] = rows
        self.project.layout["cols"] = cols
        self._scene.setSceneRect(0, 0, cols * GRID, rows * GRID)
        self._view.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._scene.update()
        self._mark_modified()

    def _on_show_grid_toggled(self, checked):
        self._scene._show_grid = checked
        self._scene.update()

    # ── Multi-file support ──────────────────────────────────────

    def _set_source_dir(self, path: str):
        self.project.source_dir = path
        self.project.entry_point = "main.py"
        entry = os.path.join(path, "main.py")
        if not os.path.isfile(entry):
            with open(entry, "w", encoding="utf-8") as f:
                f.write(
                    "# HMI Script — entry point\n"
                    "# Available: run_slot(name), log(msg), set_slot_status(name, status), sleep(s)\n\n"
                    "while True:\n"
                    "    if _stop.is_set():\n"
                    "        break\n"
                    "    sleep(1)\n"
                )

    def _rebuild_script_tab(self):
        if self._tabs.count() > 1:
            self._tabs.removeTab(1)
        script_tab = self._build_script_tab()
        self._tabs.addTab(script_tab, "Script")

    def _convert_to_multi(self):
        path = QFileDialog.getExistingDirectory(self, "Select source directory")
        if not path:
            return
        self._set_source_dir(path)
        entry = os.path.join(path, "main.py")
        if not os.path.isfile(entry):
            with open(entry, "w", encoding="utf-8") as f:
                f.write(self.script_edit.text())
        self._mark_modified()
        self._rebuild_script_tab()

    def _tab_to_spaces(self, text: str) -> str:
        return text.replace("\t", "    ")

    def _get_active_editor(self) -> ScriptEditor | None:
        if self._editor_tabs and self._editor_tabs.count() > 0:
            w = self._editor_tabs.currentWidget()
            if hasattr(w, 'script_edit'):
                return w.script_edit
        return self.script_edit

    def _all_breakpoints(self) -> dict[str, set[int]]:
        result = {}
        if self._editor_by_path and os.path.isdir(self.project.source_dir or ""):
            for abspath, editor in self._editor_by_path.items():
                bps = editor.get_breakpoints()
                if bps:
                    rel = os.path.relpath(abspath, self.project.source_dir).replace("\\", "/")
                    result[rel] = bps
        else:
            ed = getattr(self, 'script_edit', None)
            bps = ed.get_breakpoints() if ed else set()
            if bps:
                result["<hmi_script>"] = bps
        return result

    def _write_all_sources(self):
        if not self._editor_by_path:
            return
        for abspath, editor in self._editor_by_path.items():
            content = self._tab_to_spaces(editor.text())
            with open(abspath, "w", encoding="utf-8") as f:
                f.write(content)

    def _open_file(self, fullpath: str):
        fullpath = os.path.abspath(fullpath)
        if fullpath in self._editor_by_path:
            for i in range(self._editor_tabs.count()):
                if self._tab_path[i] == fullpath:
                    self._editor_tabs.setCurrentIndex(i)
                    return

        editor = ScriptEditor()
        try:
            with open(fullpath, "r", encoding="utf-8") as f:
                editor.setText(f.read())
        except Exception:
            pass

        wrapped = QWidget()
        wl = QVBoxLayout(wrapped)
        wl.setContentsMargins(0, 0, 0, 0)
        wl.addWidget(editor, 1)
        wrapped.script_edit = editor

        name = os.path.basename(fullpath)
        idx = self._editor_tabs.addTab(wrapped, name)
        self._editor_tabs.setCurrentIndex(idx)

        editor.textChanged.connect(self._mark_modified)
        editor.breakpoints_changed.connect(self._update_debug_visibility)
        editor.breakpoints_changed.connect(self._on_breakpoints_changed)

        self._editor_by_path[fullpath] = editor
        self._tab_path.insert(idx, fullpath)

        self._rebind_find_bar()

    def _on_tab_closed(self, index: int):
        if index < 0 or index >= len(self._tab_path):
            return
        abspath = self._tab_path.pop(index)
        editor = self._editor_by_path.pop(abspath, None)
        if editor:
            content = self._tab_to_spaces(editor.text())
            try:
                with open(abspath, "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception:
                pass
            editor.breakpoints_changed.disconnect(self._update_debug_visibility)
            editor.breakpoints_changed.disconnect(self._on_breakpoints_changed)
            editor.textChanged.disconnect(self._mark_modified)
        w = self._editor_tabs.widget(index)
        self._editor_tabs.removeTab(index)

    def _rebind_find_bar(self):
        ed = self._get_active_editor()
        if ed and ed != self._find_bar._editor:
            self._find_bar._editor = ed
            ed.set_find_bar(self._find_bar)

    def _rebuild_file_tree(self):
        if not self._file_tree or not self.project or not self.project.source_dir:
            return
        self._file_tree.clear()
        self._add_dir_to_tree(None, self.project.source_dir)

    def _add_dir_to_tree(self, parent: QTreeWidgetItem | None, path: str):
        try:
            entries = sorted(os.listdir(path),
                key=lambda n: (not os.path.isdir(os.path.join(path, n)), n.lower()))
        except Exception:
            return
        fp = QFileIconProvider()
        for name in entries:
            if name.startswith(".") or name == "__pycache__":
                continue
            full = os.path.join(path, name)
            if os.path.isdir(full):
                sub = QTreeWidgetItem([name])
                sub.setIcon(0, fp.icon(QFileIconProvider.IconType.Folder))
                sub.setData(0, Qt.ItemDataRole.UserRole, os.path.abspath(full))
                if parent is None:
                    self._file_tree.addTopLevelItem(sub)
                else:
                    parent.addChild(sub)
                self._add_dir_to_tree(sub, full)
            elif name.endswith(".py"):
                item = QTreeWidgetItem([name])
                item.setIcon(0, fp.icon(QFileInfo(full)))
                if parent is None:
                    self._file_tree.addTopLevelItem(item)
                else:
                    parent.addChild(item)
                item.setData(0, Qt.ItemDataRole.UserRole, os.path.abspath(full))

    def _on_tree_clicked(self, item: QTreeWidgetItem):
        fp = item.data(0, Qt.ItemDataRole.UserRole)
        if fp and os.path.isfile(fp):
            self._open_file(fp)

    def _on_tree_double_clicked(self, item: QTreeWidgetItem):
        fp = item.data(0, Qt.ItemDataRole.UserRole)
        if not fp or not os.path.isfile(fp):
            return
        name, ok = QInputDialog.getText(
            self, "Rename", "New name:", text=os.path.basename(fp))
        if ok and name and name != os.path.basename(fp):
            new_path = os.path.join(os.path.dirname(fp), name)
            if os.path.exists(new_path):
                QMessageBox.warning(self, "Error", f"File already exists: {name}")
                return
            os.rename(fp, new_path)
            self._editor_by_path.pop(fp, None)
            for i, p in enumerate(self._tab_path):
                if p == fp:
                    self._tab_path[i] = new_path
                    self._editor_tabs.setTabText(i, name)
                    break
            if fp in self._editor_by_path:
                ed = self._editor_by_path.pop(fp)
                self._editor_by_path[new_path] = ed
        QTimer.singleShot(0, self._rebuild_file_tree)
        self._mark_modified()

    def _add_new_file(self):
        name, ok = QInputDialog.getText(
            self, "New File", "File name:", text="new_module.py")
        if not ok or not name:
            return
        if not name.endswith(".py"):
            name += ".py"
        fp = os.path.join(self.project.source_dir, name)
        if os.path.exists(fp):
            QMessageBox.warning(self, "Error", f"File already exists: {name}")
            return
        try:
            with open(fp, "w", encoding="utf-8") as f:
                f.write("# " + name + "\n")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create file: {e}")
            return
        self._rebuild_file_tree()
        self._open_file(fp)
        self._mark_modified()

    def _delete_file(self, path: str = None):
        targets = []
        if path:
            targets.append(path)
        elif self._editor_tabs and self._editor_tabs.count() > 0:
            idx = self._editor_tabs.currentIndex()
            if idx < len(self._tab_path):
                targets.append(self._tab_path[idx])
        if not targets:
            return
        for fp in targets:
            if not os.path.isfile(fp):
                continue
            if fp.endswith(os.sep + self.project.entry_point):
                QMessageBox.warning(self, "Error",
                    f"Cannot delete entry point: {self.project.entry_point}")
                continue
            btn = QMessageBox.question(self, "Delete",
                f"Delete {os.path.basename(fp)}?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if btn != QMessageBox.StandardButton.Yes:
                continue
            os.remove(fp)
            self._editor_by_path.pop(fp, None)
            for i, p in enumerate(self._tab_path):
                if p == fp:
                    self._on_tab_closed(i)
                    break
        self._rebuild_file_tree()
        self._mark_modified()

    def _move_file(self, old_path: str, new_path: str):
        if os.path.exists(new_path):
            QMessageBox.warning(self, "Error", f"Target exists: {os.path.basename(new_path)}")
            return
        if not os.path.isfile(old_path):
            return
        os.rename(old_path, new_path)
        ed = self._editor_by_path.pop(old_path, None)
        if ed:
            self._editor_by_path[new_path] = ed
            for i, p in enumerate(self._tab_path):
                if p == old_path:
                    self._tab_path[i] = new_path
                    self._editor_tabs.setTabText(i, os.path.basename(new_path))
                    break
        self._rebuild_file_tree()
        self._mark_modified()

    def _on_tree_context_menu(self, pos):
        item = self._file_tree.itemAt(pos)
        menu = self._build_context_menu(item)
        if menu:
            menu.exec(self._file_tree.viewport().mapToGlobal(pos))

    def _build_context_menu(self, item: QTreeWidgetItem | None) -> QMenu | None:
        menu = QMenu(self)
        if item is None:
            menu.addAction("New File...", lambda: self._new_file_in(self.project.source_dir))
            menu.addAction("New Folder...", lambda: self._new_folder_in(self.project.source_dir))
            return menu
        fp = item.data(0, Qt.ItemDataRole.UserRole)
        if fp and os.path.isfile(fp):
            menu.addAction("Open", lambda: self._open_file(fp))
            menu.addSeparator()
            menu.addAction("Rename...", lambda: self._on_tree_double_clicked(item))
            menu.addSeparator()
            menu.addAction("Delete", lambda: self._delete_file(fp))
            return menu
        dir_path = None
        if fp and os.path.isdir(fp):
            dir_path = fp
        else:
            dir_path = self._resolve_tree_dir(item)
        if dir_path:
            menu.addAction("New File...", lambda: self._new_file_in(dir_path))
            menu.addAction("New Folder...", lambda: self._new_folder_in(dir_path))
            menu.addSeparator()
            fp_item = item.data(0, Qt.ItemDataRole.UserRole)
            if fp_item and os.path.isdir(fp_item):
                menu.addAction("Rename...", lambda: self._rename_folder(fp_item))
                menu.addAction("Delete Folder", lambda: self._delete_folder(fp_item))
            return menu
        return None

    def _resolve_tree_dir(self, item: QTreeWidgetItem) -> str:
        parts = []
        it = item
        while it:
            fp = it.data(0, Qt.ItemDataRole.UserRole)
            if fp and os.path.isfile(fp):
                return os.path.dirname(fp)
            parts.append(it.text(0))
            it = it.parent()
        parts.reverse()
        return os.path.join(self.project.source_dir, *parts)

    def _new_file_in(self, dir_path: str):
        name, ok = QInputDialog.getText(
            self, "New File", "File name:", text="new_module.py")
        if not ok or not name:
            return
        if not name.endswith(".py"):
            name += ".py"
        fp = os.path.join(dir_path, name)
        if os.path.exists(fp):
            QMessageBox.warning(self, "Error", f"File already exists: {name}")
            return
        try:
            os.makedirs(dir_path, exist_ok=True)
            with open(fp, "w", encoding="utf-8") as f:
                f.write("# " + name + "\n")
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create file: {e}")
            return
        self._rebuild_file_tree()
        self._open_file(fp)
        self._mark_modified()

    def _new_folder_in(self, dir_path: str):
        name, ok = QInputDialog.getText(
            self, "New Folder", "Folder name:", text="new_folder")
        if not ok or not name:
            return
        fp = os.path.join(dir_path, name)
        if os.path.exists(fp):
            QMessageBox.warning(self, "Error", f"Already exists: {name}")
            return
        try:
            os.makedirs(fp, exist_ok=True)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to create folder: {e}")
            return
        self._rebuild_file_tree()
        self._mark_modified()

    def _delete_folder(self, path: str):
        if not os.path.isdir(path):
            return
        btn = QMessageBox.question(self, "Delete Folder",
            f"Delete '{os.path.basename(path)}' and all contents?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
        if btn != QMessageBox.StandardButton.Yes:
            return
        closed = set()
        for abspath in list(self._editor_by_path.keys()):
            if abspath.startswith(path + os.sep):
                for i, p in enumerate(self._tab_path):
                    if p == abspath:
                        closed.add(i)
                        break
        for i in sorted(closed, reverse=True):
            self._on_tab_closed(i)
        shutil.rmtree(path)
        self._rebuild_file_tree()
        self._mark_modified()

    def _rename_folder(self, path: str):
        if not os.path.isdir(path):
            return
        parent = os.path.dirname(path)
        old_name = os.path.basename(path)
        name, ok = QInputDialog.getText(
            self, "Rename Folder", "New name:", text=old_name)
        if not ok or not name or name == old_name:
            return
        new_path = os.path.join(parent, name)
        if os.path.exists(new_path):
            QMessageBox.warning(self, "Error", f"Already exists: {name}")
            return
        for abspath in list(self._editor_by_path.keys()):
            if abspath.startswith(path + os.sep):
                new_abspath = new_path + abspath[len(path):]
                editor = self._editor_by_path.pop(abspath)
                self._editor_by_path[new_abspath] = editor
                for i, p in enumerate(self._tab_path):
                    if p == abspath:
                        self._tab_path[i] = new_abspath
                        break
        os.rename(path, new_path)
        self._rebuild_file_tree()
        self._mark_modified()

    def _build_script_tab_multi(self):
        script_tab = QWidget()
        sl = QVBoxLayout(script_tab)
        sl.setContentsMargins(0, 0, 0, 0)

        self._script_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._script_splitter.setHandleWidth(6)

        file_panel = QWidget()
        fpl = QVBoxLayout(file_panel)
        fpl.setContentsMargins(0, 0, 0, 0)
        self._file_tree = QTreeWidget()
        self._file_tree.setHeaderHidden(True)
        self._file_tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self._file_tree.customContextMenuRequested.connect(self._on_tree_context_menu)
        self._file_tree.itemClicked.connect(self._on_tree_clicked)
        self._file_tree.itemDoubleClicked.connect(self._on_tree_double_clicked)
        self._file_tree.setDragEnabled(True)
        self._file_tree.setAcceptDrops(True)
        self._file_tree.setDropIndicatorShown(True)
        self._file_tree.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)

        def tree_drop(e):
            src_item = self._file_tree.currentItem()
            src_fp = src_item.data(0, Qt.ItemDataRole.UserRole) if src_item else None
            src_path = os.path.abspath(src_fp) if src_fp else None
            if not src_path:
                e.ignore()
                return
            tgt_item = self._file_tree.itemAt(e.position().toPoint())
            if tgt_item:
                tfp = tgt_item.data(0, Qt.ItemDataRole.UserRole)
                if tfp and os.path.isdir(tfp):
                    new_path = os.path.join(tfp, os.path.basename(src_path))
                elif tfp and os.path.isfile(tfp):
                    new_path = os.path.join(os.path.dirname(tfp), os.path.basename(src_path))
                else:
                    td = self._resolve_tree_dir(tgt_item)
                    new_path = os.path.join(td, os.path.basename(src_path))
            else:
                new_path = os.path.join(self.project.source_dir, os.path.basename(src_path))
            if new_path and new_path != src_path:
                if os.path.isdir(src_path):
                    if not new_path.startswith(src_path + os.sep):
                        shutil.move(src_path, new_path)
                    else:
                        e.ignore()
                        return
                else:
                    self._move_file(src_path, new_path)
                QTimer.singleShot(0, self._rebuild_file_tree)
            e.accept()

        self._file_tree.dropEvent = tree_drop
        fpl.addWidget(self._file_tree, 1)
        self._script_splitter.addWidget(file_panel)

        editor_panel = QWidget()
        epl = QVBoxLayout(editor_panel)
        epl.setContentsMargins(0, 0, 0, 0)

        self._debug_vsplitter = QSplitter(Qt.Orientation.Vertical)
        self._debug_vsplitter.setHandleWidth(6)

        editor_area = QWidget()
        eal = QVBoxLayout(editor_area)
        eal.setContentsMargins(0, 0, 0, 0)
        self._editor_tabs = QTabWidget()
        self._editor_tabs.setTabsClosable(True)
        self._editor_tabs.tabCloseRequested.connect(self._on_tab_closed)
        self._editor_tabs.currentChanged.connect(lambda i: self._rebind_find_bar())
        eal.addWidget(self._editor_tabs, 3)

        self._find_bar = FindReplaceBar(None)
        eal.addWidget(self._find_bar)

        self._debug_vsplitter.addWidget(editor_area)

        self._debug_container = QWidget()
        dc_lyt = QVBoxLayout(self._debug_container)
        dc_lyt.setContentsMargins(0, 0, 0, 0)
        dc_lyt.setSpacing(0)

        self._debug_toolbar = DebugToolbar()
        self._debug_toolbar.continue_requested.connect(self._on_debug_continue)
        self._debug_toolbar.step_over_requested.connect(self._on_debug_step_over)
        self._debug_toolbar.step_into_requested.connect(self._on_debug_step_into)
        self._debug_toolbar.step_out_requested.connect(self._on_debug_step_out)
        self._debug_toolbar.stop_requested.connect(self._on_debug_stop)
        dc_lyt.addWidget(self._debug_toolbar)

        self._debug_splitter = QSplitter(Qt.Orientation.Horizontal)
        self._debug_splitter.setHandleWidth(6)
        self._var_inspector = VariableInspector()
        self._var_inspector.show()
        self._debug_splitter.addWidget(self._var_inspector)
        self._watch_panel = WatchPanel()
        self._watch_panel.show()
        self._debug_splitter.addWidget(self._watch_panel)
        self._debug_splitter.setSizes([350, 350])
        dc_lyt.addWidget(self._debug_splitter, 1)

        self._debug_vsplitter.addWidget(self._debug_container)
        self._debug_vsplitter.setSizes([500, 300])
        self._debug_vsplitter.splitterMoved.connect(self._on_debug_splitter_moved)

        epl.addWidget(self._debug_vsplitter, 1)

        self._script_splitter.addWidget(editor_panel)
        self._script_splitter.setSizes([180, 700])
        sl.addWidget(self._script_splitter, 1)

        sc_find = QShortcut(QKeySequence("Ctrl+F"), self)
        sc_find.activated.connect(self._on_script_find)
        sc_find_replace = QShortcut(QKeySequence("Ctrl+H"), self)
        sc_find_replace.activated.connect(self._on_script_find_replace)

        sc = QShortcut(QKeySequence("F5"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_debug_continue)
        self._sc_continue = sc
        sc = QShortcut(QKeySequence("F10"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_debug_step_over)
        self._sc_step_over = sc
        sc = QShortcut(QKeySequence("F11"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_debug_step_into)
        self._sc_step_into = sc
        sc = QShortcut(QKeySequence("Shift+F11"), self)
        sc.setContext(Qt.ShortcutContext.WindowShortcut)
        sc.activated.connect(self._on_debug_step_out)
        self._sc_step_out = sc

        QTimer.singleShot(0, self._update_debug_visibility)

        self._rebuild_file_tree()
        entry_path = os.path.abspath(os.path.join(self.project.source_dir, self.project.entry_point))
        if os.path.isfile(entry_path):
            self._open_file(entry_path)

        sl.addStretch()
        return script_tab


class _AddWidgetDialog(QDialog):
    def __init__(self, types: list[str], parent=None):
        super().__init__(parent)
        self.selected_type = None
        self.setWindowTitle("Add Widget")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select widget type:"))
        self._combo = QComboBox()
        self._combo.addItems(types)
        layout.addWidget(self._combo)
        btns = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._on_ok)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _on_ok(self):
        self.selected_type = self._combo.currentText()
        self.accept()


class _NewProjectDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New HMI Project")
        self.setMinimumWidth(480)
        self.project_type = "single"
        self.source_dir = ""
        self.entry_point = "main.py"

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Project type:"))
        self._rb_single = QRadioButton("Single embedded script")
        self._rb_multi = QRadioButton("Directory-based (multi-file)")
        self._rb_single.setChecked(True)
        self.project_type = "single"
        self._rb_single.toggled.connect(self._on_type_changed)
        layout.addWidget(self._rb_single)
        layout.addWidget(self._rb_multi)

        layout.addSpacing(8)

        self._dir_row = QWidget()
        dr = QHBoxLayout(self._dir_row)
        dr.setContentsMargins(0, 0, 0, 0)
        dr.addWidget(QLabel("Source directory:"))
        self._dir_edit = QLineEdit()
        self._dir_edit.setPlaceholderText("Select a directory...")
        dr.addWidget(self._dir_edit, 1)
        btn_browse = QPushButton("Browse")
        btn_browse.clicked.connect(self._on_browse)
        dr.addWidget(btn_browse)
        layout.addWidget(self._dir_row)

        self._ep_row = QWidget()
        epr = QHBoxLayout(self._ep_row)
        epr.setContentsMargins(0, 0, 0, 0)
        epr.addWidget(QLabel("Entry point:"))
        self._ep_edit = QLineEdit("main.py")
        epr.addWidget(self._ep_edit, 1)
        layout.addWidget(self._ep_row)

        layout.addSpacing(12)

        btns = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        btns.accepted.connect(self._on_create)
        btns.rejected.connect(self.reject)
        btns.button(QDialogButtonBox.StandardButton.Ok).setText("Create")
        layout.addWidget(btns)

        self._dir_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self._ep_row.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, False)
        self.setStyleSheet("""
            QLineEdit:disabled { color: #555; border: 1px solid #333; background: #2d2d2d; }
            QPushButton:disabled { color: #444; border: 1px solid #2d2d2d; }
        """)
        self._on_type_changed()

    def _on_type_changed(self):
        single = self._rb_single.isChecked()
        self._dir_row.setEnabled(not single)
        self._ep_row.setEnabled(not single)

    def _on_browse(self):
        path = QFileDialog.getExistingDirectory(self, "Select source directory")
        if path:
            self._dir_edit.setText(path)

    def _on_create(self):
        if self._rb_single.isChecked():
            self.project_type = "single"
            self.source_dir = ""
            self.accept()
            return
        sd = self._dir_edit.text().strip()
        if not sd:
            QMessageBox.warning(self, "Error", "Select a source directory")
            return
        if not os.path.isdir(sd):
            btn = QMessageBox.question(self, "Create Directory",
                f"Directory does not exist:\n{sd}\n\nCreate it?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if btn != QMessageBox.StandardButton.Yes:
                return
            try:
                os.makedirs(sd, exist_ok=True)
            except Exception as e:
                QMessageBox.warning(self, "Error", f"Failed to create directory: {e}")
                return
        self.project_type = "multi"
        self.source_dir = sd
        self.entry_point = self._ep_edit.text().strip() or "main.py"
        self.accept()
