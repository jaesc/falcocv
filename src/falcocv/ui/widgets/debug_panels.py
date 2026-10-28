from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QLineEdit, QTreeWidget, QTreeWidgetItem, QSplitter,
)
from PyQt6.QtCore import Qt, pyqtSignal

from falcocv.ui.style import (
    SCRIPT_BG, SCRIPT_TEXT, SCRIPT_INPUT_BG, SCRIPT_BAR_BG,
    TEXT, TEXT_MUTED, BORDER, PANEL_BG, SUCCESS, WARNING, TOOLTIP_BG,
)


class DebugToolbar(QWidget):
    continue_requested = pyqtSignal()
    step_over_requested = pyqtSignal()
    step_into_requested = pyqtSignal()
    step_out_requested = pyqtSignal()
    stop_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)

        btn_style = "font-size: 13px; padding: 4px 10px;"
        self.btn_continue = QPushButton("\u25b6 Continue")
        self.btn_continue.setToolTip("Continue (F5)")
        self.btn_continue.setStyleSheet(btn_style)
        self.btn_continue.clicked.connect(self.continue_requested.emit)
        layout.addWidget(self.btn_continue)

        self.btn_step_over = QPushButton("\u2b07 Step Over")
        self.btn_step_over.setToolTip("Step Over (F10)")
        self.btn_step_over.setStyleSheet(btn_style)
        self.btn_step_over.clicked.connect(self.step_over_requested.emit)
        layout.addWidget(self.btn_step_over)

        self.btn_step_into = QPushButton("\u2198 Step Into")
        self.btn_step_into.setToolTip("Step Into (F11)")
        self.btn_step_into.setStyleSheet(btn_style)
        self.btn_step_into.clicked.connect(self.step_into_requested.emit)
        layout.addWidget(self.btn_step_into)

        self.btn_step_out = QPushButton("\u2b06 Step Out")
        self.btn_step_out.setToolTip("Step Out (Shift+F11)")
        self.btn_step_out.setStyleSheet(btn_style)
        self.btn_step_out.clicked.connect(self.step_out_requested.emit)
        layout.addWidget(self.btn_step_out)

        self.btn_stop = QPushButton("\u25a0 Stop")
        self.btn_stop.setToolTip("Stop Debugging")
        self.btn_stop.setStyleSheet(btn_style)
        self.btn_stop.clicked.connect(self.stop_requested.emit)
        layout.addWidget(self.btn_stop)

        layout.addStretch()

        self.label_state = QLabel("Running...")
        self.label_state.setStyleSheet(f"color: {SUCCESS}; font-weight: bold; font-size: 13px;")
        layout.addWidget(self.label_state)

        self.setStyleSheet(f"background-color: {SCRIPT_BAR_BG}; border-top: 1px solid {BORDER}; border-bottom: 1px solid {BORDER};")
        self.hide()

    def set_state(self, text, color=SUCCESS):
        self.label_state.setText(text)
        self.label_state.setStyleSheet(f"color: {color}; font-weight: bold;")


from falcocv.core.config import ConfigManager
_MAX_DEPTH = ConfigManager().data.get("debug_max_depth", 10)


class VariableInspector(QTreeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setHeaderLabels(["Variable", "Value", "Type"])
        self.setColumnWidth(0, 160)
        self.setColumnWidth(1, 300)
        self.setAlternatingRowColors(True)
        self.setStyleSheet(f"""
            QTreeWidget {{ background: {SCRIPT_BG}; color: {TEXT}; border: none; font: 12px Consolas; }}
            QTreeWidget::item {{ padding: 1px 4px; }}
            QTreeWidget::item:alternate {{ background: {TOOLTIP_BG}; }}
            QHeaderView::section {{ background: {PANEL_BG}; color: {TEXT_MUTED}; border: none; padding: 2px; font-weight: bold; }}
        """)
        self.hide()

    def update_variables(self, locals_dict):
        self.clear()
        if not locals_dict:
            return
        for name in sorted(locals_dict.keys()):
            val = locals_dict[name]
            _TreeItem(self, name, val, depth=0, max_depth=_MAX_DEPTH)

    def clear_variables(self):
        self.clear()


class WatchPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self._watches = []
        self._last_locals = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        input_row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText("Watch expression...")
        self.input.setStyleSheet(f"background: {SCRIPT_INPUT_BG}; color: {TEXT}; border: 1px solid {BORDER}; padding: 2px 4px;")
        self.input.returnPressed.connect(self._add_watch)
        self.btn_add = QPushButton("+")
        self.btn_add.setFixedWidth(24)
        self.btn_add.clicked.connect(self._add_watch)
        input_row.addWidget(self.input, 1)
        input_row.addWidget(self.btn_add)

        label_bar = QLabel("Watch")
        label_bar.setStyleSheet(f"color: {TEXT_MUTED}; font-size: 11px; font-weight: bold; padding: 1px 0;")
        layout.addWidget(label_bar)
        layout.addLayout(input_row)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Expression", "Value", "Type"])
        self.tree.setColumnWidth(0, 120)
        self.tree.setColumnWidth(1, 200)
        self.tree.setAlternatingRowColors(True)
        self.tree.setStyleSheet(f"""
            QTreeWidget {{ background: {SCRIPT_BG}; color: {TEXT}; border: none; font: 12px Consolas; }}
            QTreeWidget::item {{ padding: 1px 4px; }}
            QTreeWidget::item:alternate {{ background: {TOOLTIP_BG}; }}
            QHeaderView::section {{ background: {PANEL_BG}; color: {TEXT_MUTED}; border: none; padding: 2px; font-weight: bold; }}
        """)
        layout.addWidget(self.tree, 1)

        self.btn_remove = QPushButton("Remove Selected")
        self.btn_remove.setStyleSheet("font-size: 11px; padding: 2px 6px;")
        self.btn_remove.clicked.connect(self._remove_selected)
        layout.addWidget(self.btn_remove)

        self.hide()

    def _add_watch(self):
        expr = self.input.text().strip()
        if not expr:
            return
        for e, _ in self._watches:
            if e == expr:
                self.input.clear()
                return
        item = QTreeWidgetItem([expr, "<evaluating...>", ""])
        item._watch_expr = expr
        self.tree.addTopLevelItem(item)
        self._watches.append((expr, item))
        self.input.clear()
        if self._last_locals:
            self._evaluate_item(item, expr)

    def _remove_selected(self):
        items = self.tree.selectedItems()
        for item in items:
            if hasattr(item, '_watch_expr'):
                expr = item._watch_expr
                self._watches[:] = [(e, i) for e, i in self._watches if e != expr]
            root = item.parent() or self.tree.invisibleRootItem()
            root.removeChild(item)

    def update_values(self, locals_dict):
        self._last_locals = locals_dict
        for expr, item in self._watches:
            self._evaluate_item(item, expr)

    def _evaluate_item(self, item, expr):
        locals_dict = self._last_locals
        if not locals_dict:
            item.setText(1, "<no locals>")
            item.setText(2, "")
            return
        try:
            result = eval(expr, {"__builtins__": __builtins__}, locals_dict)
            item.setText(1, repr(result))
            item.setText(2, type(result).__name__)
            _populate_item_children(item, result, depth=0, max_depth=_MAX_DEPTH)
        except Exception as e:
            item.setText(1, f"<{e}>")
            item.setText(2, "error")
            while item.childCount():
                item.removeChild(item.child(0))

    def clear_values(self):
        self.tree.clear()
        self._watches.clear()
        self._last_locals = {}


def _populate_item_children(parent_item, value, depth, max_depth):
    if depth >= max_depth:
        QTreeWidgetItem(parent_item, ["...", "(max depth)", ""])
        return

    if depth == 0:
        _populate_item_children._seen = set()
    val_id = id(value)
    if not hasattr(_populate_item_children, '_seen'):
        _populate_item_children._seen = set()
    _seen = _populate_item_children._seen
    if val_id in _seen:
        QTreeWidgetItem(parent_item, ["<circular reference>", "", ""])
        return
    _seen.add(val_id)

    try:
        if isinstance(value, dict):
            keys = sorted(value.keys(), key=str)
            for k in keys:
                v = value[k]
                _TreeItem(parent_item, repr(k), v, depth + 1, max_depth)
        elif isinstance(value, (list, tuple)):
            for i, v in enumerate(value):
                _TreeItem(parent_item, f"[{i}]", v, depth + 1, max_depth)
        elif isinstance(value, (set, frozenset)):
            items = sorted(value, key=str)
            for i, v in enumerate(items):
                _TreeItem(parent_item, f"[{i}]", v, depth + 1, max_depth)
        elif hasattr(value, '__dict__') and not _is_simple_type(value):
            d = value.__dict__
            keys = sorted(d.keys())
            for k in keys:
                try:
                    v = d[k]
                except Exception:
                    v = f"<error: {k}>"
                _TreeItem(parent_item, f".{k}", v, depth + 1, max_depth)
    except Exception:
        pass
    finally:
        _seen.discard(val_id)


def _is_simple_type(value):
    return isinstance(value, (type, int, float, bool, str, bytes, complex, type(None)))


class _TreeItem(QTreeWidgetItem):
    def __init__(self, parent, name, value, depth, max_depth):
        super().__init__()
        self.setText(0, str(name))
        try:
            self.setText(1, repr(value))
        except Exception:
            self.setText(1, f"<{type(value).__name__}>")
        self.setText(2, type(value).__name__)

        if isinstance(parent, QTreeWidget):
            parent.addTopLevelItem(self)
        else:
            parent.addChild(self)

        if depth < max_depth and _has_expandable_children(value):
            _populate_item_children(self, value, depth, max_depth)


def _has_expandable_children(value):
    if value is None or _is_simple_type(value):
        return False
    if isinstance(value, (dict, list, tuple, set, frozenset)):
        return len(value) > 0
    if hasattr(value, '__dict__'):
        d = value.__dict__
        return any(not k.startswith('__') for k in d.keys())
    try:
        if hasattr(value, '__len__') and len(value) > 0:
            return True
    except (TypeError, AttributeError):
        pass
    if hasattr(value, '__slots__'):
        return True
    return False
