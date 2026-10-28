"""
Tool Help Dialog — displays structured documentation for all FalcoCV tools.
Uses QTextBrowser with Markdown + embedded HTML <font color> for highlighting.
"""

import re

from PyQt6.QtWidgets import (QDialog, QHBoxLayout, QVBoxLayout, QListWidget,
                              QListWidgetItem, QPushButton, QWidget, QLabel,
                              QTextBrowser)
from PyQt6.QtCore import Qt, QSize, QTimer
from PyQt6.QtGui import QFont, QColor

from falcocv.ui.style import (DARK_BG, ALT_BG, TEXT, TEXT_MUTED, BORDER, HIGHLIGHT,
                      PANEL_BG, CAUTION, INFO,
                      SCRIPT_BG, SCRIPT_TEXT, SCRIPT_KEYWORD,
                      SCRIPT_NUMBER, SCRIPT_COMMENT)
from falcocv.ui.tool_help import TOOL_HELP, TYPE_TO_HELP_KEY


# ── Category ordering ──────────────────────────────────────────────────────

_CATEGORIES = ["Acquisition", "Vision Tools", "Logic", "HMI"]
_CATEGORY_ORDER = {c: i for i, c in enumerate(_CATEGORIES)}


def _sort_key(item):
    entry = item.get("entry", {})
    cat = entry.get("category", "Other")
    return _CATEGORY_ORDER.get(cat, 99), entry.get("name", "")


# ── Syntax highlighting (generates <font color="..."> spans) ───────────────

_CODE_TOKEN_RE = re.compile(
    r'(#.*?)$|'
    r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')|'
    r'\b(def|class|if|elif|else|for|while|return|import|from|as|try|except|finally|with|pass|break|continue|and|or|not|in|is|None|True|False|lambda|yield|raise|async|await)\b|'
    r'\b(print|len|str|int|float|list|dict|tuple|set|bool|range|type|isinstance|hasattr|getattr|setattr|repr|super|open|sorted|min|max|sum|any|all|enumerate|zip|map|filter|reversed|abs|round|hex|bin|ord|chr|format|locals|globals|dir|vars|id|callable|staticmethod|classmethod|property|object|Exception|ValueError|TypeError|KeyError|IndexError|RuntimeError|StopIteration|NotImplementedError|AttributeError|MemoryError|OSError|SystemError|SyntaxError|IndentationError|NameError|ImportError|ModuleNotFoundError|ZeroDivisionError|BaseException)\b|'
    r'\b(\d+\.?\d*)\b|'
    r'(\w+)(?=\()|'
    r'\b(self|cls)\b|'
    r'(@\w+)|'
    r'(\.)|'
    r'\b([a-z_]\w*)\b',
    re.MULTILINE
)


class SyntaxTheme:
    """Central color scheme for code highlighting."""
    comment   = "#565f89"
    string    = "#e0af68"
    keyword   = "#c0aaf6"
    builtin   = "#f7768e"
    number    = "#ff9e64"
    func_call = "#7dcfff"
    self_cls  = "#e0af68"
    decorator = "#7dcfff"
    dot       = "#3b4261"
    variable  = "#9ece6a"


_ST = SyntaxTheme()


def highlight_python(text: str) -> str:
    """Highlight Python source code. Returns HTML with <font color> tags."""
    def _tag(m):
        if m.group(1):  return f'<font color="{_ST.comment}">{m.group(1)}</font>'
        if m.group(2):  return f'<font color="{_ST.string}">{m.group(2)}</font>'
        if m.group(3):  return f'<font color="{_ST.keyword}">{m.group(3)}</font>'
        if m.group(4):  return f'<font color="{_ST.builtin}">{m.group(4)}</font>'
        if m.group(5):  return f'<font color="{_ST.number}">{m.group(5)}</font>'
        if m.group(6):  return f'<font color="{_ST.func_call}">{m.group(6)}</font>'
        if m.group(7):  return f'<font color="{_ST.self_cls}">{m.group(7)}</font>'
        if m.group(8):  return f'<font color="{_ST.decorator}">{m.group(8)}</font>'
        if m.group(9):  return f'<font color="{_ST.dot}">{m.group(9)}</font>'
        if m.group(10): return f'<font color="{_ST.variable}">{m.group(10)}</font>'
        return m.group(0)
    return _CODE_TOKEN_RE.sub(_tag, text)


# ── HTML generation (we use setHtml, not setMarkdown) ──────────────────────

def _render_html(entry: dict) -> str:
    """Build an HTML document with <font color> syntax highlighting."""
    if not entry:
        return "<html><body>Select a tool from the list.</body></html>"

    name = entry.get("name", "")
    purpose = entry.get("purpose", "")
    parts = [f'<h2>{name}</h2>']

    if purpose:
        parts.append(
            f'<table bgcolor="{ALT_BG}" width="100%" cellpadding="0" cellspacing="0"'
            f' style="border-left:4px solid #7aa2f7;margin:12px 0 20px 0;">'
            f'<tr><td style="padding:14px 18px;font-size:14px;line-height:1.6;">'
            f'{purpose}</td></tr></table>'
        )

    # Inline highlighted code helper
    def _code(val: str) -> str:
        return f'<code>{highlight_python(val)}</code>'

    def _muted(val: str) -> str:
        return f'<font color="{TEXT_MUTED}">{val}</font>'

    def _section(title: str) -> str:
        return f'<h3>\u25b8 {title}</h3>'

    def _table(headers: list[str], rows: list[str]) -> str:
        h = "".join(f"<th>{h}</th>" for h in headers)
        r = "".join(f"<tr>{row}</tr>" for row in rows)
        return f'<table><tr>{h}</tr>{r}</table>'

    def _code_block(code: str) -> str:
        highlighted = highlight_python(code)
        return (
            f'<table bgcolor="{SCRIPT_BG}" border="1" cellpadding="0" cellspacing="0"'
            f' width="100%"'
            f' style="border:1px solid {BORDER};border-collapse:collapse;margin:12px 0;">'
            f'<tr><td style="padding:14px 18px;font-family:Consolas,monospace;'
            f'font-size:13px;line-height:1.5;white-space:pre;color:{SCRIPT_TEXT};">'
            f'{highlighted}</td></tr></table>'
        )

    # Inputs
    inputs = entry.get("inputs", [])
    parts.append(_section("Inputs"))
    if inputs:
        rows = []
        for i in inputs:
            rows.append(
                f'<td>{_code(i["name"])}</td>'
                f'<td>{_muted(i["type"])}</td>'
                f'<td>{i["description"]}</td>'
            )
        parts.append(_table(["Port", "Type", "Description"], rows))
    else:
        parts.append(f'<p><font color="{TEXT_MUTED}">No input ports</font></p>')

    # Outputs
    outputs = entry.get("outputs", [])
    parts.append(_section("Outputs"))
    if outputs:
        rows = []
        for o in outputs:
            rows.append(
                f'<td>{_code(o["name"])}</td>'
                f'<td>{_muted(o["type"])}</td>'
                f'<td>{o["description"]}</td>'
            )
        parts.append(_table(["Port", "Type", "Description"], rows))
    else:
        parts.append(f'<p><font color="{TEXT_MUTED}">No output ports</font></p>')

    # Parameters
    params = entry.get("params", [])
    if params:
        parts.append(_section("Parameters"))
        rows = []
        for p in params:
            default_val = str(p.get("default", ""))
            rows.append(
                f'<td>{_code(p["name"])}</td>'
                f'<td>{_code(default_val)}</td>'
                f'<td>{p["description"]}</td>'
            )
        parts.append(_table(["Parameter", "Default", "Description"], rows))

    # Result fields
    result = entry.get("result")
    if result:
        fields = result.get("fields", [])
        if fields:
            parts.append(_section(f'Result: {result["class"]}'))
            rows = []
            for f in fields:
                rows.append(
                    f'<td>{_code(f["name"])}</td>'
                    f'<td>{_muted(f["type"])}</td>'
                    f'<td>{f["description"]}</td>'
                )
            parts.append(_table(["Field", "Type", "Description"], rows))

    # Notes
    notes = entry.get("notes")
    if notes:
        parts.append(_section("Usage Notes"))
        # Highlight inline <code> in notes
        notes = re.sub(
            r'<code>(.*?)</code>',
            lambda m: f'<code>{highlight_python(m.group(1))}</code>',
            notes
        )
        # Wrap <pre> blocks with highlighted code
        notes = re.sub(
            r'<pre>(.*?)</pre>',
            lambda m: _code_block(m.group(1)),
            notes,
            flags=re.DOTALL
        )
        parts.append(f'<div>{notes}</div>')

    # See also
    see = entry.get("see_also")
    if see:
        parts.append(_section("See Also"))
        items = ", ".join(_code(s) for s in see)
        parts.append(f'<p>{items}</p>')

    # Sub-sections
    sub_sections = {
        "filters": "Available Filters",
        "calc_ops": "Operations",
        "measurement_modes": "Measurement Modes",
        "barcode_types": "Barcode Types",
    }
    for sub_key, sub_title in sub_sections.items():
        sub_data = entry.get(sub_key)
        if not sub_data:
            continue
        parts.append(_section(sub_title))
        for sk, sv in sub_data.items():
            parts.append(f'<hr><h4>{sv.get("name", sk)}</h4>')
            if sv.get("how_it_works"):
                parts.append(f'<p><strong>How it works:</strong> {sv["how_it_works"]}</p>')
            if sv.get("effect"):
                parts.append(f'<p><strong>Effect:</strong> {sv["effect"]}</p>')
            sub_params = sv.get("params_detail", [])
            if sub_params:
                rows = []
                for sp in sub_params:
                    rows.append(
                        f'<td>{_code(sp.get("name",""))}</td>'
                        f'<td>{sp.get("detail","")}</td>'
                    )
                parts.append(_table(["Parameter", "Detail"], rows))
            if sv.get("when_to_use"):
                parts.append(f'<p><strong>When to use:</strong> {sv["when_to_use"]}</p>')
            if sv.get("limitations"):
                parts.append(f'<p><strong>Limitations:</strong> {sv["limitations"]}</p>')

    style = (
        f"body {{ background-color: {DARK_BG}; color: {TEXT}; "
        f"font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; "
        f"line-height: 1.7; padding: 12px; margin: 0; }}"
        f"h2 {{ color: #7aa2f7; font-size: 20px; font-weight: 700; "
        f"border-bottom: 2px solid #7aa2f7; padding-bottom: 8px; margin: 0 0 6px 0; }}"
        f"h3 {{ color: #7aa2f7; font-size: 12px; font-weight: 700; "
        f"text-transform: uppercase; letter-spacing: 1px; "
        f"border-bottom: 1px solid {BORDER}; padding-bottom: 6px; "
        f"margin: 24px 0 12px 0; }}"
        f"h4 {{ color: #7aa2f7; font-size: 14px; font-weight: 600; "
        f"margin: 18px 0 6px 0; }}"
        f"p {{ line-height: 1.7; margin: 8px 0; }}"
        f"table {{ border-collapse: collapse; width: 100%; margin: 10px 0; "
        f"font-size: 13px; }}"
        f"th {{ background-color: {ALT_BG}; color: #7aa2f7; text-align: left; "
        f"padding: 7px 10px; border: 1px solid {BORDER}; "
        f"font-size: 12px; font-weight: 600; }}"
        f"td {{ padding: 7px 10px; border: 1px solid {BORDER}; "
        f"border-top: none; vertical-align: top; }}"
        f"tr:nth-child(even) td {{ background-color: {ALT_BG}; }}"
        f"code {{ font-family: Consolas, monospace; font-size: 12.5px; }}"
        f"hr {{ border: none; border-top: 1px solid {BORDER}; margin: 20px 0; }}"
    )
    body = "".join(parts)
    return f'<!DOCTYPE html><html><head><meta charset="utf-8"><style>{style}</style></head><body>{body}</body></html>'


# ── Dialog ─────────────────────────────────────────────────────────────────

class ToolHelpDialog(QDialog):
    """Modal dialog listing all tools with selectable documentation."""

    def __init__(self, parent=None, preselected_key: str = ""):
        super().__init__(parent)
        self.setWindowTitle("Tool Reference")
        self.setMinimumSize(850, 600)
        self.resize(900, 650)
        self.setWindowFlags(
            self.windowFlags() |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowMinimizeButtonHint
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        # ── Left panel: tool list ─────────────────────────────────────
        self.list_widget = QListWidget()
        self.list_widget.setMinimumWidth(200)
        self.list_widget.setMaximumWidth(280)
        self.list_widget.setStyleSheet(
            f"QListWidget {{ background-color: {DARK_BG}; border: none; border-right: 1px solid {BORDER}; }}"
            f"QListWidget::item {{ padding: 8px 16px; color: {TEXT}; font-size: 12.5px; }}"
            f"QListWidget::item:selected {{ background-color: {HIGHLIGHT}; color: #ffffff; }}"
            f"QListWidget::item:hover {{ background-color: #252525; }}"
        )

        self.list_widget.blockSignals(True)
        self.list_widget.currentItemChanged.connect(self._on_selection_changed)
        layout.addWidget(self.list_widget)

        # ── Right panel: QTextBrowser with Markdown ─────────────────--
        self.browser = QTextBrowser()
        self.browser.setStyleSheet(
            f"QTextBrowser {{ background-color: {DARK_BG}; color: {TEXT}; border: none; "
            f"font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; padding: 12px; }}"
            f"QTextBrowser a {{ color: {HIGHLIGHT}; }}"
        )
        layout.addWidget(self.browser, 1)

        # ── Debounce ──────────────────────────────────────────────────
        self._debounce_timer = QTimer(self)
        self._debounce_timer.setSingleShot(True)
        self._debounce_timer.setInterval(150)
        self._debounce_timer.timeout.connect(self._do_render)
        self._pending_entry: dict | None = None
        self._current_key: str = ""

        # ── Populate (signals blocked) ────────────────────────────────
        self._items = []
        self._populate()
        self.list_widget.blockSignals(False)

        if preselected_key:
            self._select_key(preselected_key)
        else:
            for key, _ in self._items:
                if key:
                    self._select_key(key)
                    self._current_key = key
                    break

        # Initial render (immediate with QTextBrowser)
        self._pending_entry = TOOL_HELP.get(self._current_key, {})
        self._do_render()

    def _populate(self):
        """Build the categorized list from TOOL_HELP data."""
        grouped = {}
        for key, entry in TOOL_HELP.items():
            cat = entry.get("category", "Other")
            grouped.setdefault(cat, []).append((key, entry))

        first = True
        for cat in _CATEGORIES:
            items = grouped.get(cat, [])
            if not items:
                continue
            if not first:
                sep = QListWidgetItem()
                sep.setFlags(Qt.ItemFlag.NoItemFlags)
                sep.setSizeHint(QSize(0, 6))
                self.list_widget.addItem(sep)
                self._items.append(("", {}))
            first = False

            header = QListWidgetItem(f"\u25b8  {cat}")
            header.setFlags(Qt.ItemFlag.NoItemFlags)
            header.setData(Qt.ItemDataRole.UserRole, "")
            header.setForeground(QColor("#7aa2f7"))
            header_font = header.font()
            header_font.setBold(True)
            header_font.setPointSize(9)
            header.setFont(header_font)
            header.setSizeHint(QSize(0, 24))
            self.list_widget.addItem(header)
            self._items.append(("", {}))

            for key, entry in sorted(items, key=lambda x: x[1].get("name", "")):
                item = QListWidgetItem(f"\u00b7  {entry.get('name', key)}")
                item.setData(Qt.ItemDataRole.UserRole, key)
                self.list_widget.addItem(item)
                self._items.append((key, entry))

    def _select_key(self, key: str):
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item and item.data(Qt.ItemDataRole.UserRole) == key:
                self.list_widget.setCurrentItem(item)
                self._current_key = key
                return

    def _on_selection_changed(self, current, previous):
        if not current:
            return
        self._current_key = current.data(Qt.ItemDataRole.UserRole) or ""
        self._pending_entry = TOOL_HELP.get(self._current_key, {})
        self._debounce_timer.start()

    def _do_render(self):
        if self._pending_entry:
            html = _render_html(self._pending_entry)
            self.browser.setHtml(html)
            self._pending_entry = None

    @staticmethod
    def show_for_tool(tool, parent=None):
        cls_name = tool.__class__.__name__
        key = TYPE_TO_HELP_KEY.get(cls_name, tool.name)
        dialog = ToolHelpDialog(parent, preselected_key=key)
        dialog.exec()
