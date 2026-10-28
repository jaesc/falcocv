from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QPalette, QColor

# ---------------------------------------------------------------------------
# Base shared by all dark themes (avoids repeating ~25 generic keys)
# ---------------------------------------------------------------------------
_BASE_DARK = {
    "TOOLTIP_TEXT": "#ffffff",
    "BRIGHT_TEXT": "#ffffff",
    "HIGHLIGHTED_TEXT": "#ffffff",
    "SELECTION_TEXT": "#ffffff",
    "LINK": "#73b1ff",
    "LINK_VISITED": "#95c7ff",
    "SHADOW": "#000000",
    "MIDLIGHT": "#2d2d2d",
    "MENU_FG": "#e8e8e8",
    "MENU_SELECTED": "#333333",
    "HEADER_BG": "#2b2b2b",
    "HEADER_FG": "#dddddd",
    "TABLE_BG": "#141414",
    "TABLE_ALT": "#1d1d1d",
    "GRIDLINE": "#2b2b2b",
    "BUTTON_PRESSED": "#323232",
    "SCROLL_HOVER": "#505050",
    "WHITE": "#ffffff",
    "FOOTER_BG": "#252525",
}

# ---------------------------------------------------------------------------
# Theme definitions – each dict only overrides what differs from _BASE_DARK
# ---------------------------------------------------------------------------
THEMES = {
    "dark": {
        **_BASE_DARK,
        "DARK_BG": "#121212",
        "ALT_BG": "#1f1f1f",
        "PANEL_BG": "#181818",
        "TEXT": "#e0e0e0",
        "TEXT_MUTED": "#aaaaaa",
        "HIGHLIGHT": "#3d6dad",
        "SUCCESS": "#4caf50",
        "WARNING": "#f44336",
        "DANGER": "#d32f2f",
        "INFO": "#2196f3",
        "CAUTION": "#ffa726",
        "TOOLTIP_BG": "#252525",
        "BORDER": "#333333",
        "SCROLLBAR_BG": "#212121",
        "SCROLL_HANDLE": "#3a3a3a",
        "TEXT_DIM": "#666666",
        "SCRIPT_BG": "#1E1E1E",
        "SCRIPT_TEXT": "#D4D4D4",
        "SCRIPT_KEYWORD": "#569CD6",
        "SCRIPT_NUMBER": "#CE9178",
        "SCRIPT_COMMENT": "#6A9955",
        "SCRIPT_DECORATOR": "#D7BA7D",
        "SCRIPT_BP_BG": "#E06C75",
        "SCRIPT_BP_FG": "#BE5046",
        "SCRIPT_ARROW_BG": "#E5C07B",
        "SCRIPT_ARROW_FG": "#D19A66",
        "SCRIPT_HIGHLIGHT_BG": "#3A3A1A",
        "SCRIPT_MARGIN_BG": "#252526",
        "SCRIPT_MARGIN_FG": "#858585",
        "SCRIPT_LINE_HIGHLIGHT": "#2A2D2E",
        "SCRIPT_CARET": "#AEAFAD",
        "SCRIPT_SELECTION_BG": "#264F78",
        "SCRIPT_EDGE": "#3A3D3E",
        "SCRIPT_DEBUG_LINE_BG": "#3A3A2A",
        "SCRIPT_INPUT_BG": "#3C3C3C",
        "SCRIPT_BAR_BG": "#2A2A2A",
        "SCRIPT_FOLD_EDGE": "#2D2D2D",
        "HIST_BG": "#1e1e1e",
        "HIST_BAR": "#4a90d9",
        "HIST_BAR_BORDER": "#3a7bc8",
        "HIST_STAT": "#ff9800",
        "HIST_LABEL": "#888888",
        "MENU_BG": "#2a2a2a",
        "SPLASH_MSG": "#666666",
        "ROI_ACTIVE_BG": "#ff9800",
        "DISABLED_BORDER": "#222222",
        "DISABLED_TEXT": "#555555",
    },
    "material": {
        **_BASE_DARK,
        "DARK_BG": "#1e1e2e",
        "ALT_BG": "#25253a",
        "PANEL_BG": "#222233",
        "TEXT": "#e8e8e8",
        "TEXT_MUTED": "#a0a0b0",
        "HIGHLIGHT": "#4285f4",
        "SUCCESS": "#34a853",
        "WARNING": "#ea4335",
        "DANGER": "#ea4335",
        "INFO": "#4285f4",
        "CAUTION": "#fbbc04",
        "TOOLTIP_BG": "#2a2a3c",
        "FOOTER_BG": "#2a2a3c",
        "BORDER": "#2a2a3c",
        "SCROLLBAR_BG": "#1e1e2e",
        "SCROLL_HANDLE": "#3a3a50",
        "TEXT_DIM": "#666688",
        "SCRIPT_BG": "#1E1E2E",
        "SCRIPT_TEXT": "#E8E8E8",
        "SCRIPT_KEYWORD": "#73b1ff",
        "SCRIPT_NUMBER": "#f4a261",
        "SCRIPT_COMMENT": "#7ee787",
        "SCRIPT_DECORATOR": "#fbbc04",
        "SCRIPT_BP_BG": "#ea4335",
        "SCRIPT_BP_FG": "#b9362a",
        "SCRIPT_ARROW_BG": "#fbbc04",
        "SCRIPT_ARROW_FG": "#c99703",
        "SCRIPT_HIGHLIGHT_BG": "#2A2A1A",
        "SCRIPT_MARGIN_BG": "#25253a",
        "SCRIPT_MARGIN_FG": "#666688",
        "SCRIPT_LINE_HIGHLIGHT": "#2a2a3c",
        "SCRIPT_CARET": "#e8e8e8",
        "SCRIPT_SELECTION_BG": "#1a3a6b",
        "SCRIPT_EDGE": "#2a2a3c",
        "SCRIPT_DEBUG_LINE_BG": "#2a2a1a",
        "SCRIPT_INPUT_BG": "#2a2a3c",
        "SCRIPT_BAR_BG": "#1e1e2e",
        "SCRIPT_FOLD_EDGE": "#2a2a3c",
        "HIST_BG": "#1e1e2e",
        "HIST_BAR": "#4285f4",
        "HIST_BAR_BORDER": "#346ac3",
        "HIST_STAT": "#fbbc04",
        "HIST_LABEL": "#666688",
        "MENU_BG": "#25253a",
        "SPLASH_MSG": "#555577",
        "ROI_ACTIVE_BG": "#fbbc04",
        "DISABLED_BORDER": "#1a1a28",
        "DISABLED_TEXT": "#444455",
        "MENU_FG": "#e8e8e8",
        "MENU_SELECTED": "#2a2a3c",
        "HEADER_BG": "#2a2a3c",
        "HEADER_FG": "#bbbbcc",
        "TABLE_BG": "#1e1e2e",
        "TABLE_ALT": "#242438",
        "GRIDLINE": "#2a2a3c",
        "BUTTON_PRESSED": "#1a1a25",
        "SCROLL_HOVER": "#4a4a60",
        "MIDLIGHT": "#2a2a3c",
    },
    "light": {
        "TOOLTIP_TEXT": "#24292e",
        "BRIGHT_TEXT": "#24292e",
        "HIGHLIGHTED_TEXT": "#ffffff",
        "SELECTION_TEXT": "#24292e",
        "LINK": "#0969da",
        "LINK_VISITED": "#8250df",
        "SHADOW": "#d0d7de",
        "MIDLIGHT": "#eaeef2",
        "MENU_FG": "#24292e",
        "MENU_SELECTED": "#ddf4ff",
        "HEADER_BG": "#f6f8fa",
        "HEADER_FG": "#24292e",
        "TABLE_BG": "#ffffff",
        "TABLE_ALT": "#f6f8fa",
        "GRIDLINE": "#d0d7de",
        "BUTTON_PRESSED": "#e1e4e8",
        "SCROLL_HOVER": "#8c959f",
        "WHITE": "#ffffff",
        "FOOTER_BG": "#f6f8fa",
        "DARK_BG": "#f5f5f5",
        "ALT_BG": "#ffffff",
        "PANEL_BG": "#f0f0f0",
        "TEXT": "#24292e",
        "TEXT_MUTED": "#6e7781",
        "HIGHLIGHT": "#0969da",
        "SUCCESS": "#1a7f37",
        "WARNING": "#cf222e",
        "DANGER": "#cf222e",
        "INFO": "#0969da",
        "CAUTION": "#9a6700",
        "TOOLTIP_BG": "#ffffff",
        "BORDER": "#d0d7de",
        "SCROLLBAR_BG": "#e1e4e8",
        "SCROLL_HANDLE": "#babfc4",
        "TEXT_DIM": "#8c959f",
        "SCRIPT_BG": "#ffffff",
        "SCRIPT_TEXT": "#24292e",
        "SCRIPT_KEYWORD": "#cf222e",
        "SCRIPT_NUMBER": "#0550ae",
        "SCRIPT_COMMENT": "#6e7781",
        "SCRIPT_DECORATOR": "#9a6700",
        "SCRIPT_BP_BG": "#cf222e",
        "SCRIPT_BP_FG": "#a40e26",
        "SCRIPT_ARROW_BG": "#9a6700",
        "SCRIPT_ARROW_FG": "#7a5200",
        "SCRIPT_HIGHLIGHT_BG": "#fff8c5",
        "SCRIPT_MARGIN_BG": "#f6f8fa",
        "SCRIPT_MARGIN_FG": "#8c959f",
        "SCRIPT_LINE_HIGHLIGHT": "#f6f8fa",
        "SCRIPT_CARET": "#24292e",
        "SCRIPT_SELECTION_BG": "#ddf4ff",
        "SCRIPT_EDGE": "#d0d7de",
        "SCRIPT_DEBUG_LINE_BG": "#fff8c5",
        "SCRIPT_INPUT_BG": "#f6f8fa",
        "SCRIPT_BAR_BG": "#f5f5f5",
        "SCRIPT_FOLD_EDGE": "#d0d7de",
        "HIST_BG": "#f5f5f5",
        "HIST_BAR": "#0969da",
        "HIST_BAR_BORDER": "#0550ae",
        "HIST_STAT": "#9a6700",
        "HIST_LABEL": "#8c959f",
        "MENU_BG": "#f6f8fa",
        "SPLASH_MSG": "#8c959f",
        "ROI_ACTIVE_BG": "#9a6700",
        "DISABLED_BORDER": "#eaeef2",
        "DISABLED_TEXT": "#8c959f",
    },
    "monokai": {
        **_BASE_DARK,
        "DARK_BG": "#272822",
        "ALT_BG": "#31332c",
        "PANEL_BG": "#2d2f28",
        "TEXT": "#f8f8f2",
        "TEXT_MUTED": "#a0a090",
        "HIGHLIGHT": "#a6e22e",
        "SUCCESS": "#a6e22e",
        "WARNING": "#f92672",
        "DANGER": "#f92672",
        "INFO": "#66d9ef",
        "CAUTION": "#e6db74",
        "TOOLTIP_BG": "#383830",
        "FOOTER_BG": "#383830",
        "BORDER": "#3e3d32",
        "SCROLLBAR_BG": "#272822",
        "SCROLL_HANDLE": "#49483e",
        "TEXT_DIM": "#75715e",
        "SCRIPT_BG": "#272822",
        "SCRIPT_TEXT": "#f8f8f2",
        "SCRIPT_KEYWORD": "#66d9ef",
        "SCRIPT_NUMBER": "#ae81ff",
        "SCRIPT_COMMENT": "#75715e",
        "SCRIPT_DECORATOR": "#e6db74",
        "SCRIPT_BP_BG": "#f92672",
        "SCRIPT_BP_FG": "#c41e5a",
        "SCRIPT_ARROW_BG": "#e6db74",
        "SCRIPT_ARROW_FG": "#b8ae5c",
        "SCRIPT_HIGHLIGHT_BG": "#3a3a1a",
        "SCRIPT_MARGIN_BG": "#31332c",
        "SCRIPT_MARGIN_FG": "#75715e",
        "SCRIPT_LINE_HIGHLIGHT": "#383830",
        "SCRIPT_CARET": "#f8f8f2",
        "SCRIPT_SELECTION_BG": "#49483e",
        "SCRIPT_EDGE": "#3e3d32",
        "SCRIPT_DEBUG_LINE_BG": "#3a3a1a",
        "SCRIPT_INPUT_BG": "#383830",
        "SCRIPT_BAR_BG": "#272822",
        "SCRIPT_FOLD_EDGE": "#3e3d32",
        "HIST_BG": "#272822",
        "HIST_BAR": "#a6e22e",
        "HIST_BAR_BORDER": "#85c41a",
        "HIST_STAT": "#e6db74",
        "HIST_LABEL": "#75715e",
        "MENU_BG": "#31332c",
        "SPLASH_MSG": "#75715e",
        "ROI_ACTIVE_BG": "#e6db74",
        "DISABLED_BORDER": "#1f201b",
        "DISABLED_TEXT": "#4e4d42",
        "MENU_FG": "#f8f8f2",
        "MENU_SELECTED": "#3e3d32",
        "HEADER_BG": "#3e3d32",
        "HEADER_FG": "#cfcfc2",
        "TABLE_BG": "#272822",
        "TABLE_ALT": "#2f3029",
        "GRIDLINE": "#3e3d32",
        "BUTTON_PRESSED": "#1e1f1a",
        "SCROLL_HOVER": "#5a5848",
        "MIDLIGHT": "#3e3d32",
    },
    "synthwave": {
        **_BASE_DARK,
        "DARK_BG": "#1a0a2e",
        "ALT_BG": "#2d1b4e",
        "PANEL_BG": "#241044",
        "TEXT": "#e0e0ff",
        "TEXT_MUTED": "#b088c0",
        "HIGHLIGHT": "#00f5ff",
        "SUCCESS": "#39ff14",
        "WARNING": "#ff006e",
        "DANGER": "#ff006e",
        "INFO": "#00f5ff",
        "CAUTION": "#ffe600",
        "TOOLTIP_BG": "#3a2060",
        "FOOTER_BG": "#2d1b4e",
        "BORDER": "#4a3070",
        "SCROLLBAR_BG": "#1a0a2e",
        "SCROLL_HANDLE": "#4a3070",
        "TEXT_DIM": "#8060a0",
        "SCRIPT_BG": "#1a0a2e",
        "SCRIPT_TEXT": "#e0e0ff",
        "SCRIPT_KEYWORD": "#ff006e",
        "SCRIPT_NUMBER": "#ffe600",
        "SCRIPT_COMMENT": "#8060a0",
        "SCRIPT_DECORATOR": "#ffe600",
        "SCRIPT_BP_BG": "#ff006e",
        "SCRIPT_BP_FG": "#cc0058",
        "SCRIPT_ARROW_BG": "#ffe600",
        "SCRIPT_ARROW_FG": "#ccb800",
        "SCRIPT_HIGHLIGHT_BG": "#3a2050",
        "SCRIPT_MARGIN_BG": "#2d1b4e",
        "SCRIPT_MARGIN_FG": "#8060a0",
        "SCRIPT_LINE_HIGHLIGHT": "#2d1b4e",
        "SCRIPT_CARET": "#00f5ff",
        "SCRIPT_SELECTION_BG": "#4a2070",
        "SCRIPT_EDGE": "#4a3070",
        "SCRIPT_DEBUG_LINE_BG": "#3a2040",
        "SCRIPT_INPUT_BG": "#2d1b4e",
        "SCRIPT_BAR_BG": "#1a0a2e",
        "SCRIPT_FOLD_EDGE": "#4a3070",
        "HIST_BG": "#1a0a2e",
        "HIST_BAR": "#00f5ff",
        "HIST_BAR_BORDER": "#00c8d0",
        "HIST_STAT": "#ffe600",
        "HIST_LABEL": "#8060a0",
        "MENU_BG": "#2d1b4e",
        "SPLASH_MSG": "#8060a0",
        "ROI_ACTIVE_BG": "#ffe600",
        "DISABLED_BORDER": "#1a0a2e",
        "DISABLED_TEXT": "#604080",
        "MENU_FG": "#e0e0ff",
        "MENU_SELECTED": "#4a3070",
        "HEADER_BG": "#2d1b4e",
        "HEADER_FG": "#b088c0",
        "TABLE_BG": "#1a0a2e",
        "TABLE_ALT": "#241044",
        "GRIDLINE": "#4a3070",
        "BUTTON_PRESSED": "#1a0a2e",
        "SCROLL_HOVER": "#5a4080",
        "MIDLIGHT": "#4a3070",
    },
    "gruvbox": {
        **_BASE_DARK,
        "DARK_BG": "#1d2021",
        "ALT_BG": "#282828",
        "PANEL_BG": "#232526",
        "TEXT": "#fbf1c7",
        "TEXT_MUTED": "#a89984",
        "HIGHLIGHT": "#d65d0e",
        "SUCCESS": "#689d6a",
        "WARNING": "#cc241d",
        "DANGER": "#cc241d",
        "INFO": "#458588",
        "CAUTION": "#d79921",
        "TOOLTIP_BG": "#3c3836",
        "FOOTER_BG": "#282828",
        "BORDER": "#3c3836",
        "SCROLLBAR_BG": "#1d2021",
        "SCROLL_HANDLE": "#504945",
        "TEXT_DIM": "#7c6f64",
        "SCRIPT_BG": "#1d2021",
        "SCRIPT_TEXT": "#fbf1c7",
        "SCRIPT_KEYWORD": "#fb4934",
        "SCRIPT_NUMBER": "#d79921",
        "SCRIPT_COMMENT": "#7c6f64",
        "SCRIPT_DECORATOR": "#d79921",
        "SCRIPT_BP_BG": "#cc241d",
        "SCRIPT_BP_FG": "#9d1812",
        "SCRIPT_ARROW_BG": "#d79921",
        "SCRIPT_ARROW_FG": "#b0761a",
        "SCRIPT_HIGHLIGHT_BG": "#3c3836",
        "SCRIPT_MARGIN_BG": "#282828",
        "SCRIPT_MARGIN_FG": "#7c6f64",
        "SCRIPT_LINE_HIGHLIGHT": "#32302f",
        "SCRIPT_CARET": "#fbf1c7",
        "SCRIPT_SELECTION_BG": "#504945",
        "SCRIPT_EDGE": "#3c3836",
        "SCRIPT_DEBUG_LINE_BG": "#3c3836",
        "SCRIPT_INPUT_BG": "#32302f",
        "SCRIPT_BAR_BG": "#1d2021",
        "SCRIPT_FOLD_EDGE": "#3c3836",
        "HIST_BG": "#1d2021",
        "HIST_BAR": "#d65d0e",
        "HIST_BAR_BORDER": "#b34a0a",
        "HIST_STAT": "#d79921",
        "HIST_LABEL": "#7c6f64",
        "MENU_BG": "#282828",
        "SPLASH_MSG": "#7c6f64",
        "ROI_ACTIVE_BG": "#d79921",
        "DISABLED_BORDER": "#1a1c1c",
        "DISABLED_TEXT": "#504945",
        "MENU_FG": "#fbf1c7",
        "MENU_SELECTED": "#504945",
        "HEADER_BG": "#32302f",
        "HEADER_FG": "#d5c4a1",
        "TABLE_BG": "#1d2021",
        "TABLE_ALT": "#232526",
        "GRIDLINE": "#3c3836",
        "BUTTON_PRESSED": "#1a1c1c",
        "SCROLL_HOVER": "#665c54",
        "MIDLIGHT": "#3c3836",
    },
    "nord": {
        **_BASE_DARK,
        "DARK_BG": "#2e3440",
        "ALT_BG": "#3b4252",
        "PANEL_BG": "#353b4a",
        "TEXT": "#eceff4",
        "TEXT_MUTED": "#8190a0",
        "HIGHLIGHT": "#8fbcbb",
        "SUCCESS": "#a3be8c",
        "WARNING": "#bf616a",
        "DANGER": "#bf616a",
        "INFO": "#81a1c1",
        "CAUTION": "#ebcb8b",
        "TOOLTIP_BG": "#434c5e",
        "FOOTER_BG": "#3b4252",
        "BORDER": "#434c5e",
        "SCROLLBAR_BG": "#2e3440",
        "SCROLL_HANDLE": "#4c566a",
        "TEXT_DIM": "#6c7890",
        "SCRIPT_BG": "#2e3440",
        "SCRIPT_TEXT": "#eceff4",
        "SCRIPT_KEYWORD": "#81a1c1",
        "SCRIPT_NUMBER": "#8fbcbb",
        "SCRIPT_COMMENT": "#6c7890",
        "SCRIPT_DECORATOR": "#ebcb8b",
        "SCRIPT_BP_BG": "#bf616a",
        "SCRIPT_BP_FG": "#964b50",
        "SCRIPT_ARROW_BG": "#ebcb8b",
        "SCRIPT_ARROW_FG": "#c4a46e",
        "SCRIPT_HIGHLIGHT_BG": "#3b4252",
        "SCRIPT_MARGIN_BG": "#3b4252",
        "SCRIPT_MARGIN_FG": "#6c7890",
        "SCRIPT_LINE_HIGHLIGHT": "#3b4252",
        "SCRIPT_CARET": "#eceff4",
        "SCRIPT_SELECTION_BG": "#434c5e",
        "SCRIPT_EDGE": "#434c5e",
        "SCRIPT_DEBUG_LINE_BG": "#3b4252",
        "SCRIPT_INPUT_BG": "#3b4252",
        "SCRIPT_BAR_BG": "#2e3440",
        "SCRIPT_FOLD_EDGE": "#434c5e",
        "HIST_BG": "#2e3440",
        "HIST_BAR": "#8fbcbb",
        "HIST_BAR_BORDER": "#70a5a4",
        "HIST_STAT": "#ebcb8b",
        "HIST_LABEL": "#6c7890",
        "MENU_BG": "#3b4252",
        "SPLASH_MSG": "#6c7890",
        "ROI_ACTIVE_BG": "#ebcb8b",
        "DISABLED_BORDER": "#262c36",
        "DISABLED_TEXT": "#4c566a",
        "MENU_FG": "#eceff4",
        "MENU_SELECTED": "#434c5e",
        "HEADER_BG": "#434c5e",
        "HEADER_FG": "#d8dee9",
        "TABLE_BG": "#2e3440",
        "TABLE_ALT": "#353b4a",
        "GRIDLINE": "#434c5e",
        "BUTTON_PRESSED": "#262c36",
        "SCROLL_HOVER": "#5e6a80",
        "MIDLIGHT": "#434c5e",
    },
}

_current_theme = "dark"


def apply_theme(name):
    """Reassign all string colour constants from a theme dict (module globals)."""
    global _current_theme
    theme = THEMES.get(name, THEMES["dark"])
    for k, v in theme.items():
        globals()[k] = v
    _current_theme = name


def current_theme() -> str:
    return _current_theme


# QColor constants (immutable objects, not themeable via string swap)
HANDLE_STROKE = QColor(120, 120, 120)
HANDLE_FILL = QColor(200, 200, 200)
HIGHLIGHT_ALPHA = QColor(61, 109, 173, 40)
OVERLAY_FALLBACK = QColor(0, 255, 0)
PIN_INPUT = QColor(0, 150, 255)
PIN_OUTPUT = QColor(255, 90, 90)
PIN_BORDER = QColor(70, 70, 70)
NODE_TITLE = QColor(240, 240, 240)
NODE_SELECT_BG = QColor(255, 255, 255, 20)
NODE_SELECT_BORDER = QColor(255, 255, 255, 50)
CONNECTION_LINE = QColor(100, 150, 255, 180)
TEMP_CONNECTION = QColor(100, 160, 255, 160)
ARROW_COLOR = QColor("#cccccc")

import json
import os

from falcocv.core.config_paths import config_path as _cfg_path_fn

_initial_theme = "dark"
_cfg_path = _cfg_path_fn()
if os.path.exists(_cfg_path):
    try:
        with open(_cfg_path, encoding="utf-8") as _f:
            _data = json.load(_f)
        _initial_theme = _data.get("theme", "dark")
    except Exception:
        pass

apply_theme(_initial_theme)


def disabled_css() -> str:
    return f"""
        QGroupBox {{ border-color: {DISABLED_BORDER}; }}
        QGroupBox::title {{ color: {DISABLED_TEXT}; }}
        QGroupBox:disabled QPushButton {{ color: {DISABLED_TEXT}; background-color: {PANEL_BG}; border: 1px solid {DISABLED_BORDER}; }}
        QGroupBox:disabled QDoubleSpinBox {{ color: {DISABLED_TEXT}; background-color: {PANEL_BG}; border: 1px solid {DISABLED_BORDER}; }}
    """


def apply_qt_theme(app: QApplication):
    app.setStyle("Fusion")
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(DARK_BG))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor(PANEL_BG))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(ALT_BG))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(TOOLTIP_BG))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TOOLTIP_TEXT))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor(ALT_BG))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.BrightText, QColor(BRIGHT_TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(HIGHLIGHT))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor(HIGHLIGHTED_TEXT))
    palette.setColor(QPalette.ColorRole.Link, QColor(LINK))
    palette.setColor(QPalette.ColorRole.LinkVisited, QColor(LINK_VISITED))
    palette.setColor(QPalette.ColorRole.Shadow, QColor(SHADOW))
    palette.setColor(QPalette.ColorRole.Mid, QColor(BORDER))
    palette.setColor(QPalette.ColorRole.Midlight, QColor(MIDLIGHT))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(TEXT_MUTED))
    app.setPalette(palette)
    app.setStyleSheet(f"""
        QMainWindow {{ background-color: {DARK_BG}; }}
        QTreeView, QListView {{ background-color: {DARK_BG}; }}
        QSplitter::handle {{ background-color: {BORDER}; }}
        QToolTip {{ color: {TOOLTIP_TEXT}; background-color: {TOOLTIP_BG}; border: 1px solid {BORDER}; }}
        QDialog, QMessageBox, QMenu, QToolTip {{ background-color: {ALT_BG}; color: {TEXT}; }}
        QGroupBox {{ background-color: {ALT_BG}; border: 1px solid {BORDER}; border-radius: 6px; margin-top: 16px; }}
        QGroupBox::title {{ color: {TEXT}; subcontrol-origin: margin; left: 10px; padding: 0 5px; }}
        QTabWidget::pane {{ border: 1px solid {BORDER}; background: {PANEL_BG}; }}
        QTabBar::tab {{ background: {ALT_BG}; color: {TEXT}; padding: 6px 12px; border: 1px solid transparent; border-bottom: none; }}
        QTabBar::tab:selected {{ background: {PANEL_BG}; border-color: {HIGHLIGHT}; }}
        QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox, QDoubleSpinBox, QDateEdit, QTimeEdit, QDateTimeEdit {{ background-color: {ALT_BG}; color: {TEXT}; border: 1px solid {BORDER}; selection-background-color: {HIGHLIGHT}; selection-color: {SELECTION_TEXT}; }}
        QPushButton {{ background-color: {ALT_BG}; color: {TEXT}; border: 1px solid {BORDER}; border-radius: 4px; padding: 4px 8px; }}
        QPushButton:hover {{ background-color: {PANEL_BG}; }}
        QPushButton:pressed {{ background-color: {BUTTON_PRESSED}; }}
        QCheckBox, QRadioButton {{ color: {TEXT}; }}
        QMenuBar, QMenu, QToolBar {{ background-color: {ALT_BG}; color: {MENU_FG}; }}
        QMenu::item {{ background-color: transparent; color: {MENU_FG}; }}
        QMenu::item:selected {{ background-color: {MENU_SELECTED}; }}
        QHeaderView::section {{ background-color: {HEADER_BG}; color: {HEADER_FG}; border: none; padding: 4px; }}
        QTableWidget, QTreeWidget, QListWidget, QTreeView, QTableView {{ background-color: {TABLE_BG}; alternate-background-color: {TABLE_ALT}; color: {TEXT}; gridline-color: {GRIDLINE}; }}
        QScrollBar:vertical, QScrollBar:horizontal {{ background: {SCROLLBAR_BG}; }}
        QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{ background: {SCROLL_HANDLE}; min-height: 20px; border-radius: 6px; }}
        QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover {{ background: {SCROLL_HOVER}; }}
        QScrollBar::add-line, QScrollBar::sub-line, QScrollBar::add-page, QScrollBar::sub-page {{ border: none; background: none; }}
        QDockWidget {{ background-color: {ALT_BG}; }}
        QSplitter::handle {{ background-color: {BORDER}; }}
    """)


COMPACT_CSS = """
    QGroupBox { margin-top: 10px; }
    QGroupBox::title { padding: 0 3px; }
    QTabBar::tab { padding: 3px 8px; }
    QPushButton { padding: 2px 6px; }
    QHeaderView::section { padding: 2px; }
    QScrollBar::handle:vertical, QScrollBar::handle:horizontal { min-height: 16px; }
"""
