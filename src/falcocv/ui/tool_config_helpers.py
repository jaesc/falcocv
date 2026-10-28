"""Common helpers for tool configuration widgets.

Provides a uniform layout for parameter panel + viewer:
- Scrollable left parameter panel (like GigeCamera).
- Horizontal splitter with fixed left stretch and expanding right viewer.
- Persist/restore splitter sizes via ConfigManager.
- Common execution-time footer.
"""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QSplitter,
    QWidget,
)
from PyQt6.QtCore import Qt

from falcocv.core.config import ConfigManager
from falcocv.ui.style import ALT_BG, BORDER, PANEL_BG, SUCCESS


__all__ = [
    "wrap_params_scroll",
    "configure_params_splitter",
    "save_params_splitter",
    "create_execution_footer",
    "connect_execution_time",
]


def wrap_params_scroll(panel: QWidget) -> QScrollArea:
    """Wrap a parameter panel in a QScrollArea so it scrolls when too tall."""
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    scroll.setWidget(panel)
    # Give the scroll area a subtle background consistent with the theme.
    scroll.setStyleSheet(f"QScrollArea {{ background-color: {PANEL_BG}; border: none; }}")
    return scroll


def _min_left_width(panel: QWidget) -> int:
    """Compute a sensible minimum width for the parameter panel."""
    hint = panel.minimumSizeHint().width()
    # Add a small margin for the scrollbar and internal spacing.
    return max(180, hint + 12)


def configure_params_splitter(
    splitter: QSplitter,
    scroll_left: QScrollArea,
    right_panel: QWidget,
    tool_type: str,
) -> None:
    """Configure the splitter shared by all tool config widgets.

    - Left panel is wrapped in a scroll area and cannot be collapsed.
    - Left panel does not grow when the dialog is resized.
    - Right panel absorbs all extra space.
    - Sizes are restored from ConfigManager if valid; otherwise the left
      panel is sized to a sensible default (capped to avoid an overly wide
      panel when some control groups have a large minimumSizeHint).
    """
    panel = scroll_left.widget()
    min_left = _min_left_width(panel)
    # Allow the user to resize down to 180 px; the scroll area will show a
    # horizontal scrollbar if a control group is wider than the viewport.
    scroll_left.setMinimumWidth(180)
    # Cap the initial/restored width to avoid an overly wide left panel.
    initial_left = max(240, min_left)

    splitter.setCollapsible(0, False)
    splitter.setCollapsible(1, False)
    splitter.setStretchFactor(0, 0)
    splitter.setStretchFactor(1, 1)

    saved = ConfigManager().get_tool_splitter(tool_type)
    if saved and len(saved) >= 2 and saved[0] >= 180 and sum(saved) > 180:
        splitter.setSizes([max(saved[0], 240), saved[1]])
    else:
        splitter.setSizes([initial_left, 9999])


def save_params_splitter(splitter: QSplitter, tool_type: str) -> None:
    """Persist current splitter sizes to ConfigManager."""
    sizes = splitter.sizes()
    if sizes and len(sizes) >= 2 and sizes[0] > 0 and sizes[1] > 0:
        ConfigManager().set_tool_splitter(tool_type, sizes)


def create_execution_footer(parent: QWidget) -> tuple[QWidget, QLabel]:
    """Create the standard footer with the execution time label."""
    footer = QWidget(parent)
    footer.setFixedHeight(20)
    footer.setStyleSheet(
        f"background-color: {ALT_BG}; border-top: 1px solid {BORDER};"
    )
    layout = QHBoxLayout(footer)
    layout.setContentsMargins(5, 0, 5, 0)
    label_time = QLabel("0.0 ms")
    label_time.setStyleSheet(
        f"color: {SUCCESS}; font-weight: bold; font-size: 11px;"
    )
    layout.addWidget(label_time)
    layout.addStretch()
    return footer, label_time


def connect_execution_time(tool, label_time: QLabel) -> None:
    """Connect tool.data_changed so label_time shows last_execution_time_ms."""
    def _update() -> None:
        try:
            label_time.setText(f"{tool.last_execution_time_ms:.1f} ms")
        except RuntimeError:
            # The config dialog was closed and its widgets destroyed.
            pass

    tool.data_changed.connect(_update)

    def _disconnect() -> None:
        signal = getattr(tool, "data_changed", None)
        if signal is not None:
            signal.disconnect(_update)

    label_time.destroyed.connect(_disconnect)
