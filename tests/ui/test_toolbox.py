"""Tool discovery: every registered tool must be reachable from the UI."""

import pytest

import falcocv.tools  # noqa: F401  (populate the registry)
from falcocv.core.tool_registry import get_all_tools
from falcocv.ui.widgets.toolbox import ToolButton, Toolbox


@pytest.fixture
def toolbox(qtbot):
    box = Toolbox(embedded=True)
    qtbot.addWidget(box)
    return box


def test_every_registered_tool_has_a_toolbox_button(toolbox):
    registered = {reg.tool_type for reg in get_all_tools()}
    shown = {button.tool_type for button in toolbox.findChildren(ToolButton)}
    assert shown == registered, f"missing from the toolbox: {sorted(registered - shown)}"


def test_every_registered_group_has_a_section(toolbox):
    from PyQt6.QtWidgets import QGroupBox

    titles = {group.title() for group in toolbox.findChildren(QGroupBox)}
    groups = {reg.group for reg in get_all_tools() if reg.group}
    assert groups <= titles, f"missing toolbox groups: {sorted(groups - titles)}"
    assert "Inspection" in titles


def test_help_categories_include_data_only_categories():
    from falcocv.ui.tool_help import TOOL_HELP
    from falcocv.ui.widgets.help_dialog import help_categories

    categories = {entry.get("category", "Other") for entry in TOOL_HELP.values()}
    ordered = help_categories(categories)

    assert set(ordered) >= categories
    assert "HMI" in ordered
    assert "Inspection" in ordered
