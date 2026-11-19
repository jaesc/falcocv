"""Help dialog: guides, search, tool pages and internal links."""

import pytest
from PyQt6.QtCore import Qt, QUrl

from falcocv.ui.widgets.help_dialog import ToolHelpDialog

pytestmark = pytest.mark.ui


def _dialog(qtbot, key: str = "") -> ToolHelpDialog:
    dialog = ToolHelpDialog(preselected_key=key)
    qtbot.addWidget(dialog)
    return dialog


def _labels(dialog: ToolHelpDialog) -> list[str]:
    labels = []
    for i in range(dialog.list_widget.count()):
        item = dialog.list_widget.item(i)
        key = item.data(Qt.ItemDataRole.UserRole)
        if key:
            labels.append(key)
    return labels


def test_guides_and_tools_are_listed(qtbot):
    dialog = _dialog(qtbot)
    labels = _labels(dialog)

    assert "Getting Started" in labels
    assert "Command Line" in labels
    assert "User Libraries" in labels
    assert "Blob Detector" in labels


def test_guide_renders_sections_without_port_tables(qtbot):
    dialog = _dialog(qtbot, "Getting Started")
    text = dialog.browser.toPlainText()

    assert "Ways to run FalcoCV" in text
    assert "No input ports" not in text


def test_tool_page_keeps_the_port_tables(qtbot):
    dialog = _dialog(qtbot, "Blob Detector")
    text = dialog.browser.toPlainText()

    assert "Inputs" in text
    assert "Outputs" in text


def test_search_filters_the_page_list(qtbot):
    dialog = _dialog(qtbot)

    dialog.search.setText("libraries")

    labels = _labels(dialog)
    assert "User Libraries" in labels
    assert "Blob Detector" not in labels


def test_clearing_the_search_restores_the_list(qtbot):
    dialog = _dialog(qtbot)
    before = dialog.list_widget.count()

    dialog.search.setText("libraries")
    dialog.search.clear()

    assert dialog.list_widget.count() == before


def test_internal_link_selects_the_page(qtbot):
    dialog = _dialog(qtbot)

    dialog._on_anchor(QUrl("page:Command Line"))

    assert dialog._current_key == "Command Line"
    assert "falcocv run" in dialog.browser.toPlainText()
