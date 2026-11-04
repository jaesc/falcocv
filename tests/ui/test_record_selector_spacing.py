"""Shared record-selector layout spacing."""

from PyQt6.QtWidgets import QHBoxLayout, QWidget

from falcocv.ui.tool_config_helpers import (
    add_record_selector,
    make_record_selector,
)


def test_record_selector_has_left_and_right_spacing(qtbot):
    parent = QWidget()
    layout = QHBoxLayout(parent)
    combo = make_record_selector(parent)

    add_record_selector(layout, combo, left=8, right=8)
    qtbot.addWidget(parent)

    assert layout.count() == 3
    left = layout.itemAt(0).spacerItem()
    right = layout.itemAt(2).spacerItem()
    assert left is not None
    assert right is not None
    assert left.sizeHint().width() == 8
    assert right.sizeHint().width() == 8
    assert layout.itemAt(1).widget() is combo
