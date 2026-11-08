"""Global wheel guard: wheel scrolls panels, never edits values."""

import pytest
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QWheelEvent
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QScrollArea,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from falcocv.ui.wheel_guard import install_wheel_guard

pytestmark = pytest.mark.ui


def _wheel_event(widget):
    center = widget.rect().center()
    return QWheelEvent(
        QPointF(center),
        QPointF(widget.mapToGlobal(center)),
        QPoint(0, 0),
        QPoint(0, -120),
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )


def _send_wheel(widget):
    QApplication.sendEvent(widget, _wheel_event(widget))


def _scroll_area(qtbot, spin, combo):
    content = QWidget()
    layout = QVBoxLayout(content)
    layout.addWidget(spin)
    layout.addWidget(combo)
    filler = QWidget()
    filler.setMinimumHeight(800)
    layout.addWidget(filler)

    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setWidget(content)
    scroll.resize(320, 220)
    qtbot.addWidget(scroll)
    scroll.show()
    qtbot.wait(50)
    return scroll


def test_wheel_scrolls_without_changing_spin_or_combo(qtbot, qapp):
    install_wheel_guard(qapp)
    spin = QSpinBox()
    spin.setRange(0, 10)
    spin.setValue(1)
    combo = QComboBox()
    combo.addItems(["A", "B"])
    combo.setCurrentIndex(0)
    scroll = _scroll_area(qtbot, spin, combo)
    bar = scroll.verticalScrollBar()
    assert bar.maximum() > 0

    bar.setValue(0)
    _send_wheel(spin)
    assert spin.value() == 1
    assert bar.value() > 0

    bar.setValue(0)
    _send_wheel(combo)
    assert combo.currentIndex() == 0
    assert bar.value() > 0


def test_wheel_on_editable_line_edit_does_not_change_value(qtbot, qapp):
    install_wheel_guard(qapp)
    spin = QSpinBox()
    spin.setRange(0, 10)
    spin.setValue(2)
    combo = QComboBox()
    combo.setEditable(True)
    combo.addItems(["A", "B"])
    combo.setCurrentIndex(0)
    scroll = _scroll_area(qtbot, spin, combo)
    bar = scroll.verticalScrollBar()
    assert bar.maximum() > 0

    bar.setValue(0)
    _send_wheel(spin.lineEdit())
    assert spin.value() == 2
    assert bar.value() > 0

    bar.setValue(0)
    _send_wheel(combo.lineEdit())
    assert combo.currentIndex() == 0
    assert bar.value() > 0


def test_wheel_outside_scroll_area_does_not_change_value(qtbot, qapp):
    install_wheel_guard(qapp)
    host = QWidget()
    layout = QVBoxLayout(host)
    spin = QSpinBox()
    spin.setRange(0, 10)
    spin.setValue(3)
    combo = QComboBox()
    combo.addItems(["A", "B"])
    combo.setCurrentIndex(0)
    layout.addWidget(spin)
    layout.addWidget(combo)
    qtbot.addWidget(host)
    host.show()
    qtbot.wait(20)

    _send_wheel(spin)
    _send_wheel(combo)
    assert spin.value() == 3
    assert combo.currentIndex() == 0


def test_keyboard_still_changes_spin_value(qtbot, qapp):
    install_wheel_guard(qapp)
    spin = QSpinBox()
    spin.setRange(0, 10)
    spin.setValue(1)
    qtbot.addWidget(spin)
    spin.show()
    spin.setFocus()
    qtbot.wait(20)
    qtbot.keyClick(spin, Qt.Key.Key_Up)
    assert spin.value() == 2
