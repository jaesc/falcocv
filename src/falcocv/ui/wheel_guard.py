"""Application-wide guard against accidental wheel value changes.

Qt spin boxes and combo boxes accept wheel events while merely hovering.
In scrollable settings panels that silently changes a value.  This filter
turns the wheel into scrolling for those controls everywhere, while keeping
keyboard and click editing untouched.
"""

from __future__ import annotations

from PyQt6.QtCore import QEvent, QObject
from PyQt6.QtWidgets import (
    QAbstractScrollArea,
    QAbstractSpinBox,
    QApplication,
    QComboBox,
    QLineEdit,
)

_GUARDS: list["WheelGuard"] = []


class WheelGuard(QObject):
    """Ignore wheel changes on spin boxes and combo boxes.

    The wheel event is forwarded to the nearest scroll area viewport so the
    settings panel keeps scrolling under the cursor.
    """

    def eventFilter(self, obj, event):
        if event.type() != QEvent.Type.Wheel:
            return False
        control = self._value_control(obj)
        if control is None:
            return False

        scroll = control.parentWidget()
        while scroll is not None and not isinstance(scroll, QAbstractScrollArea):
            scroll = scroll.parentWidget()
        if scroll is not None:
            QApplication.sendEvent(scroll.viewport(), event)
        return True

    @staticmethod
    def _value_control(obj):
        if isinstance(obj, (QAbstractSpinBox, QComboBox)):
            return obj
        if not isinstance(obj, QLineEdit):
            return None
        parent = obj.parentWidget()
        for _ in range(3):
            if isinstance(parent, (QAbstractSpinBox, QComboBox)):
                return parent
            if parent is None:
                return None
            parent = parent.parentWidget()
        return None


def install_wheel_guard(app: QApplication) -> None:
    """Install the guard once for ``app``."""
    for guard in _GUARDS:
        if guard.parent() is app:
            return
    guard = WheelGuard(app)
    app.installEventFilter(guard)
    _GUARDS.append(guard)
