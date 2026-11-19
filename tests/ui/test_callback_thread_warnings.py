"""CallbackList warns when a GUI widget slot would run off the GUI thread."""

import threading
import warnings

import pytest
from PyQt6.QtWidgets import QWidget

from falcocv.core.tool_base import CallbackList

pytestmark = pytest.mark.ui


class _Receiver(QWidget):
    def __init__(self):
        super().__init__()
        self.values = []

    def on_event(self, value):
        self.values.append(value)


def _emit_from_worker(signal, value=1):
    errors = []

    def target():
        try:
            signal.emit(value)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    thread = threading.Thread(target=target, name="signal-worker")
    thread.start()
    thread.join()
    return errors


def test_widget_slot_emitted_off_thread_warns(qtbot):
    receiver = _Receiver()
    qtbot.addWidget(receiver)
    signal = CallbackList()
    signal.connect(receiver.on_event)

    with pytest.warns(RuntimeWarning, match="non-GUI thread"):
        _emit_from_worker(signal, 5)

    assert receiver.values == [5]


def test_widget_slot_on_the_gui_thread_does_not_warn(qtbot):
    receiver = _Receiver()
    qtbot.addWidget(receiver)
    signal = CallbackList()
    signal.connect(receiver.on_event)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        signal.emit(7)

    assert receiver.values == [7]
    assert [w for w in caught if issubclass(w.category, RuntimeWarning)] == []


def test_plain_callbacks_do_not_warn(qtbot):
    signal = CallbackList()
    seen = []
    signal.connect(seen.append)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _emit_from_worker(signal, 3)

    assert seen == [3]
    assert caught == []


def test_strict_mode_raises(qtbot, monkeypatch):
    monkeypatch.setenv("FALCOCV_STRICT_SIGNALS", "1")
    receiver = _Receiver()
    qtbot.addWidget(receiver)
    signal = CallbackList()
    signal.connect(receiver.on_event)

    errors = _emit_from_worker(signal, 1)

    assert len(errors) == 1
    assert isinstance(errors[0], RuntimeError)
    assert "non-GUI thread" in str(errors[0])
    assert receiver.values == []
