"""Tool callbacks emitted from worker threads must reach the GUI thread."""

import threading

import pytest
from PyQt6.QtCore import QThread
from PyQt6.QtWidgets import QApplication

from falcocv.core.tool_base import ToolSignal

pytestmark = pytest.mark.ui


def test_signal_from_worker_thread_is_marshalled_to_gui(qtbot):
    signal = ToolSignal()
    seen = []
    signal.connect(
        lambda: seen.append(
            QThread.currentThread() is QApplication.instance().thread()
        )
    )

    worker = threading.Thread(target=signal.emit)
    worker.start()
    worker.join(timeout=2)
    qtbot.waitUntil(lambda: bool(seen), timeout=2000)

    assert seen == [True]


def test_suspended_signal_is_coalesced_on_resume(qtbot):
    signal = ToolSignal()
    seen = []
    signal.connect(lambda: seen.append(1))

    signal.suspend()
    signal.emit()
    signal.emit()
    assert seen == []

    signal.resume()
    assert seen == [1]
