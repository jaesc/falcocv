"""The Python Script debugger must never touch Qt from a worker thread."""

import pytest

from falcocv.tools.python_script import config as config_module
from falcocv.tools.python_script.config import PythonScriptConfig, _qt_wait
from falcocv.tools.python_script.logic import PythonScriptTool

pytestmark = pytest.mark.ui


def test_pause_updates_the_editor_on_the_gui_thread(qtbot):
    tool = PythonScriptTool("Script")
    config = PythonScriptConfig(tool)
    qtbot.addWidget(config)

    config._on_debugger_paused(3, {"value": 42})

    assert config.editor._debug_line == 2


def test_pause_resume_round_trip_does_not_crash(qtbot):
    tool = PythonScriptTool("Script")
    config = PythonScriptConfig(tool)
    qtbot.addWidget(config)

    config._on_debugger_paused(2, {})
    config._on_debugger_resumed()

    assert config.editor._debug_line == -1


class _Event:
    def __init__(self, answers):
        self._answers = list(answers)
        self.calls = 0

    def wait(self, timeout=None):
        self.calls += 1
        if not self._answers:
            return True
        return self._answers.pop(0)


def test_qt_wait_blocks_without_pumping_on_a_worker_thread(monkeypatch):
    pumped = []

    class _FakeThread:
        @staticmethod
        def currentThread():
            return "worker"

    class _FakeApp:
        def thread(self):
            return "gui"

    class _FakeApplication:
        @staticmethod
        def instance():
            return _FakeApp()

        @staticmethod
        def processEvents(*args, **kwargs):
            pumped.append(True)

    monkeypatch.setattr(config_module, "QThread", _FakeThread)
    monkeypatch.setattr(config_module, "QApplication", _FakeApplication)

    event = _Event([])
    _qt_wait(event)

    assert event.calls == 1
    assert pumped == []


def test_qt_wait_pumps_events_on_the_gui_thread(monkeypatch):
    pumped = []

    class _FakeThread:
        @staticmethod
        def currentThread():
            return "gui"

    class _FakeApp:
        def thread(self):
            return "gui"

    class _FakeApplication:
        @staticmethod
        def instance():
            return _FakeApp()

        @staticmethod
        def processEvents(*args, **kwargs):
            pumped.append(True)

    monkeypatch.setattr(config_module, "QThread", _FakeThread)
    monkeypatch.setattr(config_module, "QApplication", _FakeApplication)

    event = _Event([False, True])
    _qt_wait(event)

    assert event.calls == 2
    assert pumped == [True]
