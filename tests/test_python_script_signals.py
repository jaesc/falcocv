"""Python Script debugger signals must be per-instance and thread-dispatched."""

from falcocv.core import tool_base
from falcocv.tools.python_script.logic import PythonScriptTool


def test_debugger_signals_are_tool_signals_and_per_instance():
    first = PythonScriptTool("Script A")
    second = PythonScriptTool("Script B")

    assert isinstance(first.debugger_paused, tool_base.ToolSignal)
    assert isinstance(first.debugger_resumed, tool_base.ToolSignal)
    assert first.debugger_paused is not second.debugger_paused
    assert first.debugger_resumed is not second.debugger_resumed


def test_debugger_signals_use_the_installed_dispatcher():
    tool = PythonScriptTool("Script")
    direct = []
    dispatched = []
    tool.debugger_paused.connect(lambda line, locs: direct.append((line, locs)))
    tool.debugger_resumed.connect(lambda: direct.append("resumed"))

    def fake_dispatcher(callback, args, kwargs):
        dispatched.append((callback, args, kwargs))

    previous = tool_base.get_tool_signal_dispatcher()
    tool_base.set_tool_signal_dispatcher(fake_dispatcher)
    try:
        tool.debugger_paused.emit(7, {"x": 1})
        tool.debugger_resumed.emit()
    finally:
        tool_base.set_tool_signal_dispatcher(previous)

    assert direct == []
    assert dispatched[0][1] == (7, {"x": 1})
    assert dispatched[0][2] == {}
    assert dispatched[1][1] == ()
