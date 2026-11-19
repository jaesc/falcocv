"""Tool signals connected to widgets must be thread-dispatched ToolSignals."""

import falcocv.tools  # noqa: F401  (populate the registry)
from falcocv.core import tool_base
from falcocv.core.tool_registry import get_all_tools
from falcocv.tools.gige_camera.logic import GigeCameraLoader


def test_no_tool_shares_a_class_level_callback_list():
    for reg in get_all_tools():
        for name, value in vars(reg.cls).items():
            assert not isinstance(value, tool_base.CallbackList), (
                f"{reg.tool_type}.{name} is a class-level CallbackList shared "
                "by every instance; create a per-instance ToolSignal instead"
            )


def test_gige_connection_signal_is_per_instance_and_dispatched():
    first = GigeCameraLoader("Camera A")
    second = GigeCameraLoader("Camera B")

    assert isinstance(first.connection_changed, tool_base.ToolSignal)
    assert first.connection_changed is not second.connection_changed

    direct = []
    dispatched = []
    first.connection_changed.connect(lambda connected: direct.append(connected))

    def fake_dispatcher(callback, args, kwargs):
        dispatched.append((callback, args, kwargs))

    previous = tool_base.get_tool_signal_dispatcher()
    tool_base.set_tool_signal_dispatcher(fake_dispatcher)
    try:
        first.connection_changed.emit(True)
    finally:
        tool_base.set_tool_signal_dispatcher(previous)

    assert direct == []
    assert dispatched[0][1] == (True,)
