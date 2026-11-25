"""Depth Range Loader edits must mark the job as dirty."""

import pytest

from falcocv.tools.depth_range_loader.config import DepthRangeLoaderConfig
from falcocv.tools.depth_range_loader.logic import DepthRangeLoader

pytestmark = pytest.mark.ui


@pytest.mark.parametrize(
    "widget_name, value, param, expected",
    [
        ("spin_scale", 2.5, "scale", 2.5),
        ("spin_xr", 0.5, "x_res", 0.5),
        ("spin_yr", 0.6, "y_res", 0.6),
        ("spin_zr", 0.7, "z_res", 0.7),
        ("spin_zo", -12.5, "z_offset", -12.5),
        ("spin_delay", 250, "delay", 250),
    ],
)
def test_param_change_emits_parameter_changed(
    qtbot, widget_name, value, param, expected
):
    tool = DepthRangeLoader()
    config = DepthRangeLoaderConfig(tool)
    qtbot.addWidget(config)

    seen = []
    tool.parameter_changed.connect(lambda: seen.append(True))

    getattr(config, widget_name).setValue(value)

    assert seen, f"{widget_name} did not emit parameter_changed"
    assert tool.params[param] == pytest.approx(expected)


def test_scale_change_marks_the_main_window_modified(main_window, qtbot):
    window = main_window
    window.editor.handle_add_tool("DepthRangeLoader")
    qtbot.wait(20)
    # The toolbox drop normally connects the new tool to dirty tracking;
    # calling handle_add_tool directly in a test bypasses that signal.
    window._connect_tool_signals()

    tool = next(
        t for t in window.workflow.tools
        if type(t).__name__ == "DepthRangeLoader"
    )
    config = tool.get_config_widget()
    qtbot.addWidget(config)

    window._modified = False
    config.spin_scale.setValue(2.5)

    assert window._modified is True
    assert tool.params["scale"] == pytest.approx(2.5)
