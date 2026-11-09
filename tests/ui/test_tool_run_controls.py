"""Run / Auto behavior in tool config headers."""

import time

import pytest

pytestmark = pytest.mark.ui

from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import create_tool
from falcocv.core.workflow import Workflow
from falcocv.core.workflow_controller import WorkflowController
from falcocv.ui.execution_service import ExecutionService
from falcocv.ui.widgets.tool_config_dialog import ToolConfigDialog
from falcocv.ui.widgets.tool_run_controls import ToolRunControls


def _open_config(qtbot, tool_type="BlobDetector"):
    tool = create_tool(tool_type)
    dialog = ToolConfigDialog(tool)
    qtbot.addWidget(dialog)
    dialog.resize(1200, 800)
    dialog.show()
    qtbot.wait(50)
    controls = dialog._config_widget.findChild(ToolRunControls)
    assert controls is not None
    return tool, dialog, controls


def test_run_button_executes_once(qtbot):
    tool, dialog, controls = _open_config(qtbot)
    calls = []
    tool.execute = lambda *args, **kwargs: calls.append(1)

    assert controls.run_button.text() == "RUN"
    assert controls.auto_button.text() == "AUTO"
    assert "QPushButton:checked" in controls.auto_button.styleSheet()
    assert (
        controls.auto_button.fontMetrics().horizontalAdvance("AUTO ON")
        <= controls.auto_button.width()
    )

    controls.run_button.click()
    assert calls == [1]

    dialog.close()


def test_auto_runs_once_after_debounce(qtbot):
    tool, dialog, controls = _open_config(qtbot)
    calls = []
    tool.execute = lambda *args, **kwargs: calls.append(1)

    controls.auto_button.setChecked(True)
    qtbot.wait(20)
    assert calls == [1]  # enabling Auto runs once immediately
    assert controls.auto_button.text() == "AUTO ON"

    tool.mark_dirty()
    tool.mark_dirty()
    qtbot.wait(300)
    assert calls == [1, 1]

    controls.auto_button.setChecked(False)
    tool.mark_dirty()
    qtbot.wait(300)
    assert calls == [1, 1]

    dialog.close()


def test_run_button_uses_the_background_service(qtbot):
    class SlowTool(BaseTool):
        def __init__(self):
            super().__init__("slow")

        def run(self, advance=False):
            time.sleep(0.1)

    service = ExecutionService(WorkflowController(Workflow()))
    try:
        tool = SlowTool()
        tool._execution_service = service
        controls = ToolRunControls(tool)
        qtbot.addWidget(controls)
        controls.show()

        with qtbot.waitSignal(service.tool_run_finished, timeout=5000):
            controls.run_button.click()
            assert not controls.run_button.isEnabled()
        assert controls.run_button.isEnabled()
        controls.shutdown()
    finally:
        service.shutdown()


def test_closing_dialog_disconnects_auto(qtbot):
    tool, dialog, controls = _open_config(qtbot)
    calls = []
    tool.execute = lambda *args, **kwargs: calls.append(1)

    controls.auto_button.setChecked(True)
    qtbot.wait(20)
    assert calls == [1]

    dialog.close()
    tool.mark_dirty()
    qtbot.wait(300)
    assert calls == [1]
