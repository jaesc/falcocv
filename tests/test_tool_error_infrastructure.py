"""Tool-level error infrastructure and stop-on-error behaviour."""

from falcocv.core.tool_base import BaseTool
from falcocv.core.workflow import Workflow
from falcocv.core.workflow_controller import WorkflowController


class DummyTool(BaseTool):
    def __init__(self, name, fail=False):
        super().__init__(name)
        self.fail = fail
        self.executions = 0

    def run(self, advance=False):
        self.executions += 1
        if self.fail:
            self.set_error("boom")


def test_tool_error_survives_execute_and_clears_on_success():
    tool = DummyTool("A", fail=True)

    tool.execute()
    assert tool.last_error == "boom"

    tool.fail = False
    tool.execute()
    assert tool.last_error is None


def test_workflow_stops_on_first_tool_error():
    first = DummyTool("A", fail=True)
    second = DummyTool("B")
    workflow = Workflow()
    workflow.add_tool(first)
    workflow.add_tool(second)
    workflow.stop_on_error = True

    workflow.run()

    assert first.executions == 1
    assert second.executions == 0
    assert first.last_error == "boom"
    assert workflow.last_error == "A: boom"


def test_workflow_continues_without_stop_on_error():
    first = DummyTool("A", fail=True)
    second = DummyTool("B")
    workflow = Workflow()
    workflow.add_tool(first)
    workflow.add_tool(second)
    workflow.stop_on_error = False

    workflow.run()

    assert first.executions == 1
    assert second.executions == 1
    assert second.last_error is None
    assert workflow.last_error == "A: boom"


def test_workflow_keeps_the_first_error_when_tools_keep_failing():
    first = DummyTool("A", fail=True)
    second = DummyTool("B", fail=True)
    workflow = Workflow()
    workflow.add_tool(first)
    workflow.add_tool(second)
    workflow.stop_on_error = False

    workflow.run()

    assert first.last_error == "boom"
    assert second.last_error == "boom"
    assert workflow.last_error == "A: boom"


def test_exception_uses_the_same_error_channel():
    class RaisingTool(DummyTool):
        def run(self, advance=False):
            self.executions += 1
            raise ValueError("bad input")

    tool = RaisingTool("A")
    workflow = Workflow()
    workflow.add_tool(tool)
    workflow.stop_on_error = True

    workflow.run()

    assert tool.executions == 1
    assert tool.last_error == "ValueError: bad input"
    assert workflow.last_error == "A: ValueError: bad input"


def test_stop_on_error_survives_serialization():
    workflow = Workflow()
    workflow.stop_on_error = True
    data = workflow.serialize()

    restored = Workflow()
    restored.deserialize(data)

    assert restored.stop_on_error is True


def test_run_clears_status_of_unexecuted_tools():
    first = DummyTool("A", fail=True)
    second = DummyTool("B")
    workflow = Workflow()
    workflow.add_tool(first)
    workflow.add_tool(second)
    workflow.stop_on_error = True
    second.last_execution_time_ms = 50.0
    second.last_error = "old error"

    workflow.run()

    assert first.executions == 1
    assert second.executions == 0
    assert first.last_error == "boom"
    assert second.last_execution_time_ms == 0.0
    assert second.last_error is None


def test_loop_stops_on_error_when_stop_on_error_is_set():
    tool = DummyTool("A", fail=True)
    workflow = Workflow()
    workflow.add_tool(tool)
    workflow.stop_on_error = True
    controller = WorkflowController(workflow)
    scheduled = []
    controller.set_scheduler(scheduled.append)

    controller.run_loop_start()
    while scheduled:
        scheduled.pop(0)()

    assert tool.executions == 1
    assert controller.is_looping() is False
    assert scheduled == []


def test_disabled_tool_clears_its_error():
    tool = DummyTool("A", fail=True)
    workflow = Workflow()
    workflow.add_tool(tool)
    tool.execute()
    assert tool.last_error == "boom"

    tool.enabled = False
    workflow.run()

    assert tool.last_error is None
    assert tool.last_execution_time_ms == 0.0
