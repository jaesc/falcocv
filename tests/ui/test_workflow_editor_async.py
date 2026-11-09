"""WorkflowEditor must delegate runs to the background execution service."""

import time

import pytest

import falcocv.tools  # noqa: F401  (populate the tool registry)
from falcocv.core.tool_base import BaseTool
from falcocv.core.workflow import Workflow
from falcocv.ui.widgets.workflow_editor import WorkflowEditor

pytestmark = pytest.mark.ui


class SlowTool(BaseTool):
    def __init__(self, delay=0.2):
        super().__init__("slow")
        self.delay = delay
        self.calls = 0
        self.observed = []

    def run(self, advance=False):
        self.calls += 1
        self.observed.append(self.last_execution_time_ms)
        time.sleep(self.delay)


def _editor(qtbot, workflow):
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)
    editor.resize(1000, 700)
    editor.show()
    qtbot.wait(30)
    return editor


def test_run_workflow_is_asynchronous(qtbot):
    workflow = Workflow()
    tool = SlowTool()
    workflow.add_tool(tool)
    editor = _editor(qtbot, workflow)

    with qtbot.waitSignal(editor.engine.workflow_finished, timeout=5000):
        editor.run_workflow(advance=False)
        assert not editor.btn_refresh.isEnabled()
        assert not editor.btn_step.isEnabled()
    qtbot.wait(30)

    assert tool.calls == 1
    assert editor.btn_refresh.isEnabled()
    assert editor.btn_step.isEnabled()


def test_parameter_editing_stays_enabled_while_running(qtbot):
    workflow = Workflow()
    workflow.add_tool(SlowTool(delay=0.2))
    editor = _editor(qtbot, workflow)

    with qtbot.waitSignal(editor.engine.workflow_finished, timeout=5000):
        editor.run_workflow(advance=False)
        # Only the run buttons are disabled; the tree and parameter panels
        # remain editable while the worker runs.
        assert editor.tree.isEnabled()
        assert editor.btn_loop.isEnabled()
        assert editor.splitter.isEnabled()


class FailOnceTool(BaseTool):
    def __init__(self, delay=0.05):
        super().__init__("fail_once")
        self.delay = delay
        self.calls = 0

    def run(self, advance=False):
        self.calls += 1
        time.sleep(self.delay)
        if self.calls >= 2:
            self.set_error("boom")


def test_error_keeps_unexecuted_tools_empty_after_the_run(qtbot):
    workflow = Workflow()
    workflow.stop_on_error = True
    failing = FailOnceTool()
    below = SlowTool(delay=0.05)
    workflow.add_tool(failing)
    workflow.add_tool(below)
    editor = _editor(qtbot, workflow)
    below_item = editor.tree.topLevelItem(1)

    with qtbot.waitSignal(editor.engine.workflow_finished, timeout=5000):
        editor.run_workflow(advance=False)
    qtbot.wait(30)
    assert below_item.text(1).endswith("ms")

    with qtbot.waitSignal(editor.engine.workflow_finished, timeout=5000):
        editor.run_workflow(advance=False)
        qtbot.waitUntil(
            lambda: editor.tree.is_status_frozen and failing.calls >= 2,
            timeout=3000,
        )
        # Previous value is still visible while the run is in progress.
        assert below_item.text(1).endswith("ms")

    qtbot.wait(30)
    # The tool below the error never ran in the second pass, so its real
    # state is empty and the UI reflects that once the run has finished.
    assert below.last_execution_time_ms == 0.0
    assert below_item.text(1) == ""


def test_previous_tool_time_stays_visible_during_run(qtbot):
    workflow = Workflow()
    tool = SlowTool(delay=0.1)
    workflow.add_tool(tool)
    editor = _editor(qtbot, workflow)
    item = editor.tree.topLevelItem(0)

    with qtbot.waitSignal(editor.engine.workflow_finished, timeout=5000):
        editor.run_workflow(advance=False)
    qtbot.wait(30)
    first = item.text(1)
    assert first.endswith("ms")

    tool.delay = 0.25
    with qtbot.waitSignal(editor.engine.workflow_finished, timeout=5000):
        editor.run_workflow(advance=False)
        qtbot.waitUntil(
            lambda: editor.tree.is_status_frozen and len(tool.observed) >= 2,
            timeout=3000,
        )
        # The live state is cleared at the start of the run...
        assert tool.last_execution_time_ms == 0.0
        # ...but the tree still shows the previous result.
        assert item.text(1) == first
        editor.tree.refresh_status(force=True)
        assert item.text(1) == first

    qtbot.wait(30)
    second = item.text(1)
    assert second.endswith("ms")
    assert second != first


def test_loop_never_clears_tool_times(qtbot):
    workflow = Workflow()
    tool = SlowTool(delay=0.1)
    workflow.add_tool(tool)
    editor = _editor(qtbot, workflow)
    item = editor.tree.topLevelItem(0)

    with qtbot.waitSignal(editor.engine.workflow_finished, timeout=5000):
        editor.run_workflow(advance=False)
    qtbot.wait(30)
    assert item.text(1).endswith("ms")

    editor.toggle_loop(True)
    try:
        deadline = time.perf_counter() + 0.6
        while time.perf_counter() < deadline:
            qtbot.wait(20)
            assert item.text(1).endswith("ms"), item.text(1)
    finally:
        editor.toggle_loop(False)
        qtbot.waitUntil(lambda: not editor.engine.is_busy, timeout=5000)
    assert item.text(1).endswith("ms")


def test_structural_edit_is_blocked_while_running(qtbot):
    workflow = Workflow()
    workflow.add_tool(SlowTool(delay=0.3))
    editor = _editor(qtbot, workflow)
    before = len(workflow.tools)

    editor.run_workflow(advance=False)
    editor.handle_add_tool("ImageLoader")
    assert len(workflow.tools) == before
    assert "Stop the current run" in editor.label_error.text()

    qtbot.waitUntil(lambda: not editor.engine.is_busy, timeout=5000)
    editor.handle_add_tool("ImageLoader")
    assert len(workflow.tools) == before + 1


def test_set_workflow_replaces_and_stops_the_engine(qtbot):
    editor = _editor(qtbot, Workflow())
    old_engine = editor.engine
    old_thread = old_engine._thread
    editor.tree.set_status_frozen(True)

    editor.set_workflow(Workflow())

    assert editor.engine is not old_engine
    assert not old_thread.isRunning()
    assert not editor.tree.is_status_frozen


def test_shutdown_stops_the_engine(qtbot):
    editor = _editor(qtbot, Workflow())
    thread = editor.engine._thread
    editor.shutdown()
    assert not thread.isRunning()
