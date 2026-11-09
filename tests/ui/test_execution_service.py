"""The Qt execution service must keep the GUI thread free while tools run."""

import time

import pytest
from PyQt6.QtCore import QThread, QTimer
from PyQt6.QtWidgets import QApplication

from falcocv.core.tool_base import BaseTool
from falcocv.core.workflow import Workflow
from falcocv.core.workflow_controller import WorkflowController
from falcocv.ui.execution_service import ExecutionService

pytestmark = pytest.mark.ui


class SlowTool(BaseTool):
    def __init__(self, delay=0.3):
        super().__init__("slow")
        self.delay = delay
        self.calls = 0
        self.outputs = {"count": 0}

    def run(self, advance=False):
        self.calls += 1
        time.sleep(self.delay)
        self.outputs["count"] = self.calls


class ErrorTool(BaseTool):
    def __init__(self):
        super().__init__("error")
        self.calls = 0

    def run(self, advance=False):
        self.calls += 1
        self.set_error("boom")


def _max_gap_ms(ticks):
    if len(ticks) < 2:
        return 0.0
    return max(b - a for a, b in zip(ticks, ticks[1:])) * 1000.0


def _heartbeat(qtbot, duration_ms=120):
    ticks = []
    timer = QTimer()
    timer.setInterval(16)
    timer.timeout.connect(lambda: ticks.append(time.perf_counter()))
    timer.start()
    qtbot.wait(duration_ms)
    return timer, ticks


def test_workflow_run_does_not_block_the_gui(qtbot):
    service = ExecutionService(WorkflowController(Workflow()))
    try:
        tool = SlowTool()
        service.controller.workflow.add_tool(tool)
        timer, ticks = _heartbeat(qtbot)

        with qtbot.waitSignal(service.workflow_finished, timeout=5000):
            assert service.run_workflow(advance=False)

        qtbot.wait(120)
        timer.stop()
        assert tool.calls == 1
        # The synchronous path blocks for the full 300 ms tool; the worker
        # path measured ~18 ms on the development machine.  Keep a generous
        # CI margin while still catching a regression to the old behaviour.
        assert _max_gap_ms(ticks) < 100.0
    finally:
        service.shutdown()


def test_single_tool_run_does_not_block_the_gui(qtbot):
    service = ExecutionService(WorkflowController(Workflow()))
    try:
        tool = SlowTool()
        timer, ticks = _heartbeat(qtbot)

        with qtbot.waitSignal(service.tool_run_finished, timeout=5000):
            assert service.run_tool(tool)

        qtbot.wait(120)
        timer.stop()
        assert tool.calls == 1
        # The synchronous path blocks for the full 300 ms tool; the worker
        # path measured ~18 ms on the development machine.  Keep a generous
        # CI margin while still catching a regression to the old behaviour.
        assert _max_gap_ms(ticks) < 100.0
    finally:
        service.shutdown()


def test_tool_signals_are_suspended_during_a_run(qtbot):
    service = ExecutionService(WorkflowController(Workflow()))
    try:
        tool = SlowTool(delay=0.1)
        service.controller.workflow.add_tool(tool)
        calls = []

        def on_data_changed():
            calls.append(
                QThread.currentThread() is QApplication.instance().thread()
            )

        tool.data_changed.connect(on_data_changed)
        with qtbot.waitSignal(service.workflow_finished, timeout=5000):
            service.run_workflow(advance=False)

        assert calls == [True]
    finally:
        service.shutdown()


def test_loop_runs_until_stopped(qtbot):
    service = ExecutionService(WorkflowController(Workflow()))
    try:
        tool = SlowTool(delay=0.05)
        service.controller.workflow.add_tool(tool)
        service.start_loop()
        qtbot.waitUntil(lambda: tool.calls >= 3, timeout=5000)
        service.stop_loop()
        qtbot.waitUntil(lambda: not service.is_busy, timeout=5000)
        calls = tool.calls
        qtbot.wait(200)
        assert tool.calls == calls
        assert not service.is_looping
    finally:
        service.shutdown()


def test_loop_stops_on_error_when_configured(qtbot):
    service = ExecutionService(WorkflowController(Workflow()))
    try:
        workflow = service.controller.workflow
        workflow.stop_on_error = True
        tool = ErrorTool()
        workflow.add_tool(tool)
        service.start_loop()
        qtbot.waitUntil(lambda: tool.calls >= 1, timeout=5000)
        qtbot.waitUntil(lambda: not service.is_busy, timeout=5000)
        assert tool.calls == 1
        assert not service.is_looping
    finally:
        service.shutdown()


def test_tool_run_is_rejected_while_busy(qtbot):
    service = ExecutionService(WorkflowController(Workflow()))
    try:
        workflow_tool = SlowTool(delay=0.2)
        service.controller.workflow.add_tool(workflow_tool)
        single_tool = SlowTool(delay=0.05)
        assert service.run_workflow(advance=False)
        assert not service.run_tool(single_tool)
        with qtbot.waitSignal(service.workflow_finished, timeout=5000):
            pass
        with qtbot.waitSignal(service.tool_run_finished, timeout=5000):
            assert service.run_tool(single_tool)
        assert single_tool.calls == 1
    finally:
        service.shutdown()


def test_shutdown_stops_the_worker(qtbot):
    service = ExecutionService(WorkflowController(Workflow()))
    thread = service._thread
    service.shutdown()
    assert not thread.isRunning()
