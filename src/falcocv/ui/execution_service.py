"""Background workflow and single-tool execution for the Qt UI.

The service owns one worker thread.  Every workflow run, loop iteration and
single-tool Auto run is serialised through it, so the GUI thread never executes
a tool and two executions can never touch the same tool at the same time.

Tool ``data_changed`` signals are suspended for the duration of a run.  They
are resumed on the GUI thread when the run finishes, which refreshes open
config dialogs once instead of once per tool from a worker thread.
"""

from __future__ import annotations

import sys
import threading
import traceback

from PyQt6.QtCore import (
    QEventLoop,
    QObject,
    QThread,
    QTimer,
    Qt,
    pyqtSignal,
    pyqtSlot,
)

from falcocv.core.workflow_controller import WorkflowController


#: Services whose worker outlived the normal shutdown timeout.  Keeping them
#: referenced prevents a QThread destructor from aborting while it still runs.
_ORPHANED_SERVICES: list = []

#: One workflow/tool execution at a time across all editors.  A Block config
#: has its own editor and worker; without this lock the main loop and a nested
#: tool could touch the same nested tools concurrently.
_RUN_LOCK = threading.Lock()


def _tool_tree(tool) -> list:
    """Return ``tool`` plus every tool inside a Block."""
    tools = [tool]
    child = getattr(tool, "workflow", None)
    if child is not None:
        tools.extend(child.iter_tools_recursive())
    return tools


def _suspend_tools(tools) -> None:
    for tool in tools:
        for name in ("data_changed", "structure_changed"):
            signal = getattr(tool, name, None)
            if signal is not None:
                signal.suspend()


def _resume_tools(tools, emit_pending: bool = True) -> None:
    for tool in tools:
        for name in ("data_changed", "structure_changed"):
            signal = getattr(tool, name, None)
            if signal is not None:
                signal.resume(emit_pending=emit_pending)


class _ExecutionWorker(QObject):
    workflow_started = pyqtSignal()
    tool_started = pyqtSignal(object)
    tool_finished = pyqtSignal(object)
    workflow_finished = pyqtSignal(object)
    tool_run_finished = pyqtSignal(object, str)

    def __init__(self):
        super().__init__()
        self._cancel = threading.Event()

    def _observe(self, event, tool):
        if self._cancel.is_set():
            return
        if event == "tool_started":
            self.tool_started.emit(tool)
        elif event == "tool_finished":
            self.tool_finished.emit(tool)

    @pyqtSlot(object, bool)
    def run_workflow(self, controller, advance):
        tools = []
        payload = {
            "total_time_ms": 0.0,
            "error": "Execution failed before the workflow started.",
            "tools": tools,
        }
        try:
            workflow = getattr(controller, "workflow", None)
            tools = (
                list(workflow.iter_tools_recursive())
                if workflow is not None
                else []
            )
            payload["tools"] = tools
            self.workflow_started.emit()
            with _RUN_LOCK:
                if self._cancel.is_set():
                    payload["error"] = "Execution cancelled"
                else:
                    _suspend_tools(tools)
                    controller.run(advance=advance, observer=self._observe)
                    payload = {
                        "total_time_ms": controller.last_execution_time_ms,
                        "error": controller.last_error,
                        "tools": tools,
                    }
        except Exception as exc:
            traceback.print_exc()
            payload = {
                "total_time_ms": 0.0,
                "error": f"{type(exc).__name__}: {exc}",
                "tools": tools,
            }
        self.workflow_finished.emit(payload)

    @pyqtSlot(object)
    def run_tool(self, tool):
        error = ""
        try:
            tools = _tool_tree(tool)
            with _RUN_LOCK:
                if self._cancel.is_set():
                    error = "Execution cancelled"
                else:
                    _suspend_tools(tools)
                    tool.execute()
                    error = tool.last_error or ""
        except Exception as exc:
            traceback.print_exc()
            error = f"{type(exc).__name__}: {exc}"
        self.tool_run_finished.emit(tool, error)

class ExecutionService(QObject):
    """Qt-side coordinator for background workflow and tool runs."""

    workflow_started = pyqtSignal()
    tool_started = pyqtSignal(object)
    tool_finished = pyqtSignal(object)
    workflow_finished = pyqtSignal(object)
    tool_run_finished = pyqtSignal(object, str)
    busy_changed = pyqtSignal(bool)

    _request_workflow = pyqtSignal(object, bool)
    _request_tool = pyqtSignal(object)

    def __init__(self, controller: WorkflowController, parent=None):
        super().__init__(parent)
        self.controller = controller
        self._busy = False
        self._finishing = False
        self._loop = False
        self._shutdown = False
        self._active_tools: list = []

        self._thread = QThread(self)
        self._thread.setObjectName("FalcoCV-Execution")
        self._worker = _ExecutionWorker()
        self._worker.moveToThread(self._thread)

        self._request_workflow.connect(
            self._worker.run_workflow, Qt.ConnectionType.QueuedConnection
        )
        self._request_tool.connect(
            self._worker.run_tool, Qt.ConnectionType.QueuedConnection
        )

        self._worker.workflow_started.connect(self._on_workflow_started)
        self._worker.tool_started.connect(self.tool_started)
        self._worker.tool_finished.connect(self.tool_finished)
        self._worker.workflow_finished.connect(self._on_workflow_finished)
        self._worker.tool_run_finished.connect(self._on_tool_run_finished)

        self._thread.start()

    # -- state ---------------------------------------------------------
    @property
    def is_busy(self) -> bool:
        return self._busy

    @property
    def is_looping(self) -> bool:
        return self._loop

    @property
    def is_finishing(self) -> bool:
        """True while run-finished callbacks are being replayed on the UI."""
        return self._finishing

    def _set_busy(self, busy: bool) -> None:
        if self._busy == busy:
            return
        self._busy = busy
        self.busy_changed.emit(busy)

    # -- commands ------------------------------------------------------
    def run_workflow(self, advance: bool = False) -> bool:
        if self._shutdown or self._busy:
            return False
        try:
            self._active_tools = list(
                self.controller.workflow.iter_tools_recursive()
            )
        except Exception:
            traceback.print_exc()
            self._active_tools = []
        self._set_busy(True)
        self._request_workflow.emit(self.controller, bool(advance))
        return True

    def run_tool(self, tool) -> bool:
        if self._shutdown or self._busy:
            # ToolRunControls owns the pending request and retries when idle.
            return False
        try:
            self._active_tools = _tool_tree(tool)
        except Exception:
            traceback.print_exc()
            self._active_tools = [tool]
        self._set_busy(True)
        self._request_tool.emit(tool)
        return True

    def start_loop(self) -> None:
        if self._shutdown:
            return
        self._loop = True
        if not self._busy:
            self.run_workflow(advance=True)

    def stop_loop(self) -> None:
        self._loop = False

    def _loop_tick(self) -> None:
        if self._loop and not self._busy:
            self.run_workflow(advance=True)

    def wait_idle(self, timeout_ms: int = 5000) -> bool:
        """Process events until the current run finishes, or time out."""
        if not self._busy:
            return True
        loop = QEventLoop()
        timer = QTimer()
        timer.setSingleShot(True)
        timer.timeout.connect(loop.quit)

        def on_busy(busy):
            if not busy:
                loop.quit()

        self.busy_changed.connect(on_busy)
        timer.start(timeout_ms)
        try:
            loop.exec()
        finally:
            timer.stop()
            try:
                self.busy_changed.disconnect(on_busy)
            except (TypeError, RuntimeError):
                pass
        return not self._busy

    # -- worker events -------------------------------------------------
    @pyqtSlot()
    def _on_workflow_started(self):
        self.workflow_started.emit()

    @pyqtSlot(object)
    def _on_workflow_finished(self, payload):
        self._finishing = True
        try:
            try:
                _resume_tools(payload.get("tools", []), emit_pending=True)
            except Exception:
                traceback.print_exc()
        finally:
            self._finishing = False
            self._active_tools = []
            self._set_busy(False)

        error = payload.get("error")
        workflow = self.controller.workflow
        if self._loop and error and getattr(workflow, "stop_on_error", False):
            self._loop = False

        self.workflow_finished.emit(payload)
        if self._loop:
            QTimer.singleShot(0, self._loop_tick)

    @pyqtSlot(object, str)
    def _on_tool_run_finished(self, tool, error):
        self._finishing = True
        try:
            try:
                _resume_tools(self._active_tools, emit_pending=True)
            except Exception:
                traceback.print_exc()
        finally:
            self._finishing = False
            self._active_tools = []
            self._set_busy(False)

        self.tool_run_finished.emit(tool, error)
        if self._loop:
            QTimer.singleShot(0, self._loop_tick)

    # -- lifecycle -----------------------------------------------------
    def shutdown(self, timeout_ms: int = 5000) -> bool:
        """Stop the loop and wait for the worker before the window closes.

        Returns ``True`` when the thread stopped.  If a tool is still blocking,
        the service is detached instead of deleting a running ``QThread``.
        """
        self._shutdown = True
        self._loop = False
        # A running tool cannot be interrupted safely; the flag stops the
        # observer from emitting more progress events.
        self._worker._cancel.set()
        self._thread.quit()
        if not self._thread.wait(timeout_ms):
            print(
                f"[ExecutionService] worker did not stop within {timeout_ms} ms",
                file=sys.stderr,
            )
            self.setParent(None)
            self._thread.setParent(None)
            _ORPHANED_SERVICES.append(self)
            return False
        # A cancelled run may have left a signal suspended.
        _resume_tools(self._active_tools, emit_pending=False)
        self._active_tools = []
        return True
