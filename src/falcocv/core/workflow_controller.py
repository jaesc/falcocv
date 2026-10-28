from enum import Enum
from falcocv.core.workflow import Workflow
from falcocv.core.tool_base import CallbackList


class ControllerState(Enum):
    IDLE = "idle"
    DIRTY = "dirty"
    RUNNING = "running"
    LOOPING = "looping"


class WorkflowController:
    def __init__(self, workflow=None):
        self.workflow = workflow or Workflow()
        self._state = ControllerState.IDLE
        self._running = False
        self._looping = False
        self._schedule = None
        self.last_execution_time_ms = 0.0
        self.last_error = None
        self.state_changed = CallbackList()
        self.execution_started = CallbackList()
        self.execution_finished = CallbackList()

    @property
    def state(self):
        return self._state

    def set_scheduler(self, schedule_fn):
        """schedule_fn(callback) must execute callback() as soon as possible
        without blocking. Qt: QTimer.singleShot(0, callback).
        asyncio: loop.call_soon(callback).
        CLI: callback() directly (sync)."""
        self._schedule = schedule_fn

    def mark_dirty(self, tool=None):
        if self._state == ControllerState.IDLE:
            self._set_state(ControllerState.DIRTY)

    def run(self, advance=False):
        if self._running:
            return
        self._running = True
        self._set_state(ControllerState.RUNNING)
        self.execution_started.emit()
        try:
            self.workflow.run(advance=advance)
            self.last_execution_time_ms = self.workflow.last_execution_time_ms
            self.last_error = self.workflow.last_error
        finally:
            self._running = False
            self._set_state(ControllerState.IDLE)
            self.execution_finished.emit({
                "total_time_ms": self.last_execution_time_ms,
                "error": self.last_error,
            })
            if self._looping and self._schedule:
                self._schedule(lambda: self.run(advance=True))

    def run_loop_start(self):
        self._looping = True
        if self._schedule:
            self._schedule(lambda: self.run(advance=True))

    def run_loop_stop(self):
        self._looping = False

    def is_looping(self):
        return self._looping

    def _set_state(self, state):
        if self._state != state:
            self._state = state
            self.state_changed.emit(state)
