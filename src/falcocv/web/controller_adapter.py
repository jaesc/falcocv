import asyncio
from falcocv.core.workflow_controller import WorkflowController, ControllerState
from falcocv.core.tool_base import CallbackList


class WebControllerAdapter:
    def __init__(self, workflow=None):
        self._controller = WorkflowController(workflow)
        self._loop = None
        self._ws_callbacks = CallbackList()
        self._controller.state_changed.connect(self._on_state_changed)
        self._controller.execution_finished.connect(self._on_execution_finished)

    def start(self):
        self._loop = asyncio.get_running_loop()
        self._controller.set_scheduler(lambda cb: self._loop.call_soon(cb))

    @property
    def controller(self):
        return self._controller

    @property
    def workflow(self):
        return self._controller.workflow

    @property
    def state(self):
        return self._controller.state

    def set_ws_broadcast(self, broadcast_fn):
        self._ws_callbacks.connect(broadcast_fn)

    def _on_state_changed(self, state):
        self._ws_callbacks.emit({"type": "state_changed", "state": state.value})

    def _on_execution_finished(self, data):
        self._ws_callbacks.emit({
            "type": "execution_finished",
            "state": self._controller.state.value,
            "total_time_ms": data.get("total_time_ms", 0),
            "error": data.get("error"),
        })

    def mark_dirty(self, tool=None):
        self._controller.mark_dirty(tool)

    def run(self, advance=False):
        self._controller.run(advance=advance)

    def run_loop_start(self):
        self._controller.run_loop_start()

    def run_loop_stop(self):
        self._controller.run_loop_stop()

    def is_looping(self):
        return self._controller.is_looping()
