from PyQt6.QtCore import QTimer
from falcocv.core.workflow_controller import WorkflowController, ControllerState


class ControllerAdapter:
    """Qt adapter for the pure-Python WorkflowController.

    Wraps the controller, adding QTimer-based loop support
    and translating CallbackList signals to Qt-friendly patterns.
    """

    def __init__(self, workflow=None):
        self._controller = WorkflowController(workflow)
        self._loop_timer = QTimer()
        self._loop_timer.timeout.connect(self._on_loop_tick)
        self._controller.set_loop_tick_callback(self._on_loop_tick)

    @property
    def controller(self):
        return self._controller

    @property
    def workflow(self):
        return self._controller.workflow

    @property
    def state(self):
        return self._controller.state

    def mark_dirty(self, tool=None):
        self._controller.mark_dirty(tool)

    def run(self, advance=False):
        self._controller.run(advance=advance)

    def _on_loop_tick(self):
        self._controller.run(advance=True)

    def run_loop_start(self, interval_ms):
        self._loop_timer.start(interval_ms)

    def run_loop_stop(self):
        self._loop_timer.stop()

    def is_looping(self):
        return self._loop_timer.isActive()

    def update_loop_interval(self, interval_ms):
        if self._loop_timer.isActive():
            self._loop_timer.setInterval(interval_ms)
