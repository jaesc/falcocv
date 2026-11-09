"""Marshal tool callbacks emitted from worker threads to the Qt GUI thread.

Tools normally emit ``data_changed`` from the thread that executes them.  The
workflow editor runs tools on a background ``QThread``, and a few tools also
emit from their own helper threads (for example the TemplateMatch cache
builder).  Qt widgets must only be touched from the GUI thread.

The dispatcher is installed once, after ``QApplication`` exists.  Headless
callers never install it, so ``ToolSignal`` keeps its direct behaviour.
"""

from __future__ import annotations

import traceback

from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal, pyqtSlot
from PyQt6.QtWidgets import QApplication

from falcocv.core.tool_base import set_tool_signal_dispatcher


class ToolSignalDispatcher(QObject):
    """Call callbacks on the GUI thread, coalescing nothing by itself."""

    _dispatch = pyqtSignal(object, tuple, dict)

    def __init__(self):
        super().__init__()
        self._dispatch.connect(
            self._run, Qt.ConnectionType.QueuedConnection
        )

    def __call__(self, callback, args, kwargs):
        app = QApplication.instance()
        if app is not None and QThread.currentThread() is app.thread():
            self._call(callback, args, kwargs)
            return
        self._dispatch.emit(callback, args, kwargs)

    @pyqtSlot(object, tuple, dict)
    def _run(self, callback, args, kwargs):
        self._call(callback, args, kwargs)

    @staticmethod
    def _call(callback, args, kwargs):
        try:
            callback(*args, **kwargs)
        except Exception:
            # A failing preview must never abort the Qt event loop.
            traceback.print_exc()


_dispatcher: ToolSignalDispatcher | None = None


def install_tool_signal_dispatcher() -> ToolSignalDispatcher:
    """Create and install the process-wide GUI dispatcher."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ToolSignalDispatcher()
        set_tool_signal_dispatcher(_dispatcher)
    return _dispatcher


def uninstall_tool_signal_dispatcher() -> None:
    """Clear the dispatcher (used by tests and shutdown)."""
    global _dispatcher
    set_tool_signal_dispatcher(None)
    _dispatcher = None
