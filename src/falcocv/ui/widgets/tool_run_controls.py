"""Shared per-tool Run / Auto controls for tool config dialogs."""

from __future__ import annotations

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QWidget

from falcocv.ui.style import HIGHLIGHT, PANEL_BG, SUCCESS


class ToolRunControls(QWidget):
    """Run once, or debounce-run after the tool's settings change.

    The controls execute only the owning tool.  They do not run upstream or
    downstream tools, so they are intended for tuning a single tool against
    its current inputs.
    """

    DEBOUNCE_MS = 250

    def __init__(self, tool, parent=None):
        super().__init__(parent)
        self.tool = tool
        self._shutdown = False
        self._executing = False
        self._pending = False
        self._service = None
        self._service_connected = False

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.run_button = QPushButton("RUN", self)
        self.run_button.setFixedWidth(50)
        self.run_button.setFixedHeight(22)
        self.run_button.setStyleSheet(
            f"font-weight: bold; color: {SUCCESS};"
        )
        self.run_button.setToolTip("Run this tool now with its current inputs.")
        self.run_button.clicked.connect(self.run_once)
        layout.addWidget(self.run_button)

        self.auto_button = QPushButton("AUTO", self)
        self.auto_button.setCheckable(True)
        self.auto_button.setFixedHeight(22)
        self.auto_button.setStyleSheet(
            f"QPushButton {{ font-weight: bold; color: {SUCCESS}; }} "
            f"QPushButton:checked {{ background-color: {PANEL_BG}; "
            f"border: 1px solid {HIGHLIGHT}; color: {SUCCESS}; }}"
        )
        auto_text_width = self.auto_button.fontMetrics().horizontalAdvance("AUTO ON")
        self.auto_button.setFixedWidth(max(64, auto_text_width + 20))
        self._set_auto_tooltip(False)
        self.auto_button.toggled.connect(self._on_auto_toggled)
        layout.addWidget(self.auto_button)

        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(self.DEBOUNCE_MS)
        self._timer.timeout.connect(self._on_timer)

        self.tool.parameter_changed.connect(self._on_parameter_changed)
        self._bind_service()
        self.destroyed.connect(lambda *_: self.shutdown())

    def _bind_service(self):
        """Follow the editor's execution service if it changed."""
        service = getattr(self.tool, "_execution_service", None)
        if service is self._service:
            return
        self._disconnect_service()
        self._service = service
        if service is None:
            return
        service.tool_run_finished.connect(self._on_tool_run_finished)
        service.busy_changed.connect(self._on_service_busy_changed)
        self._service_connected = True

    def _disconnect_service(self):
        if not self._service_connected or self._service is None:
            return
        try:
            self._service.tool_run_finished.disconnect(self._on_tool_run_finished)
        except (TypeError, RuntimeError):
            pass
        try:
            self._service.busy_changed.disconnect(self._on_service_busy_changed)
        except (TypeError, RuntimeError):
            pass
        self._service_connected = False
        self._service = None

    def run_once(self):
        """Execute the owning tool immediately."""
        if self._shutdown:
            return
        self._timer.stop()
        self._execute()

    def _set_auto_tooltip(self, active):
        if active:
            self.auto_button.setToolTip(
                f"Auto-run is ON: runs {self.DEBOUNCE_MS} ms after the last "
                "setting change. Click to stop."
            )
        else:
            self.auto_button.setToolTip(
                f"Run this tool {self.DEBOUNCE_MS} ms after the last setting change."
            )

    def _on_auto_toggled(self, checked):
        if self._shutdown:
            return
        self.auto_button.setText("AUTO ON" if checked else "AUTO")
        self._set_auto_tooltip(checked)
        if checked:
            # Let the current UI event finish before running.
            self._timer.start(0)
        else:
            self._timer.stop()

    def _on_parameter_changed(self):
        if self._shutdown or not self.auto_button.isChecked():
            return
        self._bind_service()
        if self._executing:
            self._pending = True
            return
        self._timer.start()

    def _on_timer(self):
        if not self._shutdown and self.auto_button.isChecked():
            self._execute()

    def _set_running(self, running):
        self._executing = running
        self.run_button.setText("RUN..." if running else "RUN")
        self.run_button.setEnabled(not running)

    def _execute(self):
        if self._shutdown or self._executing:
            return
        self._bind_service()
        service = self._service
        if service is None:
            self._execute_sync()
            return
        if service.is_busy:
            self._pending = True
            return
        self._set_running(True)
        if not service.run_tool(self.tool):
            # Another request won the race; retry after the current run.
            self._set_running(False)
            self._pending = True

    def _execute_sync(self):
        self._set_running(True)
        try:
            self.tool.execute()
        finally:
            self._set_running(False)
            if self._pending:
                self._pending = False
                if self.auto_button.isChecked() and not self._shutdown:
                    self._timer.start()

    def _on_tool_run_finished(self, tool, error):
        if self._shutdown or tool is not self.tool:
            return
        self._set_running(False)
        if self._pending:
            self._pending = False
            if self.auto_button.isChecked():
                self._timer.start()
            elif not self._shutdown:
                self._execute()

    def _on_service_busy_changed(self, busy):
        if self._shutdown:
            return
        if busy:
            return
        if self._pending and not self._executing:
            self._pending = False
            if self.auto_button.isChecked():
                self._timer.start()
            else:
                self._execute()

    def shutdown(self):
        """Stop the timer and detach from the tool signal."""
        if self._shutdown:
            return
        self._shutdown = True
        try:
            self._timer.stop()
        except RuntimeError:
            pass
        try:
            self.tool.parameter_changed.disconnect(self._on_parameter_changed)
        except Exception:
            pass
        self._disconnect_service()
        try:
            if self.auto_button.isChecked():
                self.auto_button.blockSignals(True)
                self.auto_button.setChecked(False)
                self.auto_button.setText("AUTO")
                self.auto_button.blockSignals(False)
        except RuntimeError:
            pass
