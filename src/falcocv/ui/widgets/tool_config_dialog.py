from PyQt6.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QWidget
from PyQt6.QtCore import Qt, QTimer, QRect


class ToolConfigDialog(QDialog):
    def __init__(self, tool, parent=None, geometry=None):
        super().__init__(parent)
        self._tool = tool
        self._config_widget = tool.get_config_widget()

        self.setWindowTitle(f"Edit {tool.name}")
        self.setWindowFlags(
            self.windowFlags() |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowMinimizeButtonHint
        )

        layout = QVBoxLayout(self)
        layout.addWidget(self._config_widget)

        help_bar = QWidget()
        help_bar.setFixedHeight(24)
        help_bar.setStyleSheet("background-color: transparent;")
        help_lyt = QHBoxLayout(help_bar)
        help_lyt.setContentsMargins(0, 0, 6, 0)
        help_lyt.addStretch()
        help_btn = QPushButton("?")
        help_btn.setFixedWidth(24)
        help_btn.setFixedHeight(20)
        help_btn.setToolTip("Show tool documentation")
        # Imported lazily: documentation is optional and must never prevent
        # the tool configuration dialog from opening.
        help_btn.clicked.connect(lambda: self._show_help())
        help_lyt.addWidget(help_btn)
        layout.addWidget(help_bar)

        for btn in self.findChildren(QPushButton):
            btn.setAutoDefault(False)
            btn.setDefault(False)

        if geometry is not None:
            self.move(geometry.topLeft())
            self.resize(geometry.size())
        else:
            self.resize(1000, 700)

        # Keep a reference to avoid garbage collection
        tool._config_window = self

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(50, self._fit_viewers)

    def _fit_viewers(self):
        if self._config_widget is not None:
            for w in self._config_widget.findChildren(QWidget):
                if hasattr(w, 'fit_in_view'):
                    w.fit_in_view()

    def _show_help(self):
        try:
            from falcocv.ui.widgets.help_dialog import ToolHelpDialog
            ToolHelpDialog.show_for_tool(self._tool, parent=self)
        except Exception:
            # Documentation is non-critical: log the failure but keep the
            # configuration dialog usable.
            import traceback
            traceback.print_exc()

    def closeEvent(self, event):
        if self._config_widget is not None and hasattr(self._config_widget, 'on_config_closed'):
            try:
                self._config_widget.on_config_closed()
            except Exception:
                pass
        # Drop callbacks bound to the config widget so a later tool execution
        # cannot touch deleted Qt objects.
        if self._config_widget is not None:
            for signal_name in ("data_changed", "structure_changed", "connection_changed"):
                signal = getattr(self._tool, signal_name, None)
                if signal is not None and hasattr(signal, "disconnect_owner"):
                    signal.disconnect_owner(self._config_widget)
        self._tool._config_window = None
        super().closeEvent(event)


def open_tool_config(tool, parent=None, geometry=None):
    from PyQt6 import sip

    wnd = tool._config_window
    if wnd is not None and sip.isdeleted(wnd):
        # The Qt object was destroyed without closeEvent() running (e.g. the
        # parent window was destroyed). Drop the stale Python reference.
        wnd = None
        tool._config_window = None

    if wnd is not None:
        if geometry is not None:
            wnd.move(geometry.topLeft())
            wnd.resize(geometry.size())
        elif not wnd.isVisible():
            wnd.resize(1000, 700)
        wnd.show()
        wnd.raise_()
        wnd.activateWindow()
    else:
        dialog = ToolConfigDialog(tool, parent, geometry)
        dialog.show()
