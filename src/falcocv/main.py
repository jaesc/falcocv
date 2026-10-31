import sys
import os
import subprocess
import faulthandler
import threading
import traceback
import warnings
warnings.filterwarnings("ignore", message="distutils Version classes")

# ── Crash log ─────────────────────────────────────────────────────────
# Captures every print(), Qt message and traceback to
# %APPDATA%/FalcoCV/falcocv.log.  Appended (rotated at 5 MB) so a crash log
# survives the next launch.
if sys.platform == "win32":
    _log_base = os.environ.get("APPDATA", os.path.expanduser("~"))
else:
    _log_base = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
_log_path = os.path.join(_log_base, "FalcoCV", "falcocv.log")


# ── Qt message handler (catches OpenGL/Qt warnings to stderr) ────────
from PyQt6.QtCore import qInstallMessageHandler, QtMsgType

def _qt_msg_handler(mode, context, message):
    print(f"[Qt {mode.name}] {message}", file=sys.stderr)

qInstallMessageHandler(_qt_msg_handler)


class _Tee:
    """Write to multiple streams at once (console + log file)."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
                s.flush()
            except Exception:
                pass

    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass


_log_fh = None
try:
    os.makedirs(os.path.dirname(_log_path), exist_ok=True)
    try:
        if os.path.getsize(_log_path) > 5 * 1024 * 1024:
            os.replace(_log_path, _log_path + ".1")
    except OSError:
        pass
    _log_fh = open(_log_path, "a", encoding="utf-8", buffering=1)
    _log_fh.write(
        f"=== FalcoCV {__import__('datetime').datetime.now()} | {__file__} ===\n"
    )
    sys.stdout = _Tee(sys.stdout, _log_fh)
    sys.stderr = _Tee(sys.stderr, _log_fh)
except Exception:
    _log_fh = None


# ── Unhandled exceptions ──────────────────────────────────────────────
# PyQt6 calls qFatal() (and aborts the process) when a Python exception
# escapes a slot.  A custom excepthook is honoured by PyQt and keeps the app
# alive while leaving the full traceback in the log.
def _log_unhandled(exc_type, exc_value, exc_tb):
    try:
        print("=== UNHANDLED PYTHON EXCEPTION ===", file=sys.stderr, flush=True)
        traceback.print_exception(exc_type, exc_value, exc_tb)
        sys.stderr.flush()
    except Exception:
        pass


sys.excepthook = _log_unhandled


def _thread_excepthook(args):
    _log_unhandled(args.exc_type, args.exc_value, args.exc_traceback)


threading.excepthook = _thread_excepthook

# Hard crashes (segfaults) go straight to the log file as well.
try:
    if _log_fh is not None:
        faulthandler.enable(file=_log_fh, all_threads=True)
    else:
        faulthandler.enable(all_threads=True)
except Exception:
    pass
# ────────────────────────────────────────────────────────────────────────

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QSplitter, QMenu, QMessageBox, QDockWidget)
from PyQt6.QtGui import QAction, QActionGroup, QColor, QIcon, QGuiApplication
from PyQt6.QtCore import Qt, QTimer
from falcocv.ui.style import ALT_BG, apply_qt_theme, BORDER, CAUTION, COMPACT_CSS, DARK_BG, SUCCESS, TEXT_DIM
from falcocv.ui.window_geometry import load_window_geometry, save_window_geometry
from falcocv.core.config import ConfigManager
from falcocv.version import __version__, __build__, __semver__


def _app_icon_path() -> str | None:
    """Resolve the theme-aware app icon for dev and frozen bundles."""
    if getattr(sys, "frozen", False):
        base = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    else:
        base = os.path.dirname(os.path.abspath(__file__))
    icon_name = "icon.ico"
    try:
        from falcocv.core.config import ConfigManager
        if ConfigManager().get_theme().lower() == "light":
            icon_name = "icon_light.ico"
    except Exception:
        pass
    icon_path = os.path.join(base, "ui", "assets", icon_name)
    return icon_path if os.path.exists(icon_path) else None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"FalcoCV v{__version__} — Modular Machine Vision Environment")
        icon_path = _app_icon_path()
        if icon_path:
            self.setWindowIcon(QIcon(icon_path))

        self.config = ConfigManager()
        rect, maximized = load_window_geometry(self.config)
        self.setGeometry(rect)
        if maximized:
            self.showMaximized()
            
        self.workflow = Workflow()
        self._modified = False

        # Central Widget
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QVBoxLayout(self.central_widget)

        # The Workflow Editor (Extracted reusable component)
        self.editor = WorkflowEditor(self.workflow)
        main_layout.addWidget(self.editor)

        # Global Toolbox
        self.toolbox = Toolbox(embedded=True)
        self.toolbox_dock = QDockWidget("Toolbox", self)
        self.toolbox_dock.setWidget(self.toolbox)
        self.toolbox_dock.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea
        )
        self.toolbox_dock.setFeatures(
            QDockWidget.DockWidgetFeature.DockWidgetClosable |
            QDockWidget.DockWidgetFeature.DockWidgetMovable |
            QDockWidget.DockWidgetFeature.DockWidgetFloatable
        )
        self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, self.toolbox_dock)

        # Restore dock visibility
        if not self.config.get_toolbox_visible():
            self.toolbox_dock.hide()

        # Menu Bar
        self.setup_menus()
        self.setup_tools_menu()
        self.setup_help_menu()
        self._connect_modification_signals()
        self._compact_mode = False
        self._adjust_for_small_screen()

        # Web server polling timer
        self._web_pending = None
        self._web_timer = QTimer()
        self._web_timer.timeout.connect(self._web_check)
        self._web_timer.start(200)

        # Web server blink indicator
        self._web_blink = QTimer()
        self._web_blink.timeout.connect(self._web_blink_tick)
        self._web_blink_on = False

        self._web_start()

    def closeEvent(self, event):
        if not self._confirm_save_if_modified():
            event.ignore()
            return

        # Close all tool configuration windows
        for tool in self.workflow.tools:
            if hasattr(tool, "_config_window") and tool._config_window:
                tool._config_window.close()

        save_window_geometry(self.config, self.geometry(), self.isMaximized())
        self.config.set_editor_splitter(self.editor.splitter.sizes())
        self.config.set_toolbox_visible(self.toolbox_dock.isVisible())
        self.editor.controller.run_loop_stop()
        self._web_stop()
        super().closeEvent(event)

    def setup_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("File")
        
        # Job Manager
        self.job_manager = JobManager()
        
        new_action = QAction("New Job", self)
        new_action.triggered.connect(self.new_job)
        file_menu.addAction(new_action)
        
        open_action = QAction("Open Job...", self)
        open_action.triggered.connect(self.open_job)
        file_menu.addAction(open_action)

        file_menu.addSeparator()

        load_tool_action = QAction("Load Tool from Disk...", self)
        load_tool_action.triggered.connect(self.load_tool_from_disk)
        file_menu.addAction(load_tool_action)

        file_menu.addSeparator()

        self.recent_menu = QMenu("Open Recent", self)
        file_menu.addMenu(self.recent_menu)
        self.update_recent_menu()
        
        save_action = QAction("Save Job", self)
        save_action.triggered.connect(self.save_job)
        file_menu.addAction(save_action)
        
        save_as_action = QAction("Save Job As...", self)
        save_as_action.triggered.connect(self.save_job_as)
        file_menu.addAction(save_as_action)
        
        
        file_menu.addSeparator()

        job_config_action = QAction("Job Configuration...", self)
        job_config_action.triggered.connect(self.open_job_config)
        file_menu.addAction(job_config_action)

        file_menu.addSeparator()
        
        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        view_menu = menubar.addMenu("View")
        view_menu.addAction(self.toolbox_dock.toggleViewAction())

        self.overlays_action = QAction("Overlays", self)
        self.overlays_action.setCheckable(True)
        self.overlays_action.setChecked(True)
        self.overlays_action.triggered.connect(self._toggle_overlays)
        view_menu.addAction(self.overlays_action)

        reset_action = QAction("Reset defaults", self)
        reset_action.triggered.connect(self._reset_view_defaults)
        view_menu.addAction(reset_action)

        view_menu.addSeparator()

        res_menu = view_menu.addMenu("Resolution mode")

        self.res_auto = QAction("Auto", self, checkable=True)
        self.res_high = QAction("High", self, checkable=True)
        self.res_low = QAction("Low", self, checkable=True)

        self.res_group = QActionGroup(self)
        self.res_group.setExclusive(True)
        for a in (self.res_auto, self.res_high, self.res_low):
            self.res_group.addAction(a)
            res_menu.addAction(a)

        mode = self.config.get_resolution_mode()
        {"auto": self.res_auto, "high": self.res_high, "low": self.res_low}[mode].setChecked(True)
        self.res_group.triggered.connect(self._on_resolution_mode_changed)

        view_menu.addSeparator()

        point_size_menu = view_menu.addMenu("3D point size")
        self.point_size_group = QActionGroup(self)
        self.point_size_group.setExclusive(True)
        current_size = self.config.get_3d_point_size()
        for size in (1.0, 2.0, 3.0, 4.0, 5.0):
            a = QAction(f"{size:.1f}px", self, checkable=True)
            a.setChecked(abs(current_size - size) < 0.01)
            a.triggered.connect(lambda checked, s=size: self._on_3d_point_size_changed(s))
            self.point_size_group.addAction(a)
            point_size_menu.addAction(a)

        view_menu.addSeparator()
        theme_menu = view_menu.addMenu("Theme")
        self.theme_group = QActionGroup(self)
        for t in ("Dark", "Material", "Monokai", "Light", "Synthwave", "Gruvbox", "Nord"):
            a = QAction(t, self, checkable=True)
            a.setChecked(self.config.get_theme().lower() == t.lower())
            a.triggered.connect(lambda checked, name=t.lower(): self._on_theme_changed(name))
            self.theme_group.addAction(a)
            theme_menu.addAction(a)

    def open_job_config(self):
        dlg = JobConfigDialog(self.workflow, self)
        dlg.exec()

    def _toggle_overlays(self, checked):
        from falcocv.core.config import ConfigManager
        ConfigManager().set_show_overlays(checked)
        WorkflowEditor._global_show_overlays = checked
        self.overlays_action.setChecked(checked)
        self.editor.update_viewer_image()
        self._refresh_block_overlays(self.workflow.tools)

    def _reset_view_defaults(self):
        from falcocv.core.config import ConfigManager
        cfg = ConfigManager()
        cfg.reset_view_defaults()
        self.editor.splitter.setSizes(cfg.get_editor_splitter())
        self.toolbox_dock.setVisible(cfg.get_toolbox_visible())
        visible = cfg.get_show_overlays()
        self.overlays_action.setChecked(visible)
        WorkflowEditor._global_show_overlays = visible
        self.editor.update_viewer_image()
        self._refresh_block_overlays(self.workflow.tools)
        self.res_auto.setChecked(True)
        self._apply_resolution_mode("auto", startup=False)

    def _on_resolution_mode_changed(self, action):
        mode = {self.res_auto: "auto", self.res_high: "high", self.res_low: "low"}[action]
        self.config.set_resolution_mode(mode)
        self._apply_resolution_mode(mode, startup=False)

    def _on_3d_point_size_changed(self, size: float):
        self.config.set_3d_point_size(size)

    def _on_theme_changed(self, name):
        old = self.config.get_theme()
        if name == old:
            return
        self.config.set_theme(name)
        QMessageBox.information(self, "Theme",
            f"Theme set to '{name.capitalize()}'. "
            "Restart the application for the change to take effect.")

    def _apply_resolution_mode(self, mode, startup=True):
        if mode == "auto":
            rect = self.geometry()
            screen = QGuiApplication.screenAt(rect.center()) or QGuiApplication.primaryScreen()
            avail = screen.availableGeometry()
            resolved = "low" if (avail.width() < 1600 or avail.height() < 900) else "high"
        else:
            resolved = mode

        app = QApplication.instance()
        current_ss = app.styleSheet()
        has_compact = COMPACT_CSS in current_ss
        if resolved == "low" and not has_compact:
            app.setStyleSheet(current_ss + COMPACT_CSS)
            self._compact_mode = True
        elif resolved == "high" and has_compact:
            app.setStyleSheet(current_ss.replace(COMPACT_CSS, ""))
            self._compact_mode = False

        if startup and resolved == "low":
            self.showMaximized()
            self.editor.splitter.setSizes([250, 750])

    def _adjust_for_small_screen(self):
        mode = self.config.get_resolution_mode()
        if mode == "auto":
            rect = self.geometry()
            screen = QGuiApplication.screenAt(rect.center()) or QGuiApplication.primaryScreen()
            avail = screen.availableGeometry()
            resolved = "low" if (avail.width() < 1600 or avail.height() < 900) else "high"
            self.res_auto.setChecked(True)
        else:
            resolved = mode
        self._apply_resolution_mode(resolved, startup=True)

    def _refresh_block_overlays(self, tools):
        from falcocv.tools.block import Block
        for tool in list(tools):
            if isinstance(tool, Block):
                if tool._config_window:
                    try:
                        cfg = tool._config_window._config_widget
                        if hasattr(cfg, 'editor'):
                            cfg.editor.update_viewer_image()
                    except Exception:
                        pass
                self._refresh_block_overlays(tool.workflow.tools)

    def update_title(self):
        path = self.job_manager.get_current_job_path()
        name = path if path else "Untitled"
        suffix = " *" if self._modified else ""
        self.setWindowTitle(f"FalcoCV v{__version__} — {name}{suffix}")

    def setup_tools_menu(self):
        menubar = self.menuBar()
        tools_menu = menubar.addMenu("Tools")
        hmi_action = QAction("HMI Designer...", self)
        hmi_action.triggered.connect(self.open_hmi_designer)
        tools_menu.addAction(hmi_action)

        tools_menu.addSeparator()
        self.web_action = QAction("Start Web Server", self)
        self.web_action.triggered.connect(self.toggle_web_server)
        tools_menu.addAction(self.web_action)

    def open_hmi_designer(self):
        import time, socket
        from falcocv.core.config import ConfigManager

        auto_web = False

        if not self._is_web_running():
            auto_web = True
            self._web_start()

            port = ConfigManager().get_web_port()
            for _ in range(40):
                QApplication.processEvents()
                if self._is_web_running():
                    break
                try:
                    s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                    s.close()
                    for _ in range(5):
                        QApplication.processEvents()
                        if self._is_web_running():
                            break
                        time.sleep(0.1)
                    break
                except OSError:
                    time.sleep(0.5)
            else:
                auto_web = False

        if self._is_web_running():
            self.editor.set_web_status(
                f"Web: ● :{ConfigManager().get_web_port()}", "#4ec9b0")

        from falcocv.ui.widgets.hmi_designer_dialog import HMIDesignerDialog
        from falcocv.core.hmi import HMIProject
        dlg = HMIDesignerDialog(parent=self)
        if dlg.exec():
            project = dlg.get_project()
            self.workflow.hmi_layout = project.layout
            self._mark_modified()

        if auto_web:
            self._web_stop()

    def toggle_web_server(self):
        if self._is_web_running():
            self._web_stop()
        else:
            self._web_start()
        self.web_action.setText(
            "Stop Web Server" if self._is_web_running()
            else "Start Web Server"
        )

    # ── Web Server ─────────────────────────────────────────────────────────

    _web_proc: subprocess.Popen | None = None
    _web_pid: int | None = None

    def _is_web_running(self) -> bool:
        return self._web_proc is not None and self._web_proc.poll() is None

    def _web_start(self):
        if self._is_web_running():
            return
        from falcocv.core.config import ConfigManager
        port = ConfigManager().get_web_port()
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        creation_flags = 0
        if sys.platform == "win32":
            creation_flags = subprocess.CREATE_NO_WINDOW
        env = os.environ.copy()
        if getattr(sys, "frozen", False):
            server_exe = os.path.join(os.path.dirname(sys.executable), "FalcoCVServer.exe")
            cmd = [server_exe, "--host", "0.0.0.0", "--port", str(port)]
        else:
            env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
            cmd = [sys.executable, "-m", "uvicorn", "falcocv.web.server:app",
                   "--host", "0.0.0.0", "--port", str(port)]
        proc = subprocess.Popen(
            cmd, cwd=root, env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            creationflags=creation_flags,
        )

        self.editor.set_web_status("Web: ⟳", "#ffa500")
        self._web_blink_on = False
        self._web_blink.start(500)
        self.web_action.setText("Starting...")
        self.web_action.setEnabled(False)

        import threading as _t
        _t.Thread(target=self._web_await, args=(proc, port), daemon=True).start()

    def _web_await(self, proc, port):
        import time, socket, threading as _t
        def _read_stderr(stream):
            try:
                for line in iter(stream.readline, b''):
                    if line: print(f"[web] {line.decode().strip()}", flush=True)
            except ValueError:
                pass
        _t.Thread(target=_read_stderr, args=(proc.stderr,), daemon=True).start()

        hosts = ["127.0.0.1", "localhost"]
        start = time.time()
        ready = False
        last_err = ""
        while time.time() - start < 60:
            if proc.poll() is not None:
                print(f"[web] process died with code {proc.poll()}", flush=True)
                break
            for host in hosts:
                try:
                    s = socket.create_connection((host, port), timeout=1)
                    s.close()
                    ready = True
                    break
                except OSError as e:
                    last_err = str(e)
            if ready:
                break
            time.sleep(0.5)

        if not ready:
            print(f"[web] last connection error: {last_err}", flush=True)

        self._web_pending = (proc, port, ready)

    def _web_check(self):
        pending = self._web_pending
        if pending is not None:
            proc, port, ready = pending
            self._web_pending = None
            self._web_start_done(proc, ready, port)

    def _web_blink_tick(self):
        self._web_blink_on = not self._web_blink_on
        c = "#ffa500" if self._web_blink_on else "#2d2d2d"
        self.editor.label_web.setStyleSheet(
            f"color: {c}; font-family: Consolas, monospace; font-size: 10px;")

    def _web_start_done(self, proc, ready, port):
        self._web_blink.stop()
        self.web_action.setEnabled(True)
        if ready:
            self._web_proc = proc
            self._web_pid = proc.pid
            self.editor.set_web_status(f"Web: ● :{port}", "#4ec9b0")
        else:
            print(f"[web] ERROR: server on port {port} not ready after 60s", flush=True)
            proc.kill()
            proc.stderr.close()
            self.editor.set_web_status("Web: ✗", "#f14c4c")
        self.web_action.setText("Stop Web Server" if self._is_web_running() else "Start Web Server")

    def _web_stop(self):
        proc = self._web_proc
        pid = self._web_pid
        self._web_proc = None
        self._web_pid = None

        if proc:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                try:
                    proc.kill()
                    proc.wait(timeout=2)
                except Exception:
                    pass
            try:
                proc.stderr.close()
            except Exception:
                pass
        elif pid:
            try:
                subprocess.run(
                    ['taskkill', '/F', '/PID', str(pid)],
                    capture_output=True, timeout=5,
                )
            except Exception:
                pass

        self.editor.set_web_status("Web: ○", "#888")

    def setup_help_menu(self):
        menubar = self.menuBar()
        help_menu = menubar.addMenu("Help")
        ref_action = QAction("Tool Reference...", self)
        ref_action.triggered.connect(self.show_tool_reference)
        help_menu.addAction(ref_action)
        help_menu.addSeparator()
        about_action = QAction("About FalcoCV", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)

    def show_tool_reference(self):
        from falcocv.ui.widgets.help_dialog import ToolHelpDialog
        ToolHelpDialog(parent=self).exec()

    def show_about(self):
        from datetime import date
        build_date = ""
        build_info = ""
        # Try to read build info from the bundled file (frozen builds)
        import os as _os
        here = _os.path.dirname(_os.path.abspath(__file__))
        info_path = _os.path.join(here, "_build_info.txt")
        if _os.path.exists(info_path):
            build_info = _os.path.getmtime(info_path)
            build_date = date.fromtimestamp(build_info).isoformat()
        else:
            build_date = date.today().isoformat()

        QMessageBox.about(
            self,
            "About FalcoCV",
            f"<style>p,td {{ font-size: 11pt; }}</style>"
            f"<h2>FalcoCV v{__version__}</h2>"
            f"<p>Modular Machine Vision Environment</p>"
            f"<hr>"
            f"<p><b>Build:</b> {__build__}</p>"
            f"<p><b>Date:</b> {build_date}</p>"
            f"<hr>"
            f"<p><b>License:</b> GNU GPL v3 or later</p>"
            f"<p style='font-size: 10pt'>FalcoCV is free software: you can redistribute it and/or "
            f"modify it under the terms of the GNU General Public License as published by the "
            f"Free Software Foundation, either version 3 of the License, or (at your option) "
            f"any later version.</p>",
        )

    def _mark_modified(self):
        """Mark the job as having unsaved changes (skip during workflow execution)."""
        if self.editor.controller.state in (ControllerState.RUNNING, ControllerState.LOOPING):
            return
        self._modified = True
        self.update_title()

    def _connect_modification_signals(self):
        """Hook tree and tool signals to dirty tracking."""
        self.editor.tree.order_changed.connect(self._mark_modified)
        self.editor.tree.order_changed.connect(self._connect_tool_signals)
        self.editor.tree.tool_deleted.connect(self._mark_modified)
        self.editor.tree.tool_dropped.connect(self._mark_modified)
        self.editor.tree.tool_dropped.connect(self._connect_tool_signals)
        self.editor.tree.tool_pasted.connect(self._mark_modified)
        self.editor.tree.tool_pasted.connect(self._connect_tool_signals)
        self.editor.tree.connection_changed.connect(self._mark_modified)
        self._connect_tool_signals()

    def _connect_tool_signals(self):
        """Connect each tool's data/structure signals to dirty tracking,
        but skip when the workflow is running (to avoid false triggers from execution)."""
        for tool in self.workflow.tools:
            try:
                tool.data_changed.disconnect(self._mark_modified)
            except (TypeError, RuntimeError):
                pass
            try:
                tool.structure_changed.disconnect(self._mark_modified)
            except (TypeError, RuntimeError):
                pass
            tool.data_changed.connect(self._mark_modified)
            tool.structure_changed.connect(self._mark_modified)

    def _confirm_save_if_modified(self):
        """If the job has unsaved changes, ask the user what to do.

        Returns:
            True  → proceed with the action (saved or discarded)
            False → cancel the action
        """
        if not self._modified:
            return True

        from PyQt6.QtWidgets import QMessageBox
        btn = QMessageBox.question(
            self, "Unsaved Changes",
            "The current job has unsaved changes.\nDo you want to save before continuing?",
            QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Save,
        )

        if btn == QMessageBox.StandardButton.Save:
            self.save_job()
            # If save was cancelled (e.g. user closed the file dialog), don't proceed
            return not self._modified
        elif btn == QMessageBox.StandardButton.Discard:
            return True
        else:  # Cancel
            return False

    def _cleanup_workflow(self, keep_cameras=None):
        keep_cameras = keep_cameras or set()
        for tool in self.workflow.tools:
            if hasattr(tool, "_config_window") and tool._config_window:
                try:
                    tool._config_window.close()
                except Exception:
                    pass
                tool._config_window = None

            if hasattr(tool, "disconnect"):
                if hasattr(tool, "connection_key") and tool.connection_key() in keep_cameras:
                    continue
                try:
                    tool.disconnect()
                except Exception:
                    pass

    def new_job(self):
        if not self._confirm_save_if_modified():
            return
        self._cleanup_workflow()
        self.workflow = Workflow()
        self.editor.set_workflow(self.workflow)
        self.job_manager = JobManager() # Reset manager
        self._modified = False
        self._connect_tool_signals()
        self.update_title()

    def update_recent_menu(self):
        self.recent_menu.clear()
        recents = self.config.get_recent_files()
        if not recents:
            self.recent_menu.addAction("No Recent Files").setEnabled(False)
            return
            
        for path in recents:
            action = QAction(path, self)
            action.triggered.connect(lambda checked, p=path: self.open_job_from_path(p))
            self.recent_menu.addAction(action)

    def open_job_from_path(self, path):
        if not self._confirm_save_if_modified():
            return
        from PyQt6.QtWidgets import QMessageBox
        import os
        if not os.path.exists(path):
            QMessageBox.warning(self, "Warning", f"File not found: {path}")
            return
        try:
            temp_workflow = Workflow()
            self.job_manager.load_job(temp_workflow, path)

            keep_cameras = set()
            for tool in temp_workflow.tools:
                if getattr(tool, "should_persist_connection", lambda: False)():
                    if hasattr(tool, "connection_key"):
                        keep_cameras.add(tool.connection_key())

            self._cleanup_workflow(keep_cameras)
            self.workflow = temp_workflow
            self.editor.set_workflow(self.workflow)
            self._modified = False
            self._connect_tool_signals()
            self.update_title()
            self.config.add_recent_file(path)
            self.update_recent_menu()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load job: {e}")

    def open_job(self):
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(self, "Open Job", "", "FalcoCV Jobs (*.cvj)")
        if path:
            self.open_job_from_path(path)

    def save_job(self):
        if self.job_manager.get_current_job_path():
            path = self.job_manager.get_current_job_path()
            self.job_manager.save_job(self.workflow, path)
            self._modified = False
            self.update_title()
            self.config.add_recent_file(path)
            self.update_recent_menu()
        else:
            self.save_job_as()

    def save_job_as(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        path, _ = QFileDialog.getSaveFileName(self, "Save Job As", "", "FalcoCV Jobs (*.cvj)")
        if path:
            try:
                if not path.endswith('.cvj'):
                    path += '.cvj'
                self.job_manager.save_job(self.workflow, path)
                self._modified = False
                self.update_title()
                self.config.add_recent_file(path)
                self.update_recent_menu()
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save job: {e}")

    def load_tool_from_disk(self):
        from PyQt6.QtWidgets import QFileDialog, QMessageBox
        import json
        import uuid
        filepath, _ = QFileDialog.getOpenFileName(self, "Load Tool from Disk", "", "FalcoCV Tool (*.cvtool)")
        if not filepath:
            return
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            name = data.get("name", "Tool")
            existing_names = {t.name for t in self.workflow.tools}
            if name in existing_names:
                counter = 1
                while f"{name}_{counter}" in existing_names:
                    counter += 1
                data["name"] = f"{name}_{counter}"
            tool = self.workflow.create_tool_from_data(data)
            if tool:
                tool.uuid = str(uuid.uuid4())
                self.workflow.add_tool(tool)
                self.editor.tree.set_workflow(self.workflow)
                self.editor.tree.tool_pasted.emit(tool)
                self._mark_modified()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load tool: {e}")

def main():
    global ConfigManager, Workflow, ControllerState, JobManager
    global WorkflowEditor, Toolbox, JobConfigDialog
    import time as _time
    _t_start = _time.monotonic()
    _MIN_SPLASH_MS = 3000

    from PyQt6.QtWidgets import QApplication
    app = QApplication(sys.argv)

    icon_path = _app_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))

    # Show splash immediately — before any heavy work
    from falcocv.ui.splash import create_splash
    splash = create_splash()
    splash.show()
    app.processEvents()

    # Theme while splash is visible
    splash.showMessage("Applying theme…", Qt.AlignmentFlag.AlignLeft, QColor(TEXT_DIM))
    app.processEvents()
    apply_qt_theme(app)

    # First-run model check/download. Blocks startup with retries as agreed:
    # a missing model must not let the app start in a half-working state.
    splash.showMessage("Checking model files…", Qt.AlignmentFlag.AlignLeft, QColor(TEXT_DIM))
    app.processEvents()
    try:
        from falcocv.core.models import ensure_models

        def _model_progress(name, done, total, index, count):
            pct = int(done * 100 / total) if total else 0
            splash.showMessage(
                f"Models {index}/{count}: {name}  {pct}%",
                Qt.AlignmentFlag.AlignLeft, QColor(TEXT_DIM),
            )
            app.processEvents()

        ensure_models(progress=_model_progress)
    except Exception as exc:
        splash.close()
        QMessageBox.critical(
            None, "FalcoCV models",
            "Could not prepare the model files.\n\n"
            f"{exc}\n\n"
            "Check your network connection or set FALCOCV_MODELS_URL to a "
            "reachable mirror, then start FalcoCV again.",
        )
        sys.exit(1)

    if getattr(sys, 'frozen', False):
        splash.showMessage("Setting up Python environment…", Qt.AlignmentFlag.AlignLeft, QColor(TEXT_DIM))
        app.processEvents()
        from falcocv.core.environment import ensure_venv, get_venv_site_packages
        ensure_venv()
        sp = get_venv_site_packages()
        if sp and sp not in sys.path:
            sys.path.insert(0, sp)

    # Preload all tools during splash
    splash.showMessage("Loading tools…", Qt.AlignmentFlag.AlignLeft, QColor(TEXT_DIM))
    app.processEvents()
    import falcocv.tools

    # Preload ML libraries  si
    splash.showMessage("Loading ML libraries…", Qt.AlignmentFlag.AlignLeft, QColor(TEXT_DIM))
    app.processEvents()
    # Preload optional ML / DNN libraries; log failures instead of silently ignoring.
    try:
        import onnxruntime
        import Levenshtein
        import pylibdmtx.pylibdmtx
    except ImportError as e:
        print(f"Optional ML/DLL libraries not available: {e}")

    try:
        from falcocv.core.ocr.industrial_ocr_engine import preload_engine
        from falcocv.tools.ocr import OCR
        preload_engine("server")
    except Exception as e:
        print(f"OCR preload skipped: {e}")

    try:
        from falcocv.core.classifier.inference import preload_classifier_model
        preload_classifier_model()
    except Exception as e:
        print(f"Classifier preload skipped: {e}")

    # MainWindow dependencies (imported after splash to avoid blocking startup)
    from falcocv.core.config import ConfigManager
    from falcocv.core.workflow import Workflow
    from falcocv.core.workflow_controller import ControllerState
    from falcocv.core.job_manager import JobManager
    from falcocv.ui.widgets.workflow_editor import WorkflowEditor
    from falcocv.ui.widgets.toolbox import Toolbox
    from falcocv.ui.widgets.job_config_dialog import JobConfigDialog

    window = MainWindow()
    if window._compact_mode:
        splash.showMessage("Compact mode — low resolution", Qt.AlignmentFlag.AlignLeft, QColor(CAUTION))
    else:
        splash.showMessage("Ready", Qt.AlignmentFlag.AlignLeft, QColor(SUCCESS))
    app.processEvents()

    window.show()

    _elapsed = (_time.monotonic() - _t_start) * 1000
    if _elapsed < _MIN_SPLASH_MS:
        _time.sleep((_MIN_SPLASH_MS - _elapsed) / 1000)

    splash.finish(window)
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
