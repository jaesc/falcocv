"""Standalone HMI Designer — launch without the full FalcoCV app.

Usage:
    python hmi_designer.py
    python hmi_designer.py demo.hmi
"""
import sys
import os
import subprocess
import socket
import time

from falcocv.core.runtime import app_dir, is_frozen, server_executable


def _app_icon_path() -> str | None:
    """Resolve the theme-aware app icon; the assets live next to this module."""
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


def main():
    from falcocv.core.user_libs import install_user_libs

    install_user_libs()
    if is_frozen():
        from falcocv.core.environment import (
            bundled_python_available,
            ensure_venv,
            get_venv_site_packages,
        )
        if bundled_python_available():
            ensure_venv()
            sp = get_venv_site_packages()
            if sp and sp not in sys.path:
                sys.path.insert(0, sp)

    from falcocv.core.config import ConfigManager

    port = ConfigManager().get_web_port()
    auto_web = False
    proc = None

    try:
        s = socket.create_connection(("127.0.0.1", port), timeout=1)
        s.close()
    except OSError:
        auto_web = True
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env = os.environ.copy()
        if is_frozen():
            cmd = [str(server_executable()), "--host", "0.0.0.0", "--port", str(port)]
            cwd = str(app_dir())
        else:
            env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
            cmd = [sys.executable, "-m", "uvicorn", "falcocv.web.server:app",
                   "--host", "0.0.0.0", "--port", str(port)]
            cwd = root
        proc = subprocess.Popen(
            cmd, cwd=cwd, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

        for _ in range(40):
            try:
                s = socket.create_connection(("127.0.0.1", port), timeout=0.5)
                s.close()
                break
            except OSError:
                time.sleep(0.5)
        else:
            auto_web = False

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QIcon
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_designer_dialog import HMIDesignerDialog
    from falcocv.ui.style import apply_qt_theme

    app = QApplication(sys.argv)
    from falcocv.ui.tool_signal_dispatcher import install_tool_signal_dispatcher
    install_tool_signal_dispatcher()
    apply_qt_theme(app)

    icon_path = _app_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))

    project = None
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            project = HMIProject.load(path)
            ConfigManager().add_recent_hmi_file(path)

    dialog = HMIDesignerDialog(project=project)
    if icon_path:
        dialog.setWindowIcon(QIcon(icon_path))
    dialog.show()
    exit_code = app.exec()

    if auto_web:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
