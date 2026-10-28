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


def main():
    if getattr(sys, 'frozen', False):
        from falcocv.core.environment import ensure_venv, get_venv_site_packages
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
        if getattr(sys, "frozen", False):
            server_exe = os.path.join(os.path.dirname(sys.executable), "FalcoCVServer.exe")
            cmd = [server_exe, "--host", "0.0.0.0", "--port", str(port)]
        else:
            env["PYTHONPATH"] = root + os.pathsep + env.get("PYTHONPATH", "")
            cmd = [sys.executable, "-m", "uvicorn", "falcocv.web.server:app",
                   "--host", "0.0.0.0", "--port", str(port)]
        proc = subprocess.Popen(
            cmd, cwd=root, env=env,
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
    apply_qt_theme(app)

    icon_path = _app_icon_path()
    if icon_path:
        app.setWindowIcon(QIcon(icon_path))

    project = None
    if len(sys.argv) > 1:
        path = sys.argv[1]
        if os.path.isfile(path):
            project = HMIProject.load(path)

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
