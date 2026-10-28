"""Window geometry helpers for the Qt UI.

These functions were moved out of core.config so that core/ remains
free of PyQt6 imports.
"""

from __future__ import annotations


def load_window_geometry(config_manager):
    """Return (QRect, maximized) for the main window, validating the screen."""
    from PyQt6.QtCore import QRect
    from PyQt6.QtGui import QGuiApplication

    w = config_manager.data["window"]
    rect = QRect(w["x"], w["y"], w["width"], w["height"])

    screens = QGuiApplication.screens()
    on_screen = any(screen.geometry().intersects(rect) for screen in screens)

    if not on_screen:
        d = config_manager.defaults["window"]
        rect = QRect(d["x"], d["y"], d["width"], d["height"])
        return rect, False

    return rect, w["maximized"]


def save_window_geometry(config_manager, rect, maximized):
    """Persist the main window geometry to the config manager."""
    config_manager.data["window"] = {
        "x": rect.x(),
        "y": rect.y(),
        "width": rect.width(),
        "height": rect.height(),
        "maximized": maximized,
        "editor_splitter": config_manager.data["window"].get("editor_splitter", [400, 600]),
    }
    config_manager.save()
