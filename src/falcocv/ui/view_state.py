"""Persistence helpers for window and panel view state."""

from __future__ import annotations

import base64

from PyQt6.QtCore import QByteArray, Qt
from PyQt6.QtGui import QGuiApplication
from PyQt6.QtWidgets import QHeaderView, QSplitter, QWidget

from falcocv.core.config import ConfigManager

_VIEWS_KEY = "views"


def _views() -> dict:
    views = ConfigManager().data.setdefault(
        _VIEWS_KEY, {"windows": {}, "splitters": {}, "headers": {}}
    )
    views.setdefault("windows", {})
    views.setdefault("splitters", {})
    views.setdefault("headers", {})
    return views


def _encode(data: QByteArray) -> str:
    return base64.b64encode(bytes(data)).decode("ascii")


def _decode(data: str) -> QByteArray:
    return QByteArray.fromBase64(data.encode("ascii"))


def save_window(window: QWidget, key: str) -> None:
    views = _views()
    views["windows"][key] = {
        "geometry": _encode(window.saveGeometry()),
        "maximized": bool(window.isMaximized()),
    }
    ConfigManager().save()


def restore_window(
    window: QWidget, key: str, default_size: tuple[int, int] = (1000, 700)
) -> bool:
    state = _views()["windows"].get(key)
    restored = False
    if isinstance(state, dict) and state.get("geometry"):
        try:
            restored = window.restoreGeometry(_decode(state["geometry"]))
        except Exception:
            restored = False
    if restored:
        rect = window.frameGeometry()
        if not any(
            screen.availableGeometry().intersects(rect)
            for screen in QGuiApplication.screens()
        ):
            window.resize(*default_size)
            restored = False
        elif state.get("maximized"):
            window.setWindowState(
                window.windowState() | Qt.WindowState.WindowMaximized
            )
    if not restored:
        window.resize(*default_size)
    return restored


def save_window_layout(window, key: str) -> None:
    if not hasattr(window, "saveState"):
        return
    _views()["windows"][f"{key}.layout"] = _encode(window.saveState())
    ConfigManager().save()


def restore_window_layout(window, key: str) -> bool:
    if not hasattr(window, "restoreState"):
        return False
    state = _views()["windows"].get(f"{key}.layout")
    if not state:
        return False
    try:
        return bool(window.restoreState(_decode(state)))
    except Exception:
        return False


def save_splitter(splitter: QSplitter, key: str) -> None:
    if splitter is None:
        return
    _views()["splitters"][key] = _encode(splitter.saveState())
    ConfigManager().save()


def restore_splitter(
    splitter: QSplitter, key: str, default_sizes: list[int]
) -> bool:
    if splitter is None:
        return False
    state = _views()["splitters"].get(key)
    if state:
        try:
            if splitter.restoreState(_decode(state)):
                return True
        except Exception:
            pass
    splitter.setSizes(default_sizes)
    return False


def save_header(header: QHeaderView, key: str) -> None:
    if header is None:
        return
    _views()["headers"][key] = _encode(header.saveState())
    ConfigManager().save()


def restore_header(header: QHeaderView, key: str) -> bool:
    if header is None:
        return False
    state = _views()["headers"].get(key)
    if not state:
        return False
    try:
        return bool(header.restoreState(_decode(state)))
    except Exception:
        return False


def reset_keys(prefixes: tuple[str, ...]) -> None:
    views = _views()
    for group in views.values():
        for key in list(group):
            if any(key == prefix or key.startswith(prefix + ".") for prefix in prefixes):
                del group[key]
    ConfigManager().save()
