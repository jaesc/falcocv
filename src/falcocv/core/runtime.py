"""Filesystem layout helpers shared by development and frozen builds.

The standalone distribution is a PyInstaller onedir bundle: the executables
(``FalcoCV``, ``FalcoCVServer``, ``FalcoCVCLI``, ``FalcoCVHMI``) live next to a
``_internal`` directory that holds the interpreter and the packages, and the
optional ``models`` directory sits next to the executables.

``core`` must stay Qt-free; this module only uses the standard library.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

SERVER_BASENAME = "FalcoCVServer"


def is_frozen() -> bool:
    """True inside a PyInstaller (or compatible) bundle."""
    return bool(getattr(sys, "frozen", False))


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def bundle_dir() -> Path:
    """Read-only resource root: ``_MEIPASS`` when frozen, the checkout if not."""
    if is_frozen():
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return _repo_root()


def app_dir() -> Path:
    """Directory with the executables when frozen, the checkout root if not."""
    if is_frozen():
        return Path(sys.executable).resolve().parent
    return _repo_root()


def server_executable() -> Path:
    """Path of the web/HMI server executable next to the GUI executable."""
    suffix = ".exe" if os.name == "nt" else ""
    return app_dir() / f"{SERVER_BASENAME}{suffix}"
