"""Cross-platform config directory resolution."""

import os
import sys


def state_dir() -> str:
    """Per-user directory for config.json and UI settings.

    ``FALCOCV_CONFIG_DIR`` overrides the location (used by tests and portable
    installs).  Migrates an existing CVision configuration folder on first run.
    """
    override = os.environ.get("FALCOCV_CONFIG_DIR")
    if override:
        os.makedirs(override, exist_ok=True)
        return override

    if sys.platform == "win32":
        base = os.environ.get("APPDATA", os.path.expanduser("~"))
    else:
        base = os.environ.get(
            "XDG_CONFIG_HOME", os.path.expanduser("~/.config")
        )
    path = os.path.join(base, "FalcoCV")
    legacy = os.path.join(base, "CVision")
    if not os.path.isdir(path) and os.path.isdir(legacy):
        try:
            import shutil
            shutil.copytree(legacy, path)
        except Exception:
            pass
    os.makedirs(path, exist_ok=True)
    return path


def config_path() -> str:
    """Return the path to config.json (same directory as state files)."""
    return os.path.join(state_dir(), "config.json")
