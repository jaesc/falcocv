from __future__ import annotations

import os
import sys
import subprocess
import shutil
import json
import platform
from pathlib import Path


def _get_app_data() -> Path:
    if platform.system() == "Windows":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "FalcoCV"


APP_DATA = _get_app_data()
PYTHON_DIR = APP_DATA / "python"
VENV_DIR = APP_DATA / "venv"


def _get_bundled_python_dir() -> Path | None:
    if not getattr(sys, "frozen", False):
        return None
    exe_dir = Path(sys.executable).parent
    bundled = exe_dir / "python"
    if bundled.exists():
        return bundled
    return None


def _get_python_exe(python_dir: Path) -> Path:
    if platform.system() == "Windows":
        return python_dir / "python.exe"
    return python_dir / "bin" / "python3"


def _get_venv_python() -> Path:
    if platform.system() == "Windows":
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python3"


def _get_py_ver() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def get_venv_site_packages() -> str | None:
    if not VENV_DIR.exists():
        return None
    if platform.system() == "Windows":
        sp = VENV_DIR / "Lib" / "site-packages"
    else:
        sp = VENV_DIR / "lib" / f"python{_get_py_ver()}" / "site-packages"
    if sp.exists():
        return str(sp)
    return None


def ensure_python() -> str | None:
    if not getattr(sys, "frozen", False):
        return sys.executable
    python_exe = _get_python_exe(PYTHON_DIR)
    if python_exe.exists():
        return str(python_exe)
    bundled = _get_bundled_python_dir()
    if bundled and bundled.exists():
        PYTHON_DIR.mkdir(parents=True, exist_ok=True)
        for item in bundled.iterdir():
            dst = PYTHON_DIR / item.name
            if item.is_dir():
                shutil.copytree(item, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(item, dst)
        if python_exe.exists():
            return str(python_exe)
    return None


def ensure_venv() -> bool:
    if VENV_DIR.exists() and _get_venv_python().exists():
        return True
    python_exe = ensure_python()
    if not python_exe:
        return False
    VENV_DIR.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [python_exe, "-m", "venv", str(VENV_DIR)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return False
    subprocess.run(
        [str(_get_venv_python()), "-m", "pip", "install", "--upgrade", "pip"],
        capture_output=True, timeout=60,
    )
    return True


def pip_install(packages: list[str]) -> tuple[bool, str]:
    if not ensure_venv():
        return False, "Could not create the virtual environment"
    result = subprocess.run(
        [str(_get_venv_python()), "-m", "pip", "install"] + packages,
        capture_output=True, text=True, timeout=300,
    )
    return result.returncode == 0, result.stdout + result.stderr


def pip_list() -> list[dict]:
    python = _get_venv_python()
    if not python.exists():
        return []
    result = subprocess.run(
        [str(python), "-m", "pip", "list", "--format=json"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode == 0 and result.stdout.strip():
        return json.loads(result.stdout)
    return []


def pip_uninstall(packages: list[str]) -> tuple[bool, str]:
    python = _get_venv_python()
    if not python.exists():
        return False, "The virtual environment does not exist"
    result = subprocess.run(
        [str(python), "-m", "pip", "uninstall", "-y"] + packages,
        capture_output=True, text=True, timeout=60,
    )
    return result.returncode == 0, result.stdout + result.stderr
