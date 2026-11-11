"""User libraries for the standalone bundle.

Vision-pipeline scripts (``PythonScriptTool``) and HMI scripts run inside the
FalcoCV process, so every third-party module they import must be on
``sys.path``.  User directories are **appended**: the packages bundled with
FalcoCV (numpy, OpenCV, Qt, ...) always win, and a user package can never
shadow them with an incompatible build.

Resolution order:

1. ``FALCOCV_USER_LIBS`` (``os.pathsep`` separated) - deployment override.
2. ``<app_dir>/site-packages`` - libraries shipped by the integrator.
3. ``%APPDATA%/FalcoCV/user-packages`` - installed by ``falcocv packages``.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

from .environment import APP_DATA
from .runtime import app_dir

USER_PACKAGES_ENV = "FALCOCV_USER_LIBS"
WHEELHOUSE_ENV = "FALCOCV_WHEELHOUSE"
UV_ENV = "FALCOCV_UV"


def user_packages_dir() -> Path:
    """Directory managed by ``falcocv packages``."""
    return APP_DATA / "user-packages"


def user_dirs() -> list[Path]:
    """Existing user library directories, in import priority order."""
    candidates: list[Path] = []
    for raw in os.environ.get(USER_PACKAGES_ENV, "").split(os.pathsep):
        if raw.strip():
            candidates.append(Path(raw).expanduser())
    candidates.append(app_dir() / "site-packages")
    candidates.append(user_packages_dir())

    seen: set[str] = set()
    result: list[Path] = []
    for candidate in candidates:
        key = str(candidate)
        if key in seen or not candidate.is_dir():
            continue
        seen.add(key)
        result.append(candidate)
    return result


def install_user_libs() -> list[str]:
    """Append the user directories to ``sys.path``; returns the active ones."""
    added: list[str] = []
    for directory in user_dirs():
        path = str(directory)
        if path not in sys.path:
            sys.path.append(path)
            added.append(path)
    return added


def find_uv() -> Path | None:
    """Locate the ``uv`` binary: explicit env, bundle, then ``PATH``."""
    explicit = os.environ.get(UV_ENV)
    if explicit:
        candidate = Path(explicit).expanduser()
        return candidate if candidate.is_file() else None

    bundled = app_dir() / ("uv.exe" if os.name == "nt" else "uv")
    if bundled.is_file():
        return bundled

    alongside_python = Path(sys.executable).parent / (
        "uv.exe" if os.name == "nt" else "uv"
    )
    if alongside_python.is_file():
        return alongside_python

    found = shutil.which("uv")
    return Path(found) if found else None


def _uv_platform() -> str:
    if sys.platform == "win32":
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    return "darwin"


def bundled_modules(names: list[str]) -> list[str]:
    """Package names that already resolve inside the bundle."""
    found = []
    for name in names:
        module = name.replace("-", "_")
        try:
            if importlib.util.find_spec(module) is not None:
                found.append(name)
        except (ImportError, ValueError):
            continue
    return found


def install_packages(
    names: list[str], wheelhouse: str | None = None
) -> tuple[bool, str]:
    """Install wheels into :func:`user_packages_dir` with ``uv``."""
    if not names:
        return False, "No packages were given"

    uv = find_uv()
    if uv is None:
        return False, (
            "uv is not available. Copy uv.exe next to the FalcoCV executables, "
            "set FALCOCV_UV, or install uv on PATH."
        )

    target = user_packages_dir()
    target.mkdir(parents=True, exist_ok=True)

    command = [
        str(uv),
        "pip",
        "install",
        "--target",
        str(target),
        "--python-version",
        f"{sys.version_info.major}.{sys.version_info.minor}",
        "--python-platform",
        _uv_platform(),
        "--only-binary=:all:",
        *names,
    ]

    links = wheelhouse or os.environ.get(WHEELHOUSE_ENV)
    if links:
        command += ["--find-links", str(links), "--no-index"]

    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    output = (result.stdout + result.stderr).strip()
    return result.returncode == 0, output


def _dist_info_name(dist: Path, suffix: str = ".dist-info") -> tuple[str, str]:
    stem = dist.name[: -len(suffix)]
    name, _, version = stem.rpartition("-")
    return (name or stem, version)


def _normalize(name: str) -> str:
    return name.lower().replace("-", "_").replace(".", "_")


def list_packages() -> list[dict]:
    """Distributions installed in :func:`user_packages_dir`."""
    root = user_packages_dir()
    if not root.is_dir():
        return []
    packages = []
    for dist in sorted(root.glob("*.dist-info")):
        name, version = _dist_info_name(dist)
        packages.append({"name": name, "version": version, "path": str(dist)})
    return packages


def remove_package(name: str) -> tuple[bool, str]:
    """Remove a distribution installed in :func:`user_packages_dir`."""
    root = user_packages_dir()
    wanted = _normalize(name)
    matches = [
        dist
        for dist in root.glob("*.dist-info")
        if _normalize(_dist_info_name(dist)[0]) == wanted
    ]
    if not matches:
        return False, f"{name} is not installed in {root}"

    removed = 0
    for dist in matches:
        record = dist / "RECORD"
        if record.is_file():
            for line in record.read_text(encoding="utf-8").splitlines():
                relative = line.split(",", 1)[0].strip()
                if not relative:
                    continue
                path = root / relative
                try:
                    if path.is_file() or path.is_symlink():
                        path.unlink()
                        removed += 1
                except OSError:
                    continue
        shutil.rmtree(dist, ignore_errors=True)

    # Drop directories left empty by the removal.
    for path in sorted(root.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if path.is_dir():
            try:
                path.rmdir()
            except OSError:
                pass

    return True, f"removed {name} ({removed} files)"
