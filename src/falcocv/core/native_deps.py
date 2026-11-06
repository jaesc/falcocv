"""Resolution of optional native libraries used by device drivers.

Vendor SDKs are deliberately not bundled in the FalcoCV wheel.  A driver can
resolve a library from an explicit path, from the per-user native directory or
from the normal system search path.
"""

from __future__ import annotations

import ctypes.util
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .environment import APP_DATA


def native_root() -> Path:
    """Return the root directory for optional native libraries."""

    override = os.environ.get("FALCOCV_NATIVE_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return (APP_DATA / "native").resolve()


def native_driver_dir(driver: str) -> Path:
    return native_root() / driver


def architecture_suffix() -> str:
    """Return the compiler architecture suffix used by Windows DLL names."""

    return "64" if sys.maxsize > 2**32 else ""


@dataclass(frozen=True)
class NativeLibrary:
    """Result of resolving one native library."""

    path: Path | None
    source: str
    searched: tuple[Path, ...] = ()

    @property
    def found(self) -> bool:
        return self.path is not None and self.path.is_file()


def _candidate_names(names: Sequence[str]) -> tuple[str, ...]:
    return tuple(name for name in names if name)


def resolve_library(
    names: Sequence[str],
    *,
    driver: str,
    configured: str = "",
) -> NativeLibrary:
    """Resolve a native library without loading it.

    Search order:

    1. ``configured`` file or directory.
    2. ``<native root>/<driver>/<name>``.
    3. System library search (``PATH`` and ``ctypes.util.find_library``).
    """

    names = _candidate_names(names)
    searched: list[Path] = []
    if configured:
        configured_path = Path(configured).expanduser()
        if configured_path.is_file():
            return NativeLibrary(
                configured_path.resolve(), "configured file"
            )
        if configured_path.is_dir():
            for name in names:
                candidate = configured_path / name
                searched.append(candidate)
                if candidate.is_file():
                    return NativeLibrary(
                        candidate.resolve(), "configured directory", tuple(searched)
                    )
        else:
            searched.append(configured_path)

    directory = native_driver_dir(driver)
    for name in names:
        candidate = directory / name
        searched.append(candidate)
        if candidate.is_file():
            return NativeLibrary(candidate.resolve(), "native directory", tuple(searched))

    for name in names:
        found = shutil.which(name)
        if found:
            return NativeLibrary(Path(found).resolve(), "system", tuple(searched))
        by_module = ctypes.util.find_library(Path(name).stem)
        if by_module:
            candidate = Path(by_module)
            if candidate.is_file():
                return NativeLibrary(candidate.resolve(), "system", tuple(searched))

    return NativeLibrary(None, "not found", tuple(searched))


def resolve_tcads_dll() -> NativeLibrary:
    """Resolve pyads' ``TcAdsDll.dll`` from native, PATH or TwinCAT."""

    searched: list[Path] = []
    native = native_driver_dir("beckhoff") / "TcAdsDll.dll"
    searched.append(native)
    if native.is_file():
        return NativeLibrary(native.resolve(), "native directory", tuple(searched))

    twin_dir = os.environ.get("TWINCAT3DIR")
    if twin_dir:
        root = Path(twin_dir).expanduser()
        common = "Common64" if sys.maxsize > 2**32 else "Common32"
        for base in (root.parent, root):
            candidate = base / common / "TcAdsDll.dll"
            searched.append(candidate)
            if candidate.is_file():
                return NativeLibrary(
                    candidate.resolve(),
                    "TWINCAT3DIR",
                    tuple(searched),
                )

    result = resolve_library(("TcAdsDll.dll",), driver="beckhoff")
    if result.found:
        return result
    return NativeLibrary(None, result.source, tuple(searched) + result.searched)


def add_native_dll_directory(driver: str):
    """Add the per-user native directory to the Windows DLL search path.

    Returns an ``os.add_dll_directory`` handle that the caller must keep alive
    while the library is loaded, or ``None`` when there is nothing to add.
    """

    if os.name != "nt":
        return None
    directory = native_driver_dir(driver)
    if not directory.is_dir():
        return None
    return os.add_dll_directory(str(directory))


def native_library_hint(driver: str, names: Sequence[str]) -> str:
    """Return a user-facing hint describing where to install a library."""

    directory = native_driver_dir(driver)
    files = ", ".join(names)
    return (
        f"Put {files} in {directory} "
        "(or set FALCOCV_NATIVE_DIR to a portable native folder)."
    )
