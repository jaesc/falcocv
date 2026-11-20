# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the FalcoCV standalone (onedir) distribution.

Run it through ``scripts/build_standalone.py``: the script generates the
Windows version resource and copies the models next to the executables.

Two bootloaders are built from a single analysis:

* ``FalcoCV`` (windowed) -> copied to ``FalcoCV.exe`` and ``FalcoCVHMI.exe``
* ``FalcoCVConsole``     -> copied to ``FalcoCVServer.exe`` and ``FalcoCVCLI.exe``
"""

import sys
from pathlib import Path

from PyInstaller.utils.hooks import (
    collect_data_files,
    collect_dynamic_libs,
    collect_submodules,
)

SPEC_DIR = Path(SPECPATH).resolve()
ROOT = SPEC_DIR.parent
SRC = ROOT / "src"

sys.path.insert(0, str(SRC))

import falcocv  # noqa: E402

PACKAGE_DIR = Path(falcocv.__file__).resolve().parent
ICON = PACKAGE_DIR / "ui" / "assets" / "icon.ico"
VERSION_FILE = SPEC_DIR / "version_info.txt"
IS_WINDOWS = sys.platform == "win32"


def _kernel_binaries():
    """Locate the pybind11 kernel in the source tree, site-packages or cpp/bin.

    Editable installs build the extension into ``site-packages/falcocv`` while
    ``falcocv.__file__`` points at ``src/falcocv``.  On Windows the CMake
    ``LIBRARY_OUTPUT_DIRECTORY`` does not apply to the ``.pyd`` runtime output,
    so ``cpp/bin`` may also be empty.
    """
    candidates = []
    for folder in (PACKAGE_DIR, ROOT / "src" / "falcocv"):
        candidates.extend(folder.glob("_falco_kernel*"))

    try:
        import site
        import sysconfig

        site_dirs = set(site.getsitepackages())
        purelib = sysconfig.get_paths().get("purelib")
        if purelib:
            site_dirs.add(purelib)
        for site_dir in site_dirs:
            candidates.extend((Path(site_dir) / "falcocv").glob("_falco_kernel*"))
    except Exception:
        pass

    candidates.extend((ROOT / "cpp" / "bin").glob("_falco_kernel*"))

    seen = set()
    binaries = []
    for path in candidates:
        if path.suffix not in {".pyd", ".so"} or path.name in seen:
            continue
        seen.add(path.name)
        binaries.append((str(path), "falcocv"))
    return binaries


def _data_files():
    data = [
        (str(PACKAGE_DIR / "ui" / "assets"), "falcocv/ui/assets"),
        (str(PACKAGE_DIR / "web" / "hmi"), "falcocv/web/hmi"),
    ]
    for package in ("pyzbar", "pylibdmtx", "harvesters", "genicam"):
        try:
            data += collect_data_files(package)
        except Exception:
            pass
    return data


def _imagecodecs_codecs():
    """imagecodecs loads its 60+ codec extensions dynamically."""
    try:
        import imagecodecs
    except Exception:
        return []
    folder = Path(imagecodecs.__file__).resolve().parent
    patterns = ("*.pyd",) if IS_WINDOWS else ("*.so", "*.so.*")
    found = []
    for pattern in patterns:
        found.extend(folder.glob(pattern))
    return [(str(path), "imagecodecs") for path in sorted(set(found))]


def _binaries():
    binaries = []
    for package in ("pyzbar", "pylibdmtx", "imagecodecs"):
        try:
            binaries += collect_dynamic_libs(package)
        except Exception:
            pass
    return binaries + _kernel_binaries() + _imagecodecs_codecs()


hiddenimports = [
    # PyOpenGL loads its platform backend dynamically; the standard hook only
    # covers glx/win32/darwin. FalcoCV selects EGL on Wayland.
    "OpenGL.platform.egl",
    "OpenGL.platform.osmesa",
    "falcocv.web.server",
    "falcocv.web.hmi_worker",
    "falcocv.core.ocr.industrial_ocr_engine",
    "uvicorn.logging",
    "uvicorn.loops.auto",
    "uvicorn.protocols.http.auto",
    "uvicorn.protocols.http.h11_impl",
    "uvicorn.protocols.websockets.auto",
    "uvicorn.protocols.websockets.websockets_impl",
    "uvicorn.lifespan.on",
    "websockets.legacy.server",
    "watchfiles",
]
for package in ("falcocv.tools", "falcocv.comms.drivers", "falcocv.comms.servers"):
    hiddenimports += collect_submodules(package)

excludes = [
    "tkinter",
    "matplotlib",
    "scipy",
    "pandas",
    "IPython",
    "pytest",
    "PyQt6.QtWebEngineCore",
    "PyQt6.QtWebEngineWidgets",
    "PyQt6.QtWebEngineQuick",
    "PyQt6.QtQuick",
    "PyQt6.QtQml",
    "PyQt6.QtQuickWidgets",
    "PyQt6.Qt3DCore",
    "PyQt6.Qt3DRender",
    "PyQt6.QtMultimedia",
    "PyQt6.QtMultimediaWidgets",
    "PyQt6.QtDesigner",
    "PyQt6.QtHelp",
    "PyQt6.QtBluetooth",
    "PyQt6.QtNfc",
    "PyQt6.QtPositioning",
    "PyQt6.QtSerialPort",
    "PyQt6.QtTest",
]

a = Analysis(
    [str(SPEC_DIR / "entrypoint.py")],
    pathex=[str(SRC)],
    binaries=_binaries(),
    datas=_data_files(),
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    noarchive=False,
)

pyz = PYZ(a.pure)


def make_exe(name, console):
    options = dict(
        name=name,
        console=console,
        debug=False,
        strip=False,
        upx=False,
        runtime_tmpdir=None,
        exclude_binaries=True,
    )
    if ICON.exists():
        options["icon"] = str(ICON)
    if IS_WINDOWS and VERSION_FILE.exists():
        options["version"] = str(VERSION_FILE)
    return EXE(pyz, a.scripts, [], **options)


exe_gui = make_exe("FalcoCV", console=False)
exe_console = make_exe("FalcoCVConsole", console=True)

coll = COLLECT(
    exe_gui,
    exe_console,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="FalcoCV",
)
