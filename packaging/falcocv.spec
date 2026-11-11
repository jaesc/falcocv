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
    """The pybind11 kernel lives in the package (wheel) or ``cpp/bin`` (dev)."""
    candidates = list(PACKAGE_DIR.glob("_falco_kernel*")) or list(
        (ROOT / "cpp" / "bin").glob("_falco_kernel*")
    )
    return [
        (str(path), "falcocv")
        for path in candidates
        if path.suffix in {".pyd", ".so"}
    ]


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
