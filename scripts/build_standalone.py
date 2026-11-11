"""Build the FalcoCV standalone (PyInstaller onedir) bundle.

The script assumes the current interpreter already has the package installed
(``pip install -e .[dev,plc,protocols,standalone]``) and only orchestrates:

1. version stamping for the Windows executable resource,
2. a single PyInstaller run from ``packaging/falcocv.spec``,
3. the final executable names expected by the runtime,
4. the ``models`` directory next to the executables,
5. an optional portable ZIP.

Usage:
    python scripts/build_standalone.py
    python scripts/build_standalone.py --no-models --clean --zip
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = ROOT / "packaging" / "falcocv.spec"
VERSION_PY = ROOT / "src" / "falcocv" / "version.py"
VERSION_RESOURCE = ROOT / "packaging" / "version_info.txt"
BUNDLE_NAME = "FalcoCV"

# Final executable name -> bootloader produced by the spec.
EXECUTABLES = {
    "FalcoCV": "gui",
    "FalcoCVHMI": "gui",
    "FalcoCVServer": "console",
    "FalcoCVCLI": "console",
}


def read_version() -> str:
    match = re.search(
        r'__version__\s*=\s*"([^"]+)"', VERSION_PY.read_text(encoding="utf-8")
    )
    if not match:
        raise SystemExit(f"Could not read __version__ from {VERSION_PY}")
    return match.group(1)


def _version_tuple(version: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in re.findall(r"\d+", version)[:4]]
    return tuple(parts + [0] * (4 - len(parts)))  # type: ignore[return-value]


def write_version_resource(version: str) -> Path:
    filevers = prodvers = _version_tuple(version)
    content = f"""VSVersionInfo(
  ffi=FixedFileInfo(
    filevers={filevers},
    prodvers={prodvers},
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0),
  ),
  kids=[
    StringFileInfo([
      StringTable('040904B0', [
        StringStruct('CompanyName', 'FalcoCV contributors'),
        StringStruct('FileDescription', 'FalcoCV - industrial machine vision'),
        StringStruct('FileVersion', '{version}'),
        StringStruct('InternalName', 'FalcoCV'),
        StringStruct('OriginalFilename', 'FalcoCV.exe'),
        StringStruct('ProductName', 'FalcoCV'),
        StringStruct('ProductVersion', '{version}'),
      ])
    ]),
    VarFileInfo([VarStruct('Translation', [1033, 1200])]),
  ],
)
"""
    VERSION_RESOURCE.write_text(content, encoding="utf-8")
    return VERSION_RESOURCE


def run_pyinstaller(dist: Path, work: Path, clean: bool) -> None:
    try:
        import PyInstaller  # noqa: F401
    except ImportError:
        raise SystemExit(
            "PyInstaller is missing. Install the build extra first:\n"
            '  pip install -e ".[dev,plc,protocols,standalone]"'
        )

    command = [
        sys.executable,
        "-m",
        "PyInstaller",
        str(SPEC),
        "--noconfirm",
        "--distpath",
        str(dist),
        "--workpath",
        str(work),
    ]
    if clean:
        command.append("--clean")
    print("[build] " + " ".join(command))
    subprocess.run(command, cwd=ROOT, check=True)


def rename_executables(bundle: Path) -> None:
    suffix = ".exe" if sys.platform == "win32" else ""
    sources = {
        "gui": bundle / f"FalcoCV{suffix}",
        "console": bundle / f"FalcoCVConsole{suffix}",
    }
    for name, kind in EXECUTABLES.items():
        source = sources[kind]
        target = bundle / f"{name}{suffix}"
        if not source.exists():
            raise SystemExit(f"missing bootloader {source}")
        if target.resolve() != source.resolve():
            shutil.copy2(source, target)
        if sys.platform != "win32":
            target.chmod(0o755)

    console_stub = sources["console"]
    if console_stub.exists() and console_stub.name not in {
        f"{name}{suffix}" for name in EXECUTABLES
    }:
        console_stub.unlink()


def copy_models(bundle: Path, models_dir: Path) -> None:
    if not models_dir.is_dir():
        print(f"[build] models directory not found: {models_dir} (skipped)")
        return
    target = bundle / "models"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(models_dir, target)
    size_mb = sum(p.stat().st_size for p in target.rglob("*") if p.is_file()) / 1e6
    print(f"[build] models: {size_mb:.1f} MB")


def copy_uv(bundle: Path) -> None:
    """Ship ``uv`` so users can install their own wheels at runtime."""
    alongside_python = Path(sys.executable).parent / (
        "uv.exe" if os.name == "nt" else "uv"
    )
    uv = shutil.which("uv")
    if alongside_python.is_file():
        uv = str(alongside_python)
    if not uv:
        print(
            "[build] uv not found: 'falcocv packages install' will be disabled "
            '(pip install ".[standalone]")'
        )
        return
    target = bundle / Path(uv).name
    shutil.copy2(uv, target)
    print(f"[build] uv: {target.name} ({target.stat().st_size / 1e6:.1f} MB)")


def create_site_packages(bundle: Path) -> None:
    """Create the administrator drop-in directory for user libraries."""
    target = bundle / "site-packages"
    target.mkdir(exist_ok=True)
    readme = target / "README.txt"
    if not readme.exists():
        readme.write_text(
            "Drop pure-Python packages or unpacked cp313 wheels here.\n"
            "This directory is appended to sys.path, so the packages bundled\n"
            "with FalcoCV keep priority.\n\n"
            "Use 'FalcoCVCLI packages install <name>' to install into\n"
            "%APPDATA%/FalcoCV/user-packages instead.\n",
            encoding="utf-8",
        )


def make_zip(bundle: Path, version: str) -> Path:
    platform = "windows" if sys.platform == "win32" else "linux"
    archive = bundle.parent / f"FalcoCV-{version}-{platform}-x86_64"
    # compresslevel=1 keeps the release build in the "minutes" budget.
    import zipfile

    zip_path = archive.with_suffix(".zip")
    zip_path.unlink(missing_ok=True)
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                zf.write(path, path.relative_to(bundle.parent))
    return zip_path


def report(bundle: Path) -> None:
    files = [p for p in bundle.rglob("*") if p.is_file()]
    size_mb = sum(p.stat().st_size for p in files) / 1e6
    kernel = list((bundle / "_internal" / "falcocv").glob("_falco_kernel*"))
    print(f"[build] bundle: {bundle}")
    print(f"[build] size:   {size_mb:.1f} MB in {len(files)} files")
    print(f"[build] kernel: {kernel[0].name if kernel else 'MISSING'}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the FalcoCV standalone bundle")
    parser.add_argument(
        "--dist", default=str(ROOT / "dist" / "standalone"), help="Output directory"
    )
    parser.add_argument(
        "--work", default=str(ROOT / "build" / "standalone"), help="Work directory"
    )
    parser.add_argument(
        "--models-dir", default=str(ROOT / "models"), help="Models to bundle"
    )
    parser.add_argument(
        "--no-models", action="store_true", help="Do not copy the models directory"
    )
    parser.add_argument("--zip", action="store_true", help="Create a portable ZIP")
    parser.add_argument(
        "--clean", action="store_true", help="Remove the PyInstaller cache first"
    )
    args = parser.parse_args(argv)

    started = time.time()
    version = read_version()
    dist = Path(args.dist).resolve()
    work = Path(args.work).resolve()

    print(f"[build] FalcoCV {version}")
    write_version_resource(version)
    run_pyinstaller(dist, work, args.clean)

    bundle = dist / BUNDLE_NAME
    if not bundle.is_dir():
        raise SystemExit(f"PyInstaller did not produce {bundle}")

    rename_executables(bundle)
    copy_uv(bundle)
    create_site_packages(bundle)
    if not args.no_models:
        copy_models(bundle, Path(args.models_dir).resolve())
    report(bundle)

    if args.zip:
        archive = make_zip(bundle, version)
        print(f"[build] zip:    {archive} ({archive.stat().st_size / 1e6:.1f} MB)")

    print(f"[build] done in {time.time() - started:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
