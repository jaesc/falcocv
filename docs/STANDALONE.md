# FalcoCV — Standalone bundle

The PyPI wheel is the integration path for Python users.  The standalone
bundle is the end-user path: it ships the interpreter, Qt, OpenCV, ONNX
Runtime, the native kernel and the models, so the customer installs FalcoCV
without Python or pip.

## Layout

```
FalcoCV/
  FalcoCV.exe        GUI (Vision Editor)
  FalcoCVHMI.exe     HMI Designer
  FalcoCVServer.exe  web dashboard, HMI runtime and `--worker` processes
  FalcoCVCLI.exe     self-test, run, packages, python, native-deps
  uv.exe             installs user wheels (`falcocv packages install`)
  models/            ONNX models and the OCR dictionary
  site-packages/     drop-in user libraries (appended to sys.path)
  _internal/         Python 3.13, packages, Qt plugins and the native kernel
```

`FalcoCV.exe` and `FalcoCVHMI.exe` are the same windowed bootloader;
`FalcoCVServer.exe` and `FalcoCVCLI.exe` are the same console bootloader.  The
single entry script (`packaging/entrypoint.py`) dispatches on the executable
name, so the build only analyses the dependency graph once.

## Build

The bundle is built with PyInstaller `onedir`; cross-compilation is not
supported, so each platform builds on its own machine.

```bash
pip install -e ".[dev,plc,protocols,standalone]"
python scripts/build_standalone.py            # dist/standalone/FalcoCV
python scripts/build_standalone.py --zip      # + portable ZIP
python scripts/build_standalone.py --no-models
python scripts/build_standalone.py --clean    # drop the PyInstaller cache
```

A cold build takes ~2 minutes on Windows x64; the ZIP adds a few seconds.
Models are copied next to the executables unless `--no-models` is used.

## Smoke test

```bash
python scripts/smoke_standalone.py
```

It checks the four executables, the native kernel, the models directory,
`FalcoCVCLI self-test` (tools, config widgets, help, pipeline, job round-trip
and a kernel call), `FalcoCVCLI native-deps`, and starts `FalcoCVServer` to
verify `/api/health` reports `standalone: true`.

## Runtime layout

`falcocv/core/runtime.py` is the only place that knows about frozen builds:

| Helper | Development | Frozen bundle |
|---|---|---|
| `is_frozen()` | `False` | `True` |
| `app_dir()` | repository root | directory with the executables |
| `bundle_dir()` | repository root | `sys._MEIPASS` |
| `server_executable()` | `<root>/FalcoCVServer[.exe]` | `<app_dir>/FalcoCVServer[.exe]` |

Do not hardcode `.exe` names, `sys._MEIPASS` or `sys.executable` paths in tool
code; use these helpers.

Models are resolved in this order: `FALCOCV_MODELS_DIR`, the development
checkout, `<app_dir>/models` (standalone bundle), then the user data
directory.

## User libraries

Pipeline scripts (`PythonScriptTool`) and HMI scripts run inside the FalcoCV
process, so their imports come from `sys.path`.  User directories are
**appended**, never prepended: the packages bundled with FalcoCV (numpy,
OpenCV, Qt, ONNX Runtime) always win, so a user wheel can never shadow them
with an incompatible build.

Resolution order:

1. `FALCOCV_USER_LIBS` - `os.pathsep` separated list of directories.
2. `<install>/site-packages` - drop-in directory for the integrator.
3. `%APPDATA%/FalcoCV/user-packages` (`$XDG_DATA_HOME/FalcoCV/user-packages`
   on Linux) - managed by `falcocv packages`.

```bash
FalcoCVCLI packages install pandas requests
FalcoCVCLI packages install --find-links D:\wheels openpyxl   # offline
FalcoCVCLI packages list
FalcoCVCLI packages remove pandas
FalcoCVCLI packages dir
FalcoCVCLI python -c "import pandas; print(pandas.__version__)"
```

`uv.exe` ships next to the executables and installs only binary wheels
(`cp313`) for the frozen interpreter.  Offline plants set
`FALCOCV_WHEELHOUSE` or pass `--find-links`.  If `uv` is missing, the rest of
the application keeps working and only `packages install` is disabled.

Limitations: a user package that requires a different numpy/OpenCV version
than the bundled one will use the bundled copy; packages that need a C
compiler cannot be installed (wheels only).

## Installers

### Windows (Inno Setup 6)

```bash
python scripts/build_standalone.py --zip --installer
# dist/standalone/FalcoCV-<version>-windows-x64-setup.exe
```

- Per-user install by default; the privilege dialog allows a machine-wide one.
- Start-menu shortcuts, optional desktop icon, `.cvj` → Vision Editor and
  `.hmi` → HMI Designer associations.
- Silent install for deployments:
  `FalcoCV-<version>-windows-x64-setup.exe /VERYSILENT /DIR="C:\FalcoCV"`.
- Uninstall removes the bundle and the associations; `%APPDATA%\FalcoCV`
  (config, models, user packages, native DLLs) is preserved on purpose.
- Rebuild only the installer after editing `packaging/falcocv.iss`:
  `python scripts/build_standalone.py --skip-build --installer`.

### Linux (AppImage)

```bash
# appimagetool on PATH or APPIMAGETOOL=/path/to/appimagetool
python scripts/build_standalone.py --zip --appimage
# dist/standalone/FalcoCV-<version>-linux-x86_64.AppImage
```

Build on Ubuntu 22.04 or newer (PyQt6 needs glibc 2.34).  The AppDir embeds
`AppRun`, the desktop entry and the MIME definitions for `.cvj`/`.hmi`.

### CI

`.github/workflows/standalone.yml` builds both platforms in parallel on tags
`v*` or `workflow_dispatch`, runs the smoke test and uploads the ZIP,
installer and AppImage as artifacts.

## Platform notes

- **Windows**: the bundle is the input for the Inno Setup installer
  (Program Files or per-user, shortcuts, `.cvj`/`.hmi` associations).
- **Linux**: build on Ubuntu 22.04 or newer (PyQt6 needs glibc 2.34) and wrap
  the bundle in an AppImage.  The server executable has no `.exe` suffix; the
  runtime helper handles that.
- **Native vendor DLLs** (ADLINK DASK, Beckhoff `TcAdsDll.dll`) are not
  bundled; they are resolved from `%APPDATA%/FalcoCV/native/<driver>`, the
  system path or the vendor installation.  `falcocv native-deps` reports them.

## Pitfalls

- `imagecodecs` loads its 60+ codec extensions dynamically; the spec collects
  every `_*.pyd`/`_*.so` from the package explicitly.
- `uvicorn.run("module:app")` hides the app module from the analysis; the
  frozen server passes the app object instead.
- The HMI worker is spawned as `<server exe> --worker`, which is why the
  console bootloader must keep working with inherited stdout/stdin pipes.
- Do not add UPX or `--onefile`: both inflate build time and trigger
  antivirus false positives.
