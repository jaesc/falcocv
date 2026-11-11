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
  FalcoCVCLI.exe     self-test, run, native-deps, download-models
  models/            ONNX models and the OCR dictionary
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
