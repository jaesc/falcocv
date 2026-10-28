# FalcoCV — System Architecture

## 1. Overview

FalcoCV is an industrial machine vision framework with two operating modes:
**desktop** (workflow editor) and **HMI** (operator interface via web).
Both share the same execution engine (`src/falcocv/core/`) but have separate runtimes
with process isolation to prevent C-level resource leaks.

> **Current architecture (hybrid):** the core and all tools are Python. The
> only native code is `cpp/kernel` (`_falco_kernel`), which implements
> point-cloud generation, plane fitting and shape-match voting because
> benchmarks showed a >2× end-to-end win. It has no OpenCV dependency. See
> `benchmarks/bench_tools.py`.

---

## 2. Process tree

```
┌──────────────────────────┐  ┌──────────────────────┐  ┌──────────────────────┐
│  python -m falcocv          │  │  falcocv-hmi        │  │  python -m falcocv.web       │
│  ─────────────────────── │  │                     │  │  (or falcocv-server)  │
│  PyQt6 + WorkflowEditor  │  │  ─────────────────── │  │  ─────────────────── │
│  Workflow editor         │  │  standalone Qt modal │  │  FastAPI + uvicorn   │
│  HMI Designer (modal)    │  │  No server required  │  │  + WebSocket server  │
│                          │  │  Optional server     │  │  Dashboard at /      │
└──────────────────────────┘  └──────────────────────┘  └──────────┬───────────┘
                                                                   │
                          ┌────────────────────────────────────────┘
                          │  (n processes, one per project_id)
                    ┌─────▼──────┐  ┌─────▼──────┐
                    │ hmi_worker │  │ hmi_worker │  ...
                    │ (ws10)     │  │ (ws20)     │
                    │ subprocess │  │ subprocess │
                    │ isolated   │  │ isolated   │
                    └────────────┘  └────────────┘
```

> FalcoCV runs **from source**. No packaged binaries are distributed: anyone who needs an
> executable can generate it with PyInstaller/Nuitka, but that packaging is not part of the
> repository build.

### Web server (Process B)

```
GET  /                     → dashboard.html    (landing page with all projects)
GET  /?id=project          → index.html         (HMI for the specific project)
GET  /api/server/projects  → list {project_id, alive, pid, memory_mb, uptime, slots, grid}
GET  /api/hmi?pid=xxx      → layout + slots of project xxx
POST /api/hmi/load         → receives .hmi, spawns worker, returns project_id
POST /api/hmi/reload       → redeploy with cached payload (same project_id)
POST /api/hmi/stop         → kills worker and releases resources
WS   /api/ws?pid=xxx       → runtime events + debug commands
```

### Why subprocesses instead of threads

| Reason                       | Thread              | Subprocess                                       |
|------------------------------|---------------------|--------------------------------------------------|
| OpenCV/CUDA crash            | Kills the server    | Only the worker dies                             |
| `sys.settrace`               | Affects the async loop | Isolated in the worker                        |
| C-level leaks (Harvester, CUDA) | Accumulate across cycles | OS releases everything on process kill      |
| Resource cleanup             | Manual, error-prone | `process.kill()` = full cleanup                  |
| Memory isolation             | Shared              | Independent                                      |

### Worker lifecycle

```
  Deploy                Runtime               Stop
  ───────               ───────               ─────
  POST /api/hmi/load    worker in loop        POST /api/hmi/stop
     │                     │                     │
     ▼                     ▼                     ▼
  spawn worker.py       receives start cmd    receives stop cmd
     │                     │                     │
     ▼                     ▼                     ▼
  send start cmd        loads jobs            stop_script()
  + project JSON           │                    │
  + breakpoints            ▼                    ▼
     │                 runs script            disconnect cameras
     ▼                    │                    │
  worker sends            ▼                    ▼
  "ready"             events via stdout      raise SystemExit
     │                 state_changed            │
     ▼                 slot_finished            ▼
  server responds     paused/resumed        process ends
  200 OK              logs/status               │
                                                 ▼
                                            OS releases resources
```

---

## 3. Project structure

```
FalcoCV/
??? src/falcocv/
?   ??? core/                  # workflow, VisionImage, overlays, ROI, models
?   ??? tools/                 # logic.py (processing) + config.py (PyQt6)
?   ??? ui/                    # viewers, ROI editors, HMI designer, assets
?   ??? web/                   # FastAPI server, HMI workers, dashboard
?   ??? main.py                # Vision Editor
?   ??? hmi_designer.py        # standalone HMI Designer
?   ??? web_runner.py          # standalone web server
?   ??? cli.py                 # falcocv / python -m falcocv
?   ??? version.py
??? cpp/kernel/                # _falco_kernel: point cloud + plane fit (no OpenCV)
??? tests/                     # pytest suite
??? models/                    # local models for development (downloaded in production)
??? docs/                      # tool help + HMI API reference
??? pyproject.toml             # scikit-build-core + metadata
??? README.md
??? ARCHITECTURE.md
??? AGENTS.md
??? CREDITS.md
```

---

## 4. Design principles

### 4.1 Core ↔ UI separation

```
src/falcocv/core/  ──── 0 PyQt6 imports ──────────────
  │                                      │
  │  CallbackList (replaces pyqtSignal)  │
  │  abstract ABCs                       │
  │                                      │
src/falcocv/ui/   ──── PyQt6 only ────────────────────
```

- `core/` does not know Qt. It uses `CallbackList` (a list of callbacks) instead of `pyqtSignal`.
- `BaseTool` is a pure ABC, with no `QObject` and no `get_config_widget()`.
- The `ToolConfigDialog` (Qt) is created externally and injected.

### 4.2 Injected scheduler

```python
# core/workflow_controller.py
controller.set_scheduler(lambda fn: QTimer.singleShot(0, fn))
```

- `WorkflowController` (pure Python) does not know Qt.
- It receives an abstract scheduler via callback.
- On desktop: `QTimer.singleShot(0, ...)`.
- In tests: `lambda fn: fn()` (synchronous).

### 4.3 Tool registry vs factory

`core/tool_registry.py` exposes two APIs:

| API | Used by | Purpose |
|-----|-----------|-----------|
| `@register_tool(prefix, default_name, group)` | Each `tools/X/logic.py` | Class decorator: registers the tool in `_TOOL_REGISTRY` |
| `create_tool(tool_type, **kwargs)` | `Workflow.deserialize()` | Factory: instantiates a tool by type name |
| `get_all_tools()` | `ui/widgets/toolbox.py` | Populate the toolbox with all available tools |
| `get_registration(tool_type)` | `ui/widgets/workflow_editor.py` | Get ToolReg to create a new tool |
| `get_registration_by_class(cls)` | `ui/widgets/workflow_tree.py` | Get the prefix of an existing tool |

The worker **does not use the registry**. It creates tools via `Workflow.create_tool_from_data()` with lazy imports.

---

## 5. HMI endpoints

The `src/falcocv/web/server.py` server only exposes HMI endpoints (no leftovers from the web editor).
All endpoints accept `project_id` for concurrent multi-HMI support.

### Frontend

| Route | Serves | Description |
|------|-------|-------------|
| `/` | `dashboard.html` | Landing page: lists all projects with PID, memory, uptime |
| `/?id=xxx` | `index.html` | HMI of project `xxx` |
| `/hmi` | `index.html` | HMI without a specific project |
| `/hmi/**` | `hmi/*` | Static assets (CSS, JS) |

### REST API

| Method | Route | Parameters | Description |
|--------|------|------------|-------------|
| `GET` | `/api/server/projects` | — | Lists all workers: `{project_id, alive, pid, memory_mb, uptime, layout, slots}` |
| `GET` | `/api/hmi` | `?project_id=` | Layout + slots of the project (or default if dead) |
| `POST` | `/api/hmi/load` | body: `{script, layout, slots, files?, entry_point?, bp_by_file, project_id}` | Deploy: spawns worker, returns `{project_id}` |
| `POST` | `/api/hmi/reload` | body: `{project_id}` | Re-deploy from the cached payload of the last deploy |
| `POST` | `/api/hmi/stop` | body: `{project_id}` | Kills the worker and releases resources |
| `POST` | `/api/hmi/set-bps` | body: `{bps, bps_by_file?, project_id}` | Sets breakpoints pre-deploy |
| `GET` | `/api/hmi/logs` | `?project_id=&since=N` | Incremental logs |
| `GET` | `/api/hmi/slot/{name}/output/{out}` | `?project_id=&quality=` | Image frame (base64 JPEG) |

### WebSocket

**URL:** `ws://.../api/ws?project_id=xxx`

**Events (server → client):**

| Type | Payload |
|------|---------|
| `connected` | `{project_id}` |
| `hmi_loaded` | `{slots, project_id}` |
| `hmi_stopped` | `{}` |
| `hmi_debug_paused` | `{line, file, locals}` |
| `hmi_debug_resumed` | `{}` |
| `hmi_state_changed` | `{slot, state}` |
| `hmi_slot_finished` | `{slot, error, total_time_ms, outputs}` |
| `hmi_log` | `{entry: {level, message, timestamp}}` |
| `hmi_status` | `{slot, status}` |

**Commands (client → server):**

| Command | Payload |
|---------|---------|
| `hmi_debug_continue` | `{}` |
| `hmi_debug_step_over` | `{}` |
| `hmi_debug_step_into` | `{}` |
| `hmi_debug_step_out` | `{}` |
| `hmi_debug_stop` | `{}` |
| `hmi_set_breakpoints` | `{bps: [lines] \| {filename: [lines]}}` |

---

## 6. .hmi format (HMI Project)

```json
{
    "version": "2.0",
    "project_id": "estacion20",
    "slots": [
        {"name": "job1", "job_path": "C:\\jobs\\inspection.cvj"}
    ],
    "layout": {
        "rows": 28,
        "cols": 50,
        "widgets": [
            {
                "id": "w_af29e900", "type": "ImageViewer",
                "x": 2, "y": 2, "w": 10, "h": 8, "z": 0,
                "slot": "job1", "output": "image_out", "label": "Camera View"
            }
        ]
    },
    "script": "# HMI Script\nwhile True:\n    run_slot('job1')\n    sleep(0.5)\n",
    "source_dir": "my_project/",       // multi-file only
    "entry_point": "main.py"           // multi-file only
}
```

| Field | Description |
|-------|-------------|
| `version` | `"1.0"` (single) or `"2.0"` (multi-file) |
| `project_id` | Unique identifier for the HMI (user-friendly, editable in Settings) |
| `script` | Embedded Python code (v1.0) |
| `source_dir` | Directory on disk with .py files (v2.0 multi-file) |
| `entry_point` | Main Python file inside `source_dir` |

v1.0 `.hmi` files without `project_id` get an auto-generated one on load (`uuid.uuid4()[:8]`).

### Widget types

| Type | Properties |
|------|-------------|
| `ImageViewer` | `slot`, `output`, `label` |
| `StatusIndicator` | `slot`, `label` |
| `MeasureLabel` | `slot`, `output`, `field`, `label`, `min`, `max` |
| `TextLabel` | `label` (static text) |
| `LogPanel` | (no extra properties) |

---

## 7. .cvj format (Job / Workflow)

Jobs serialized as ZIP archives containing `workflow.json`:

```json
{
    "version": "2",
    "stop_on_error": false,
    "tools": [
        {
            "type": "GigeCameraLoader",
            "params": {
                "camera_serial": "21906264",
                "cti_path": "C:\\Program Files\\...\\mvGenTLProducer.cti",
                "trigger_mode": false,
                "pixel_format": "BayerRG8"
            },
            "inputs": {},
            "outputs": {"image_out": "image"},
            "x": 0, "y": 0,
            "uuid": "a1b2c3d4-..."
        },
        {
            "type": "EdgeDetection",
            "params": {"algorithm": "canny", "threshold1": 50, "threshold2": 150},
            "inputs": {"image_in": {"source_tool": "a1b2c3d4-...", "source_output": "image_out"}},
            "outputs": {"edge_out": "image"},
            "x": 1, "y": 0,
            "uuid": "e5f6g7h8-..."
        }
    ],
    "connections": [
        {"from": "a1b2c3d4-...", "from_output": "image_out",
         "to": "e5f6g7h8-...", "to_input": "image_in"}
    ]
}
```

---

## 8. Camera management (GigeCameraLoader)

### Pool per process

```python
# tools/gige_camera/logic.py
_CAMERA_POOL = {}       # {"serial": {"h": Harvester, "ia": ImageAcquirer}}
_POOL_LOCK = threading.Lock()
```

- It is a **per-process** dict (each `sys.modules["tools.gige_camera.logic"]` has its own).
- Protected by `_POOL_LOCK` for intra-process concurrency.
- **There is no global or cross-process pool** — actual contention happens at the GenTL transport level.

### Contention between editor and HMI

```
Process A (editor) ──► Harvester.create(idx) ──► camera busy
                                                       │
Process C (worker) ──► Harvester.create(idx) ──► ❌ ERROR (already in use)
```

**Solution:** the HMI Designer calls `MainWindow._cleanup_workflow()` (in `main.py`)
before deploy, forcing `GigeCameraLoader.disconnect()` in the editor.

### disconnect()

```python
# tools/gige_camera/logic.py
def disconnect(self):
    _POOL_LOCK.acquire()
    self._ia.stop()
    self._ia.destroy()
    del _CAMERA_POOL[self.params["camera_serial"]]
    self._h.reset()
    _POOL_LOCK.release()
```

---

## 9. TraceDebugger (sys.settrace)

### How it works

```python
# core/script_debugger.py
class TraceDebugger:
    _breakpoints: dict[str, set[int]]  # {filename: line_numbers} — snapped
    _script_filename: str               # "<hmi_script>" or real file name
    _script_filenames: frozenset        # set of multi-file files to trace
    _step_mode: str | None              # "step_over" | "step_into" | "step_out"

    def _trace_func(self, frame, event, arg):
        fn = frame.f_code.co_filename
        # Filter: only HMI project files
        if self._script_filenames and fn not in self._script_filenames:
            return None
        if fn != self._script_filename and not self._script_filenames:
            return None

        # Breakpoint check: snapped + live HMI (dual)
        bps = self._breakpoints.get(fn, set())
        bps |= self.get_breakpoints_for_file(fn)
        bps |= self.get_breakpoints()
        if frame.f_lineno in bps:
            self._pause(frame)

        # Step mode: step_into / step_over / step_out
        ...
```

### Modes

| Mode | `_script_filenames` | `get_breakpoints_for_file` | Use |
|------|---------------------|---------------------------|-----|
| Single-file | `None` | lambda returning `set()` | `python_script` tool |
| HMI single | `None` | lambda returns `_hmi_breakpoints["<hmi_script>"]` | Deploy v1.0 |
| HMI multi | `frozenset` of files | lookup via `os.path.relpath(fn, source_dir)` | Deploy v2.0 |

---

## 10. Data flow: HMI deploy

```
HMI Designer (Qt)                   Server (FastAPI)                Worker (subprocess)
─────────────────                   ────────────────                ───────────────────
       │                                    │                               │
       │ 1. _cleanup_workflow()             │                               │
       │    (releases editor cameras)       │                               │
       │                                    │                               │
       │ 2. _flush_to_project()             │                               │
       │    (sync UI → HMIProject)          │                               │
       │                                    │                               │
       │ 3. _check_compile()                │                               │
       │    (syntax error → dialog)         │                               │
       │                                    │                               │
       │ 4. POST /api/hmi/load              │                               │
       │    JSON body:                      │                               │
       │    { script, layout, slots,        │                               │
       │      bp_by_file, project_id,       │                               │
       │      files?, entry_point? }        │                               │
       │───────────────────────────────────►│                               │
       │                                    │  snap BPs via AST             │
       │                                    │  _stop_worker(pid)            │
       │                                    │  (no broadcast stopped)       │
       │                                    │  worker = HMIWorkerManager()  │
       │                                    │  worker.start(...)            │
       │                                    │──────────────────────────────►│
       │                                    │                               │  load project
       │                                    │                               │  load jobs (.cvj)
       │                                    │                               │  connect slots
       │                                    │                               │  start script
       │                                    │          "ready"              │
       │                                    │◄──────────────────────────────│
       │              200 OK {project_id}   │                               │
       │◄───────────────────────────────────│                               │
       │                                    │                               │
       │ 5. WS: hmi_loaded (changes status  │                               │
       │    to "Deployed")                  │                               │
       │◄───────────────────────────────────│                               │
       │                                    │                               │
       │ 6. WS: hmi_state_changed (running) │                               │
       │◄───────────────────────────────────│                               │
```

### Hot breakpoints (after deploy)

Breakpoints are sent in the deploy payload itself (`bp_by_file`). If the user
changes a BP after deploy, it is sent via WS:

```
F9 in ScriptEditor
       │
       ▼
  _on_breakpoints_changed()
       │
       │ WS: hmi_set_breakpoints
       │ { bps: [lines] | {filename: [lines]} }
       ▼
  Server → worker: set_bps
  (updates _hmi_breakpoints in real time)
```

---

## 11. C-level resource management

### Cameras (GigeCameraLoader)

| Event | Editor (Process A) | Worker (Process C) |
|--------|-------------------|-------------------|
| HMI deploy | `_cleanup_workflow()` → `disconnect()` | `connect()` in `run()` |
| HMI stop | — | `sys.exit(0)` → OS releases GenTL |
| Run workflow in editor | `disconnect()` if the worker holds it → editor `connect()` | — |
| Close editor | `_web_stop()` → kills subprocesses | `process.kill()` → OS releases |

### OpenCV / numpy / CUDA

- Each subprocess has its own instance of the DLLs.
- When the worker is killed, GPU VRAM and CUDA contexts are released.
- Without a subprocess: VRAM leaks between deploys.

---

## 12. Multi-file HMI projects (v2.0 implemented)

### Format

```
inspection_line/
├── inspection.hmi        # layout + slots + meta (version 2.0)
│   version: "2.0"
│   source_dir: "my_project/"
│   entry_point: "main.py"
├── my_project/
│   ├── main.py           # entry point: `__import__("main")`
│   ├── utils.py
│   └── lib/
│       └── vision.py
```

### Execution in the worker

The `.py` files are packed into the payload as `files: {"main.py": "...", ...}`.
The worker writes them to a tempdir and imports the entry point:

```python
files = cmd["project"]["files"]
entry = cmd["project"].get("entry_point", "main.py")
tmpdir = tempfile.mkdtemp()
for relpath, content in files.items():
    full = os.path.join(tmpdir, relpath)
    os.makedirs(os.path.dirname(full), exist_ok=True)
    Path(full).write_text(content, encoding="utf-8")
sys.path.insert(0, tmpdir)
modname = entry.replace(".py", "").replace("/", ".").replace("\\", ".")
__import__(modname)
```

The HMI script can do normal imports:
```python
from utils import preprocess
from lib.vision import analyze
```

### Editor

- **Script tab:** horizontal QSplitter: QTreeWidget (file tree) | QTabWidget (editors)
- **Operations:** New File/Folder, Rename, Delete, Drag & Drop (moves on disk + remaps tabs)
- **Breakpoints per-file:** `{"main.py": {1, 5}, "utils.py": {3}}`
- **Debug highlight:** on the tab of the file where execution is paused

### Multi-file debugger

| Component | v1.0 | v2.0 |
|------------|------|------|
| `_breakpoints` | `set[int]` | `dict[str, set[int]]` per-file |
| Trace filter | `co_filename == "<hmi_script>"` | `co_filename in source_filenames` |
| WS paused | `{"line": 10}` | `{"file": "main.py", "line": 10}` |
| WS set_bps | `{"bps": [1, 5]}` | `{"bps": {"main.py": [1,5]}}` |
| `_run_script_*` | `_run_script_single()` | `_run_script_multi()` |

---

## 13. Configuration

### config.json (`%APPDATA%/FalcoCV/config.json`)

```json
{
    "window": {"x": 0, "y": 0, "width": 1600, "height": 900,
               "maximized": false, "editor_splitter": [300, 700]},
    "colors": {},
    "recent_files": [],
    "gige_camera": {"cti_path": ""},
    "depth_viewer": {"colormap": "TURBO", "vmin": null, "vmax": null,
                     "use_colormap": true},
    "viewer3d": {"point_size": 3.0},
    "toolbox_visible": true,
    "show_overlays": true,
    "tool_splitters": {},
    "resolution_mode": "auto",
    "theme": "dark",
    "web_port": 8000,
    "debug_max_depth": 10
}
```

| Key | Type | Default | Description |
|-------|------|---------|-------------|
| `window` | dict | — | Window geometry + editor splitter |
| `colors` | dict | — | Custom colors for tool ports |
| `recent_files` | list | [] | Recent files |
| `gige_camera.cti_path` | str | "" | Path to the GenTL Producer (`.cti`) |
| `depth_viewer` | dict | TURBO | Colormap and range of the depth viewer |
| `viewer3d.point_size` | float | 3.0 | Point size in the 3D viewer |
| `toolbox_visible` | bool | true | Visibility of the toolbox dock |
| `show_overlays` | bool | true | Show overlays in the viewer |
| `tool_splitters` | dict | {} | Splitter sizes per tool |
| `resolution_mode` | str | "auto" | `auto` / `high` / `low` |
| `theme` | str | "dark" | Qt theme |
| `web_port` | int | 8000 | Web server port |
| `debug_max_depth` | int | 10 | Maximum debugger depth |

> **Brand migration:** on first launch, if `%APPDATA%/FalcoCV` does not exist but
> `%APPDATA%/CVision` does, the contents are copied automatically (config, theme, recent files).

### `get_web_port()`

All port references use `ConfigManager().get_web_port()` —
there are no hardcoded `8000` values anywhere functional in the code.

---

## 14. Code conventions

- **Python 3.13** (`py -3.13`)
- `src/falcocv/core/`: 0 PyQt6 imports, 0 FastAPI imports
- `src/falcocv/ui/`: PyQt6 only
- `src/falcocv/web/`: FastAPI + stdlib only (0 Qt imports)
- `src/falcocv/tools/*/logic.py`: core + numpy/opencv only
- `src/falcocv/tools/*/config.py`: PyQt6 only (configuration widget). Some legacy tools still contain business logic that is being migrated to `logic.py`; see audit item #8.
- No unnecessary comments
- Types with modern syntax (`str | None`, `dict[str, int]`)
- No commits without explicit permission
