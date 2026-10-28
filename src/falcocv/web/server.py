import os
import sys
import asyncio
import json
import subprocess
import threading
import time
import uuid
import tempfile
import warnings
import psutil
warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")
warnings.filterwarnings("ignore", message="remove second argument of ws_handler")
from typing import Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

_ws_clients: dict[str, list[WebSocket]] = {}
_workers: dict[str, "HMIWorkerManager"] = {}
_pending_bps: dict[str, dict[str, list[int]]] = {}
_main_loop = None


def broadcast(project_id: str, data: dict):
    loop = _main_loop
    if loop is None or not loop.is_running():
        return
    clients = _ws_clients.get(project_id, [])
    async def _send():
        for ws in list(clients):
            try:
                await ws.send_json(data)
            except Exception:
                try:
                    _ws_clients[project_id].remove(ws)
                except (ValueError, KeyError):
                    pass
    try:
        asyncio.run_coroutine_threadsafe(_send(), loop)
    except RuntimeError:
        pass


@asynccontextmanager
async def lifespan(application: FastAPI):
    global _main_loop
    _main_loop = asyncio.get_running_loop()
    yield


app = FastAPI(title="FalcoCV Web API", version="1.0.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── WebSocket --------------------------------------------------------------

@app.websocket("/api/ws")
async def websocket_endpoint(ws: WebSocket, project_id: str = "default"):
    await ws.accept()
    _ws_clients.setdefault(project_id, []).append(ws)
    try:
        await ws.send_json({"type": "connected", "project_id": project_id})
        while True:
            msg = await ws.receive_json()
            cmd = msg.get("cmd")
            if cmd == "ping":
                await ws.send_json({"type": "pong"})
                continue
            worker = _workers.get(project_id)
            if not worker or not worker.alive:
                continue
            if cmd == "hmi_set_breakpoints":
                worker.send({"cmd": "set_bps", "bps": msg.get("bps", [])})
            elif cmd == "hmi_debug_continue":
                worker.send({"cmd": "continue"})
            elif cmd == "hmi_debug_step_over":
                worker.send({"cmd": "step_over"})
            elif cmd == "hmi_debug_step_into":
                worker.send({"cmd": "step_into"})
            elif cmd == "hmi_debug_step_out":
                worker.send({"cmd": "step_out"})
            elif cmd == "hmi_debug_stop":
                worker.send({"cmd": "debug_stop"})
            elif cmd == "prompt_response":
                worker.send(msg)
    except WebSocketDisconnect:
        pass
    finally:
        try:
            _ws_clients[project_id].remove(ws)
        except (ValueError, KeyError):
            pass


# ── HMI Subprocess Worker ----------------------------

class HMIWorkerManager:
    def __init__(self, project_id: str = "default"):
        self.project_id = project_id
        self.process: subprocess.Popen | None = None
        self._reader: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._pending: dict[str, asyncio.Event] = {}
        self._pending_results: dict[str, Any] = {}
        self._last_payload: dict | None = None
        self.layout: dict = {}
        self.slots: list[dict] = []
        self.logs: list[dict] = []

    @property
    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def set_loop(self, loop):
        self._loop = loop

    def start(self, script: str, bps: set[int] | dict[str, list[int]], layout: dict,
              slots_raw: list[dict], files: dict[str, str] | None = None,
              entry_point: str = "main.py"):
        self.stop()
        self.layout = dict(layout)
        self.slots = [
            {
                "name": sd.get("name", ""),
                "job_path": sd.get("job_path", ""),
                "state": "unloaded",
                "last_error": None,
                "last_time_ms": 0,
                "outputs": [],
                "status": "idle",
            }
            for sd in slots_raw
        ]
        self.logs.clear()

        if getattr(sys, 'frozen', False):
            cmd = [sys.executable, "--worker"]
        else:
            worker_path = os.path.join(os.path.dirname(__file__), "hmi_worker.py")
            cmd = [sys.executable, "-u", worker_path]
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )

        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()

        self._stderr_reader = threading.Thread(target=self._stderr_loop, daemon=True)
        self._stderr_reader.start()

        cmd: dict = {
            "cmd": "start",
            "project": {"script": script, "layout": layout, "slots": slots_raw},
            "bps": bps,
        }
        if files:
            cmd["files"] = files
            cmd["project"]["entry_point"] = entry_point
        self.send(cmd)

        self._last_payload = {
            "script": script,
            "bps": bps,
            "layout": layout,
            "slots": slots_raw,
            "files": files,
            "entry_point": entry_point,
        }

    def reload(self) -> bool:
        if not self._last_payload:
            return False
        p = self._last_payload
        self.start(p["script"], p["bps"], p["layout"], p["slots"],
                   files=p["files"], entry_point=p["entry_point"])
        return True

    def stop(self):
        if self.process:
            self.send({"cmd": "exit"})
            try:
                self.process.wait(timeout=3)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=3)
            self.process = None
            self._reader = None

    def send(self, cmd: dict):
        if self.process and self.process.stdin and not self.process.stdin.closed:
            self.process.stdin.write(json.dumps(cmd, default=str) + "\n")
            self.process.stdin.flush()

    def _reader_loop(self):
        try:
            for line in self.process.stdout:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    self._dispatch(event)
                except json.JSONDecodeError:
                    continue
        except (OSError, ValueError, AttributeError):
            pass
        finally:
            self.process = None

    def _stderr_loop(self):
        try:
            for line in self.process.stderr:
                line = line.strip()
                if line:
                    print(f"[worker] {line}", file=sys.stderr, flush=True)
        except (OSError, ValueError, AttributeError):
            pass

    def _dispatch(self, event: dict):
        t = event.get("type")
        req_id = event.get("_req_id")

        if req_id and req_id in self._pending:
            self._pending_results[req_id] = event.get("data")
            loop = self._loop
            if loop:
                loop.call_soon_threadsafe(self._pending[req_id].set)
            return

        if t == "log":
            entry = event.get("entry", {})
            self.logs.append(entry)
            if len(self.logs) > 500:
                self.logs = self.logs[-500:]

        elif t == "status":
            for s in self.slots:
                if s["name"] == event.get("slot"):
                    s["status"] = event.get("status", "idle")
                    break

        elif t == "state_changed":
            for s in self.slots:
                if s["name"] == event.get("slot"):
                    s["state"] = event.get("state", "unknown")
                    break

        elif t == "slot_finished":
            for s in self.slots:
                if s["name"] == event.get("slot"):
                    s["last_error"] = event.get("error")
                    s["last_time_ms"] = event.get("total_time_ms", 0)
                    s["outputs"] = event.get("outputs", [])
                    break

        # Map worker event types → WS broadcast types
        _MAP = {
            "paused": "hmi_debug_paused",
            "resumed": "hmi_debug_resumed",
            "finished": "hmi_stopped",
            "ready": "hmi_loaded",
            "log": "hmi_log",
            "status": "hmi_status",
            "state_changed": "hmi_state_changed",
            "slot_finished": "hmi_slot_finished",
            "error": "hmi_error",
            "prompt": "hmi_prompt",
            "prompt_closed": "hmi_prompt_closed",
        }
        mapped = _MAP.get(t, t)
        data = {"type": mapped, **{k: v for k, v in event.items() if k != "type"}}
        broadcast(self.project_id, data)

    async def request_output(self, slot_name: str, output_name: str, quality: int = 85, field: str = "", widget_id: str = "", timeout=15):
        req_id = str(uuid.uuid4())
        ev = asyncio.Event()
        self._pending[req_id] = ev
        cmd = {"cmd": "get_output", "slot": slot_name, "output": output_name, "quality": quality, "_req_id": req_id}
        if field:
            cmd["field"] = field
        if widget_id:
            cmd["widget_id"] = widget_id
        self.send(cmd)
        try:
            await asyncio.wait_for(ev.wait(), timeout)
        except asyncio.TimeoutError:
            self._pending.pop(req_id, None)
            return None
        result = self._pending_results.pop(req_id, None)
        self._pending.pop(req_id, None)
        return result


# ── Helpers ──────────────────────────────────────────────────

def _stop_worker(pid: str, broadcast_stopped: bool = True):
    worker = _workers.get(pid)
    if worker:
        worker.stop()
        if broadcast_stopped:
            broadcast(pid, {"type": "hmi_stopped"})


class HmiLoadRequest(BaseModel):
    script: str = ""
    layout: dict = {}
    slots: list[dict] = []
    files: dict[str, str] | None = None
    entry_point: str = "main.py"
    bp_by_file: dict[str, list[int]] = {}
    project_id: str = ""


@app.post("/api/hmi/load")
def hmi_load(req: HmiLoadRequest):
    from falcocv.core.script_debugger import TraceDebugger
    bps: dict[str, list[int]] = {}
    pid = req.project_id or str(uuid.uuid4())[:8]

    if req.files:
        for fn, lines in req.bp_by_file.items():
            src = req.files.get(fn, "")
            if src:
                snapped = TraceDebugger._snap_breakpoints(set(lines), src)
                bps[fn] = list(snapped)
            else:
                bps[fn] = lines
    else:
        if req.bp_by_file:
            lines = req.bp_by_file.get("<hmi_script>", req.bp_by_file.get(list(req.bp_by_file.keys())[0] if req.bp_by_file else "", []))
            snapped = TraceDebugger._snap_breakpoints(set(lines), req.script)
            bps = {"<hmi_script>": list(snapped)}
        elif req.script:
            bps = _pending_bps.pop(pid, {})

    _stop_worker(pid, broadcast_stopped=False)
    worker = HMIWorkerManager(project_id=pid)
    worker.set_loop(_main_loop)
    worker.start(
        req.script, bps, req.layout, req.slots,
        files=req.files, entry_point=req.entry_point,
    )
    _workers[pid] = worker
    return {"ok": True, "slots": len(req.slots), "project_id": pid}


class SetBpsRequest(BaseModel):
    bps: list[int] = []
    bps_by_file: dict[str, list[int]] | None = None
    project_id: str = ""


@app.post("/api/hmi/set-bps")
def hmi_set_bps(req: SetBpsRequest):
    pid = req.project_id or "default"
    if req.bps_by_file:
        _pending_bps[pid] = dict(req.bps_by_file)
    else:
        _pending_bps[pid] = {"<hmi_script>": list(req.bps)}
    return {"ok": True}


class StopRequest(BaseModel):
    project_id: str = ""


@app.post("/api/hmi/stop")
def hmi_stop(req: StopRequest = StopRequest()):
    pid = req.project_id or "default"
    worker = _workers.get(pid)
    if not worker or not worker.alive:
        return {"error": "No HMI project loaded"}, 400
    _stop_worker(pid)
    return {"ok": True}


@app.post("/api/hmi/reload")
def hmi_reload(req: StopRequest = StopRequest()):
    pid = req.project_id or "default"
    worker = _workers.get(pid)
    if not worker:
        return {"error": "Project not found"}, 404
    if not worker.reload():
        return {"error": "No previous deploy payload"}, 400
    return {"ok": True, "project_id": pid}


@app.get("/api/hmi")
def hmi_status(project_id: str = "default"):
    worker = _workers.get(project_id)
    if not worker or not worker.alive:
        return {"layout": {"rows": 0, "cols": 0, "widgets": []}, "slots": []}
    return {"layout": worker.layout, "slots": worker.slots}


@app.get("/api/hmi/logs")
def hmi_logs(since: int = 0, project_id: str = "default"):
    worker = _workers.get(project_id)
    if not worker or not worker.alive:
        return {"logs": [], "count": 0}
    logs = worker.logs[since:]
    return {"logs": logs, "count": len(worker.logs)}


@app.get("/api/hmi/slot/{slot_name}/output/{output_name}")
async def hmi_slot_output(slot_name: str, output_name: str, quality: int = 85, field: str = "", widget_id: str = "", project_id: str = "default"):
    worker = _workers.get(project_id)
    if not worker or not worker.alive:
        return {"error": "No HMI project"}, 404
    data = await worker.request_output(slot_name, output_name, quality, field=field, widget_id=widget_id)
    if data is None:
        return {"data_b64": None, "width": 0, "height": 0, "overlays": []}
    return data


# ── Package management (venv) ────────────────────────────────


class PackageInstallRequest(BaseModel):
    packages: list[str]


@app.get("/api/packages")
def package_list():
    from falcocv.core.environment import pip_list
    return {"packages": pip_list()}


@app.post("/api/packages/install")
def package_install(req: PackageInstallRequest):
    from falcocv.core.environment import pip_install
    ok, msg = pip_install(req.packages)
    return {"ok": ok, "message": msg}


@app.post("/api/packages/uninstall")
def package_uninstall(req: PackageInstallRequest):
    from falcocv.core.environment import pip_uninstall
    ok, msg = pip_uninstall(req.packages)
    return {"ok": ok, "message": msg}


# ── Server dashboard ──────────────────────────────────────────


@app.get("/api/server/projects")
def server_projects():
    result = []
    now = time.time()
    for pid, worker in _workers.items():
        info = {
            "project_id": pid,
            "alive": worker.alive,
            "pid": worker.process.pid if worker.alive and worker.process else 0,
            "layout_rows": worker.layout.get("rows", 0),
            "layout_cols": worker.layout.get("cols", 0),
            "slot_count": len(worker.slots),
            "memory_mb": 0.0,
            "uptime_seconds": 0.0,
        }
        if worker.alive and worker.process:
            try:
                p = psutil.Process(worker.process.pid)
                info["memory_mb"] = round(p.memory_info().rss / (1024 * 1024), 1)
                info["uptime_seconds"] = round(now - p.create_time(), 0)
            except Exception:
                pass
        result.append(info)
    return result


# ── Static files ──────────────────────────────────────────────

hmi_dir = os.path.join(os.path.dirname(__file__), "hmi")

@app.get("/{full_path:path}")
async def serve_frontend(full_path: str, request: Request = None):
    from fastapi.responses import JSONResponse
    if full_path.startswith("api/") or full_path.startswith("ws"):
        return JSONResponse({"error": "Not found"}, status_code=404)
    if not full_path:
        if request is not None and request.query_params.get("id"):
            hmi_index = os.path.join(hmi_dir, "index.html")
            if os.path.isfile(hmi_index):
                return FileResponse(hmi_index)
        dash = os.path.join(hmi_dir, "dashboard.html")
        if os.path.isfile(dash):
            return FileResponse(dash)
        return JSONResponse({"error": "Dashboard not found"}, status_code=404)
    if full_path == "hmi":
        hmi_index = os.path.join(hmi_dir, "index.html")
        if os.path.isfile(hmi_index):
            return FileResponse(hmi_index)
    if full_path.startswith("hmi/"):
        hmi_file = os.path.join(hmi_dir, full_path[4:])
        if os.path.isfile(hmi_file):
            return FileResponse(hmi_file)
    return JSONResponse({"error": "Not found"}, status_code=404)
