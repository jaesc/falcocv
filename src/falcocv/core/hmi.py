import datetime
import json
import os
import threading
import time as _time
import uuid
from dataclasses import dataclass, field
from typing import Any

from falcocv.core.workflow import Workflow
from falcocv.core.workflow_controller import WorkflowController
from falcocv.core.job_manager import JobManager
from falcocv.core.tool_base import CallbackList


@dataclass
class HMISlot:
    name: str = ""
    job_path: str = ""
    _controller: Any = None

    @property
    def controller(self) -> WorkflowController | None:
        return self._controller

    def load_job(self):
        if not self.job_path or not os.path.isfile(self.job_path):
            return
        wf = Workflow()
        JobManager().load_job(wf, self.job_path)
        self._controller = WorkflowController(wf)

    def run(self, advance=True):
        if self._controller:
            self._controller.run(advance=advance)


WIDGET_TYPES = [
    "ImageViewer",
    "MeasureLabel",
    "StatusIndicator",
    "LedIndicator",
    "TextLabel",
    "LogPanel",
]

WIDGET_DEFAULT_SIZES = {
    "ImageViewer": (8, 6),
    "MeasureLabel": (6, 3),
    "StatusIndicator": (4, 3),
    "LedIndicator": (3, 3),
    "TextLabel": (4, 2),
    "LogPanel": (8, 6),
}

DEFAULT_LAYOUT: dict[str, Any] = {
    "rows": 56,
    "cols": 100,
    "widgets": [],
}


class HMIProject:
    def __init__(self):
        self.slots: list[HMISlot] = []
        self.layout: dict = dict(DEFAULT_LAYOUT)
        self.script: str = (
            "# HMI Script — runs in a background thread\n"
            "# Available functions:\n"
            "#   run_slot(name: str)            — execute one job slot\n"
            "#   log(msg: str, level='info')    — add message to log (levels: info, error, warning, debug)\n"
            "#   set_slot_status(name, status)  — 'pass', 'fail', 'pending' or 'idle'\n"
            "#   sleep(s: float)                — pause execution\n"
            "#   help()                         — print this API reference to log\n"
            "#   proj.widgets[name]             — access/edit any widget by name\n"
            "#   proj.get_widget_value(name)    — evaluate field+condition, returns {'value':..., 'status':'pass/fail'}\n"
            "#   val = proj.input('msg')        — prompt operator for text (blocking)\n"
            "#   ok = proj.confirm('msg')       — Yes/No/Cancel (blocking)\n"
            "#   c = proj.confirm('msg',['A','B']) — custom options (blocking)\n"
            "#   proj.input('msg', width='400px', height='200px', font_size='18px') — styled\n"
            "#\n"
            "# Widget config examples:\n"
            "#   w = proj.widgets['My counter']\n"
            "#   w['min'] = 0; w['max'] = 50\n"
            "#\n"
            "#   proj.widgets['Title']['label'] = 'Hello'\n"
            "#   proj.widgets['Title']['bg_color'] = [0, 100, 200]\n"
            "#   proj.widgets['Title']['font_size'] = 24\n"
            "#\n"
            "# MeasureLabel conditions:\n"
            "#   w['value_type'] = 'numeric'\n"
            "#   w['condition'] = 'between';  w['min']=0; w['max']=50\n"
            "#   w['condition'] = '>';   w['compare_value'] = '10'\n"
            "#\n"
            "#   w['value_type'] = 'text'\n"
            "#   w['condition'] = 'contains';  w['compare_value'] = 'OK'\n"
            "#   w['condition'] = 'regex';    w['regex'] = '^OK.*'\n"
            "#\n"
            "while True:\n"
            "    if _stop.is_set():\n"
            "        break\n"
            "    run_slot('Slot1')\n"
            "    sleep(1)\n"
        )
        self.source_dir: str | None = None
        self.entry_point: str = "main.py"
        self.project_id: str = str(uuid.uuid4())[:8]
        self._script_thread: threading.Thread | None = None
        self._script_stop = threading.Event()
        self._logs: list[dict] = []
        self._log_lock = threading.Lock()
        self._log_listeners: list = []
        self._slot_statuses: dict[str, str] = {}
        self._status_listeners: list = []
        self._pending_prompts: dict[str, dict] = {}
        self._prompt_listeners: list = []
        self._hmi_breakpoints: dict[str, set[int]] = {}
        self._debugger: Any = None
        self.debugger_paused = CallbackList()
        self.debugger_resumed = CallbackList()

    def set_project_id(self, value: str):
        clean = value.strip().replace(" ", "_")
        if not clean or not all(c.isalnum() or c in "_-" for c in clean):
            raise ValueError("Project ID must be alphanumeric (a-z, 0-9, _, -)")
        self.project_id = clean

    def add_slot(self, name: str = "", job_path: str = "") -> HMISlot:
        slot = HMISlot(name=name, job_path=job_path)
        self.slots.append(slot)
        return slot

    def remove_slot(self, index: int):
        if 0 <= index < len(self.slots):
            del self.slots[index]

    def _widget_name_exists(self, name: str) -> bool:
        return any(w.get("name") == name for w in self.layout.get("widgets", []))

    def _default_widget_name(self, widget_type: str) -> str:
        existing = [w.get("name", "") for w in self.layout.get("widgets", [])]
        count = sum(1 for n in existing if n.startswith(f"{widget_type}_"))
        return f"{widget_type}_{count + 1}"

    @property
    def widgets(self) -> dict[str, dict]:
        return {w["name"]: w for w in self.layout.get("widgets", []) if w.get("name")}

    def add_widget(self, widget_type: str, x: int = 0, y: int = 0, name: str | None = None) -> dict:
        import uuid
        dw, dh = WIDGET_DEFAULT_SIZES.get(widget_type, (4, 3))
        existing_z = [w.get("z", 0) for w in self.layout.get("widgets", [])]
        next_z = max(existing_z, default=-1) + 1
        if not name:
            name = self._default_widget_name(widget_type)
        elif self._widget_name_exists(name):
            raise ValueError(f"Widget name '{name}' already exists")
        wid = {
            "id": f"w_{uuid.uuid4().hex[:8]}",
            "name": name,
            "type": widget_type,
            "x": max(0, min(x, self.layout["cols"] - dw)),
            "y": max(0, min(y, self.layout["rows"] - dh)),
            "w": dw,
            "h": dh,
            "z": next_z,
        }
        self.layout.setdefault("widgets", []).append(wid)
        return wid

    def remove_widget(self, widget_id: str):
        self.layout["widgets"] = [w for w in self.layout.get("widgets", []) if w["id"] != widget_id]

    def rename_widget(self, widget_id: str, new_name: str):
        if self._widget_name_exists(new_name):
            raise ValueError(f"Widget name '{new_name}' already exists")
        for w in self.layout.get("widgets", []):
            if w["id"] == widget_id:
                w["name"] = new_name
                return
        raise KeyError(f"Widget id '{widget_id}' not found")

    def get_source_files(self) -> dict[str, str]:
        """Read all .py files from source_dir. Returns {relpath: content}."""
        if not self.source_dir or not os.path.isdir(self.source_dir):
            return {}
        files = {}
        for root, _, filenames in os.walk(self.source_dir):
            for fn in filenames:
                if fn.endswith(".py"):
                    full = os.path.join(root, fn)
                    try:
                        with open(full, "r", encoding="utf-8") as f:
                            content = f.read()
                    except Exception:
                        continue
                    rel = os.path.relpath(full, self.source_dir).replace("\\", "/")
                    files[rel] = content
        return files

    def save(self, path: str):
        version = "2.0" if self.source_dir else "1.0"
        data: dict = {
            "version": version,
            "project_id": self.project_id,
            "slots": [
                {"name": s.name, "job_path": s.job_path}
                for s in self.slots
            ],
            "layout": self.layout,
            "script": self.script,
        }
        if self.source_dir:
            data["source_dir"] = self.source_dir
            data["entry_point"] = self.entry_point
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "HMIProject":
        with open(path, "r") as f:
            data = json.load(f)
        proj = cls()
        proj.project_id = data.get("project_id") or str(uuid.uuid4())[:8]
        if data.get("source_dir"):
            proj.source_dir = data["source_dir"]
            proj.entry_point = data.get("entry_point", "main.py")
        for sd in data.get("slots", []):
            slot = proj.add_slot(name=sd.get("name", ""), job_path=sd.get("job_path", ""))
        loaded_layout = data.get("layout", {})
        layout = dict(DEFAULT_LAYOUT)
        layout.update(loaded_layout)
        for key in ("rows", "cols", "widgets"):
            layout[key] = loaded_layout.get(key, DEFAULT_LAYOUT[key])
        proj.layout = layout
        # Migrate old widgets: assign auto-name if missing
        seen = set()
        for w in proj.layout.get("widgets", []):
            if not w.get("name"):
                base = w.get("type", "Widget")
                name = f"{base}_1"
                i = 1
                while name in seen:
                    i += 1
                    name = f"{base}_{i}"
                w["name"] = name
            seen.add(w["name"])
        proj.script = data.get("script", proj.script)
        return proj

    def run_slot(self, slot_name: str, advance: bool = True):
        for slot in self.slots:
            if slot.name == slot_name:
                if self._script_stop.is_set():
                    return
                slot.run(advance=advance)
                return

    def load_all_jobs(self):
        for slot in self.slots:
            slot.load_job()

    def start_script(self):
        if self._script_thread and self._script_thread.is_alive():
            return
        self._script_stop.clear()
        self._script_thread = threading.Thread(
            target=self._run_script, daemon=True
        )
        self._script_thread.start()

    def stop_script(self):
        self._script_stop.set()
        self._log_listeners.clear()
        self._status_listeners.clear()
        if self._debugger:
            self._debugger.paused.clear()
            self._debugger.resumed.clear()
            self._debugger.stop()
            self._debugger = None
        if self._script_thread and self._script_thread.is_alive():
            self._script_thread.join(timeout=2)
        self._script_thread = None
        self._log_listeners = []
        self._status_listeners = []
        with self._log_lock:
            self._logs.clear()
        for slot in self.slots:
            ctrl = slot._controller
            if ctrl:
                ctrl.state_changed.clear()
                ctrl.execution_started.clear()
                ctrl.execution_finished.clear()
                for tool in ctrl.workflow.tools:
                    for k in tool.outputs:
                        tool.outputs[k] = None
            slot._controller = None

    def log(self, msg: str, level: str = "info"):
        entry = {
            "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
            "message": str(msg),
            "level": level,
        }
        with self._log_lock:
            self._logs.append(entry)
            if len(self._logs) > 500:
                self._logs = self._logs[-500:]
        for listener in self._log_listeners:
            try:
                listener(entry)
            except Exception:
                pass

    def get_logs(self, since_index: int = 0) -> list[dict]:
        with self._log_lock:
            if since_index >= len(self._logs):
                return []
            return self._logs[since_index:]

    def set_slot_status(self, slot_name: str, status: str):
        self._slot_statuses[slot_name] = status
        for listener in self._status_listeners:
            try:
                listener({"slot": slot_name, "status": status})
            except Exception:
                pass

    def _prompt(self, msg: str, input_type: str = "text", options: list[str] | None = None, timeout: float = 300, width: str = "", height: str = "", font_size: str = "") -> str | None:
        """Send a prompt to all frontends and wait for a response. Blocking."""
        import uuid as _uuid
        req_id = _uuid.uuid4().hex[:12]
        ev = threading.Event()
        self._pending_prompts[req_id] = {"event": ev, "result": None}
        payload = {"req_id": req_id, "msg": msg, "input_type": input_type}
        if options:
            payload["options"] = options
        if width:
            payload["width"] = width
        if height:
            payload["height"] = height
        if font_size:
            payload["font_size"] = font_size
        for cb in self._prompt_listeners:
            try:
                cb(payload)
            except Exception:
                pass
        ev.wait(timeout=timeout)
        entry = self._pending_prompts.pop(req_id, {})
        return entry.get("result")

    def input(self, msg: str, width: str = "", height: str = "", font_size: str = "") -> str | None:
        """Prompt operator for text input. Returns the entered string or None if cancelled/timed out."""
        return self._prompt(msg, "text", width=width, height=height, font_size=font_size)

    def confirm(self, msg: str, options: list[str] | None = None, width: str = "", height: str = "", font_size: str = "") -> bool | str | None:
        """Prompt operator for confirmation. Returns True/False for default options, or the selected string for custom options."""
        opts = options or ["Yes", "No", "Cancel"]
        result = self._prompt(msg, "confirm", options=opts, width=width, height=height, font_size=font_size)
        if options:
            return result
        if result == "Yes": return True
        if result == "No": return False
        return None

    def _resolve_prompt(self, req_id: str, value):
        entry = self._pending_prompts.get(req_id)
        if entry:
            entry["result"] = value
            entry["event"].set()
            payload = {"req_id": req_id}
            for cb in self._prompt_listeners:
                try:
                    cb({**payload, "type": "prompt_closed"})
                except Exception:
                    pass

    def get_widget_value(self, name: str):
        """Evaluate a MeasureLabel widget's field and condition, return value and status."""
        import re as _re
        w = self.widgets.get(name)
        if not w:
            raise KeyError(f"Widget '{name}' not found")
        slot_name = w.get("slot", "")
        output_name = w.get("output", "")
        field = w.get("field", "")
        ctrl = None
        for slot in self.slots:
            if slot.name == slot_name:
                ctrl = slot.controller
                break
        if ctrl is None or not ctrl.workflow or not ctrl.workflow.tools:
            return {"value": None, "status": "error"}
        if " > " in output_name:
            tool_name, _, out_name = output_name.partition(" > ")
            tool = ctrl.workflow.get_tool(tool_name)
            if tool is None:
                return {"value": None, "status": "error"}
            val = tool.get_output(out_name)
        else:
            last_tool = ctrl.workflow.tools[-1]
            val = last_tool.get_output(output_name)
        if val is None:
            return {"value": None, "status": "error"}
        if hasattr(val, 'data'):
            return {"value": None, "status": "error"}
        if field:
            val = self._evaluate_field(val, field)
        value = val
        status = self._evaluate_condition(value, w)
        return {"value": value, "status": status}

    def _evaluate_field(self, val, field_expr):
        """Evaluate a simple field expression like [0].area or Count."""
        import re as _re
        if not field_expr:
            return str(val)
        parts = _re.findall(r'\[(\d+)\]|\["([^"]+)"\]|\[(\w+)\]|\.(\w+)', field_expr)
        for idx, str_key, raw_key, attr in parts:
            key = int(idx) if idx else (str_key or raw_key or attr)
            if isinstance(key, int):
                val = val[key]
            elif hasattr(val, '__getitem__'):
                try:
                    val = val[key]
                except (KeyError, TypeError, IndexError):
                    val = getattr(val, key, val)
            else:
                val = getattr(val, key, val)
        return val

    @staticmethod
    def _evaluate_condition(value, w: dict) -> str:
        """Evaluate a widget's condition against a value. Returns 'pass' or 'fail'."""
        import re as _re
        vt = w.get("value_type", "numeric")
        cond = w.get("condition", "between")
        if vt == "numeric":
            try:
                num_value = float(value) if value is not None else None
            except (TypeError, ValueError):
                num_value = None
            if cond == "between":
                lo = w.get("min")
                hi = w.get("max")
                if lo is not None and hi is not None and num_value is not None:
                    try:
                        lo, hi = float(lo), float(hi)
                    except (TypeError, ValueError):
                        return "pass"
                    return "pass" if lo <= num_value <= hi else "fail"
                return "pass"
            cv = w.get("compare_value")
            if cv is None or cv == "" or num_value is None:
                return "pass"
            try:
                cv = float(cv)
            except (TypeError, ValueError):
                return "pass"
            if cond == ">":   return "pass" if num_value > cv else "fail"
            if cond == ">=":  return "pass" if num_value >= cv else "fail"
            if cond == "<":   return "pass" if num_value < cv else "fail"
            if cond == "<=":  return "pass" if num_value <= cv else "fail"
            if cond == "=":   return "pass" if num_value == cv else "fail"
            if cond == "!=":  return "pass" if num_value != cv else "fail"
        else:
            sval = str(value) if value is not None else ""
            if cond == "equals":
                return "pass" if sval == w.get("compare_value", "") else "fail"
            if cond == "!equals":
                return "pass" if sval != w.get("compare_value", "") else "fail"
            if cond == "contains":
                return "pass" if w.get("compare_value", "") in sval else "fail"
            if cond == "!contains":
                return "pass" if w.get("compare_value", "") not in sval else "fail"
            if cond == "regex":
                try:
                    return "pass" if _re.match(w.get("regex", ""), sval) else "fail"
                except _re.error:
                    return "error"
        return "pass"

    def _script_help(self):
        self.log("=== HMI Script API ===")
        self.log("run_slot(name)       — execute a job slot")
        self.log("log(msg, level)      — write to log panel")
        self.log("set_slot_status(n,s) — 'pass'/'fail'/'pending'/'idle'")
        self.log("sleep(seconds)       — interruptible sleep")
        self.log("help()               — this help")
        self.log("proj.input(msg)       — prompt operator for text (blocking)")
        self.log("proj.confirm(msg)     — prompt operator Yes/No/Cancel (blocking)")
        self.log("proj.confirm(msg,opt) — custom options e.g. ['Auto','Manual']")
        self.log("proj.input(msg,width,font_size) — styled prompt")
        self.log("")
        self.log("Objects:")
        self.log("  proj                  — the HMIProject")
        self.log("  proj.widgets[name]    — access any widget")
        self.log("  proj.get_widget_value(name) — evaluate field+condition")
        self.log("  proj.slots            — job slots")
        self.log("  proj.layout           — raw layout dict")
        self.log("")
        self.log("Widget properties (writable):")
        self.log("  w['label']         — display text")
        self.log("  w['bg_color']      — [R,G,B] background")
        self.log("  w['text_color']    — [R,G,B] text color")
        self.log("  w['font_size']     — px")
        self.log("  w['halign']/valign — left/center/right/top/bottom")
        self.log("  w['label_layout']  — 'stacked' (above) or 'inline' (same line)")
        self.log("  w['slot']/output   — data binding")
        self.log("  w['field']         — field expression")
        self.log("  w['x']/y/w/h/z     — layout geometry")
        self.log("")
        self.log("MeasureLabel conditions:")
        self.log("  w['value_type'] = 'numeric'   (default)")
        self.log("  w['condition'] = 'between' | '>' | '>=' | '<' | '<=' | '=' | '!='")
        self.log("  w['min']/w['max'] — for 'between'")
        self.log("  w['compare_value'] — for other numeric conditions")
        self.log("  w['value_type'] = 'text'")
        self.log("  w['condition'] = 'equals' | '!equals' | 'contains' | '!contains' | 'regex'")
        self.log("  w['compare_value'] — for text conditions (except regex)")
        self.log("  w['regex'] — regex pattern for 'regex' condition")
        self.log("")
        self.log("Example:")
        self.log("  w = proj.widgets['My counter']")
        self.log("  w['min'] = 0; w['max'] = 50")
        self.log("  result = proj.get_widget_value('My counter')")
        self.log("  log(f\"Value: {result['value']} Status: {result['status']}\")")

    def _interruptible_sleep(self, seconds: float):
        step = 0.1
        for _ in range(int(seconds / step)):
            if self._script_stop.is_set():
                return
            _time.sleep(step)

    def _run_script(self):
        if self.source_dir and os.path.isdir(self.source_dir):
            self._run_script_multi()
        else:
            self._run_script_single()

    def _run_script_single(self):
        from falcocv.core.script_debugger import TraceDebugger
        import types

        ns = {
            "run_slot": self.run_slot,
            "log": self.log,
            "set_slot_status": self.set_slot_status,
            "sleep": self._interruptible_sleep,
            "help": self._script_help,
            "_stop": self._script_stop,
            "proj": self,
        }

        code = compile(self.script, "<hmi_script>", "exec")
        ns["__script_code"] = code

        wrapper_code = compile("exec(__script_code)", "<hmi_script>", "exec")
        run_fn = types.FunctionType(wrapper_code, ns)

        self._debugger = TraceDebugger()
        self._debugger.get_breakpoints = lambda: self._hmi_breakpoints.get("<hmi_script>", set())
        self._debugger.paused.connect(self._emit_debug_paused)
        self._debugger.resumed.connect(self._emit_debug_resumed)

        try:
            self._debugger.run_with_trace(run_fn, script_filename="<hmi_script>")
        except Exception as e:
            self.log(f"HMI Script error: {e}", level="error")
        finally:
            if self._debugger:
                self._debugger.paused.clear()
                self._debugger.resumed.clear()
                self._debugger.get_breakpoints = lambda: set()
                self._debugger.get_breakpoints_for_file = lambda fn: set()
                self._debugger = None

    def _run_script_multi(self):
        import sys as _sys
        from falcocv.core.script_debugger import TraceDebugger

        ns = {
            "run_slot": self.run_slot,
            "log": self.log,
            "set_slot_status": self.set_slot_status,
            "sleep": self._interruptible_sleep,
            "help": self._script_help,
            "_stop": self._script_stop,
            "proj": self,
        }

        source_dir = os.path.abspath(self.source_dir)
        _sys.path.insert(0, source_dir)

        entry_path = os.path.join(source_dir, self.entry_point)
        if not os.path.isfile(entry_path):
            self.log(f"Entry point not found: {entry_path}", level="error")
            return

        source_files = self.get_source_files()
        filenames = frozenset(
            os.path.abspath(os.path.join(source_dir, f)) for f in source_files
        )

        with open(entry_path, "r", encoding="utf-8") as f:
            entry_source = f.read()

        code = compile(entry_source, os.path.abspath(entry_path), "exec")
        ns["__script_code"] = code
        wrapper_code = compile("exec(__script_code)", os.path.abspath(entry_path), "exec")

        import types
        run_fn = types.FunctionType(wrapper_code, ns)

        self._debugger = TraceDebugger()
        sd = os.path.abspath(source_dir)
        self._debugger.get_breakpoints_for_file = (
            lambda fn: self._hmi_breakpoints.get(
                os.path.relpath(fn, sd).replace("\\", "/"), set()
            )
        )
        self._debugger.paused.connect(self._emit_debug_paused)
        self._debugger.resumed.connect(self._emit_debug_resumed)

        try:
            self._debugger.run_with_trace(
                run_fn,
                script_filename=os.path.abspath(entry_path),
                script_filenames=filenames,
            )
        except Exception as e:
            self.log(f"HMI Script error: {e}", level="error")
        finally:
            if self._debugger:
                self._debugger.paused.clear()
                self._debugger.resumed.clear()
                self._debugger.get_breakpoints = lambda: set()
                self._debugger.get_breakpoints_for_file = lambda fn: set()
                self._debugger = None

    def _emit_debug_paused(self, line, locals_dict):
        extra = dict(locals_dict) if locals_dict else {}
        debug_file = extra.pop("_debug_file", "<hmi_script>")
        extra["hmi_debug_file"] = debug_file
        slot_info = {}
        for slot in self.slots:
            info = {
                "name": slot.name,
                "job_path": slot.job_path,
                "status": self._slot_statuses.get(slot.name, "idle"),
            }
            ctrl = slot.controller
            if ctrl and ctrl.workflow and ctrl.workflow.tools:
                last = ctrl.workflow.tools[-1]
                info["last_error"] = ctrl.last_error
                info["last_time_ms"] = ctrl.last_execution_time_ms
                outputs = {}
                for k, v in last.outputs.items():
                    if v is not None:
                        try:
                            outputs[k] = repr(v)[:300]
                        except Exception:
                            outputs[k] = "<error>"
                    else:
                        outputs[k] = None
                info["last_outputs"] = outputs
            slot_info[slot.name] = info
        extra["hmi_slots"] = slot_info
        extra["log_count"] = len(self._logs)
        self.debugger_paused.emit(line, extra)

    def _emit_debug_resumed(self):
        self.debugger_resumed.emit()
