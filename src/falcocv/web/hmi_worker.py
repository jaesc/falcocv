import json
import sys
import os
import re
import threading
import tempfile
from dataclasses import asdict

if not getattr(sys, 'frozen', False):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from falcocv.core.hmi import HMIProject
from falcocv.core.script_debugger import TraceDebugger
from falcocv.core.environment import get_venv_site_packages as _get_venv_sp


_proj: HMIProject | None = None


def _evaluate_field(val, field_expr):
    """Evaluate a simple field expression like [0].area or Count or [0]["value"].
    
    Supports:
        [N]     → val[N] (index access)
        .name   → getattr(val, "name") or val["name"]
        ["key"] → val["key"]
    """
    if not field_expr:
        return str(val)
    parts = re.findall(r'\[(\d+)\]|\["([^"]+)"\]|\[(\w+)\]|\.(\w+)', field_expr)
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


_finished = threading.Event()
_site_packages: str | None = None


def _tag_dict(tag) -> dict:
    return {
        "id": tag.id,
        "device": tag.device,
        "name": tag.name,
        "address": tag.address,
        "type": tag.type,
        "access": tag.access,
        "poll_ms": tag.poll_ms,
        "default": tag.default,
        "description": tag.description,
    }


def _serialise_tag_values(manager) -> dict:
    return {
        tag_id: {
            "value": value.value,
            "quality": value.quality,
            "timestamp": value.timestamp,
            "error": value.error,
        }
        for tag_id, value in manager.snapshot().items()
    }


def _comms_snapshot() -> dict:
    """Read the deployed project's communications state for the Live monitor."""

    from falcocv.comms import CommsError, parse_comms_config

    if _proj is None:
        return {"error": "No HMI project loaded", "tags": [], "values": {}}

    manager = _proj.comms
    statuses = []
    values = {}
    tags = []
    warning = ""
    if manager is not None:
        tags = [_tag_dict(tag) for tag in manager.tags]
        readable = [
            tag.id for tag in manager.tags if tag.access in ("r", "rw")
        ]
        if readable:
            try:
                manager.read_many(readable)
            except CommsError as exc:
                warning = str(exc)
        values = _serialise_tag_values(manager)
        statuses = [asdict(status) for status in manager.statuses()]
    else:
        try:
            config = parse_comms_config(_proj.comms_config)
            tags = [_tag_dict(tag) for device in config.devices for tag in device.tags]
        except Exception as exc:
            return {
                "error": str(exc),
                "tags": [],
                "values": {},
                "statuses": [],
            }
    return {
        "tags": tags,
        "values": values,
        "statuses": statuses,
        "warning": warning,
    }


def _comms_values() -> dict:
    """Lightweight values-only refresh used after the first Live snapshot."""

    from falcocv.comms import CommsError

    if _proj is None or _proj.comms is None:
        return {
            "error": "Communications runtime is not running",
            "values": {},
            "statuses": [],
        }
    manager = _proj.comms
    warning = ""
    readable = [
        tag.id for tag in manager.tags if tag.access in ("r", "rw")
    ]
    if readable:
        try:
            manager.read_many(readable)
        except CommsError as exc:
            warning = str(exc)
    return {
        "values": _serialise_tag_values(manager),
        "statuses": [asdict(status) for status in manager.statuses()],
        "warning": warning,
    }


def _comms_widget_values() -> dict:
    """Return evaluated values for layout widgets bound to tags."""

    from falcocv.comms import CommsError

    out = {}
    if _proj is None:
        return {"widgets": out}
    tag_widgets = [
        widget
        for widget in _proj.layout.get("widgets", [])
        if widget.get("source") == "tag" and widget.get("tag")
    ]
    if not tag_widgets:
        return {"widgets": out}

    manager = _proj.comms
    if manager is None:
        for widget in tag_widgets:
            out[widget["id"]] = {
                "value": None,
                "quality": "bad",
                "status": "idle",
            }
        return {"widgets": out}

    tag_ids = sorted({widget["tag"] for widget in tag_widgets})
    try:
        manager.read_many(tag_ids)
    except CommsError:
        pass
    for widget in tag_widgets:
        tag_value = manager.tag(widget["tag"])
        if tag_value is None:
            out[widget["id"]] = {
                "value": None,
                "quality": "bad",
                "status": "idle",
            }
            continue
        quality = tag_value.quality
        if quality != "good":
            status = "idle"
        elif widget.get("type") in ("StatusIndicator", "LedIndicator"):
            status = "pass" if bool(tag_value.value) else "fail"
        else:
            status = _proj._evaluate_condition(tag_value.value, widget)
        out[widget["id"]] = {
            "value": tag_value.value,
            "quality": quality,
            "status": status,
        }
    return {"widgets": out}


def _get_site_packages():
    global _site_packages
    if _site_packages is None:
        sp = _get_venv_sp()
        if not sp:
            approot = os.path.dirname(sys.executable) if getattr(sys, 'frozen', False) else os.path.dirname(__file__)
            sp = os.path.join(approot, "site-packages")
            if not os.path.isdir(sp):
                sp = ""
        _site_packages = sp
    return _site_packages or None


def write_event(data: dict):
    sys.stdout.write(json.dumps(data, default=str) + "\n")
    sys.stdout.flush()


def _log_stderr(msg):
    print(f"[worker] {msg}", file=sys.stderr, flush=True)


def handle_cmd(cmd: dict):
    global _proj
    c = cmd.get("cmd")

    if c == "start":
        data = cmd.get("project", {})
        bp_raw = cmd.get("bps", {})
        if isinstance(bp_raw, list):
            bps = {"<hmi_script>": set(bp_raw)}
        else:
            bps = {fn: set(lines) for fn, lines in bp_raw.items()}
        files = cmd.get("files", None)

        if _proj:
            _proj.stop_script()
            _proj.stop_runtime()

        proj = HMIProject()
        if data.get("comms"):
            proj.comms_config = data["comms"]

        if files:
            tmpdir = tempfile.mkdtemp(prefix="hmi_src_")
            for relpath, content in files.items():
                full = os.path.join(tmpdir, relpath)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                with open(full, "w", encoding="utf-8") as wf:
                    wf.write(content)
            sp = _get_site_packages()
            if sp:
                sys.path.insert(0, sp)
            sys.path.insert(0, tmpdir)
            proj.source_dir = tmpdir
            proj.entry_point = data.get("entry_point", "main.py")
        else:
            proj.script = data.get("script", "")
        proj.layout = data.get("layout", {})
        for sd in data.get("slots", []):
            slot = proj.add_slot(name=sd["name"], job_path=sd.get("job_path", ""))
            if sd.get("job_path"):
                slot.load_job()
                if slot.controller is None:
                    _log_stderr(f"load_job FAILED: slot='{sd['name']}' job_path='{sd.get('job_path')}'")

        proj._hmi_breakpoints = bps

        proj._log_listeners.append(_on_log)
        proj._status_listeners.append(_on_status)
        proj._prompt_listeners.append(_on_prompt)
        proj.debugger_paused.connect(_on_debug_paused)
        proj.debugger_resumed.connect(lambda: write_event({"type": "resumed"}))

        for slot in proj.slots:
            ctrl = slot.controller
            if ctrl is None:
                _log_stderr(f"connecting slot '{slot.name}': NO CONTROLLER (load_job failed)")
                continue
            n_tools = len(ctrl.workflow.tools) if ctrl.workflow and ctrl.workflow.tools else 0
            name_slot = slot.name
            ctrl.execution_finished.connect(
                lambda data, n=name_slot, c=ctrl: _on_slot_finished(n, data, c)
            )
            ctrl.state_changed.connect(
                lambda state, n=name_slot: write_event({
                    "type": "state_changed", "slot": n, "state": state.value
                })
            )

        _finished.clear()
        try:
            proj.start_runtime()
        except Exception as exc:
            proj.log(f"Communications start failed: {exc}", level="error")
        proj.start_script()
        this_proj = proj
        _proj = proj

        def _watch():
            if this_proj._script_thread:
                this_proj._script_thread.join()
            this_proj.stop_runtime()
            if _proj is this_proj and not _finished.is_set():
                _finished.set()
                write_event({"type": "finished"})

        threading.Thread(target=_watch, daemon=True).start()

        write_event({"type": "ready", "slots": len(data.get("slots", []))})

    elif c == "stop":
        if not _finished.is_set():
            _finished.set()
            if _proj:
                _proj.stop_script()
                _proj.stop_runtime()
                _proj = None
                write_event({"type": "finished"})

    elif c in ("continue", "step_over", "step_into", "step_out"):
        if _proj and _proj._debugger:
            {
                "continue": lambda: _proj._debugger.continue_exec(),
                "step_over": lambda: _proj._debugger.step_over(),
                "step_into": lambda: _proj._debugger.step_into(),
                "step_out": lambda: _proj._debugger.step_out(),
            }[c]()

    elif c == "debug_stop":
        if _proj and _proj._debugger:
            _proj._debugger.stop()

    elif c == "set_bps":
        if _proj:
            bp_raw = cmd.get("bps", {})
            if isinstance(bp_raw, list):
                bps = set(bp_raw)
                if _proj.script:
                    bps = TraceDebugger._snap_breakpoints(bps, _proj.script)
                _proj._hmi_breakpoints = {"<hmi_script>": bps}
            else:
                result = {}
                if _proj.source_dir:
                    files = _proj.get_source_files()
                    for fn, lines in bp_raw.items():
                        src = files.get(fn, "")
                        if src:
                            result[fn] = TraceDebugger._snap_breakpoints(set(lines), src)
                        else:
                            result[fn] = set(lines)
                else:
                    for fn, lines in bp_raw.items():
                        result[fn] = set(lines)
                _proj._hmi_breakpoints = result

    elif c == "run_slot":
        if _proj:
            _proj.run_slot(cmd.get("slot", ""), advance=True)

    elif c == "get_output":
        req_id = cmd["_req_id"]
        slot_name = cmd["slot"]
        output_name = cmd["output"]
        quality = cmd.get("quality", 85)
        field = cmd.get("field", "")
        result = {"data_b64": None, "width": 0, "height": 0, "overlays": []}
        try:
            if _proj:
                for slot in _proj.slots:
                    if slot.name == slot_name:
                        ctrl = slot.controller
                        if ctrl is None:
                            _log_stderr(f"get_output: slot '{slot_name}' has no controller")
                            break
                        if not ctrl.workflow or not ctrl.workflow.tools:
                            _log_stderr(f"get_output: slot '{slot_name}' no tools in workflow")
                            break
                        if " > " in output_name:
                            tool_name, _, out_name = output_name.partition(" > ")
                            tool = ctrl.workflow.get_tool(tool_name)
                            if tool is None:
                                _log_stderr(f"get_output: tool '{tool_name}' not found in workflow")
                                break
                            val = tool.get_output(out_name)
                        else:
                            last_tool = ctrl.workflow.tools[-1]
                            val = last_tool.get_output(output_name)
                        if val is None:
                            _log_stderr(f"get_output: output '{output_name}' is None")
                            break
                        if hasattr(val, 'data'):
                            from falcocv.web.image_utils import encode_vision_image as _enc
                            result = _enc(val, quality=quality)
                        elif field:
                            extracted = _evaluate_field(val, field)
                            result = {"value": extracted}
                        else:
                            result = {"value": str(val)}
                        widget_id = cmd.get("widget_id")
                        if widget_id and _proj:
                            for w in _proj.layout.get("widgets", []):
                                if w.get("id") == widget_id:
                                    if "min" in w:
                                        result["min"] = w["min"]
                                    if "max" in w:
                                        result["max"] = w["max"]
                                    result["status"] = _proj._evaluate_condition(result.get("value"), w)
                                    break
                        break
                else:
                    _log_stderr(f"get_output: slot '{slot_name}' not found")
        except Exception as e:
            _log_stderr(f"get_output({slot_name}/{output_name}): {e}")
            import traceback
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
        write_event({"type": "cmd_response", "_req_id": req_id, "data": result})

    elif c == "comms_snapshot":
        req_id = cmd["_req_id"]
        try:
            data = _comms_snapshot()
        except Exception as exc:
            data = {"error": str(exc), "tags": [], "values": {}}
        write_event({"type": "cmd_response", "_req_id": req_id, "data": data})

    elif c == "comms_values":
        req_id = cmd["_req_id"]
        try:
            data = _comms_values()
        except Exception as exc:
            data = {"error": str(exc), "values": {}, "statuses": []}
        write_event({"type": "cmd_response", "_req_id": req_id, "data": data})

    elif c == "comms_widget_values":
        req_id = cmd["_req_id"]
        try:
            data = _comms_widget_values()
        except Exception as exc:
            data = {"error": str(exc), "widgets": {}}
        write_event({"type": "cmd_response", "_req_id": req_id, "data": data})

    elif c == "comms_write":
        req_id = cmd["_req_id"]
        tag_id = cmd.get("tag", "")
        try:
            if _proj is None or _proj.comms is None:
                raise RuntimeError("Communications runtime is not running")
            _proj.comms.write(tag_id, cmd.get("value"))
            data = {"ok": True, "tag": tag_id}
        except Exception as exc:
            data = {"ok": False, "tag": tag_id, "error": str(exc)}
        write_event({"type": "cmd_response", "_req_id": req_id, "data": data})

    elif c == "prompt_response":
        if _proj:
            _proj._resolve_prompt(cmd["req_id"], cmd.get("value"))

    elif c == "exit":
        if _proj:
            _proj.stop_script()
            _proj.stop_runtime()
            _proj = None
        raise SystemExit(0)


def _on_log(entry):
    write_event({"type": "log", "entry": entry})


def _on_prompt(payload):
    write_event({"type": "prompt", **payload})


def _on_status(data):
    write_event({"type": "status", "slot": data["slot"], "status": data["status"]})


def _on_debug_paused(line, locals_dict):
    fn = locals_dict.pop("hmi_debug_file", "<hmi_script>")
    serialized = TraceDebugger.serialize_locals(locals_dict)
    write_event({"type": "paused", "line": line, "file": fn, "locals": serialized})


def _on_slot_finished(slot_name, data, ctrl):
    write_event({
        "type": "slot_finished",
        "slot": slot_name,
        "error": data.get("error"),
        "total_time_ms": data.get("total_time_ms", 0),
    })


def main():
    if getattr(sys, 'frozen', False):
        sp = _get_venv_sp()
        if sp:
            sys.path.insert(0, sp)
    try:
        import pylibdmtx.pylibdmtx
        from falcocv.core.ocr.industrial_ocr_engine import preload_engine
        preload_engine("server")
        from falcocv.core.classifier.inference import preload_classifier_model
        preload_classifier_model()
    except Exception:
        pass
    try:
        for raw in sys.stdin:
            line = raw.strip()
            if not line:
                continue
            try:
                cmd = json.loads(line)
                handle_cmd(cmd)
            except SystemExit:
                break
            except Exception as e:
                import traceback
                traceback.print_exc(file=sys.stderr)
                write_event({"type": "error", "message": f"{type(e).__name__}: {e}"})
    except (EOFError, KeyboardInterrupt):
        pass


if __name__ == "__main__":
    main()
