import os
import sys
import importlib.util
from falcocv.core.tool_base import BaseTool, CallbackList
from falcocv.core.tool_registry import register_tool
from falcocv.core.script_debugger import TraceDebugger


_PY_VER = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


@register_tool(prefix="[PYTH]", default_name="Python Script", group="Logic")
class PythonScriptTool(BaseTool):
    debugger_paused = CallbackList()
    debugger_resumed = CallbackList()

    def __init__(self, name="Python Script"):
        super().__init__(name)
        self.params = {"script_path": "", "script_code": "", "input_ports": ["image_in"], "output_ports": ["image_out"]}
        self._instance = None
        self._script_modified_time = 0
        self._script_code_hash = 0
        self.engine_context = None
        self._debug_breakpoints = set()
        self._debug_editor = None

        self._debugger = TraceDebugger()
        self._debugger.get_breakpoints = lambda: self._get_breakpoints()
        self._debugger.paused.connect(lambda line, locs: self.debugger_paused.emit(line, locs))
        self._debugger.resumed.connect(lambda: self.debugger_resumed.emit())

        self._sync_ports()

    @property
    def _debug_active(self):
        return self._debugger.active

    def _sync_ports(self):
        self.inputs = {p: self.inputs.get(p, None) for p in self.params.get("input_ports", [])}
        self.outputs = {p: self.outputs.get(p, None) for p in self.params.get("output_ports", [])}

    def add_input_port(self, name: str | None = None) -> str:
        ports = list(self.params.get("input_ports", []))
        new_name = name or f"in_{len(ports)}"
        ports.append(new_name)
        self.params["input_ports"] = ports
        self._sync_ports()
        return new_name

    def add_output_port(self, name: str | None = None) -> str:
        ports = list(self.params.get("output_ports", []))
        new_name = name or f"out_{len(ports)}"
        ports.append(new_name)
        self.params["output_ports"] = ports
        self._sync_ports()
        return new_name

    def remove_input_port(self, index: int):
        ports = list(self.params.get("input_ports", []))
        if 0 <= index < len(ports):
            ports.pop(index)
            self.params["input_ports"] = ports
            self._sync_ports()

    def remove_output_port(self, index: int):
        ports = list(self.params.get("output_ports", []))
        if 0 <= index < len(ports):
            ports.pop(index)
            self.params["output_ports"] = ports
            self._sync_ports()

    def set_input_ports(self, names: list[str]):
        self.params["input_ports"] = [n.strip() for n in names if n.strip()]
        self._sync_ports()

    def set_output_ports(self, names: list[str]):
        self.params["output_ports"] = [n.strip() for n in names if n.strip()]
        self._sync_ports()

    def _load_module(self):
        sp = self.params.get("script_path", "")
        if sp and os.path.exists(sp):
            mt = os.path.getmtime(sp)
            if self._instance and self._script_modified_time == mt:
                return
            mn = f"falcocv_custom_{self.uuid.replace('-', '_')}"
            spec = importlib.util.spec_from_file_location(mn, sp)
            m = importlib.util.module_from_spec(spec)
            sys.modules[mn] = m
            spec.loader.exec_module(m)
            self._instance = m.CustomTool(engine_context=self.engine_context)
            self._script_modified_time = mt
            return
        code = self.params.get("script_code", "")
        if not code:
            self._instance = None
            return
        ch = hash(code)
        if self._instance and self._script_code_hash == ch:
            return
        ns = {}
        exec(code, ns)
        if "CustomTool" not in ns:
            raise AttributeError("Script must define 'CustomTool'")
        self._instance = ns["CustomTool"](engine_context=self.engine_context)
        self._script_code_hash = ch
        self._script_modified_time = 0

    def _get_breakpoints(self):
        return self._debug_editor.get_breakpoints() if self._debug_editor else self._debug_breakpoints

    def run(self, advance=False):
        self._sync_ports()
        if not self.params.get("script_path") and not self.params.get("script_code", "").strip():
            return
        self._load_module()
        if not self._instance:
            raise RuntimeError("Script not loaded.")
        if not hasattr(self._instance, "process"):
            raise AttributeError("Need process()")

        source_text = self._debug_editor.text() if self._debug_editor else self.params.get("script_code", "")
        script_filename = self._instance.process.__code__.co_filename

        try:
            results = self._debugger.run_with_debug(
                lambda: self._instance.process(self.inputs),
                script_filename=script_filename,
                source_text=source_text,
            )
        except Exception as e:
            import traceback
            raise RuntimeError(f"Script: {''.join(traceback.format_exception_only(type(e), e)).strip()}")

        if isinstance(results, dict):
            for k, v in results.items():
                if k in self.outputs:
                    self.outputs[k] = v
        self.data_changed.emit()

    def _debug_continue(self):
        self._debugger.continue_exec()

    def _debug_step_over(self):
        self._debugger.step_over()

    def _debug_step_into(self):
        self._debugger.step_into()

    def _debug_step_out(self):
        self._debugger.step_out()

    def _stop_debug(self):
        self._debugger.stop()

    def get_config_widget(self):
        from .config import PythonScriptConfig
        return PythonScriptConfig(self)

    def serialize(self, job_manager=None) -> dict:
        self._sync_ports()
        path = self.params.get("script_path")
        if path and os.path.exists(path):
            try:
                with open(path, "r") as f:
                    self.params["script_code"] = f.read()
            except:
                pass
        return super().serialize(job_manager)

    def deserialize(self, data: dict, job_manager=None):
        super().deserialize(data, job_manager)
        self._sync_ports()
        self._instance = None
