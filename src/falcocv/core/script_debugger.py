import ast
import sys
import threading
import types
from falcocv.core.tool_base import CallbackList


class TraceDebugger:
    """Generic sys.settrace-based debugger for Python scripts.

    Single-file usage (backward compat):
        debugger = TraceDebugger()
        debugger.get_breakpoints = lambda: my_breakpoint_set

        def my_run():
            return my_function(inputs)

        debugger.run_with_debug(my_run, script_filename="<string>", source_text=source)

    Multi-file usage (v2.0):
        debugger = TraceDebugger()
        debugger.get_breakpoints_for_file = lambda fn: per_file_bps.get(fn, set())

        def my_run():
            return my_function(inputs)

        bps = {fn: TraceDebugger._snap_breakpoints(b, sources[fn])
               for fn, b in per_file_bps.items()}
        debugger.run_with_debug(my_run, script_filenames=frozenset(sources.keys()),
                                source_text=None, breakpoints=bps)
    """

    def __init__(self):
        self.get_breakpoints = lambda: set()
        self.get_breakpoints_for_file = lambda fn: set()
        self.paused = CallbackList()
        self.resumed = CallbackList()

        self._active = False
        self._breakpoints: dict[str, set[int]] = {}
        self._script_filenames: frozenset[str] = frozenset()
        self._step_mode = "continue"
        self._event: threading.Event | None = None
        self._frame = None
        self._depth = 0
        self._target_depth = -1
        self._step_over_depth = 0
        self._step_over_start_line = 0
        self._last_event_call = False
        self._last_line = 0
        self._last_code = None
        self._script_filename: str = ""
        self._wait_strategy = None

    @property
    def active(self):
        return self._active

    def set_wait_strategy(self, fn):
        """Inject a custom wait function for the paused event loop.
        fn(event: threading.Event) is called instead of event.wait().
        Allows Qt processEvents() to run during debugger pause."""
        self._wait_strategy = fn

    # --- Public API ---

    def run_with_debug(self, fn, script_filename="<string>", source_text=None,
                        script_filenames: frozenset[str] | None = None,
                        breakpoints: dict[str, set[int]] | None = None):
        """Run fn with trace only if breakpoints exist.

        Backward compat: set script_filename and source_text for single-file.
        Multi-file: pass script_filenames and breakpoints dict.
        """
        if script_filenames and breakpoints is not None:
            self._breakpoints = breakpoints
            self._script_filenames = script_filenames
            flat = set()
            for s in breakpoints.values():
                flat |= s
            if not flat:
                return fn()
        else:
            self._script_filename = script_filename
            bps = self.get_breakpoints()
            if not bps:
                return fn()
            snapped = self._snap_breakpoints(bps, source_text)
            if not snapped:
                return fn()
            self._breakpoints = {script_filename: snapped}
            self._script_filenames = frozenset([script_filename])

        self._depth = 0
        self._target_depth = -1
        self._step_mode = "continue"
        self._last_event_call = False
        self._active = True
        old = sys.gettrace()
        try:
            sys.settrace(self._trace_func)
            result = fn()
        finally:
            self._active = False
            self._breakpoints.clear()
            sys.settrace(old)
        self.resumed.emit()
        return result

    def run_with_trace(self, fn, script_filename="<string>",
                       script_filenames: frozenset[str] | None = None):
        """Install trace unconditionally and check breakpoints dynamically."""
        self._script_filename = script_filename
        if script_filenames:
            self._script_filenames = script_filenames
        else:
            self._script_filenames = frozenset([script_filename])
        self._depth = 0
        self._target_depth = -1
        self._step_mode = "continue"
        self._last_event_call = False
        self._active = True
        old = sys.gettrace()
        try:
            sys.settrace(self._trace_func)
            result = fn()
        finally:
            self._active = False
            sys.settrace(old)
        self.resumed.emit()
        return result

    def stop(self):
        self._active = False
        if self._event is not None:
            self._event.set()
        self._frame = None
        self.resumed.emit()

    def continue_exec(self):
        self._step_mode = "continue"
        self._target_depth = -1
        if self._event is not None:
            self._event.set()

    def step_over(self):
        self._step_mode = "step_over"
        self._step_over_depth = self._depth
        self._step_over_start_line = self._last_line
        if self._event is not None:
            self._event.set()

    def step_into(self):
        self._step_mode = "step_into"
        self._target_depth = -1
        if self._event is not None:
            self._event.set()

    def step_out(self):
        self._step_mode = "step_out"
        self._target_depth = self._depth - 1
        if self._event is not None:
            self._event.set()

    # --- Internal ---

    def _trace_func(self, frame, event, arg):
        if not self._active:
            return None
        if event == 'call':
            self._depth += 1
            in_script = (frame.f_code.co_filename in self._script_filenames)
            if self._step_mode == "step_into":
                if in_script:
                    self._frame = frame
                    self._last_event_call = True
                    self._pause(frame)
                else:
                    self._step_mode = "step_over"
                    self._step_over_depth = self._depth - 1
                    self._step_over_start_line = self._last_line
            elif self._step_mode == "step_over":
                pass

            if self._step_mode == "continue" and not in_script:
                return None

            return self._trace_func
        elif event == 'return':
            self._depth -= 1
            if self._step_mode == "step_out" and self._depth <= self._target_depth:
                self._frame = frame
                self._last_event_call = False
                self._pause(frame)
            return self._trace_func
        elif event == 'line':
            self._last_event_call = False
            ln = frame.f_lineno
            fn = frame.f_code.co_filename
            in_script = (fn in self._script_filenames)
            if in_script:
                if self._step_mode == "step_over":
                    d = getattr(self, '_step_over_depth', self._depth)
                    if self._depth <= d and ln != self._step_over_start_line:
                        self._frame = frame
                        self._pause(frame)
                        return self._trace_func
                if self._step_mode == "step_into":
                    self._frame = frame
                    self._pause(frame)
                    return self._trace_func
            if self._step_mode == "continue" and in_script:
                bps = self._breakpoints.get(fn, set())
                bps |= self.get_breakpoints_for_file(fn)
                bps |= self.get_breakpoints()
                if ln in bps:
                    self._frame = frame
                    self._pause(frame)
            return self._trace_func
        elif event == 'exception':
            return self._trace_func
        return None

    def _pause(self, frame):
        self._last_line = frame.f_lineno
        self._last_code = frame.f_code
        locals_copy = dict(frame.f_locals)
        locals_copy["_debug_file"] = frame.f_code.co_filename
        self._event = threading.Event()
        self.paused.emit(frame.f_lineno, locals_copy)
        saved_trace = sys.gettrace()
        sys.settrace(None)
        try:
            if self._wait_strategy:
                self._wait_strategy(self._event)
            else:
                self._event.wait()
        finally:
            sys.settrace(saved_trace)
        self._event = None
        self.resumed.emit()

    @staticmethod
    def _snap_breakpoints(bps, source_text):
        if not bps or not source_text:
            return bps
        try:
            tree = ast.parse(source_text)
            exec_lines = {node.lineno for node in ast.walk(tree) if hasattr(node, 'lineno') and isinstance(node, ast.stmt)}
            if not exec_lines:
                return bps
            se = sorted(exec_lines)
            snapped = set()
            for bp in bps:
                if bp in exec_lines:
                    snapped.add(bp)
                else:
                    for exe in se:
                        if exe > bp:
                            snapped.add(exe)
                            break
                    else:
                        snapped.add(bp)
            return snapped
        except SyntaxError:
            return bps

    @staticmethod
    def serialize_locals(locals_dict, max_depth=3):
        """Serialize a frame's f_locals to JSON-safe dict for WS transport."""
        _seen = set()
        def _val(v, depth):
            if depth > max_depth:
                return "<max depth>"
            vid = id(v)
            if vid in _seen:
                return "<circular>"
            _seen.add(vid)
            try:
                if isinstance(v, (int, float, bool, str, type(None))):
                    return v
                if isinstance(v, (list, tuple)):
                    return [_val(item, depth+1) for item in v[:20]]
                if isinstance(v, dict):
                    return {str(k): _val(val, depth+1) for k, val in list(v.items())[:20]}
                if isinstance(v, types.FunctionType):
                    return f"<fn {v.__name__}>"
                if isinstance(v, types.ModuleType):
                    return f"<module {v.__name__}>"
                return repr(v)[:200]
            except Exception:
                return "<error>"
            finally:
                _seen.discard(vid)

        result = {}
        for name in sorted(locals_dict.keys()):
            if name.startswith('_'):
                continue
            try:
                result[name] = _val(locals_dict[name], 0)
            except Exception:
                result[name] = "<error>"
        return result
