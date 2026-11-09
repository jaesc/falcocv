import abc
import uuid
import time

from falcocv.core.port_types import normalize_kind


class ToolResult:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)

    def items(self):
        return self.__dict__.items()

    def keys(self):
        return self.__dict__.keys()

    def get(self, key, default=None):
        return self.__dict__.get(key, default)

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.__dict__[key]
        raise KeyError(key)

    def __contains__(self, key):
        return isinstance(key, str) and key in self.__dict__


class CallbackList(list):
    def connect(self, cb):
        if cb not in self:
            self.append(cb)

    def disconnect(self, cb):
        try:
            self.remove(cb)
        except ValueError:
            pass

    def disconnect_owner(self, owner):
        """Drop callbacks bound to ``owner`` (including ``safe_refresh`` wrappers)."""
        def callback_owner(cb):
            bound = getattr(cb, "__self__", None)
            if bound is not None:
                return bound
            # functools.wraps() stores the original bound method here.
            wrapped = getattr(cb, "__wrapped__", None)
            return getattr(wrapped, "__self__", None)

        self[:] = [cb for cb in self if callback_owner(cb) is not owner]

    def emit(self, *args, **kwargs):
        for cb in self:
            cb(*args, **kwargs)


#: Optional hook installed by the Qt UI.  Tool signals are emitted from worker
#: threads during a background run, so the UI needs a way to marshal its
#: callbacks back to the GUI thread.  Headless callers leave this as ``None``.
_TOOL_SIGNAL_DISPATCHER = None


def set_tool_signal_dispatcher(dispatcher):
    """Install (or clear) the thread dispatcher used by :class:`ToolSignal`."""
    global _TOOL_SIGNAL_DISPATCHER
    _TOOL_SIGNAL_DISPATCHER = dispatcher


def get_tool_signal_dispatcher():
    return _TOOL_SIGNAL_DISPATCHER


class ToolSignal(CallbackList):
    """Callback list with suspend/resume and optional thread dispatch.

    ``data_changed`` is emitted by tools while a workflow runs on a worker
    thread.  Suspending the signal for the duration of the run avoids touching
    Qt widgets from that thread; resuming on the GUI thread emits one coalesced
    refresh when the run finishes.
    """

    def __init__(self):
        super().__init__()
        self._suspend_depth = 0
        self._pending = False

    @property
    def suspended(self) -> bool:
        return self._suspend_depth > 0

    def suspend(self) -> None:
        self._suspend_depth += 1

    def resume(self, emit_pending: bool = True) -> None:
        if self._suspend_depth <= 0:
            return
        self._suspend_depth -= 1
        if self._suspend_depth == 0 and self._pending:
            self._pending = False
            if emit_pending:
                self.emit()

    def emit(self, *args, **kwargs):
        if self._suspend_depth:
            self._pending = True
            return
        callbacks = list(self)
        dispatcher = _TOOL_SIGNAL_DISPATCHER
        if dispatcher is None:
            for cb in callbacks:
                cb(*args, **kwargs)
            return
        for cb in callbacks:
            dispatcher(cb, args, kwargs)


class ToolResults(list):
    @property
    def Count(self):
        return len(self)

    def __repr__(self):
        return f"ToolResults(Count={self.Count})"


class BaseTool(abc.ABC):
    #: Declared port kinds, e.g. ``{"input image": IMAGE}``.  Ports that are
    #: not listed are untyped (``ANY``); never inferred from the port name.
    input_port_types: dict[str, str] = {}
    output_port_types: dict[str, str] = {}
    #: Whether the config dialog offers the per-tool Run / Auto controls.
    #: Sources and special tools opt out.
    supports_manual_run = True
    #: Runtime-only context injected by the HMI (live PLC/scanner objects, ...).
    engine_context = None

    def __init__(self, name):
        self.name = name
        self.prefix = ""
        self.uuid = str(uuid.uuid4())
        self.inputs = {}
        self.outputs = {}
        self.params = {}
        self.port_metadata = {"inputs": {}, "outputs": {}}
        self.roi = None
        self.enabled = True
        self.last_execution_time_ms = 0.0
        self.last_error: str | None = None
        self._controller = None
        self._config_window = None
        #: UI-only service that serialises background workflow/tool runs.
        self._execution_service = None
        self.data_changed = ToolSignal()
        self.parameter_changed = ToolSignal()
        self.structure_changed = ToolSignal()

    def set_controller(self, controller):
        self._controller = controller

    def mark_dirty(self):
        if self._controller:
            self._controller.mark_dirty(self)
        self.data_changed.emit()
        self.parameter_changed.emit()

    def execute(self, advance=False):
        start_ns = time.perf_counter_ns()
        self.last_error = None
        try:
            self.run(advance=advance)
        except Exception as e:
            self.last_error = type(e).__name__ + ": " + str(e)
        finally:
            end_ns = time.perf_counter_ns()
            self.last_execution_time_ms = (end_ns - start_ns) / 1e6
        self.data_changed.emit()

    def set_error(self, message: str) -> None:
        """Mark the tool as failed without raising an exception.

        The workflow treats the tool as failed and honours its
        ``stop_on_error`` policy.  Outputs are kept so callers can inspect
        partial results or a zero count.
        """
        self.last_error = str(message)

    def fail_if_below_min_results(
        self, count: int, singular: str = "result", plural: str | None = None
    ) -> None:
        """Apply the tool's ``min_results`` policy to ``count``.

        ``min_results == 0`` disables the check.  A positive value marks the
        tool as failed when fewer results were produced.
        """
        minimum = int(self.params.get("min_results", 0) or 0)
        if minimum > 0 and count < minimum:
            noun = singular if count == 1 else (plural or f"{singular}s")
            self.set_error(f"Found {count} {noun}; minimum is {minimum}")

    @abc.abstractmethod
    def run(self, advance=False):
        pass

    def get_config_schema(self) -> dict:
        """Return JSON Schema describing the tool's configurable parameters.
        
        Default returns an empty schema. Tools override to provide
        structured parameter metadata for dynamic UI generation.
        """
        return {
            "type": "object",
            "properties": {},
        }

    def set_input(self, name, value):
        self.inputs[name] = value

    def get_input_port_type(self, name: str) -> str:
        declared = self.input_port_types.get(name)
        if declared is None:
            declared = self.port_metadata.get("inputs", {}).get(name)
        return normalize_kind(declared)

    def get_output_port_type(self, name: str) -> str:
        declared = self.output_port_types.get(name)
        if declared is None:
            declared = self.port_metadata.get("outputs", {}).get(name)
        return normalize_kind(declared)

    def get_output(self, name):
        return self.outputs.get(name)

    def set_roi(self, roi):
        self.roi = roi

    def _copy_overlays_to_input(self, output_vimg, input_vimg):
        if not self.params.get("copy_overlays_to_input", True):
            return
        out_layer = output_vimg.get_overlay_layer("default", source_tool=self.name)
        in_layer = input_vimg.get_overlay_layer("default", source_tool=self.name)
        in_layer.elements = [
            e for e in in_layer.elements
            if getattr(e, "source_tool", None) != self.name
        ]
        for elem in out_layer.elements:
            if getattr(elem, "source_tool", None) == self.name:
                in_layer.elements.append(elem)

    def serialize(self, job_manager=None) -> dict:
        return {
            "type": self.__class__.__name__,
            "uuid": self.uuid,
            "name": self.name,
            "prefix": self.prefix,
            "params": self.params,
            "enabled": self.enabled,
            "ports": {
                "inputs": list(self.inputs.keys()),
                "outputs": list(self.outputs.keys()),
            },
        }

    def deserialize(self, data: dict, job_manager=None):
        self.uuid = data.get("uuid", self.uuid)
        self.name = data.get("name", self.name)
        self.prefix = data.get("prefix", "")
        loaded_params = data.get("params", {})
        if isinstance(loaded_params, dict):
            self.params.update(loaded_params)
        self.enabled = data.get("enabled", True)
