import abc
import uuid
import time


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


class ToolResults(list):
    @property
    def Count(self):
        return len(self)

    def __repr__(self):
        return f"ToolResults(Count={self.Count})"


class BaseTool(abc.ABC):
    def __init__(self, name):
        self.name = name
        self.prefix = ""
        self.uuid = str(uuid.uuid4())
        self.inputs = {}
        self.outputs = {}
        self.params = {}
        self.roi = None
        self.enabled = True
        self.last_execution_time_ms = 0.0
        self.last_error: str | None = None
        self._controller = None
        self._config_window = None
        self.data_changed = CallbackList()
        self.structure_changed = CallbackList()

    def set_controller(self, controller):
        self._controller = controller

    def mark_dirty(self):
        if self._controller:
            self._controller.mark_dirty(self)
        self.data_changed.emit()

    def execute(self, advance=False):
        start_ns = time.perf_counter_ns()
        try:
            self.run(advance=advance)
            self.last_error = None
        except Exception as e:
            self.last_error = type(e).__name__ + ": " + str(e)
        finally:
            end_ns = time.perf_counter_ns()
            self.last_execution_time_ms = (end_ns - start_ns) / 1e6
        self.data_changed.emit()

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
