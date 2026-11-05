# HMI runtime context

The HMI script and the job Python Script tools run in the same HMI worker
process. `HMIProject` exposes a runtime `context` dictionary that is passed to
job workflows through `engine_context`.

The context is intentionally **not serialized**. It can carry live objects and
callbacks such as:

- the HMI log callback;
- an already-connected PLC;
- a camera/scanner controller;
- a shared client or device handle.

## HMI script

The HMI namespace contains both `context` and `proj.context`:

```python
context["log"] = log
context["plc"] = plc

while not _stop.is_set():
    run_slot("Slot1")
    sleep(1)
```

`run_slot()` injects the same dictionary into the slot's
`workflow.engine_context`.

When the project has a `comms` section, `HMIProject.start_runtime()` also adds
the communications manager as `context["comms"]`. See
[`COMMS.md`](COMMS.md).

## Job Python Script

`PythonScriptTool` passes `engine_context` to the `CustomTool` constructor.
The `log` entry is the bound `HMIProject.log` method, so calls from the job
reach the same HMI log panel:

```python
class CustomTool:
    def __init__(self, engine_context):
        self.engine = engine_context or {}
        self.log = self.engine.get("log")

    def process(self, inputs):
        if callable(self.log):
            self.log("processing image", "info")
        plc = self.engine.get("plc")
```

The instance is recreated when the context identity changes. Nested workflows
inside a `Block` receive the same context.

## Lifecycle

- The HMI script owns and disconnects the live objects.
- The context dictionary is shared by reference; mutate keys instead of
  replacing the dictionary while a job is running.
- `vision`/`falcocv` tools other than Python Script do not consume the context;
  they keep using their declared input ports.
