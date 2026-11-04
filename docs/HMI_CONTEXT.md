# HMI runtime context

The HMI script and the job Python Script tools run in the same HMI worker
process. `HMIProject` exposes a runtime `context` dictionary that is passed to
job workflows through `engine_context`.

The context is intentionally **not serialized**. It is for live objects such as:

- an already-connected PLC;
- a camera/scanner controller;
- a shared client or device handle.

## HMI script

The HMI namespace contains both `context` and `proj.context`:

```python
context["plc"] = plc
context["mecheye"] = mecheye_ctl

while not _stop.is_set():
    run_slot("Slot1")
    sleep(1)
```

`run_slot()` injects the same dictionary into the slot's
`workflow.engine_context`.

## Job Python Script

`PythonScriptTool` passes `engine_context` to the `CustomTool` constructor:

```python
class CustomTool:
    def __init__(self, engine_context):
        self.engine = engine_context or {}

    def process(self, inputs):
        plc = self.engine.get("plc")
        mecheye = self.engine.get("mecheye")
```

The instance is recreated when the context identity changes. Nested workflows
inside a `Block` receive the same context.

## Lifecycle

- The HMI script owns and disconnects the live objects.
- The context dictionary is shared by reference; mutate keys instead of
  replacing the dictionary while a job is running.
- `vision`/`falcocv` tools other than Python Script do not consume the context;
  they keep using their declared input ports.
