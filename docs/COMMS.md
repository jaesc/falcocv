# Communications runtime

FalcoCV has a generic communications layer for HMI projects. It is independent
of Qt and of any specific PLC, Modbus or socket library.

The current implementation ships the `dummy` driver, which is enough to build
and test the HMI/job integration without hardware. Real drivers will be added
without changing the project or script API.

## Concepts

```text
HMI       = operator dashboard and designer
Station   = .hmi project + job slots + HMI script + communications
Comms     = devices, tags, quality, reconnection
Job       = vision workflow
```

The `.hmi` file is the single deployment unit. Communications are configured in
an optional `comms` section:

```json
{
  "comms": {
    "version": 1,
    "devices": [
      {
        "id": "dummy",
        "driver": "dummy",
        "enabled": true,
        "config": {
          "read_delay_ms": 0,
          "write_delay_ms": 0
        },
        "tags": [
          {
            "name": "trigger",
            "address": "trigger",
            "type": "bool",
            "access": "rw",
            "default": false
          },
          {
            "name": "count",
            "address": "count",
            "type": "int",
            "access": "rw",
            "default": 0
          }
        ]
      }
    ]
  }
}
```

A tag is addressed globally as `device_id.tag_name`, for example
`dummy.trigger`. The runtime state is never serialized; only the definitions are
stored in `.hmi`.

## Runtime API

`HMIProject.start_runtime()` validates the configuration, creates a
`CommsManager`, starts its devices and injects it into the HMI context:

```python
comms = context["comms"]
```

The HMI script also gets a direct `comms` namespace entry. A job Python Script
receives the same object through `engine_context`:

```python
class CustomTool:
    def __init__(self, engine_context):
        self.engine = engine_context or {}

    def process(self, inputs):
        comms = self.engine.get("comms")
        comms.write("dummy.count", 1)
        return {"output image": None}
```

Available calls:

```python
comms.read("dummy.trigger")             # direct read
comms.read_tag("dummy.trigger")         # TagValue(value, quality, timestamp)
comms.write("dummy.count", 3)
comms.read_many(["dummy.trigger", "dummy.count"])
comms.write_many({"dummy.count": 3, "dummy.message": "OK"})
comms.wait_for("dummy.trigger", True, timeout=10)

comms.tag("dummy.count")                # cached TagValue
comms.snapshot()                        # all cached values
comms.on_change("dummy.count", callback)
comms.status("dummy")
```

Quality is one of `good`, `stale`, `bad` or `disabled`. A failed read/write
marks the tag as `bad`; it must not be presented as a valid value.

## Dummy driver

The dummy driver keeps values in memory. It supports:

- initial values from each tag's `default`;
- optional `read_delay_ms` / `write_delay_ms`;
- `fail_reads` / `fail_writes` counters for error testing;
- `disconnect_on_start` for disconnected-state testing;
- simulated changing signals.

Example simulation:

```json
"config": {
  "simulate": {
    "counter": {"kind": "counter", "period_ms": 1000, "step": 1},
    "toggle":  {"kind": "toggle",  "period_ms": 500},
    "sine":    {"kind": "sine",    "period_ms": 2000, "min": 0, "max": 100}
  }
}
```

The signal is computed when the tag is read, so the monitor can show a counter,
toggle or sine changing without writing anything. The manager only starts an
internal polling thread for tags with `poll_ms > 0`.

## HMI Designer

The HMI Designer has a **Communications** tab. It edits only the serialisable
`comms` section of the project; it never opens devices. You can:

- add/remove devices;
- select the driver and edit its known configuration with generated fields
  (for example `read_delay_ms` and `fail_reads` for the dummy);
- use the **Advanced config (JSON)** checkbox for driver-specific keys not
  covered by the schema (for example dummy `simulate`); when such keys exist
  the checkbox is already checked and the editor open when the device loads;
- add/remove tags and edit name, address, type, access, poll interval, default
  and description.

Driver schemas live with the driver (`ConfigField` / `DriverSchema`); adding a
real driver also adds its UI fields without hard-coding them in the HMI
Designer. The right-hand configuration pane is scrollable so the Tags table
remains reachable when the advanced JSON editor is open.

The tab validates the configuration on save, Check and Deploy. The worker is the
only place where devices are actually opened.

The designer edits an isolated in-memory working copy. Nothing is written to the
`.hmi` until Save. If Deploy is pressed with unsaved changes, it offers
**Save and Deploy**, **Deploy Without Saving** or **Cancel**; the default is to
save first.

The **Signal Monitor** tab has two modes:

- **Local test**: creates a temporary `CommsMonitorSession` from the project
  configuration and starts it explicitly with **Connect**. It must not be used
  while another process owns the same hardware.
- **Live worker**: talks to the deployed worker over HTTP, so it reads and writes
  the same `CommsManager` instance as the running HMI. It does not open a second
  PLC connection. The target is the server URL plus the designer project's
  `project_id`; that project must be deployed and its worker alive.

Both modes show tag values, quality and timestamps and let you read, toggle or
write a selected tag. The selected-tag editor sits above the table; bool tags use
an explicit False/True selector plus a Toggle button. While you are editing a
value, auto-refresh updates the table but does not overwrite your draft. Reads
and writes run in a worker thread so a slow PLC does not freeze the designer.

Live mode uses:

```text
GET  /api/hmi/comms/snapshot?project_id=...
POST /api/hmi/comms/write
```

The monitor shows an explicit `DISCONNECTED` / `CONNECTING` / `CONNECTED` /
`ERROR` badge. In Live mode it only becomes `CONNECTED` after the first
successful snapshot. If the project is not deployed or is stopped, the server
returns a structured 404 and the monitor leaves `Connect` enabled for a retry.

The worker handles the snapshot by reading the tags through its existing
`CommsManager` and returns tag definitions, values, quality and device status.
Writes are executed by the deployed worker, not by the HMI Designer.

## Driver batching

`DeviceDriver` has two optional hooks:

```python
def read_many(self, tags) -> dict[str, Any]:
    return {tag.id: self.read(tag) for tag in tags}

def write_many(self, items) -> None:
    for tag, value in items:
        self.write(tag, value)
```

They are not required. The default implementation keeps the per-tag contract,
but a driver can override them to batch network I/O or perform one coherent
transaction. The manager groups requested tags by device and calls these hooks;
the polling thread uses `read_many` too.

This is where protocol-specific optimizations belong:

- Omron can read whole structs and cache snapshots with a TTL;
- Beckhoff and Siemens can use their native symbol/address reads;
- raw TCP/Modbus can batch or keep the default.

The core only knows `read_many(tags)` and `write_many(items)`. It does not know
what a struct, CIP session or S7 address is.

Write verification and protocol-specific retries stay inside the driver. If a
batch write fails, the driver raises `CommsError` and the manager marks the
affected tags as `bad`.

## Lifecycle

The HMI worker calls:

```python
proj.start_runtime()
proj.start_script()
# ...
proj.stop_script()
proj.stop_runtime()
```

`start_runtime()` is idempotent. `stop_runtime()` removes `context["comms"]`
and disconnects every active device.

A project without a `comms` section behaves exactly as before.

## Future drivers and widgets

The tag model is the seam for future drivers:

- raw TCP / framed sockets;
- Modbus TCP;
- Beckhoff ADS, Siemens S7 and other PLCs;
- future MQTT/OPC UA adapters.

Future HMI widgets will read the same `TagStore`:

```json
{
  "type": "MeasureLabel",
  "source": "tag",
  "tag": "dummy.count",
  "field": "value"
}
```

Widgets will not talk to drivers directly, so changing a driver does not change
the dashboard configuration.
