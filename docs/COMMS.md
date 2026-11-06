# Communications runtime

FalcoCV has a generic communications layer for HMI projects. It is independent
of Qt and of any specific PLC, Modbus or socket library.

The current implementation ships the `dummy` driver for development and the
`omron`, `beckhoff` and `siemens` drivers for real controllers. Real drivers are
added without changing the project or script API.

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
- add/remove tags and edit name, address, type, access, poll interval, default,
  description and string length.

Driver schemas live with the driver (`ConfigField` / `DriverSchema`); adding a
real driver also adds its UI fields without hard-coding them in the HMI
Designer.

Changing the driver of a device with configuration or tags asks what to do with
its tags:

- **Keep tags** preserves the tag names, types, access, polling, defaults,
  descriptions and string lengths, but clears every address because address
  formats are driver-specific. The tag metadata and the device ID/enabled flag
  survive the change.
- **Delete tags** removes the tags for that device.
- **Cancel** keeps the current driver.

Both Keep and Delete clear the old driver-specific configuration and advanced
JSON, then rebuild the fields with the new driver's defaults. Tags kept with an
empty address are valid draft data: they can be saved, but Deploy and Signal
Monitor remain blocked until the addresses are filled.

The device ID is updated in the Devices list while typing. Because it is part
of every tag id (`device.tag`), changing it also rewrites tag-bound widget
references from the old prefix to the new one. HMI script references are not
rewritten automatically; the tab shows a warning when the script still contains
the old id.

In the Tags table, **Address** is free text because its format belongs to the
driver (`trigger` for dummy, `Vision.Trigger` for Omron, `MAIN.bTrigger` for
Beckhoff and `DB1.DBW2` for Siemens). **Type** is a selector
limited to `bool`, `int`, `float` and `str`; **Access** is a selector limited to
`r` (read-only) and `rw` (read/write).

`Poll ms` controls the manager's optional background polling, not whether the
tag can be read:

- `0` means on-demand reads only (HMI widgets, Signal Monitor, scripts,
  `comms.read()` or `wait_for()`);
- `N > 0` reads tags every `N` ms in the background and updates the tag store,
  quality and `on_change` callbacks.

The dashboard already requests tag-widget values periodically, so `poll_ms` does
not change that refresh rate; it is mainly useful for cached values, scripts and
change callbacks.

The Device configuration is in the upper scrollable pane of a vertical
splitter. It can be compressed so the Tags table gets more room; when the
configuration does not fit, only that upper pane scrolls. The tags pane expands
to fill the remaining height. Both the Tags table and the Signal Monitor table
use interactive column widths, so they can be resized by dragging the headers.

New tags get a type-appropriate default: `false` for `bool`, `0` for `int`,
`0.0` for `float` and empty (no default) for `str`. Changing the Type updates an
automatic default, but preserves a custom value when it is valid for the new
type.

The tab distinguishes structural errors from incomplete data:

- **Structural errors** (unknown driver, duplicate ids, invalid type/access or a
  default that cannot be converted) block saving and execution.
- **Incomplete data** (a required driver field such as Siemens `host`, or an
  empty tag address) is allowed while editing and saving. The Communications
  tab shows a persistent warning and the HMI Designer marks the tab as
  `Communications ⚠`.

Save and Save As accept incomplete projects, so a partially prepared
configuration is never lost. Check reports the incomplete fields as warnings.
Deploy, local Signal Monitor Connect and the worker startup path require a
complete configuration and block with the list of missing fields. The worker is
the only place where devices are actually opened.

The designer edits an isolated in-memory working copy. Nothing is written to the
`.hmi` until Save. If Deploy is pressed with unsaved changes, it offers
**Save and Deploy**, **Deploy Without Saving** or **Cancel**; the default is to
save first.

The **Signal Monitor** tab has two modes:

- **Local test**: creates a temporary `CommsMonitorSession` from the current
  Communications working copy and starts it explicitly with **Connect**. It
  applies the draft automatically, so saving is not required before testing,
  but the configuration must be complete. It must not be used while another
  process owns the same hardware. A device that fails to start is shown as an
  error, not as connected.
- **Live worker**: talks to the deployed worker over HTTP, so it reads and writes
  the same `CommsManager` instance as the running HMI. It does not open a second
  PLC connection. The target is the server URL plus the designer project's
  `project_id`; that project must be deployed and its worker alive.

Both modes show tag values, quality and timestamps and let you read, toggle or
write a selected tag. The selected-tag editor sits above the table; bool tags use
explicit **Set True** / **Set False** and **Toggle** buttons. Pressing Enter in
the value field writes the selected tag; the monitor consumes the Enter key and
its buttons are not dialog default buttons, so Enter cannot trigger Deploy or
Stop. While you are editing a value,
auto-refresh updates the table but does not overwrite your draft; focusing the
textbox selects its content and pauses value replacement. Table rows are updated
in place, preserving selection and scroll; the next refresh is scheduled after
the previous one completes instead of queuing fixed timer ticks. Reads and writes
run in a worker thread so a slow PLC does not freeze the designer.

Live mode uses:

```text
GET  /api/hmi/comms/snapshot?project_id=...
GET  /api/hmi/comms/values?project_id=...
POST /api/hmi/comms/write
```

Live mode fetches full tag definitions once, then uses the lightweight values
endpoint for subsequent refreshes. The default local URL uses `127.0.0.1`
rather than `localhost`; the monitor normalizes `localhost`/`::1` to the IPv4
loopback because on some Windows setups `localhost` adds ~2 seconds per HTTP
connection. A stale response from an old connection is ignored after
`Disconnect`.

The monitor shows an explicit `DISCONNECTED` / `CONNECTING` / `CONNECTED` /
`ERROR` badge. In Live mode it only becomes `CONNECTED` after the first
successful snapshot. If the project is not deployed or is stopped, the server
returns a structured 404 and the monitor leaves `Connect` enabled for a retry.

## Tag-bound widgets

HMI widgets can display configured tags read-only. `source` defaults to `job`;
when it is `tag`, the widget reads the namespaced tag id:

```json
{
  "type": "MeasureLabel",
  "source": "tag",
  "tag": "dummy.manual_value",
  "condition": "between",
  "min": 0,
  "max": 100
}
```

`StatusIndicator` and `LedIndicator` use the boolean value of the tag. The
designer populates the tag selector from `project.comms_config`; devices do not
need to be connected while editing.

The dashboard uses:

```text
GET /api/hmi/comms/widget-values?project_id=...
```

The worker evaluates widget conditions and returns `value`, `quality` and
`status`. A tag with quality other than `good` is shown as `—`/idle, never as a
valid measurement. Tag widgets are read-only in this phase.

The HMI Designer's **Settings > Dashboard tag refresh** menu stores the
dashboard refresh interval in the project layout as `tag_refresh_ms`. The
available values are `250 ms`, `500 ms` (the default), `1 s` and `2 s`. This
setting only controls tag-bound dashboard widgets: HMI scripts,
`comms.read()`, `wait_for()` and the Signal Monitor are not limited by it.

The dashboard makes a single in-flight request at a time, schedules the next
request after the previous one finishes, and pauses tag-widget refreshes while
the browser tab is hidden. If there are no tag-bound widgets, it performs no
tag-value requests.

The worker handles the snapshot by reading the tags through its existing
`CommsManager` and returns tag definitions, values, quality and device status.
Writes are executed by the deployed worker, not by the HMI Designer.

## Omron driver

The public driver name is `omron`. It implements the `omron-fast` strategy
internally:

- one network read per struct, shared by all requested members;
- short-lived snapshot cache controlled by `snapshot_ttl_ms`;
- shadow-copy writes;
- optional read-back verification, disabled by default;
- optional verification retries, visible only when verification is enabled.

```json
{
  "host": "192.168.1.10",
  "snapshot_ttl_ms": 50,
  "verify_writes": false,
  "write_retries": 0
}
```

Tags use `Struct.Member` addresses. There is no dictionary file at all: the
schema does not expose `dictionary_file`, unknown config keys are rejected, and
the driver never calls aphyt's `load_dictionary_file` or
`save_current_dictionary`. Aphyt builds its variable dictionary in memory for
the current session; stopping the driver discards it.

`aphyt` is loaded lazily when the driver starts, so the project can still be
edited without the dependency installed; connecting will report a clear
missing-package error.

## Beckhoff ADS driver

The public driver name is `beckhoff`.

```json
{
  "ams_net_id": "192.168.1.10.1.1",
  "ams_port": 851,
  "ip_address": "",
  "verify_writes": false,
  "write_retries": 0
}
```

- `ams_net_id` is mandatory.
- `ip_address` is optional. On Windows with a TwinCAT ADS route it is not used;
  it is needed on Linux or routed/NAT networks where pyads must create a route.
- Tags use symbolic names, for example `MAIN.bTrigger`.
- `read_many` uses `read_list_by_name`; `write_many` uses
  `write_list_by_name`.
- `verify_writes=true` reads the written symbols back once.
- `write_retries` is only visible when verification is enabled and controls
  extra write + verify attempts after a mismatch. `0` means one attempt.

The ADS route must be configured in TwinCAT before deploying.

## Siemens S7 driver

The public driver name is `siemens`.

```json
{
  "host": "192.168.1.20",
  "rack": 0,
  "slot": 1,
  "port": 102,
  "verify_writes": false,
  "write_retries": 0
}
```

Supported absolute addresses:

```text
I0.0 / Q0.0 / M0.0
IB0 / IW0 / ID0
QB0 / QW0 / QD0
MB0 / MW0 / MD0
DB1.DBX0.0
DB1.DBB0
DB1.DBW0
DB1.DBD0
```

- `X` is a bit and requires `bool`.
- `B` is a byte (8-bit integer).
- `W` is a 16-bit word (`int`).
- `D` is a 32-bit value (`int` or `float`/REAL).
- S7 `str` tags require a DB byte address and the tag **Length** field, and use
  the standard `STRING[n]` layout with its two-byte header.
- Multi-byte values are big-endian.
- Reads and writes are grouped by area/DB and byte range; writes use
  read-modify-write to preserve neighbouring bytes.
- `verify_writes=true` reads each written range back once.
- `write_retries` is only visible when verification is enabled and controls
  extra write + verify attempts after a mismatch. `0` means one attempt.

S7-1200/1500 controllers require PUT/GET enabled and non-optimized DBs for
absolute addressing. S7-300/400 use absolute addressing normally.

## Driver dependencies

Install all PLC libraries with:

```bash
pip install -e ".[plc]"
```

The extra installs `aphyt`, `pyads` and `python-snap7`. Driver imports are lazy,
so FalcoCV still starts when one of them is missing.

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

Write verification stays inside the driver. Without verification the driver
trusts the library response and never reads back or retries. With verification,
`write_retries` controls extra write + verify attempts after a value mismatch;
transport/read errors are not retried. If a batch write ultimately fails, the
driver raises `CommsError` and the manager marks the affected tags as `bad`.

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
- other PLCs;
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
