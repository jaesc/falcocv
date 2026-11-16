"""Evaluate four connector Height Blobs 3D tools against JSON nominals.

Setup in the Python Script tool
--------------------------------
Inputs (type **Results**):
    conn1, conn2, conn3, conn4
        Connect each one to the ``results`` output of its Height Blobs 3D tool.

Outputs:
    results  (type **Results**)  one row per pin
    ok       (type **Any**)      global OK/NG (bool)

Each ``results`` row carries the connector, pin, measured ``u/v/h``, the
per-axis ``min``/``max``/``ac_name``, per-axis OK flags and the pin OK.

Set ``sort_mode="grid"`` on every Height Blobs 3D tool.  Pins are paired by the
order of the results list: pin1 is the first result, pin2 the second, and so on.
The number of pins is always read from the JSON, never hard-coded here.

JSON
----
By default the file is ``connectors.json`` next to this script.  Change
``NOMINALS_JSON`` below, or set ``engine_context["connectors_json"]`` at
runtime.  The structure is::

    {
      "conn1": {
        "pin1_u": {
          "ac_name": "VAC_CONN1_PIN1_U_MM_VISION",
          "type": "number",
          "comparison": "range",
          "min": 10.2,
          "max": 10.9
        },
        "pin1_v": { "...": "..." },
        "pin1_h": { "...": "..." }
      },
      "conn2": { "...": "..." }
    }

Master mode
-----------
Set ``MASTER_MODE`` below, or set ``engine_context["connectors_master"]``::

    True                       # master all four connectors
    "conn1"                    # master only conn1
    ["conn1", "conn3"]         # several
    {"conn1": True}            # per connector

Only the targeted connectors are processed while mastering.  If a connector
section is missing or has no pins, it is generated from the measured blobs.
If it already has pins, their ``ac_name``/``type``/``comparison`` and any other
manual fields are preserved and only ``min``/``max`` are updated.  A pin-count
mismatch aborts the write for that connector, so a bad detection cannot destroy
the nominals.  Delete or empty a section to regenerate it.

``min``/``max`` are written as ``value +/- MASTER_TOL[axis]``.  The JSON is
backed up to ``connectors.json.bak`` and written atomically.
"""

from __future__ import annotations

import copy
import json
import math
import os
import re
import shutil
import tempfile

from falcocv.core.tool_base import ToolResult, ToolResults

# ── Configuration ─────────────────────────────────────────────────────

CONNECTORS = ("conn1", "conn2", "conn3", "conn4")
AXES = ("u", "v", "h")

#: Fallback when ``engine_context["connectors_master"]`` is not set.
MASTER_MODE = False

#: Fixed mastering tolerance per axis (mm): min = value - tol, max = value + tol.
MASTER_TOL = {"u": 0.7, "v": 0.7, "h": 0.7}

#: Generated tag name for auto-created pin entries.
AC_NAME_PATTERN = "VAC_{conn}_{pin}_{axis}_MM_VISION"

try:
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:  # Embedded scripts have no __file__.
    _SCRIPT_DIR = os.getcwd()

#: Relative paths are resolved against the script directory.
NOMINALS_JSON = "connectors.json"

_PIN_RE = re.compile(r"^pin(\d+)_(u|v|h)$")


class CustomTool:
    def __init__(self, engine_context):
        self.engine = engine_context or {}
        self.log = self.engine.get("log")

        self._path = None
        self._mtime = None
        self._error_mtime = None
        self._data = None
        self._missing_logged = False
        self._logged_keys = set()

    # ── Logging and JSON I/O ─────────────────────────────────────────

    def _log(self, message, level="info"):
        if callable(self.log):
            try:
                self.log(message, level)
                return
            except Exception:
                pass
        print(f"[CONN] {message}")

    def _log_once(self, key, message, level="warning"):
        if key in self._logged_keys:
            return
        self._logged_keys.add(key)
        self._log(message, level)

    def _json_path(self):
        raw = self.engine.get("connectors_json") or NOMINALS_JSON
        raw = os.path.expanduser(os.fspath(raw))
        if not os.path.isabs(raw):
            raw = os.path.join(_SCRIPT_DIR, raw)
        return os.path.abspath(raw)

    def _load(self, allow_missing=False):
        path = self._json_path()
        if path != self._path:
            self._path = path
            self._mtime = None
            self._error_mtime = None
            self._data = None
            self._missing_logged = False

        if not os.path.exists(path):
            self._data = None
            self._mtime = None
            self._error_mtime = None
            if not allow_missing and not self._missing_logged:
                self._log(f"Nominals JSON not found: {path}", "warning")
                self._missing_logged = True
            return {}

        try:
            mtime = os.path.getmtime(path)
        except OSError as exc:
            self._log(f"Cannot stat {path}: {exc}", "error")
            return None

        if self._data is not None and self._mtime == mtime:
            return self._data
        if self._error_mtime == mtime:
            return None

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if not isinstance(data, dict):
                raise ValueError("top-level JSON must be an object")
        except (OSError, ValueError) as exc:
            self._error_mtime = mtime
            self._data = None
            self._log(f"Cannot read {path}: {exc}", "error")
            return None

        self._data = data
        self._mtime = mtime
        self._error_mtime = None
        self._missing_logged = False
        return data

    def _save(self, data):
        path = self._json_path()
        try:
            folder = os.path.dirname(path) or "."
            os.makedirs(folder, exist_ok=True)
            if os.path.exists(path):
                shutil.copy2(path, path + ".bak")
            fd, tmp = tempfile.mkstemp(
                prefix=".connectors_", suffix=".json.tmp", dir=folder
            )
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as fh:
                    json.dump(data, fh, indent=2, ensure_ascii=False)
                    fh.write("\n")
                os.replace(tmp, path)
            except Exception:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as exc:
            self._log(f"Cannot write {path}: {exc}", "error")
            # Force a reload from disk on the next run.
            self._data = None
            self._mtime = None
            self._error_mtime = None
            return False

        self._data = data
        try:
            self._mtime = os.path.getmtime(path)
        except OSError:
            self._mtime = None
        self._error_mtime = None
        self._missing_logged = False
        return True

    # ── Master flag ──────────────────────────────────────────────────

    def _master_targets(self):
        raw = self.engine.get("connectors_master", MASTER_MODE)
        names = set(CONNECTORS)
        if isinstance(raw, bool):
            return names if raw else set()
        if isinstance(raw, (int, float)):
            return names if raw else set()
        if isinstance(raw, str):
            return {part.strip() for part in raw.split(",") if part.strip()} & names
        if isinstance(raw, dict):
            return {
                str(key)
                for key, value in raw.items()
                if value and str(key) in names
            }
        if isinstance(raw, (list, tuple, set, frozenset)):
            return {str(value) for value in raw if str(value) in names}
        return set()

    # ── Small helpers ────────────────────────────────────────────────

    @staticmethod
    def _pin_numbers(section):
        if not isinstance(section, dict):
            return []
        pins = set()
        for key in section:
            match = _PIN_RE.match(key)
            if match and int(match.group(1)) > 0:
                pins.add(int(match.group(1)))
        return sorted(pins)

    @staticmethod
    def _blob_value(blob, axis):
        if blob is None:
            return None
        if hasattr(blob, "get"):
            value = blob.get(axis)
        else:
            value = getattr(blob, axis, None)
        try:
            value = float(value)
        except (TypeError, ValueError):
            return None
        return value if math.isfinite(value) else None

    @staticmethod
    def _as_list(value):
        if value is None:
            return []
        if isinstance(value, (list, tuple)):
            return list(value)
        return [value]

    def _row(
        self,
        conn,
        pin,
        values=None,
        limits=None,
        checks=None,
        ok=False,
        mastered=False,
        message="",
    ):
        values = values or {}
        limits = limits or {}
        checks = checks or {}

        def field(axis, key):
            spec = limits.get(axis)
            return spec.get(key) if isinstance(spec, dict) else None

        pin = int(pin)
        return ToolResult(
            connector=conn,
            pin=pin,
            pin_name=f"pin{pin}" if pin else "",
            u=values.get("u"),
            v=values.get("v"),
            h=values.get("h"),
            u_min=field("u", "min"),
            u_max=field("u", "max"),
            v_min=field("v", "min"),
            v_max=field("v", "max"),
            h_min=field("h", "min"),
            h_max=field("h", "max"),
            u_ac_name=field("u", "ac_name"),
            v_ac_name=field("v", "ac_name"),
            h_ac_name=field("h", "ac_name"),
            u_ok=checks.get("u"),
            v_ok=checks.get("v"),
            h_ok=checks.get("h"),
            ok=bool(ok),
            mastered=bool(mastered),
            message=message,
        )

    # ── Master ───────────────────────────────────────────────────────

    def _master_connector(self, data, conn, blobs):
        section = data.get(conn)
        if not isinstance(section, dict):
            section = {}

        pins = self._pin_numbers(section)
        if not blobs:
            message = "no results to master"
            self._log_once(f"{conn}:master:no-results", f"{conn}: {message}", "warning")
            return [self._row(conn, 0, ok=False, message=message)], False

        if pins and len(pins) != len(blobs):
            message = (
                f"pin count mismatch: JSON={len(pins)} measured={len(blobs)}; "
                "JSON not written"
            )
            self._log_once(
                f"{conn}:master:count-mismatch", f"{conn}: {message}", "error"
            )
            return [
                self._row(conn, pin, ok=False, message=message)
                for pin in pins
            ], False

        target_pins = pins or list(range(1, len(blobs) + 1))
        new_section = copy.deepcopy(section)
        generated = []

        for pin, blob in zip(target_pins, blobs):
            values = {axis: self._blob_value(blob, axis) for axis in AXES}
            missing = [axis for axis, value in values.items() if value is None]
            if missing:
                message = f"pin{pin}: invalid measured {', '.join(missing)}"
                self._log_once(
                    f"{conn}:master:invalid-values", f"{conn}: {message}", "error"
                )
                return [
                    self._row(conn, other, ok=False, message=f"not written: {message}")
                    for other in target_pins
                ], False

            for axis in AXES:
                key = f"pin{pin}_{axis}"
                spec = new_section.get(key)
                if not isinstance(spec, dict):
                    spec = {}
                if not spec.get("ac_name"):
                    spec["ac_name"] = AC_NAME_PATTERN.format(
                        conn=conn.upper(),
                        pin=f"PIN{pin}",
                        axis=axis.upper(),
                    )
                spec.setdefault("type", "number")
                spec.setdefault("comparison", "range")
                tol = float(MASTER_TOL.get(axis, 0.0) or 0.0)
                spec["min"] = round(values[axis] - tol, 4)
                spec["max"] = round(values[axis] + tol, 4)
                new_section[key] = spec

            generated.append((pin, values))

        data[conn] = new_section
        rows = []
        for pin, values in generated:
            limits = {
                axis: new_section[f"pin{pin}_{axis}"]
                for axis in AXES
            }
            rows.append(
                self._row(
                    conn,
                    pin,
                    values=values,
                    limits=limits,
                    checks={axis: True for axis in AXES},
                    ok=True,
                    mastered=True,
                )
            )
        return rows, True

    # ── Run ──────────────────────────────────────────────────────────

    def _evaluate_connector(self, data, conn, blobs):
        section = data.get(conn)
        if not isinstance(section, dict):
            message = "missing JSON section"
            self._log_once(
                f"{conn}:run:missing-section", f"{conn}: {message}", "warning"
            )
            return [self._row(conn, 0, ok=False, message=message)], False

        pins = self._pin_numbers(section)
        if not pins:
            message = "no pins defined in JSON"
            self._log_once(f"{conn}:run:no-pins", f"{conn}: {message}", "warning")
            return [self._row(conn, 0, ok=False, message=message)], False

        if len(blobs) != len(pins):
            message = (
                f"blob count mismatch: JSON={len(pins)} measured={len(blobs)}"
            )
            return [
                self._row(conn, pin, ok=False, message=message)
                for pin in pins
            ], False

        connector_ok = True
        rows = []
        for pin, blob in zip(pins, blobs):
            values = {axis: self._blob_value(blob, axis) for axis in AXES}
            checks = {}
            limits = {}
            messages = []

            for axis in AXES:
                spec = section.get(f"pin{pin}_{axis}")
                if not isinstance(spec, dict):
                    continue
                limits[axis] = spec

                value = values[axis]
                if value is None:
                    checks[axis] = False
                    messages.append(f"{axis}: invalid measured value")
                    continue

                comparison = str(spec.get("comparison", "range")).lower()
                if comparison != "range":
                    checks[axis] = False
                    messages.append(f"{axis}: unsupported comparison {comparison!r}")
                    continue

                try:
                    vmin = float(spec["min"])
                    vmax = float(spec["max"])
                except (KeyError, TypeError, ValueError):
                    checks[axis] = False
                    messages.append(f"{axis}: invalid min/max")
                    continue

                checks[axis] = vmin <= value <= vmax
                if not checks[axis]:
                    messages.append(
                        f"{axis}: {value:.4f} not in [{vmin:.4f}, {vmax:.4f}]"
                    )

            pin_ok = bool(checks) and all(checks.values())
            if not checks:
                messages.append("no u/v/h ranges defined")
            connector_ok = connector_ok and pin_ok
            rows.append(
                self._row(
                    conn,
                    pin,
                    values=values,
                    limits=limits,
                    checks=checks,
                    ok=pin_ok,
                    message="; ".join(messages),
                )
            )
        return rows, connector_ok

    # ── Entry point ──────────────────────────────────────────────────

    def process(self, inputs):
        targets = self._master_targets()
        master_run = bool(targets)
        active = (
            [conn for conn in CONNECTORS if conn in targets]
            if master_run
            else list(CONNECTORS)
        )

        data = self._load(allow_missing=master_run)
        data_error = data is None
        if data is None:
            data = {}

        before = None
        if master_run and not data_error:
            before = json.dumps(data, sort_keys=True)

        rows = []
        connector_ok = []
        for conn in active:
            blobs = self._as_list(inputs.get(conn))
            if master_run:
                if data_error:
                    message = "cannot read nominals JSON"
                    rows.append(self._row(conn, 0, ok=False, message=message))
                    connector_ok.append(False)
                    continue
                conn_rows, ok = self._master_connector(data, conn, blobs)
            else:
                conn_rows, ok = self._evaluate_connector(data, conn, blobs)
            rows.extend(conn_rows)
            connector_ok.append(ok)

        saved = True
        if master_run and not data_error and before is not None:
            after = json.dumps(data, sort_keys=True)
            if after != before:
                saved = self._save(data)
                if saved:
                    self._log(
                        f"Mastered {', '.join(sorted(targets))} -> "
                        f"{self._json_path()}",
                        "info",
                    )
                else:
                    connector_ok.append(False)

        return {
            "results": ToolResults(rows),
            "ok": bool(connector_ok) and all(connector_ok) and saved,
            "output image": inputs.get("input image"),
        }
