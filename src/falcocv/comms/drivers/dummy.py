"""In-memory communications driver used for development and tests."""

from __future__ import annotations

import math
import time
from typing import Any

from ..base import CommsError, DeviceConfig, DeviceDriver, TagDef, coerce_tag_value
from .schema import ConfigField, DriverSchema


DUMMY_SCHEMA = DriverSchema(
    name="dummy",
    label="Dummy",
    description="In-memory device for development and signal monitoring.",
    fields=(
        ConfigField(
            "read_delay_ms", "Read delay (ms)", "int", 0, minimum=0,
            help="Simulated latency for each read.",
        ),
        ConfigField(
            "write_delay_ms", "Write delay (ms)", "int", 0, minimum=0,
            help="Simulated latency for each write.",
        ),
        ConfigField(
            "fail_reads", "Fail next reads", "int", 0, minimum=0,
            help="Number of following reads that should fail.",
        ),
        ConfigField(
            "fail_writes", "Fail next writes", "int", 0, minimum=0,
            help="Number of following writes that should fail.",
        ),
        ConfigField(
            "disconnect_on_start", "Disconnect on start", "bool", False,
            help="Start the dummy in a disconnected state.",
        ),
    ),
)


class DummyDriver(DeviceDriver):
    """A deterministic device with in-memory tags.

    Driver config:
      - ``read_delay_ms`` / ``write_delay_ms``: simulated latency.
      - ``fail_reads`` / ``fail_writes``: number of next operations that fail.
      - ``disconnect_on_start``: start in a disconnected state.
    """

    driver_name = "dummy"
    config_schema = DUMMY_SCHEMA

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self._tags = {tag.name: tag for tag in config.tags}
        self._values: dict[str, Any] = {
            tag.name: tag.default for tag in config.tags
        }
        self._connected = False
        self._start_time = time.monotonic()
        raw_simulate = config.config.get("simulate", {})
        self._simulate = raw_simulate if isinstance(raw_simulate, dict) else {}
        self._read_delay = max(
            0.0, float(config.config.get("read_delay_ms", 0) or 0) / 1000.0
        )
        self._write_delay = max(
            0.0, float(config.config.get("write_delay_ms", 0) or 0) / 1000.0
        )
        self._fail_reads = max(
            0, int(config.config.get("fail_reads", 0) or 0)
        )
        self._fail_writes = max(
            0, int(config.config.get("fail_writes", 0) or 0)
        )
        self._disconnect_on_start = bool(
            config.config.get("disconnect_on_start", False)
        )

    def start(self) -> None:
        self._start_time = time.monotonic()
        self._connected = not self._disconnect_on_start

    def stop(self) -> None:
        self._connected = False

    def is_connected(self) -> bool:
        return self._connected

    def read(self, tag: TagDef) -> Any:
        self._ensure_connected(tag, "read")
        if self._read_delay:
            time.sleep(self._read_delay)
        if self._fail_reads:
            self._fail_reads -= 1
            raise CommsError(
                "dummy read failure",
                device=self.config.id,
                tag=tag.id,
            )
        if tag.name in self._simulate:
            return self._simulated_value(tag)
        return self._values.get(tag.name)

    def write(self, tag: TagDef, value: Any) -> None:
        self._ensure_connected(tag, "write")
        if tag.access == "r":
            raise CommsError(
                "tag is read-only",
                device=self.config.id,
                tag=tag.id,
            )
        value = coerce_tag_value(tag, value)
        if self._write_delay:
            time.sleep(self._write_delay)
        if self._fail_writes:
            self._fail_writes -= 1
            raise CommsError(
                "dummy write failure",
                device=self.config.id,
                tag=tag.id,
            )
        if tag.name not in self._tags:
            raise CommsError(
                "tag is not defined on this device",
                device=self.config.id,
                tag=tag.id,
            )
        self._values[tag.name] = value

    def _simulated_value(self, tag: TagDef) -> Any:
        spec = self._simulate.get(tag.name)
        if not isinstance(spec, dict):
            raise CommsError(
                f"invalid simulation spec for tag '{tag.name}'",
                device=self.config.id,
                tag=tag.id,
            )
        kind = str(spec.get("kind", "counter")).lower()
        elapsed_ms = (time.monotonic() - self._start_time) * 1000.0
        try:
            if kind == "counter":
                period = max(float(spec.get("period_ms", 1000)), 0.001)
                step = int(float(spec.get("step", 1)))
                return int(elapsed_ms // period) * step
            if kind == "toggle":
                period = max(float(spec.get("period_ms", 500)), 0.001)
                return int(elapsed_ms // period) % 2 == 1
            if kind == "sine":
                period = max(float(spec.get("period_ms", 2000)), 0.001)
                minimum = float(spec.get("min", 0.0))
                maximum = float(spec.get("max", 100.0))
                phase = (elapsed_ms / period) * 2.0 * math.pi
                return minimum + (maximum - minimum) * ((math.sin(phase) + 1.0) / 2.0)
        except (TypeError, ValueError) as exc:
            raise CommsError(
                f"invalid simulation spec for tag '{tag.name}'",
                device=self.config.id,
                tag=tag.id,
                cause=exc,
            ) from exc
        raise CommsError(
            f"unsupported simulation kind '{kind}' for tag '{tag.name}'",
            device=self.config.id,
            tag=tag.id,
        )

    def _ensure_connected(self, tag: TagDef, operation: str) -> None:
        if not self._connected:
            raise CommsError(
                f"dummy device is not connected for {operation}",
                device=self.config.id,
                tag=tag.id,
            )
