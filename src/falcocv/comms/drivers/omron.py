"""Omron NJ/NX driver using the fast struct/snapshot strategy.

The public driver name is ``omron``.  It uses the same approach as the
production ``omron-fast`` implementation: one network read per struct, a
short-lived snapshot cache, shadow writes and optional write verification.

No dictionary file is persisted.  The dictionary is built by aphyt for the
current session only and is discarded when the driver stops.
"""

from __future__ import annotations

import threading
import time
from typing import Any

from ..base import (
    CommsError,
    DeviceConfig,
    DeviceDriver,
    TagDef,
    coerce_tag_value,
)
from .schema import ConfigField, DriverSchema
from .verify import values_match

OMRON_SCHEMA = DriverSchema(
    name="omron",
    label="Omron (omron-fast)",
    description="Omron NJ/NX using aphyt and struct snapshot caching.",
    allow_extra=False,
    fields=(
        ConfigField(
            "host", "Host", "str", required=True,
            placeholder="192.168.1.10",
            help="IP address of the Omron controller.",
        ),
        ConfigField(
            "snapshot_ttl_ms", "Snapshot TTL (ms)", "int", 50, minimum=0,
            help="Reuse a struct snapshot for this long; 0 disables caching.",
        ),
        ConfigField(
            "verify_writes", "Verify writes", "bool", False,
            help="Read back each written struct once and compare the values.",
        ),
        ConfigField(
            "write_retries", "Write retries", "int", 0, minimum=0,
            visible_when=("verify_writes", True),
            help=(
                "Extra write + verify attempts after a verification mismatch. "
                "0 means one write attempt."
            ),
        ),
    ),
)


def _create_client(host: str):
    try:
        from aphyt import omron as aphyt_omron
    except Exception as exc:
        raise CommsError(
            "driver 'omron' requires the aphyt package",
            device="omron",
            cause=exc,
        ) from exc
    return aphyt_omron.n_series.NSeries()


class OmronFastDriver(DeviceDriver):
    driver_name = "omron"
    config_schema = OMRON_SCHEMA

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self._client = None
        self._connected = False
        self._lock = threading.RLock()
        self._structs: dict[str, Any] = {}
        self._snap_t: dict[str, float] = {}
        self._ttl = max(
            0.0,
            float(config.config.get("snapshot_ttl_ms", 50) or 0) / 1000.0,
        )
        self._verify_writes = bool(config.config.get("verify_writes", False))
        self._write_retries = (
            max(0, int(config.config.get("write_retries", 0) or 0))
            if self._verify_writes
            else 0
        )
        # Validate addresses at construction time; the actual reads/writes still
        # resolve them per operation so a tag can be reused safely.
        for tag in config.tags:
            self._split_address(tag)

    @staticmethod
    def _split_address(tag: TagDef) -> tuple[str, str]:
        if "." not in tag.address:
            raise CommsError(
                f"Omron tag '{tag.id}' address must be 'Struct.Member'",
                device=tag.device,
                tag=tag.id,
            )
        struct, member = tag.address.split(".", 1)
        if not struct or not member:
            raise CommsError(
                f"Omron tag '{tag.id}' has an empty struct or member",
                device=tag.device,
                tag=tag.id,
            )
        return struct, member

    def start(self) -> None:
        if self._client is not None:
            return
        self._structs.clear()
        self._snap_t.clear()
        host = str(self.config.config.get("host", "")).strip()
        if not host:
            raise CommsError(
                "Omron driver requires config.host",
                device=self.config.id,
            )
        client = _create_client(host)
        try:
            client.connect_explicit(host)
            client.register_session()
        except Exception as exc:
            try:
                client.close_explicit()
            except Exception:
                pass
            raise CommsError(
                f"could not connect to Omron PLC at {host}: {exc}",
                device=self.config.id,
                cause=exc,
            ) from exc
        self._client = client
        self._connected = True

    def stop(self) -> None:
        client = self._client
        self._client = None
        self._connected = False
        self._structs.clear()
        self._snap_t.clear()
        if client is not None:
            try:
                client.close_explicit()
            except Exception:
                pass

    def is_connected(self) -> bool:
        client = self._client
        if not self._connected or client is None:
            return False
        try:
            dispatcher = client.connected_cip_dispatcher
            return bool(dispatcher.is_connected_explicit)
        except Exception:
            return True

    def read(self, tag: TagDef) -> Any:
        return self.read_many([tag])[tag.id]

    def read_many(self, tags) -> dict[str, Any]:
        self._ensure_connected()
        grouped: dict[str, list[tuple[TagDef, str]]] = {}
        for tag in tags:
            struct, member = self._split_address(tag)
            grouped.setdefault(struct, []).append((tag, member))

        values: dict[str, Any] = {}
        with self._lock:
            for struct, items in grouped.items():
                data = self._get_struct(struct)
                for tag, member in items:
                    values[tag.id] = self._read_member(
                        data, member, tag
                    )
        return values

    def write(self, tag: TagDef, value: Any) -> None:
        self.write_many([(tag, value)])

    def write_many(self, items) -> None:
        self._ensure_connected()
        grouped: dict[str, list[tuple[TagDef, Any, str]]] = {}
        for tag, value in items:
            if tag.access == "r":
                raise CommsError(
                    "tag is read-only",
                    device=self.config.id,
                    tag=tag.id,
                )
            struct, member = self._split_address(tag)
            grouped.setdefault(struct, []).append(
                (tag, coerce_tag_value(tag, value), member)
            )

        with self._lock:
            for struct, entries in grouped.items():
                self._write_struct(struct, entries)

    # ── internals ─────────────────────────────────────────────────────────

    def _ensure_connected(self) -> None:
        if self._client is None or not self._connected:
            raise CommsError(
                "Omron PLC is not connected",
                device=self.config.id,
            )

    def _fresh(self, struct: str) -> bool:
        if self._ttl <= 0 or struct not in self._structs:
            return False
        return (time.perf_counter() - self._snap_t.get(struct, 0.0)) < self._ttl

    def _get_struct(self, struct: str):
        if self._fresh(struct):
            return self._structs[struct]
        return self._read_struct(struct)

    def _read_struct(self, struct: str):
        try:
            data = self._client.read_variable(struct)
        except Exception as exc:
            self._structs.pop(struct, None)
            self._snap_t.pop(struct, None)
            raise CommsError(
                f"could not read Omron struct '{struct}': {exc}",
                device=self.config.id,
                cause=exc,
            ) from exc
        self._structs[struct] = data
        self._snap_t[struct] = time.perf_counter()
        return data

    def _read_member(self, data, member: str, tag: TagDef) -> Any:
        obj = data
        try:
            for key in member.split("."):
                obj = obj[key]
            return obj.value()
        except Exception as exc:
            raise CommsError(
                f"could not read Omron member '{member}' of '{tag.address}'",
                device=self.config.id,
                tag=tag.id,
                cause=exc,
            ) from exc

    def _write_member(self, data, member: str, value: Any, tag: TagDef) -> None:
        obj = data
        try:
            keys = member.split(".")
            for key in keys[:-1]:
                if key not in obj:
                    obj[key] = {}
                obj = obj[key]
            obj[keys[-1]].from_value(value)
        except Exception as exc:
            raise CommsError(
                f"could not set Omron member '{member}' of '{tag.address}'",
                device=self.config.id,
                tag=tag.id,
                cause=exc,
            ) from exc

    def _write_struct(self, struct: str, entries) -> None:
        try:
            shadow = self._get_struct(struct)
            for tag, value, member in entries:
                self._write_member(shadow, member, value, tag)
            self._client.write_variable(struct, shadow)
        except Exception as exc:
            self._structs.pop(struct, None)
            self._snap_t.pop(struct, None)
            if isinstance(exc, CommsError):
                raise
            raise CommsError(
                f"Omron write failed: {exc}",
                device=self.config.id,
                cause=exc,
            ) from exc

        if not self._verify_writes:
            self._structs[struct] = shadow
            self._snap_t[struct] = time.perf_counter()
            return

        last_error: CommsError | None = None
        for attempt in range(self._write_retries + 1):
            try:
                check = self._client.read_variable(struct)
            except Exception as exc:
                self._structs.pop(struct, None)
                self._snap_t.pop(struct, None)
                raise CommsError(
                    f"Omron write verification read failed: {exc}",
                    device=self.config.id,
                    cause=exc,
                ) from exc

            for tag, value, member in entries:
                actual = self._read_member(check, member, tag)
                if not values_match(tag, value, actual):
                    last_error = CommsError(
                        f"Omron write verification failed for '{tag.id}': "
                        f"wrote {value!r}, read {actual!r}",
                        device=self.config.id,
                        tag=tag.id,
                    )
                    break
            else:
                self._structs[struct] = check
                self._snap_t[struct] = time.perf_counter()
                return

            if attempt >= self._write_retries:
                break
            try:
                shadow = self._read_struct(struct)
                for tag, value, member in entries:
                    self._write_member(shadow, member, value, tag)
                self._client.write_variable(struct, shadow)
            except Exception as exc:
                self._structs.pop(struct, None)
                self._snap_t.pop(struct, None)
                if isinstance(exc, CommsError):
                    raise
                raise CommsError(
                    f"Omron retry write failed: {exc}",
                    device=self.config.id,
                    cause=exc,
                ) from exc

        self._structs.pop(struct, None)
        self._snap_t.pop(struct, None)
        raise last_error or CommsError(
            "Omron write verification failed",
            device=self.config.id,
        )
