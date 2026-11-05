"""Siemens S7 driver using standard absolute addresses."""

from __future__ import annotations

import re
import struct
import threading
from dataclasses import dataclass
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

SIEMENS_SCHEMA = DriverSchema(
    name="siemens",
    label="Siemens S7",
    description="Siemens S7 using standard absolute addresses.",
    allow_extra=False,
    fields=(
        ConfigField(
            "host", "Host", "str", required=True,
            placeholder="192.168.1.10",
            help="IP address of the S7 PLC.",
        ),
        ConfigField(
            "rack", "Rack", "int", 0, minimum=0, maximum=7,
            help="S7 rack number.",
        ),
        ConfigField(
            "slot", "Slot", "int", 1, minimum=0, maximum=31,
            help="S7 slot number; often 1 for S7-1200/1500 and 2 for S7-300.",
        ),
        ConfigField(
            "port", "TCP port", "int", 102, minimum=1, maximum=65535,
            help="S7 TCP port.",
        ),
        ConfigField(
            "verify_writes", "Verify writes", "bool", False,
            help=(
                "Read each written range back once and compare the values. "
                "No write is repeated."
            ),
        ),
    ),
)

_DB_RE = re.compile(
    r"^DB(?P<db>\d+)\.(?P<kind>DBX|DBB|DBW|DBD)"
    r"(?P<offset>\d+)(?:\.(?P<bit>[0-7]))?$",
    re.IGNORECASE,
)
_AREA_BIT_RE = re.compile(
    r"^(?P<area>[IQM])(?P<offset>\d+)\.(?P<bit>[0-7])$",
    re.IGNORECASE,
)
_AREA_SIZE_RE = re.compile(
    r"^(?P<area>[IQM])(?P<kind>[BWD])(?P<offset>\d+)$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _S7Tag:
    tag: TagDef
    area: str
    db: int
    offset: int
    width: int
    kind: str
    bit: int | None = None

    @property
    def end(self) -> int:
        return self.offset + self.width


def _tag_error(tag: TagDef, message: str) -> CommsError:
    return CommsError(message, device=tag.device, tag=tag.id)


def _parse_tag(tag: TagDef) -> _S7Tag:
    address = tag.address.strip().upper()
    match = _DB_RE.match(address)
    if match:
        db = int(match.group("db"))
        kind = match.group("kind").upper()
        offset = int(match.group("offset"))
        bit_text = match.group("bit")
        if kind == "DBX":
            if tag.type != "bool" or bit_text is None:
                raise _tag_error(
                    tag,
                    f"S7 address '{tag.address}' requires bool and a bit "
                    "(DBX...bit)",
                )
            return _S7Tag(tag, "DB", db, offset, 1, "bool", int(bit_text))
        if bit_text is not None:
            raise _tag_error(
                tag,
                f"S7 address '{tag.address}' does not accept a bit suffix",
            )
        if kind == "DBB":
            if tag.type == "str":
                if not tag.length or tag.length <= 0 or tag.length > 254:
                    raise _tag_error(
                        tag,
                        f"S7 string tag '{tag.id}' requires length 1..254",
                    )
                return _S7Tag(
                    tag, "DB", db, offset, int(tag.length) + 2, "string"
                )
            if tag.type == "int":
                return _S7Tag(tag, "DB", db, offset, 1, "int8")
            raise _tag_error(
                tag,
                f"S7 address '{tag.address}' is only valid for int or str",
            )
        if kind == "DBW":
            if tag.type != "int":
                raise _tag_error(
                    tag,
                    f"S7 address '{tag.address}' requires int",
                )
            return _S7Tag(tag, "DB", db, offset, 2, "int16")
        if tag.type == "float":
            return _S7Tag(tag, "DB", db, offset, 4, "float")
        if tag.type == "int":
            return _S7Tag(tag, "DB", db, offset, 4, "int32")
        raise _tag_error(
            tag,
            f"S7 address '{tag.address}' is only valid for int or float",
        )

    match = _AREA_BIT_RE.match(address)
    if match:
        if tag.type != "bool":
            raise _tag_error(
                tag,
                f"S7 address '{tag.address}' requires bool",
            )
        return _S7Tag(
            tag,
            match.group("area").upper(),
            0,
            int(match.group("offset")),
            1,
            "bool",
            int(match.group("bit")),
        )

    match = _AREA_SIZE_RE.match(address)
    if match:
        area = match.group("area").upper()
        kind = match.group("kind").upper()
        offset = int(match.group("offset"))
        if kind == "B":
            if tag.type != "int":
                raise _tag_error(
                    tag,
                    f"S7 address '{tag.address}' requires int",
                )
            return _S7Tag(tag, area, 0, offset, 1, "int8")
        if kind == "W":
            if tag.type != "int":
                raise _tag_error(
                    tag,
                    f"S7 address '{tag.address}' requires int",
                )
            return _S7Tag(tag, area, 0, offset, 2, "int16")
        if tag.type == "float":
            return _S7Tag(tag, area, 0, offset, 4, "float")
        if tag.type == "int":
            return _S7Tag(tag, area, 0, offset, 4, "int32")
        raise _tag_error(
            tag,
            f"S7 address '{tag.address}' is only valid for int or float",
        )

    raise _tag_error(
        tag,
        f"invalid S7 address '{tag.address}'; expected DB1.DBX0.0, "
        "DB1.DBW2, DB1.DBD4, M0.0, MW2, I0.0 or Q0.0",
    )


def _create_client(config: DeviceConfig):
    try:
        from s7 import Client
    except Exception:
        try:
            from snap7.client import Client
        except Exception as exc:
            raise CommsError(
                "driver 'siemens' requires the python-snap7 package",
                device=config.id,
                cause=exc,
            ) from exc
    return Client()


@dataclass(frozen=True)
class _Range:
    area: str
    db: int
    start: int
    end: int
    tags: tuple[_S7Tag, ...]


class SiemensS7Driver(DeviceDriver):
    driver_name = "siemens"
    config_schema = SIEMENS_SCHEMA
    _MAX_GAP_BYTES = 16

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self._client = None
        self._connected = False
        self._lock = threading.RLock()
        self._verify_writes = bool(config.config.get("verify_writes", False))
        self._specs: dict[str, _S7Tag] = {}
        for tag in config.tags:
            self._specs[tag.id] = _parse_tag(tag)

    def start(self) -> None:
        if self._client is not None:
            return
        host = str(self.config.config.get("host", "")).strip()
        if not host:
            raise CommsError(
                "Siemens driver requires config.host",
                device=self.config.id,
            )
        client = _create_client(self.config)
        try:
            client.connect(
                host,
                int(self.config.config.get("rack", 0) or 0),
                int(self.config.config.get("slot", 1) or 0),
                int(self.config.config.get("port", 102) or 102),
            )
        except Exception as exc:
            try:
                client.disconnect()
            except Exception:
                pass
            raise CommsError(
                f"could not connect to Siemens PLC at {host}: {exc}",
                device=self.config.id,
                cause=exc,
            ) from exc
        self._client = client
        self._connected = True

    def stop(self) -> None:
        client = self._client
        self._client = None
        self._connected = False
        if client is not None:
            try:
                client.disconnect()
            except Exception:
                pass

    def is_connected(self) -> bool:
        client = self._client
        if not self._connected or client is None:
            return False
        try:
            return bool(client.get_connected())
        except Exception:
            return True

    def read(self, tag: TagDef) -> Any:
        return self.read_many([tag])[tag.id]

    def read_many(self, tags) -> dict[str, Any]:
        self._ensure_connected()
        specs = [self._spec_for(tag) for tag in tags]
        if not specs:
            return {}
        with self._lock:
            values: dict[str, Any] = {}
            for data_range in self._ranges(specs):
                data = self._read_bytes(
                    data_range.area,
                    data_range.db,
                    data_range.start,
                    data_range.end - data_range.start,
                )
                for spec in data_range.tags:
                    values[spec.tag.id] = self._decode(spec, data_range.start, data)
            return values

    def write(self, tag: TagDef, value: Any) -> None:
        self.write_many([(tag, value)])

    def write_many(self, items) -> None:
        self._ensure_connected()
        entries = []
        for tag, value in items:
            if tag.access == "r":
                raise CommsError(
                    "tag is read-only",
                    device=self.config.id,
                    tag=tag.id,
                )
            entries.append((self._spec_for(tag), coerce_tag_value(tag, value)))
        if not entries:
            return

        with self._lock:
            for data_range in self._ranges([spec for spec, _ in entries]):
                raw = self._read_bytes(
                    data_range.area,
                    data_range.db,
                    data_range.start,
                    data_range.end - data_range.start,
                )
                data = bytearray(raw)
                range_entries = [
                    (spec, value)
                    for spec, value in entries
                    if spec in data_range.tags
                ]
                for spec, value in range_entries:
                    self._encode(spec, data, data_range.start, value)
                self._write_bytes(
                    data_range.area,
                    data_range.db,
                    data_range.start,
                    data,
                )
                if self._verify_writes:
                    self._verify_range(data_range, range_entries)

    def _spec_for(self, tag: TagDef) -> _S7Tag:
        spec = self._specs.get(tag.id)
        if spec is None:
            spec = _parse_tag(tag)
            self._specs[tag.id] = spec
        return spec

    def _ranges(self, specs: list[_S7Tag]) -> list[_Range]:
        grouped: dict[tuple[str, int], list[_S7Tag]] = {}
        for spec in specs:
            grouped.setdefault((spec.area, spec.db), []).append(spec)

        ranges: list[_Range] = []
        for (area, db), group in grouped.items():
            group = sorted(group, key=lambda item: (item.offset, item.end))
            start = group[0].offset
            end = group[0].end
            current: list[_S7Tag] = [group[0]]
            for spec in group[1:]:
                if spec.offset <= end + self._MAX_GAP_BYTES:
                    current.append(spec)
                    end = max(end, spec.end)
                else:
                    ranges.append(_Range(area, db, start, end, tuple(current)))
                    start = spec.offset
                    end = spec.end
                    current = [spec]
            ranges.append(_Range(area, db, start, end, tuple(current)))
        return ranges

    def _read_bytes(self, area: str, db: int, start: int, size: int) -> bytes:
        try:
            if area == "DB":
                return bytes(self._client.db_read(db, start, size))
            if area == "M":
                return bytes(self._client.mb_read(start, size))
            if area == "I":
                return bytes(self._client.eb_read(start, size))
            if area == "Q":
                return bytes(self._client.ab_read(start, size))
        except Exception as exc:
            raise CommsError(
                f"S7 read failed for {area}{db if area == 'DB' else ''}"
                f"[{start}:{size}]: {exc}",
                device=self.config.id,
                cause=exc,
            ) from exc
        raise CommsError(
            f"unsupported S7 area '{area}'",
            device=self.config.id,
        )

    def _write_bytes(self, area: str, db: int, start: int, data: bytes) -> None:
        if area not in ("DB", "M", "I", "Q"):
            raise CommsError(
                f"unsupported S7 area '{area}'",
                device=self.config.id,
            )
        try:
            if area == "DB":
                result = self._client.db_write(db, start, data)
            elif area == "M":
                result = self._client.mb_write(start, data)
            elif area == "I":
                result = self._client.eb_write(start, data)
            else:
                result = self._client.ab_write(start, data)
            if result not in (None, 0):
                raise CommsError(
                    f"S7 write returned status {result!r}",
                    device=self.config.id,
                )
        except CommsError:
            raise
        except Exception as exc:
            raise CommsError(
                f"S7 write failed for {area}{db if area == 'DB' else ''}"
                f"[{start}:{len(data)}]: {exc}",
                device=self.config.id,
                cause=exc,
            ) from exc

    def _decode(self, spec: _S7Tag, range_start: int, data: bytes) -> Any:
        offset = spec.offset - range_start
        chunk = data[offset:offset + spec.width]
        if len(chunk) != spec.width:
            raise CommsError(
                f"S7 read returned a short range for '{spec.tag.id}'",
                device=self.config.id,
                tag=spec.tag.id,
            )
        if spec.kind == "bool":
            return bool((chunk[0] >> int(spec.bit or 0)) & 1)
        if spec.kind == "int8":
            return struct.unpack(">B", chunk)[0]
        if spec.kind == "int16":
            return struct.unpack(">h", chunk)[0]
        if spec.kind == "int32":
            return struct.unpack(">i", chunk)[0]
        if spec.kind == "float":
            return struct.unpack(">f", chunk)[0]
        if spec.kind == "string":
            actual = chunk[1] if len(chunk) > 1 else 0
            characters = chunk[2:2 + actual]
            return characters.decode("latin-1", errors="replace")
        raise CommsError(
            f"unsupported S7 data kind '{spec.kind}'",
            device=self.config.id,
            tag=spec.tag.id,
        )

    def _encode(self, spec: _S7Tag, data: bytearray, range_start: int, value: Any) -> None:
        offset = spec.offset - range_start
        try:
            if spec.kind == "bool":
                mask = 1 << int(spec.bit or 0)
                if value:
                    data[offset] |= mask
                else:
                    data[offset] &= ~mask & 0xFF
                return
            if spec.kind == "int8":
                struct.pack_into(">B", data, offset, int(value))
                return
            if spec.kind == "int16":
                struct.pack_into(">h", data, offset, int(value))
                return
            if spec.kind == "int32":
                struct.pack_into(">i", data, offset, int(value))
                return
            if spec.kind == "float":
                struct.pack_into(">f", data, offset, float(value))
                return
            if spec.kind == "string":
                length = int(spec.tag.length or 0)
                encoded = str(value).encode("latin-1", errors="replace")
                if len(encoded) > length:
                    raise CommsError(
                        f"S7 string value for '{spec.tag.id}' is longer "
                        f"than {length} characters",
                        device=self.config.id,
                        tag=spec.tag.id,
                    )
                data[offset] = length
                data[offset + 1] = len(encoded)
                payload = encoded.ljust(length, b"\x00")
                data[offset + 2:offset + 2 + length] = payload
                return
        except CommsError:
            raise
        except Exception as exc:
            raise CommsError(
                f"could not encode S7 value for '{spec.tag.id}': {exc}",
                device=self.config.id,
                tag=spec.tag.id,
                cause=exc,
            ) from exc
        raise CommsError(
            f"unsupported S7 data kind '{spec.kind}'",
            device=self.config.id,
            tag=spec.tag.id,
        )

    def _verify_range(self, data_range: _Range, entries) -> None:
        try:
            data = self._read_bytes(
                data_range.area,
                data_range.db,
                data_range.start,
                data_range.end - data_range.start,
            )
        except Exception as exc:
            raise CommsError(
                f"S7 write verification read failed: {exc}",
                device=self.config.id,
                cause=exc,
            ) from exc
        for spec, expected in entries:
            actual = self._decode(spec, data_range.start, data)
            if not values_match(spec.tag, expected, actual):
                raise CommsError(
                    f"S7 write verification failed for '{spec.tag.id}': "
                    f"wrote {expected!r}, read {actual!r}",
                    device=self.config.id,
                    tag=spec.tag.id,
                )

    def _ensure_connected(self) -> None:
        if self._client is None or not self._connected:
            raise CommsError(
                "Siemens S7 is not connected",
                device=self.config.id,
            )
