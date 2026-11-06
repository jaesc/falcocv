"""Modbus TCP client driver.

Address syntax::

    coil:0                 bool, read/write
    discrete:0             bool, read-only
    holding:0              int16
    input:0                int16, read-only
    holding:0.3            bit 3 of a holding register
    holding:100:int32      signed 32-bit value in two registers
    holding:100:uint32     unsigned 32-bit value in two registers
    holding:100:float32    IEEE-754 float in two registers

The default register type is ``int16`` unless the tag type is ``float``.
"""

from __future__ import annotations

import re
import struct
import threading
from dataclasses import dataclass
from typing import Any, Sequence

from ..base import (
    CommsError,
    DeviceConfig,
    DeviceDriver,
    TagDef,
    coerce_tag_value,
)
from .schema import ConfigField, DriverSchema
from .verify import values_match

MODBUS_TCP_SCHEMA = DriverSchema(
    name="modbus_tcp",
    label="Modbus TCP",
    description="Modbus TCP client using coils, inputs and holding registers.",
    allow_extra=False,
    fields=(
        ConfigField(
            "host", "Host", "str", required=True,
            placeholder="192.168.1.30",
            help="IP address or host name of the Modbus TCP server.",
        ),
        ConfigField("port", "TCP port", "int", 502, minimum=1, maximum=65535),
        ConfigField("unit_id", "Unit ID", "int", 1, minimum=0, maximum=255),
        ConfigField(
            "timeout", "Timeout (s)", "float", 3.0, minimum=0.1,
            maximum=120.0,
        ),
        ConfigField(
            "word_order", "32-bit word order", "enum", "big",
            choices=(("Big endian words", "big"), ("Little endian words", "little")),
            help="Word order used by int32/uint32/float32 register values.",
        ),
        ConfigField(
            "verify_writes", "Verify writes", "bool", False,
            help="Read each written value back and compare it.",
        ),
        ConfigField(
            "write_retries", "Write retries", "int", 0, minimum=0,
            visible_when=("verify_writes", True),
            help=(
                "Extra write + verify attempts after a mismatch. "
                "0 means one attempt."
            ),
        ),
    ),
)

_ADDRESS_RE = re.compile(
    r"^(COIL|DISCRETE|HOLDING|INPUT):(\d+)"
    r"(?:\.(\d+))?"
    r"(?::(INT16|UINT16|INT32|UINT32|FLOAT32))?$",
    re.IGNORECASE,
)
_ALIASES = {
    "C": "COIL",
    "D": "DISCRETE",
    "HR": "HOLDING",
    "IR": "INPUT",
}


@dataclass(frozen=True)
class _ModbusTag:
    area: str
    address: int
    bit: int | None
    data_type: str
    words: int


def _tag_error(tag: TagDef, message: str) -> CommsError:
    return CommsError(message, device=tag.device, tag=tag.id)


def _parse_tag(tag: TagDef, word_order: str) -> _ModbusTag:
    text = tag.address.strip().upper()
    if ":" in text:
        prefix, rest = text.split(":", 1)
        prefix = _ALIASES.get(prefix, prefix)
        text = f"{prefix}:{rest}"
    match = _ADDRESS_RE.match(text)
    if not match:
        raise _tag_error(
            tag,
            f"Modbus address '{tag.address}' must be "
            "coil:N, discrete:N, holding:N or input:N "
            "(optional .bit and :int32/:uint32/:float32)",
        )
    area = match.group(1).upper()
    address = int(match.group(2))
    bit_text = match.group(3)
    type_text = match.group(4)
    bit = int(bit_text) if bit_text is not None else None
    if address > 65535:
        raise _tag_error(tag, "Modbus address must be between 0 and 65535")

    if area in ("COIL", "DISCRETE"):
        if bit is not None or type_text is not None:
            raise _tag_error(
                tag, f"{area} addresses do not accept a bit or type suffix"
            )
        if tag.type != "bool":
            raise _tag_error(tag, f"{area} tags must use type 'bool'")
        if area == "DISCRETE" and tag.access != "r":
            raise _tag_error(tag, "discrete input tags must use access 'r'")
        return _ModbusTag(area, address, None, "bool", 1)

    if area == "INPUT" and tag.access != "r":
        raise _tag_error(tag, "input register tags must use access 'r'")
    if bit is not None:
        if not 0 <= bit <= 15:
            raise _tag_error(tag, "register bit must be between 0 and 15")
        if type_text is not None:
            raise _tag_error(tag, "a register bit cannot have a type suffix")
        if tag.type != "bool":
            raise _tag_error(tag, "bit register tags must use type 'bool'")
        return _ModbusTag(area, address, bit, "bool", 1)

    if tag.type == "str":
        raise _tag_error(tag, "Modbus string tags are not supported")
    if tag.type == "bool":
        if type_text is not None and type_text.lower() not in ("uint16", "int16"):
            raise _tag_error(tag, "bool register tags use int16/uint16")
        return _ModbusTag(area, address, None, "bool", 1)

    data_type = (type_text or "").lower()
    if not data_type:
        data_type = "float32" if tag.type == "float" else "int16"
    if data_type not in ("int16", "uint16", "int32", "uint32", "float32"):
        raise _tag_error(tag, f"unsupported Modbus register type '{data_type}'")
    if data_type in ("int16", "uint16"):
        if tag.type != "int":
            raise _tag_error(tag, f"Modbus {data_type} tags must use type 'int'")
        return _ModbusTag(area, address, None, data_type, 1)
    if tag.type != "float" and data_type == "float32":
        raise _tag_error(tag, "Modbus float32 tags must use type 'float'")
    if tag.type != "int" and data_type in ("int32", "uint32"):
        raise _tag_error(tag, f"Modbus {data_type} tags must use type 'int'")
    if word_order not in ("big", "little"):
        raise _tag_error(tag, "word_order must be 'big' or 'little'")
    return _ModbusTag(area, address, None, data_type, 2)


def _check_response(response):
    if response is None:
        raise CommsError("Modbus client returned no response")
    if response.isError():
        code = getattr(response, "exception_code", None)
        raise CommsError(
            f"Modbus exception response (code {code})"
            if code is not None
            else "Modbus exception response"
        )
    return response


def _decode(tag: _ModbusTag, registers: list[int], word_order: str):
    data_type = tag.data_type
    if data_type == "bool":
        return bool(registers[0])
    if data_type in ("int16", "uint16"):
        value = int(registers[0]) & 0xFFFF
        if data_type == "int16" and value & 0x8000:
            value -= 0x10000
        return value
    if len(registers) < 2:
        raise CommsError("Modbus 32-bit value requires two registers")
    low, high = int(registers[0]), int(registers[1])
    if word_order == "little":
        low, high = high, low
    packed = struct.pack(">HH", low, high)
    if data_type == "float32":
        return struct.unpack(">f", packed)[0]
    value = struct.unpack(">i" if data_type == "int32" else ">I", packed)[0]
    return value


def _encode(tag: _ModbusTag, value: Any, word_order: str) -> list[int]:
    data_type = tag.data_type
    if data_type == "bool":
        return [1 if value else 0]
    if data_type in ("int16", "uint16"):
        value = int(value)
        if data_type == "int16":
            return [value & 0xFFFF]
        return [value & 0xFFFF]
    if data_type == "float32":
        packed = struct.pack(">f", float(value))
        high, low = struct.unpack(">HH", packed)
    elif data_type == "int32":
        packed = struct.pack(">i", int(value))
        high, low = struct.unpack(">HH", packed)
    else:
        packed = struct.pack(">I", int(value))
        high, low = struct.unpack(">HH", packed)
    if word_order == "little":
        return [low, high]
    return [high, low]


def _create_client(config: DeviceConfig):
    try:
        from pymodbus.client import ModbusTcpClient
    except Exception as exc:
        raise CommsError(
            "Modbus TCP driver requires the pymodbus package "
            "(install falcocv[protocols])",
            device=config.id,
            cause=exc,
        ) from exc
    return ModbusTcpClient(
        str(config.config.get("host", "")).strip(),
        port=int(config.config.get("port", 502) or 502),
        timeout=float(config.config.get("timeout", 3.0) or 3.0),
        retries=0,
    )


class ModbusTcpDriver(DeviceDriver):
    driver_name = "modbus_tcp"
    config_schema = MODBUS_TCP_SCHEMA

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self._client = None
        self._connected = False
        self._lock = threading.RLock()
        self._unit_id = int(config.config.get("unit_id", 1) or 1)
        self._word_order = str(config.config.get("word_order", "big"))
        self._verify_writes = bool(config.config.get("verify_writes", False))
        self._write_retries = (
            max(0, int(config.config.get("write_retries", 0) or 0))
            if self._verify_writes
            else 0
        )
        self._tags: dict[str, _ModbusTag] = {}
        for tag in config.tags:
            self._tags[tag.id] = _parse_tag(tag, self._word_order)

    def start(self) -> None:
        with self._lock:
            if self._client is not None:
                return
            host = str(self.config.config.get("host", "")).strip()
            if not host:
                raise CommsError(
                    "Modbus TCP driver requires config.host",
                    device=self.config.id,
                )
            client = _create_client(self.config)
            if not client.connect():
                client.close()
                raise CommsError(
                    f"could not connect to Modbus TCP server {host}",
                    device=self.config.id,
                )
            self._client = client
            self._connected = True

    def stop(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._connected = False
            if client is not None:
                try:
                    client.close()
                except Exception:
                    pass

    def is_connected(self) -> bool:
        client = self._client
        if not self._connected or client is None:
            return False
        try:
            return bool(client.connected)
        except Exception:
            return True

    def _require_client(self, tag: TagDef | None = None):
        if self._client is None or not self._connected:
            raise CommsError(
                "Modbus TCP client is not connected",
                device=self.config.id,
                tag=tag.id if tag is not None else "",
            )
        return self._client

    def _read_one(self, client, tag: TagDef):
        address = self._tags[tag.id]
        try:
            if address.area == "COIL":
                response = client.read_coils(
                    address.address, count=1, device_id=self._unit_id
                )
                _check_response(response)
                return bool(response.bits[0])
            if address.area == "DISCRETE":
                response = client.read_discrete_inputs(
                    address.address, count=1, device_id=self._unit_id
                )
                _check_response(response)
                return bool(response.bits[0])
            method = (
                client.read_holding_registers
                if address.area == "HOLDING"
                else client.read_input_registers
            )
            response = method(
                address.address,
                count=address.words,
                device_id=self._unit_id,
            )
            _check_response(response)
            registers = list(response.registers)
            if address.bit is not None:
                return bool(registers[0] & (1 << address.bit))
            return _decode(address, registers, self._word_order)
        except CommsError:
            raise
        except Exception as exc:
            raise CommsError(
                f"Modbus read failed: {exc}",
                device=self.config.id,
                tag=tag.id,
                cause=exc,
            ) from exc

    def read(self, tag: TagDef) -> Any:
        with self._lock:
            return self._read_one(self._require_client(tag), tag)

    def read_many(self, tags: Sequence[TagDef]) -> dict[str, Any]:
        with self._lock:
            client = self._require_client(tags[0] if tags else None)
            return {tag.id: self._read_one(client, tag) for tag in tags}

    def _write_one(self, client, tag: TagDef, value: Any):
        address = self._tags[tag.id]
        attempts = self._write_retries + 1 if self._verify_writes else 1
        for attempt in range(attempts):
            try:
                if address.area == "COIL":
                    response = client.write_coil(
                        address.address,
                        bool(value),
                        device_id=self._unit_id,
                    )
                    expected = bool(value)
                elif address.area == "DISCRETE":
                    raise CommsError(
                        "discrete inputs are read-only", tag=tag.id
                    )
                elif address.bit is not None:
                    read_response = client.read_holding_registers(
                        address.address,
                        count=1,
                        device_id=self._unit_id,
                    )
                    _check_response(read_response)
                    register = int(read_response.registers[0])
                    if value:
                        register |= 1 << address.bit
                    else:
                        register &= ~(1 << address.bit)
                    response = client.write_register(
                        address.address,
                        register,
                        device_id=self._unit_id,
                    )
                    expected = bool(value)
                else:
                    registers = _encode(address, value, self._word_order)
                    response = client.write_registers(
                        address.address,
                        registers,
                        device_id=self._unit_id,
                    )
                    expected = value
                _check_response(response)
            except CommsError:
                raise
            except Exception as exc:
                raise CommsError(
                    f"Modbus write failed: {exc}",
                    device=self.config.id,
                    tag=tag.id,
                    cause=exc,
                ) from exc
            if not self._verify_writes:
                return
            actual = self._read_one(client, tag)
            if values_match(tag, expected, actual):
                return
            if attempt + 1 >= attempts:
                raise CommsError(
                    f"Modbus write verification failed: expected {expected}, "
                    f"read {actual}",
                    device=self.config.id,
                    tag=tag.id,
                )

    def write(self, tag: TagDef, value: Any) -> None:
        self.write_many([(tag, value)])

    def write_many(self, items: Sequence[tuple[TagDef, Any]]) -> None:
        with self._lock:
            client = self._require_client(items[0][0] if items else None)
            for tag, value in items:
                self._write_one(client, tag, coerce_tag_value(tag, value))
