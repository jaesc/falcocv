"""Modbus TCP server exposing CommsManager tags."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from typing import Any

from ..async_runtime import AsyncRuntime
from ..base import CommsError, TagDef
from ..drivers import modbus_tcp as modbus_driver
from .base import ServerRuntime, ServerRuntimeError
from .config import ModbusTcpServerConfig, ServerMapping


@dataclass(frozen=True)
class _ServerTag:
    mapping: ServerMapping
    tag: TagDef
    parsed: Any


class ModbusTcpServerRuntime(ServerRuntime):
    server_type = "modbus_tcp_server"

    def __init__(self, config: ModbusTcpServerConfig, manager, log=None):
        super().__init__(config, manager, log)
        self._runtime = AsyncRuntime(f"modbus-server-{config.id}")
        self._server = None
        self._exc = None

    def _tag_value(self, tag: TagDef) -> Any:
        current = self.manager.tag(tag.id)
        value = current.value if current is not None else tag.default
        if value is None:
            if tag.type == "bool":
                return False
            if tag.type in ("int", "float"):
                return 0
            return ""
        if tag.type == "bool":
            return bool(value)
        if tag.type == "int":
            return int(value)
        if tag.type == "float":
            return float(value)
        return value

    def _write_tag(self, tag: TagDef, value: Any):
        try:
            self.manager.write(tag.id, value)
        except CommsError as exc:
            self.log(
                f"Modbus server write to '{tag.id}' failed: {exc}", "error"
            )
            return self._exc.DEVICE_FAILURE
        return None

    def _build_mappings(self):
        tags = {tag.id: tag for tag in self.manager.tags}
        serves: list[_ServerTag] = []
        occupied: dict[str, set[int]] = {
            "coil": set(),
            "discrete": set(),
            "holding": set(),
            "input": set(),
        }
        for mapping in self.config.mappings:
            tag = tags.get(mapping.tag)
            if tag is None:
                raise ServerRuntimeError(
                    f"Modbus server '{self.config.id}' maps unknown tag "
                    f"'{mapping.tag}'"
                )
            if mapping.access == "rw" and tag.access != "rw":
                raise ServerRuntimeError(
                    f"Modbus mapping '{mapping.address}' requests rw for "
                    f"read-only tag '{tag.id}'"
                )
            try:
                parsed = modbus_driver._parse_tag(
                    replace(tag, address=mapping.address),
                    self.config.word_order,
                )
            except CommsError as exc:
                raise ServerRuntimeError(
                    f"Modbus server '{self.config.id}': {exc}"
                ) from exc
            if parsed.area in ("HOLDING", "INPUT") and parsed.bit is not None:
                raise ServerRuntimeError(
                    f"Modbus server mapping '{mapping.address}' uses a "
                    "register bit; use coil/discrete for bits"
                )
            area = parsed.area.lower()
            registers = range(parsed.address, parsed.address + parsed.words)
            if area in ("coil", "discrete"):
                if parsed.address in occupied[area]:
                    raise ServerRuntimeError(
                        f"duplicate Modbus {area} address {parsed.address}"
                    )
                occupied[area].add(parsed.address)
            else:
                overlap = occupied[area].intersection(registers)
                if overlap:
                    raise ServerRuntimeError(
                        f"overlapping Modbus {area} register "
                        f"{min(overlap)}"
                    )
                occupied[area].update(registers)
            serves.append(_ServerTag(mapping, tag, parsed))
        return serves, occupied

    def _build_device(self):
        from pymodbus.simulator import DataType, SimData, SimDevice

        serves, occupied = self._build_mappings()
        self._serves = serves
        self._coils = {
            item.parsed.address: item
            for item in serves
            if item.parsed.area == "COIL"
        }
        self._discretes = {
            item.parsed.address: item
            for item in serves
            if item.parsed.area == "DISCRETE"
        }
        self._holding: dict[int, _ServerTag] = {}
        self._input: dict[int, _ServerTag] = {}
        for item in serves:
            if item.parsed.area == "HOLDING":
                for address in range(
                    item.parsed.address,
                    item.parsed.address + item.parsed.words,
                ):
                    self._holding[address] = item
            elif item.parsed.area == "INPUT":
                for address in range(
                    item.parsed.address,
                    item.parsed.address + item.parsed.words,
                ):
                    self._input[address] = item

        coil_count = max(occupied["coil"], default=-1) + 1
        discrete_count = max(occupied["discrete"], default=-1) + 1
        holding_count = max(occupied["holding"], default=-1) + 1
        input_count = max(occupied["input"], default=-1) + 1
        blocks = (
            [SimData(0, count=max(1, coil_count), values=False,
                     datatype=DataType.BITS)],
            [SimData(0, count=max(1, discrete_count), values=False,
                     datatype=DataType.BITS)],
            [SimData(0, count=max(1, holding_count), values=0,
                     datatype=DataType.REGISTERS)],
            [SimData(0, count=max(1, input_count), values=0,
                     datatype=DataType.REGISTERS)],
        )
        return SimDevice(
            id=self.config.unit_id,
            simdata=blocks,
            use_bit_addressing=True,
            action=self._action,
        )

    def _set_bit(self, registers: list[int], address: int, value: bool):
        index = address // 16
        bit = address % 16
        if not 0 <= index < len(registers):
            return
        if value:
            registers[index] |= 1 << bit
        else:
            registers[index] &= ~(1 << bit)

    def _handle_bits(
        self,
        func_code: int,
        start_address: int,
        address: int,
        count: int,
        current_registers: list[int],
        set_values: list[bool] | None,
    ):
        is_discrete = func_code == 2
        mapping = self._discretes if is_discrete else self._coils
        writing = set_values is not None
        if writing and is_discrete:
            return self._exc.ILLEGAL_FUNCTION
        for offset in range(count):
            bit_address = address + offset
            item = mapping.get(bit_address)
            if item is None:
                return self._exc.ILLEGAL_ADDRESS
            if writing:
                if item.mapping.access != "rw":
                    return self._exc.ILLEGAL_ADDRESS
                error = self._write_tag(
                    item.tag, bool(set_values[offset])
                )
                if error:
                    return error
            value = self._tag_value(item.tag)
            self._set_bit(current_registers, bit_address, bool(value))
        return None

    def _handle_registers(
        self,
        func_code: int,
        start_address: int,
        address: int,
        count: int,
        current_registers: list[int],
        set_values: list[int] | None,
    ):
        area = "holding" if func_code in (3, 6, 16) else "input"
        mapping = self._holding if area == "holding" else self._input
        writing = set_values is not None
        if writing and area == "input":
            return self._exc.ILLEGAL_FUNCTION
        requested = set(range(address, address + count))
        covered: list[_ServerTag] = []
        seen: set[int] = set()
        for register in requested:
            item = mapping.get(register)
            if item is None:
                return self._exc.ILLEGAL_ADDRESS
            if id(item) not in seen:
                seen.add(id(item))
                covered.append(item)
        for item in covered:
            expected = set(
                range(
                    item.parsed.address,
                    item.parsed.address + item.parsed.words,
                )
            )
            if not expected.issubset(requested):
                return self._exc.ILLEGAL_ADDRESS
            if writing and item.mapping.access != "rw":
                return self._exc.ILLEGAL_ADDRESS

        for item in covered:
            item_start = item.parsed.address
            item_count = item.parsed.words
            if writing:
                offset = item_start - address
                values = list(set_values[offset:offset + item_count])
                try:
                    value = modbus_driver._decode(
                        item.parsed,
                        values,
                        self.config.word_order,
                    )
                except Exception:
                    return self._exc.ILLEGAL_VALUE
                error = self._write_tag(item.tag, value)
                if error:
                    return error

            value = self._tag_value(item.tag)
            try:
                registers = modbus_driver._encode(
                    item.parsed,
                    value,
                    self.config.word_order,
                )
            except Exception:
                return self._exc.DEVICE_FAILURE
            for index, register in enumerate(registers):
                target = item_start + index - start_address
                if 0 <= target < len(current_registers):
                    current_registers[target] = register & 0xFFFF
        return None

    async def _action(
        self,
        func_code: int,
        start_address: int,
        address: int,
        count: int,
        current_registers: list[int],
        set_values: list[Any] | None,
    ):
        try:
            if func_code in (1, 2, 5, 15):
                return self._handle_bits(
                    func_code,
                    start_address,
                    address,
                    count,
                    current_registers,
                    set_values,
                )
            if func_code in (3, 4, 6, 16):
                return self._handle_registers(
                    func_code,
                    start_address,
                    address,
                    count,
                    current_registers,
                    set_values,
                )
            return self._exc.ILLEGAL_FUNCTION
        except Exception as exc:
            self.log(f"Modbus server action failed: {exc}", "error")
            return self._exc.DEVICE_FAILURE

    def start(self) -> None:
        if self.status.running:
            return
        try:
            from pymodbus.constants import ExcCodes
            from pymodbus.server import ModbusTcpServer
        except Exception as exc:
            raise ServerRuntimeError(
                "Modbus server requires the pymodbus package "
                "(install falcocv[protocols])"
            ) from exc
        self._exc = ExcCodes
        device = self._build_device()

        async def serve():
            self._server = ModbusTcpServer(
                device,
                address=(self.config.host, self.config.port),
            )
            await self._server.serve_forever(background=True)

        try:
            self._runtime.run(serve(), timeout=5.0)
        except Exception as exc:
            self._runtime.stop()
            self.status.last_error = str(exc)
            raise ServerRuntimeError(
                f"could not start Modbus server '{self.config.id}': {exc}"
            ) from exc
        self.status.running = True
        self.status.last_error = ""
        self.log(
            f"Modbus TCP server '{self.config.id}' listening on "
            f"{self.config.host}:{self.config.port}"
        )

    def stop(self) -> None:
        server = self._server
        self._server = None
        self.status.running = False
        if server is not None:
            try:
                self._runtime.run(server.shutdown(), timeout=5.0)
            except Exception as exc:
                self.status.last_error = str(exc)
                self.log(
                    f"Modbus server '{self.config.id}' stop failed: {exc}",
                    "warning",
                )
        self._runtime.stop()
