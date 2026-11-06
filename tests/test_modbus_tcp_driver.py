"""Modbus TCP client driver tests."""

from __future__ import annotations

import struct

import pytest

from falcocv.comms import CommsError, parse_comms_config
from falcocv.comms.drivers import available_drivers, get_driver_schema
from falcocv.comms.drivers import modbus_tcp as modbus_driver


class _Response:
    def __init__(self, *, bits=None, registers=None, error=None):
        self.bits = list(bits or [])
        self.registers = list(registers or [])
        self.exception_code = error
        self._error = error is not None

    def isError(self):
        return self._error


class _FakeClient:
    connected = True

    def __init__(self):
        self.coils = {0: False, 3: True}
        self.discretes = {2: True}
        self.holding = {10: 0, 11: 0, 20: 0, 21: 0}
        self.inputs = {5: 7, 6: 9}
        self.calls = []

    def read_coils(self, address, *, count=1, device_id=1):
        self.calls.append(("read_coils", address, count, device_id))
        return _Response(bits=[self.coils.get(address + i, False) for i in range(count)])

    def read_discrete_inputs(self, address, *, count=1, device_id=1):
        self.calls.append(("read_discrete", address, count, device_id))
        return _Response(
            bits=[self.discretes.get(address + i, False) for i in range(count)]
        )

    def read_holding_registers(self, address, *, count=1, device_id=1):
        self.calls.append(("read_holding", address, count, device_id))
        return _Response(
            registers=[self.holding.get(address + i, 0) for i in range(count)]
        )

    def read_input_registers(self, address, *, count=1, device_id=1):
        self.calls.append(("read_input", address, count, device_id))
        return _Response(
            registers=[self.inputs.get(address + i, 0) for i in range(count)]
        )

    def write_coil(self, address, value, *, device_id=1):
        self.calls.append(("write_coil", address, value, device_id))
        self.coils[address] = bool(value)
        return _Response()

    def write_register(self, address, value, *, device_id=1):
        self.calls.append(("write_register", address, value, device_id))
        self.holding[address] = value & 0xFFFF
        return _Response()

    def write_registers(self, address, values, *, device_id=1):
        self.calls.append(("write_registers", address, list(values), device_id))
        for offset, value in enumerate(values):
            self.holding[address + offset] = value & 0xFFFF
        return _Response()

    def connect(self):
        return True

    def close(self):
        self.connected = False


def _config(tags, **overrides):
    config = {
        "host": "127.0.0.1",
        "port": 15020,
        "unit_id": 1,
        "timeout": 2.0,
        "word_order": "big",
        "verify_writes": False,
        "write_retries": 0,
    }
    config.update(overrides)
    return {
        "version": 1,
        "devices": [
            {
                "id": "mb",
                "driver": "modbus_tcp",
                "config": config,
                "tags": tags,
            }
        ],
    }


def _driver(tags, **overrides):
    return modbus_driver.ModbusTcpDriver(
        parse_comms_config(_config(tags, **overrides)).devices[0]
    )


def test_modbus_schema_and_aliases():
    schema = get_driver_schema("modbus_tcp")
    fields = {field.key: field for field in schema.fields}
    assert "modbus_tcp" in available_drivers()
    assert fields["port"].default == 502
    assert fields["word_order"].default == "big"
    assert fields["write_retries"].visible_when == ("verify_writes", True)


@pytest.mark.parametrize(
    "address, tag_type, access, expected",
    [
        ("coil:0", "bool", "rw", ("COIL", 0, None, "bool", 1)),
        ("discrete:2", "bool", "r", ("DISCRETE", 2, None, "bool", 1)),
        ("holding:10", "int", "rw", ("HOLDING", 10, None, "int16", 1)),
        ("input:5", "int", "r", ("INPUT", 5, None, "int16", 1)),
        ("holding:10.3", "bool", "rw", ("HOLDING", 10, 3, "bool", 1)),
        ("HR:20", "int", "rw", ("HOLDING", 20, None, "int16", 1)),
        ("holding:20:int32", "int", "rw", ("HOLDING", 20, None, "int32", 2)),
        ("holding:20:uint32", "int", "rw", ("HOLDING", 20, None, "uint32", 2)),
        ("holding:20:float32", "float", "rw", ("HOLDING", 20, None, "float32", 2)),
    ],
)
def test_modbus_address_parser(address, tag_type, access, expected):
    tag = modbus_driver.TagDef(
        device="mb",
        name="tag",
        address=address,
        type=tag_type,
        access=access,
    )
    parsed = modbus_driver._parse_tag(tag, "big")
    assert (
        parsed.area,
        parsed.address,
        parsed.bit,
        parsed.data_type,
        parsed.words,
    ) == expected


def test_modbus_reads_all_areas(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(modbus_driver, "_create_client", lambda _config: client)
    driver = _driver(
        [
            {"name": "c", "address": "coil:0", "type": "bool", "access": "rw"},
            {"name": "d", "address": "discrete:2", "type": "bool", "access": "r"},
            {"name": "h", "address": "holding:10", "type": "int", "access": "rw"},
            {"name": "i", "address": "input:5", "type": "int", "access": "r"},
        ]
    )
    driver.start()
    try:
        assert driver.read_many(driver.config.tags) == {
            "mb.c": False,
            "mb.d": True,
            "mb.h": 0,
            "mb.i": 7,
        }
        assert client.calls[0][3] == 1
    finally:
        driver.stop()


def test_modbus_reads_and_writes_32_bit_values(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(modbus_driver, "_create_client", lambda _config: client)
    driver = _driver(
        [
            {"name": "i", "address": "holding:20:int32", "type": "int", "access": "rw"},
            {"name": "f", "address": "holding:30:float32", "type": "float", "access": "rw"},
        ],
        verify_writes=True,
        word_order="little",
    )
    driver.start()
    try:
        driver.write(driver.config.tags[0], -123456)
        driver.write(driver.config.tags[1], 12.5)
        high, low = struct.unpack(">HH", struct.pack(">i", -123456))
        assert client.holding[20] == low
        assert client.holding[21] == high
        assert driver.read(driver.config.tags[0]) == -123456
        assert driver.read(driver.config.tags[1]) == 12.5
    finally:
        driver.stop()


def test_modbus_verify_retries_after_mismatch(monkeypatch):
    client = _FakeClient()

    original_write = client.write_coil
    reject = [True]

    def flaky_write(address, value, *, device_id=1):
        if reject[0]:
            reject[0] = False
            client.calls.append(("rejected", address, value, device_id))
            return _Response()
        return original_write(address, value, device_id=device_id)

    client.write_coil = flaky_write
    monkeypatch.setattr(modbus_driver, "_create_client", lambda _config: client)
    driver = _driver(
        [{"name": "c", "address": "coil:3", "type": "bool", "access": "rw"}],
        verify_writes=True,
        write_retries=1,
    )
    driver.start()
    try:
        driver.write(driver.config.tags[0], False)
        assert client.coils[3] is False
    finally:
        driver.stop()


def test_modbus_rejects_invalid_addresses_and_access():
    with pytest.raises(CommsError, match="must be"):
        _driver(
            [{"name": "bad", "address": "0", "type": "bool", "access": "r"}]
        )
    with pytest.raises(CommsError, match="access 'r'"):
        _driver(
            [{"name": "bad", "address": "input:0", "type": "int", "access": "rw"}]
        )
    with pytest.raises(CommsError, match="bit"):
        _driver(
            [{"name": "bad", "address": "holding:0.16", "type": "bool", "access": "rw"}]
        )


def test_modbus_registers_are_encoded_big_endian_by_default():
    tag = modbus_driver.TagDef(
        device="mb",
        name="float",
        address="holding:0:float32",
        type="float",
        access="rw",
    )
    parsed = modbus_driver._parse_tag(tag, "big")
    assert modbus_driver._encode(parsed, 1.0, "big") == [0x3F80, 0x0000]
    assert modbus_driver._decode(parsed, [0x3F80, 0x0000], "big") == 1.0
    assert modbus_driver._encode(parsed, 1.0, "little") == [0x0000, 0x3F80]
    assert modbus_driver._decode(parsed, [0x0000, 0x3F80], "little") == 1.0
