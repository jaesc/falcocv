"""Siemens S7 driver tests using a fake snap7 client."""

from __future__ import annotations

import struct

import pytest

from falcocv.comms import CommsError, parse_comms_config
from falcocv.comms.drivers import available_drivers, get_driver_schema
from falcocv.comms.drivers import siemens as siemens_driver


class _FakeClient:
    def __init__(self, reject_writes=0, write_status=None):
        self.memory = {
            ("DB", 1): bytearray(128),
            ("M", 0): bytearray(32),
            ("I", 0): bytearray(32),
            ("Q", 0): bytearray(32),
        }
        self.reject_writes = reject_writes
        self.write_status = write_status
        self.connected = False
        self.read_calls = []
        self.write_calls = []

    def _area(self, area, db):
        return self.memory[(area, db)]

    def _read(self, area, db, start, size):
        self.read_calls.append((area, db, start, size))
        return bytes(self._area(area, db)[start:start + size])

    def _write(self, area, db, start, data):
        self.write_calls.append((area, db, start, bytes(data)))
        if self.reject_writes:
            self.reject_writes -= 1
            return 0
        if self.write_status is not None:
            return self.write_status
        self._area(area, db)[start:start + len(data)] = data
        return 0

    def connect(self, host, rack, slot, port=102):
        self.connection = (host, rack, slot, port)
        self.connected = True

    def disconnect(self):
        self.connected = False

    def get_connected(self):
        return self.connected

    def db_read(self, db, start, size):
        return self._read("DB", db, start, size)

    def db_write(self, db, start, data):
        return self._write("DB", db, start, data)

    def mb_read(self, start, size):
        return self._read("M", 0, start, size)

    def mb_write(self, start, data):
        return self._write("M", 0, start, data)

    def eb_read(self, start, size):
        return self._read("I", 0, start, size)

    def eb_write(self, start, data):
        return self._write("I", 0, start, data)

    def ab_read(self, start, size):
        return self._read("Q", 0, start, size)

    def ab_write(self, start, data):
        return self._write("Q", 0, start, data)


def _config(verify=False, tags=None, retries=0):
    return {
        "version": 1,
        "devices": [
            {
                "id": "s7",
                "driver": "siemens",
                "config": {
                    "host": "192.168.1.20",
                    "rack": 0,
                    "slot": 1,
                    "verify_writes": verify,
                    "write_retries": retries,
                },
                "tags": tags or [
                    {"name": "trigger", "address": "DB1.DBX0.0", "type": "bool", "access": "rw"},
                    {"name": "count", "address": "DB1.DBW2", "type": "int", "access": "rw"},
                    {"name": "level", "address": "DB1.DBD4", "type": "float", "access": "rw"},
                    {"name": "message", "address": "DB1.DBB10", "type": "str", "access": "rw", "length": 4},
                ],
            }
        ],
    }


def _device(verify=False, tags=None, retries=0):
    return parse_comms_config(_config(verify, tags, retries)).devices[0]


def _client(monkeypatch, **kwargs):
    client = _FakeClient(**kwargs)
    monkeypatch.setattr(siemens_driver, "_create_client", lambda _config: client)
    return client


def _start(monkeypatch, verify=False, tags=None, retries=0, **kwargs):
    client = _client(monkeypatch, **kwargs)
    device = _device(verify, tags, retries)
    driver = siemens_driver.SiemensS7Driver(device)
    driver.start()
    return client, device, driver


def _seed(client):
    data = client.memory[("DB", 1)]
    data[0] = 0b00000001
    struct.pack_into(">h", data, 2, 7)
    struct.pack_into(">f", data, 4, 1.5)
    data[10] = 4
    data[11] = 3
    data[12:15] = b"abc"


def test_siemens_schema_and_addresses():
    schema = get_driver_schema("siemens")
    fields = {field.key: field for field in schema.fields}
    assert "siemens" in available_drivers()
    assert fields["verify_writes"].default is False
    assert fields["write_retries"].default == 0
    assert fields["write_retries"].visible_when == ("verify_writes", True)


def test_siemens_rejects_incompatible_addresses():
    bad_tags = [
        ({"name": "a", "address": "DB1.DBW0", "type": "bool"}, "requires int"),
        ({"name": "b", "address": "M0.0", "type": "int"}, "requires bool"),
        ({"name": "c", "address": "DB1.DBX0.0", "type": "int"}, "requires bool"),
        ({"name": "d", "address": "DB1.DBB0", "type": "str"}, "requires length"),
        ({"name": "e", "address": "DB1.DBW0", "type": "float"}, "requires int"),
        ({"name": "f", "address": "nonsense", "type": "int"}, "invalid S7 address"),
    ]
    for tag, message in bad_tags:
        with pytest.raises(CommsError, match=message):
            siemens_driver.SiemensS7Driver(_device(tags=[tag]))


def test_siemens_read_batches_db_ranges(monkeypatch):
    client, device, driver = _start(monkeypatch)
    _seed(client)
    try:
        values = driver.read_many(device.tags)
        assert values["s7.trigger"] is True
        assert values["s7.count"] == 7
        assert values["s7.level"] == pytest.approx(1.5)
        assert values["s7.message"] == "abc"
        assert len(client.read_calls) == 1
    finally:
        driver.stop()


def test_siemens_supports_memory_bit_and_word(monkeypatch):
    client = _FakeClient()
    client.memory[("M", 0)][0] = 0b00000001
    struct.pack_into(">h", client.memory[("M", 0)], 2, 11)
    monkeypatch.setattr(siemens_driver, "_create_client", lambda _config: client)
    device = _device(
        tags=[
            {"name": "flag", "address": "M0.0", "type": "bool", "access": "rw"},
            {"name": "word", "address": "MW2", "type": "int", "access": "rw"},
        ]
    )
    driver = siemens_driver.SiemensS7Driver(device)
    driver.start()
    try:
        assert driver.read_many(device.tags) == {
            "s7.flag": True,
            "s7.word": 11,
        }
    finally:
        driver.stop()


def test_siemens_write_batches_and_verifies_with_one_read(monkeypatch):
    client, device, driver = _start(monkeypatch, verify=True)
    _seed(client)
    try:
        driver.write_many(
            [
                (device.tags[0], False),
                (device.tags[1], 9),
                (device.tags[2], 2.5),
                (device.tags[3], "ok"),
            ]
        )
        assert len(client.write_calls) == 1
        # One RMW read, one verification read.
        assert len(client.read_calls) == 2
        data = client.memory[("DB", 1)]
        assert data[0] & 1 == 0
        assert struct.unpack_from(">h", data, 2)[0] == 9
        assert struct.unpack_from(">f", data, 4)[0] == pytest.approx(2.5)
        assert data[10] == 4
        assert data[11] == 2
        assert data[12:14] == b"ok"
    finally:
        driver.stop()


def test_siemens_verify_write_failure_does_not_retry(monkeypatch):
    client, device, driver = _start(
        monkeypatch, verify=True, reject_writes=1
    )
    _seed(client)
    try:
        with pytest.raises(CommsError, match="verification"):
            driver.write(device.tags[1], 9)
        assert len(client.write_calls) == 1
        assert len(client.read_calls) == 2
    finally:
        driver.stop()


def test_siemens_write_status_error_does_not_retry(monkeypatch):
    client, device, driver = _start(
        monkeypatch, verify=True, retries=3, write_status=1
    )
    try:
        with pytest.raises(CommsError, match="status"):
            driver.write(device.tags[1], 9)
        assert len(client.write_calls) == 1
    finally:
        driver.stop()


def test_siemens_verification_retries_until_match(monkeypatch):
    client, device, driver = _start(
        monkeypatch, verify=True, retries=1, reject_writes=1
    )
    _seed(client)
    try:
        driver.write(device.tags[1], 9)
        assert len(client.write_calls) == 2
        assert len(client.read_calls) == 4
        assert struct.unpack_from(">h", client.memory[("DB", 1)], 2)[0] == 9
    finally:
        driver.stop()


def test_siemens_verify_false_ignores_retries(monkeypatch):
    client, device, driver = _start(
        monkeypatch, verify=False, retries=3
    )
    _seed(client)
    try:
        driver.write(device.tags[1], 9)
        assert len(client.write_calls) == 1
        assert len(client.read_calls) == 1
    finally:
        driver.stop()


def test_siemens_missing_snap7_is_reported(monkeypatch):
    def fail(_config):
        raise CommsError("driver 'siemens' requires the python-snap7 package")

    monkeypatch.setattr(siemens_driver, "_create_client", fail)
    driver = siemens_driver.SiemensS7Driver(_device())
    with pytest.raises(CommsError, match="python-snap7"):
        driver.start()
