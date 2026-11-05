"""Omron fast driver tests using a fake aphyt client."""

from __future__ import annotations

import copy

import pytest

from falcocv.comms import CommsConfigError, CommsError, parse_comms_config
from falcocv.comms.drivers import available_drivers, get_driver_schema
from falcocv.comms.drivers import omron as omron_driver


class _Item:
    def __init__(self, value):
        self._value = value

    def value(self):
        return self._value

    def from_value(self, value):
        self._value = value


class _FakeClient:
    def __init__(self, values, fail_writes=0):
        self.values = values
        self.read_calls = 0
        self.write_calls = 0
        self.fail_writes = fail_writes
        self.dictionary_calls = []
        self.connected_cip_dispatcher = type(
            "Dispatcher", (), {"is_connected_explicit": True}
        )()

    def connect_explicit(self, host):
        self.host = host

    def register_session(self):
        pass

    def close_explicit(self):
        self.connected_cip_dispatcher.is_connected_explicit = False

    def read_variable(self, name):
        self.read_calls += 1
        return copy.deepcopy(self.values[name])

    def write_variable(self, name, data):
        self.write_calls += 1
        if self.fail_writes:
            self.fail_writes -= 1
            raise OSError("temporary write failure")
        self.values[name] = copy.deepcopy(data)

    def save_current_dictionary(self, *args, **kwargs):
        self.dictionary_calls.append("save")

    def load_dictionary_file(self, *args, **kwargs):
        self.dictionary_calls.append("load")

    def load_dictionary_file_if_present(self, *args, **kwargs):
        self.dictionary_calls.append("load_if_present")


def _config(ttl=1000, verify=True, retries=2):
    return {
        "version": 1,
        "devices": [
            {
                "id": "plc",
                "driver": "omron",
                "config": {
                    "host": "192.168.1.10",
                    "snapshot_ttl_ms": ttl,
                    "verify_writes": verify,
                    "write_retries": retries,
                },
                "tags": [
                    {
                        "name": "trigger",
                        "address": "Vision.Trigger",
                        "type": "bool",
                        "access": "rw",
                        "default": False,
                    },
                    {
                        "name": "count",
                        "address": "Vision.Count",
                        "type": "int",
                        "access": "rw",
                        "default": 0,
                    },
                ],
            }
        ],
    }


def test_omron_schema_has_no_dictionary_file():
    schema = get_driver_schema("omron")
    assert "omron" in available_drivers()
    assert "host" in {field.key for field in schema.fields}
    assert "snapshot_ttl_ms" in {field.key for field in schema.fields}
    assert "dictionary_file" not in {field.key for field in schema.fields}

    config = _config()
    config["devices"][0]["config"]["dictionary_file"] = "legacy.pkl"
    with pytest.raises(CommsConfigError, match="unknown config"):
        parse_comms_config(config)


def test_omron_read_groups_by_struct_and_uses_ttl(monkeypatch):
    client = _FakeClient(
        {"Vision": {"Trigger": _Item(False), "Count": _Item(7)}}
    )
    monkeypatch.setattr(omron_driver, "_create_client", lambda host: client)
    device = parse_comms_config(_config()).devices[0]
    driver = omron_driver.OmronFastDriver(device)

    driver.start()
    try:
        values = driver.read_many([device.tags[0], device.tags[1]])
        assert values == {"plc.trigger": False, "plc.count": 7}
        assert client.read_calls == 1

        values = driver.read_many([device.tags[0], device.tags[1]])
        assert values == {"plc.trigger": False, "plc.count": 7}
        assert client.read_calls == 1
        assert client.dictionary_calls == []
    finally:
        driver.stop()


def test_omron_ttl_zero_reads_every_time(monkeypatch):
    client = _FakeClient(
        {"Vision": {"Trigger": _Item(False), "Count": _Item(7)}}
    )
    monkeypatch.setattr(omron_driver, "_create_client", lambda host: client)
    device = parse_comms_config(_config(ttl=0)).devices[0]
    driver = omron_driver.OmronFastDriver(device)

    driver.start()
    try:
        driver.read_many([device.tags[0]])
        driver.read_many([device.tags[0]])
        assert client.read_calls == 2
    finally:
        driver.stop()


def test_omron_write_groups_and_verifies(monkeypatch):
    client = _FakeClient(
        {"Vision": {"Trigger": _Item(False), "Count": _Item(7)}}
    )
    monkeypatch.setattr(omron_driver, "_create_client", lambda host: client)
    device = parse_comms_config(_config(verify=True)).devices[0]
    driver = omron_driver.OmronFastDriver(device)

    driver.start()
    try:
        driver.write_many(
            [(device.tags[0], True), (device.tags[1], 9)]
        )
        assert client.write_calls == 1
        assert driver.read_many([device.tags[0], device.tags[1]]) == {
            "plc.trigger": True,
            "plc.count": 9,
        }
    finally:
        driver.stop()


def test_omron_retries_a_failed_write(monkeypatch):
    client = _FakeClient(
        {"Vision": {"Trigger": _Item(False), "Count": _Item(7)}},
        fail_writes=1,
    )
    monkeypatch.setattr(omron_driver, "_create_client", lambda host: client)
    device = parse_comms_config(_config(retries=1)).devices[0]
    driver = omron_driver.OmronFastDriver(device)

    driver.start()
    try:
        driver.write(device.tags[1], 11)
        assert client.write_calls == 2
        assert driver.read(device.tags[1]) == 11
    finally:
        driver.stop()


def test_omron_missing_aphyt_is_reported(monkeypatch):
    def fail(host):
        raise CommsError("driver 'omron' requires the aphyt package")

    monkeypatch.setattr(omron_driver, "_create_client", fail)
    device = parse_comms_config(_config()).devices[0]
    driver = omron_driver.OmronFastDriver(device)
    with pytest.raises(CommsError, match="aphyt"):
        driver.start()


def test_omron_rejects_non_struct_address():
    config = _config()
    config["devices"][0]["tags"][0]["address"] = "Trigger"
    with pytest.raises(CommsError, match="Struct.Member"):
        omron_driver.OmronFastDriver(
            parse_comms_config(config).devices[0]
        )
