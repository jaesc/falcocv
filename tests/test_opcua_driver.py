"""OPC UA client driver tests with an async fake client."""

from __future__ import annotations

import pytest

from falcocv.comms import CommsError, parse_comms_config
from falcocv.comms.drivers import available_drivers, get_driver_schema
from falcocv.comms.drivers import opcua as opcua_driver


class _FakeNode:
    def __init__(self, value):
        self.value = value
        self.reads = 0
        self.writes = []

    async def read_value(self):
        self.reads += 1
        return self.value

    async def write_value(self, value):
        self.writes.append(value)
        self.value = value


class _FakeClient:
    def __init__(self):
        self.values = {"ns=2;s=Count": 7, "ns=2;s=Flag": True}
        self.nodes = {key: _FakeNode(value) for key, value in self.values.items()}
        self.connected = False
        self.disconnected = False

    async def connect(self):
        self.connected = True

    async def disconnect(self):
        self.disconnected = True
        self.connected = False

    def get_node(self, address):
        return self.nodes[address]


def _config(tags, **overrides):
    config = {"endpoint": "opc.tcp://127.0.0.1:4840", "timeout": 2.0}
    config.update(overrides)
    return {
        "version": 1,
        "devices": [
            {
                "id": "ua",
                "driver": "opcua",
                "config": config,
                "tags": tags,
            }
        ],
    }


def _driver(tags, **overrides):
    return opcua_driver.OpcUaDriver(
        parse_comms_config(_config(tags, **overrides)).devices[0]
    )


def test_opcua_schema_and_driver_registered():
    schema = get_driver_schema("opcua")
    fields = {field.key: field for field in schema.fields}
    assert "opcua" in available_drivers()
    assert fields["endpoint"].required
    assert fields["password"].secret
    assert fields["write_retries"].visible_when == ("verify_writes", True)


def test_opcua_reads_writes_and_verifies(monkeypatch):
    client = _FakeClient()
    monkeypatch.setattr(opcua_driver, "_create_client", lambda _config: client)
    driver = _driver(
        [
            {
                "name": "count",
                "address": "ns=2;s=Count",
                "type": "int",
                "access": "rw",
            },
            {
                "name": "flag",
                "address": "ns=2;s=Flag",
                "type": "bool",
                "access": "r",
            },
        ],
        verify_writes=True,
    )
    driver.start()
    try:
        assert client.connected
        assert driver.read_many(driver.config.tags) == {
            "ua.count": 7,
            "ua.flag": True,
        }
        driver.write(driver.config.tags[0], 11)
        assert client.nodes["ns=2;s=Count"].value == 11
        assert driver.read(driver.config.tags[0]) == 11
    finally:
        driver.stop()
    assert client.disconnected


def test_opcua_missing_endpoint_is_reported():
    from falcocv.comms import DeviceConfig, TagDef

    driver = opcua_driver.OpcUaDriver(
        DeviceConfig(
            id="ua",
            driver="opcua",
            config={"endpoint": ""},
            tags=(
                TagDef(
                    device="ua",
                    name="x",
                    address="ns=2;s=X",
                    type="int",
                    access="r",
                ),
            ),
        )
    )
    with pytest.raises(CommsError, match="endpoint"):
        driver.start()


def test_opcua_type_conversion():
    tag = opcua_driver.TagDef(
        device="ua", name="x", address="ns=2;s=X", type="int"
    )
    assert opcua_driver._convert_read(tag, 3.0) == 3
    assert opcua_driver._convert_read(
        opcua_driver.TagDef(
            device="ua", name="y", address="ns=2;s=Y", type="bool"
        ),
        1,
    ) is True
