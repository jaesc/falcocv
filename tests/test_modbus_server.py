"""Loopback tests for the Modbus TCP server runtime."""

from __future__ import annotations

import socket
import time

import pytest

pytest.importorskip("pymodbus")

from falcocv.comms import TagDef, TagValue
from falcocv.comms.servers.config import (
    ModbusTcpServerConfig,
    ServerMapping,
    ServerConfigError,
    parse_servers_config,
)
from falcocv.comms.servers.modbus import ModbusTcpServerRuntime


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _FakeManager:
    def __init__(self):
        self.tags = (
            TagDef("dev", "flag", "coil:0", "bool", "rw", default=False),
            TagDef("dev", "count", "holding:0", "int", "rw", default=0),
            TagDef("dev", "value", "holding:10", "float", "rw", default=1.5),
        )
        self.values = {tag.id: tag.default for tag in self.tags}

    def tag(self, tag_id):
        return TagValue(self.values.get(tag_id), "good", time.time())

    def write(self, tag_id, value):
        self.values[tag_id] = value


def test_modbus_server_maps_tags_to_coils_and_registers():
    from pymodbus.client import ModbusTcpClient

    manager = _FakeManager()
    port = _free_port()
    config = ModbusTcpServerConfig(
        id="mb",
        enabled=True,
        host="127.0.0.1",
        port=port,
        unit_id=1,
        word_order="big",
        mappings=(
            ServerMapping("coil:0", "dev.flag", "rw"),
            ServerMapping("holding:0", "dev.count", "rw"),
            ServerMapping("holding:10", "dev.value", "rw"),
        ),
    )
    server = ModbusTcpServerRuntime(config, manager)
    server.start()
    client = ModbusTcpClient("127.0.0.1", port=port, timeout=2)
    try:
        assert client.connect()
        response = client.read_coils(0, count=1, device_id=1)
        assert not response.isError()
        assert response.bits[0] is False

        assert not client.write_coil(0, True, device_id=1).isError()
        assert manager.values["dev.flag"] is True
        assert client.read_coils(0, count=1, device_id=1).bits[0] is True

        assert not client.write_register(0, 42, device_id=1).isError()
        assert manager.values["dev.count"] == 42
        assert client.read_holding_registers(0, count=1, device_id=1).registers == [42]

        assert not client.write_registers(
            10, [0x4049, 0x0FDB], device_id=1
        ).isError()
        assert manager.values["dev.value"] == pytest.approx(3.1415927, rel=1e-5)
        assert client.read_holding_registers(
            10, count=2, device_id=1
        ).registers == [0x4049, 0x0FDB]
    finally:
        client.close()
        server.stop()


def test_modbus_client_driver_reads_and_writes_through_server():
    from falcocv.comms import parse_comms_config
    from falcocv.comms.drivers.modbus_tcp import ModbusTcpDriver

    manager = _FakeManager()
    port = _free_port()
    server = ModbusTcpServerRuntime(
        ModbusTcpServerConfig(
            id="mb",
            enabled=True,
            host="127.0.0.1",
            port=port,
            unit_id=1,
            word_order="big",
            mappings=(
                ServerMapping("coil:0", "dev.flag", "rw"),
                ServerMapping("holding:0", "dev.count", "rw"),
            ),
        ),
        manager,
    )
    server.start()
    driver = ModbusTcpDriver(
        parse_comms_config(
            {
                "version": 1,
                "devices": [
                    {
                        "id": "mb",
                        "driver": "modbus_tcp",
                        "config": {"host": "127.0.0.1", "port": port},
                        "tags": [
                            {
                                "name": "flag",
                                "address": "coil:0",
                                "type": "bool",
                                "access": "rw",
                            },
                            {
                                "name": "count",
                                "address": "holding:0",
                                "type": "int",
                                "access": "rw",
                            },
                        ],
                    }
                ],
            }
        ).devices[0]
    )
    try:
        driver.start()
        assert driver.read_many(driver.config.tags) == {
            "mb.flag": False,
            "mb.count": 0,
        }
        driver.write(driver.config.tags[0], True)
        driver.write(driver.config.tags[1], 99)
        assert manager.values["dev.flag"] is True
        assert manager.values["dev.count"] == 99
        assert driver.read_many(driver.config.tags) == {
            "mb.flag": True,
            "mb.count": 99,
        }
    finally:
        driver.stop()
        server.stop()


def test_modbus_server_rejects_unmapped_addresses():
    from pymodbus.client import ModbusTcpClient

    manager = _FakeManager()
    port = _free_port()
    config = ModbusTcpServerConfig(
        id="mb",
        enabled=True,
        host="127.0.0.1",
        port=port,
        unit_id=1,
        word_order="big",
        mappings=(ServerMapping("coil:0", "dev.flag", "rw"),),
    )
    server = ModbusTcpServerRuntime(config, manager)
    server.start()
    client = ModbusTcpClient("127.0.0.1", port=port, timeout=2)
    try:
        assert client.connect()
        response = client.read_coils(1, count=1, device_id=1)
        assert response.isError()
    finally:
        client.close()
        server.stop()


def test_servers_config_validation():
    assert parse_servers_config({}) == ()
    with pytest.raises(ServerConfigError, match="duplicate server"):
        parse_servers_config(
            {
                "version": 1,
                "servers": [
                    {"id": "a", "type": "modbus_tcp_server"},
                    {"id": "a", "type": "opcua_server"},
                ],
            }
        )
    with pytest.raises(ServerConfigError, match="access"):
        parse_servers_config(
            {
                "version": 1,
                "servers": [
                    {
                        "id": "a",
                        "type": "modbus_tcp_server",
                        "mappings": [
                            {"address": "coil:0", "tag": "x", "access": "w"}
                        ],
                    }
                ],
            }
        )
