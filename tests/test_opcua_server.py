"""Loopback tests for the OPC UA server runtime."""

from __future__ import annotations

import asyncio
import socket
import time

import pytest

pytest.importorskip("asyncua")

from falcocv.comms import TagDef, TagValue
from falcocv.comms.servers.config import OpcUaServerConfig, ServerMapping
from falcocv.comms.servers.opcua import OpcUaServerRuntime


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


class _FakeManager:
    def __init__(self):
        self.tags = (
            TagDef("dev", "count", "ns=2;s=Count", "int", "rw", default=7),
            TagDef("dev", "flag", "ns=2;s=Flag", "bool", "r", default=True),
        )
        self.values = {tag.id: tag.default for tag in self.tags}

    def tag(self, tag_id):
        return TagValue(self.values.get(tag_id), "good", time.time())

    def write(self, tag_id, value):
        self.values[tag_id] = value


def test_opcua_server_reads_writes_and_rejects_read_only():
    from asyncua import Client

    manager = _FakeManager()
    port = _free_port()
    endpoint = f"opc.tcp://127.0.0.1:{port}/falcocv"
    config = OpcUaServerConfig(
        id="ua",
        enabled=True,
        endpoint=endpoint,
        namespace="urn:falcocv",
        mappings=(
            ServerMapping("ns=2;s=Count", "dev.count", "rw"),
            ServerMapping("ns=2;s=Flag", "dev.flag", "r"),
        ),
    )
    server = OpcUaServerRuntime(config, manager)
    server.start()

    async def exercise():
        client = Client(endpoint, timeout=3)
        await client.connect()
        try:
            count = client.get_node("ns=2;s=Count")
            assert await count.read_value() == 7
            await count.write_value(11)
            assert manager.values["dev.count"] == 11
            assert await count.read_value() == 11

            flag = client.get_node("ns=2;s=Flag")
            assert await flag.read_value() is True
            with pytest.raises(Exception):
                await flag.write_value(False)
        finally:
            await client.disconnect()

    try:
        asyncio.run(exercise())
    finally:
        server.stop()


def test_opcua_client_driver_reads_and_writes_through_server():
    from falcocv.comms import parse_comms_config
    from falcocv.comms.drivers.opcua import OpcUaDriver

    manager = _FakeManager()
    port = _free_port()
    endpoint = f"opc.tcp://127.0.0.1:{port}/falcocv"
    server = OpcUaServerRuntime(
        OpcUaServerConfig(
            id="ua",
            enabled=True,
            endpoint=endpoint,
            namespace="urn:falcocv",
            mappings=(
                ServerMapping("ns=2;s=Count", "dev.count", "rw"),
                ServerMapping("ns=2;s=Flag", "dev.flag", "r"),
            ),
        ),
        manager,
    )
    server.start()
    driver = OpcUaDriver(
        parse_comms_config(
            {
                "version": 1,
                "devices": [
                    {
                        "id": "ua",
                        "driver": "opcua",
                        "config": {"endpoint": endpoint, "timeout": 3},
                        "tags": [
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
                    }
                ],
            }
        ).devices[0]
    )
    try:
        driver.start()
        assert driver.read_many(driver.config.tags) == {
            "ua.count": 7,
            "ua.flag": True,
        }
        driver.write(driver.config.tags[0], 12)
        assert manager.values["dev.count"] == 12
        assert driver.read(driver.config.tags[0]) == 12
    finally:
        driver.stop()
        server.stop()


def test_opcua_server_rejects_unknown_tag_mapping():
    manager = _FakeManager()
    port = _free_port()
    config = OpcUaServerConfig(
        id="ua",
        enabled=True,
        endpoint=f"opc.tcp://127.0.0.1:{port}/falcocv",
        namespace="urn:falcocv",
        mappings=(ServerMapping("ns=2;s=Missing", "dev.missing", "r"),),
    )
    server = OpcUaServerRuntime(config, manager)
    with pytest.raises(Exception, match="unknown tag"):
        server.start()
