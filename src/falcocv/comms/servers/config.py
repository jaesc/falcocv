"""Configuration for the optional Modbus/OPC UA server runtimes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

_ACCESS = {"r", "rw"}
_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
_MODBUS_ADDRESS_RE = re.compile(
    r"^(COIL|DISCRETE|HOLDING|INPUT):\d+(?:\.\d+)?"
    r"(?::(INT16|UINT16|INT32|UINT32|FLOAT32))?$",
    re.IGNORECASE,
)


class ServerConfigError(ValueError):
    pass


@dataclass(frozen=True)
class ServerMapping:
    address: str
    tag: str
    access: str = "r"


@dataclass(frozen=True)
class ModbusTcpServerConfig:
    id: str
    enabled: bool
    host: str
    port: int
    unit_id: int
    word_order: str
    mappings: tuple[ServerMapping, ...]


@dataclass(frozen=True)
class OpcUaServerConfig:
    id: str
    enabled: bool
    endpoint: str
    namespace: str
    mappings: tuple[ServerMapping, ...]


ServerConfig = ModbusTcpServerConfig | OpcUaServerConfig


def default_servers_config() -> dict[str, Any]:
    return {"version": 1, "servers": []}


def parse_servers_config(data: Any) -> tuple[ServerConfig, ...]:
    if data is None or data == {}:
        return ()
    if not isinstance(data, dict):
        raise ServerConfigError("servers must be a JSON object")
    version = data.get("version", 1)
    if version != 1:
        raise ServerConfigError(
            f"unsupported servers.version {version}; expected 1"
        )
    raw_servers = data.get("servers", [])
    if not isinstance(raw_servers, list):
        raise ServerConfigError("servers.servers must be a list")

    servers: list[ServerConfig] = []
    ids: set[str] = set()
    for index, raw in enumerate(raw_servers):
        where = f"servers.servers[{index}]"
        if not isinstance(raw, dict):
            raise ServerConfigError(f"{where} must be an object")
        server_id = raw.get("id")
        if not isinstance(server_id, str) or not _ID_RE.match(server_id):
            raise ServerConfigError(
                f"{where}.id must match {_ID_RE.pattern}"
            )
        if server_id in ids:
            raise ServerConfigError(f"duplicate server id '{server_id}'")
        ids.add(server_id)
        server_type = raw.get("type")
        enabled = raw.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ServerConfigError(f"{where}.enabled must be a boolean")
        mappings = _parse_mappings(
            raw.get("mappings", []), where
        )
        if server_type == "modbus_tcp_server":
            host = raw.get("host", "0.0.0.0")
            if not isinstance(host, str) or not host.strip():
                raise ServerConfigError(f"{where}.host must be a string")
            port = raw.get("port", 502)
            if isinstance(port, bool) or not isinstance(port, int):
                raise ServerConfigError(f"{where}.port must be an integer")
            if not 1 <= port <= 65535:
                raise ServerConfigError(f"{where}.port must be 1..65535")
            unit_id = raw.get("unit_id", 1)
            if isinstance(unit_id, bool) or not isinstance(unit_id, int):
                raise ServerConfigError(f"{where}.unit_id must be an integer")
            if not 0 <= unit_id <= 255:
                raise ServerConfigError(f"{where}.unit_id must be 0..255")
            word_order = raw.get("word_order", "big")
            if word_order not in ("big", "little"):
                raise ServerConfigError(
                    f"{where}.word_order must be 'big' or 'little'"
                )
            servers.append(
                ModbusTcpServerConfig(
                    id=server_id,
                    enabled=enabled,
                    host=host.strip(),
                    port=port,
                    unit_id=unit_id,
                    word_order=word_order,
                    mappings=mappings,
                )
            )
        elif server_type == "opcua_server":
            endpoint = raw.get(
                "endpoint", "opc.tcp://0.0.0.0:4840/falcocv"
            )
            if not isinstance(endpoint, str) or not endpoint.strip():
                raise ServerConfigError(
                    f"{where}.endpoint must be a non-empty string"
                )
            namespace = raw.get("namespace", "urn:falcocv")
            if not isinstance(namespace, str) or not namespace.strip():
                raise ServerConfigError(
                    f"{where}.namespace must be a non-empty string"
                )
            servers.append(
                OpcUaServerConfig(
                    id=server_id,
                    enabled=enabled,
                    endpoint=endpoint.strip(),
                    namespace=namespace.strip(),
                    mappings=mappings,
                )
            )
        else:
            raise ServerConfigError(
                f"unknown server type '{server_type}' for {where}"
            )
    return tuple(servers)


def _parse_mappings(raw_mappings: Any, where: str) -> tuple[ServerMapping, ...]:
    if not isinstance(raw_mappings, list):
        raise ServerConfigError(f"{where}.mappings must be a list")
    mappings: list[ServerMapping] = []
    addresses: set[str] = set()
    tags: set[str] = set()
    for index, raw in enumerate(raw_mappings):
        mwhere = f"{where}.mappings[{index}]"
        if not isinstance(raw, dict):
            raise ServerConfigError(f"{mwhere} must be an object")
        address = raw.get("address", raw.get("node_id"))
        if not isinstance(address, str) or not address.strip():
            raise ServerConfigError(
                f"{mwhere}.address must be a non-empty string"
            )
        address = address.strip()
        tag = raw.get("tag")
        if not isinstance(tag, str) or not tag.strip():
            raise ServerConfigError(f"{mwhere}.tag must be a non-empty string")
        tag = tag.strip()
        access = raw.get("access", "r")
        if access not in _ACCESS:
            raise ServerConfigError(
                f"{mwhere}.access must be 'r' or 'rw'"
            )
        if address.lower().startswith(
            ("coil:", "discrete:", "holding:", "input:")
        ):
            if not _MODBUS_ADDRESS_RE.match(address):
                raise ServerConfigError(
                    f"{mwhere}.address is not a valid Modbus address"
                )
        if address in addresses:
            raise ServerConfigError(
                f"duplicate server address '{address}' in {where}"
            )
        if tag in tags:
            raise ServerConfigError(
                f"duplicate server tag '{tag}' in {where}"
            )
        addresses.add(address)
        tags.add(tag)
        mappings.append(ServerMapping(address, tag, access))
    return tuple(mappings)
