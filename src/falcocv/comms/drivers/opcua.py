"""OPC UA client driver using asyncua."""

from __future__ import annotations

import asyncio
import threading
from typing import Any, Sequence

from ..async_runtime import AsyncRuntime
from ..base import (
    CommsError,
    DeviceConfig,
    DeviceDriver,
    TagDef,
    coerce_tag_value,
)
from .schema import ConfigField, DriverSchema
from .verify import values_match

OPCUA_SCHEMA = DriverSchema(
    name="opcua",
    label="OPC UA",
    description="OPC UA client using NodeId addresses.",
    allow_extra=False,
    fields=(
        ConfigField(
            "endpoint",
            "Endpoint",
            "str",
            required=True,
            placeholder="opc.tcp://192.168.1.40:4840",
            help="OPC UA endpoint URL.",
        ),
        ConfigField(
            "username",
            "Username",
            "str",
            "",
            help="Optional OPC UA username.",
        ),
        ConfigField(
            "password",
            "Password",
            "str",
            "",
            secret=True,
            help="Optional OPC UA password.",
        ),
        ConfigField(
            "timeout",
            "Timeout (s)",
            "float",
            5.0,
            minimum=0.2,
            maximum=120.0,
        ),
        ConfigField(
            "verify_writes",
            "Verify writes",
            "bool",
            False,
            help="Read each written value back and compare it.",
        ),
        ConfigField(
            "write_retries",
            "Write retries",
            "int",
            0,
            minimum=0,
            visible_when=("verify_writes", True),
            help=(
                "Extra write + verify attempts after a mismatch. "
                "0 means one attempt."
            ),
        ),
    ),
)


def _create_client(config: DeviceConfig):
    try:
        from asyncua import Client
    except Exception as exc:
        raise CommsError(
            "OPC UA driver requires the asyncua package "
            "(install falcocv[protocols])",
            device=config.id,
            cause=exc,
        ) from exc
    client = Client(
        str(config.config.get("endpoint", "")).strip(),
        timeout=float(config.config.get("timeout", 5.0) or 5.0),
    )
    username = str(config.config.get("username", "") or "").strip()
    password = str(config.config.get("password", "") or "")
    if username:
        client.set_user(username)
        client.set_password(password)
    return client


def _convert_read(tag: TagDef, value: Any) -> Any:
    if tag.type == "bool":
        return bool(value)
    if tag.type == "int":
        return int(value)
    if tag.type == "float":
        return float(value)
    if tag.type == "str":
        return str(value)
    return value


class OpcUaDriver(DeviceDriver):
    driver_name = "opcua"
    config_schema = OPCUA_SCHEMA

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self._client = None
        self._connected = False
        self._lock = threading.RLock()
        self._runtime = AsyncRuntime(f"opcua-{config.id}")
        self._timeout = float(config.config.get("timeout", 5.0) or 5.0)
        self._verify_writes = bool(config.config.get("verify_writes", False))
        self._write_retries = (
            max(0, int(config.config.get("write_retries", 0) or 0))
            if self._verify_writes
            else 0
        )

    def start(self) -> None:
        with self._lock:
            if self._client is not None:
                return
            endpoint = str(self.config.config.get("endpoint", "")).strip()
            if not endpoint:
                raise CommsError(
                    "OPC UA driver requires config.endpoint",
                    device=self.config.id,
                )
            client = _create_client(self.config)
            try:
                self._runtime.run(client.connect(), timeout=self._timeout + 5.0)
            except Exception as exc:
                self._runtime.stop()
                raise CommsError(
                    f"could not connect to OPC UA endpoint {endpoint}: {exc}",
                    device=self.config.id,
                    cause=exc,
                ) from exc
            self._client = client
            self._connected = True

    def stop(self) -> None:
        with self._lock:
            client = self._client
            self._client = None
            self._connected = False
            if client is not None:
                try:
                    self._runtime.run(
                        client.disconnect(), timeout=self._timeout + 2.0
                    )
                except Exception:
                    pass
            self._runtime.stop()

    def is_connected(self) -> bool:
        return self._connected and self._client is not None

    def _require_client(self, tag: TagDef | None = None):
        if self._client is None or not self._connected:
            raise CommsError(
                "OPC UA client is not connected",
                device=self.config.id,
                tag=tag.id if tag is not None else "",
            )
        return self._client

    async def _read_many_async(self, client, tags: Sequence[TagDef]):
        nodes = [client.get_node(tag.address) for tag in tags]
        values = await asyncio.gather(
            *(node.read_value() for node in nodes)
        )
        return {
            tag.id: _convert_read(tag, value)
            for tag, value in zip(tags, values)
        }

    async def _write_many_async(
        self, client, items: Sequence[tuple[TagDef, Any]]
    ):
        for tag, value in items:
            node = client.get_node(tag.address)
            await node.write_value(value)

    async def _read_one_async(self, client, tag: TagDef):
        node = client.get_node(tag.address)
        return _convert_read(tag, await node.read_value())

    def read(self, tag: TagDef) -> Any:
        with self._lock:
            client = self._require_client(tag)
            try:
                return self._runtime.run(
                    self._read_one_async(client, tag),
                    timeout=self._timeout,
                )
            except Exception as exc:
                raise CommsError(
                    f"OPC UA read failed: {exc}",
                    device=self.config.id,
                    tag=tag.id,
                    cause=exc,
                ) from exc

    def read_many(self, tags: Sequence[TagDef]) -> dict[str, Any]:
        with self._lock:
            client = self._require_client(tags[0] if tags else None)
            try:
                return self._runtime.run(
                    self._read_many_async(client, tags),
                    timeout=self._timeout,
                )
            except Exception as exc:
                raise CommsError(
                    f"OPC UA read failed: {exc}",
                    device=self.config.id,
                    tag=tags[0].id if tags else "",
                    cause=exc,
                ) from exc

    def write(self, tag: TagDef, value: Any) -> None:
        self.write_many([(tag, value)])

    def write_many(self, items: Sequence[tuple[TagDef, Any]]) -> None:
        with self._lock:
            client = self._require_client(items[0][0] if items else None)
            prepared = [
                (tag, coerce_tag_value(tag, value))
                for tag, value in items
            ]
            attempts = self._write_retries + 1 if self._verify_writes else 1
            for attempt in range(attempts):
                try:
                    self._runtime.run(
                        self._write_many_async(client, prepared),
                        timeout=self._timeout,
                    )
                    if not self._verify_writes:
                        return
                    values = self._runtime.run(
                        self._read_many_async(
                            client, [tag for tag, _value in prepared]
                        ),
                        timeout=self._timeout,
                    )
                except Exception as exc:
                    raise CommsError(
                        f"OPC UA write failed: {exc}",
                        device=self.config.id,
                        tag=prepared[0][0].id if prepared else "",
                        cause=exc,
                    ) from exc
                mismatches = [
                    (tag, value, values[tag.id])
                    for tag, value in prepared
                    if not values_match(tag, value, values[tag.id])
                ]
                if not mismatches:
                    return
                if attempt + 1 >= attempts:
                    tag, expected, actual = mismatches[0]
                    raise CommsError(
                        f"OPC UA write verification failed for '{tag.id}': "
                        f"expected {expected}, read {actual}",
                        device=self.config.id,
                        tag=tag.id,
                    )
