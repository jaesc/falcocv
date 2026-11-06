"""OPC UA server exposing CommsManager tags."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..async_runtime import AsyncRuntime
from ..base import CommsError, TagDef
from .base import ServerRuntime, ServerRuntimeError
from .config import OpcUaServerConfig, ServerMapping


@dataclass(frozen=True)
class _ServerTag:
    mapping: ServerMapping
    tag: TagDef
    node_id: Any
    variant_type: Any


def _variant_type(tag: TagDef):
    from asyncua import ua

    if tag.type == "bool":
        return ua.VariantType.Boolean
    if tag.type == "int":
        return ua.VariantType.Int32
    if tag.type == "float":
        return ua.VariantType.Double
    return ua.VariantType.String


class OpcUaServerRuntime(ServerRuntime):
    server_type = "opcua_server"

    def __init__(self, config: OpcUaServerConfig, manager, log=None):
        super().__init__(config, manager, log)
        self._runtime = AsyncRuntime(f"opcua-server-{config.id}")
        self._server = None
        self._ua = None
        self._tags: dict[str, _ServerTag] = {}
        self._nodes: dict[str, Any] = {}

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

    def _build_tags(self):
        from asyncua import ua

        tags = {tag.id: tag for tag in self.manager.tags}
        result: dict[str, _ServerTag] = {}
        node_ids: set[str] = set()
        for mapping in self.config.mappings:
            tag = tags.get(mapping.tag)
            if tag is None:
                raise ServerRuntimeError(
                    f"OPC UA server '{self.config.id}' maps unknown tag "
                    f"'{mapping.tag}'"
                )
            if mapping.access == "rw" and tag.access != "rw":
                raise ServerRuntimeError(
                    f"OPC UA mapping '{mapping.address}' requests rw for "
                    f"read-only tag '{tag.id}'"
                )
            try:
                node_id = ua.NodeId.from_string(mapping.address)
            except Exception as exc:
                raise ServerRuntimeError(
                    f"OPC UA server mapping '{mapping.address}' is not a "
                    "valid NodeId"
                ) from exc
            key = str(node_id)
            if key in node_ids:
                raise ServerRuntimeError(
                    f"duplicate OPC UA NodeId '{mapping.address}'"
                )
            node_ids.add(key)
            result[mapping.tag] = _ServerTag(
                mapping, tag, node_id, _variant_type(tag)
            )
        return result

    def _read_callback(self, node_id, _attr):
        tag = self._tags.get(str(node_id))
        if tag is None:
            return self._ua.DataValue(
                self._ua.Variant(None),
                StatusCode=self._ua.StatusCode(
                    self._ua.StatusCodes.BadNodeIdUnknown
                ),
            )
        value = self._tag_value(tag.tag)
        return self._ua.DataValue(
            self._ua.Variant(value, tag.variant_type)
        )

    def _write_setter(self, node_data, attr, value):
        node_id = str(node_data.nodeid)
        item = self._tags.get(node_id)
        if item is None:
            return
        try:
            raw = value.Value.Value
        except Exception:
            raw = value.Value
        try:
            self.manager.write(item.tag.id, raw)
        except CommsError as exc:
            self.log(
                f"OPC UA server write to '{item.tag.id}' failed: {exc}",
                "error",
            )
            return
        # Keep the local node value in sync for data-change subscribers.
        # Keep value_callback: subsequent reads still come from the manager.
        node_data.attributes[attr].value = value

    async def _serve(self):
        from asyncua import Server, ua

        self._ua = ua
        server = Server()
        await server.init()
        server.set_endpoint(self.config.endpoint)
        server.set_server_name(f"FalcoCV {self.config.id}")
        index = await server.register_namespace(self.config.namespace)
        objects = server.get_objects_node()
        parent = await objects.add_folder(index, "FalcoCV")

        for item in self._tags.values():
            node = await parent.add_variable(
                item.node_id,
                item.tag.name,
                self._tag_value(item.tag),
                varianttype=item.variant_type,
            )
            self._nodes[item.tag.id] = node
            server.set_attribute_value_callback(
                node.nodeid, self._read_callback
            )
            if item.mapping.access == "rw":
                await node.set_writable()
                server.set_attribute_value_setter(
                    node.nodeid, self._write_setter
                )
        await server.start()
        self._server = server
        self.status.running = True
        self.log(
            f"OPC UA server '{self.config.id}' listening on "
            f"{self.config.endpoint}"
        )

    def start(self) -> None:
        if self.status.running:
            return
        try:
            self._tags = {
                str(item.node_id): item
                for item in self._build_tags().values()
            }
        except ServerRuntimeError:
            raise
        except Exception as exc:
            raise ServerRuntimeError(
                "OPC UA server requires the asyncua package "
                "(install falcocv[protocols])"
            ) from exc
        try:
            self._runtime.run(self._serve(), timeout=10.0)
        except Exception as exc:
            self._runtime.stop()
            self.status.last_error = str(exc)
            raise ServerRuntimeError(
                f"could not start OPC UA server '{self.config.id}': {exc}"
            ) from exc

    def stop(self) -> None:
        server = self._server
        self._server = None
        self.status.running = False
        if server is not None:
            try:
                self._runtime.run(server.stop(), timeout=10.0)
            except Exception as exc:
                self.status.last_error = str(exc)
                self.log(
                    f"OPC UA server '{self.config.id}' stop failed: {exc}",
                    "warning",
                )
        self._runtime.stop()
