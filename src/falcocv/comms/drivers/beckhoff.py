"""Beckhoff TwinCAT ADS driver using symbolic variable names."""

from __future__ import annotations

import threading
from typing import Any

from ..base import (
    CommsError,
    DeviceConfig,
    DeviceDriver,
    TagDef,
    coerce_tag_value,
)
from .schema import ConfigField, DriverSchema
from .verify import values_match

BECKHOFF_SCHEMA = DriverSchema(
    name="beckhoff",
    label="Beckhoff ADS",
    description="Beckhoff TwinCAT ADS using symbolic variable names.",
    allow_extra=False,
    fields=(
        ConfigField(
            "ams_net_id", "AMS Net ID", "str", required=True,
            placeholder="192.168.1.10.1.1",
            help="AMS Net ID of the TwinCAT target.",
        ),
        ConfigField(
            "ams_port", "AMS port", "int", 851, minimum=1, maximum=65535,
            help="ADS port; 851 is the TwinCAT 3 PLC runtime.",
        ),
        ConfigField(
            "ip_address", "IP address (optional)", "str", "",
            help=(
                "Only needed for Linux route creation or routed/NAT networks. "
                "Windows uses the TwinCAT ADS route."
            ),
        ),
        ConfigField(
            "verify_writes", "Verify writes", "bool", False,
            help=(
                "Read each written batch back once and compare the values. "
                "No write is repeated."
            ),
        ),
    ),
)


def _create_client(config: DeviceConfig):
    try:
        import pyads
    except Exception as exc:
        raise CommsError(
            "driver 'beckhoff' requires the pyads package",
            device=config.id,
            cause=exc,
        ) from exc
    ip_address = str(config.config.get("ip_address", "") or "").strip()
    return pyads.Connection(
        str(config.config.get("ams_net_id", "")).strip(),
        int(config.config.get("ams_port", 851) or 851),
        ip_address or None,
    )


def _status_ok(status: Any) -> bool:
    if status is None:
        return True
    if isinstance(status, str):
        return status.strip().lower() == "no error"
    return status == 0


class BeckhoffAdsDriver(DeviceDriver):
    driver_name = "beckhoff"
    config_schema = BECKHOFF_SCHEMA

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self._client = None
        self._connected = False
        self._lock = threading.RLock()
        self._verify_writes = bool(config.config.get("verify_writes", False))

    def start(self) -> None:
        if self._client is not None:
            return
        ams_net_id = str(self.config.config.get("ams_net_id", "")).strip()
        if not ams_net_id:
            raise CommsError(
                "Beckhoff driver requires config.ams_net_id",
                device=self.config.id,
            )
        client = _create_client(self.config)
        try:
            client.open()
            client.read_state()
        except Exception as exc:
            try:
                client.close()
            except Exception:
                pass
            raise CommsError(
                f"could not reach Beckhoff ADS target: {exc}",
                device=self.config.id,
                cause=exc,
            ) from exc
        self._client = client
        self._connected = True

    def stop(self) -> None:
        client = self._client
        self._client = None
        self._connected = False
        if client is not None:
            try:
                client.close()
            except Exception:
                pass

    def is_connected(self) -> bool:
        client = self._client
        if not self._connected or client is None:
            return False
        try:
            return bool(client.is_open)
        except Exception:
            return True

    def read(self, tag: TagDef) -> Any:
        return self.read_many([tag])[tag.id]

    def read_many(self, tags) -> dict[str, Any]:
        self._ensure_connected()
        tags = list(tags)
        if not tags:
            return {}
        addresses = list(dict.fromkeys(tag.address for tag in tags))

        with self._lock:
            try:
                values = self._read_values(addresses)
            except Exception as exc:
                raise CommsError(
                    f"Beckhoff read failed: {exc}",
                    device=self.config.id,
                    cause=exc,
                ) from exc
            if not isinstance(values, dict):
                raise CommsError(
                    "Beckhoff read did not return a value dictionary",
                    device=self.config.id,
                )

            result: dict[str, Any] = {}
            for tag in tags:
                if tag.address not in values:
                    raise CommsError(
                        f"Beckhoff read did not return '{tag.address}'",
                        device=self.config.id,
                        tag=tag.id,
                    )
                result[tag.id] = values[tag.address]
        return result

    def write(self, tag: TagDef, value: Any) -> None:
        self.write_many([(tag, value)])

    def write_many(self, items) -> None:
        self._ensure_connected()
        entries = []
        for tag, value in items:
            if tag.access == "r":
                raise CommsError(
                    "tag is read-only",
                    device=self.config.id,
                    tag=tag.id,
                )
            entries.append((tag, coerce_tag_value(tag, value)))
        if not entries:
            return

        with self._lock:
            try:
                if hasattr(self._client, "write_list_by_name"):
                    statuses = self._client.write_list_by_name(
                        {tag.address: value for tag, value in entries}
                    )
                else:
                    statuses = {}
                    for tag, value in entries:
                        self._client.write_by_name(tag.address, value)
            except Exception as exc:
                raise CommsError(
                    f"Beckhoff write failed: {exc}",
                    device=self.config.id,
                    cause=exc,
                ) from exc

            if statuses is not None and not isinstance(statuses, dict):
                raise CommsError(
                    "Beckhoff write did not return a status dictionary",
                    device=self.config.id,
                )
            if isinstance(statuses, dict) and statuses:
                for tag, _value in entries:
                    if tag.address not in statuses:
                        raise CommsError(
                            f"Beckhoff write returned no status for "
                            f"'{tag.address}'",
                            device=self.config.id,
                            tag=tag.id,
                        )
                for address, status in statuses.items():
                    if not _status_ok(status):
                        raise CommsError(
                            f"Beckhoff write failed for '{address}': {status}",
                            device=self.config.id,
                        )

            if self._verify_writes:
                self._verify_entries(entries)

    def _read_values(self, addresses) -> dict[str, Any]:
        if hasattr(self._client, "read_list_by_name"):
            return self._client.read_list_by_name(addresses)
        return {
            address: self._client.read_by_name(address)
            for address in addresses
        }

    def _verify_entries(self, entries) -> None:
        addresses = [tag.address for tag, _value in entries]
        try:
            values = self._read_values(addresses)
        except Exception as exc:
            raise CommsError(
                f"Beckhoff write verification read failed: {exc}",
                device=self.config.id,
                cause=exc,
            ) from exc
        if not isinstance(values, dict):
            raise CommsError(
                "Beckhoff verification read did not return a value dictionary",
                device=self.config.id,
            )
        for tag, expected in entries:
            if tag.address not in values:
                raise CommsError(
                    f"Beckhoff verification did not return '{tag.address}'",
                    device=self.config.id,
                    tag=tag.id,
                )
            actual = values[tag.address]
            if not values_match(tag, expected, actual):
                raise CommsError(
                    f"Beckhoff write verification failed for '{tag.id}': "
                    f"wrote {expected!r}, read {actual!r}",
                    device=self.config.id,
                    tag=tag.id,
                )

    def _ensure_connected(self) -> None:
        if self._client is None or not self._connected:
            raise CommsError(
                "Beckhoff ADS is not connected",
                device=self.config.id,
            )
