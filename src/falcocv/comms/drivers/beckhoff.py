"""Beckhoff TwinCAT ADS driver using symbolic variable names."""

from __future__ import annotations

import importlib.util
import sys
import threading
from typing import Any

from falcocv.core.native_deps import (
    add_native_dll_directory,
    resolve_tcads_dll,
)

from ..base import (
    CommsError,
    DeviceConfig,
    DeviceDriver,
    TagDef,
    coerce_tag_value,
)
from ..dependencies import DependencyStatus
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
                "Read each written batch back once and compare the values."
            ),
        ),
        ConfigField(
            "write_retries", "Write retries", "int", 0, minimum=0,
            visible_when=("verify_writes", True),
            help=(
                "Extra write + verify attempts after a verification mismatch. "
                "0 means one write attempt."
            ),
        ),
    ),
)


_NATIVE_DLL_HANDLE = None


def _create_client(config: DeviceConfig):
    global _NATIVE_DLL_HANDLE
    if _NATIVE_DLL_HANDLE is None:
        _NATIVE_DLL_HANDLE = add_native_dll_directory("beckhoff")
    try:
        import pyads
    except Exception as exc:
        raise CommsError(
            "driver 'beckhoff' requires pyads and TcAdsDll.dll "
            "(install TwinCAT or put TcAdsDll.dll in the native/beckhoff "
            "folder)",
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

    @classmethod
    def check_dependencies(cls, config: dict) -> tuple[DependencyStatus, ...]:
        pyads_installed = importlib.util.find_spec("pyads") is not None
        tcads = resolve_tcads_dll()
        statuses: list[DependencyStatus] = []
        statuses.append(
            DependencyStatus(
                "pyads",
                "pyads package",
                "ok" if pyads_installed else "missing",
                (
                    "pyads is installed."
                    if pyads_installed
                    else "Install the FalcoCV plc extra (pyads)."
                ),
                help_key="Beckhoff driver",
            )
        )
        if not tcads.found:
            statuses.append(
                DependencyStatus(
                    "tcads",
                    "TwinCAT ADS runtime",
                    "missing",
                    "Install TwinCAT 3 x64 or provide TcAdsDll.dll in the "
                    "native/beckhoff folder.",
                    help_key="Beckhoff driver",
                )
            )
            return tuple(statuses)

        tcads_state = "ok"
        tcads_detail = "TcAdsDll.dll detected."
        if (
            sys.maxsize > 2**32
            and "common32" in str(tcads.path).lower()
        ):
            tcads_state = "error"
            tcads_detail = (
                "A 32-bit TcAdsDll.dll was found; install TwinCAT x64 for "
                "64-bit Python."
            )
        statuses.append(
            DependencyStatus(
                "tcads",
                "TwinCAT ADS runtime",
                tcads_state,
                tcads_detail,
                help_key="Beckhoff driver",
            )
        )
        if pyads_installed and tcads_state == "ok":
            statuses.append(
                DependencyStatus(
                    "ads_route",
                    "ADS route",
                    "unknown",
                    "Configure an ADS route from this PC to the target "
                    "AMS Net ID.",
                    help_key="Beckhoff driver",
                    test_kind="ads_connection",
                )
            )
        return tuple(statuses)

    @classmethod
    def probe(cls, config: dict) -> str:
        device = DeviceConfig(
            id="probe",
            driver=cls.driver_name,
            config=dict(config),
        )
        client = _create_client(device)
        try:
            client.open()
            client.read_state()
            return "ADS connection OK"
        finally:
            try:
                client.close()
            except Exception:
                pass

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        self._client = None
        self._connected = False
        self._lock = threading.RLock()
        self._verify_writes = bool(config.config.get("verify_writes", False))
        self._write_retries = (
            max(0, int(config.config.get("write_retries", 0) or 0))
            if self._verify_writes
            else 0
        )

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
            self._write_batch(entries)
            if not self._verify_writes:
                return

            last_error: CommsError | None = None
            for attempt in range(self._write_retries + 1):
                last_error = self._verify_entries(entries)
                if last_error is None:
                    return
                if attempt >= self._write_retries:
                    break
                self._write_batch(entries)
            raise last_error or CommsError(
                "Beckhoff write verification failed",
                device=self.config.id,
            )

    def _write_batch(self, entries) -> None:
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

    def _read_values(self, addresses) -> dict[str, Any]:
        if hasattr(self._client, "read_list_by_name"):
            return self._client.read_list_by_name(addresses)
        return {
            address: self._client.read_by_name(address)
            for address in addresses
        }

    def _verify_entries(self, entries) -> CommsError | None:
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
                return CommsError(
                    f"Beckhoff verification did not return '{tag.address}'",
                    device=self.config.id,
                    tag=tag.id,
                )
            actual = values[tag.address]
            if not values_match(tag, expected, actual):
                return CommsError(
                    f"Beckhoff write verification failed for '{tag.id}': "
                    f"wrote {expected!r}, read {actual!r}",
                    device=self.config.id,
                    tag=tag.id,
                )
        return None

    def _ensure_connected(self) -> None:
        if self._client is None or not self._connected:
            raise CommsError(
                "Beckhoff ADS is not connected",
                device=self.config.id,
            )
