"""ADLINK DASK digital and analog I/O driver.

The public driver name is ``adlink``.  It uses the vendor DASK DLL through
``ctypes``; the DLL is resolved from the user native-library directory (or an
explicit path) and is never bundled with FalcoCV.
"""

from __future__ import annotations

import ctypes
import os
import re
import threading
import time
from dataclasses import dataclass
from typing import Any, Sequence

from falcocv.core.native_deps import (
    architecture_suffix,
    native_library_hint,
    resolve_library,
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


@dataclass(frozen=True)
class _ModelCapabilities:
    di_ports: int | None = None
    do_ports: int | None = None
    lines_per_port: int | None = None
    ai_channels: int | None = None
    ao_channels: int | None = None


@dataclass(frozen=True)
class _AdlinkModel:
    key: str
    label: str
    family: str
    card_type: int
    capabilities: _ModelCapabilities | None = None


_MODEL_CAPABILITIES = {
    # Verified on hardware: one 8-bit DI port, one 8-bit relay output port.
    "USB_7250": _ModelCapabilities(
        di_ports=1,
        do_ports=1,
        lines_per_port=8,
        ai_channels=0,
        ao_channels=0,
    ),
}


def _model_label(key: str) -> str:
    prefix, _, rest = key.partition("_")
    return f"{prefix}-{rest.replace('_', '-')}"


# The names and numeric ids mirror ADLINK's DASK card type enum.  USB and PCI
# ids live in the same numeric space here because they are passed to different
# DLL entry points.
_ADLINK_MODELS = (
    ("USB_1902", "USB", 1),
    ("USB_1903", "USB", 2),
    ("USB_1902A", "USB", 2),
    ("USB_1901", "USB", 3),
    ("USB_1902L", "USB", 3),
    ("USB_2401", "USB", 4),
    ("USB_7250", "USB", 5),
    ("USB_7230", "USB", 6),
    ("USB_2405", "USB", 7),
    ("USB_1210", "USB", 8),
    ("USB_2401A3", "USB", 9),
    ("USB_101", "USB", 10),
    ("PCI_6208V", "PCI", 1),
    ("PCI_6208A", "PCI", 2),
    ("PCI_6308V", "PCI", 3),
    ("PCI_6308A", "PCI", 4),
    ("PCI_7200", "PCI", 5),
    ("PCI_7230", "PCI", 6),
    ("PCI_7233", "PCI", 7),
    ("PCI_7234", "PCI", 8),
    ("PCI_7248", "PCI", 9),
    ("PCI_7249", "PCI", 10),
    ("PCI_7250", "PCI", 11),
    ("PCI_7252", "PCI", 12),
    ("PCI_7296", "PCI", 13),
    ("PCI_7300A_RevA", "PCI", 14),
    ("PCI_7300A_RevB", "PCI", 15),
    ("PCI_7432", "PCI", 16),
    ("PCI_7433", "PCI", 17),
    ("PCI_7434", "PCI", 18),
    ("PCI_8554", "PCI", 19),
    ("PCI_9111DG", "PCI", 20),
    ("PCI_9111HR", "PCI", 21),
    ("PCI_9112", "PCI", 22),
    ("PCI_9113", "PCI", 23),
    ("PCI_9114DG", "PCI", 24),
    ("PCI_9114HG", "PCI", 25),
    ("PCI_9118DG", "PCI", 26),
    ("PCI_9118HG", "PCI", 27),
    ("PCI_9118HR", "PCI", 28),
    ("PCI_9810", "PCI", 29),
    ("PCI_9812", "PCI", 30),
    ("PCI_7396", "PCI", 31),
    ("PCI_9116", "PCI", 32),
    ("PCI_7256", "PCI", 33),
    ("PCI_7258", "PCI", 34),
    ("PCI_7260", "PCI", 35),
    ("PCI_7452", "PCI", 36),
    ("PCI_7442", "PCI", 37),
    ("PCI_7443", "PCI", 38),
    ("PCI_7444", "PCI", 39),
    ("PCI_9221", "PCI", 40),
    ("PCI_9524", "PCI", 41),
    ("PCI_6202", "PCI", 42),
    ("PCI_9222", "PCI", 43),
    ("PCI_9223", "PCI", 44),
    ("PCI_7433C", "PCI", 45),
    ("PCI_7434C", "PCI", 46),
    ("PCI_922A", "PCI", 47),
    ("PCI_7350", "PCI", 48),
    ("PCI_7360", "PCI", 49),
    ("PCI_7300A_RevC", "PCI", 50),
    ("PCIe_9524", "PCI", 51),
    ("PCIe_9101", "PCI", 52),
    ("PCIe_9121", "PCI", 53),
    ("PCIe_9141", "PCI", 54),
    ("PCIe_9103", "PCI", 55),
)

ADLINK_MODELS = tuple(
    _AdlinkModel(
        key=key,
        label=_model_label(key),
        family=family,
        card_type=card_type,
        capabilities=_MODEL_CAPABILITIES.get(key),
    )
    for key, family, card_type in _ADLINK_MODELS
)
_ADLINK_MODEL_BY_KEY = {model.key: model for model in ADLINK_MODELS}
ADLINK_MODEL_CHOICES = tuple((model.label, model.key) for model in ADLINK_MODELS)

_AD_RANGES = (
    ("+/-10 V", 1),
    ("+/-5 V", 2),
    ("+/-2.5 V", 3),
    ("+/-1.25 V", 4),
    ("+/-0.625 V", 5),
    ("+/-0.3125 V", 6),
    ("+/-0.5 V", 7),
    ("+/-0.05 V", 8),
    ("+/-0.005 V", 9),
    ("+/-1 V", 10),
    ("+/-0.1 V", 11),
    ("+/-0.01 V", 12),
    ("+/-0.001 V", 13),
    ("+/-2 V", 23),
    ("+/-0.25 V", 24),
    ("+/-0.2 V", 25),
)


def adlink_dll_names(model_key: str | None = None) -> tuple[str, ...]:
    """Return the expected DASK DLL names for a model or for both families."""

    suffix = architecture_suffix()
    if model_key is None:
        return (f"PCI-Dask{suffix}.dll", f"USB-Dask{suffix}.dll")
    model = _ADLINK_MODEL_BY_KEY.get(model_key)
    if model is None:
        return ()
    base = "USB-Dask" if model.family == "USB" else "PCI-Dask"
    return (f"{base}{suffix}.dll",)


ADLINK_SCHEMA = DriverSchema(
    name="adlink",
    label="ADLINK DASK I/O",
    description="ADLINK DASK digital and analog I/O cards.",
    allow_extra=False,
    fields=(
        ConfigField(
            "card_type",
            "Card type",
            "enum",
            "USB_7250",
            choices=ADLINK_MODEL_CHOICES,
            help="ADLINK DASK card model connected to this device.",
        ),
        ConfigField(
            "card_number",
            "Card number",
            "int",
            0,
            minimum=0,
            maximum=65535,
            help="Zero-based index of the card when several are installed.",
        ),
        ConfigField(
            "dll_path",
            "DASK DLL / folder",
            "str",
            "",
            placeholder="Leave empty to use the native library folder",
            help=(
                "Optional path to PCI-Dask64.dll / USB-Dask64.dll or their "
                "containing folder. The default native folder is "
                "%APPDATA%/FalcoCV/native/adlink."
            ),
        ),
        ConfigField(
            "ai_range",
            "Analog input range",
            "enum",
            1,
            choices=_AD_RANGES,
            help="DASK AD_B_* range used by AI tags.",
        ),
        ConfigField(
            "verify_writes",
            "Verify writes",
            "bool",
            False,
            help="Read back each written digital output and compare it.",
        ),
        ConfigField(
            "write_retries",
            "Write retries",
            "int",
            0,
            minimum=0,
            visible_when=("verify_writes", True),
            help=(
                "Extra write + verify attempts after a verification mismatch. "
                "0 means one write attempt."
            ),
        ),
        ConfigField(
            "verify_delay_ms",
            "Verify delay (ms)",
            "int",
            20,
            minimum=0,
            visible_when=("verify_writes", True),
            help=(
                "Wait after a digital write before reading it back. "
                "ADLINK relay outputs need a short settling time."
            ),
        ),
    ),
)

_DIO_RE = re.compile(r"^(DI|DO)(\d+)\.(\d+)$", re.IGNORECASE)
_ANALOG_RE = re.compile(r"^(AI|AO)(\d+)$", re.IGNORECASE)

_DASK_ERROR_TEXT = {
    -2: "invalid ADLINK card number",
    -3: "too many ADLINK cards registered",
    -4: "ADLINK card is not registered",
    -5: "function not supported by this ADLINK card",
    -13: "could not open the ADLINK driver",
    -22: "invalid or unsupported ADLINK card type",
    -31: "invalid digital port for this ADLINK card",
    -32: "invalid digital line for this ADLINK card",
    -302: (
        "no ADLINK USB module found; install the USB-DASK driver, "
        "check the USB connection and try again"
    ),
}


def _dask_error_message(operation: str, code: int) -> str:
    detail = _DASK_ERROR_TEXT.get(code)
    message = f"ADLINK {operation} failed with code {code}"
    if detail:
        message += f": {detail}"
    return message


@dataclass(frozen=True)
class _AdlinkTag:
    kind: str
    port: int | None = None
    line: int | None = None
    channel: int | None = None


def _tag_error(tag: TagDef, message: str) -> CommsError:
    return CommsError(message, device=tag.device, tag=tag.id)


def _parse_tag(tag: TagDef, model: _AdlinkModel) -> _AdlinkTag:
    address = tag.address.strip().upper()
    caps = model.capabilities
    match = _DIO_RE.match(address)
    if match:
        kind = match.group(1).upper()
        port = int(match.group(2))
        line = int(match.group(3))
        if kind == "DI" and tag.access != "r":
            raise _tag_error(tag, "DI tags must use access 'r'")
        if kind == "DO" and tag.access not in ("r", "rw"):
            raise _tag_error(tag, "DO tags must use access 'r' or 'rw'")
        if tag.type != "bool":
            raise _tag_error(tag, f"{kind} tags must use type 'bool'")
        if caps is not None:
            if kind == "DI":
                if caps.di_ports is not None and port >= caps.di_ports:
                    raise _tag_error(
                        tag,
                        f"{model.label} has no digital input port {port}",
                    )
            elif caps.do_ports is not None and port >= caps.do_ports:
                raise _tag_error(
                    tag,
                    f"{model.label} has no digital output port {port}",
                )
            if (
                caps.lines_per_port is not None
                and line >= caps.lines_per_port
            ):
                raise _tag_error(
                    tag,
                    f"{model.label} has no digital line {line} on port {port}",
                )
        return _AdlinkTag(kind=kind, port=port, line=line)

    match = _ANALOG_RE.match(address)
    if match:
        kind = match.group(1).upper()
        channel = int(match.group(2))
        if kind == "AI":
            if tag.access != "r":
                raise _tag_error(tag, "AI tags must use access 'r'")
            if tag.type != "float":
                raise _tag_error(tag, "AI tags must use type 'float'")
            if caps is not None:
                if caps.ai_channels == 0:
                    raise _tag_error(
                        tag, f"{model.label} has no analog inputs"
                    )
                if (
                    caps.ai_channels is not None
                    and channel >= caps.ai_channels
                ):
                    raise _tag_error(
                        tag,
                        f"{model.label} has no analog input {channel}",
                    )
        else:
            if tag.access != "rw":
                raise _tag_error(tag, "AO tags must use access 'rw'")
            if tag.type != "float":
                raise _tag_error(tag, "AO tags must use type 'float'")
            if caps is not None:
                if caps.ao_channels == 0:
                    raise _tag_error(
                        tag, f"{model.label} has no analog outputs"
                    )
                if (
                    caps.ao_channels is not None
                    and channel >= caps.ao_channels
                ):
                    raise _tag_error(
                        tag,
                        f"{model.label} has no analog output {channel}",
                    )
        return _AdlinkTag(kind=kind, channel=channel)

    raise _tag_error(
        tag,
        f"ADLINK address '{tag.address}' must be DI<port>.<line>, "
        "DO<port>.<line>, AI<channel> or AO<channel>",
    )


class _DaskApi:
    """Thin, typed wrapper around the PCI-Dask or USB-Dask entry points."""

    def __init__(
        self, library, model: _AdlinkModel, card_number: int = 0
    ):
        self._lib = library
        self._model = model
        self._card_number = card_number
        self._card = -1
        self.directory_handle = None
        self._register = self._bind(
            "Register_Card",
            "UD_Register_Card",
            [ctypes.c_ushort, ctypes.c_ushort],
            ctypes.c_short,
        )
        self._release = self._bind(
            "Release_Card",
            "UD_Release_Card",
            [ctypes.c_ushort],
            ctypes.c_short,
        )
        self._di_read_line = self._bind(
            "DI_ReadLine",
            "UD_DI_ReadLine",
            [
                ctypes.c_ushort,
                ctypes.c_ushort,
                ctypes.c_ushort,
                ctypes.POINTER(ctypes.c_ushort),
            ],
            ctypes.c_short,
        )
        self._di_read_port = self._bind(
            "DI_ReadPort",
            "UD_DI_ReadPort",
            [
                ctypes.c_ushort,
                ctypes.c_ushort,
                ctypes.POINTER(ctypes.c_uint32),
            ],
            ctypes.c_short,
        )
        self._do_read_line = self._bind(
            "DO_ReadLine",
            "UD_DO_ReadLine",
            [
                ctypes.c_ushort,
                ctypes.c_ushort,
                ctypes.c_ushort,
                ctypes.POINTER(ctypes.c_ushort),
            ],
            ctypes.c_short,
        )
        self._do_read_port = self._bind(
            "DO_ReadPort",
            "UD_DO_ReadPort",
            [
                ctypes.c_ushort,
                ctypes.c_ushort,
                ctypes.POINTER(ctypes.c_uint32),
            ],
            ctypes.c_short,
        )
        self._do_write_line = self._bind(
            "DO_WriteLine",
            "UD_DO_WriteLine",
            [
                ctypes.c_ushort,
                ctypes.c_ushort,
                ctypes.c_ushort,
                ctypes.c_ushort,
            ],
            ctypes.c_short,
        )
        self._ai_read = self._bind(
            "AI_VReadChannel",
            "UD_AI_VReadChannel",
            [
                ctypes.c_ushort,
                ctypes.c_ushort,
                ctypes.c_ushort,
                ctypes.POINTER(ctypes.c_double),
            ],
            ctypes.c_short,
        )
        self._ao_write = self._bind(
            "AO_VWriteChannel",
            "UD_AO_VWriteChannel",
            [ctypes.c_ushort, ctypes.c_ushort, ctypes.c_double],
            ctypes.c_short,
        )

    def _bind(self, pci_name: str, usb_name: str, argtypes, restype):
        name = usb_name if self._model.family == "USB" else pci_name
        function = getattr(self._lib, name)
        function.argtypes = argtypes
        function.restype = restype
        return function

    def _check(self, result: int, operation: str):
        if result < 0:
            raise CommsError(
                _dask_error_message(operation, result),
                device=self._model.key,
            )

    def register(self) -> int:
        handle = self._register(
            ctypes.c_ushort(self._model.card_type),
            ctypes.c_ushort(self._card_number),
        )
        if handle < 0:
            raise CommsError(
                _dask_error_message("Register_Card", handle)
                + f" (model {self._model.label}, card {self._card_number})"
            )
        self._card = int(handle)
        return self._card

    def release(self):
        if self._card < 0:
            return
        result = self._release(ctypes.c_ushort(self._card))
        self._card = -1
        self._check(result, "Release_Card")

    def read_di_line(self, port: int, line: int) -> bool:
        value = ctypes.c_ushort(0)
        result = self._di_read_line(
            ctypes.c_ushort(self._card),
            ctypes.c_ushort(port),
            ctypes.c_ushort(line),
            ctypes.byref(value),
        )
        self._check(result, "DI_ReadLine")
        return bool(value.value)

    def read_di_port(self, port: int) -> int:
        value = ctypes.c_uint32(0)
        result = self._di_read_port(
            ctypes.c_ushort(self._card),
            ctypes.c_ushort(port),
            ctypes.byref(value),
        )
        self._check(result, "DI_ReadPort")
        return int(value.value)

    def read_do_line(self, port: int, line: int) -> bool:
        value = ctypes.c_ushort(0)
        result = self._do_read_line(
            ctypes.c_ushort(self._card),
            ctypes.c_ushort(port),
            ctypes.c_ushort(line),
            ctypes.byref(value),
        )
        self._check(result, "DO_ReadLine")
        return bool(value.value)

    def read_do_port(self, port: int) -> int:
        value = ctypes.c_uint32(0)
        result = self._do_read_port(
            ctypes.c_ushort(self._card),
            ctypes.c_ushort(port),
            ctypes.byref(value),
        )
        self._check(result, "DO_ReadPort")
        return int(value.value)

    def write_do_line(self, port: int, line: int, value: bool):
        result = self._do_write_line(
            ctypes.c_ushort(self._card),
            ctypes.c_ushort(port),
            ctypes.c_ushort(line),
            ctypes.c_ushort(1 if value else 0),
        )
        self._check(result, "DO_WriteLine")

    def read_ai(self, channel: int, ai_range: int) -> float:
        value = ctypes.c_double(0.0)
        result = self._ai_read(
            ctypes.c_ushort(self._card),
            ctypes.c_ushort(channel),
            ctypes.c_ushort(ai_range),
            ctypes.byref(value),
        )
        self._check(result, "AI_VReadChannel")
        return float(value.value)

    def write_ao(self, channel: int, value: float):
        result = self._ao_write(
            ctypes.c_ushort(self._card),
            ctypes.c_ushort(channel),
            ctypes.c_double(float(value)),
        )
        self._check(result, "AO_VWriteChannel")


def _load_dask_api(
    model: _AdlinkModel,
    configured_path: str,
    card_number: int = 0,
) -> _DaskApi:
    if os.name != "nt":
        raise CommsError(
            "ADLINK DASK driver is only available on Windows"
        )
    names = adlink_dll_names(model.key)
    resolved = resolve_library(names, driver="adlink", configured=configured_path)
    directory_handle = None
    try:
        if resolved.path is not None:
            directory_handle = os.add_dll_directory(
                str(resolved.path.parent)
            )
            library = ctypes.WinDLL(str(resolved.path))
        else:
            library = ctypes.WinDLL(names[0])
        api = _DaskApi(library, model, card_number=card_number)
    except Exception as exc:
        if directory_handle is not None:
            directory_handle.close()
        source = (
            f"from {resolved.path}"
            if resolved.path is not None
            else "from the system search path"
        )
        raise CommsError(
            f"ADLINK DASK library could not be loaded for {model.label} "
            f"{source}: {exc}. "
            + native_library_hint("adlink", names),
            cause=exc,
        ) from exc
    api.directory_handle = directory_handle
    return api


class AdlinkDaskDriver(DeviceDriver):
    driver_name = "adlink"
    config_schema = ADLINK_SCHEMA

    @classmethod
    def check_dependencies(cls, config: dict) -> tuple[DependencyStatus, ...]:
        model_key = str(config.get("card_type", "USB_7250"))
        model = _ADLINK_MODEL_BY_KEY.get(model_key)
        if os.name != "nt":
            return (
                DependencyStatus(
                    "adlink",
                    "ADLINK DASK runtime",
                    "missing",
                    "ADLINK DASK is only available on Windows x64.",
                    help_key="ADLINK driver",
                ),
            )
        if model is None:
            return (
                DependencyStatus(
                    "adlink",
                    "ADLINK DASK runtime",
                    "missing",
                    f"Unknown card type '{model_key}'.",
                    help_key="ADLINK driver",
                ),
            )
        names = adlink_dll_names(model.key)
        resolved = resolve_library(
            names,
            driver="adlink",
            configured=str(config.get("dll_path", "") or ""),
        )
        if resolved.found:
            return (
                DependencyStatus(
                    "adlink",
                    "ADLINK DASK runtime",
                    "ok",
                    "DASK runtime detected.",
                    help_key="ADLINK driver",
                    test_kind="adlink_card",
                ),
            )
        package = "USB-DASK" if model.family == "USB" else "PCI-DASK"
        return (
            DependencyStatus(
                "adlink",
                "ADLINK DASK runtime",
                "missing",
                f"Install the ADLINK {package} 64-bit SDK for {model.label}.",
                help_key="ADLINK driver",
            ),
        )

    @classmethod
    def probe(cls, config: dict) -> str:
        if os.name != "nt":
            raise CommsError("ADLINK DASK is only available on Windows x64")
        model_key = str(config.get("card_type", "USB_7250"))
        model = _ADLINK_MODEL_BY_KEY.get(model_key)
        if model is None:
            raise CommsError(f"unknown ADLINK card type '{model_key}'")
        card_number = int(config.get("card_number", 0) or 0)
        api = _load_dask_api(
            model,
            str(config.get("dll_path", "") or ""),
            card_number,
        )
        try:
            api.register()
            return f"{model.label} detected (card {card_number})"
        finally:
            try:
                api.release()
            finally:
                if api.directory_handle is not None:
                    api.directory_handle.close()
                    api.directory_handle = None

    def __init__(self, config: DeviceConfig):
        super().__init__(config)
        model_key = str(config.config.get("card_type", "USB_7250"))
        model = _ADLINK_MODEL_BY_KEY.get(model_key)
        if model is None:
            raise CommsError(
                f"unknown ADLINK card type '{model_key}'",
                device=config.id,
            )
        self._model = model
        self._card_number = int(config.config.get("card_number", 0) or 0)
        self._ai_range = int(config.config.get("ai_range", 1) or 1)
        self._dll_path = str(config.config.get("dll_path", "") or "")
        self._verify_writes = bool(config.config.get("verify_writes", False))
        self._write_retries = (
            max(0, int(config.config.get("write_retries", 0) or 0))
            if self._verify_writes
            else 0
        )
        self._verify_delay = (
            max(0, int(config.config.get("verify_delay_ms", 20) or 0)) / 1000.0
            if self._verify_writes
            else 0.0
        )
        self._api: _DaskApi | None = None
        self._connected = False
        self._lock = threading.RLock()
        self._tags: dict[str, _AdlinkTag] = {}
        for tag in config.tags:
            self._tags[tag.id] = _parse_tag(tag, self._model)
        self._ao_values: dict[str, float] = {
            tag.id: float(tag.default)
            for tag in config.tags
            if self._tags[tag.id].kind == "AO"
            and isinstance(tag.default, (int, float))
            and not isinstance(tag.default, bool)
        }

    def _require_api(self, tag: TagDef | None = None) -> _DaskApi:
        if self._api is None or not self._connected:
            raise CommsError(
                "ADLINK card is not open",
                device=self.config.id,
                tag=tag.id if tag is not None else "",
            )
        return self._api

    def start(self) -> None:
        with self._lock:
            if self._api is not None:
                return
            api = _load_dask_api(
                self._model, self._dll_path, self._card_number
            )
            try:
                api.register()
            except Exception:
                try:
                    self._close_api(api)
                except Exception:
                    pass
                raise
            self._api = api
            self._connected = True

    def stop(self) -> None:
        with self._lock:
            api = self._api
            self._api = None
            self._connected = False
            if api is not None:
                self._close_api(api)

    def _close_api(self, api: _DaskApi):
        try:
            api.release()
        finally:
            if api.directory_handle is not None:
                api.directory_handle.close()
                api.directory_handle = None

    def is_connected(self) -> bool:
        return self._connected and self._api is not None

    def _read_one(self, api: _DaskApi, tag: TagDef) -> Any:
        address = self._tags[tag.id]
        if address.kind == "DI":
            return api.read_di_line(address.port, address.line)
        if address.kind == "DO":
            return api.read_do_line(address.port, address.line)
        if address.kind == "AI":
            return api.read_ai(address.channel, self._ai_range)
        if address.kind == "AO":
            return self._ao_values.get(tag.id, 0.0)
        raise _tag_error(tag, "unsupported ADLINK address")

    def read(self, tag: TagDef) -> Any:
        with self._lock:
            return self._read_one(self._require_api(tag), tag)

    def read_many(self, tags: Sequence[TagDef]) -> dict[str, Any]:
        with self._lock:
            api = self._require_api(tags[0] if tags else None)
            result: dict[str, Any] = {}
            port_groups: dict[tuple[str, int], list[TagDef]] = {}
            for tag in tags:
                address = self._tags[tag.id]
                if address.kind in ("DI", "DO"):
                    port_groups.setdefault(
                        (address.kind, address.port), []
                    ).append(tag)
                elif address.kind in ("AI", "AO"):
                    result[tag.id] = self._read_one(api, tag)
                else:
                    raise _tag_error(tag, "unsupported ADLINK address")

            for (kind, port), group in port_groups.items():
                try:
                    bits = (
                        api.read_di_port(port)
                        if kind == "DI"
                        else api.read_do_port(port)
                    )
                except CommsError:
                    for tag in group:
                        result[tag.id] = self._read_one(api, tag)
                else:
                    for tag in group:
                        address = self._tags[tag.id]
                        result[tag.id] = bool(
                            bits & (1 << int(address.line))
                        )
            return result

    def _write_one(self, api: _DaskApi, tag: TagDef, value: Any):
        address = self._tags[tag.id]
        if address.kind == "DO":
            expected = bool(value)
            attempts = self._write_retries + 1 if self._verify_writes else 1
            for attempt in range(attempts):
                api.write_do_line(address.port, address.line, expected)
                if not self._verify_writes:
                    return
                if self._verify_delay:
                    time.sleep(self._verify_delay)
                actual = api.read_do_line(address.port, address.line)
                if values_match(tag, expected, actual):
                    return
                if attempt + 1 >= attempts:
                    raise _tag_error(
                        tag,
                        f"ADLINK write verification failed: expected "
                        f"{expected}, read {actual}",
                    )
            return
        if address.kind == "AO":
            voltage = float(value)
            api.write_ao(address.channel, voltage)
            self._ao_values[tag.id] = voltage
            return
        raise _tag_error(
            tag, f"ADLINK tag '{tag.id}' is not an output"
        )

    def write(self, tag: TagDef, value: Any) -> None:
        self.write_many([(tag, value)])

    def write_many(self, items: Sequence[tuple[TagDef, Any]]) -> None:
        with self._lock:
            api = self._require_api(items[0][0] if items else None)
            for tag, value in items:
                self._write_one(api, tag, coerce_tag_value(tag, value))
