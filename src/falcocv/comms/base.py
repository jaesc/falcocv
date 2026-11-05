"""Core data types for the FalcoCV communications subsystem.

This module deliberately has no dependency on Qt, the HMI project or any
particular device driver.  Drivers implement :class:`DeviceDriver`; the
:class:`~falcocv.comms.manager.CommsManager` owns their lifecycle.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

TAG_TYPES = frozenset({"bool", "int", "float", "str"})
TAG_ACCESS = frozenset({"r", "w", "rw"})
TAG_QUALITY = frozenset({"good", "stale", "bad", "disabled"})


class CommsError(RuntimeError):
    """Error raised when a communications operation cannot be completed."""

    def __init__(
        self,
        message: str,
        *,
        device: str = "",
        tag: str = "",
        cause: BaseException | None = None,
    ):
        self.device = device
        self.tag = tag
        self.cause = cause
        self.base_message = message
        detail = message
        if device:
            detail = f"[{device}] {detail}"
        if tag:
            detail = f"{detail} ({tag})"
        super().__init__(detail)


class CommsConfigError(ValueError):
    """Invalid communications configuration in an HMI project."""


@dataclass(frozen=True)
class TagDef:
    """Definition of one logical tag exposed by a device."""

    device: str
    name: str
    address: str
    type: str = "bool"
    access: str = "rw"
    poll_ms: int = 0
    default: Any = None
    scale: float = 1.0
    offset: float = 0.0
    unit: str = ""
    length: int | None = None
    description: str = ""

    @property
    def id(self) -> str:
        return f"{self.device}.{self.name}"


@dataclass(frozen=True)
class TagValue:
    """Last known value and quality of a tag."""

    value: Any
    quality: str = "good"
    timestamp: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class DeviceConfig:
    """One configured communications device."""

    id: str
    driver: str
    enabled: bool = True
    config: dict[str, Any] = field(default_factory=dict)
    tags: tuple[TagDef, ...] = ()


@dataclass(frozen=True)
class CommsConfig:
    """Validated communications section of an HMI project."""

    version: int = 1
    devices: tuple[DeviceConfig, ...] = ()


@dataclass
class DeviceStatus:
    """Runtime status for one device."""

    id: str
    driver: str
    connected: bool = False
    last_error: str = ""
    last_read_ts: float = 0.0
    last_write_ts: float = 0.0
    reconnect_count: int = 0


class DeviceDriver(ABC):
    """Minimal driver contract.

    The manager owns reconnection, status propagation and tag-store updates.
    A driver only needs to perform raw device I/O.
    """

    driver_name: str = ""

    def __init__(self, config: DeviceConfig):
        self.config = config

    @abstractmethod
    def start(self) -> None:
        """Open the connection / initialise the device."""

    @abstractmethod
    def stop(self) -> None:
        """Close the connection / release device resources."""

    @abstractmethod
    def read(self, tag: TagDef) -> Any:
        """Read one tag from the device."""

    @abstractmethod
    def write(self, tag: TagDef, value: Any) -> None:
        """Write one tag to the device."""

    @abstractmethod
    def is_connected(self) -> bool:
        """Return the current connection state."""

    def read_many(self, tags: Sequence[TagDef]) -> dict[str, Any]:
        """Read several tags from this device.

        The default implementation preserves the per-tag contract.  Drivers
        whose protocol exposes grouped data (for example PLC structs) can
        override this to batch network I/O.  Return values keyed by ``tag.id``.
        """

        return {tag.id: self.read(tag) for tag in tags}

    def write_many(self, items: Sequence[tuple[TagDef, Any]]) -> None:
        """Write several tags to this device.

        Drivers may override this to batch writes or perform one coherent
        transaction.  The default implementation writes one tag at a time.
        """

        for tag, value in items:
            self.write(tag, value)


def coerce_tag_value(tag: TagDef, value: Any) -> Any:
    """Validate and normalise a value for ``tag``.

    The function is intentionally strict: a configuration error or a script
    writing the wrong type should fail with a clear message instead of silently
    converting a value on the device boundary.
    """

    if tag.type == "bool":
        if isinstance(value, bool):
            return value
        raise CommsError(
            f"tag '{tag.id}' expects bool, got {type(value).__name__}",
            device=tag.device,
            tag=tag.id,
        )
    if tag.type == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise CommsError(
                f"tag '{tag.id}' expects int, got {type(value).__name__}",
                device=tag.device,
                tag=tag.id,
            )
        return value
    if tag.type == "float":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise CommsError(
                f"tag '{tag.id}' expects float, got {type(value).__name__}",
                device=tag.device,
                tag=tag.id,
            )
        return float(value)
    if tag.type == "str":
        if not isinstance(value, str):
            raise CommsError(
                f"tag '{tag.id}' expects str, got {type(value).__name__}",
                device=tag.device,
                tag=tag.id,
            )
        return value
    raise CommsError(
        f"tag '{tag.id}' has unsupported type '{tag.type}'",
        device=tag.device,
        tag=tag.id,
    )
