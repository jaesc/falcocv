"""FalcoCV communications subsystem.

Public API for the HMI runtime and for standalone tests.  Device drivers live
in :mod:`falcocv.comms.drivers`; this package intentionally has no Qt or HMI
dependency.
"""

from .base import (
    CommsConfig,
    CommsConfigError,
    CommsError,
    DeviceConfig,
    DeviceDriver,
    DeviceStatus,
    TagDef,
    TagValue,
)
from .config import (
    CommsValidationResult,
    default_comms_config,
    parse_comms_config,
    validate_comms_config,
)
from .manager import CommsManager
from .monitor import CommsMonitorSession
from .tags import TagStore

__all__ = [
    "CommsConfig",
    "CommsConfigError",
    "CommsError",
    "CommsManager",
    "CommsMonitorSession",
    "CommsValidationResult",
    "DeviceConfig",
    "DeviceDriver",
    "DeviceStatus",
    "TagDef",
    "TagStore",
    "TagValue",
    "default_comms_config",
    "parse_comms_config",
    "validate_comms_config",
]
