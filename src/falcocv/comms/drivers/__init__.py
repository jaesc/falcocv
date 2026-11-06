"""Communications driver registry."""

from __future__ import annotations

from ..base import CommsConfigError, DeviceConfig, DeviceDriver
from ..dependencies import DependencyStatus
from .schema import DriverSchema


def _driver_classes():
    # Imported lazily so future optional drivers do not become hard imports.
    from .adlink import AdlinkDaskDriver
    from .beckhoff import BeckhoffAdsDriver
    from .dummy import DummyDriver
    from .modbus_tcp import ModbusTcpDriver
    from .omron import OmronFastDriver
    from .opcua import OpcUaDriver
    from .siemens import SiemensS7Driver

    return {
        AdlinkDaskDriver.driver_name: AdlinkDaskDriver,
        BeckhoffAdsDriver.driver_name: BeckhoffAdsDriver,
        DummyDriver.driver_name: DummyDriver,
        ModbusTcpDriver.driver_name: ModbusTcpDriver,
        OmronFastDriver.driver_name: OmronFastDriver,
        OpcUaDriver.driver_name: OpcUaDriver,
        SiemensS7Driver.driver_name: SiemensS7Driver,
    }


def available_drivers() -> tuple[str, ...]:
    return tuple(sorted(_driver_classes()))


def get_driver_schema(driver_name: str) -> DriverSchema:
    driver_class = _driver_classes().get(driver_name)
    if driver_class is None:
        raise CommsConfigError(f"unknown communications driver '{driver_name}'")
    schema = getattr(driver_class, "config_schema", None)
    if isinstance(schema, DriverSchema):
        return schema
    return DriverSchema(name=driver_name, label=driver_name)


def create_driver(config: DeviceConfig) -> DeviceDriver:
    driver_class = _driver_classes().get(config.driver)
    if driver_class is None:
        available = ", ".join(available_drivers()) or "none"
        raise CommsConfigError(
            f"unknown communications driver '{config.driver}' for device "
            f"'{config.id}'; available: {available}"
        )
    return driver_class(config)


def driver_dependencies(
    driver_name: str, config: dict | None = None
) -> tuple[DependencyStatus, ...]:
    """Return non-invasive dependency status for a configured driver."""

    driver_class = _driver_classes().get(driver_name)
    if driver_class is None:
        return ()
    checker = getattr(driver_class, "check_dependencies", None)
    if not callable(checker):
        return ()
    return tuple(checker(dict(config or {})))


def driver_probe(driver_name: str, config: dict | None = None) -> str:
    """Run a short connection/card probe for a configured driver."""

    driver_class = _driver_classes().get(driver_name)
    if driver_class is None:
        raise CommsConfigError(
            f"unknown communications driver '{driver_name}'"
        )
    probe = getattr(driver_class, "probe", None)
    if not callable(probe):
        raise CommsConfigError(
            f"driver '{driver_name}' does not support a connection probe"
        )
    return str(probe(dict(config or {})))
