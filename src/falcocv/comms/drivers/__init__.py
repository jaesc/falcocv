"""Communications driver registry."""

from __future__ import annotations

from ..base import CommsConfigError, DeviceConfig, DeviceDriver
from .schema import DriverSchema


def _driver_classes():
    # Imported lazily so future optional drivers do not become hard imports.
    from .dummy import DummyDriver
    from .omron import OmronFastDriver

    return {
        DummyDriver.driver_name: DummyDriver,
        OmronFastDriver.driver_name: OmronFastDriver,
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
