"""Parsing and validation for the ``comms`` section of an HMI project."""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any

from .base import (
    CommsConfig,
    CommsConfigError,
    DeviceConfig,
    TagDef,
    TAG_ACCESS,
    TAG_TYPES,
    coerce_tag_value,
)

_ID_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_-]*$")
SUPPORTED_CONFIG_VERSION = 1


@dataclass(frozen=True)
class CommsValidationResult:
    """Structural errors and non-blocking incompleteness warnings."""

    config: CommsConfig | None
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    @property
    def is_complete(self) -> bool:
        return not self.errors and not self.warnings


def default_comms_config() -> dict[str, Any]:
    """Return a fresh, empty, serialisable comms configuration."""

    return {"version": SUPPORTED_CONFIG_VERSION, "devices": []}


def parse_comms_config(data: Any) -> CommsConfig:
    """Validate a project ``comms`` dictionary and return dataclasses.

    A missing or empty section is valid.  Invalid or incomplete definitions
    raise :class:`CommsConfigError` so deploy/load cannot run a partial
    configuration.
    """

    config, warnings = _parse_comms_config(data, allow_incomplete=False)
    if warnings:
        raise CommsConfigError("; ".join(warnings))
    return config


def validate_comms_config(data: Any) -> CommsValidationResult:
    """Validate a project comms section, allowing incomplete fields."""

    try:
        config, warnings = _parse_comms_config(data, allow_incomplete=True)
    except CommsConfigError as exc:
        return CommsValidationResult(None, (str(exc),), ())
    return CommsValidationResult(config, (), tuple(warnings))


def _parse_comms_config(
    data: Any, *, allow_incomplete: bool
) -> tuple[CommsConfig, list[str]]:
    warnings: list[str] = []
    if isinstance(data, CommsConfig):
        return data, warnings
    if data is None or data == {}:
        return CommsConfig(), warnings

    if not isinstance(data, dict):
        raise CommsConfigError("comms must be a JSON object")

    version = data.get("version", SUPPORTED_CONFIG_VERSION)
    if isinstance(version, bool) or not isinstance(version, int):
        raise CommsConfigError("comms.version must be an integer")
    if version != SUPPORTED_CONFIG_VERSION:
        raise CommsConfigError(
            f"unsupported comms.version {version}; "
            f"expected {SUPPORTED_CONFIG_VERSION}"
        )

    raw_devices = data.get("devices", [])
    if not isinstance(raw_devices, list):
        raise CommsConfigError("comms.devices must be a list")

    # Import lazily: the driver registry imports concrete drivers, while this
    # module is also imported while those drivers are being initialised.
    from .drivers import get_driver_schema, registered_drivers

    known_drivers = set(registered_drivers())
    devices: list[DeviceConfig] = []
    device_ids: set[str] = set()
    tag_ids: set[str] = set()

    for index, raw_device in enumerate(raw_devices):
        if not isinstance(raw_device, dict):
            raise CommsConfigError(f"comms.devices[{index}] must be an object")

        device_id = raw_device.get("id")
        if not isinstance(device_id, str) or not _ID_RE.match(device_id):
            raise CommsConfigError(
                f"comms.devices[{index}].id must match {_ID_RE.pattern}"
            )
        if device_id in device_ids:
            raise CommsConfigError(f"duplicate device id '{device_id}'")
        device_ids.add(device_id)

        driver = raw_device.get("driver")
        if not isinstance(driver, str) or not driver:
            raise CommsConfigError(
                f"comms.devices[{index}].driver must be a non-empty string"
            )
        if driver not in known_drivers:
            raise CommsConfigError(
                f"unknown communications driver '{driver}' for device "
                f"'{device_id}'; available: {', '.join(sorted(known_drivers))}"
            )

        enabled = raw_device.get("enabled", True)
        if not isinstance(enabled, bool):
            raise CommsConfigError(
                f"comms.devices[{index}].enabled must be a boolean"
            )

        driver_config = raw_device.get("config", {})
        if driver_config is None:
            driver_config = {}
        if not isinstance(driver_config, dict):
            raise CommsConfigError(
                f"comms.devices[{index}].config must be an object"
            )
        _validate_driver_config(
            device_id,
            driver,
            driver_config,
            get_driver_schema(driver),
            allow_incomplete=allow_incomplete,
            warnings=warnings,
        )

        raw_tags = raw_device.get("tags", [])
        if not isinstance(raw_tags, list):
            raise CommsConfigError(
                f"comms.devices[{index}].tags must be a list"
            )

        tags: list[TagDef] = []
        for tag_index, raw_tag in enumerate(raw_tags):
            tag = _parse_tag(
                device_id,
                raw_tag,
                index,
                tag_index,
                allow_incomplete=allow_incomplete,
                warnings=warnings,
            )
            if tag.id in tag_ids:
                raise CommsConfigError(f"duplicate tag id '{tag.id}'")
            tag_ids.add(tag.id)
            tags.append(tag)

        devices.append(
            DeviceConfig(
                id=device_id,
                driver=driver,
                enabled=enabled,
                config=dict(driver_config),
                tags=tuple(tags),
            )
        )

    return CommsConfig(version=version, devices=tuple(devices)), warnings


def _validate_driver_config(
    device_id: str,
    driver: str,
    config: dict,
    schema,
    *,
    allow_incomplete: bool,
    warnings: list[str],
) -> None:
    known = {field.key for field in schema.fields}
    for field in schema.fields:
        value = config.get(field.key)
        if field.required and (
            value is None or (isinstance(value, str) and not value.strip())
        ):
            message = (
                f"Device '{device_id}' ({driver}): "
                f"config.{field.key} is required"
            )
            if allow_incomplete:
                warnings.append(message)
                continue
            raise CommsConfigError(message)
        if field.key not in config or value is None:
            continue
        if field.kind == "bool":
            if not isinstance(value, bool):
                raise CommsConfigError(
                    f"comms device '{device_id}' config.{field.key} "
                    "must be a boolean"
                )
        elif field.kind == "int":
            if isinstance(value, bool) or not isinstance(value, int):
                raise CommsConfigError(
                    f"comms device '{device_id}' config.{field.key} "
                    "must be an integer"
                )
        elif field.kind == "float":
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise CommsConfigError(
                    f"comms device '{device_id}' config.{field.key} "
                    "must be numeric"
                )
        elif field.kind == "str":
            if not isinstance(value, str):
                raise CommsConfigError(
                    f"comms device '{device_id}' config.{field.key} "
                    "must be a string"
                )
        elif field.kind == "enum":
            allowed = [choice for _label, choice in field.choices]
            if value not in allowed:
                raise CommsConfigError(
                    f"comms device '{device_id}' config.{field.key} "
                    f"must be one of {allowed}"
                )
        if field.kind in ("int", "float") and field.minimum is not None:
            if value < field.minimum:
                raise CommsConfigError(
                    f"comms device '{device_id}' config.{field.key} "
                    f"must be >= {field.minimum}"
                )
        if field.kind in ("int", "float") and field.maximum is not None:
            if value > field.maximum:
                raise CommsConfigError(
                    f"comms device '{device_id}' config.{field.key} "
                    f"must be <= {field.maximum}"
                )

    if not schema.allow_extra:
        extra = sorted(set(config) - known)
        if extra:
            raise CommsConfigError(
                f"comms device '{device_id}' ({driver}) has unknown "
                f"config keys: {', '.join(extra)}"
            )


def _parse_tag(
    device_id: str,
    raw_tag: Any,
    device_index: int,
    tag_index: int,
    *,
    allow_incomplete: bool,
    warnings: list[str],
) -> TagDef:
    where = f"comms.devices[{device_index}].tags[{tag_index}]"
    if not isinstance(raw_tag, dict):
        raise CommsConfigError(f"{where} must be an object")

    name = raw_tag.get("name")
    if not isinstance(name, str) or not _ID_RE.match(name):
        raise CommsConfigError(f"{where}.name must match {_ID_RE.pattern}")

    address = raw_tag.get("address", name)
    if not isinstance(address, str):
        raise CommsConfigError(f"{where}.address must be a string")
    if not address:
        message = f"Tag '{device_id}.{name}': address is required"
        if allow_incomplete:
            warnings.append(message)
        else:
            raise CommsConfigError(message)

    tag_type = raw_tag.get("type", "bool")
    if not isinstance(tag_type, str) or tag_type not in TAG_TYPES:
        raise CommsConfigError(
            f"{where}.type must be one of {sorted(TAG_TYPES)}"
        )

    access = raw_tag.get("access", "rw")
    if not isinstance(access, str) or access not in TAG_ACCESS:
        raise CommsConfigError(
            f"{where}.access must be one of {sorted(TAG_ACCESS)}"
        )

    poll_ms = raw_tag.get("poll_ms", 0)
    if isinstance(poll_ms, bool) or not isinstance(poll_ms, int) or poll_ms < 0:
        raise CommsConfigError(f"{where}.poll_ms must be a non-negative integer")

    scale = raw_tag.get("scale", 1.0)
    offset = raw_tag.get("offset", 0.0)
    try:
        scale = float(scale)
        offset = float(offset)
    except (TypeError, ValueError) as exc:
        raise CommsConfigError(
            f"{where}.scale/offset must be numeric"
        ) from exc

    length = raw_tag.get("length")
    if length is not None:
        if isinstance(length, bool) or not isinstance(length, int) or length < 0:
            raise CommsConfigError(
                f"{where}.length must be a non-negative integer or null"
            )

    unit = raw_tag.get("unit", "")
    description = raw_tag.get("description", "")
    if not isinstance(unit, str) or not isinstance(description, str):
        raise CommsConfigError(f"{where}.unit/description must be strings")

    tag = TagDef(
        device=device_id,
        name=name,
        address=address,
        type=tag_type,
        access=access,
        poll_ms=poll_ms,
        default=raw_tag.get("default"),
        scale=scale,
        offset=offset,
        unit=unit,
        length=length,
        description=description,
    )

    if tag.default is not None:
        try:
            normalized = coerce_tag_value(tag, tag.default)
        except Exception as exc:
            raise CommsConfigError(
                f"{where}.default is not valid for type '{tag.type}'"
            ) from exc
        tag = replace(tag, default=normalized)

    return tag
