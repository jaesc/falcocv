"""Driver configuration metadata used by the HMI Designer.

The schema is UI-agnostic: it describes field names, labels and validation
rules; the Qt layer maps the field kinds to widgets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ConfigField:
    key: str
    label: str
    kind: str = "str"  # str | int | float | bool | enum
    default: Any = None
    required: bool = False
    placeholder: str = ""
    help: str = ""
    choices: tuple[tuple[str, Any], ...] = ()
    minimum: float | None = None
    maximum: float | None = None
    secret: bool = False
    visible_when: tuple[str, Any] | None = None


@dataclass(frozen=True)
class DriverSchema:
    name: str
    label: str
    description: str = ""
    fields: tuple[ConfigField, ...] = ()
    allow_extra: bool = True


def field_keys(schema: DriverSchema) -> set[str]:
    return {item.key for item in schema.fields}
