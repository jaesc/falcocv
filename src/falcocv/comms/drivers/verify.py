"""Value comparison helpers shared by PLC drivers."""

from __future__ import annotations

import math
from typing import Any

from ..base import TagDef


def normalise_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("latin-1", errors="replace").replace("\x00", "").strip()
    if isinstance(value, str):
        return value.replace("\x00", "").strip()
    return value


def values_match(tag: TagDef, expected: Any, actual: Any) -> bool:
    """Compare a written value with a read-back value."""

    if tag.type == "float":
        try:
            return math.isclose(
                float(expected),
                float(actual),
                rel_tol=1e-6,
                abs_tol=1e-6,
            )
        except (TypeError, ValueError):
            return False
    if tag.type == "str":
        return normalise_value(expected) == normalise_value(actual)
    if tag.type == "bool":
        return bool(expected) == bool(actual)
    return expected == actual
