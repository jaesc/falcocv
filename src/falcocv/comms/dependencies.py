"""Optional runtime-dependency status for communications drivers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DependencyState = Literal["ok", "missing", "unknown", "error"]


@dataclass(frozen=True)
class DependencyStatus:
    """One package, native library or connection prerequisite."""

    key: str
    label: str
    state: DependencyState
    detail: str = ""
    help_key: str = ""
    test_kind: str = ""

    @property
    def blocking(self) -> bool:
        return self.state in ("missing", "error")
