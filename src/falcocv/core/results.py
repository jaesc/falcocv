"""Typed result contract for measurement and inspection tools.

The legacy ``ToolResult``/``ToolResults`` collections remain unchanged.  New
tools can additionally produce ``InspectionResult`` objects so headless runs
and HMI verdicts share a stable schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class InspectionResult:
    """A measured or inspected value with its quality/verdict metadata."""

    value: float | None = None
    unit: str = ""
    tolerance: float | None = None
    verdict: str = ""          # "OK" | "NG" | ""
    score: float | None = None
    uncertainty: float | None = None
    message: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
