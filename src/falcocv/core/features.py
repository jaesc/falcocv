"""Measured 3D features shared between tools.

A feature is one detected region (a pin tip, a pad, a hole) in calibrated
world coordinates, corrected by the image fixture when one is present.  ``h``
is its signed height above the reference plane.  This lets downstream tools
compare positions with nominal coordinates without using ROIs as origins.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeaturePoint:
    """One measured region, in calibrated world/fixture coordinates."""

    id: int                  # connected-component label before sorting
    index: int               # position after the configured sort
    x: float                 # calibrated world/fixture X (range image aligned)
    y: float                 # calibrated world/fixture Y (range image aligned)
    z: float                 # calibrated world Z
    h: float                 # signed tip height above the (offset) plane
    area_px: int
    area_mm2: float
    volume_mm3: float
    height_max: float
    height_p95: float
    height_mean: float
    points: int
    touches_border: bool = False
    contour_2d: tuple[tuple[float, float], ...] = ()
