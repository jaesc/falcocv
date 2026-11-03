"""Measured 3D features shared between tools.

A feature is one detected region (a pin tip, a pad, a hole) expressed in the
frame of the reference plane, so downstream tools can compare it with nominal
coordinates without worrying about how the part is tilted.
"""

from __future__ import annotations

from dataclasses import dataclass

from falcocv.core.plane_fit import PlaneFrame


@dataclass(frozen=True)
class FeaturePoint:
    """One measured region, in the plane frame plus blob statistics."""

    id: int                  # connected-component label before sorting
    index: int               # position after the configured sort
    u: float                 # in-plane position along the frame X axis
    v: float                 # in-plane position along the frame Y axis
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
    depth: float = 0.0       # tip distance along the measurement direction


@dataclass(frozen=True)
class FeatureSet:
    """Features plus the frame they are expressed in."""

    frame: PlaneFrame
    points: tuple[FeaturePoint, ...] = ()

    def __len__(self) -> int:
        return len(self.points)

    def __iter__(self):
        return iter(self.points)

    def __getitem__(self, item):
        return self.points[item]
