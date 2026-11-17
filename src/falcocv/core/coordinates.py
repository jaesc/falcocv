"""Coordinate conventions shared by measurement tools.

The canonical order is:

    pixel/range value -> calibration -> calibrated world -> fixture correction

Fixtures only correct the image plane (X/Y); range Z is always the calibrated
world Z.  ROIs select search regions and never define coordinate origins.
"""

from __future__ import annotations

import math

import numpy as np


def world_xy_from_pixels(cols, rows, image_width, scale_x, scale_y):
    """Map image columns/rows to calibrated world XY.

    X is mirrored so image-left maps to world +X, matching the point cloud.
    """
    cols = np.asarray(cols, dtype=np.float64)
    rows = np.asarray(rows, dtype=np.float64)
    world_x = (float(image_width) - 1.0 - cols) * float(scale_x)
    world_y = rows * float(scale_y)
    return world_x, world_y


def fixture_transform(image, image_width):
    """Return ``(x0, y0, cos_a, sin_a)`` for the image fixture, or ``None``.

    ``x0/y0`` is the fixture origin in calibrated world XY.  ``a`` is the
    image-space fixture angle (clockwise positive).  An identity fixture
    (x=y=angle=0) means "no fixture" and leaves coordinates untouched.
    """
    if image is None:
        return None
    try:
        fx = float(getattr(image, "x", 0.0))
        fy = float(getattr(image, "y", 0.0))
        angle = float(getattr(image, "angle", 0.0))
        scale_x = float(getattr(image, "scale_x", 1.0))
        scale_y = float(getattr(image, "scale_y", 1.0))
    except (TypeError, ValueError):
        return None
    if (
        abs(fx) < 1e-12
        and abs(fy) < 1e-12
        and abs(angle) < 1e-12
    ):
        return None
    x0 = (float(image_width) - 1.0 - fx) * scale_x
    y0 = fy * scale_y
    radians = math.radians(angle)
    return x0, y0, math.cos(radians), math.sin(radians)


def world_to_fixture_xy(world_x, world_y, transform):
    """Correct calibrated world XY with a fixture transform.

    With no fixture the world coordinates are returned unchanged.  With a
    fixture the result follows the part frame while remaining in millimetres.
    """
    x = np.asarray(world_x, dtype=np.float64)
    y = np.asarray(world_y, dtype=np.float64)
    if transform is None:
        return x, y
    x0, y0, cos_a, sin_a = transform
    dx = x - x0
    dy = y - y0
    return cos_a * dx - sin_a * dy, sin_a * dx + cos_a * dy


def fixture_to_world_xy(fixture_x, fixture_y, transform):
    """Undo :func:`world_to_fixture_xy` (used by camera-space overlays)."""
    x = np.asarray(fixture_x, dtype=np.float64)
    y = np.asarray(fixture_y, dtype=np.float64)
    if transform is None:
        return x, y
    x0, y0, cos_a, sin_a = transform
    dx = cos_a * x + sin_a * y
    dy = -sin_a * x + cos_a * y
    return x0 + dx, y0 + dy
