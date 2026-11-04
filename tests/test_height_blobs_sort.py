"""Matrix ordering for Height Blobs 3D."""

import math

import pytest

from falcocv.core.features import FeaturePoint
from falcocv.tools.height_blobs.logic import HeightBlobs3D


def _feature(feature_id, u, v):
    return FeaturePoint(
        id=feature_id,
        index=0,
        u=u,
        v=v,
        h=0.0,
        area_px=1,
        area_mm2=1.0,
        volume_mm3=0.0,
        height_max=0.0,
        height_p95=0.0,
        height_mean=0.0,
        points=1,
    )


def _grid(angle_deg, rows=4, cols=4, step=2.0, jitter=0.0):
    angle = math.radians(angle_deg)
    cosine, sine = math.cos(angle), math.sin(angle)
    features = []
    for row in range(rows):
        for column in range(cols):
            u = column * step
            v = row * step
            features.append(
                _feature(
                    row * cols + column,
                    cosine * u - sine * v + jitter,
                    sine * u + cosine * v - jitter,
                )
            )
    return features


def _ordered_ids(angle_deg, **kwargs):
    tool = HeightBlobs3D()
    tool.params["sort_mode"] = "grid"
    return [
        feature.id
        for feature in tool._sort(_grid(angle_deg, **kwargs))
    ]


@pytest.mark.parametrize("angle", [-20, -10, -5, 0, 5, 10, 20])
def test_grid_sort_is_stable_under_rotation(angle):
    assert _ordered_ids(angle) == list(range(16))


@pytest.mark.parametrize("angle", [-20, -10, -5, 0, 5, 10, 20])
def test_three_by_two_grid_is_ordered_by_rows(angle):
    assert _ordered_ids(angle, rows=3, cols=2) == list(range(6))


def test_grid_sort_keeps_rows_in_order_with_noise():
    features = _grid(8, jitter=0.08)
    tool = HeightBlobs3D()
    tool.params["sort_mode"] = "grid"
    ids = [feature.id for feature in tool._sort(features)]
    assert ids == list(range(16))


def test_three_by_two_grid_handles_noise_without_tolerance():
    features = _grid(8, rows=3, cols=2, jitter=0.15)
    tool = HeightBlobs3D()
    tool.params["sort_mode"] = "grid"
    ids = [feature.id for feature in tool._sort(features)]
    assert ids == list(range(6))


def test_grid_sort_descending_reverses_the_matrix_order():
    tool = HeightBlobs3D()
    tool.params["sort_mode"] = "grid"
    tool.params["sort_descending"] = True
    ids = [feature.id for feature in tool._sort(_grid(7))]
    assert ids == list(reversed(range(16)))


def test_grid_sort_falls_back_for_too_few_points():
    tool = HeightBlobs3D()
    tool.params["sort_mode"] = "grid"
    ids = [feature.id for feature in tool._sort([
        _feature(2, 2.0, 1.0),
        _feature(1, 1.0, 0.0),
    ])]
    assert ids == [1, 2]
