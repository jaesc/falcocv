"""3D ROI rotation: the box and the point selection must agree with the fit."""

import numpy as np

from falcocv.core.plane_fit import _point_in_roi_xy
from falcocv.core.roi_3d import (
    extract_points_in_rois,
    roi_2d_to_3d,
    roi_3d_to_2d,
)

WIDTH = 120
HEIGHT = 90


def _grid_points(width=WIDTH, height=HEIGHT, flip_x=True):
    cols, rows = np.meshgrid(np.arange(width), np.arange(height))
    xs = (width - 1 - cols) if flip_x else cols
    return np.stack(
        [xs.ravel(), rows.ravel(), np.zeros(cols.size)],
        axis=1,
    ).astype(np.float32), cols.ravel(), rows.ravel()


def _mask_for(roi, flip_x=True, z_min=-1.0, z_max=1.0):
    points, _, _ = _grid_points(flip_x=flip_x)
    roi_entry = {"roi": list(roi), "z_min": z_min, "z_max": z_max}
    return extract_points_in_rois(
        points,
        [roi_entry],
        image_width=WIDTH,
        flip_x=flip_x,
    )


def test_roi_2d_to_3d_reports_the_world_angle():
    depth = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    _center, _size, _z_min, _z_max, angle = roi_2d_to_3d(
        [10, 12, 30, 20, 25.0], depth
    )
    assert angle == -25.0


def test_roi_3d_to_2d_round_trip_keeps_the_angle():
    depth = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    roi = [10.0, 12.0, 30.0, 20.0, 25.0]
    center, size, _z_min, _z_max, angle = roi_2d_to_3d(roi, depth)
    back = roi_3d_to_2d(
        center, size, image_width=WIDTH, flip_x=True, angle=angle
    )
    assert back[0] == roi[0]
    assert back[1] == roi[1]
    assert back[2] == roi[2]
    assert back[3] == roi[3]
    assert back[4] == roi[4]


def test_zero_angle_selects_the_pixel_rectangle():
    roi = [20.0, 15.0, 40.0, 30.0, 0.0]
    mask = _mask_for(roi)
    _points, cols, rows = _grid_points()
    expected = (
        (cols >= 20.0) & (cols <= 60.0) & (rows >= 15.0) & (rows <= 45.0)
    )
    assert np.array_equal(mask, expected)


def test_ninety_degrees_swaps_the_axes():
    roi = [40.0, 30.0, 40.0, 20.0, 90.0]
    mask = _mask_for(roi)
    _points, cols, rows = _grid_points()
    # After a 90 degree turn the 40x20 rectangle becomes 20 wide and 40 tall.
    expected = (
        (cols >= 50.0) & (cols <= 70.0) & (rows >= 20.0) & (rows <= 60.0)
    )
    assert np.array_equal(mask, expected)


def test_rotated_selection_agrees_with_the_plane_fit():
    roi = [20.0, 15.0, 40.0, 30.0, 25.0]
    mask = _mask_for(roi)
    points, _cols, _rows = _grid_points()

    x, y, w, h, angle = roi
    cx = ((WIDTH - 1) - (x + w * 0.5))
    cy = y + h * 0.5
    expected = _point_in_roi_xy(
        points[:, 0] + 0.0,
        points[:, 1] + 0.0,
        (cx, cy, w, h, -angle),
    )
    assert np.array_equal(mask, expected)


def test_flip_x_keeps_the_pixels_and_mirrors_the_world_center():
    roi = [10.0, 10.0, 20.0, 20.0, 0.0]
    flipped = _mask_for(roi, flip_x=True)
    plain = _mask_for(roi, flip_x=False)
    _points, cols, rows = _grid_points()
    # The ROI is drawn in image coordinates, so mirroring the cloud must not
    # change which pixels it selects: both runs pick the same rectangle.
    expected = (
        (cols >= 10.0) & (cols <= 30.0) & (rows >= 10.0) & (rows <= 30.0)
    )
    assert np.array_equal(plain, expected)
    assert np.array_equal(flipped, expected)

    depth = np.ones((HEIGHT, WIDTH), dtype=np.float32)
    center_flipped = roi_2d_to_3d(roi, depth, flip_x=True)[0]
    center_plain = roi_2d_to_3d(roi, depth, flip_x=False)[0]
    assert center_flipped[0] == (WIDTH - 1) - center_plain[0]


def test_z_band_filters_the_selection():
    points, _cols, _rows = _grid_points()
    points[:, 2] = np.linspace(-1.0, 1.0, len(points))
    mask = extract_points_in_rois(
        points,
        [{"roi": [20.0, 15.0, 40.0, 30.0, 0.0], "z_min": -0.2, "z_max": 0.2}],
        image_width=WIDTH,
    )
    assert mask.any()
    assert np.all(points[mask, 2] >= -0.2)
    assert np.all(points[mask, 2] <= 0.2)
