"""Robust line / circle fitting core (pure NumPy)."""

import math

import numpy as np
import pytest

from falcocv.core.fitting import fit_circle, fit_line


def test_line_fit_recovers_an_exact_line():
    xs = np.linspace(0.0, 100.0, 21)
    points = np.column_stack([xs, 0.5 * xs + 10.0])

    fit = fit_line(points)

    assert fit is not None
    assert fit.count == len(points)
    assert fit.angle_deg == pytest.approx(
        math.degrees(math.atan2(-0.5, 1.0)), abs=1e-6
    )
    assert fit.rms == pytest.approx(0.0, abs=1e-9)
    assert fit.inlier_ratio == pytest.approx(1.0)


def test_line_fit_rejects_outliers():
    xs = np.linspace(0.0, 100.0, 25)
    inliers = np.column_stack([xs, 2.0 * xs + 5.0])
    outliers = np.array([[10.0, -40.0], [50.0, -60.0], [90.0, -20.0]])

    fit = fit_line(np.vstack([inliers, outliers]), robust=True)

    assert fit is not None
    assert fit.count == len(inliers)
    assert not fit.inliers[-3:].any()
    assert fit.angle_deg == pytest.approx(
        math.degrees(math.atan2(-2.0, 1.0)), abs=1e-6
    )
    assert fit.rms < 1e-6


def test_line_fit_without_robust_keeps_every_point():
    xs = np.linspace(0.0, 100.0, 10)
    points = np.column_stack([xs, np.zeros_like(xs)])

    fit = fit_line(points, robust=False)

    assert fit.count == len(points)


def test_line_fit_needs_two_points():
    with pytest.raises(ValueError):
        fit_line([(1.0, 2.0)])


def test_fitting_ignores_non_finite_points():
    points = np.array(
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [np.nan, 5.0], [np.inf, 1.0]]
    )

    line = fit_line(points, robust=False)
    assert line.count == 3

    circle = fit_circle(np.array([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0], [np.nan, np.nan]]))
    assert circle.count == 3


def test_circle_fit_recovers_an_exact_circle():
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)
    points = np.column_stack([120.0 + 35.0 * np.cos(theta), 90.0 + 35.0 * np.sin(theta)])

    fit = fit_circle(points)

    assert fit is not None
    assert fit.center == pytest.approx([120.0, 90.0], abs=1e-6)
    assert fit.radius == pytest.approx(35.0, abs=1e-6)
    assert fit.count == len(points)
    assert fit.rms == pytest.approx(0.0, abs=1e-6)


def test_circle_fit_rejects_outliers():
    theta = np.linspace(0.0, 2.0 * np.pi, 48, endpoint=False)
    inliers = np.column_stack([50.0 + 20.0 * np.cos(theta), 60.0 + 20.0 * np.sin(theta)])
    outliers = np.array([[0.0, 0.0], [100.0, 0.0], [50.0, 120.0]])

    fit = fit_circle(np.vstack([inliers, outliers]), robust=True)

    assert fit is not None
    assert fit.count == len(inliers)
    assert not fit.inliers[-3:].any()
    assert fit.center == pytest.approx([50.0, 60.0], abs=0.05)
    assert fit.radius == pytest.approx(20.0, abs=0.05)


def test_circle_fit_returns_none_for_collinear_points():
    points = np.column_stack([np.arange(10.0), np.arange(10.0)])

    assert fit_circle(points) is None


def test_circle_fit_needs_three_points():
    with pytest.raises(ValueError):
        fit_circle([(0.0, 0.0), (1.0, 1.0)])


def test_line_fit_rejects_calipers_on_a_wrong_edge():
    # 10 calipers, 2 of them on a second edge 35 px away: the least-median
    # model and the MAD cutoff must still recover the true line.
    xs = np.linspace(0.0, 100.0, 10)
    points = np.column_stack([xs, 100.0 + 0.2 * xs])
    points[1, 1] += 35.0
    points[3, 1] += 35.0

    fit = fit_line(points, max_outlier_fraction=0.2)

    assert fit is not None
    assert fit.cap_exceeded is False
    assert fit.count == 8
    assert fit.outliers == 2
    assert fit.angle_deg == pytest.approx(
        math.degrees(math.atan2(-0.2, 1.0)), abs=1e-6
    )
    assert fit.rms < 1e-9


def test_line_fit_flags_more_outliers_than_the_cap_allows():
    xs = np.linspace(0.0, 100.0, 10)
    points = np.column_stack([xs, np.zeros_like(xs)])
    for index in (1, 3, 5, 7):
        points[index, 1] = 30.0

    # 10% allows one discard; four outliers must abort instead of fitting a
    # biased line on the remaining contaminated points.
    fit = fit_line(points, max_outlier_fraction=0.1)

    assert fit is not None
    assert fit.cap_exceeded is True
    assert fit.count == 6
    assert fit.outliers == 4


def test_line_fit_without_an_outlier_cap_keeps_the_cutoff():
    xs = np.linspace(0.0, 100.0, 10)
    points = np.column_stack([xs, np.zeros_like(xs)])
    points[2, 1] = 20.0
    points[7, 1] = -20.0

    fit = fit_line(points, max_outlier_fraction=0.0)

    assert fit.count == 8


def test_circle_fit_rejects_calipers_on_a_wrong_edge():
    theta = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    points = np.column_stack(
        [100.0 + 30.0 * np.cos(theta), 100.0 + 30.0 * np.sin(theta)]
    )
    # Two calipers land at twice the radius.
    points[4] = [100.0 + 60.0 * np.cos(theta[4]), 100.0 + 60.0 * np.sin(theta[4])]
    points[11] = [100.0 + 60.0 * np.cos(theta[11]), 100.0 + 60.0 * np.sin(theta[11])]

    fit = fit_circle(points, max_outlier_fraction=0.125)

    assert fit is not None
    assert fit.cap_exceeded is False
    assert fit.count == 14
    assert fit.outliers == 2
    assert fit.center == pytest.approx([100.0, 100.0], abs=1e-6)
    assert fit.radius == pytest.approx(30.0, abs=1e-6)
