"""Robust line and circle fitting for the metrology tools.

Pure NumPy: no OpenCV, no Qt.  Coordinates are image pixels (Y grows
downwards); ``angle_deg`` follows the measurement convention ``atan2(-dy, dx)``
so a slope of +1 in image coordinates reports -45 degrees.

Outlier rejection is a two-stage strategy:

* RANSAC keeps the model with the smallest **median** residual (least median
  of squares).  The initial model therefore does not depend on a residual
  threshold that contaminated points would inflate.
* The inlier cutoff is derived from that model's median absolute deviation and
  the number of discarded points is capped by ``max_outlier_fraction``, so a
  couple of calipers on the wrong edge can never pull the fit and the fit can
  never silently ignore most of the array.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: Residual threshold used when the caller passes ``threshold <= 0``.
_AUTO_THRESHOLD_MIN_PX = 1.0
#: Scale factor from the median absolute deviation to a Gaussian sigma.
_MAD_TO_SIGMA = 1.4826
#: Inlier cutoff in sigma units.
_AUTO_CUTOFF_SIGMAS = 3.0
_MAX_RADIUS = 1e7


@dataclass
class LineFit:
    """Result of :func:`fit_line`."""

    point: np.ndarray
    direction: np.ndarray
    angle_deg: float
    rms: float
    inliers: np.ndarray
    count: int
    cap_exceeded: bool = False

    @property
    def inlier_ratio(self) -> float:
        return self.count / max(1, len(self.inliers))

    @property
    def outliers(self) -> int:
        return len(self.inliers) - self.count


@dataclass
class CircleFit:
    """Result of :func:`fit_circle`."""

    center: np.ndarray
    radius: float
    rms: float
    inliers: np.ndarray
    count: int
    cap_exceeded: bool = False

    @property
    def inlier_ratio(self) -> float:
        return self.count / max(1, len(self.inliers))

    @property
    def outliers(self) -> int:
        return len(self.inliers) - self.count


def clean_points(points) -> np.ndarray:
    """Return an ``(N, 2)`` float64 array without NaN/inf rows."""
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    array = array.reshape(-1, 2)
    return array[np.isfinite(array).all(axis=1)]


def _line_model(points: np.ndarray):
    centroid = points.mean(axis=0)
    centered = points - centroid
    if len(points) == 2:
        direction = centered[1]
    else:
        # Total least squares: the smallest singular vector is the normal.
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        direction = vt[0]
    norm = float(np.linalg.norm(direction))
    if norm < 1e-12:
        return None
    direction = direction / norm
    # Canonical orientation: left to right, so the angle stays in (-90, 90].
    if direction[0] < 0 or (abs(direction[0]) < 1e-12 and direction[1] < 0):
        direction = -direction
    return centroid, direction


def _line_residuals(points: np.ndarray, point: np.ndarray, direction: np.ndarray):
    normal = np.array([-direction[1], direction[0]])
    return np.abs((points - point) @ normal)


def _circle_model(points: np.ndarray):
    """Algebraic (Kasa) circle fit: solve ``2x·cx + 2y·cy + c = x² + y²``."""
    if len(points) < 3:
        return None
    # Collinear samples (including three points of a RANSAC trial) have no
    # finite-radius circle; reject them before lstsq can return a huge one.
    centered = points - points.mean(axis=0)
    singular = np.linalg.svd(centered, compute_uv=False)
    if len(singular) >= 2 and singular[-1] <= max(float(singular[0]), 1.0) * 1e-12:
        return None
    x, y = points[:, 0], points[:, 1]
    design = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    target = x * x + y * y
    try:
        solution, residuals, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    if rank < 3:
        return None
    cx, cy, constant = solution
    radius_sq = constant + cx * cx + cy * cy
    if not np.isfinite(radius_sq) or radius_sq <= 0:
        return None
    radius = math.sqrt(radius_sq)
    if not np.isfinite(radius) or radius > _MAX_RADIUS:
        return None
    return np.array([cx, cy], dtype=np.float64), float(radius)


def _circle_residuals(points: np.ndarray, center: np.ndarray, radius: float):
    return np.abs(np.linalg.norm(points - center, axis=1) - radius)


def _robust_cutoff(residuals: np.ndarray) -> float:
    """Median-MAD inlier cutoff, never below :data:`_AUTO_THRESHOLD_MIN_PX`."""
    if len(residuals) == 0:
        return _AUTO_THRESHOLD_MIN_PX
    median = float(np.median(residuals))
    mad = float(np.median(np.abs(residuals - median)))
    return max(
        _AUTO_THRESHOLD_MIN_PX,
        median + _AUTO_CUTOFF_SIGMAS * _MAD_TO_SIGMA * mad,
    )


def _least_median_model(
    points: np.ndarray,
    model_fn,
    residuals_fn,
    sample_size: int,
    iterations: int,
    min_keep: int,
    seed: int,
):
    """RANSAC variant that ranks candidates by their median residual."""
    rng = np.random.default_rng(seed)
    best_model, best_key = None, None
    core_size = max(min_keep, len(points) // 2)
    for _ in range(max(1, int(iterations))):
        sample = rng.choice(len(points), size=sample_size, replace=False)
        candidate = model_fn(points[sample])
        if candidate is None:
            continue
        residuals = residuals_fn(points, *candidate)
        core = np.sort(residuals)[:core_size]
        key = (float(np.median(residuals)), float(np.mean(core**2)))
        if best_key is None or key < best_key:
            best_key, best_model = key, candidate
    return best_model


def _minimum_mask(residuals: np.ndarray, min_keep: int) -> np.ndarray:
    order = np.argsort(residuals, kind="stable")
    mask = np.zeros(len(residuals), dtype=bool)
    mask[order[:min_keep]] = True
    return mask


def _trim_model(
    points: np.ndarray,
    model,
    model_fn,
    residuals_fn,
    *,
    threshold: float,
    fraction: float,
    min_keep: int,
    passes: int = 2,
):
    """Robust cutoff and refit loop shared by both fitters.

    ``fraction`` is the largest fraction of points the fit may ignore
    (``0`` = no limit).  When the cutoff rejects more than that, the returned
    ``cap_exceeded`` flag is ``True``: the caller must report the failure
    instead of publishing a fit computed from points it was not allowed to
    discard.
    """
    total = len(points)
    residuals = residuals_fn(points, *model)
    cutoff = float(threshold) if threshold > 0 else _robust_cutoff(residuals)
    mask = residuals <= cutoff
    if int(mask.sum()) < min_keep:
        mask = _minimum_mask(residuals, min_keep)

    allowed = None
    if fraction > 0:
        allowed = max(0, min(int(round(total * float(fraction))), total - min_keep))
    if allowed is not None and total - int(mask.sum()) > allowed:
        return model, mask, True

    for _ in range(max(1, int(passes))):
        refined = model_fn(points[mask]) if int(mask.sum()) >= min_keep else None
        if refined is None:
            break
        model = refined
        residuals = residuals_fn(points, *model)
        if threshold <= 0:
            cutoff = _robust_cutoff(residuals[mask])
        new_mask = residuals <= cutoff
        if int(new_mask.sum()) < min_keep:
            new_mask = _minimum_mask(residuals, min_keep)
        if np.array_equal(new_mask, mask):
            break
        mask = new_mask

    if allowed is not None and total - int(mask.sum()) > allowed:
        return model, mask, True
    return model, mask, False


def fit_line(
    points,
    *,
    robust: bool = True,
    threshold: float = 0.0,
    iterations: int = 200,
    max_outlier_fraction: float = 0.0,
    seed: int = 0,
) -> LineFit | None:
    """Fit a line, optionally rejecting outliers.

    ``threshold`` is the maximum perpendicular residual in pixels; ``0`` derives
    a robust cutoff from the median absolute deviation.  ``max_outlier_fraction``
    (``0`` .. ``0.5``) caps the fraction of points the fit may discard; ``0``
    disables the cap.  Returns ``None`` for coincident points.  Raises
    ``ValueError`` with fewer than two valid points.
    """
    pts = clean_points(points)
    if len(pts) < 2:
        raise ValueError(f"line fit needs at least 2 valid points (found {len(pts)})")

    model = _line_model(pts)
    if model is None:
        return None

    if robust and len(pts) >= 3:
        selected = _least_median_model(
            pts, _line_model, _line_residuals, 2, iterations, 2, seed
        )
        if selected is not None:
            model = selected
        model, inliers, cap_exceeded = _trim_model(
            pts,
            model,
            _line_model,
            _line_residuals,
            threshold=float(threshold),
            fraction=min(max(float(max_outlier_fraction), 0.0), 0.5),
            min_keep=2,
        )
    else:
        inliers = np.ones(len(pts), dtype=bool)
        cap_exceeded = False

    residuals = _line_residuals(pts, *model)
    rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
    point, direction = model
    angle = math.degrees(math.atan2(-direction[1], direction[0]))
    return LineFit(
        point=point,
        direction=direction,
        angle_deg=angle,
        rms=rms,
        inliers=inliers,
        count=int(inliers.sum()),
        cap_exceeded=cap_exceeded,
    )


def fit_circle(
    points,
    *,
    robust: bool = True,
    threshold: float = 0.0,
    iterations: int = 200,
    max_outlier_fraction: float = 0.0,
    seed: int = 0,
) -> CircleFit | None:
    """Fit a circle, optionally rejecting outliers.

    ``threshold`` and ``max_outlier_fraction`` behave like in :func:`fit_line`.
    Returns ``None`` when the points are collinear or the model is degenerate
    (non-finite or absurdly large radius).  Raises ``ValueError`` with fewer
    than three valid points.
    """
    pts = clean_points(points)
    if len(pts) < 3:
        raise ValueError(f"circle fit needs at least 3 valid points (found {len(pts)})")

    model = _circle_model(pts)
    if model is None:
        return None

    if robust and len(pts) >= 4:
        selected = _least_median_model(
            pts, _circle_model, _circle_residuals, 3, iterations, 3, seed
        )
        if selected is not None:
            model = selected
        model, inliers, cap_exceeded = _trim_model(
            pts,
            model,
            _circle_model,
            _circle_residuals,
            threshold=float(threshold),
            fraction=min(max(float(max_outlier_fraction), 0.0), 0.5),
            min_keep=3,
        )
    else:
        inliers = np.ones(len(pts), dtype=bool)
        cap_exceeded = False

    residuals = _circle_residuals(pts, *model)
    rms = float(np.sqrt(np.mean(residuals[inliers] ** 2)))
    center, radius = model
    return CircleFit(
        center=center,
        radius=float(radius),
        rms=rms,
        inliers=inliers,
        count=int(inliers.sum()),
        cap_exceeded=cap_exceeded,
    )
