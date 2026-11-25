"""Piecewise-linear pixel mapping for the Filters tool.

The stored state is the final transfer function, not the history of edits:
a list of control points sorted by input value.  Two consecutive points with
the same input value form a vertical step.  The first and last points cover
the input domain; values outside it pass through unchanged.

The editor (``pixel_map_editor.py``) only mutates this curve; the runtime
below never needs to know how the curve was built.
"""

from __future__ import annotations

import math

import cv2
import numpy as np


def identity_points(domain) -> list[list[float]]:
    """Return the identity curve for ``domain``."""
    lo, hi = float(domain[0]), float(domain[1])
    if hi <= lo:
        hi = lo + 1.0
    return [[lo, lo], [hi, hi]]


def input_domain(data) -> tuple[float, float] | None:
    """Return the natural input domain for ``data``.

    Integer images use the full dtype range.  Float images use the finite
    data range, extended down to 0 so the invalid-depth value stays visible.
    """
    if data is None:
        return None
    kind = data.dtype.kind
    if kind in "ui":
        info = np.iinfo(data.dtype)
        return float(info.min), float(info.max)
    if kind == "f":
        finite = data[np.isfinite(data)]
        if finite.size == 0:
            return 0.0, 1.0
        lo = float(finite.min())
        hi = float(finite.max())
        lo = min(0.0, lo)
        if hi <= lo:
            hi = lo + 1.0
        return lo, hi
    return None


def output_range(domain, dtype) -> tuple[float, float]:
    """Return the output range shown in the editor for ``dtype``."""
    if dtype is not None and dtype.kind in "ui":
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    return float(domain[0]), float(domain[1])


def curve_limits(points, x):
    """Return ``(left, right)`` limits of the curve at input ``x``.

    A vertical step at ``x`` has different limits; a regular point or the
    interior of a segment has equal limits.
    """
    if not points:
        return float(x), float(x)
    exact = [point for point in points if abs(float(point[0]) - float(x)) < 1e-12]
    if exact:
        return float(exact[0][1]), float(exact[-1][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if float(x0) < float(x) < float(x1):
            span = float(x1) - float(x0)
            if span <= 0.0:
                continue
            t = (float(x) - float(x0)) / span
            value = float(y0) + t * (float(y1) - float(y0))
            return value, value
    return float(x), float(x)


def validate_curve(points, domain, lo, hi) -> str | None:
    """Return an error message when ``points`` is not a valid curve."""
    if not isinstance(points, (list, tuple)) or len(points) < 2:
        return "A pixel map needs at least two control points."
    xs = []
    for point in points:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return "Every control point needs an input and an output value."
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return "Control points must be numeric."
        if not (math.isfinite(x) and math.isfinite(y)):
            return "Control points must be finite."
        xs.append(x)
    for previous, current in zip(xs, xs[1:]):
        if current < previous:
            return "Control points must be sorted by input value."
    d0, d1 = float(domain[0]), float(domain[1])
    if abs(xs[0] - d0) > 1e-9 or abs(xs[-1] - d1) > 1e-9:
        return "The curve must start and end at the input domain edges."
    for x in xs:
        if x < d0 - 1e-9 or x > d1 + 1e-9:
            return "Control points must lie inside the input domain."
    for point in points:
        y = float(point[1])
        if y < lo - 1e-9 or y > hi + 1e-9:
            return "Control points must lie inside the output range."
    return None


def normalize_curve(points, domain) -> list[list[float]]:
    """Sort, clamp and collapse the curve into its canonical form."""
    d0, d1 = float(domain[0]), float(domain[1])
    items = [
        (min(d1, max(d0, float(point[0]))), float(point[1]))
        for point in points or ()
        if isinstance(point, (list, tuple)) and len(point) >= 2
    ]
    items.sort(key=lambda item: item[0])

    collapsed: list[list[float]] = []
    index = 0
    while index < len(items):
        end = index
        while (
            end + 1 < len(items)
            and abs(items[end + 1][0] - items[index][0]) < 1e-12
        ):
            end += 1
        if end == index:
            collapsed.append([items[index][0], items[index][1]])
        else:
            # A vertical line only keeps its entry and exit values.
            collapsed.append([items[index][0], items[index][1]])
            collapsed.append([items[end][0], items[end][1]])
        index = end + 1

    deduped: list[list[float]] = []
    for x, y in collapsed:
        if (
            deduped
            and abs(deduped[-1][0] - x) < 1e-12
            and abs(deduped[-1][1] - y) < 1e-12
        ):
            continue
        deduped.append([x, y])

    if len(deduped) < 2:
        return identity_points(domain)
    if deduped[0][0] > d0 + 1e-12:
        deduped.insert(0, [d0, d0])
    if deduped[-1][0] < d1 - 1e-12:
        deduped.append([d1, d1])
    return deduped


def apply_range(points, domain, in0, in1, out0, out1) -> list[list[float]]:
    """Rewrite the curve so ``[in0, in1]`` maps to ``[out0, out1]``.

    Values outside the interval keep their previous mapping; a discontinuity
    at either edge is represented by two control points sharing that input.
    """
    d0, d1 = float(domain[0]), float(domain[1])
    in0, in1 = float(in0), float(in1)
    out0, out1 = float(out0), float(out1)
    if not (d0 <= in0 < in1 <= d1):
        raise ValueError(
            "The input range must be inside the domain and not empty."
        )

    left, _ = curve_limits(points, in0)
    _, right = curve_limits(points, in1)

    result: list[list[float]] = [
        [float(x), float(y)] for x, y in points if float(x) < in0
    ]
    if in0 > d0:
        if not result or result[-1][0] < in0 - 1e-12:
            result.append([in0, left])
        if (
            abs(result[-1][0] - in0) > 1e-12
            or abs(result[-1][1] - out0) > 1e-12
        ):
            result.append([in0, out0])
    else:
        result.append([in0, out0])
    result.append([in1, out1])
    if in1 < d1:
        result.append([in1, right])
    result.extend([float(x), float(y)] for x, y in points if float(x) > in1)
    return normalize_curve(result, domain)


def fit_curve_to_range(points, domain, lo, hi) -> list[list[float]]:
    """Clamp a curve to a runtime domain and output range."""
    fitted = []
    for point in points or ():
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            continue
        x = min(domain[1], max(domain[0], float(point[0])))
        y = float(point[1])
        if math.isfinite(lo) or math.isfinite(hi):
            y = min(hi, max(lo, y))
        fitted.append([x, y])
    if len(fitted) < 2:
        return identity_points(domain)
    return normalize_curve(fitted, domain)


def build_lut(points, lo: int, hi: int) -> np.ndarray:
    """Build an integer lookup table for the curve over ``[lo, hi]``."""
    xs = np.arange(lo, hi + 1, dtype=np.float64)
    ys = xs.copy()
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        x0 = float(x0)
        x1 = float(x1)
        if x1 <= x0:
            continue
        mask = (xs >= x0) & (xs <= x1)
        if not mask.any():
            continue
        ys[mask] = (
            float(y0)
            + (xs[mask] - x0) * (float(y1) - float(y0)) / (x1 - x0)
        )
    return np.clip(np.rint(ys), lo, hi).astype(np.int64)


def _map_curve(data: np.ndarray, points) -> np.ndarray:
    """Evaluate the curve on ``data`` with inclusive segment masks.

    Later segments win at a shared boundary, so the value at a step uses the
    right-hand side.
    """
    out = data.astype(np.float64, copy=True)
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        x0 = float(x0)
        x1 = float(x1)
        if x1 <= x0:
            continue
        mask = (data >= x0) & (data <= x1)
        if not mask.any():
            continue
        out[mask] = (
            float(y0)
            + (data[mask] - x0) * (float(y1) - float(y0)) / (x1 - x0)
        )
    return out


def _map_integer(data: np.ndarray, points) -> np.ndarray:
    lo, hi = int(np.iinfo(data.dtype).min), int(np.iinfo(data.dtype).max)
    points = fit_curve_to_range(points, (lo, hi), lo, hi)
    if data.dtype.kind == "u" and data.dtype.itemsize <= 2:
        lut = build_lut(points, lo, hi).astype(data.dtype)
        try:
            return cv2.LUT(data, lut)
        except cv2.error:
            pass
    mapped = np.clip(np.rint(_map_curve(data, points)), lo, hi)
    return mapped.astype(data.dtype)


def _build_float_lut(points, lo: int, hi: int) -> np.ndarray:
    """Evaluate the curve on every integer input in ``[lo, hi]`` (no clip)."""
    xs = np.arange(lo, hi + 1, dtype=np.float64)
    ys = xs.copy()
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        x0 = float(x0)
        x1 = float(x1)
        if x1 <= x0:
            continue
        mask = (xs >= x0) & (xs <= x1)
        if mask.any():
            ys[mask] = (
                float(y0)
                + (xs[mask] - x0) * (float(y1) - float(y0)) / (x1 - x0)
            )
    return ys.astype(np.float32)


def _map_float(data: np.ndarray, points, preserve_invalid: bool) -> np.ndarray:
    """Map a float range image.

    Integer-valued counts (raw depth) use an exact one-unit LUT.  Other float
    ranges use a 65536-bin LUT over the domain; the quantisation error is at
    most half a bin (``span / 131072``).  Very wide domains fall back to the
    exact per-segment masks.
    """
    if not points:
        return data
    d0 = float(points[0][0])
    d1 = float(points[-1][0])
    span = d1 - d0
    finite = np.isfinite(data)
    out = None

    if span > 0.0 and _looks_integer(data):
        lo = int(math.ceil(d0))
        hi = int(math.floor(d1))
        count = hi - lo + 1
        if 0 < count <= 65536:
            values = _build_float_lut(points, lo, hi)
            lut = np.empty(65536, dtype=np.float32)
            lut[:count] = values
            lut[count:] = values[-1]
            index = np.rint(data)
            index -= lo
            np.clip(index, 0, count - 1, out=index)
            if not finite.all():
                index[~finite] = 0.0
            out = cv2.LUT(index.astype(np.uint16), lut)

    if out is None and 0.0 < span <= 65536.0:
        samples = np.linspace(d0, d1, 65536, dtype=np.float64)
        values = samples.copy()
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            x0 = float(x0)
            x1 = float(x1)
            if x1 <= x0:
                continue
            mask = (samples >= x0) & (samples <= x1)
            if mask.any():
                values[mask] = (
                    float(y0)
                    + (samples[mask] - x0)
                    * (float(y1) - float(y0))
                    / (x1 - x0)
                )
        lut = values.astype(np.float32)
        scale = 65535.0 / span
        index = data.copy()
        index -= d0
        index *= scale
        np.clip(index, 0.0, 65535.0, out=index)
        if not finite.all():
            index[~finite] = 0.0
        out = cv2.LUT(index.astype(np.uint16), lut)

    if out is None:
        out = data.copy()
        valid = None if not preserve_invalid else (
            finite & (data != 0.0)
        )
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            x0 = float(x0)
            x1 = float(x1)
            if x1 <= x0:
                continue
            mask = (data >= x0) & (data <= x1)
            if valid is not None:
                mask &= valid
            if mask.any():
                out[mask] = (
                    float(y0)
                    + (data[mask] - x0) * (float(y1) - float(y0)) / (x1 - x0)
                )
        return out.astype(data.dtype, copy=False)

    restore = ~finite
    if preserve_invalid:
        restore |= data == 0.0
    if span > 0.0:
        restore |= (data < d0) | (data > d1)
    if restore.any():
        out[restore] = data[restore]
    return out.astype(data.dtype, copy=False)


def _looks_integer(data: np.ndarray) -> bool:
    """Sample the array and report whether it looks like integer counts."""
    if data.size == 0:
        return False
    step = max(1, data.size // 200_000)
    sample = data.reshape(-1)[::step]
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        return False
    return bool(np.all(sample == np.rint(sample)))



def apply_pixel_map(data: np.ndarray, params: dict) -> np.ndarray:
    """Apply the stored curve to ``data``, preserving its dtype.

    Raises ``ValueError`` when the curve cannot be applied, so the tool can
    report a visible configuration error instead of publishing a wrong frame.
    """
    if data is None or data.size == 0:
        return data
    points = params.get("points")
    if not points:
        return data

    if data.dtype.kind in "ui":
        info = np.iinfo(data.dtype)
        return _map_integer(data, points)

    if data.dtype.kind != "f":
        raise ValueError(f"Unsupported pixel-map dtype: {data.dtype}")

    domain = params.get("domain") or input_domain(data)
    if not domain or float(domain[1]) <= float(domain[0]):
        domain = input_domain(data)
        if not domain:
            return data
    d0, d1 = float(domain[0]), float(domain[1])
    points = fit_curve_to_range(
        points, (d0, d1), -float("inf"), float("inf")
    )
    error = validate_curve(points, (d0, d1), -float("inf"), float("inf"))
    if error:
        raise ValueError(error)
    return _map_float(
        data, points, bool(params.get("preserve_invalid", True))
    )
