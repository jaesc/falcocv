"""Shared helpers for the metrology fit tools (LineFit / CircleFit).

Two point sources feed the robust fitters:

* a caliper array driven by the tool ROI (the default), which detects
  subpixel edges and returns one point per caliper, and
* existing points from overlay, contour or polygon elements.

The module also filters points with the tool ROI and prepares the output
VisionImage with the overlay-ownership rules every processing tool follows.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from falcocv.core.overlay_layer import OverlayType
from falcocv.core.pointcloud_utils import propagate_3d
from falcocv.core.vision_image import VisionImage


def _point_from_item(item):
    if isinstance(item, dict):
        if "center" in item:
            return item["center"]
        return item.get("x"), item.get("y")
    if isinstance(item, (tuple, list)):
        return item[:2] if len(item) >= 2 else None
    center = getattr(item, "center", None)
    if center is not None:
        return center
    return getattr(item, "x", None), getattr(item, "y", None)


def normalize_points(raw) -> np.ndarray:
    """Coerce tuples, dicts, result objects or arrays into an ``(N, 2)`` array.

    Accepts a NumPy array, a list of ``(x, y)`` pairs, dictionaries with
    ``x``/``y`` (or ``center``) and result objects with ``x``/``y`` attributes,
    so an upstream tool's ``ToolResults`` can feed the point overlays directly.
    """
    if raw is None:
        return np.empty((0, 2), dtype=np.float64)
    if isinstance(raw, np.ndarray):
        return _finite(raw)
    try:
        iterator = iter(raw)
    except TypeError:
        return np.empty((0, 2), dtype=np.float64)
    if np.isscalar(raw):
        return np.empty((0, 2), dtype=np.float64)
    pairs = []
    for item in iterator:
        point = _point_from_item(item)
        if point is None:
            continue
        pairs.append(point)
    return _finite(pairs)


def _finite(points) -> np.ndarray:
    array = np.asarray(points, dtype=np.float64)
    if array.size == 0:
        return np.empty((0, 2), dtype=np.float64)
    array = array.reshape(-1, 2)
    return array[np.isfinite(array).all(axis=1)]


def collect_overlay_points(
    v_image: VisionImage,
    *,
    use_points: bool = True,
    use_contours: bool = False,
    exclude_source: str = "",
) -> np.ndarray:
    """Points of the visible POINT / CONTOUR / POLYGON overlays."""
    points = []
    for layer in v_image.overlay_layers.values():
        for element in layer.elements:
            if not element.visible:
                continue
            if exclude_source and element.source_tool == exclude_source:
                # Never feed the tool its own overlays back into the next run.
                continue
            if element.element_type == OverlayType.POINT and use_points:
                center = element.data.get("center")
                if center is not None:
                    points.append(center)
            elif (
                element.element_type in (OverlayType.CONTOUR, OverlayType.POLYGON)
                and use_contours
            ):
                raw = element.data.get("contour") or element.data.get("points") or []
                points.extend(raw)
    return normalize_points(points)


def roi_to_image(v_image: VisionImage, roi) -> list[float] | None:
    """ROI stored in fixture coordinates → absolute image coordinates."""
    if not roi or len(roi) < 4 or float(roi[2]) <= 0 or float(roi[3]) <= 0:
        return None
    return list(v_image.fixture_to_image(list(roi)))


def filter_by_roi(points: np.ndarray, roi_image) -> np.ndarray:
    """Keep the points inside the rotated ROI (all points when it is empty)."""
    if roi_image is None or len(points) == 0:
        return points
    rx, ry, rw, rh, angle = roi_image
    theta = np.deg2rad(angle)
    cos_a, sin_a = np.cos(theta), np.sin(theta)
    delta = points - np.array([rx + rw / 2.0, ry + rh / 2.0])
    u = cos_a * delta[:, 0] + sin_a * delta[:, 1]
    v = -sin_a * delta[:, 0] + cos_a * delta[:, 1]
    mask = (np.abs(u) <= rw / 2.0) & (np.abs(v) <= rh / 2.0)
    return points[mask]


@dataclass
class CaliperScan:
    """Edge points found by a caliper array plus the scan segments to draw."""

    points: np.ndarray
    segments: list

    @property
    def total(self) -> int:
        return len(self.segments)


#: Auto caliper band length as a fraction of the ROI's short side.
_AUTO_SCAN_FRACTION = 0.35


def _to_gray(image_data):
    if image_data.ndim == 2:
        return image_data
    if image_data.shape[2] == 4:
        return cv2.cvtColor(image_data, cv2.COLOR_BGRA2GRAY)
    return cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)


def derivative_of_gaussian(profile, sigma=1.0):
    """Smooth a 1D profile and return its central-difference derivative."""
    profile = np.asarray(profile, dtype=np.float32)
    if sigma and sigma > 0.1:
        size = int(6 * sigma) | 1
        x = np.arange(size) - size // 2
        kernel = np.exp(-x**2 / (2 * sigma**2))
        kernel /= kernel.sum()
        pad = size // 2
        profile = np.convolve(
            np.pad(profile, pad, mode="edge"), kernel, mode="valid"
        )
    grad = np.zeros_like(profile)
    if len(profile) >= 3:
        grad[1:-1] = (profile[2:] - profile[:-2]) * 0.5
    return grad


def _subpixel_peak(grad, idx):
    if idx <= 0 or idx >= len(grad) - 1:
        return float(idx)
    left, mid, right = float(grad[idx - 1]), float(grad[idx]), float(grad[idx + 1])
    denom = 2.0 * (left + right - 2.0 * mid)
    if abs(denom) < 1e-6:
        return float(idx)
    return float(idx) + (left - right) / denom


def _edge_peak(grad, threshold, polarity, selector):
    """Return ``(position, strength)`` of the chosen edge in a 1D gradient."""
    if len(grad) < 3:
        return None
    if polarity == "Dark\u2192Light":
        peaks = (grad[1:-1] > grad[:-2]) & (grad[1:-1] > grad[2:])
        idx = np.where(peaks)[0] + 1
        strengths = grad[idx]
    elif polarity == "Light\u2192Dark":
        peaks = (grad[1:-1] < grad[:-2]) & (grad[1:-1] < grad[2:])
        idx = np.where(peaks)[0] + 1
        strengths = -grad[idx]
    else:
        abs_grad = np.abs(grad)
        peaks = (abs_grad[1:-1] > abs_grad[:-2]) & (abs_grad[1:-1] > abs_grad[2:])
        idx = np.where(peaks)[0] + 1
        strengths = abs_grad[idx]
    keep = strengths >= max(1e-6, float(threshold))
    idx, strengths = idx[keep], strengths[keep]
    if idx.size == 0:
        return None
    if selector == "First":
        pick = 0
    elif selector == "Last":
        pick = -1
    else:
        pick = int(np.argmax(strengths))
    return _subpixel_peak(grad, int(idx[pick])), float(strengths[pick])


def scan_line_calipers(
    image_data,
    roi_image,
    *,
    count=9,
    length=0.0,
    width=5.0,
    polarity="Any",
    threshold=5.0,
    sigma=1.0,
    selector="Best",
    subpixel=True,
) -> CaliperScan:
    """Scan a ROI with ``count`` parallel calipers across its local V axis.

    The ROI's long (U) axis is the fit direction; every caliper averages
    ``width`` pixels along U and scans ``length`` pixels across V.  ``length``
    and ``0`` means "the full ROI height".  Profile direction follows +V, so
    ``Dark\u2192Light`` is a brightening edge towards the ROI's local bottom.
    """
    empty = CaliperScan(np.empty((0, 2), dtype=np.float64), [])
    if roi_image is None:
        return empty
    x, y, w, h, angle = [float(value) for value in roi_image]
    if w < 1.0 or h < 3.0:
        return empty

    count = max(1, int(count))
    gray = _to_gray(image_data)
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
    theta = np.deg2rad(angle)
    u = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
    v = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)

    scan_len = float(length) if float(length) > 0 else h
    scan_len = max(3.0, min(scan_len, h))
    caliper_w = max(1.0, min(float(width), w))
    if count == 1:
        stations = np.array([0.0])
    else:
        step = w / count
        stations = np.linspace(-w / 2.0 + step / 2.0, w / 2.0 - step / 2.0, count)

    num_s = max(3, int(round(scan_len)))
    num_q = max(1, int(round(caliper_w)))
    s_values = np.linspace(-scan_len / 2.0, scan_len / 2.0, num_s)
    q_values = np.linspace(-caliper_w / 2.0, caliper_w / 2.0, num_q)
    Q, S = np.meshgrid(q_values, s_values)

    points, segments = [], []
    for station in stations:
        base = center + u * station
        map_x = (base[0] + u[0] * Q + v[0] * S).astype(np.float32)
        map_y = (base[1] + u[1] * Q + v[1] * S).astype(np.float32)
        samples = cv2.remap(
            gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        profile = samples.mean(axis=1, dtype=np.float32)
        segments.append(
            (
                tuple(float(value) for value in base + v * (-scan_len / 2.0)),
                tuple(float(value) for value in base + v * (scan_len / 2.0)),
            )
        )
        peak = _edge_peak(
            derivative_of_gaussian(profile, sigma), threshold, polarity, selector
        )
        if peak is None:
            continue
        position, _ = peak
        if not subpixel:
            position = float(round(position))
        if position < 0 or position > num_s - 1:
            continue
        offset = -scan_len / 2.0 + (position / (num_s - 1)) * scan_len
        points.append(base + v * offset)

    return CaliperScan(np.asarray(points, dtype=np.float64).reshape(-1, 2), segments)


def scan_radial_calipers(
    image_data,
    roi_image,
    *,
    count=16,
    radius=0.0,
    length=0.0,
    width=3.0,
    polarity="Any",
    threshold=5.0,
    sigma=1.0,
    selector="Best",
    subpixel=True,
    direction="inward",
) -> CaliperScan:
    """Scan a circle of ``count`` radial calipers around the ROI centre.

    The ring sits at ``radius`` (``0`` = the ROI's inscribed radius) and every
    caliper scans over ``length`` pixels (``0`` = 35% of the ROI's short side,
    at least 6 px) towards the centre (``direction="inward"``, the default) or
    away from it (``"outward"``).  The profile follows the scan direction, so
    ``Dark\u2192Light`` means the edge brightens towards the centre when
    scanning inwards and towards the outside when scanning outwards.
    """
    outward = str(direction or "inward").strip().lower().startswith("out")
    empty = CaliperScan(np.empty((0, 2), dtype=np.float64), [])
    if roi_image is None:
        return empty
    x, y, w, h, _angle = [float(value) for value in roi_image]
    if w < 3.0 or h < 3.0:
        return empty

    count = max(3, int(count))
    gray = _to_gray(image_data)
    center = np.array([x + w / 2.0, y + h / 2.0], dtype=np.float64)
    short = min(w, h)
    r_outer = float(radius) if float(radius) > 0 else short / 2.0
    r_outer = max(r_outer, 3.0)
    band = float(length) if float(length) > 0 else max(6.0, _AUTO_SCAN_FRACTION * short)
    band = max(3.0, band)
    if not outward:
        band = min(band, r_outer)
    caliper_w = max(1.0, float(width))

    num_s = max(3, int(round(band)))
    num_q = max(1, int(round(caliper_w)))
    s_values = np.linspace(0.0, band, num_s)
    q_values = np.linspace(-caliper_w / 2.0, caliper_w / 2.0, num_q)
    Q, S = np.meshgrid(q_values, s_values)

    points, segments = [], []
    for index in range(count):
        theta = 2.0 * np.pi * index / count
        unit = np.array([np.cos(theta), np.sin(theta)], dtype=np.float64)
        tangent = np.array([-np.sin(theta), np.cos(theta)], dtype=np.float64)
        radial = r_outer + S if outward else r_outer - S
        map_x = (center[0] + unit[0] * radial + tangent[0] * Q).astype(np.float32)
        map_y = (center[1] + unit[1] * radial + tangent[1] * Q).astype(np.float32)
        samples = cv2.remap(
            gray, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE
        )
        profile = samples.mean(axis=1, dtype=np.float32)
        end_radius = r_outer + band if outward else r_outer - band
        segments.append(
            (
                tuple(float(value) for value in center + unit * r_outer),
                tuple(float(value) for value in center + unit * end_radius),
            )
        )
        peak = _edge_peak(
            derivative_of_gaussian(profile, sigma), threshold, polarity, selector
        )
        if peak is None:
            continue
        position, _ = peak
        if not subpixel:
            position = float(round(position))
        if position < 0 or position > num_s - 1:
            continue
        offset = (position / (num_s - 1)) * band
        point_radius = r_outer + offset if outward else r_outer - offset
        points.append(center + unit * point_radius)

    return CaliperScan(np.asarray(points, dtype=np.float64).reshape(-1, 2), segments)


def drawable_roi(roi_image, image_shape) -> list[float] | None:
    """Clip the ROI to the image; ``None`` means "covers the whole image"."""
    if roi_image is None:
        return None
    height, width = image_shape[:2]
    x, y, w, h, angle = roi_image
    x0 = max(0, int(np.floor(x)))
    y0 = max(0, int(np.floor(y)))
    x1 = min(width, int(np.ceil(x + w)))
    y1 = min(height, int(np.ceil(y + h)))
    if x1 <= x0 or y1 <= y0:
        return None
    if x0 == 0 and y0 == 0 and x1 >= width and y1 >= height and abs(angle) < 1e-9:
        return None
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0), float(angle)]


def prepare_output(input_image, tool_name: str):
    """Return ``(input VisionImage, output VisionImage, claimed layer)``.

    The output inherits the upstream overlays, claims the default layer for
    ``tool_name`` and drops that tool's own stale elements, as required by the
    overlay flow.
    """
    img = (
        input_image
        if isinstance(input_image, VisionImage)
        else VisionImage.from_array(input_image)
    )
    output = VisionImage(
        img.data,
        x=img.x,
        y=img.y,
        angle=img.angle,
        scale_x=img.scale_x,
        scale_y=img.scale_y,
        scale_z=img.scale_z,
        offset_z=img.offset_z,
    )
    output.inherit_overlays_from(img)
    propagate_3d(img, output, regenerate=False)
    output.get_overlay_layer("default", source_tool=tool_name)
    layer = output.get_overlay_layer("default")
    layer.source_tool = tool_name
    layer.elements = [
        element
        for element in layer.elements
        if getattr(element, "source_tool", None) != tool_name
    ]
    return img, output, layer
