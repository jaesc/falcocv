"""Plane fitting for 3D point clouds.

Python reference/fallback implementation of the native ``_falco_kernel.plane_fit``
function.  It mirrors the C++ compute: ROI selection in world coordinates, auto
Z ranges, LSQ/RANSAC fitting and the plane/image-box intersection polygon.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

try:
    from _falco_kernel import plane_fit as _kernel_plane_fit
except ImportError:
    _kernel_plane_fit = None


@dataclass(frozen=True)
class PlaneModel:
    """A plane in world coordinates with a deterministic normal sense.

    ``normal`` is unit length and points towards world +Z/up
    (``normal[2] >= 0``), so ``height(point) > 0`` means the point is above the
    plane.  A plane has two equivalent normals; fixing the sense here keeps
    every consumer from having to guess.

    ``calib`` is the ``(scale_x, scale_y, scale_z, offset_z)`` used to build
    the world coordinates.  It is a guard, not geometry: a consumer with a
    different calibration would get wrong heights without any other symptom.
    """

    normal: tuple[float, float, float]
    distance: float
    center: tuple[float, float, float]
    rms_error: float = 0.0
    inlier_ratio: float = 0.0
    point_count: int = 0
    calib: tuple[float, float, float, float] | None = None
    edge_on: bool = False

    def height(self, points) -> np.ndarray:
        """Signed height of ``points`` above the plane (positive = +Z/up)."""
        points = np.asarray(points, dtype=np.float64)
        normal = np.asarray(self.normal, dtype=np.float64)
        center = np.asarray(self.center, dtype=np.float64)
        return (points - center) @ normal

    def offset(self, distance) -> "PlaneModel":
        """Return the same plane shifted ``distance`` along +normal.

        Positive moves the plane up (+Z), so measured heights decrease by
        ``distance``.  Negative lowers the plane, which is what you need when
        the features sit below the fitted surface.
        """
        distance = float(distance)
        if distance == 0.0:
            return self
        normal = np.asarray(self.normal, dtype=np.float64)
        center = np.asarray(self.center, dtype=np.float64) + normal * distance
        return replace(
            self,
            distance=self.distance + distance,
            center=tuple(float(value) for value in center),
        )

    def frame(self, x_hint=None) -> "PlaneFrame":
        """Frame anchored to this plane; ``x_hint`` fixes the in-plane X."""
        return _plane_frame(self, x_hint)


def plane_height(plane: PlaneModel, points) -> np.ndarray:
    """Signed height of ``points`` above ``plane`` (positive = +Z/up)."""
    return plane.height(points)


@dataclass(frozen=True)
class PlaneFrame:
    """Right-handed frame anchored to a plane.

    ``z`` is the plane normal, ``x`` the fixture direction projected onto the
    plane and ``y = z x x``.  Coordinates in this frame are invariant to how
    the part is tilted relative to the camera.
    """

    origin: tuple[float, float, float]
    x: tuple[float, float, float]
    y: tuple[float, float, float]
    z: tuple[float, float, float]

    def to_plane(self, points):
        """World points -> (u, v, h) in the frame."""
        values = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        origin = np.asarray(self.origin, dtype=np.float64)
        relative = values - origin
        u = relative @ np.asarray(self.x, dtype=np.float64)
        v = relative @ np.asarray(self.y, dtype=np.float64)
        h = relative @ np.asarray(self.z, dtype=np.float64)
        return u, v, h

    def to_world(self, u, v, h):
        """(u, v, h) in the frame -> world points."""
        u = np.asarray(u, dtype=np.float64)
        v = np.asarray(v, dtype=np.float64)
        h = np.asarray(h, dtype=np.float64)
        origin = np.asarray(self.origin, dtype=np.float64)
        axes = [
            np.asarray(self.x, dtype=np.float64),
            np.asarray(self.y, dtype=np.float64),
            np.asarray(self.z, dtype=np.float64),
        ]
        return origin + u[..., None] * axes[0] + v[..., None] * axes[1] + h[..., None] * axes[2]


def _plane_frame(plane, x_hint=None):
    normal = np.asarray(plane.normal, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(normal))
    if length < 1e-12:
        normal = np.array([0.0, 0.0, 1.0])
    else:
        normal = normal / length

    hint = np.asarray(
        x_hint if x_hint is not None else (1.0, 0.0, 0.0),
        dtype=np.float64,
    ).reshape(3)
    axis = hint - float(hint @ normal) * normal
    if float(np.linalg.norm(axis)) < 1e-6:
        for candidate in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)):
            trial = np.asarray(candidate, dtype=np.float64)
            axis = trial - float(trial @ normal) * normal
            if float(np.linalg.norm(axis)) >= 1e-6:
                break
    axis = axis / float(np.linalg.norm(axis))
    other = np.cross(normal, axis)
    return PlaneFrame(
        origin=tuple(float(value) for value in plane.center),
        x=tuple(float(value) for value in axis),
        y=tuple(float(value) for value in other),
        z=tuple(float(value) for value in normal),
    )


def normalize_plane(normal, distance, z_epsilon=1e-3):
    """Normalise a plane and fix the normal sense.

    Returns ``(unit_normal, distance, edge_on)``.  ``edge_on`` is True when the
    plane is almost parallel to the optical axis, where a height map is not
    meaningful.
    """
    values = np.asarray(normal, dtype=np.float64).reshape(3)
    length = float(np.linalg.norm(values))
    if length < 1e-12:
        return (0.0, 0.0, 1.0), float(distance), True
    values = values / length
    plane_distance = float(distance) / length
    if values[2] < 0.0:
        values = -values
        plane_distance = -plane_distance
    elif abs(values[2]) < z_epsilon:
        # Edge-on: the Z sense carries no information, so pick a deterministic
        # one from the first non-zero component.
        for value in values:
            if abs(value) > 1e-9:
                if value < 0.0:
                    values = -values
                    plane_distance = -plane_distance
                break
    edge_on = abs(float(values[2])) < z_epsilon
    return tuple(float(value) for value in values), plane_distance, edge_on


def _svd_fit_plane(points):
    pts = np.asarray(points, dtype=np.float64)
    n = len(pts)
    center = pts.mean(axis=0)
    q = pts - center
    cov = (q.T @ q) / n
    _, _, vt = np.linalg.svd(cov)
    normal = vt[-1]
    length = float(np.linalg.norm(normal))
    if length > 1e-12:
        normal = normal / length
    dist = float(np.dot(normal, center))
    return normal.astype(np.float32), center.astype(np.float32), dist


def _point_in_roi_xy(x, y, roi):
    cx, cy, sx, sy, angle = roi
    if abs(angle) < 1e-6:
        return (x >= cx - sx * 0.5) & (x <= cx + sx * 0.5) & \
               (y >= cy - sy * 0.5) & (y <= cy + sy * 0.5)
    a_rad = math.radians(angle)
    ca, sa = math.cos(a_rad), math.sin(a_rad)
    dx = x - cx
    dy = y - cy
    lx = dx * ca + dy * sa
    ly = -dx * sa + dy * ca
    return (np.abs(lx) <= sx * 0.5) & (np.abs(ly) <= sy * 0.5)


def _compute_plane_box_intersection(normal, dist, x_min, x_max, y_min, y_max,
                                    z_min, z_max, scale_x, scale_y, img_w):
    edges = [
        (x_min, y_min, z_min, x_max, y_min, z_min),
        (x_min, y_max, z_min, x_max, y_max, z_min),
        (x_min, y_min, z_max, x_max, y_min, z_max),
        (x_min, y_max, z_max, x_max, y_max, z_max),
        (x_min, y_min, z_min, x_min, y_max, z_min),
        (x_max, y_min, z_min, x_max, y_max, z_min),
        (x_min, y_min, z_max, x_min, y_max, z_max),
        (x_max, y_min, z_max, x_max, y_max, z_max),
        (x_min, y_min, z_min, x_min, y_min, z_max),
        (x_max, y_min, z_min, x_max, y_min, z_max),
        (x_min, y_max, z_min, x_min, y_max, z_max),
        (x_max, y_max, z_min, x_max, y_max, z_max),
    ]
    nx, ny, nz = (float(normal[0]), float(normal[1]), float(normal[2]))
    pts = []
    for x0, y0, z0, x1, y1, z1 in edges:
        dx, dy, dz = x1 - x0, y1 - y0, z1 - z0
        denom = nx * dx + ny * dy + nz * dz
        if abs(denom) < 1e-12:
            continue
        t = (dist - (nx * x0 + ny * y0 + nz * z0)) / denom
        if t < 0.0 or t > 1.0:
            continue
        px = x0 + t * dx
        py = y0 + t * dy
        col = (img_w - 1) - (px / scale_x)
        row = py / scale_y
        pts.append((float(col), float(row)))

    if len(pts) < 3:
        pts = [
            (0.0, 0.0),
            (float(img_w - 1), 0.0),
            (float(img_w - 1), float(y_max / scale_y)),
            (0.0, float(y_max / scale_y)),
        ]

    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    pts.sort(key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    return [[p[0], p[1]] for p in pts]


def plane_fit(input_dict):
    """Fit a plane to ``points`` constrained by optional 3D ROIs.

    Returns the same dict shape as the native kernel. On invalid input it
    returns ``{"error": "..."}`` instead of raising.
    """
    if _kernel_plane_fit is not None:
        return _kernel_plane_fit(input_dict)

    try:
        params = input_dict.get("params", {})
        points = np.asarray(input_dict["points"], dtype=np.float32)
        if points.ndim != 2 or points.shape[1] != 3:
            return {"error": "points must be (N, 3) float32"}
        n = len(points)
        if n < 3:
            return {"error": "at least 3 points required for plane fitting"}

        depth = input_dict.get("image")
        if depth is not None:
            depth = np.asarray(depth, dtype=np.float32)
        if depth is not None and depth.ndim == 2:
            img_h, img_w = depth.shape
        else:
            img_w = max(1, int(math.ceil(math.sqrt(n))))
            img_h = max(1, int(math.ceil(n / img_w)))

        method = str(params.get("method", "ransac"))
        dist_thresh = float(params.get("distance_threshold", 1.0))
        max_iter = int(params.get("max_iterations", 100))
        min_points = int(params.get("min_points", 10))
        max_fit_points = max(100, int(params.get("max_fit_points", 50000)))

        calib = input_dict.get("calib", [1.0, 1.0, 1.0, 0.0])
        try:
            scale_x = float(calib[0]) if len(calib) >= 1 else 1.0
            scale_y = float(calib[1]) if len(calib) >= 2 else 1.0
            scale_z = float(calib[2]) if len(calib) >= 3 else 1.0
            offset_z = float(calib[3]) if len(calib) >= 4 else 0.0
        except Exception:
            scale_x = scale_y = scale_z = 1.0
            offset_z = 0.0
        if not np.isfinite(scale_x) or abs(scale_x) < 1e-12:
            scale_x = 1.0
        if not np.isfinite(scale_y) or abs(scale_y) < 1e-12:
            scale_y = 1.0
        if not np.isfinite(scale_z) or abs(scale_z) < 1e-12:
            scale_z = 1.0
        if not np.isfinite(offset_z):
            offset_z = 0.0

        img_z_min, img_z_max = 0.0, 1.0
        if depth is not None and depth.ndim == 2:
            finite_depth = depth[np.isfinite(depth)]
            if finite_depth.size:
                img_z_min = float(finite_depth.min()) * scale_z + offset_z
                img_z_max = float(finite_depth.max()) * scale_z + offset_z
        if img_z_max - img_z_min < 1e-6:
            img_z_max = img_z_min + 1.0

        # Parse ROIs (absolute image coordinates already converted by the tool).
        rois = []
        for entry in input_dict.get("rois", []) or []:
            roi = entry.get("roi") if isinstance(entry, dict) else None
            if roi is None or len(roi) < 4:
                continue
            x, y, w, h = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
            a_deg = float(roi[4]) if len(roi) > 4 else 0.0
            cx_img = x + w * 0.5
            cx = ((img_w - 1) - cx_img) * scale_x
            cy = (y + h * 0.5) * scale_y
            sx = w * scale_x
            sy = h * scale_y
            z_min = float(entry.get("z_min", 0.0)) if isinstance(entry, dict) else 0.0
            z_max = float(entry.get("z_max", 1.0)) if isinstance(entry, dict) else 1.0
            rois.append({
                "cx": cx, "cy": cy, "sx": sx, "sy": sy,
                "angle": -a_deg, "z_min": z_min, "z_max": z_max,
                "auto_z": z_min == 0.0 and z_max == 1.0,
            })

        if not rois:
            rois = [{
                "cx": 0.0, "cy": 0.0,
                "sx": float("inf"), "sy": float("inf"),
                "angle": 0.0, "z_min": -float("inf"), "z_max": float("inf"),
                "auto_z": False,
            }]

        xs = points[:, 0].astype(np.float64)
        ys = points[:, 1].astype(np.float64)
        zs = points[:, 2].astype(np.float64)

        for roi in rois:
            if not roi["auto_z"]:
                continue
            xy = _point_in_roi_xy(xs, ys, (roi["cx"], roi["cy"], roi["sx"], roi["sy"], roi["angle"]))
            z_valid = zs[xy & np.isfinite(zs)]
            if z_valid.size:
                z_min = float(z_valid.min())
                z_max = float(z_valid.max())
            else:
                z_min, z_max = 0.0, 1.0
            if z_max - z_min < 1e-6:
                z_max = z_min + 1.0
            roi["z_min"] = z_min
            roi["z_max"] = z_max

        selected_mask = np.zeros(n, dtype=bool)
        for roi in rois:
            xy = _point_in_roi_xy(xs, ys, (roi["cx"], roi["cy"], roi["sx"], roi["sy"], roi["angle"]))
            selected_mask |= xy & (zs >= roi["z_min"]) & (zs <= roi["z_max"]) & np.isfinite(zs)
        selected_idx = np.nonzero(selected_mask)[0].astype(np.int64)
        selected_n = len(selected_idx)
        if selected_n < 3:
            return {"error": "not enough points inside ROIs for plane fitting"}

        if selected_n > max_fit_points:
            stride = max(1, selected_n // max_fit_points)
            fit_idx = selected_idx[::stride]
        else:
            fit_idx = selected_idx
        fit_pts = points[fit_idx]

        best_normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        best_center = np.zeros(3, dtype=np.float32)
        best_dist = 0.0
        best_inlier_count = 0
        best_rms = 0.0

        if method == "ransac":
            rng = np.random.default_rng(42)
            fit_n = len(fit_pts)
            for _ in range(max_iter):
                i0, i1, i2 = rng.integers(0, fit_n, size=3)
                if i0 == i1 or i0 == i2 or i1 == i2:
                    continue
                p0, p1, p2 = fit_pts[i0], fit_pts[i1], fit_pts[i2]
                normal = np.cross(p1 - p0, p2 - p0)
                length = float(np.linalg.norm(normal))
                if length < 1e-12:
                    continue
                normal = normal / length
                d = float(np.dot(normal, p0))
                dists = np.abs(fit_pts @ normal - d)
                inlier_count = int(np.count_nonzero(dists < dist_thresh))
                if inlier_count > best_inlier_count and inlier_count >= min_points:
                    refit_normal, refit_center, refit_dist = _svd_fit_plane(fit_pts[dists < dist_thresh])
                    best_inlier_count = inlier_count
                    best_normal = refit_normal
                    best_center = refit_center
                    best_dist = refit_dist

            if best_inlier_count == 0:
                best_normal, best_center, best_dist = _svd_fit_plane(fit_pts)
                best_inlier_count = len(fit_pts)

            best_inliers = np.zeros(selected_n, dtype=bool)
            sum_sq = 0.0
            for i, idx in enumerate(selected_idx):
                dp = abs(float(np.dot(points[idx], best_normal)) - best_dist)
                inlier = dp < dist_thresh
                best_inliers[i] = inlier
                if inlier:
                    sum_sq += dp * dp
            best_rms = math.sqrt(sum_sq / max(1, best_inlier_count))
        else:
            best_normal, best_center, best_dist = _svd_fit_plane(fit_pts)
            best_inlier_count = len(fit_pts)
            best_inliers = np.ones(selected_n, dtype=bool)
            dp = np.abs(points[selected_idx] @ best_normal - best_dist)
            best_rms = float(math.sqrt(float(np.sum(dp * dp)) / selected_n))

        box_x_min = 0.0
        box_x_max = float(img_w - 1) * scale_x
        box_y_min = 0.0
        box_y_max = float(img_h - 1) * scale_y
        polygon = _compute_plane_box_intersection(
            best_normal, best_dist,
            box_x_min, box_x_max, box_y_min, box_y_max,
            img_z_min, img_z_max, scale_x, scale_y, img_w,
        )

        return {
            "normal": best_normal.astype(np.float32),
            "center": best_center.astype(np.float32),
            "distance": float(best_dist),
            "inlier_count": int(best_inlier_count),
            "inlier_ratio": float(best_inlier_count) / max(1, selected_n),
            "rms_error": float(best_rms),
            "total_points": int(selected_n),
            "inlier_mask": best_inliers.astype(np.uint8),
            "selected_indices": selected_idx,
            "polygon_2d": polygon,
        }
    except Exception as exc:
        return {"error": str(exc)}
