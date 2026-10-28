"""3D ROI utilities — conversion between 2D image ROIs and 3D bounding boxes."""

import numpy as np


def roi_2d_to_3d(roi_2d, depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0, flip_x=True):
    """Convert a 2D image ROI to a 3D bounding box using depth and calibration.

    Parameters
    ----------
    roi_2d : list-like [x, y, w, h, angle]
        ROI in pixel/fixture coordinates.
    depth : (H, W) float32 array
        Depth/range image.
    scale_x, scale_y, scale_z, offset_z : float
        Calibration parameters mapping pixel→metric.
    flip_x : bool
        If True (default), image columns are mirrored to world X so that the
        3D point cloud matches the 2D image left/right.

    Returns
    -------
    center_3d : (3,) float32
        Box center in 3D space.
    size_3d : (3,) float32
        Box dimensions in 3D space.
    z_min, z_max : float
        Depth range in metric units.
    """
    x, y, w, h, angle = float(roi_2d[0]), float(roi_2d[1]), float(roi_2d[2]), float(roi_2d[3]), float(roi_2d[4]) if len(roi_2d) >= 5 else 0.0

    img_w = depth.shape[1] if depth is not None and hasattr(depth, "shape") and depth.ndim >= 2 else None
    cx_img = x + w * 0.5
    if flip_x and img_w is not None:
        cx_img = (img_w - 1) - cx_img

    cx = cx_img * scale_x
    cy = (y + h * 0.5) * scale_y
    sx = w * scale_x
    sy = h * scale_y

    if depth is not None and hasattr(depth, "ndim") and depth.ndim == 2:
        auto_z = _auto_z_range(depth, x, y, w, h, scale_z, offset_z)
        z_min, z_max = auto_z
    else:
        z_min, z_max = 0.0, 1.0

    cz = (z_min + z_max) * 0.5
    sz = z_max - z_min

    center_3d = np.array([cx, cy, cz], dtype=np.float32)
    size_3d = np.array([sx, sy, max(sz, 1e-6)], dtype=np.float32)
    return center_3d, size_3d, z_min, z_max


def roi_3d_to_2d(center_3d, size_3d, scale_x=1.0, scale_y=1.0, image_width=None, flip_x=True):
    """Convert a 3D bounding box XY extent to a 2D image ROI.

    Parameters
    ----------
    center_3d : (3,) float32
        Box center in 3D space.
    size_3d : (3,) float32
        Box dimensions in 3D space.
    scale_x, scale_y : float
        Calibration parameters mapping metric→pixel.
    image_width : int or None
        Width of the source image in pixels. Required when flip_x is True.
    flip_x : bool
        If True (default), undo the world-X mirroring used by the point cloud.

    Returns
    -------
    roi_2d : [x, y, w, h, angle]
    """
    cx_3d, cy_3d = float(center_3d[0]), float(center_3d[1])
    sx_3d, sy_3d = float(size_3d[0]), float(size_3d[1])

    w = sx_3d / scale_x if abs(scale_x) > 1e-12 else 0.0
    h = sy_3d / scale_y if abs(scale_y) > 1e-12 else 0.0
    if abs(scale_x) > 1e-12:
        cx_img = cx_3d / scale_x
        if flip_x and image_width is not None:
            cx_img = (image_width - 1) - cx_img
        x = cx_img - w * 0.5
    else:
        x = 0.0
    y = cy_3d / scale_y - h * 0.5 if abs(scale_y) > 1e-12 else 0.0

    return [x, y, w, h, 0.0]


def _auto_z_range(depth, rx, ry, rw, rh, scale_z, offset_z):
    """Auto-detect Z min/max within a 2D rectangle on a depth map."""
    h_img, w_img = depth.shape[:2]
    x0 = max(0, int(round(rx)))
    y0 = max(0, int(round(ry)))
    x1 = min(w_img, max(x0 + 1, int(round(rx + rw))))
    y1 = min(h_img, max(y0 + 1, int(round(ry + rh))))

    patch = depth[y0:y1, x0:x1]
    if patch.size == 0:
        return 0.0, 1.0

    valid = patch[np.isfinite(patch)]
    if valid.size == 0:
        return 0.0, 1.0

    z_min = float(np.min(valid)) * scale_z + offset_z
    z_max = float(np.max(valid)) * scale_z + offset_z
    if z_max - z_min < 1e-6:
        z_max = z_min + 1.0

    return z_min, z_max


def extract_points_in_rois(points, rois, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0):
    """Extract points falling within any of the given 3D ROIs.

    Parameters
    ----------
    points : (N, 3) float32
        Full point cloud.
    rois : list of dict
        Each dict: {"roi": [x, y, w, h, angle], "z_min": float, "z_max": float}

    Returns
    -------
    mask : (N,) bool
        True for points inside any ROI.
    """
    if len(points) == 0 or not rois:
        return np.zeros(len(points), dtype=bool)

    mask = np.zeros(len(points), dtype=bool)
    for roi_entry in rois:
        roi_2d = roi_entry["roi"]
        z_min = float(roi_entry.get("z_min", 0.0))
        z_max = float(roi_entry.get("z_max", 1.0))
        x, y, w, h = float(roi_2d[0]), float(roi_2d[1]), float(roi_2d[2]), float(roi_2d[3])

        cx_3d = (x + w * 0.5) * scale_x
        cy_3d = (y + h * 0.5) * scale_y
        sx_3d = w * scale_x
        sy_3d = h * scale_y
        sz_3d = z_max - z_min
        cz_3d = (z_min + z_max) * 0.5

        half = np.array([sx_3d, sy_3d, sz_3d], dtype=np.float32) * 0.5
        center = np.array([cx_3d, cy_3d, cz_3d], dtype=np.float32)
        lo = center - half
        hi = center + half
        roi_mask = np.all((points >= lo) & (points <= hi), axis=1)
        mask |= roi_mask

    return mask


def roi_2d_from_params(params, index=0):
    """Get the 2D ROI definition from tool params for a given index.

    Returns [x, y, w, h, angle] or None.
    """
    rois = params.get("rois", [])
    if not rois or index >= len(rois):
        return None
    return rois[index].get("roi", None)


def z_range_from_params(params, index=0):
    """Get the Z range from tool params for a given ROI index.

    Returns (z_min, z_max) or (0.0, 1.0).
    """
    rois = params.get("rois", [])
    if not rois or index >= len(rois):
        return 0.0, 1.0
    entry = rois[index]
    return float(entry.get("z_min", 0.0)), float(entry.get("z_max", 1.0))


def roi_3d_from_params(params, index=0, depth=None, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0, fixture_to_image_fn=None):
    """Get full 3D ROI (center_3d, size_3d) from tool params for a given index.

    If missing or zero, auto-compute from depth. ``fixture_to_image_fn`` (typically
    ``VisionImage.fixture_to_image``) converts the stored fixture-coordinate ROI
    to absolute image pixels before sampling depth.
    """
    roi_2d = roi_2d_from_params(params, index)
    if roi_2d is None or (roi_2d[2] == 0 and roi_2d[3] == 0):
        return None, None

    roi_abs = list(roi_2d)
    if fixture_to_image_fn is not None:
        try:
            roi_abs = list(fixture_to_image_fn(roi_2d))
        except Exception:
            roi_abs = list(roi_2d)

    z_min, z_max = z_range_from_params(params, index)
    center_3d, size_3d, z_min_auto, z_max_auto = roi_2d_to_3d(
        roi_abs, depth, scale_x, scale_y, scale_z, offset_z
    )

    if z_min == 0.0 and z_max == 1.0:
        z_min, z_max = z_min_auto, z_max_auto
        cz = (z_min + z_max) * 0.5
        sz = z_max - z_min
        center_3d[2] = cz
        size_3d[2] = max(sz, 1e-6)

    return center_3d, size_3d


def all_rois_3d(params, depth=None, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0, fixture_to_image_fn=None):
    """Return list of (center_3d, size_3d) for all ROIs in params."""
    rois = params.get("rois", [])
    result = []
    for i in range(len(rois)):
        c, s = roi_3d_from_params(params, i, depth, scale_x, scale_y, scale_z, offset_z,
                                  fixture_to_image_fn=fixture_to_image_fn)
        if c is not None:
            result.append((c, s))
    return result


def update_roi_2d_in_params(params, index, roi_2d):
    """Update the 2D ROI for a given index in params."""
    rois = params.get("rois", [])
    if not rois or index >= len(rois):
        return
    rois[index]["roi"] = [float(v) for v in roi_2d]


def update_z_range_in_params(params, index, z_min, z_max):
    """Update Z range for a given ROI index in params."""
    rois = params.get("rois", [])
    if not rois or index >= len(rois):
        return
    rois[index]["z_min"] = float(z_min)
    rois[index]["z_max"] = float(z_max)


def add_roi(params, roi_2d=None, z_min=0.0, z_max=1.0):
    """Add a new ROI entry to params."""
    if "rois" not in params:
        params["rois"] = []
    if roi_2d is None:
        roi_2d = [0.0, 0.0, 0.0, 0.0, 0.0]
    params["rois"].append({
        "roi": [float(v) for v in roi_2d],
        "z_min": float(z_min),
        "z_max": float(z_max),
    })
    return len(params["rois"]) - 1


def remove_roi(params, index):
    """Remove a ROI entry from params by index."""
    rois = params.get("rois", [])
    if 0 <= index < len(rois):
        del rois[index]


def auto_fit_z(params, index, depth, scale_z=1.0, offset_z=0.0, fixture_to_image_fn=None):
    """Auto-detect Z range for a given ROI and update params.

    The ROI stored in ``params`` is in fixture/relative coordinates. ``depth``
    is indexed in absolute image coordinates, so when there is a fixture offset
    or rotation, pass ``fixture_to_image_fn`` (typically ``VisionImage.fixture_to_image``)
    to convert the ROI to absolute pixels before indexing.
    """
    roi_2d = roi_2d_from_params(params, index)
    if roi_2d is None:
        return None
    roi_abs = list(roi_2d)
    if fixture_to_image_fn is not None:
        try:
            roi_abs = list(fixture_to_image_fn(roi_2d))
        except Exception:
            roi_abs = list(roi_2d)
    x, y, w, h = float(roi_abs[0]), float(roi_abs[1]), float(roi_abs[2]), float(roi_abs[3])
    z_min, z_max = _auto_z_range(depth, x, y, w, h, scale_z, offset_z)
    update_z_range_in_params(params, index, z_min, z_max)
    return z_min, z_max


def calib_from_vimg(vimg):
    """Extract calibration tuple from a VisionImage."""
    return (
        float(getattr(vimg, "scale_x", 1.0)),
        float(getattr(vimg, "scale_y", 1.0)),
        float(getattr(vimg, "scale_z", 1.0)),
        float(getattr(vimg, "offset_z", 0.0)),
    )


def get_3d_data(vimg):
    """Get 3D point cloud data from a VisionImage."""
    if vimg is None:
        return None
    return getattr(vimg, "_3d_data", None)
