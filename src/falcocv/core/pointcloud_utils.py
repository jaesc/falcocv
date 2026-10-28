"""3D point cloud utilities â€” PLY I/O, range-to-pointcloud, normal estimation, ROI extraction."""

import cv2
import numpy as np

try:
    from _falco_kernel import generate_pointcloud as _kernel_generate_pointcloud
except ImportError:
    _kernel_generate_pointcloud = None


def _is_depth_like(vimg):
    """Return True if vimg looks like a 2D float depth map."""
    if vimg is None:
        return False
    data = vimg.data if hasattr(vimg, "data") else vimg
    return (
        isinstance(data, np.ndarray)
        and data.ndim == 2
        and data.dtype.kind == "f"
    )


def _calib_tuple(vimg):
    return (
        float(getattr(vimg, "scale_x", 1.0)),
        float(getattr(vimg, "scale_y", 1.0)),
        float(getattr(vimg, "scale_z", 1.0)),
        float(getattr(vimg, "offset_z", 0.0)),
    )


def _intensity_colors(intensity, colormap):
    from falcocv.core.depth_viz import resolve_colormap_id
    intensity = np.asarray(intensity, dtype=np.float32)
    count = len(intensity)
    colors = np.zeros((count, 3), dtype=np.uint8)
    cmap_id = resolve_colormap_id(str(colormap))
    if count == 0 or cmap_id < 0:
        return colors
    lo = float(intensity.min())
    hi = float(intensity.max())
    span = hi - lo
    if span > 0.0:
        u8 = np.clip((intensity - lo) / span * 255.0 + 0.5, 0, 255).astype(np.uint8)
    else:
        u8 = np.zeros(count, dtype=np.uint8)
    bgr = cv2.applyColorMap(u8.reshape(1, -1), cmap_id)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)


def generate_pointcloud(depth, x_res=1.0, y_res=1.0, z_res=1.0, z_offset=0.0,
                        target_points=1_000_000, zero_as_invalid=True,
                        colormap="", compute_bbox=False):
    """Generate a metric point cloud from a 2D depth/range map.

    This is the Python fallback for the native ``_falco_kernel`` function and
    mirrors its semantics exactly: uniform stride, mirrored X axis,
    ``z = depth * z_res + z_offset``, optional TURBO colors and bounding box.
    """
    if _kernel_generate_pointcloud is not None:
        cloud = _kernel_generate_pointcloud(
            depth, x_res, y_res, z_res, z_offset, target_points,
            zero_as_invalid, "", compute_bbox,
        )
        if colormap:
            cloud["colors"] = _intensity_colors(cloud["intensity"], colormap)
        return cloud

    depth = np.ascontiguousarray(depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got {depth.ndim}")

    h, w = depth.shape
    total = h * w
    step = 1
    if target_points > 0 and total > target_points:
        step = max(1, int(np.ceil(np.sqrt(total / target_points))))

    sampled = depth[::step, ::step]
    valid = np.isfinite(sampled)
    if zero_as_invalid:
        valid &= sampled != 0.0

    rows, cols = np.nonzero(valid)
    d = sampled[valid]
    count = len(rows)
    points = np.empty((count, 3), dtype=np.float32)
    intensity = d.astype(np.float32, copy=True)
    points[:, 0] = ((w - 1) - cols * step).astype(np.float32) * x_res
    points[:, 1] = (rows * step).astype(np.float32) * y_res
    points[:, 2] = d * np.float32(z_res) + np.float32(z_offset)

    result = {"points": points, "intensity": intensity}

    if colormap:
        result["colors"] = _intensity_colors(intensity, colormap)

    if compute_bbox:
        if count > 0:
            lo = points.min(axis=0)
            hi = points.max(axis=0)
        else:
            lo = np.zeros(3, dtype=np.float32)
            hi = np.ones(3, dtype=np.float32)
        center = (lo + hi) * 0.5
        size = hi - lo
        size = np.maximum(size, 1e-6)
        result["bbox_center"] = center.astype(np.float32)
        result["bbox_size"] = size.astype(np.float32)

    return result


def ensure_pointcloud_data(vimg, target_points=1_000_000, zero_as_invalid=True):
    """Generate or validate 3D point-cloud data for a depth-like VisionImage.

    Uses the image's calibration (scale_x/y/z, offset_z). The result is cached
    on the image via ``_3d_data`` and ``_3d_cache_key`` so repeated calls are
    cheap when calibration has not changed.

    ``target_points`` is the desired upper bound for the number of points.
    Use ``target_points <= 0`` to keep every valid pixel (no subsampling).

    Returns True if the image now has _3d_data, False otherwise.
    """
    if not _is_depth_like(vimg):
        vimg._3d_data = None
        vimg._3d_cache_key = None
        return False

    calib = _calib_tuple(vimg)
    cache_key = (*calib, int(target_points))
    current_key = getattr(vimg, "_3d_cache_key", None)
    if current_key == cache_key and getattr(vimg, "_3d_data", None) is not None:
        return True

    x_res, y_res, z_res, z_offset = calib
    cloud = generate_pointcloud(
        vimg.data, x_res, y_res, z_res, z_offset, target_points,
        zero_as_invalid, colormap="", compute_bbox=False,
    )
    vimg._3d_data = {
        "points": np.asarray(cloud["points"], dtype=np.float32),
        "intensity": np.asarray(cloud["intensity"], dtype=np.float32),
        "colors": None,
    }
    vimg._3d_cache_key = cache_key
    vimg.metadata["_3d_target_points"] = target_points
    return True


def propagate_3d(input_vimg, output_vimg, regenerate=True, target_points=1_000_000, zero_as_invalid=True):
    """Propagate calibration and optional 3D point cloud from input to output.

    If ``regenerate`` is True and the input had a 3D point cloud, the output
    cloud is regenerated from the output's current pixel data. If False, the
    input cloud is copied unchanged (useful for tools that do not modify pixels).
    """
    # Always propagate calibration
    output_vimg.scale_x = float(getattr(input_vimg, "scale_x", 1.0))
    output_vimg.scale_y = float(getattr(input_vimg, "scale_y", 1.0))
    output_vimg.scale_z = float(getattr(input_vimg, "scale_z", 1.0))
    output_vimg.offset_z = float(getattr(input_vimg, "offset_z", 0.0))

    input_3d = getattr(input_vimg, "_3d_data", None)
    if input_3d is None:
        output_vimg._3d_data = None
        output_vimg._3d_cache_key = None
        return

    if not regenerate:
        output_vimg._3d_data = input_3d
        output_vimg._3d_cache_key = getattr(input_vimg, "_3d_cache_key", None)
        return

    tp = int(getattr(input_vimg, "metadata", {}).get("_3d_target_points", target_points))
    ensure_pointcloud_data(output_vimg, tp, zero_as_invalid)


def range_to_pointcloud_metric(depth, x_res=1.0, y_res=1.0, z_res=1.0, z_offset=0.0, step=None, target_points=1_000_000):
    """Convert a 2D float32 depth/range map to a 3D point cloud using
    metric resolution (no pinhole model).

    Each valid pixel maps to a 3D point via simple scaling:
        x = col * x_res
        y = row * y_res
        z = depth[row, col] * z_res + z_offset

    Parameters
    ----------
    x_res : float
        Physical resolution per column (e.g. mm/pixel in X direction).
    y_res : float
        Physical resolution per row (e.g. mm/pixel in Y direction).
    z_res : float
        Scale factor to convert depth values to physical Z (e.g. mm/count).
    z_offset : float
        Offset added to Z after scaling (e.g. depth_range).
    step : int or None
        Subsampling step.  If None, it is chosen automatically so that the
        resulting cloud has roughly ``target_points`` points.
    target_points : int
        Desired upper bound for the number of points when ``step`` is None.

    Returns
    -------
    points : (N, 3) float32
    colors : (N, 3) uint8
    intensity : (N,) float32
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got {depth.ndim}")
    depth = np.ascontiguousarray(depth, dtype=np.float32)

    h, w = depth.shape
    if step is None:
        total = h * w
        step = 1 if total <= target_points else max(1, int(round((total / target_points) ** 0.5)))

    r_vals = np.arange(0, h, step, dtype=np.int32)
    c_vals = np.arange(0, w, step, dtype=np.int32)
    d = depth[r_vals[:, None], c_vals[None, :]]

    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        return empty_pts, np.zeros((0, 3), dtype=np.uint8), np.zeros(0, dtype=np.float32)

    r_grid, c_grid = np.broadcast_arrays(r_vals[:, None], c_vals[None, :])
    r = r_grid[valid]
    c = c_grid[valid]
    # Mirror image columns so that column 0 (left) maps to world +X. This makes
    # the 3D point cloud match the 2D image left/right with the default camera.
    x = (w - 1 - c).astype(np.float32) * x_res
    y = r.astype(np.float32) * y_res
    z = d[valid] * z_res + z_offset

    points = np.column_stack([x, y, z])
    intensity = d[valid].copy()

    lo, hi = float(intensity.min()), float(intensity.max())
    if hi > lo:
        t = np.clip((intensity - lo) / (hi - lo), 0, 1)
    else:
        t = np.zeros_like(intensity)
    r_ch = np.clip(255 * t, 0, 255).astype(np.uint8)
    g_ch = np.clip(255 * (1 - np.abs(2 * t - 1)), 0, 255).astype(np.uint8)
    b_ch = np.clip(255 * (1 - t), 0, 255).astype(np.uint8)
    colors = np.column_stack([r_ch, g_ch, b_ch])

    return points, colors, intensity


def range_to_pointcloud(depth, fx, fy, cx, cy, z_scale=1.0, step=None, target_points=1_000_000):
    """Convert a 2D float32 depth/range map to a 3D point cloud.

    Each valid pixel (finite, non-NaN) projects to a 3D point via the
    standard pinhole model:
        x = (col - cx) * z / fx
        y = (row - cy) * z / fy
        z = depth[row, col] * z_scale

    Parameters
    ----------
    step : int or None
        Subsampling step.  If None, it is chosen automatically so that the
        resulting cloud has roughly ``target_points`` points.
    target_points : int
        Desired upper bound for the number of points when ``step`` is None.

    Returns
    -------
    points : (N, 3) float32
    colors : (N, 3) uint8   â€” range values mapped to heat gradient
    intensity : (N,) float32 â€” raw depth/range value per point
    """
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2D, got {depth.ndim}")
    depth = np.ascontiguousarray(depth, dtype=np.float32)

    h, w = depth.shape
    if step is None:
        total = h * w
        step = 1 if total <= target_points else max(1, int(round((total / target_points) ** 0.5)))

    r_vals = np.arange(0, h, step, dtype=np.int32)
    c_vals = np.arange(0, w, step, dtype=np.int32)
    d = depth[r_vals[:, None], c_vals[None, :]]

    valid = np.isfinite(d) & (d > 0)
    if not valid.any():
        empty_pts = np.zeros((0, 3), dtype=np.float32)
        return empty_pts, np.zeros((0, 3), dtype=np.uint8), np.zeros(0, dtype=np.float32)

    r_grid, c_grid = np.broadcast_arrays(r_vals[:, None], c_vals[None, :])
    r = r_grid[valid]
    c = c_grid[valid]
    z = d[valid] * z_scale
    x = (c.astype(np.float32) - cx) * z / fx
    y = (r.astype(np.float32) - cy) * z / fy

    points = np.column_stack([x, y, z])
    intensity = d[valid].copy()

    # Default colors: map depth to heat-like gradient (full range)
    lo, hi = float(intensity.min()), float(intensity.max())
    if hi > lo:
        t = np.clip((intensity - lo) / (hi - lo), 0, 1)
    else:
        t = np.zeros_like(intensity)
    r_ch = np.clip(255 * t, 0, 255).astype(np.uint8)
    g_ch = np.clip(255 * (1 - np.abs(2 * t - 1)), 0, 255).astype(np.uint8)
    b_ch = np.clip(255 * (1 - t), 0, 255).astype(np.uint8)
    colors = np.column_stack([r_ch, g_ch, b_ch])

    return points, colors, intensity


def load_ply(path):
    """Load an ASCII PLY point cloud file.

    Returns
    -------
    points : (N, 3) float32
    normals : (N, 3) float32 or None
    colors : (N, 3) uint8 or None
    """
    with open(path, "r") as f:
        lines = f.readlines()

    if not lines or lines[0].strip() != "ply":
        raise ValueError("Not a PLY file (missing 'ply' header)")

    fmt = None
    elem_vertex_count = 0
    props = []
    line_no = 1
    while line_no < len(lines):
        line = lines[line_no].strip()
        line_no += 1
        if not line:
            continue
        if line.startswith("format "):
            parts = line.split()
            fmt = " ".join(parts[1:3]) if len(parts) >= 3 else parts[1]
        elif line.startswith("element vertex "):
            elem_vertex_count = int(line.split()[2])
        elif line == "end_header":
            break
        elif line.startswith("property "):
            parts = line.split()
            if len(parts) >= 3:
                props.append((parts[1], parts[2]))
        elif line == "end_header":
            break

    if fmt != "ascii 1.0":
        raise ValueError(f"Only ASCII PLY supported, got format '{fmt}'")

    if elem_vertex_count == 0:
        return np.zeros((0, 3), dtype=np.float32), None, None

    data_start = line_no
    raw_data = []
    for i in range(elem_vertex_count):
        if data_start + i >= len(lines):
            break
        parts = lines[data_start + i].strip().split()
        if len(parts) >= len(props):
            raw_data.append(parts)

    n = len(raw_data)
    points = np.zeros((n, 3), dtype=np.float32)
    normals = None
    colors = None

    x_idx = y_idx = z_idx = -1
    nx_idx = ny_idx = nz_idx = -1
    r_idx = g_idx = b_idx = -1
    for idx, (ptype, pname) in enumerate(props):
        if pname == "x": x_idx = idx
        elif pname == "y": y_idx = idx
        elif pname == "z": z_idx = idx
        elif pname == "nx": nx_idx = idx
        elif pname == "ny": ny_idx = idx
        elif pname == "nz": nz_idx = idx
        elif pname == "red": r_idx = idx
        elif pname == "green": g_idx = idx
        elif pname == "blue": b_idx = idx

    for i in range(n):
        if x_idx >= 0: points[i, 0] = float(raw_data[i][x_idx])
        if y_idx >= 0: points[i, 1] = float(raw_data[i][y_idx])
        if z_idx >= 0: points[i, 2] = float(raw_data[i][z_idx])

    if all(idx >= 0 for idx in (nx_idx, ny_idx, nz_idx)):
        normals = np.zeros((n, 3), dtype=np.float32)
        for i in range(n):
            normals[i, 0] = float(raw_data[i][nx_idx])
            normals[i, 1] = float(raw_data[i][ny_idx])
            normals[i, 2] = float(raw_data[i][nz_idx])

    if all(idx >= 0 for idx in (r_idx, g_idx, b_idx)):
        colors = np.zeros((n, 3), dtype=np.uint8)
        for i in range(n):
            colors[i, 0] = int(raw_data[i][r_idx])
            colors[i, 1] = int(raw_data[i][g_idx])
            colors[i, 2] = int(raw_data[i][b_idx])

    return points, normals, colors


def save_ply(path, points, normals=None, colors=None):
    """Save point cloud to ASCII PLY."""
    n = len(points)
    header = ["ply", "format ascii 1.0",
              f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    has_normals = normals is not None and len(normals) == n
    has_colors = colors is not None and len(colors) == n
    if has_normals:
        header.extend(["property float nx", "property float ny", "property float nz"])
    if has_colors:
        header.extend(["property uchar red", "property uchar green", "property uchar blue"])
    header.append("end_header")

    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        for i in range(n):
            line = f"{points[i,0]} {points[i,1]} {points[i,2]}"
            if has_normals:
                line += f" {normals[i,0]} {normals[i,1]} {normals[i,2]}"
            if has_colors:
                line += f" {int(colors[i,0])} {int(colors[i,1])} {int(colors[i,2])}"
            f.write(line + "\n")


def extract_roi_3d(points, center, size):
    """Filter points inside an axis-aligned bounding box.

    Parameters
    ----------
    points : (N, 3) float32
    center : (3,) â€” box center
    size : (3,) â€” box dimensions (w, h, d)

    Returns
    -------
    indices : (M,) int â€” indices of points inside the box
    """
    half = np.asarray(size, dtype=np.float32) * 0.5
    c = np.asarray(center, dtype=np.float32)
    lo = c - half
    hi = c + half
    mask = np.all((points >= lo) & (points <= hi), axis=1)
    return np.where(mask)[0]


def axis_aligned_bounding_box(points):
    """Compute axis-aligned bounding box for a point cloud.

    Returns (center, size) as (3,) float32 arrays.
    """
    lo = points.min(axis=0)
    hi = points.max(axis=0)
    center = (lo + hi) * 0.5
    size = hi - lo
    size[size < 1e-6] = 1e-6
    return center, size


class VoxelIndex:
    """Simple uniform voxel grid for fast spatial queries on large point clouds.

    This is intentionally lightweight: it stores one representative point index
    per occupied voxel. Useful for nearest-neighbor / region queries when a
    full KD-tree would be too expensive to build or too heavy in memory.
    """

    def __init__(self, points, voxel_size=None):
        self.points = np.asarray(points, dtype=np.float32)
        if len(self.points) == 0:
            self.origin = np.zeros(3, dtype=np.float32)
            self.voxel_size = float(voxel_size) if voxel_size else 1.0
            self._grid = {}
            return

        lo = self.points.min(axis=0)
        hi = self.points.max(axis=0)
        size = hi - lo
        size[size < 1e-6] = 1e-6

        if voxel_size is None:
            # Aim for roughly one voxel per ~1000 points.
            target_cells = max(1, len(self.points) // 1000)
            volume = size.prod()
            voxel_size = max(1e-6, (volume / target_cells) ** (1.0 / 3.0))

        self.voxel_size = float(voxel_size)
        self.origin = lo

        # Map each point to its voxel; keep the first point in each voxel.
        coords = np.floor((self.points - self.origin) / self.voxel_size).astype(np.int64)
        keys = (coords[:, 0] << 42) | (coords[:, 1] << 21) | coords[:, 2]
        _, unique_idx = np.unique(keys, return_index=True)

        self._grid = {}
        for idx in unique_idx:
            key = int(keys[idx])
            self._grid[key] = int(idx)

    def _key(self, pt):
        c = np.floor((np.asarray(pt, dtype=np.float32) - self.origin) / self.voxel_size).astype(np.int64)
        return int((c[0] << 42) | (c[1] << 21) | c[2])

    def nearest(self, pt):
        """Return the index of the closest representative point, or -1."""
        if not self._grid:
            return -1
        center_key = self._key(pt)
        # Search in a 3x3x3 neighborhood around the query voxel.
        best_idx = -1
        best_d2 = float("inf")
        cx = (center_key >> 42) & 0x1FFFFF
        cy = (center_key >> 21) & 0x1FFFFF
        cz = center_key & 0x1FFFFF
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    key = ((cx + dx) << 42) | ((cy + dy) << 21) | (cz + dz)
                    idx = self._grid.get(key)
                    if idx is None:
                        continue
                    d2 = float(np.sum((self.points[idx] - pt) ** 2))
                    if d2 < best_d2:
                        best_d2 = d2
                        best_idx = idx
        return best_idx

    def radius_search(self, pt, radius):
        """Return indices of representative points within radius of pt."""
        if not self._grid:
            return np.array([], dtype=np.int64)
        center_key = self._key(pt)
        r = int(np.ceil(radius / self.voxel_size))
        cx = (center_key >> 42) & 0x1FFFFF
        cy = (center_key >> 21) & 0x1FFFFF
        cz = center_key & 0x1FFFFF
        result = []
        r2 = radius * radius
        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                for dz in range(-r, r + 1):
                    key = ((cx + dx) << 42) | ((cy + dy) << 21) | (cz + dz)
                    idx = self._grid.get(key)
                    if idx is None:
                        continue
                    d2 = float(np.sum((self.points[idx] - pt) ** 2))
                    if d2 <= r2:
                        result.append(idx)
        return np.array(result, dtype=np.int64)


__all__ = [
    "range_to_pointcloud", "range_to_pointcloud_metric",
    "load_ply", "save_ply",
    "extract_roi_3d", "axis_aligned_bounding_box",
    "VoxelIndex",
]