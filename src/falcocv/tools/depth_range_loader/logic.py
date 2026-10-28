import os
import collections
import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.pointcloud_utils import generate_pointcloud

_CACHE_MAX = 4
_LOAD_CACHE: dict[str, np.ndarray] = {}
_LOAD_KEYS: collections.OrderedDict = collections.OrderedDict()

# Optional fast TIFF reader (often 2-3x faster than cv2.imread on compressed TIFFs).
try:
    import tifffile
    _HAS_TIFFFILE = True
except Exception:
    tifffile = None  # type: ignore[assignment]
    _HAS_TIFFFILE = False


def _cache_get(path: str):
    arr = _LOAD_CACHE.get(path)
    if arr is not None:
        _LOAD_KEYS.move_to_end(path)
    return arr


def _cache_put(path: str, arr: np.ndarray):
    if path in _LOAD_CACHE:
        _LOAD_KEYS.move_to_end(path)
        return
    if len(_LOAD_CACHE) >= _CACHE_MAX:
        oldest = next(iter(_LOAD_KEYS))
        del _LOAD_CACHE[oldest]
        del _LOAD_KEYS[oldest]
    _LOAD_CACHE[path] = arr
    _LOAD_KEYS[path] = None


def _cache_clear():
    _LOAD_CACHE.clear()
    _LOAD_KEYS.clear()


_VALID_EXTS = {".tif", ".tiff", ".png", ".bmp", ".jpg", ".jpeg"}


def _load_depth_raw(path):
    raw = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if raw is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    if raw.ndim == 3:
        if raw.shape[2] >= 3:
            raw = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        else:
            raw = raw[:, :, 0]
    return raw


def _load_depth_tifffile(path):
    """Load a TIFF faster than cv2.imread for many compression schemes."""
    if tifffile is None:
        raise RuntimeError("tifffile is not installed")
    raw = tifffile.imread(path)
    if raw is None:
        raise RuntimeError(f"Failed to decode image with tifffile: {path}")
    # tifffile may return a (H, W, 1) array for some single-channel TIFFs.
    if raw.ndim == 3:
        if raw.shape[2] == 1:
            raw = raw[:, :, 0]
        elif raw.shape[2] >= 3:
            # Convert to grayscale. Use standard luminance weights.
            raw = (
                raw[:, :, 0] * 0.299
                + raw[:, :, 1] * 0.587
                + raw[:, :, 2] * 0.114
            ).astype(raw.dtype)
        else:
            raw = raw[:, :, 0]
    return raw


def _convert_depth(raw, scale):
    if raw.dtype.kind in "iu":
        d = raw.astype(np.float32)
        d *= np.float32(scale)
    elif raw.dtype.kind == "f":
        d = raw.astype(np.float32)
        d *= np.float32(scale)
    else:
        raise RuntimeError(f"Unsupported depth dtype: {raw.dtype}")
    return d


@register_tool(prefix="[DPTH]", default_name="Depth Range Loader", group="Acquisition")
class DepthRangeLoader(BaseTool):
    """Loads single-channel range / depth images (TIFF, PNG, etc.) as float32.
    Supports single-file and folder-stack modes.
    3D point cloud is generated eagerly in C++ when generate_3d is enabled."""

    def __init__(self, name="Depth Range Loader"):
        super().__init__(name)
        self.outputs = {"output image": None}
        self.params = {
            "mode": "File",
            "file_path": "",
            "folder_path": "",
            "sort_by": "Name",
            "current_index": 0,
            "delay": 0,
            "scale": 1.0,
            "zero_as_invalid": True,
            "x_res": 1.0,
            "y_res": 1.0,
            "z_res": 1.0,
            "z_offset": 0.0,
            "generate_3d": False,
            "target_points": 1_000_000,
            "show_all_points": False,
        }
        self._image_files: list[str] = []

    def _refresh_folder(self):
        self._image_files = []
        path = self.params.get("folder_path", "")
        if not path or not os.path.exists(path):
            return
        for f in os.listdir(path):
            if os.path.splitext(f)[1].lower() in _VALID_EXTS:
                self._image_files.append(os.path.join(path, f))
        sort_by = self.params.get("sort_by", "Name")
        if sort_by == "Name":
            self._image_files.sort()
        elif sort_by == "Date":
            self._image_files.sort(key=os.path.getmtime)
        idx = self.params.get("current_index", 0)
        if self._image_files and idx >= len(self._image_files):
            self.params["current_index"] = 0

    def _resolve_path(self, advance=False):
        mode = self.params.get("mode", "File")
        if mode == "Folder":
            if not self._image_files:
                self._refresh_folder()
            if not self._image_files:
                raise FileNotFoundError("No images found in folder")
            if advance:
                idx = self.params.get("current_index", 0) + 1
                if idx >= len(self._image_files):
                    idx = 0
                self.params["current_index"] = idx
            return self._image_files[self.params.get("current_index", 0)]
        path = self.params.get("file_path", "")
        if not path:
            raise FileNotFoundError("No file specified")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        return path

    def run(self, advance=False):
        self.outputs["output image"] = None
        path = self._resolve_path(advance)

        scale = float(self.params.get("scale", 1.0))
        zero_invalid = bool(self.params.get("zero_as_invalid", False))
        generate_3d = bool(self.params.get("generate_3d", False))
        x_res = float(self.params.get("x_res", 1.0))
        y_res = float(self.params.get("y_res", 1.0))
        z_res = float(self.params.get("z_res", 1.0))
        z_offset = float(self.params.get("z_offset", 0.0))
        show_all_points = bool(self.params.get("show_all_points", False))
        target_points = int(self.params.get("target_points", 1_000_000))
        effective_target_points = 0 if show_all_points else target_points

        # Prefer tifffile for TIFFs: often much faster on compressed files.
        ext = os.path.splitext(path)[1].lower()
        use_tifffile = _HAS_TIFFFILE and ext in {".tif", ".tiff"}

        raw = _cache_get(path)
        if raw is None:
            if use_tifffile:
                try:
                    raw = _load_depth_tifffile(path)
                except Exception as e:
                    print(f"tifffile decode failed ({e}), falling back to cv2.imread")
                    raw = _load_depth_raw(path)
            else:
                raw = _load_depth_raw(path)
            _cache_put(path, raw)

        d = _convert_depth(raw, scale)

        # Zero-as-invalid handling is done at cloud generation time. The
        # depth map for display keeps 0 values so the viewer shows a normal
        # colormap instead of rendering invalid pixels as black.
        d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        vimg = VisionImage(d)
        vimg.scale_x = x_res
        vimg.scale_y = y_res
        vimg.scale_z = z_res
        vimg.offset_z = z_offset

        if generate_3d:
            cloud = generate_pointcloud(
                d, x_res, y_res, z_res, z_offset,
                effective_target_points, zero_invalid, "TURBO", True,
            )
            vimg._3d_data = {
                "points": np.asarray(cloud["points"], dtype=np.float32),
                "intensity": np.asarray(cloud["intensity"], dtype=np.float32),
                "colors": np.asarray(cloud["colors"], dtype=np.uint8) if "colors" in cloud else None,
                "bbox_center": np.asarray(cloud["bbox_center"], dtype=np.float32) if "bbox_center" in cloud else None,
                "bbox_size": np.asarray(cloud["bbox_size"], dtype=np.float32) if "bbox_size" in cloud else None,
            }
            vimg._3d_cache_key = (x_res, y_res, z_res, z_offset, effective_target_points)
            vimg.metadata["_3d_target_points"] = effective_target_points

        self.outputs["output image"] = vimg

    def serialize(self, job_manager=None) -> dict:
        data = super().serialize(job_manager)
        data["params"] = {k: v for k, v in data["params"].items() if k != "current_index"}
        return data

    def deserialize(self, data: dict, job_manager=None):
        super().deserialize(data, job_manager)
        self.params["current_index"] = 0
        self._image_files = []
        if self.params.get("mode") == "Folder":
            self._refresh_folder()

    def get_config_widget(self):
        from .config import DepthRangeLoaderConfig
        return DepthRangeLoaderConfig(self)
