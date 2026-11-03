import os

import cv2
import numpy as np

from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.pointcloud_utils import generate_pointcloud
from falcocv.core.depth_format import load_depth_fcd

# imagecodecs is the fastest decoder for the large LZW TIFFs.  tifffile is
# kept as the last fallback for TIFF variants or environments without it.
try:
    import imagecodecs
    _HAS_IMAGECODECS = True
except Exception:
    imagecodecs = None  # type: ignore[assignment]
    _HAS_IMAGECODECS = False

try:
    import tifffile
    _HAS_TIFFFILE = True
except Exception:
    tifffile = None  # type: ignore[assignment]
    _HAS_TIFFFILE = False


_VALID_EXTS = {".tif", ".tiff", ".fcd", ".png", ".bmp", ".jpg", ".jpeg"}


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


def _as_depth_array(raw):
    """Normalise channel layouts returned by different TIFF decoders."""
    if raw is None:
        return None
    if raw.ndim != 3:
        return raw
    if raw.shape[2] == 1:
        return raw[:, :, 0]
    if raw.shape[2] < 3:
        return raw[:, :, 0]
    if raw.dtype == np.uint8:
        return cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
    # Preserve the source dtype while using standard luminance weights.
    return (
        raw[:, :, 0] * 0.299
        + raw[:, :, 1] * 0.587
        + raw[:, :, 2] * 0.114
    ).astype(raw.dtype)


def _load_depth_imagecodecs(path):
    """Decode a TIFF through imagecodecs/libtiff."""
    if imagecodecs is None:
        raise RuntimeError("imagecodecs is not installed")
    with open(path, "rb") as handle:
        raw = imagecodecs.tiff_decode(handle.read())
    if raw is None:
        raise RuntimeError(f"Failed to decode TIFF with imagecodecs: {path}")
    return _as_depth_array(raw)


def _load_depth_tifffile(path):
    """Load a TIFF through tifffile as a final fallback."""
    if tifffile is None:
        raise RuntimeError("tifffile is not installed")
    raw = tifffile.imread(path)
    if raw is None:
        raise RuntimeError(f"Failed to decode image with tifffile: {path}")
    return _as_depth_array(raw)


def _load_tiff_depth(path):
    """Decode a TIFF using the fastest available backend."""
    errors = []
    if _HAS_IMAGECODECS:
        try:
            return _load_depth_imagecodecs(path)
        except Exception as exc:
            errors.append(f"imagecodecs: {exc}")
    try:
        return _load_depth_raw(path)
    except Exception as exc:
        errors.append(f"cv2: {exc}")
    if _HAS_TIFFFILE:
        try:
            return _load_depth_tifffile(path)
        except Exception as exc:
            errors.append(f"tifffile: {exc}")
    detail = "; ".join(errors) if errors else "no decoder available"
    raise RuntimeError(f"Failed to decode TIFF {path}: {detail}")


def _convert_depth(raw, scale):
    scale = np.float32(scale)
    if raw.dtype.kind in "iu":
        # OpenCV's SIMD path performs the uint16 -> float32 conversion and
        # scaling in one pass.  It is faster than astype + multiply and keeps
        # the same exact float32 values for the supported integer dtypes.
        try:
            return cv2.addWeighted(
                raw, float(scale), raw, 0.0, 0.0, dtype=cv2.CV_32F
            )
        except cv2.error:
            pass
        d = raw.astype(np.float32)
        if scale != 1.0:
            d *= scale
    elif raw.dtype.kind == "f":
        # Preserve an existing float32 buffer when no scaling is needed.
        d = np.asarray(raw, dtype=np.float32)
        if scale != 1.0:
            d = d * scale
    else:
        raise RuntimeError(f"Unsupported depth dtype: {raw.dtype}")
    return d


@register_tool(prefix="[DPTH]", default_name="Depth Range Loader", group="Acquisition")
class DepthRangeLoader(BaseTool):
    """Loads single-channel range / depth images (FCD, TIFF, PNG, etc.) as float32.
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

        # imagecodecs decodes the large LZW TIFFs fastest; cv2 and tifffile
        # remain fallbacks.  .fcd is a native raw-uint16 + LZ4 container.
        ext = os.path.splitext(path)[1].lower()

        if ext == ".fcd":
            raw = load_depth_fcd(path)
        elif ext in {".tif", ".tiff"}:
            raw = _load_tiff_depth(path)
        else:
            raw = _load_depth_raw(path)
        if raw is None:
            raise RuntimeError(f"Failed to decode image: {path}")

        d = _convert_depth(raw, scale)

        # Zero-as-invalid handling is done at cloud generation time. The
        # depth map for display keeps 0 values so the viewer shows a normal
        # colormap instead of rendering invalid pixels as black.
        # Integer sources cannot contain NaN/Inf and are by far the common
        # case for range images; avoid a full float64 sanitisation pass.
        if raw.dtype.kind == "f" and not np.isfinite(d).all():
            d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)

        vimg = VisionImage(d)
        vimg.scale_x = x_res
        vimg.scale_y = y_res
        vimg.scale_z = z_res
        vimg.offset_z = z_offset

        if generate_3d:
            cloud = generate_pointcloud(
                d, x_res, y_res, z_res, z_offset,
                effective_target_points, zero_invalid, "", True,
            )
            vimg._3d_data = {
                "points": np.asarray(cloud["points"], dtype=np.float32),
                "intensity": np.asarray(cloud["intensity"], dtype=np.float32),
                # Colors are generated lazily by the 3D viewer.
                "colors": None,
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
