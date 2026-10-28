"""Depth / range image visualization helpers (colormap + value range)."""
from __future__ import annotations

import cv2
import numpy as np

# OpenCV colormaps exposed in the UI (name -> cv2.COLORMAP_*)
COLORMAP_NAMES: list[str] = [
    "TURBO",
    "JET",
    "VIRIDIS",
    "PLASMA",
    "INFERNO",
    "MAGMA",
    "CIVIDIS",
    "HOT",
    "COOL",
    "BONE",
    "HSV",
    "OCEAN",
    "RAINBOW",
    "PARULA",
    "TWILIGHT",
    "TWILIGHT_SHIFTED",
    "DEEPGREEN",
    "AUTUMN",
    "WINTER",
    "SPRING",
    "SUMMER",
]


def resolve_colormap_id(name: str) -> int:
    key = (name or "TURBO").strip().upper()
    attr = f"COLORMAP_{key}"
    cid = getattr(cv2, attr, None)
    if cid is None:
        return cv2.COLORMAP_TURBO
    return int(cid)


def apply_depth_colormap(
    depth: np.ndarray,
    colormap_name: str,
    vmin: float | None,
    vmax: float | None,
) -> np.ndarray:
    """
    Map a 2D depth/range array to BGR uint8 using OpenCV colormap.
    Non-finite values render as black.
    """
    d = np.asarray(depth)
    if d.ndim != 2:
        raise ValueError("depth must be 2D")
    if d.size == 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    work = d.astype(np.float64, copy=False)
    valid = np.isfinite(work)
    if not np.any(valid):
        return np.zeros((d.shape[0], d.shape[1], 3), dtype=np.uint8)

    lo = float(np.min(work[valid])) if vmin is None else float(vmin)
    hi = float(np.max(work[valid])) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-9

    norm = (work - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    norm_u8 = (norm * 255.0).astype(np.uint8)
    norm_u8[~valid] = 0

    cmap_id = resolve_colormap_id(colormap_name)
    bgr = cv2.applyColorMap(norm_u8, cmap_id)
    bgr[~valid] = (0, 0, 0)
    return bgr


def depth_to_gray_u8(
    depth: np.ndarray,
    vmin: float | None,
    vmax: float | None,
) -> np.ndarray:
    """Linear normalize 2D depth to uint8 grayscale (no colormap)."""
    d = np.asarray(depth)
    if d.ndim != 2:
        raise ValueError("depth must be 2D")
    work = d.astype(np.float64, copy=False)
    valid = np.isfinite(work)
    if not np.any(valid):
        return np.zeros(d.shape, dtype=np.uint8)

    lo = float(np.min(work[valid])) if vmin is None else float(vmin)
    hi = float(np.max(work[valid])) if vmax is None else float(vmax)
    if hi <= lo:
        hi = lo + 1e-9
    norm = (work - lo) / (hi - lo)
    norm = np.clip(norm, 0.0, 1.0)
    out = (norm * 255.0).astype(np.uint8)
    out[~valid] = 0
    return out


def should_offer_depth_controls(arr: np.ndarray, port_name: str | None = None) -> bool:
    """Whether the workflow viewer should show depth colormap / range controls."""
    if not isinstance(arr, np.ndarray) or arr.ndim != 2:
        return False
    k = arr.dtype.kind
    if k == "f":
        return True
    if k in "iu" and arr.itemsize >= 2:
        return True
    if k == "u" and arr.itemsize == 1:
        hints = ("depth", "range", "z_map", "height", "disparity", "distance")
        pn = (port_name or "").lower()
        return any(h in pn for h in hints)
    return False


def map_depth_to_rgb(values, cmap_name="TURBO", vmin=None, vmax=None):
    """Map a float array to an (N, 3) uint8 RGB image using an OpenCV colormap.

    Accepts any shape; the output is flattened to (N, 3) with N = values.size.
    Values outside [vmin, vmax] are clamped. NaNs are treated as vmin.
    """
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0:
        return np.zeros((0, 3), dtype=np.uint8)

    finite = np.isfinite(values)
    if vmin is None:
        vmin = float(values[finite].min()) if finite.any() else 0.0
    if vmax is None:
        vmax = float(values[finite].max()) if finite.any() else 1.0

    span = vmax - vmin
    if span <= 0:
        u8 = np.zeros(values.shape, dtype=np.uint8)
    else:
        safe = np.where(finite, values, vmin)
        u8 = np.clip((safe - vmin) / span * 255.0, 0, 255).astype(np.uint8)

    cmap_id = resolve_colormap_id(str(cmap_name))
    colored_bgr = cv2.applyColorMap(u8.reshape(1, -1), cmap_id)
    colored_rgb = cv2.cvtColor(colored_bgr, cv2.COLOR_BGR2RGB).reshape(-1, 3)

    colored_rgb[~finite.reshape(-1)] = 0
    return colored_rgb
