"""FalcoCV calibrated raw-depth container (``.fcd`` v3).

The container stores a 2D raw depth array as ``uint16`` and the calibration
needed to convert it to millimetres.  The payload is compressed with LZ4; the
JSON header stores shape, dtype, raw-depth calibration and provenance.

FCD v1/v2 are intentionally not supported anymore.
"""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import numpy as np

try:
    import imagecodecs
except Exception:  # pragma: no cover - optional dependency fallback
    imagecodecs = None  # type: ignore[assignment]


MAGIC = b"FCD3"
VERSION = 3
_HEADER_LENGTH = struct.Struct("<I")
_CALIBRATION_KEYS = ("x_res", "y_res", "z_scale", "z_offset")


def _require_imagecodecs() -> None:
    if imagecodecs is None:
        raise RuntimeError(
            "imagecodecs is required to read/write .fcd depth files"
        )


def _normalise_metadata(metadata) -> dict:
    meta = dict(metadata or {})
    calibration = dict(meta.pop("calibration", {}) or {})
    for key in _CALIBRATION_KEYS:
        if key in meta:
            calibration[key] = meta.pop(key)
    for key, default in (
        ("x_res", 1.0),
        ("y_res", 1.0),
        ("z_scale", 1.0),
        ("z_offset", 0.0),
    ):
        calibration.setdefault(key, default)
    meta["calibration"] = {
        key: float(calibration[key]) for key in _CALIBRATION_KEYS
    }
    meta.setdefault("value_type", "raw_counts")
    meta.setdefault("unit", "mm")
    meta.setdefault("invalid_value", 0)
    return meta


def save_depth_fcd(path, data: np.ndarray, metadata=None, level=None) -> Path:
    """Write a raw uint16 depth image and its calibration to ``.fcd`` v3.

    ``data`` must be a 2D uint16 raw-depth image.  Invalid samples are 0.  The
    millimetre Z value is reconstructed as
    ``raw * z_scale + z_offset``.
    """
    _require_imagecodecs()
    arr = np.ascontiguousarray(data)
    if arr.ndim != 2:
        raise ValueError(f".fcd requires a 2D array, got shape {arr.shape}")
    if arr.dtype != np.uint16:
        raise ValueError(f".fcd v3 requires uint16 raw depth, got {arr.dtype}")

    payload = imagecodecs.lz4_encode(arr, level=level, header=True)
    header = {
        "version": VERSION,
        "shape": [int(v) for v in arr.shape],
        "dtype": arr.dtype.str,
        "compression": "lz4",
        "payload_size": len(payload),
        **_normalise_metadata(metadata),
    }
    header_bytes = json.dumps(
        header, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with open(tmp, "wb") as handle:
            handle.write(MAGIC)
            handle.write(_HEADER_LENGTH.pack(len(header_bytes)))
            handle.write(header_bytes)
            handle.write(payload)
        os.replace(tmp, path)
    except Exception:
        try:
            tmp.unlink()
        except FileNotFoundError:
            pass
        raise
    return path


def _read_header_payload(path) -> tuple[dict, bytes]:
    path = Path(path)
    with open(path, "rb") as handle:
        magic = handle.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"Not a FalcoCV depth v3 file: {path}")
        length_bytes = handle.read(_HEADER_LENGTH.size)
        if len(length_bytes) != _HEADER_LENGTH.size:
            raise ValueError(f"Truncated .fcd header: {path}")
        (header_length,) = _HEADER_LENGTH.unpack(length_bytes)
        header_bytes = handle.read(header_length)
        if len(header_bytes) != header_length:
            raise ValueError(f"Truncated .fcd header: {path}")
        payload = handle.read()

    try:
        header = json.loads(header_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid .fcd header: {path}") from exc
    if int(header.get("version", -1)) != VERSION:
        raise ValueError(
            f"Unsupported .fcd version: {header.get('version')!r}; "
            "only FCD v3 is supported"
        )
    if header.get("compression") != "lz4":
        raise ValueError(
            f"Unsupported .fcd compression: {header.get('compression')!r}"
        )
    expected_size = int(header.get("payload_size", -1))
    if expected_size != len(payload):
        raise ValueError(
            f"Truncated .fcd payload: expected {expected_size}, got {len(payload)}"
        )
    return header, payload


def load_depth_fcd_with_metadata(path) -> tuple[np.ndarray, dict]:
    """Read a ``.fcd`` v3 file and return ``(raw_uint16, metadata)``."""
    _require_imagecodecs()
    header, payload = _read_header_payload(path)

    try:
        shape = tuple(int(v) for v in header["shape"])
        dtype = np.dtype(header["dtype"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid .fcd shape/dtype: {path}") from exc
    if len(shape) != 2 or any(v < 0 for v in shape):
        raise ValueError(f"Invalid .fcd shape: {shape}")
    if dtype != np.uint16:
        raise ValueError(f".fcd v3 requires uint16 raw depth, got {dtype}")

    try:
        decoded = imagecodecs.lz4_decode(payload, header=True)
    except Exception as exc:
        raise ValueError(f"Could not decompress .fcd file: {path}") from exc

    expected_bytes = int(np.prod(shape, dtype=np.int64)) * dtype.itemsize
    if len(decoded) != expected_bytes:
        raise ValueError(
            f"Invalid .fcd payload size: expected {expected_bytes}, "
            f"got {len(decoded)}"
        )
    arr = np.frombuffer(decoded, dtype=dtype).reshape(shape)
    return arr, header


def load_depth_fcd(path) -> np.ndarray:
    """Read a ``.fcd`` v3 file and return its raw uint16 depth array."""
    arr, _ = load_depth_fcd_with_metadata(path)
    return arr
