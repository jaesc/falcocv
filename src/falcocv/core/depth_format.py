"""FalcoCV depth container (``.fcd``).

The container stores a raw 2D depth array compressed with LZ4.  It is not a
TIFF: TIFF has no standard LZ4 compression code.  The small JSON header keeps
the array shape and dtype self-describing, without a sidecar file.
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


MAGIC = b"FCD1"
VERSION = 1
_HEADER_LENGTH = struct.Struct("<I")


def _require_imagecodecs() -> None:
    if imagecodecs is None:
        raise RuntimeError(
            "imagecodecs is required to read/write .fcd depth files"
        )


def save_depth_fcd(path, data: np.ndarray, level=None) -> Path:
    """Write a 2D depth array to a ``.fcd`` file.

    ``data`` is kept as-is (no dtype conversion).  The payload is compressed
    with LZ4 and the header records shape/dtype so it can be restored without
    external metadata.
    """
    _require_imagecodecs()
    arr = np.ascontiguousarray(data)
    if arr.ndim != 2:
        raise ValueError(f".fcd requires a 2D array, got shape {arr.shape}")

    payload = imagecodecs.lz4_encode(arr, level=level, header=True)
    header = {
        "version": VERSION,
        "shape": [int(v) for v in arr.shape],
        "dtype": arr.dtype.str,
        "compression": "lz4",
        "payload_size": len(payload),
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


def load_depth_fcd(path) -> np.ndarray:
    """Read a ``.fcd`` file and return its 2D depth array."""
    _require_imagecodecs()
    path = Path(path)
    with open(path, "rb") as handle:
        magic = handle.read(len(MAGIC))
        if magic != MAGIC:
            raise ValueError(f"Not a FalcoCV depth file: {path}")
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
            f"Unsupported .fcd version: {header.get('version')!r}"
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

    try:
        shape = tuple(int(v) for v in header["shape"])
        dtype = np.dtype(header["dtype"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Invalid .fcd shape/dtype: {path}") from exc
    if len(shape) != 2 or any(v < 0 for v in shape):
        raise ValueError(f"Invalid .fcd shape: {shape}")

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
    arr = np.frombuffer(decoded, dtype=dtype)
    return arr.reshape(shape)
