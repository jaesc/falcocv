"""Convert TIFF depth frames to the FalcoCV ``.fcd`` LZ4 container."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from falcocv.core.depth_format import load_depth_fcd, save_depth_fcd

try:
    import imagecodecs
except Exception:  # pragma: no cover
    imagecodecs = None

try:
    import tifffile
except Exception:  # pragma: no cover
    tifffile = None


def _as_depth_array(raw):
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
    return (
        raw[:, :, 0] * 0.299
        + raw[:, :, 1] * 0.587
        + raw[:, :, 2] * 0.114
    ).astype(raw.dtype)


def _decode_tiff(path: Path) -> np.ndarray:
    errors = []
    if imagecodecs is not None:
        try:
            return _as_depth_array(
                imagecodecs.tiff_decode(path.read_bytes())
            )
        except Exception as exc:
            errors.append(f"imagecodecs: {exc}")
    raw = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if raw is not None:
        return _as_depth_array(raw)
    errors.append("cv2: decode failed")
    if tifffile is not None:
        try:
            return _as_depth_array(tifffile.imread(str(path)))
        except Exception as exc:
            errors.append(f"tifffile: {exc}")
    raise RuntimeError(f"Could not decode {path}: {'; '.join(errors)}")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, type=Path, help="folder with source TIFFs"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="destination folder"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="replace existing .fcd files"
    )
    parser.add_argument(
        "--verify", action=argparse.BooleanOptionalAction, default=True,
        help="decode each output and compare it with the source",
    )
    parser.add_argument(
        "--level", type=int, default=None,
        help="optional LZ4 compression level",
    )
    args = parser.parse_args(argv)

    source_dir = args.input.resolve()
    output_dir = args.output.resolve()
    if not source_dir.is_dir():
        parser.error(f"input folder does not exist: {source_dir}")

    sources = sorted(
        p for p in source_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in {".tif", ".tiff"}
    )
    if not sources:
        parser.error(f"no TIFF files found in {source_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    converted = 0
    skipped = 0
    total_source_bytes = 0
    total_output_bytes = 0

    for source in sources:
        relative = source.relative_to(source_dir).with_suffix(".fcd")
        destination = output_dir / relative
        if destination.exists() and not args.overwrite:
            skipped += 1
            continue

        raw = _decode_tiff(source)
        if raw is None:
            raise RuntimeError(f"decoded image is empty: {source}")
        save_depth_fcd(destination, raw, level=args.level)
        if args.verify:
            restored = load_depth_fcd(destination)
            if not np.array_equal(restored, raw):
                raise RuntimeError(f"verification failed: {destination}")

        converted += 1
        total_source_bytes += source.stat().st_size
        total_output_bytes += destination.stat().st_size
        print(
            f"[{converted}/{len(sources)}] {relative} "
            f"{raw.shape} {raw.dtype} "
            f"{source.stat().st_size / 1e6:.1f} -> "
            f"{destination.stat().st_size / 1e6:.1f} MB"
        )

    elapsed = time.perf_counter() - started
    print(
        f"converted={converted} skipped={skipped} "
        f"source={total_source_bytes / 1e6:.1f} MB "
        f"output={total_output_bytes / 1e6:.1f} MB "
        f"time={elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
