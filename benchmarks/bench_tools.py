"""Benchmark current C++ tool implementations against the pure-Python legacy
implementations recovered from git history.

Run from the repository root:
    py -3.13 benchmarks/bench_tools.py

Legacy sources are extracted from pinned commits into benchmarks/.legacy/
(gitignored). Both variants run against the current core wrappers, so the
comparison isolates the compute implementation.
"""
from __future__ import annotations

import importlib
import os
import pathlib
import statistics
import subprocess
import sys
import time

import cv2
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
LEGACY = ROOT / "benchmarks" / ".legacy"
LEGACY.mkdir(parents=True, exist_ok=True)

LEGACY_SPECS = {
    # pure-Python implementations (parents of the C++ migrations)
    "hist_pure.py": ("eded9ca^", "tools/histogram/logic.py"),
    "blob_py.py": ("2f04f1a^", "tools/blob_detector/logic.py"),
    "imgcalc_pure.py": ("eded9ca^", "tools/image_calculator/logic.py"),
    "filters_py.py": ("a386d5e^", "tools/filters/logic.py"),
    "edge_py.py": ("a386d5e^", "tools/edge_detection/logic.py"),
    "pattern_py.py": ("d421b8a^", "tools/pattern_matcher/logic.py"),
    "pointcloud_py.py": ("bb155aa^", "core/pointcloud_utils.py"),
}


def extract_legacy() -> None:
    for name, (rev, path) in LEGACY_SPECS.items():
        dst = LEGACY / name
        if dst.exists():
            continue
        data = subprocess.check_output(
            ["git", "show", f"{rev}:{path}"], cwd=ROOT
        )
        dst.write_bytes(data)
    sys.path.insert(0, str(LEGACY))


def bench(fn, n=10, warm=2):
    for _ in range(warm):
        fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append((time.perf_counter() - t0) * 1000.0)
    times.sort()
    return {
        "median": statistics.median(times),
        "min": times[0],
        "p95": times[min(len(times) - 1, int(len(times) * 0.95))],
    }


def fmt(r):
    return f"{r['median']:8.2f} ms"


def load_image(name):
    img = cv2.imread(str(ROOT / name))
    if img is None:
        raise SystemExit(f"missing test image: {name}")
    return img


def main() -> None:
    extract_legacy()

    from falcocv.core.vision_image import VisionImage
    from falcocv.tools.histogram.logic import HistogramTool
    from falcocv.tools.blob_detector.logic import BlobDetector
    from falcocv.tools.image_calculator.logic import ImageCalculatorTool
    from falcocv.tools.filters.logic import FiltersTool
    from falcocv.tools.edge_detection.logic import EdgeDetection
    from falcocv.tools.pattern_matcher.logic import PatternMatcher

    hist_py = importlib.import_module("hist_pure").HistogramTool
    blob_py = importlib.import_module("blob_py").BlobDetector
    imgcalc_py = importlib.import_module("imgcalc_pure").ImageCalculatorTool
    filters_py = importlib.import_module("filters_py").FiltersTool
    edge_py = importlib.import_module("edge_py").EdgeDetection
    pattern_py = importlib.import_module("pattern_py").PatternMatcher

    images = {
        "5MP": load_image("test_pictures/barcode_1D/1.jpg"),
    }
    # The 20 MP sample is an optional local benchmark image (customer images are
    # not distributed with the repository).
    large = os.environ.get("FALCOCV_BENCH_IMAGE")
    if large and os.path.isfile(large):
        images["20MP"] = load_image(large)
    else:
        print("[skip] 20 MP sample: set FALCOCV_BENCH_IMAGE to a large image")

    results = []

    def add(name, size, label, cpp, py):
        if cpp is None or py is None:
            results.append((name, size, label, cpp, py, None))
            return
        ratio = py["median"] / cpp["median"] if cpp["median"] > 0 else 0
        results.append((name, size, label, cpp, py, ratio))

    for size, img in images.items():
        h, w = img.shape[:2]
        v = VisionImage(img)

        # Histogram (stats only; plot disabled for both)
        hist_cpp = HistogramTool()
        hist_cpp.inputs["input image"] = v
        hist_cpp.params["enable_output_image"] = False
        hist_legacy = hist_py()
        hist_legacy.inputs["input image"] = v
        hist_legacy.params["enable_output_image"] = False
        add("Histogram", size, "stats", bench(hist_cpp.run), bench(hist_legacy.run))

        # Blob detector
        blob_cpp = BlobDetector()
        blob_cpp.inputs["input image"] = v
        blob_legacy = blob_py()
        blob_legacy.inputs["input image"] = v
        add("Blob", size, "threshold+contours", bench(blob_cpp.run), bench(blob_legacy.run))

        # Image calculator (add)
        calc_cpp = ImageCalculatorTool()
        calc_cpp.inputs["image_A"] = v
        calc_cpp.inputs["image_B"] = VisionImage(img)
        calc_cpp.params["ops"] = [{"enabled": True, "type": "add"}]
        calc_legacy = imgcalc_py()
        calc_legacy.inputs["image_A"] = v
        calc_legacy.inputs["image_B"] = VisionImage(img)
        calc_legacy.params["ops"] = [{"enabled": True, "type": "add"}]
        add("ImageCalc", size, "add", bench(calc_cpp.run), bench(calc_legacy.run))

        # Filters (Gaussian 5x5)
        filt_cpp = FiltersTool()
        filt_cpp.inputs["input image"] = v
        filt_cpp.params["filters"] = [{"enabled": True, "type": "gaussian_blur",
                                       "kernel_size": 5, "sigma": 1.0}]
        filt_legacy = filters_py()
        filt_legacy.inputs["input image"] = v
        filt_legacy.params["filters"] = [{"enabled": True, "type": "gaussian_blur",
                                          "kernel_size": 5, "sigma": 1.0}]
        add("Filters", size, "gaussian 5x5", bench(filt_cpp.run), bench(filt_legacy.run))

        # Edge detection (full-image ROI)
        edge_cpp = EdgeDetection()
        edge_cpp.inputs["input image"] = v
        edge_cpp.params["roi"] = [0, 0, w, h, 0]
        edge_legacy = edge_py()
        edge_legacy.inputs["input image"] = v
        edge_legacy.params["roi"] = [0, 0, w, h, 0]
        add("Edge", size, "full ROI", bench(edge_cpp.run), bench(edge_legacy.run))

        # Pattern matching: train on a 128x128 patch, single angle/scale.
        cx, cy = w // 2, h // 2
        p_cpp = PatternMatcher()
        p_cpp.inputs["input image"] = v
        p_cpp.save_train_image()
        p_cpp.train_from_roi(cx, cy, 128, 128)
        p_cpp.params.update({
            "angle_start": 0, "angle_end": 0, "angle_step": 1,
            "scale_start": 1.0, "scale_end": 1.0, "scale_step": 0.1,
            "pyramid_levels": 3, "score_threshold": 0.7, "max_matches": 1,
        })
        p_legacy = pattern_py()
        p_legacy.inputs["input image"] = v
        p_legacy.save_train_image()
        p_legacy.params["train_roi"] = [cx, cy, 128, 128, 0]
        p_legacy.train_pattern()
        p_legacy.params.update({
            "angle_start": 0, "angle_end": 0, "angle_step": 1,
            "scale_start": 1.0, "scale_end": 1.0, "scale_step": 0.1,
            "pyramid_levels": 3, "score_threshold": 0.7, "max_matches": 1,
        })
        add("Pattern", size, "128px patch, 1 angle",
            bench(p_cpp.run, n=3), bench(p_legacy.run, n=3))

    print("\n## Tool benchmarks (median of run())")
    print(f"{'Tool':12} {'Size':5} {'Case':<26} {'C++':>12} {'Python':>12} {'Py/C++':>8}")
    print("-" * 86)
    for name, size, label, cpp, py, ratio in results:
        c = fmt(cpp) if cpp else "      n/a"
        p = fmt(py) if py else "      n/a"
        r = f"{ratio:6.2f}x" if ratio else "   n/a"
        print(f"{name:12} {size:5} {label:26} {c:>12} {p:>12} {r:>8}")

    # ---- Point cloud ---------------------------------------------------
    from _falco_kernel import generate_pointcloud as _gen_pc
    pc_py = importlib.import_module("pointcloud_py").range_to_pointcloud_metric

    ph, pw = 1080, 1920
    rng = np.random.default_rng(0)
    depth = rng.uniform(0.3, 2.0, (ph, pw)).astype(np.float32)
    depth[::7, ::7] = 0.0
    sx, sy, sz, oz = 0.05, 0.05, 0.001, -0.5

    def py_optimized():
        valid = np.isfinite(depth) & (depth != 0)
        rows, cols = np.nonzero(valid)
        pts = np.empty((len(rows), 3), dtype=np.float32)
        pts[:, 0] = (pw - 1 - cols) * sx
        pts[:, 1] = rows * sy
        pts[:, 2] = depth[valid] * sz + oz
        return pts

    cpp_pc = bench(lambda: _gen_pc(depth, sx, sy, sz, oz, 10_000_000, True), n=6)
    legacy_pc = bench(lambda: pc_py(depth, sx, sy, sz, oz, target_points=10_000_000), n=3)
    opt_pc = bench(py_optimized, n=6)
    print("\n## Point cloud (1920x1080 depth)")
    print(f"  C++                 {fmt(cpp_pc)}")
    print(f"  Legacy Python       {fmt(legacy_pc)}")
    print(f"  Optimized Python    {fmt(opt_pc)}")
    print(f"  legacy/C++          {legacy_pc['median'] / cpp_pc['median']:.2f}x")
    print(f"  optimized/C++       {opt_pc['median'] / cpp_pc['median']:.2f}x")

    # ---- Barcode -------------------------------------------------------
    from falcocv.tools.barcode_decoder.logic import BarcodeDecoder
    from pyzbar.pyzbar import decode as pyzbar_decode
    import zxingcpp

    img = images["5MP"]
    h, w = img.shape[:2]
    v = VisionImage(img)

    def make_barcode(mode):
        t = BarcodeDecoder()
        t.inputs["input image"] = v
        t.params["code_type"] = mode
        t.params["roi"] = [0, 0, w, h, 0]
        return t

    print("\n## Barcode (5MP, full image)")
    for mode in ("1D", "Auto"):
        r = bench(lambda: make_barcode(mode).run(), n=3, warm=1)
        print(f"  current {mode:5}       {fmt(r)}")
    print(f"  pyzbar full         {fmt(bench(lambda: pyzbar_decode(img), n=3, warm=1))}")
    print(f"  zxingcpp full       {fmt(bench(lambda: zxingcpp.read_barcodes(img), n=3, warm=1))}")


if __name__ == "__main__":
    main()
