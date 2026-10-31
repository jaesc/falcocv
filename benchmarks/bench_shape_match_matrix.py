"""Interleaved ShapeMatch matrix benchmark.

Runs every case with several runtime configurations in alternating rounds so
thermal drift and background load do not favour one configuration.  Reports
median latency and pose accuracy per (case, configuration).

Usage:
    python benchmarks/bench_shape_match_matrix.py
    python benchmarks/bench_shape_match_matrix.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "cpp" / "bin"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from falcocv.accuracy.synthetic import make_template, render_scene  # noqa: E402
from falcocv.core.vision_image import VisionImage  # noqa: E402
from falcocv.tools.shape_match.logic import ShapeMatch  # noqa: E402


def _tool(template: np.ndarray) -> ShapeMatch:
    tool = ShapeMatch()
    tool.params.update(
        {
            "angle_start": -30.0,
            "angle_end": 30.0,
            "scale_start": 0.9,
            "scale_end": 1.1,
            "min_score": 0.25,
            "max_matches": 1,
            "max_features": 800,
            "show_search_roi": False,
            "show_found_region": False,
            "show_found_origin": False,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool._static_train_image = template.copy()
    tool.params["train_roi"] = [0, 0, template.shape[1], template.shape[0], 0]
    tool.train_pattern()
    return tool


CASES = (
    ("480x360", (360, 480), (120, 90)),
    ("1MP", (960, 1280), (120, 90)),
    ("2.7MP", (1440, 1920), (120, 90)),
    ("big5MP", (1940, 2588), (404, 407)),
)

CONFIGS = {
    "default": None,
    "cv1": 1,
    "cv2": 2,
    "cv4": 4,
}


def _case(template_size, scene_size):
    base = make_template()
    width, height = template_size
    if (width, height) != (base.shape[1], base.shape[0]):
        template = cv2.resize(base, (width, height), interpolation=cv2.INTER_CUBIC)
    else:
        template = base
    scene_height, scene_width = scene_size
    cx = scene_width / 2.0
    cy = scene_height / 2.0
    angle = 12.0
    scale = 1.05
    image = render_scene(
        template,
        cx=cx,
        cy=cy,
        angle=angle,
        scale=scale,
        gamma=0.9,
        noise=3.0,
        scene_size=scene_size,
        seed=7,
    )
    return template, image, (cx, cy, angle, scale)


def _apply(config):
    if config is None:
        return cv2.getNumThreads()
    previous = cv2.getNumThreads()
    cv2.setNumThreads(config)
    return previous


def _restore(previous):
    cv2.setNumThreads(previous)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Write results JSON")
    parser.add_argument("--rounds", type=int, default=5)
    args = parser.parse_args(argv)

    report = {}
    for case_name, scene_size, template_size in CASES:
        template, image, truth = _case(template_size, scene_size)
        tool = _tool(template)
        tool.inputs["input image"] = VisionImage(image)
        tool.run()
        results = {name: {"times": [], "error": None} for name in CONFIGS}

        for _ in range(args.rounds):
            for name, config in CONFIGS.items():
                previous = _apply(config)
                try:
                    tool.inputs["input image"] = VisionImage(image)
                    start = time.perf_counter()
                    tool.run()
                    elapsed = (time.perf_counter() - start) * 1000.0
                finally:
                    _restore(previous)
                results[name]["times"].append(elapsed)

        for name, config in CONFIGS.items():
            previous = _apply(config)
            try:
                tool.inputs["input image"] = VisionImage(image)
                tool.run()
            finally:
                _restore(previous)
            if tool.outputs.get("count"):
                result = tool.outputs["results"][0]
                results[name]["error"] = {
                    "translation_px": float(
                        np.hypot(result.x - truth[0], result.y - truth[1])
                    ),
                    "angle_deg": float(abs(result.angle - truth[2])),
                    "scale_rel": float(abs(result.scale - truth[3]) / truth[3]),
                }

        print(f"== {case_name} ==")
        report[case_name] = {}
        for name in CONFIGS:
            times = results[name]["times"]
            median = float(np.median(times))
            error = results[name]["error"] or {}
            report[case_name][name] = {
                "median_ms": median,
                "min_ms": float(min(times)),
                "error": error,
            }
            print(
                f"  {name:8s} median {median:6.1f} ms  min {min(times):6.1f} ms"
                f"  err {error.get('translation_px', float('nan')):.3f} px"
                f" / {error.get('angle_deg', float('nan')):.3f} deg"
                f" / {100.0 * error.get('scale_rel', float('nan')):.3f} %"
            )

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
