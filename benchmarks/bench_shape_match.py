"""ShapeMatch performance benchmark and phase profile.

Usage:
    python benchmarks/bench_shape_match.py
    python benchmarks/bench_shape_match.py --json results.json
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

from falcocv.accuracy.synthetic import (  # noqa: E402
    BACKGROUND,
    make_template,
    render_scene,
    scene_suite,
)
from falcocv.core.vision_image import VisionImage  # noqa: E402
from falcocv.tools.pattern_matcher.logic import PatternMatcher  # noqa: E402
from falcocv.tools.shape_match.logic import ShapeMatch  # noqa: E402


def _shape_tool(
    template: np.ndarray,
    max_matches: int = 1,
    angle_start: float = -30.0,
    angle_end: float = 30.0,
    scale_start: float = 0.9,
    scale_end: float = 1.1,
) -> ShapeMatch:
    tool = ShapeMatch()
    tool.params.update(
        {
            "angle_start": angle_start,
            "angle_end": angle_end,
            "scale_start": scale_start,
            "scale_end": scale_end,
            "min_score": 0.25,
            "max_matches": max_matches,
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


def _pattern_tool(template: np.ndarray) -> PatternMatcher:
    tool = PatternMatcher()
    tool.params.update(
        {
            "angle_start": -30,
            "angle_end": 30,
            "angle_step": 2,
            "scale_start": 1.0,
            "scale_end": 1.0,
            "pyramid_levels": 2,
            "score_threshold": 0.3,
            "max_matches": 1,
            "show_search_roi": False,
            "show_overlays": False,
            "show_labels": False,
            "show_found_region": False,
            "show_found_origin": False,
            "copy_overlays_to_input": False,
        }
    )
    tool._static_train_image = template.copy()
    tool.params["train_roi"] = [0, 0, template.shape[1], template.shape[0], 0]
    tool.train_pattern()
    return tool


def _time_tool(tool, image, repeats=5):
    times = []
    for _ in range(repeats):
        tool.inputs["input image"] = VisionImage(image)
        start = time.perf_counter()
        tool.run()
        times.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(times)), float(np.min(times))


def _profile_phases(tool, image):
    names = (
        "_kernel_coarse",
        "_coarse_pose",
        "_ncc_angle_scan",
        "_ncc_scale_scan",
        "_ncc_translate",
        "_local_alignment",
        "_ecc_refine",
        "_occlusion_refine",
        "_dt_score",
        "_icp_refine",
    )
    timings = {name.lstrip("_"): 0.0 for name in names}
    timings["matchTemplate"] = 0.0
    calls = {name.lstrip("_"): 0 for name in names}
    calls["matchTemplate"] = 0

    originals = {name: getattr(tool, name) for name in names}
    original_match = cv2.matchTemplate

    def wrap(name, original):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            result = original(*args, **kwargs)
            timings[name] += (time.perf_counter() - start) * 1000.0
            calls[name] += 1
            return result

        return wrapper

    for name in names:
        setattr(tool, name, wrap(name.lstrip("_"), originals[name]))
    cv2.matchTemplate = wrap("matchTemplate", original_match)
    try:
        tool.inputs["input image"] = VisionImage(image)
        start = time.perf_counter()
        tool.run()
        total = (time.perf_counter() - start) * 1000.0
    finally:
        cv2.matchTemplate = original_match
        for name, original in originals.items():
            setattr(tool, name, original)

    return {"total_ms": total, "phases": timings, "calls": calls}


def _scene(size):
    template = make_template()
    height, width = size
    return template, render_scene(
        template,
        cx=width / 2.0,
        cy=height / 2.0,
        angle=12.0,
        scale=1.05,
        gamma=0.9,
        noise=3.0,
        scene_size=(height, width),
        seed=7,
    )


def _two_part_scene(template):
    scene = np.full((360, 480, 3), BACKGROUND, dtype=np.uint8)
    height, width = template.shape[:2]
    for cx, cy, angle in ((180.0, 150.0, 0.0), (360.0, 230.0, 15.0)):
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1.0)
        matrix[0, 2] += cx - width / 2
        matrix[1, 2] += cy - height / 2
        warped = cv2.warpAffine(
            template, matrix, (480, 360), borderValue=(BACKGROUND,) * 3
        )
        mask = cv2.warpAffine(
            np.full((height, width), 255, np.uint8),
            matrix,
            (480, 360),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        scene = np.where(mask[..., None] > 0, warped, scene)
    return scene


def _run_case(name, tool, image, report, include_pattern=False, template=None):
    if include_pattern and template is not None:
        pattern = _pattern_tool(template)
        pat_median, pat_min = _time_tool(pattern, image)
    else:
        pat_median = pat_min = None
    shape_median, shape_min = _time_tool(tool, image)
    profile = _profile_phases(tool, image)
    report[name] = {
        "pattern_median_ms": pat_median,
        "pattern_min_ms": pat_min,
        "shape_median_ms": shape_median,
        "shape_min_ms": shape_min,
        "shape_profile": profile,
        "shape_found": tool.outputs.get("count", 0),
    }
    if not _run_case.quiet:
        print(f"== {name} ==")
        if pat_median is not None:
            print(f"  PatternMatcher median: {pat_median:.1f} ms (min {pat_min:.1f})")
        print(f"  ShapeMatch median:     {shape_median:.1f} ms (min {shape_min:.1f})")
        print(f"  ShapeMatch found:      {tool.outputs.get('count', 0)}")
        print(f"  phases: {profile['phases']}")
        print(f"  calls:  {profile['calls']}")


_run_case.quiet = False


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Write results JSON")
    parser.add_argument("--json-only", action="store_true")
    args = parser.parse_args(argv)
    _run_case.quiet = args.json_only

    template = make_template()
    report = {}

    for name, size in (("480x360", (360, 480)), ("1MP", (960, 1280))):
        template, image = _scene(size)
        _run_case(
            name,
            _shape_tool(template),
            image,
            report,
            include_pattern=True,
            template=template,
        )

    template, image = _scene((1440, 1920))
    _run_case("2.7MP", _shape_tool(template), image, report)

    scenes = {scene.name: scene for scene in scene_suite(template)}
    for name in ("occlusion20", "occlusion30", "combo"):
        scene = scenes[name]
        _run_case(name, _shape_tool(template), scene.image, report)

    _run_case(
        "multi2",
        _shape_tool(template, max_matches=2),
        _two_part_scene(template),
        report,
    )

    for name, span in (("angle+-90", (-90.0, 90.0)), ("angle+-180", (-180.0, 180.0))):
        _run_case(
            name,
            _shape_tool(template, angle_start=span[0], angle_end=span[1]),
            scenes["baseline"].image,
            report,
        )

    _run_case(
        "fixed-scale",
        _shape_tool(template, scale_start=1.0, scale_end=1.0),
        scenes["baseline"].image,
        report,
    )

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
