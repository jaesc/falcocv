"""F1 bake-off: current PatternMatcher vs generalized Hough vs SIFT+RANSAC.

Run:
    python benchmarks/bench_matcher.py
    python benchmarks/bench_matcher.py --json results.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "cpp" / "bin"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from falcocv.accuracy.synthetic import (  # noqa: E402
    PoseError,
    SyntheticScene,
    pose_error,
    scene_suite,
)
from falcocv.core.vision_image import VisionImage  # noqa: E402
from falcocv.tools.pattern_matcher.logic import PatternMatcher  # noqa: E402
from falcocv.tools.shape_match.logic import ShapeMatch  # noqa: E402


def _pattern_tool(template: np.ndarray) -> PatternMatcher:
    tool = PatternMatcher()
    tool.params.update(
        {
            "angle_start": -35,
            "angle_end": 35,
            "angle_step": 2,
            "scale_start": 0.85,
            "scale_end": 1.15,
            "scale_step": 0.05,
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


def run_pattern(tool: PatternMatcher, scene: SyntheticScene) -> dict:
    started = time.perf_counter()
    tool.inputs["input image"] = VisionImage(scene.image)
    tool.run()
    elapsed = (time.perf_counter() - started) * 1000.0
    results = tool.outputs.get("results") or []
    if not results:
        return {"found": False, "time_ms": elapsed, "score": None}
    best = results[0]
    return {
        "found": True,
        "x": float(best.x),
        "y": float(best.y),
        "angle": float(best.angle),
        "scale": float(best.scale),
        "score": float(best.score),
        "time_ms": elapsed,
    }


def _shape_tool(template: np.ndarray) -> ShapeMatch:
    tool = ShapeMatch()
    tool.params.update(
        {
            "angle_start": -35.0,
            "angle_end": 35.0,
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


def run_shape(tool: ShapeMatch, scene: SyntheticScene) -> dict:
    started = time.perf_counter()
    tool.inputs["input image"] = VisionImage(scene.image)
    tool.run()
    elapsed = (time.perf_counter() - started) * 1000.0
    results = tool.outputs.get("results") or []
    if not results:
        return {"found": False, "time_ms": elapsed, "score": None}
    best = results[0]
    return {
        "found": True,
        "x": float(best.x),
        "y": float(best.y),
        "angle": float(best.angle),
        "scale": float(best.scale),
        "score": float(best.score),
        "time_ms": elapsed,
    }


def _edge(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return cv2.Canny(gray, 50, 150)


def run_hough(template: np.ndarray, scene: SyntheticScene) -> dict:
    template_edges = _edge(template)
    scene_edges = _edge(scene.image)
    hough = cv2.createGeneralizedHoughGuil()
    hough.setMinDist(50)
    hough.setDp(1.0)
    hough.setMaxBufferSize(10000)
    hough.setCannyLowThresh(50)
    hough.setCannyHighThresh(150)
    hough.setLevels(360)
    hough.setAngleEpsilon(1.0)
    hough.setMinAngle(0)
    hough.setMaxAngle(360)
    hough.setAngleStep(5)
    hough.setAngleThresh(30)
    hough.setMinScale(0.85)
    hough.setMaxScale(1.15)
    hough.setScaleStep(0.05)
    hough.setScaleThresh(10)
    hough.setPosThresh(30)
    hough.setTemplate(template_edges)

    started = time.perf_counter()
    try:
        positions, votes = hough.detect(scene_edges)
    except cv2.error as exc:
        return {"found": False, "time_ms": (time.perf_counter() - started) * 1000.0, "error": str(exc)}
    elapsed = (time.perf_counter() - started) * 1000.0
    if positions is None or not len(positions) or not len(positions[0]):
        return {"found": False, "time_ms": elapsed}
    row = positions[0][0]
    return {
        "found": True,
        "x": float(row[0]),
        "y": float(row[1]),
        "scale": float(row[2]),
        # The Guil orientation output uses a different convention (and is
        # ambiguous for this template); keep the raw value for the record.
        "angle": None,
        "raw_angle": float(row[3]),
        "time_ms": elapsed,
    }


def run_sift(template: np.ndarray, scene: SyntheticScene) -> dict:
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    scene_gray = cv2.cvtColor(scene.image, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(nfeatures=2000)
    keypoints_t, descriptors_t = sift.detectAndCompute(template_gray, None)
    started = time.perf_counter()
    keypoints_s, descriptors_s = sift.detectAndCompute(scene_gray, None)
    if descriptors_t is None or descriptors_s is None or len(keypoints_t) < 3:
        return {"found": False, "time_ms": (time.perf_counter() - started) * 1000.0}

    matcher = cv2.BFMatcher(cv2.NORM_L2)
    matches = matcher.knnMatch(descriptors_t, descriptors_s, k=2)
    good = [m for m, n in matches if m.distance < 0.75 * n.distance]
    if len(good) < 3:
        return {"found": False, "time_ms": (time.perf_counter() - started) * 1000.0}

    src = np.float32([keypoints_t[m.queryIdx].pt for m in good])
    dst = np.float32([keypoints_s[m.trainIdx].pt for m in good])
    matrix, inliers = cv2.estimateAffinePartial2D(
        src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    elapsed = (time.perf_counter() - started) * 1000.0
    if matrix is None:
        return {"found": False, "time_ms": elapsed}

    height, width = template.shape[:2]
    center = matrix @ np.array([width / 2.0, height / 2.0, 1.0])
    # cv2.getRotationMatrix2D uses [[cos, sin], [-sin, cos]]; SIFT returns the
    # same image-space matrix, so negate to compare with the scene's angle.
    angle = -float(np.degrees(np.arctan2(matrix[1, 0], matrix[0, 0])))
    scale = float(np.hypot(matrix[0, 0], matrix[1, 0]))
    return {
        "found": True,
        "x": float(center[0]),
        "y": float(center[1]),
        "angle": angle,
        "scale": scale,
        "score": float(inliers.sum()) / max(1, len(good)),
        "time_ms": elapsed,
    }


def _error_dict(error: PoseError) -> dict:
    return asdict(error)


def _summarize(rows: list[dict]) -> dict:
    found = [row for row in rows if row["result"].get("found")]
    translation = [row["error"]["translation_px"] for row in found if row["error"]["translation_px"] is not None]
    rotation = [row["error"]["rotation_deg"] for row in found if row["error"]["rotation_deg"] is not None]
    scale = [row["error"]["scale_rel"] for row in found if row["error"]["scale_rel"] is not None]
    times = [row["result"]["time_ms"] for row in rows if "time_ms" in row["result"]]
    return {
        "scenes": len(rows),
        "found": len(found),
        "translation_mean_px": float(np.mean(translation)) if translation else None,
        "translation_max_px": float(np.max(translation)) if translation else None,
        "rotation_mean_deg": float(np.mean(rotation)) if rotation else None,
        "rotation_max_deg": float(np.max(rotation)) if rotation else None,
        "scale_mean_rel": float(np.mean(scale)) if scale else None,
        "scale_max_rel": float(np.max(scale)) if scale else None,
        "time_mean_ms": float(np.mean(times)) if times else None,
    }


def _repeatability(method, scene: SyntheticScene, runs: int) -> dict | None:
    xs: list[float] = []
    ys: list[float] = []
    for _ in range(runs):
        result = method(scene)
        if result.get("found"):
            xs.append(float(result["x"]))
            ys.append(float(result["y"]))
    if len(xs) < 2:
        return None
    return {
        "runs": len(xs),
        "x_std_px": float(np.std(xs)),
        "y_std_px": float(np.std(ys)),
    }


def _format(value, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value:.2f}{suffix}"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", help="Write the full report to this JSON file")
    args = parser.parse_args(argv)

    scenes = scene_suite()
    template = scenes[0].template
    pattern_tool = _pattern_tool(template)
    shape_tool = _shape_tool(template)

    methods = {
        "pattern_matcher": lambda scene: run_pattern(pattern_tool, scene),
        "shape_match": lambda scene: run_shape(shape_tool, scene),
        "hough_guil": lambda scene: run_hough(template, scene),
        "sift_ransac": lambda scene: run_sift(template, scene),
    }

    report: dict = {"scenes": [], "methods": {}}
    for name, method in methods.items():
        rows = []
        for scene in scenes:
            result = method(scene)
            pose = pose_error(
                scene,
                result.get("x"),
                result.get("y"),
                result.get("angle"),
                result.get("scale"),
            )
            row = {
                "scene": scene.name,
                "method": name,
                "gt": {
                    "x": scene.cx,
                    "y": scene.cy,
                    "angle": scene.angle,
                    "scale": scene.scale,
                },
                "result": result,
                "error": _error_dict(pose),
            }
            rows.append(row)
            if result.get("found"):
                print(
                    f"{name:16s} {scene.name:14s} "
                    f"err_t={_format(pose.translation_px, 'px')} "
                    f"err_a={_format(pose.rotation_deg, 'deg')} "
                    f"err_s={_format(None if pose.scale_rel is None else pose.scale_rel * 100, '%')} "
                    f"t={result['time_ms']:.1f}ms"
                )
            else:
                print(f"{name:16s} {scene.name:14s} NOT FOUND ({result['time_ms']:.1f}ms)")
        summary = _summarize(rows)
        runs = 3 if name == "hough_guil" else 10
        repeatability = _repeatability(method, scenes[0], runs)
        report["methods"][name] = {
            "summary": summary,
            "repeatability": repeatability,
            "rows": rows,
        }
        print(f"{name:16s} summary: {summary}")
        print(f"{name:16s} repeatability: {repeatability}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
