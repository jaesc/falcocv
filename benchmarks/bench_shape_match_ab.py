"""ShapeMatch A/B benchmark: interleaved variants with pose deltas.

Variants run in the same process, alternating baseline and candidate on every
repetition, so cache state and CPU frequency affect both equally.  The tool
also reports per-stage call counts so a speed change can be attributed to a
stage instead of the whole run.  Deltas are printed as "variant vs baseline":
positive means the variant is faster.

Usage:
    python benchmarks/bench_shape_match_ab.py
    python benchmarks/bench_shape_match_ab.py --job job.cvj --reps 15
    python benchmarks/bench_shape_match_ab.py --json ab.json --quiet
    python benchmarks/bench_shape_match_ab.py --max-px 0.05 --max-angle 0.01
"""

from __future__ import annotations

import argparse
import contextlib
import json
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "cpp" / "bin"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from falcocv.accuracy.synthetic import (  # noqa: E402
    SyntheticScene,
    make_template,
    pose_error,
    render_scene,
    scene_suite,
)
from falcocv.core.job_manager import JobManager  # noqa: E402
from falcocv.core.vision_image import VisionImage  # noqa: E402
from falcocv.core.workflow import Workflow  # noqa: E402
from falcocv.core.workflow_controller import WorkflowController  # noqa: E402
from falcocv.tools.shape_match.logic import ShapeMatch  # noqa: E402

# Stages are leaves: nested helpers (solve_pose, occlusion_refine) are not
# instrumented to keep the numbers additive and unambiguous.
_STAGE_METHODS = {
    "_dt_scan_xy": "translate",
    "_dt_scan_axis": "axis",
    "_alignment_score": "align",
    "_icp_refine": "icp",
    "_kernel_coarse": "vote",
    "_ecc_refine": "ecc",
}
_STAGES = ("translate", "axis", "align", "icp", "vote", "ncc", "ecc")


# ----------------------------------------------------------------------
# Variants
# ----------------------------------------------------------------------
@contextlib.contextmanager
def _baseline():
    """Current behaviour, no patch."""
    yield


@contextlib.contextmanager
def _single_translation_grid():
    """Restore the single-step translation grid (pre Step 1.1 behaviour)."""
    original = ShapeMatch._dt_scan_xy_staged

    def single(self, *args, **kwargs):
        return self._dt_scan_xy(*args, **kwargs)

    ShapeMatch._dt_scan_xy_staged = single
    try:
        yield
    finally:
        ShapeMatch._dt_scan_xy_staged = original


VARIANTS = {
    "baseline": _baseline,
    "single_grid": _single_translation_grid,
}


# ----------------------------------------------------------------------
# Case construction
# ----------------------------------------------------------------------
def _job_cases(path: Path) -> list[tuple[str, list[ShapeMatch]]]:
    workflow = Workflow()
    JobManager().load_job(workflow, str(path))
    controller = WorkflowController(workflow)
    controller.set_scheduler(lambda callback: callback())
    controller.run(advance=False)
    tools = [
        tool
        for tool in workflow.tools
        if isinstance(tool, ShapeMatch) and tool.is_trained
    ]
    if not tools:
        raise SystemExit(f"No trained ShapeMatch tool in {path}")
    return [(path.name, tools)]


def _synthetic_tool(template: np.ndarray, max_features: int) -> ShapeMatch:
    tool = ShapeMatch()
    tool.params.update(
        {
            "angle_start": -35.0,
            "angle_end": 35.0,
            "min_score": 0.25,
            "max_matches": 1,
            "max_features": max_features,
            "show_search_roi": False,
            "show_found_region": False,
            "show_found_origin": False,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool._static_train_image = template.copy()
    tool.params["train_roi"] = [
        0,
        0,
        template.shape[1],
        template.shape[0],
        0,
    ]
    tool.train_pattern()
    return tool


def _large_scene(template: np.ndarray) -> SyntheticScene:
    """A 2.7 MP scene, where the solver cost stops being noise."""
    height, width = 1440, 1920
    cx, cy = width / 2.0, height / 2.0
    return SyntheticScene(
        name="2.7MP",
        image=render_scene(
            template,
            cx=cx,
            cy=cy,
            angle=12.0,
            scale=1.05,
            gamma=0.95,
            noise=3.0,
            scene_size=(height, width),
            seed=11,
        ),
        template=template,
        cx=cx,
        cy=cy,
        angle=12.0,
        scale=1.05,
    )


def _synthetic_cases(max_features: int) -> list[tuple[str, list[ShapeMatch]]]:
    template = make_template()
    scenes = list(scene_suite(template))
    scenes.append(_large_scene(template))
    cases = []
    for scene in scenes:
        tool = _synthetic_tool(template, max_features)
        tool.inputs["input image"] = VisionImage(scene.image)
        cases.append((scene.name, [tool]))
    return cases


# ----------------------------------------------------------------------
# Measurement
# ----------------------------------------------------------------------
def _poses(tool: ShapeMatch) -> list[tuple[float, float, float, float]]:
    results = tool.outputs.get("results") or []
    return [
        (
            float(result.get("x", 0.0)),
            float(result.get("y", 0.0)),
            float(result.get("angle", 0.0)),
            float(result.get("scale", 1.0)),
        )
        for result in results
    ]


@dataclass
class CaseResult:
    name: str
    tools: list[str] = field(default_factory=list)
    medians: dict[str, dict[str, float]] = field(default_factory=dict)
    poses: dict[str, dict[str, list[tuple[float, float, float, float]]]] = field(
        default_factory=dict
    )
    profiles: dict[str, dict[str, dict[str, float]]] = field(default_factory=dict)


def _run_variant(tools, variant) -> None:
    with variant():
        for tool in tools:
            tool.run()


def _measure_case(
    name: str, tools: list[ShapeMatch], variants: dict, reps: int
) -> CaseResult:
    result = CaseResult(name=name, tools=[tool.name for tool in tools])
    samples = {
        key: {tool.name: [] for tool in tools} for key in variants
    }
    for _key, variant in variants.items():
        _run_variant(tools, variant)

    for _rep in range(reps):
        for key, variant in variants.items():
            with variant():
                for tool in tools:
                    start = time.perf_counter()
                    tool.run()
                    elapsed = (time.perf_counter() - start) * 1000.0
                    samples[key][tool.name].append(elapsed)
                    result.poses.setdefault(key, {})[tool.name] = _poses(tool)

    for key, per_tool in samples.items():
        result.medians[key] = {
            tool_name: float(statistics.median(values))
            for tool_name, values in per_tool.items()
        }
    for key, variant in variants.items():
        result.profiles[key] = _profile_case(tools, variant)
    return result


def _profile_case(tools, variant) -> dict[str, dict[str, float]]:
    profile = {}
    for tool in tools:
        stats, restore = _instrument(tool)
        try:
            _run_variant([tool], variant)
        finally:
            restore()
        profile[tool.name] = stats
    return profile


def _instrument(tool: ShapeMatch):
    stats = {
        stage: {"calls": 0, "ms": 0.0, "poses": 0}
        for stage in _STAGES
    }
    originals = {}

    def wrap(original, stage):
        def wrapper(*args, **kwargs):
            start = time.perf_counter()
            value = original(*args, **kwargs)
            stats[stage]["ms"] += (time.perf_counter() - start) * 1000.0
            stats[stage]["calls"] += 1
            stats[stage]["poses"] += _pose_count(stage, args, kwargs)
            return value

        return wrapper

    for attr, stage in _STAGE_METHODS.items():
        original = getattr(tool, attr, None)
        if original is None:
            continue
        originals[attr] = original
        setattr(tool, attr, wrap(original, stage))

    original_match = cv2.matchTemplate

    def match_wrapper(*args, **kwargs):
        start = time.perf_counter()
        value = original_match(*args, **kwargs)
        stats["ncc"]["ms"] += (time.perf_counter() - start) * 1000.0
        stats["ncc"]["calls"] += 1
        return value

    cv2.matchTemplate = match_wrapper

    def restore():
        cv2.matchTemplate = original_match
        for attr, original in originals.items():
            setattr(tool, attr, original)

    return stats, restore


def _arg(args, kwargs, index, name):
    if index < len(args):
        return args[index]
    return kwargs.get(name)


def _pose_count(stage: str, args, kwargs) -> int:
    if stage == "translate":
        span = _arg(args, kwargs, 7, "span")
        step = _arg(args, kwargs, 8, "step")
        if span is None or step in (None, 0):
            return 0
        offsets = int(2.0 * float(span) / float(step) + 1.5)
        return max(1, offsets) ** 2
    if stage == "axis":
        low = _arg(args, kwargs, 8, "low")
        high = _arg(args, kwargs, 9, "high")
        step = _arg(args, kwargs, 10, "step")
        if low is None or high is None or step in (None, 0):
            return 0
        return max(1, int((float(high) - float(low)) / float(step) + 1.5))
    if stage == "align":
        return 1
    return 0


# ----------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------
def _pose_delta(
    reference: list[tuple[float, float, float, float]],
    candidate: list[tuple[float, float, float, float]],
) -> dict[str, float]:
    if len(reference) != len(candidate):
        return {
            "count": len(candidate),
            "lost": len(reference) - len(candidate),
            "px": 0.0,
            "angle": 0.0,
            "scale_pct": 0.0,
        }
    px = angle = scale = 0.0
    for left, right in zip(reference, candidate):
        px = max(px, float(np.hypot(left[0] - right[0], left[1] - right[1])))
        angle = max(angle, abs(left[2] - right[2]))
        scale = max(scale, abs(left[3] - right[3]) * 100.0)
    return {
        "count": len(candidate),
        "lost": len(reference) - len(candidate),
        "px": px,
        "angle": angle,
        "scale_pct": scale,
    }


def _synthetic_error(case: str, poses, suite) -> str:
    scene = next((item for item in suite if item.name == case), None)
    if scene is None or not poses:
        return ""
    x, y, angle, scale = poses[0]
    error = pose_error(scene, x, y, angle, scale)
    if error.translation_px is None:
        return "pose=lost"
    return (
        f"gt px={error.translation_px:.3f} "
        f"ang={error.rotation_deg:.4f} "
        f"scale={error.scale_rel * 100:.4f}%"
    )


def _print_case(result: CaseResult, variants: dict, quiet: bool) -> None:
    baseline = "baseline"
    for tool_name in result.tools:
        base = result.medians[baseline][tool_name]
        for key in variants:
            if key == baseline:
                continue
            value = result.medians[key][tool_name]
            delta = 100.0 * (base - value) / base if base else 0.0
            poses = _pose_delta(
                result.poses[baseline][tool_name],
                result.poses[key][tool_name],
            )
            print(
                f"  {tool_name:16s} {base:7.1f} -> {value:7.1f} ms "
                f"({delta:+5.1f}%)  dpx={poses['px']:.3f} "
                f"dang={poses['angle']:.4f} dscale={poses['scale_pct']:.4f}% "
                f"n={poses['count']}"
                + (" LOST" if poses["lost"] else "")
            )
    if quiet:
        return
    for tool_name in result.tools:
        for key in variants:
            stats = result.profiles[key].get(tool_name) or {}
            parts = [
                f"{stage}={stats[stage]['calls']}c/"
                f"{stats[stage]['poses']}p/{stats[stage]['ms']:.1f}ms"
                for stage in _STAGES
                if stats.get(stage, {}).get("calls")
            ]
            if parts:
                print(f"    profile {key}/{tool_name}: " + " ".join(parts))


def _check_gate(poses, limits: dict) -> list[str]:
    failures = []
    if poses["lost"]:
        failures.append(f"lost {poses['lost']} detection(s)")
    if limits.get("px") is not None and poses["px"] > limits["px"]:
        failures.append(f"dpx {poses['px']:.3f} > {limits['px']}")
    if limits.get("angle") is not None and poses["angle"] > limits["angle"]:
        failures.append(f"dang {poses['angle']:.4f} > {limits['angle']}")
    if (
        limits.get("scale_pct") is not None
        and poses["scale_pct"] > limits["scale_pct"]
    ):
        failures.append(
            f"dscale {poses['scale_pct']:.4f}% > {limits['scale_pct']}"
        )
    return failures


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", help="Run the ShapeMatch tools of a .cvj job")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Run the synthetic scene suite (default without --job)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=800,
        help="Training budget for the synthetic tool (default 800)",
    )
    parser.add_argument("--reps", type=int, default=9)
    parser.add_argument(
        "--variants",
        default=",".join(VARIANTS),
        help=f"Comma separated subset of: {', '.join(VARIANTS)}",
    )
    parser.add_argument("--json", help="Write the results JSON")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--max-px", type=float, default=None)
    parser.add_argument("--max-angle", type=float, default=None)
    parser.add_argument("--max-scale-pct", type=float, default=None)
    args = parser.parse_args(argv)

    if not args.job and not args.synthetic:
        args.synthetic = True

    wanted = [name.strip() for name in args.variants.split(",") if name.strip()]
    missing = [name for name in wanted if name not in VARIANTS]
    if missing:
        parser.error(f"unknown variant(s): {', '.join(missing)}")
    variants = {name: VARIANTS[name] for name in wanted}
    if "baseline" not in variants:
        variants = {"baseline": VARIANTS["baseline"], **variants}

    suite = scene_suite(make_template()) if args.synthetic else []
    if args.job:
        cases = _job_cases(Path(args.job))
    else:
        cases = _synthetic_cases(args.max_features)

    limits = {
        "px": args.max_px,
        "angle": args.max_angle,
        "scale_pct": args.max_scale_pct,
    }
    failures = []
    report = {"cases": {}}
    for name, tools in cases:
        result = _measure_case(name, tools, variants, args.reps)
        if not args.quiet:
            print(f"== {name} ==")
        _print_case(result, variants, args.quiet)
        entry = {
            "tools": result.tools,
            "medians_ms": result.medians,
            "profiles": result.profiles,
            "deltas": {},
        }
        for key in variants:
            if key == "baseline":
                continue
            entry["deltas"][key] = {
                tool_name: _pose_delta(
                    result.poses["baseline"][tool_name],
                    result.poses[key][tool_name],
                )
                for tool_name in result.tools
            }
            for tool_name in result.tools:
                failures.extend(
                    f"{name}/{tool_name}: {message}"
                    for message in _check_gate(
                        entry["deltas"][key][tool_name], limits
                    )
                )
        if args.synthetic:
            for key in variants:
                for tool_name in result.tools:
                    poses = result.poses[key].get(tool_name) or []
                    text = _synthetic_error(name, poses, suite)
                    if text and not args.quiet:
                        print(f"    {key}/{tool_name}: {text}")
                    entry.setdefault("synthetic_error", {}).setdefault(
                        key, {}
                    )[tool_name] = text
        report["cases"][name] = entry

    if failures:
        print("\nGATE FAILED")
        for failure in failures:
            print(f"  - {failure}")
    elif limits["px"] is not None or limits["angle"] is not None:
        print("\nGATE PASSED")

    if args.json:
        Path(args.json).write_text(
            json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
        )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
