"""Validate ShapeMatch geometry mode against the hybrid reference.

Runs the same job twice per image, once forcing ``match_mode="geometry"`` and
once forcing ``"hybrid"``, then compares detections and poses.  Geometry is the
default for new tools but is still marked experimental, so this is the gate it
has to pass before the mode can be considered consolidated.

Usage:
    python scripts/validate_shape_match_modes.py --job job.cvj --images ./frames
    python scripts/validate_shape_match_modes.py --job job.cvj --images ./frames \\
        --limit 20 --json report.json
    python scripts/validate_shape_match_modes.py --job job.cvj --images ./frames \\
        --max-px 5 --max-angle 2 --max-scale-pct 1.5

Exit code is 1 when a detection is lost or a pose delta exceeds its threshold.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _path in (ROOT / "src", ROOT / "cpp" / "bin"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from falcocv.core.job_manager import JobManager  # noqa: E402
from falcocv.core.workflow import Workflow  # noqa: E402
from falcocv.core.workflow_controller import WorkflowController  # noqa: E402
from falcocv.tools.image_loader.logic import ImageLoader  # noqa: E402
from falcocv.tools.shape_match.logic import ShapeMatch  # noqa: E402

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
MATCH_MODES = ("hybrid", "geometry")


@dataclass
class Detection:
    x: float
    y: float
    angle: float
    scale: float
    score: float
    solver: str | None = None


@dataclass
class ToolReport:
    name: str
    frames: int = 0
    errors: int = 0
    lost: int = 0
    extra: int = 0
    max_px: float = 0.0
    max_angle: float = 0.0
    max_scale_pct: float = 0.0
    hybrid_ms: list[float] = field(default_factory=list)
    geometry_ms: list[float] = field(default_factory=list)
    worst: list[tuple[str, float, float, float]] = field(default_factory=list)

    def summary(self) -> dict:
        return {
            "frames": self.frames,
            "errors": self.errors,
            "lost": self.lost,
            "extra": self.extra,
            "max_px": round(self.max_px, 4),
            "max_angle": round(self.max_angle, 4),
            "max_scale_pct": round(self.max_scale_pct, 4),
            "hybrid_ms": round(statistics.median(self.hybrid_ms), 3)
            if self.hybrid_ms
            else None,
            "geometry_ms": round(statistics.median(self.geometry_ms), 3)
            if self.geometry_ms
            else None,
            "worst": [
                {
                    "image": name,
                    "px": round(px, 4),
                    "angle": round(angle, 4),
                    "scale_pct": round(scale, 4),
                }
                for name, px, angle, scale in sorted(
                    self.worst, key=lambda item: item[1], reverse=True
                )[:5]
            ],
        }


def _image_paths(folder: Path, pattern: str | None, limit: int | None):
    if pattern:
        files = sorted(folder.glob(pattern))
    else:
        files = sorted(
            path
            for path in folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
        )
    if limit is not None:
        files = files[:limit]
    return files


def _detections(tool: ShapeMatch) -> list[Detection]:
    results = tool.outputs.get("results") or []
    return [
        Detection(
            float(result.get("x", 0.0)),
            float(result.get("y", 0.0)),
            float(result.get("angle", 0.0)),
            float(result.get("scale", 1.0)),
            float(result.get("score", 0.0)),
            result.get("solver"),
        )
        for result in results
    ]


def _match(reference: list[Detection], candidate: list[Detection], max_px: float):
    """Greedy nearest-pose matching between the two runs."""
    pairs: list[tuple[int, int, float]] = []
    used_reference: set[int] = set()
    used_candidate: set[int] = set()
    distances = sorted(
        (
            (
                math.hypot(
                    candidate[i].x - reference[j].x,
                    candidate[i].y - reference[j].y,
                ),
                i,
                j,
            )
            for i in range(len(candidate))
            for j in range(len(reference))
        )
    )
    for distance, index, reference_index in distances:
        if distance > max_px:
            break
        if index in used_candidate or reference_index in used_reference:
            continue
        used_candidate.add(index)
        used_reference.add(reference_index)
        pairs.append((index, reference_index, distance))
    lost = [
        index for index in range(len(reference))
        if index not in used_reference
    ]
    extra = [
        index for index in range(len(candidate))
        if index not in used_candidate
    ]
    return pairs, lost, extra


def _angle_delta(left: float, right: float) -> float:
    return abs((left - right + 180.0) % 360.0 - 180.0)


def _run(workflow, controller, loader, tools, image: Path, mode: str):
    loader.params["mode"] = "File"
    loader.params["file_path"] = str(image)
    for tool in tools:
        tool.params["match_mode"] = mode
    for tool in workflow.tools:
        tool.last_error = None
    controller.run(advance=False)
    return {
        tool.name: (
            _detections(tool),
            float(tool.last_execution_time_ms),
            tool.last_error,
        )
        for tool in tools
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--job", required=True, help="Path to the .cvj job")
    parser.add_argument("--images", required=True, help="Folder with frames")
    parser.add_argument("--pattern", help="Glob inside --images (default: image suffixes)")
    parser.add_argument("--limit", type=int, help="Only the first N frames")
    parser.add_argument("--match-px", type=float, default=200.0,
                        help="Max distance to pair two detections (default 200)")
    parser.add_argument("--max-px", type=float, default=5.0)
    parser.add_argument("--max-angle", type=float, default=2.0)
    parser.add_argument("--max-scale-pct", type=float, default=1.5)
    parser.add_argument("--no-gate", action="store_true",
                        help="Report only, never fail")
    parser.add_argument("--json", help="Write the full report as JSON")
    args = parser.parse_args(argv)

    folder = Path(args.images)
    if not folder.is_dir():
        parser.error(f"--images is not a folder: {folder}")
    images = _image_paths(folder, args.pattern, args.limit)
    if not images:
        parser.error(f"no images found in {folder}")

    workflow = Workflow()
    JobManager().load_job(workflow, args.job)
    tools = [
        tool
        for tool in workflow.tools
        if isinstance(tool, ShapeMatch) and tool.is_trained
    ]
    if not tools:
        parser.error("the job has no trained ShapeMatch tool")
    loaders = [tool for tool in workflow.tools if isinstance(tool, ImageLoader)]
    if not loaders:
        parser.error("the job needs an Image Loader to feed the frames")
    connected = {
        destination.name
        for _source, _output, destination, input_name in workflow.connections
        if input_name == "input image"
    }
    missing_input = [tool.name for tool in tools if tool.name not in connected]
    if missing_input:
        parser.error(
            "ShapeMatch tool(s) without an 'input image' connection: "
            + ", ".join(missing_input)
        )
    loader = loaders[0]
    original_file = loader.params.get("file_path")
    original_mode = loader.params.get("mode")

    controller = WorkflowController(workflow)
    controller.set_scheduler(lambda callback: callback())

    reports = {tool.name: ToolReport(name=tool.name) for tool in tools}
    failures: list[str] = []

    for image in images:
        runs: dict[str, dict[str, ShapeMatch]] = {}
        for mode in MATCH_MODES:
            runs[mode] = _run(
                workflow, controller, loader, tools, image, mode
            )
        for tool in tools:
            report = reports[tool.name]
            report.frames += 1
            geometry = runs["geometry"][tool.name]
            hybrid = runs["hybrid"][tool.name]
            report.geometry_ms.append(geometry[1])
            report.hybrid_ms.append(hybrid[1])
            if geometry[2] or hybrid[2]:
                report.errors += 1
                failures.append(
                    f"{image.name}/{tool.name}: {geometry[2] or hybrid[2]}"
                )
                continue
            hybrid_detections = hybrid[0]
            geometry_detections = geometry[0]
            pairs, lost, extra = _match(
                hybrid_detections, geometry_detections, args.match_px
            )
            report.lost += len(lost)
            report.extra += len(extra)
            if lost:
                failures.append(
                    f"{image.name}/{tool.name}: lost {len(lost)} hybrid "
                    "detection(s)"
                )
            for candidate_index, reference_index, _ in pairs:
                candidate = geometry_detections[candidate_index]
                reference = hybrid_detections[reference_index]
                px = math.hypot(
                    candidate.x - reference.x, candidate.y - reference.y
                )
                angle = _angle_delta(candidate.angle, reference.angle)
                scale = abs(candidate.scale - reference.scale) / max(
                    abs(reference.scale), 1e-6
                ) * 100.0
                report.max_px = max(report.max_px, px)
                report.max_angle = max(report.max_angle, angle)
                report.max_scale_pct = max(report.max_scale_pct, scale)
                report.worst.append((image.name, px, angle, scale))
                if not args.no_gate:
                    if px > args.max_px:
                        failures.append(
                            f"{image.name}/{tool.name}: dpx {px:.3f} > "
                            f"{args.max_px}"
                        )
                    if angle > args.max_angle:
                        failures.append(
                            f"{image.name}/{tool.name}: dang {angle:.3f} > "
                            f"{args.max_angle}"
                        )
                    if scale > args.max_scale_pct:
                        failures.append(
                            f"{image.name}/{tool.name}: dscale {scale:.3f}% > "
                            f"{args.max_scale_pct}"
                        )
            if not args.no_gate:
                for detection in geometry_detections:
                    if detection.solver != "geometry":
                        failures.append(
                            f"{image.name}/{tool.name}: geometry mode "
                            f"reported solver '{detection.solver}'"
                        )

    loader.params["file_path"] = original_file
    loader.params["mode"] = original_mode

    print(f"job:        {args.job}")
    print(f"frames:     {len(images)}")
    print(
        f"thresholds: dpx<={args.max_px} dang<={args.max_angle} deg "
        f"dscale<={args.max_scale_pct} %"
    )
    header = (
        f"{'tool':16s} {'lost':>5s} {'extra':>6s} {'dpx':>8s} {'dang':>8s} "
        f"{'dscale%':>9s} {'hybrid ms':>10s} {'geom ms':>8s}"
    )
    print("")
    print(header)
    print("-" * len(header))
    for tool in tools:
        report = reports[tool.name]
        hybrid_ms = (
            statistics.median(report.hybrid_ms) if report.hybrid_ms else 0.0
        )
        geometry_ms = (
            statistics.median(report.geometry_ms)
            if report.geometry_ms
            else 0.0
        )
        print(
            f"{report.name:16s} {report.lost:5d} {report.extra:6d} "
            f"{report.max_px:8.3f} {report.max_angle:8.3f} "
            f"{report.max_scale_pct:9.3f} {hybrid_ms:10.1f} "
            f"{geometry_ms:8.1f}"
        )
    print("")
    for tool in tools:
        report = reports[tool.name]
        worst = sorted(report.worst, key=lambda item: item[1], reverse=True)[:3]
        for image_name, px, angle, scale in worst:
            print(
                f"  worst {report.name}: {image_name} "
                f"dpx={px:.3f} dang={angle:.3f} dscale={scale:.3f}%"
            )

    verdict = "FAIL" if failures and not args.no_gate else "PASS"
    print("")
    print(f"VERDICT: {verdict}")
    for failure in failures[:20]:
        print(f"  - {failure}")
    if len(failures) > 20:
        print(f"  ... {len(failures) - 20} more")

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "job": args.job,
                    "images": len(images),
                    "thresholds": {
                        "px": args.max_px,
                        "angle": args.max_angle,
                        "scale_pct": args.max_scale_pct,
                        "match_px": args.match_px,
                    },
                    "tools": {
                        name: report.summary()
                        for name, report in reports.items()
                    },
                    "failures": failures,
                    "verdict": verdict,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    return 0 if verdict == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
