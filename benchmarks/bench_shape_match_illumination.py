"""Synthetic illumination stress test for ShapeMatch pre-gate variants.

The script does not use private jobs or images.  It renders a known synthetic
scene, applies controlled illumination changes and compares the production
grayscale-NCC pre-gate with the geometric prototypes.
"""

from __future__ import annotations

import argparse
import statistics
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
import falcocv.tools.shape_match.logic as logic  # noqa: E402


def _scene(scale: str):
    template = make_template()
    if scale == "5mp":
        template = cv2.resize(template, (404, 407), interpolation=cv2.INTER_CUBIC)
        scene_size = (1940, 2588)
    else:
        scene_size = (360, 480)
    image = render_scene(
        template,
        cx=scene_size[1] / 2.0,
        cy=scene_size[0] / 2.0,
        angle=12.0,
        scale=1.05,
        gamma=0.9,
        noise=3.0,
        scene_size=scene_size,
        seed=7,
    )
    return template, image, (
        scene_size[1] / 2.0,
        scene_size[0] / 2.0,
        12.0,
        1.05,
    )


def _transform(image: np.ndarray, name: str) -> np.ndarray:
    source = image.astype(np.float32)
    if name == "identity":
        result = source
    elif name == "bright+30":
        result = source + 30.0
    elif name == "bright-30":
        result = source - 30.0
    elif name == "contrast0.7":
        result = (source - 128.0) * 0.7 + 128.0
    elif name == "contrast1.3":
        result = (source - 128.0) * 1.3 + 128.0
    elif name == "gamma0.7":
        result = np.power(np.clip(source / 255.0, 0.0, 1.0), 0.7) * 255.0
    elif name == "gamma1.4":
        result = np.power(np.clip(source / 255.0, 0.0, 1.0), 1.4) * 255.0
    elif name == "bright+70":
        result = source + 70.0
    elif name == "bright-70":
        result = source - 70.0
    elif name == "contrast0.4":
        result = (source - 128.0) * 0.4 + 128.0
    elif name == "contrast1.8":
        result = (source - 128.0) * 1.8 + 128.0
    elif name == "gamma0.45":
        result = np.power(np.clip(source / 255.0, 0.0, 1.0), 0.45) * 255.0
    elif name == "gamma2.0":
        result = np.power(np.clip(source / 255.0, 0.0, 1.0), 2.0) * 255.0
    elif name == "lowlight":
        result = source * 0.35
    elif name == "shadow":
        height, width = source.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        mask = 0.45 + 0.55 * (xx / max(1, width - 1))
        result = source * mask[..., None]
    elif name == "deep_shadow":
        height, width = source.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        mask = 0.12 + 0.88 * (xx / max(1, width - 1))
        result = source * mask[..., None]
    elif name == "blur":
        result = cv2.GaussianBlur(source, (5, 5), 1.0)
    elif name == "strong_blur":
        result = cv2.GaussianBlur(source, (9, 9), 2.0)
    elif name == "bright+gamma":
        result = (
            np.power(np.clip((source + 25.0) / 255.0, 0.0, 1.0), 0.75) * 255.0
        )
    elif name == "shadow+gamma":
        height, width = source.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width]
        mask = 0.30 + 0.70 * (xx / max(1, width - 1))
        shadowed = source * mask[..., None]
        result = (
            np.power(np.clip(shadowed / 255.0, 0.0, 1.0), 0.7) * 255.0
        )
    else:
        raise ValueError(name)
    return np.clip(result, 0, 255).astype(np.uint8)


def _make_tool(template):
    tool = logic.ShapeMatch()
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


def _error(tool, truth):
    if not tool.outputs.get("count"):
        return None
    result = tool.outputs["results"][0]
    return {
        "px": float(
            ((result.x - truth[0]) ** 2 + (result.y - truth[1]) ** 2) ** 0.5
        ),
        "deg": float(abs(result.angle - truth[2])),
        "scale": float(abs(result.scale - truth[3]) / truth[3]),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=("480", "5mp"), default="480")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument(
        "--variants",
        default="ncc,alignment,edge,geometry",
        help="comma-separated pre-gate variants (geometry is a match mode)",
    )
    args = parser.parse_args(argv)

    template, base_image, truth = _scene(args.scene)
    variants = [item.strip() for item in args.variants.split(",") if item.strip()]
    transforms = (
        "identity",
        "bright+30",
        "bright-30",
        "contrast0.7",
        "contrast1.3",
        "contrast0.4",
        "contrast1.8",
        "gamma0.7",
        "gamma1.4",
        "gamma0.45",
        "gamma2.0",
        "bright+70",
        "bright-70",
        "lowlight",
        "shadow",
        "deep_shadow",
        "blur",
        "strong_blur",
        "bright+gamma",
        "shadow+gamma",
    )
    tools = {name: _make_tool(template) for name in variants}
    records = {
        name: {transform: [] for transform in transforms} for name in variants
    }

    for transform in transforms:
        image = _transform(base_image, transform)
        for name in variants:
            tool = tools[name]
            if name == "geometry":
                tool.params["match_mode"] = "geometry"
                tool.params["pre_gate_scorer"] = "alignment"
            else:
                tool.params["match_mode"] = "hybrid"
                tool.params["pre_gate_scorer"] = name
            tool.inputs["input image"] = VisionImage(image)
            tool.run()  # warm
            for _ in range(args.rounds):
                tool.inputs["input image"] = VisionImage(image)
                start = time.perf_counter()
                tool.run()
                elapsed = (time.perf_counter() - start) * 1000.0
                records[name][transform].append(
                    {
                        "ms": elapsed,
                        "error": _error(tool, truth),
                    }
                )

    print(f"scene={args.scene} rounds={args.rounds}")
    header = "transform".ljust(14)
    for name in variants:
        header += f" | {name:>10s} det/px/deg/scale/ms"
    print(header)
    for transform in transforms:
        line = transform.ljust(14)
        for name in variants:
            rows = records[name][transform]
            detections = sum(1 for row in rows if row["error"])
            errors = [row["error"] for row in rows if row["error"]]
            if errors:
                px = max(error["px"] for error in errors)
                deg = max(error["deg"] for error in errors)
                scale = max(error["scale"] for error in errors) * 100.0
            else:
                px = deg = scale = float("nan")
            ms = statistics.median(row["ms"] for row in rows)
            line += (
                f" | {detections}/{len(rows)}"
                f" {px:5.3f}/{deg:5.3f}/{scale:5.3f}/{ms:6.1f}"
            )
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
