"""Synthetic scenes with known pose for matcher evaluation.

The template is a texture-light industrial-looking part.  Scenes are generated
by warping the template to a known (x, y, angle, scale), then applying noise,
blur, gamma/contrast and occlusion.  Ground truth is therefore exact.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

BACKGROUND = 64
SCENE_SIZE = (360, 480)


@dataclass(frozen=True)
class SyntheticScene:
    name: str
    image: np.ndarray
    template: np.ndarray
    cx: float
    cy: float
    angle: float
    scale: float
    occlusion: float = 0.0
    gamma: float = 1.0
    noise: float = 0.0
    blur: float = 0.0


@dataclass(frozen=True)
class PoseError:
    translation_px: float | None
    rotation_deg: float | None
    scale_rel: float | None


def make_template() -> np.ndarray:
    """A bright bracket with a dark hole, a diagonal rib and a notch."""
    template = np.full((90, 120, 3), BACKGROUND, dtype=np.uint8)
    cv2.rectangle(template, (18, 18), (102, 72), (210, 210, 210), -1)
    cv2.circle(template, (60, 45), 12, (BACKGROUND,) * 3, -1)
    cv2.line(template, (18, 72), (102, 18), (120, 120, 120), 4)
    cv2.rectangle(template, (25, 28), (38, 38), (BACKGROUND,) * 3, -1)
    return template


def _apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    if gamma == 1.0:
        return image
    lut = np.array(
        [((value / 255.0) ** gamma) * 255.0 for value in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(image, lut)


def render_scene(
    template: np.ndarray,
    cx: float,
    cy: float,
    angle: float = 0.0,
    scale: float = 1.0,
    occlusion: float = 0.0,
    gamma: float = 1.0,
    noise: float = 0.0,
    blur: float = 0.0,
    scene_size: tuple[int, int] = SCENE_SIZE,
    seed: int = 0,
) -> np.ndarray:
    height, width = template.shape[:2]
    scene_height, scene_width = scene_size
    scene = np.full((scene_height, scene_width, 3), BACKGROUND, dtype=np.uint8)

    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
    matrix[0, 2] += cx - width / 2.0
    matrix[1, 2] += cy - height / 2.0

    warped = cv2.warpAffine(
        template,
        matrix,
        (scene_width, scene_height),
        flags=cv2.INTER_LINEAR,
        borderValue=(BACKGROUND, BACKGROUND, BACKGROUND),
    )
    mask = cv2.warpAffine(
        np.full((height, width), 255, dtype=np.uint8),
        matrix,
        (scene_width, scene_height),
        flags=cv2.INTER_NEAREST,
        borderValue=0,
    )
    scene = np.where(mask[..., None] > 0, warped, scene)

    if blur > 0:
        scene = cv2.GaussianBlur(scene, (0, 0), blur)

    if occlusion > 0:
        span = max(0.0, min(1.0, float(occlusion))) ** 0.5
        occ_w = max(1, int(round(width * scale * span)))
        occ_h = max(1, int(round(height * scale * span)))
        x0 = int(round(cx - occ_w / 2))
        y0 = int(round(cy - occ_h / 2))
        cv2.rectangle(
            scene,
            (x0, y0),
            (x0 + occ_w, y0 + occ_h),
            (BACKGROUND, BACKGROUND, BACKGROUND),
            -1,
        )

    scene = _apply_gamma(scene, gamma)

    if noise > 0:
        rng = np.random.default_rng(seed)
        scene = np.clip(
            scene.astype(np.float32) + rng.normal(0.0, noise, scene.shape),
            0,
            255,
        ).astype(np.uint8)

    return scene


def scene_suite(template: np.ndarray | None = None) -> list[SyntheticScene]:
    template = make_template() if template is None else template
    specs = [
        ("baseline", 300, 190, 0.0, 1.0, {}),
        ("translation", 350, 230, 0.0, 1.0, {}),
        ("rotation+15", 300, 190, 15.0, 1.0, {}),
        ("rotation-20", 300, 190, -20.0, 1.0, {}),
        ("scale0.9", 300, 190, 0.0, 0.9, {}),
        ("scale1.1", 300, 190, 0.0, 1.1, {}),
        ("gamma0.7", 300, 190, 0.0, 1.0, {"gamma": 0.7}),
        ("gamma1.4", 300, 190, 0.0, 1.0, {"gamma": 1.4}),
        ("noise8", 300, 190, 0.0, 1.0, {"noise": 8.0}),
        ("blur1.5", 300, 190, 0.0, 1.0, {"blur": 1.5}),
        ("occlusion20", 300, 190, 0.0, 1.0, {"occlusion": 0.2}),
        ("occlusion30", 300, 190, 0.0, 1.0, {"occlusion": 0.3}),
        (
            "combo",
            300,
            190,
            12.0,
            1.05,
            {"gamma": 0.85, "noise": 4.0, "occlusion": 0.2},
        ),
    ]

    scenes: list[SyntheticScene] = []
    for seed, (name, cx, cy, angle, scale, effects) in enumerate(specs):
        scenes.append(
            SyntheticScene(
                name=name,
                image=render_scene(
                    template,
                    cx=cx,
                    cy=cy,
                    angle=angle,
                    scale=scale,
                    seed=seed,
                    **effects,
                ),
                template=template,
                cx=float(cx),
                cy=float(cy),
                angle=float(angle),
                scale=float(scale),
                occlusion=float(effects.get("occlusion", 0.0)),
                gamma=float(effects.get("gamma", 1.0)),
                noise=float(effects.get("noise", 0.0)),
                blur=float(effects.get("blur", 0.0)),
            )
        )
    return scenes


def pose_error(
    gt: SyntheticScene,
    x: float | None,
    y: float | None,
    angle: float | None = None,
    scale: float | None = None,
) -> PoseError:
    translation = None
    if x is not None and y is not None:
        translation = float(np.hypot(x - gt.cx, y - gt.cy))

    rotation = None
    if angle is not None:
        diff = (float(angle) - gt.angle + 180.0) % 360.0 - 180.0
        rotation = abs(diff)

    scale_rel = None
    if scale is not None and gt.scale:
        scale_rel = abs(float(scale) - gt.scale) / abs(gt.scale)

    return PoseError(translation, rotation, scale_rel)
