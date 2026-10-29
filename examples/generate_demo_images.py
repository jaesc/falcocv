"""Generate the synthetic demo dataset under ``examples/images``.

The scenes are rendered from scratch with NumPy/OpenCV: no third-party images,
models or textures are used, so the output is original artwork of this project
and can be redistributed with it.

Usage:
    python examples/generate_demo_images.py
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "images"
TRAIN_DIR = ROOT / "train"

W, H = 1280, 960
SS = 4  # supersampling factor for the part
PART_W, PART_H = 260, 170
HOLE_R = 22
HOLES = [(-70.0, 0.0), (70.0, 0.0)]


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _rounded_mask(w: int, h: int, radius: int) -> np.ndarray:
    mask = np.zeros((h, w), np.uint8)
    cv2.rectangle(mask, (radius, 0), (w - radius - 1, h - 1), 255, -1)
    cv2.rectangle(mask, (0, radius), (w - 1, h - radius - 1), 255, -1)
    for cx, cy in ((radius, radius), (w - radius - 1, radius),
                   (radius, h - radius - 1), (w - radius - 1, h - radius - 1)):
        cv2.circle(mask, (cx, cy), radius, 255, -1)
    return mask


def _brushed_metal(w: int, h: int, rng: np.random.Generator) -> np.ndarray:
    base = np.empty((h, w, 3), np.float32)
    base[..., 0] = 150.0
    base[..., 1] = 153.0
    base[..., 2] = 158.0
    streak = cv2.GaussianBlur(rng.normal(0, 7.0, (h, w)).astype(np.float32), (1, 15), 0)
    base += streak[..., None]
    vertical = np.linspace(1.07, 0.90, h, dtype=np.float32)[:, None, None]
    base *= vertical
    for _ in range(6):  # faint long scratches in the brushing direction
        y = int(rng.integers(0, h))
        x0 = int(rng.integers(-w // 3, w // 2))
        x1 = x0 + int(rng.integers(w // 3, w))
        shade = float(rng.uniform(-18, 18))
        cv2.line(base, (x0, y), (x1, y), (base[y, max(0, x0)].tolist()[0] + shade,) * 3, 1)
    return np.clip(base, 0, 255)


def _specular(w: int, h: int) -> np.ndarray:
    sheen = np.zeros((h, w), np.float32)
    cv2.ellipse(sheen, (int(w * 0.32), int(h * 0.28)), (int(w * 0.30), int(h * 0.16)),
                12.0, 0, 360, 1.0, -1)
    sheen = cv2.GaussianBlur(sheen, (0, 0), max(w, h) * 0.06)
    return np.clip(sheen, 0, 1)


def render_part(defect: str | None, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return (bgr, alpha) for the part at 1x, centred in its own canvas."""
    w, h = PART_W * SS, PART_H * SS
    mask = _rounded_mask(w, h, 18 * SS)

    # chamfered top-right corner (a distinctive feature for shape matching)
    chamfer = np.zeros_like(mask)
    cut = 34 * SS
    pts = np.array([[w - cut, 0], [w - 1, 0], [w - 1, cut]], np.int32)
    cv2.fillPoly(chamfer, [pts], 255)
    mask[chamfer > 0] = 0

    color = _brushed_metal(w, h, rng)
    sheen = _specular(w, h)
    color = np.clip(color + (sheen * 70.0)[..., None], 0, 255)
    bevel = cv2.GaussianBlur(mask, (0, 0), 3 * SS).astype(np.float32) / 255.0
    color *= (0.72 + 0.28 * bevel)[..., None]

    alpha = mask.copy()

    holes = list(HOLES)
    if defect == "missing_hole":
        holes = holes[:1]
    elif defect == "blocked_hole":
        holes = holes[:1]  # hole filled with metal -> alpha stays 255
    elif defect == "extra_hole":
        holes = holes + [(0.0, -36.0)]
    elif defect == "holes_too_close":
        holes = [(-55.0, 0.0), (55.0, 0.0)]

    for hx, hy in holes:
        radius = HOLE_R
        centre = (int(w / 2 + hx * SS), int(h / 2 + hy * SS))
        if defect == "blocked_hole" and (hx, hy) == HOLES[1]:
            cv2.circle(color, centre, HOLE_R * SS, (120, 126, 132), -1)
            continue
        cv2.circle(alpha, centre, radius * SS, 0, -1)
        cv2.circle(color, centre, radius * SS, (60, 64, 68), -1)

    if defect == "scratch":
        for _ in range(3):
            p0 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
            p1 = (p0[0] + int(rng.integers(-120, 120)) * SS, p0[1] + int(rng.integers(-90, 90)) * SS)
            cv2.line(color, p0, p1, (90, 92, 96), 2 * SS, cv2.LINE_AA)
            cv2.line(color, (p0[0] + 2, p0[1] + 2), (p1[0] + 2, p1[1] + 2), (200, 203, 208), 1 * SS, cv2.LINE_AA)
    elif defect == "stain":
        centre = (int(w * rng.uniform(0.35, 0.65)), int(h * rng.uniform(0.35, 0.65)))
        cv2.ellipse(color, centre, (8 * SS, 6 * SS), float(rng.uniform(0, 180)), 0, 360,
                    (96, 92, 86), -1)
        color = cv2.GaussianBlur(color, (0, 0), 2 * SS)

    bgr = cv2.resize(color, (PART_W, PART_H), interpolation=cv2.INTER_AREA)
    alpha1 = cv2.resize(alpha, (PART_W, PART_H), interpolation=cv2.INTER_AREA)
    return np.clip(bgr, 0, 255).astype(np.uint8), alpha1


def render_background(rng: np.random.Generator) -> np.ndarray:
    base = np.empty((H, W, 3), np.float32)
    base[..., 0] = 40.0
    base[..., 1] = 43.0
    base[..., 2] = 47.0
    blobs = rng.normal(0, 1.0, (H // 16, W // 16)).astype(np.float32)
    blobs = cv2.resize(blobs, (W, H), interpolation=cv2.INTER_CUBIC)
    base += (blobs * 9.0)[..., None]
    grain = rng.normal(0, 3.0, (H, W)).astype(np.float32)
    base += grain[..., None]
    return np.clip(base, 0, 255)


def compose(part_bgr: np.ndarray, part_alpha: np.ndarray, cx: float, cy: float,
            angle: float, scale: float, rng: np.random.Generator) -> np.ndarray:
    ph, pw = part_bgr.shape[:2]
    matrix = cv2.getRotationMatrix2D((pw / 2.0, ph / 2.0), angle, scale)
    matrix[0, 2] += cx - pw / 2.0
    matrix[1, 2] += cy - ph / 2.0
    warped = cv2.warpAffine(part_bgr, matrix, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    warped_alpha = cv2.warpAffine(part_alpha, matrix, (W, H), flags=cv2.INTER_LINEAR,
                                  borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    scene = render_background(rng)

    shadow = np.zeros((H, W), np.float32)
    shadow = cv2.warpAffine(
        (part_alpha > 0).astype(np.float32), matrix, (W, H), flags=cv2.INTER_LINEAR)
    shadow = cv2.GaussianBlur(np.roll(shadow, (16, 12), axis=(0, 1)), (0, 0), 22)
    scene *= (1.0 - 0.55 * np.clip(shadow, 0, 1))[..., None]

    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    light = 1.16 - 0.34 * np.clip(((xx - W * 0.18) ** 2 + (yy - H * 0.12) ** 2) /
                                  (W * W + H * H), 0, 1)
    scene *= light[..., None]

    a = (warped_alpha.astype(np.float32) / 255.0)[..., None]
    scene = scene * (1.0 - a) + warped.astype(np.float32) * a

    radius = np.sqrt(((xx - W / 2) ** 2 + (yy - H / 2) ** 2)) / (0.5 * np.hypot(W, H))
    scene *= (1.0 - 0.22 * radius ** 2)[..., None]

    scene = cv2.GaussianBlur(scene, (0, 0), 0.6)
    scene += rng.normal(0, 2.2, scene.shape).astype(np.float32)
    return np.clip(scene, 0, 255).astype(np.uint8)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)

    rng = _rng(20261029)
    part, alpha = render_part(None, rng)
    reference = compose(part, alpha, W / 2, H / 2, 0.0, 1.0, rng)
    cv2.imwrite(str(TRAIN_DIR / "part_reference.png"), reference)

    cases = (
        [("ok", None)] * 8
        + [("ok", "scratch"), ("ok", "stain")]  # cosmetic marks: must still pass
        + [("ng", defect) for defect in
           ("missing_hole", "blocked_hole", "extra_hole", "holes_too_close")]
    )
    entries = []
    index = 0
    for verdict, defect in cases:
        index += 1
        rng = _rng(1000 + index)
        part, alpha = render_part(defect, rng)
        cx = float(rng.uniform(0.30, 0.70) * W)
        cy = float(rng.uniform(0.34, 0.66) * H)
        angle = float(rng.uniform(-22.0, 22.0))
        scale = float(rng.uniform(0.94, 1.06))
        image = compose(part, alpha, cx, cy, angle, scale, rng)
        if verdict == "ok":
            label = "ok" if not defect else f"ok_{defect}"
        else:
            label = f"ng_{defect}"
        name = f"{index:02d}_{label}.jpg"
        cv2.imwrite(str(OUT_DIR / name), image, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        entries.append({
            "file": name,
            "verdict": verdict,
            "defect": defect or "",
            "cx": round(cx, 2),
            "cy": round(cy, 2),
            "angle": round(angle, 3),
            "scale": round(scale, 4),
        })

    manifest = {
        "generator": "examples/generate_demo_images.py",
        "license": "CC0-1.0 (original synthetic renders, no third-party assets)",
        "camera": {"width": W, "height": H},
        "part": {"width": PART_W, "height": PART_H, "hole_radius": HOLE_R,
                 "hole_spacing_px": HOLES[1][0] - HOLES[0][0]},
        "verdict_rule": ("OK when exactly 2 holes are found and the hole distance "
                         "is within 12.5-15.5 mm (10 px/mm); cosmetic marks are ignored"),
        "images": entries,
    }
    (OUT_DIR / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {len(entries)} images to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
