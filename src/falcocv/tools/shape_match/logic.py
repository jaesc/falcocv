"""ShapeMatch v3 - gradient/edge based localization.

The native kernel proposes (x, y, angle, scale) hypotheses.  A local NCC
refinement aligns rotation/translation and an explicit alignment score selects
one of two solvers: ECC for fully visible parts, or a distance-weighted
gradient/ICP solver for occluded parts.  There are no silent fallbacks.

The native ``_falco_kernel`` extension is required for matching.
"""

from __future__ import annotations

import base64
import io
import math

import cv2
import numpy as np

from falcocv.core.roi_utils import extract_roi
from falcocv.core.tool_base import BaseTool, ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.pointcloud_utils import propagate_3d

try:
    from _falco_kernel import (
        shape_match_icp as _kernel_icp,
        shape_match_score as _kernel_score,
        shape_match_vote as _kernel_vote,
    )
except ImportError:
    try:
        from falcocv._falco_kernel import (
            shape_match_icp as _kernel_icp,
            shape_match_score as _kernel_score,
            shape_match_vote as _kernel_vote,
        )
    except ImportError:
        _kernel_icp = None
        _kernel_score = None
        _kernel_vote = None

MODEL_VERSION = 2

# Angle conventions
# -----------------
# Public angles (params, results, overlays and fixture) follow the framework
# convention: positive = clockwise on screen (Qt / ROI / fixture).
# Internal ``theta`` values used by the kernel, NCC, ECC and ICP follow
# OpenCV's convention: positive = counter-clockwise.  ``theta = -angle``.
#
# The training ROI is already expressed in the framework convention, so the
# crop below can keep using ``cv2.getRotationMatrix2D(+roi_angle)``: rotating
# the content counter-clockwise straightens a clockwise-rotated ROI.

_EDGE_LOW = 50
_EDGE_HIGH = 150
_ORIENTATION_EXPONENT = 2.0

# Distance weighting used by the occlusion-tolerant score.
_DISTANCE_SIGMA = 1.0

# Solver selection.  ``alignment`` is the gradient alignment score (0..1).
# Visible parts use ECC; occluded parts use the robust solver.  If ECC fails
# just above the threshold, the candidate is reclassified as occlusion; well
# above the threshold it is rejected with a warning.
_ECC_ALIGNMENT = 0.90
_ECC_REJECT_ALIGNMENT = 0.98
_ECC_MIN_CC = 0.85
_ECC_POLISH_MIN_CC = 0.995
_MIN_NCC = 0.30
_MAX_MATCHES = 100
_MAX_REFINE = 16
_MAX_NCC_CANDIDATES = 10
_MAX_NCC_SCANS = 6
_MAX_PRE_GATE = 32
_MAX_SOLVE = 12
_MAX_KERNEL_CANDIDATES = 1024
_VOTE_PEAKS_PER_HYPOTHESIS = 32

# Coarse voting is deliberately cheap, but must not depend on the ROI extent:
# the downscale is derived from the template and the edge budget is a fixed
# per-cell quota.  Full-resolution refinement uses the complete model.
_VOTE_MODEL_FEATURES = 500
_VOTE_EDGE_QUOTA = 48
_VOTE_MIN_TEMPLATE_PX = 40
_VOTE_TARGET_TEMPLATE_PX = 96
_VOTE_MAX_FACTOR = 4
_VOTE_EDGE_CELL = 32
_VOTE_MIN_PEAK = 0.02

# NCC angle/scale scans run on a reduced view of the search ROI.  The factor is
# derived from the template size so the template keeps enough detail for a
# reliable NCC peak; nothing here depends on the job or the ROI position.
_SEARCH_MIN_TEMPLATE_PX = 48
_SEARCH_MAX_FACTOR = 8
_ECC_COARSE_MAX_FACTOR = 2


class ShapeMatchResult(ToolResult):
    def __init__(
        self,
        id,
        x,
        y,
        angle,
        scale,
        score,
        solver=None,
        ncc_score=None,
        visible_fraction=None,
        ecc_cc=None,
        warnings=None,
    ):
        super().__init__(
            id=id,
            x=x,
            y=y,
            angle=angle,
            scale=scale,
            score=score,
            solver=solver,
            ncc_score=ncc_score,
            visible_fraction=visible_fraction,
            ecc_cc=ecc_cc,
            warnings=list(warnings or []),
        )


@register_tool(prefix="[SHAP]", default_name="Shape Match", group="Vision Tools")
class ShapeMatch(BaseTool):
    """Rotation/scale/translation shape matcher with explicit solvers."""

    def __init__(self, name="Shape Match"):
        super().__init__(name)
        self.params = {
            "train_roi": [0, 0, 0, 0, 0],
            "search_roi": [0, 0, 0, 0, 0],
            "angle_start": -30.0,
            "angle_end": 30.0,
            "scale_start": 0.9,
            "scale_end": 1.1,
            "min_score": 0.45,
            "max_matches": 1,
            "max_features": 800,
            "origin_x": 0.0,
            "origin_y": 0.0,
            "show_search_roi": True,
            "show_found_region": True,
            "show_found_origin": True,
            "show_labels": True,
            "show_overlays": True,
            "act_as_fixture": False,
            "copy_overlays_to_input": True,
            "model_b64": None,
            "train_image_b64": None,
            "train_fixture": [0.0, 0.0, 0.0],
            "is_trained": False,
            "template_size": [0, 0],
        }
        self.inputs = {"input image": None}
        self.outputs = {
            "output image": None,
            "count": 0,
            "trained": False,
            "results": ToolResults(),
            "warnings": [],
        }
        self._param_keys = set(self.params)
        self._static_train_image = None
        self._static_train_fixture = {"x": 0.0, "y": 0.0, "angle": 0.0}
        self._template_gray = None
        self._template_color = None
        self._template_border = 0
        self._search_template_gray = None
        self._search_factor = 1
        self._ecc_template_gray = None
        self._ecc_factor = 1
        self._model_x = np.empty(0, dtype=np.float32)
        self._model_y = np.empty(0, dtype=np.float32)
        self._model_theta = np.empty(0, dtype=np.float32)
        self._model_mag = np.empty(0, dtype=np.float32)
        self._vote_model_x = np.empty(0, dtype=np.float32)
        self._vote_model_y = np.empty(0, dtype=np.float32)
        self._vote_model_theta = np.empty(0, dtype=np.float32)
        self._vote_model_mag = np.empty(0, dtype=np.float32)
        self._template_edges = None
        self._template_size = (0, 0)

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    @property
    def is_trained(self) -> bool:
        return (
            self._model_x.size > 0
            and self._template_edges is not None
            and self._template_gray is not None
            and self._template_size[0] > 0
        )

    def _gray(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image

    def _gradient(self, gray: np.ndarray):
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        theta = np.arctan2(gy, gx).astype(np.float32)
        return gx, gy, mag, theta

    def _build_model(self, image: np.ndarray) -> None:
        gray = self._gray(image)
        height, width = gray.shape[:2]
        _, _, mag, theta = self._gradient(gray)
        edges = cv2.Canny(gray, _EDGE_LOW, _EDGE_HIGH)

        ys, xs = np.nonzero(edges)
        if len(xs) < 20:
            threshold = np.percentile(mag, 90)
            ys, xs = np.nonzero(mag >= threshold)

        if len(xs) == 0:
            raise ValueError("Template has no usable edges.")

        values = mag[ys, xs]
        order = np.argsort(values)[::-1]
        limit = max(20, int(self.params["max_features"]))
        order = order[:limit]
        xs = xs[order].astype(np.float32)
        ys = ys[order].astype(np.float32)
        values = values[order].astype(np.float32)
        angles = theta[ys.astype(int), xs.astype(int)].astype(np.float32)

        total = float(values.sum())
        if total <= 0:
            raise ValueError("Template gradient magnitude is zero.")

        self._model_x = xs - width / 2.0
        self._model_y = ys - height / 2.0
        self._model_theta = angles
        self._model_mag = values / total
        self._template_edges = edges
        if gray.dtype == np.uint8:
            template_gray = gray
        else:
            template_gray = cv2.normalize(
                gray, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)
        self._template_gray = np.ascontiguousarray(template_gray)
        self._update_template_border()
        self._update_search_template()
        if image.ndim == 3:
            color = image
        else:
            color = cv2.cvtColor(template_gray, cv2.COLOR_GRAY2BGR)
        self._template_color = np.ascontiguousarray(color)
        self._template_size = (width, height)
        self.params["template_size"] = [width, height]
        self._rebuild_vote_model()
        self.outputs["trained"] = True

    def _update_template_border(self) -> None:
        if self._template_gray is None or self._template_gray.size == 0:
            self._template_border = 0
            return
        gray = self._template_gray
        parts = [
            gray[:2, :].ravel(),
            gray[-2:, :].ravel(),
            gray[:, :2].ravel(),
            gray[:, -2:].ravel(),
        ]
        self._template_border = int(np.median(np.concatenate(parts)))

    def _update_search_template(self) -> None:
        """Precompute the reduced templates used by the coarse stages."""
        if self._template_gray is None or self._template_gray.size == 0:
            self._search_template_gray = None
            self._search_factor = 1
            self._ecc_template_gray = None
            self._ecc_factor = 1
            return
        height, width = self._template_gray.shape[:2]
        factor = min(width, height) // _SEARCH_MIN_TEMPLATE_PX
        factor = max(1, min(_SEARCH_MAX_FACTOR, factor))
        self._search_template_gray = self._reduced_template(factor)
        self._search_factor = factor
        ecc_factor = max(1, min(_ECC_COARSE_MAX_FACTOR, factor))
        self._ecc_template_gray = self._reduced_template(ecc_factor)
        self._ecc_factor = ecc_factor

    def _reduced_template(self, factor: int):
        if factor <= 1:
            return self._template_gray
        height, width = self._template_gray.shape[:2]
        size = (
            max(8, int(round(width / factor))),
            max(8, int(round(height / factor))),
        )
        return np.ascontiguousarray(
            cv2.resize(self._template_gray, size, interpolation=cv2.INTER_AREA)
        )

    def _rebuild_vote_model(self) -> None:
        if self._model_x.size == 0:
            self._vote_model_x = np.empty(0, dtype=np.float32)
            self._vote_model_y = np.empty(0, dtype=np.float32)
            self._vote_model_theta = np.empty(0, dtype=np.float32)
            self._vote_model_mag = np.empty(0, dtype=np.float32)
            return
        count = min(_VOTE_MODEL_FEATURES, self._model_x.size)
        order = np.argsort(self._model_mag)[::-1][:count]
        self._vote_model_x = self._model_x[order].astype(np.float32)
        self._vote_model_y = self._model_y[order].astype(np.float32)
        self._vote_model_theta = self._model_theta[order].astype(np.float32)
        weights = self._model_mag[order].astype(np.float32)
        total = float(weights.sum())
        self._vote_model_mag = weights / total if total > 0 else weights

    def train_pattern(self) -> None:
        image = self._static_train_image
        if image is None:
            raise ValueError("No training image captured.")
        if image.size == 0:
            raise ValueError("Training image is empty.")

        roi_data = self.params["train_roi"]
        if len(roi_data) == 4:
            x, y, w, h = (float(v) for v in roi_data)
            angle = 0.0
        elif len(roi_data) == 5:
            x, y, w, h, angle = (float(v) for v in roi_data)
        else:
            raise ValueError(
                f"train_roi must have 4 or 5 elements, got {len(roi_data)}"
            )

        if w <= 0 or h <= 0:
            crop = image
        else:
            fixture = self._static_train_fixture
            if (
                abs(fixture.get("x", 0.0)) > 0.01
                or abs(fixture.get("y", 0.0)) > 0.01
                or abs(fixture.get("angle", 0.0)) > 0.01
            ):
                tmp = VisionImage(
                    image,
                    x=fixture["x"],
                    y=fixture["y"],
                    angle=fixture["angle"],
                )
                x, y, w, h, angle = tmp.fixture_to_image([x, y, w, h, angle])

            image_h, image_w = image.shape[:2]
            x1 = max(int(x), 0)
            y1 = max(int(y), 0)
            x2 = min(int(x + w), image_w)
            y2 = min(int(y + h), image_h)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(
                    f"Training ROI is outside the image ({image_w}x{image_h})."
                )
            center = (x1 + (x2 - x1) / 2.0, y1 + (y2 - y1) / 2.0)
            matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            warped = cv2.warpAffine(
                image, matrix, (image_w, image_h), flags=cv2.INTER_LINEAR
            )
            crop = warped[y1:y2, x1:x2]

        self._build_model(crop)
        self.params["is_trained"] = True
        self._save_model()
        self.data_changed.emit()

    def train_from_roi(self, x, y, w, h, a=0.0) -> None:
        self.params["train_roi"] = [x, y, w, h, a]
        self.train_pattern()

    def save_train_image(self) -> None:
        source = self.inputs.get("input image")
        if source is None:
            return
        data = source.data if isinstance(source, VisionImage) else source
        if not isinstance(data, np.ndarray):
            return
        self._static_train_image = data.copy()
        if isinstance(source, VisionImage):
            fixture = {
                "x": float(source.x),
                "y": float(source.y),
                "angle": float(source.angle),
            }
        else:
            fixture = {"x": 0.0, "y": 0.0, "angle": 0.0}
        self._static_train_fixture = fixture
        self.params["train_fixture"] = [
            fixture["x"],
            fixture["y"],
            fixture["angle"],
        ]
        ok, buffer = cv2.imencode(".png", data)
        if ok:
            self.params["train_image_b64"] = base64.b64encode(buffer).decode("ascii")
        self.data_changed.emit()

    def clear_training(self, keep_image=False) -> None:
        self._model_x = np.empty(0, dtype=np.float32)
        self._model_y = np.empty(0, dtype=np.float32)
        self._model_theta = np.empty(0, dtype=np.float32)
        self._model_mag = np.empty(0, dtype=np.float32)
        self._vote_model_x = np.empty(0, dtype=np.float32)
        self._vote_model_y = np.empty(0, dtype=np.float32)
        self._vote_model_theta = np.empty(0, dtype=np.float32)
        self._vote_model_mag = np.empty(0, dtype=np.float32)
        self._template_edges = None
        self._template_gray = None
        self._template_color = None
        self._template_border = 0
        self._search_template_gray = None
        self._search_factor = 1
        self._ecc_template_gray = None
        self._ecc_factor = 1
        self._template_size = (0, 0)
        self._static_train_fixture = {"x": 0.0, "y": 0.0, "angle": 0.0}
        self.params.update(
            {
                "model_b64": None,
                "is_trained": False,
                "template_size": [0, 0],
                "train_fixture": [0.0, 0.0, 0.0],
            }
        )
        if not keep_image:
            self._static_train_image = None
            self.params["train_image_b64"] = None
        self.outputs["trained"] = False
        self.data_changed.emit()

    def _save_model(self) -> None:
        if self._template_color is None or self._template_gray is None:
            raise ValueError("Cannot save an empty ShapeMatch model.")
        buffer = io.BytesIO()
        np.savez_compressed(
            buffer,
            version=np.array([MODEL_VERSION], dtype=np.int32),
            x=self._model_x,
            y=self._model_y,
            theta=self._model_theta,
            mag=self._model_mag,
            edges=self._template_edges,
            gray=self._template_gray,
            color=self._template_color,
            size=np.array(self._template_size, dtype=np.int32),
        )
        self.params["model_b64"] = base64.b64encode(buffer.getvalue()).decode("ascii")

    def _load_model(self) -> None:
        encoded = self.params.get("model_b64")
        if not encoded or not self.params.get("is_trained"):
            return
        raw = base64.b64decode(encoded)
        with np.load(io.BytesIO(raw), allow_pickle=False) as data:
            if int(data["version"][0]) != MODEL_VERSION:
                raise ValueError("Unsupported ShapeMatch model version; retrain.")
            if "gray" not in data or "color" not in data:
                raise ValueError("ShapeMatch model has no template images; retrain.")
            self._model_x = data["x"].astype(np.float32)
            self._model_y = data["y"].astype(np.float32)
            self._model_theta = data["theta"].astype(np.float32)
            self._model_mag = data["mag"].astype(np.float32)
            self._template_edges = data["edges"].astype(np.uint8)
            self._template_gray = np.ascontiguousarray(data["gray"].astype(np.uint8))
            self._template_color = np.ascontiguousarray(data["color"].astype(np.uint8))
            self._update_template_border()
            self._update_search_template()
            self._template_size = (int(data["size"][0]), int(data["size"][1]))
            self._rebuild_vote_model()
            self.outputs["trained"] = True

    def _restore_state_from_params(self) -> None:
        if self.params.get("train_image_b64"):
            raw = base64.b64decode(self.params["train_image_b64"])
            buffer = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Could not restore the ShapeMatch training image.")
            self._static_train_image = image
        fixture = self.params.get("train_fixture", [0.0, 0.0, 0.0])
        if len(fixture) >= 3:
            self._static_train_fixture = {
                "x": float(fixture[0]),
                "y": float(fixture[1]),
                "angle": float(fixture[2]),
            }
        if self.params.get("is_trained"):
            self._load_model()

    def _ensure_state(self) -> None:
        if self._static_train_image is None and self.params.get("train_image_b64"):
            raw = base64.b64decode(self.params["train_image_b64"])
            buffer = np.frombuffer(raw, dtype=np.uint8)
            image = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
            if image is None:
                raise ValueError("Could not restore the ShapeMatch training image.")
            self._static_train_image = image
        if self._template_gray is None and self.params.get("model_b64"):
            self._load_model()

    def deserialize(self, data: dict, job_manager=None) -> None:
        super().deserialize(data, job_manager)
        for key in list(self.params):
            if key not in self._param_keys:
                del self.params[key]
        self._restore_state_from_params()

    # ------------------------------------------------------------------
    # Geometry helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _even_diagonal(width: int, height: int, scale: float) -> int:
        diagonal = int(math.ceil(math.hypot(width * scale, height * scale))) + 4
        if diagonal % 2:
            diagonal += 1
        return diagonal

    def _pose_matrix(self, angle: float, scale: float, diagonal: int,
                     width: int | None = None, height: int | None = None) -> np.ndarray:
        if width is None or height is None:
            width, height = self._template_size
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        matrix[0, 2] += diagonal / 2.0 - width / 2.0
        matrix[1, 2] += diagonal / 2.0 - height / 2.0
        return matrix

    def _origin_point(self, x, y, angle, scale):
        """Offset origin in image space.

        ``angle`` is a public/framework angle (positive = clockwise), so the
        local offset is rotated clockwise around the template centre.
        """
        origin_x = float(self.params.get("origin_x", 0.0))
        origin_y = float(self.params.get("origin_y", 0.0))
        if abs(origin_x) < 1e-9 and abs(origin_y) < 1e-9:
            return x, y
        radians = math.radians(angle)
        cos_a = math.cos(radians)
        sin_a = math.sin(radians)
        tx = origin_x * scale
        ty = origin_y * scale
        return x + tx * cos_a - ty * sin_a, y + tx * sin_a + ty * cos_a

    def _render_gray(self, angle: float, scale: float,
                     template_gray: np.ndarray | None = None) -> np.ndarray | None:
        template = self._template_gray if template_gray is None else template_gray
        if template is None:
            return None
        height, width = template.shape[:2]
        diagonal = self._even_diagonal(width, height, scale)
        matrix = self._pose_matrix(angle, scale, diagonal, width, height)
        return cv2.warpAffine(
            template,
            matrix,
            (diagonal, diagonal),
            flags=cv2.INTER_LINEAR,
            borderValue=int(self._template_border),
        )

    def _render_search(self, angle: float, scale: float,
                       template_gray: np.ndarray | None = None):
        """Render the template cropped to its rotated bounding box.

        The diagonal padding of ``_render_gray`` is constant border colour and
        only adds work to every ``matchTemplate`` call.  A rotated rectangle is
        centrally symmetric, so the crop stays centred on the template centre.
        """
        template = self._template_gray if template_gray is None else template_gray
        if template is None:
            return None
        height, width = template.shape[:2]
        diagonal = self._even_diagonal(width, height, scale)
        matrix = self._pose_matrix(angle, scale, diagonal, width, height)
        rendered = cv2.warpAffine(
            template,
            matrix,
            (diagonal, diagonal),
            flags=cv2.INTER_LINEAR,
            borderValue=int(self._template_border),
        )
        corners = cv2.transform(
            np.array(
                [[[0.0, 0.0], [width, 0.0], [width, height], [0.0, height]]],
                dtype=np.float32,
            ),
            matrix,
        )[0]
        center = diagonal / 2.0
        half_w = int(math.ceil(float(np.abs(corners[:, 0] - center).max())))
        half_h = int(math.ceil(float(np.abs(corners[:, 1] - center).max())))
        half_w = min(half_w, int(center) - 1)
        half_h = min(half_h, int(center) - 1)
        if half_w < 2 or half_h < 2:
            return None
        left = int(center) - half_w
        top = int(center) - half_h
        return rendered[top:top + 2 * half_h, left:left + 2 * half_w]

    # ------------------------------------------------------------------
    # Coarse candidates
    # ------------------------------------------------------------------
    def _subpixel_peak(self, response: np.ndarray, x: int, y: int):
        def offset(left, center, right):
            denominator = left - 2.0 * center + right
            if abs(denominator) < 1e-9:
                return 0.0
            return 0.5 * (left - right) / denominator

        dx = dy = 0.0
        if 0 < x < response.shape[1] - 1:
            dx = offset(response[y, x - 1], response[y, x], response[y, x + 1])
        if 0 < y < response.shape[0] - 1:
            dy = offset(response[y - 1, x], response[y, x], response[y + 1, x])
        return dx, dy

    def _kernel_coarse(self, mag, theta, scene_edges, angles, scales,
                       model_scale_x=1.0, model_scale_y=1.0):
        if _kernel_vote is None:
            raise RuntimeError(
                "ShapeMatch requires the native _falco_kernel extension. "
                "Build it with build_kernel.bat or install the wheel."
            )
        if self._vote_model_x.size == 0:
            raise ValueError("ShapeMatch model has no voting features; retrain.")
        try:
            result = _kernel_vote(
                scene_mag=np.ascontiguousarray(mag, dtype=np.float32),
                scene_theta=np.ascontiguousarray(theta, dtype=np.float32),
                scene_mask=np.ascontiguousarray(scene_edges, dtype=np.uint8),
                model_x=np.ascontiguousarray(
                    self._vote_model_x / model_scale_x, dtype=np.float32
                ),
                model_y=np.ascontiguousarray(
                    self._vote_model_y / model_scale_y, dtype=np.float32
                ),
                model_theta=self._vote_model_theta,
                model_mag=self._vote_model_mag,
                angles=np.ascontiguousarray(angles, dtype=np.float32),
                scales=np.ascontiguousarray(scales, dtype=np.float32),
                bins=32,
                min_scene_mag=5.0,
                peaks_per_hypothesis=_VOTE_PEAKS_PER_HYPOTHESIS,
                max_candidates=_MAX_KERNEL_CANDIDATES,
            )
        except Exception as exc:
            raise RuntimeError(f"shape_match_vote failed: {exc}") from exc
        if not result:
            return []
        if "error" in result:
            raise RuntimeError(f"shape_match_vote failed: {result['error']}")
        candidates = []
        for x, y, angle, scale, votes in zip(
            result["x"], result["y"], result["angle"], result["scale"], result["votes"]
        ):
            candidates.append(
                {
                    "x": float(x),
                    "y": float(y),
                    "angle": float(angle),
                    "scale": float(scale),
                    "peak": float(votes),
                }
            )
        return candidates

    @staticmethod
    def _dedupe_candidates(candidates, min_distance,
                           angle_tolerance=180.0, scale_tolerance=1.0):
        """Drop only near-identical hypotheses.

        Two peaks at the same position but different angle or scale are
        different hypotheses and must both reach refinement; merging them by
        position alone can discard the correct scale before the (local) NCC
        scan ever sees it.
        """
        kept = []
        for candidate in candidates:
            duplicate = False
            for other in kept:
                if math.hypot(
                    candidate["x"] - other["x"], candidate["y"] - other["y"]
                ) >= min_distance:
                    continue
                if abs(candidate["angle"] - other["angle"]) > angle_tolerance:
                    continue
                if abs(candidate["scale"] / other["scale"] - 1.0) > scale_tolerance:
                    continue
                duplicate = True
                break
            if not duplicate:
                kept.append(candidate)
        return kept

    def _vote_factor(self, roi_width=0, roi_height=0) -> int:
        """Vote downscale, derived from the template only.

        The ROI extent must not change the vote resolution: a bigger search
        area may add candidates but never take detail away from the template.
        """
        min_template = min(self._template_size) if self._template_size[0] else 0
        if min_template <= 0:
            return 1
        factor = max(1, min_template // _VOTE_TARGET_TEMPLATE_PX)
        return max(1, min(_VOTE_MAX_FACTOR, factor))

    @staticmethod
    def _cap_scene_edges(mag, edges, cell=_VOTE_EDGE_CELL, quota=_VOTE_EDGE_QUOTA):
        # A fixed per-cell quota keeps the same edge budget for a given part
        # regardless of how much background the ROI contains.  A global target
        # divided by the number of cells made bigger ROIs starve the object.
        if quota <= 0:
            return edges
        ys, xs = np.nonzero(edges)
        height, width = edges.shape
        cells_x = max(1, (width + cell - 1) // cell)
        keys = (ys // cell) * cells_x + (xs // cell)
        values = mag[ys, xs]
        # Single composite sort key: cell first, then descending magnitude.
        # This is ~4x faster than a two-key lexsort over the same edges.
        weight = float(values.max()) + 1.0 if values.size else 1.0
        order = np.argsort(
            keys.astype(np.float64) * weight - values.astype(np.float64),
            kind="quicksort",
        )
        sorted_keys = keys[order]
        boundaries = np.r_[0, np.flatnonzero(np.diff(sorted_keys)) + 1]
        counts = np.diff(np.r_[boundaries, len(sorted_keys)])
        ranks = np.arange(len(sorted_keys)) - np.repeat(boundaries, counts)
        selected = order[ranks < quota]
        capped = np.zeros_like(edges)
        capped[ys[selected], xs[selected]] = 255
        return capped

    @staticmethod
    def _scene_edges(gray, cache):
        if cache[0] is None:
            cache[0] = cv2.Canny(gray, _EDGE_LOW, _EDGE_HIGH)
        return cache[0]

    # ------------------------------------------------------------------
    # NCC / ECC refinement
    # ------------------------------------------------------------------
    def _ncc_match(self, scene_gray, rendered, center_x, center_y, margin=10, search=8):
        height, width = rendered.shape[:2]
        left = int(round(center_x)) - width // 2 - margin
        top = int(round(center_y)) - height // 2 - margin
        right = left + width + 2 * margin
        bottom = top + height + 2 * margin
        left = max(0, left)
        top = max(0, top)
        right = min(scene_gray.shape[1], right)
        bottom = min(scene_gray.shape[0], bottom)
        roi = scene_gray[top:bottom, left:right]
        if roi.shape[0] < height or roi.shape[1] < width:
            return None

        response = cv2.matchTemplate(roi, rendered, cv2.TM_CCOEFF_NORMED)
        expected_x = center_x - (left + width / 2.0)
        expected_y = center_y - (top + height / 2.0)
        ix = int(round(expected_x))
        iy = int(round(expected_y))
        x0 = max(0, ix - search)
        x1 = min(response.shape[1], ix + search + 1)
        y0 = max(0, iy - search)
        y1 = min(response.shape[0], iy + search + 1)
        if x1 <= x0 or y1 <= y0:
            return None
        window = response[y0:y1, x0:x1]
        _, score, _, location = cv2.minMaxLoc(window)
        px = x0 + location[0]
        py = y0 + location[1]
        dx, dy = self._subpixel_peak(response, px, py)
        return (
            left + px + dx + width / 2.0,
            top + py + dy + height / 2.0,
            float(score),
        )

    def _ncc_angle_scan(self, scene_gray, x, y, angle, scale, angle_min, angle_max,
                        template_gray=None):
        template = self._template_gray if template_gray is None else template_gray
        if template is None:
            return None
        low = max(angle_min, angle - 8.0)
        high = min(angle_max, angle + 8.0)
        if high < low:
            return None
        grid = np.arange(low, high + 1e-6, 4.0, dtype=np.float32)
        if grid.size == 0:
            return None
        if not np.any(np.abs(grid - angle) < 1e-4):
            grid = np.concatenate([grid, np.array([angle], dtype=np.float32)])
        evaluated = []
        for candidate in grid:
            rendered = self._render_search(float(candidate), scale, template_gray)
            if rendered is None:
                continue
            match = self._ncc_match(scene_gray, rendered, x, y)
            if match is None:
                continue
            evaluated.append(
                (float(candidate), float(match[2]), float(match[0]), float(match[1]))
            )
        if not evaluated:
            return None
        best_index = max(range(len(evaluated)), key=lambda index: evaluated[index][1])
        best_angle, best_score, best_x, best_y = evaluated[best_index]
        if 0 < best_index < len(evaluated) - 1:
            left = evaluated[best_index - 1]
            center = evaluated[best_index]
            right = evaluated[best_index + 1]
            denominator = left[1] - 2.0 * center[1] + right[1]
            if abs(denominator) > 1e-12:
                step = right[0] - center[0]
                best_angle = center[0] + 0.5 * (left[1] - right[1]) / denominator * step
                best_angle = max(left[0], min(right[0], best_angle))
        return best_score, best_angle, best_x, best_y

    def _ncc_scale_scan(self, scene_gray, x, y, angle, scale, scale_min, scale_max,
                        template_gray=None):
        template = self._template_gray if template_gray is None else template_gray
        if template is None:
            return None
        offsets = (-0.10, -0.05, 0.0, 0.05, 0.10)
        values = []
        for offset in offsets:
            value = min(scale_max, max(scale_min, scale + offset))
            if not any(abs(value - existing) < 1e-6 for existing in values):
                values.append(value)
        evaluated = []
        for candidate in values:
            rendered = self._render_search(angle, candidate, template_gray)
            if rendered is None:
                continue
            match = self._ncc_match(scene_gray, rendered, x, y)
            if match is None:
                continue
            evaluated.append(
                (float(candidate), float(match[2]), float(match[0]), float(match[1]))
            )
        if not evaluated:
            return None
        best_index = max(range(len(evaluated)), key=lambda index: evaluated[index][1])
        best_scale, best_score, best_x, best_y = evaluated[best_index]
        if 0 < best_index < len(evaluated) - 1:
            left = evaluated[best_index - 1]
            center = evaluated[best_index]
            right = evaluated[best_index + 1]
            denominator = left[1] - 2.0 * center[1] + right[1]
            if abs(denominator) > 1e-12:
                step = right[0] - center[0]
                best_scale = center[0] + 0.5 * (left[1] - right[1]) / denominator * step
                best_scale = max(left[0], min(right[0], best_scale))
        return best_score, best_scale, best_x, best_y

    def _ncc_translate(self, scene_gray, x, y, angle, scale, margin=8, search=6):
        if self._template_gray is None:
            return x, y
        rendered = self._render_gray(angle, scale)
        if rendered is None:
            return x, y
        match = self._ncc_match(
            scene_gray, rendered, x, y, margin=margin, search=search
        )
        if match is None:
            return x, y
        return match[0], match[1]

    @staticmethod
    def _decompose_similarity(matrix, center_x, center_y):
        linear = matrix[:2, :2]
        translation = matrix[:2, 2]
        u, singular, vt = np.linalg.svd(linear)
        rotation = u @ vt
        if np.linalg.det(rotation) < 0:
            rotation = u @ np.diag([1.0, -1.0]) @ vt
        scale = float(np.mean(singular))
        angle = -math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        center = matrix @ np.array([center_x, center_y, 1.0])
        return float(center[0]), float(center[1]), angle, scale

    def _ecc_refine(self, scene_gray, x, y, angle, scale, template_gray=None,
                    motion=cv2.MOTION_AFFINE,
                    template_scale=(1.0, 1.0), scene_scale=(1.0, 1.0),
                    scene_offset=(0.0, 0.0)):
        template = self._template_gray if template_gray is None else template_gray
        if template is None:
            return None
        height, width = template.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        matrix[0, 2] += x - width / 2.0
        matrix[1, 2] += y - height / 2.0
        inverse = cv2.invertAffineTransform(matrix)
        patch = cv2.warpAffine(
            scene_gray,
            inverse,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderValue=0,
        )
        warp = np.eye(2, 3, dtype=np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            20,
            1e-5,
        )
        try:
            coefficient, warp = cv2.findTransformECC(
                template,
                patch,
                warp,
                motion,
                criteria,
                None,
                5,
            )
        except cv2.error:
            return None
        composed = np.vstack([matrix, [0, 0, 1]]) @ np.vstack([warp, [0, 0, 1]])
        # Undo the anisotropic resampling introduced by working on reduced
        # views.  ``template_scale`` maps full-resolution template pixels to
        # the reduced template, ``scene_scale`` does the same for the scene,
        # so conjugating by both keeps the pose in full-resolution units.
        template_x, template_y = template_scale
        scene_x, scene_y = scene_scale
        offset_x, offset_y = scene_offset
        if (
            abs(template_x - 1.0) > 1e-9
            or abs(template_y - 1.0) > 1e-9
            or abs(scene_x - 1.0) > 1e-9
            or abs(scene_y - 1.0) > 1e-9
            or abs(offset_x) > 1e-9
            or abs(offset_y) > 1e-9
        ):
            template_matrix = np.diag([template_x, template_y, 1.0])
            # ``view = full * scale + offset``, so full = (view - offset)/scale.
            scene_matrix = np.array(
                [
                    [1.0 / scene_x, 0.0, -offset_x / scene_x],
                    [0.0, 1.0 / scene_y, -offset_y / scene_y],
                    [0.0, 0.0, 1.0],
                ]
            )
            composed = scene_matrix @ composed @ template_matrix
            center_x = (width / 2.0) / template_x
            center_y = (height / 2.0) / template_y
            refined_x, refined_y, refined_angle, refined_scale = (
                self._decompose_similarity(composed, center_x, center_y)
            )
            return (
                refined_x,
                refined_y,
                refined_angle,
                refined_scale,
                float(coefficient),
            )
        refined_x, refined_y, refined_angle, refined_scale = self._decompose_similarity(
            composed, width / 2.0, height / 2.0
        )
        return (
            refined_x,
            refined_y,
            refined_angle,
            refined_scale,
            float(coefficient),
        )

    # ------------------------------------------------------------------
    # Occlusion-tolerant score
    # ------------------------------------------------------------------
    @staticmethod
    def _bilinear(array, xs, ys):
        height, width = array.shape
        x0 = np.floor(xs).astype(np.int32)
        y0 = np.floor(ys).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = xs - x0
        wy = ys - y0
        valid = (x0 >= 0) & (y0 >= 0) & (x1 < width) & (y1 < height)
        x0 = np.clip(x0, 0, width - 1)
        x1 = np.clip(x1, 0, width - 1)
        y0 = np.clip(y0, 0, height - 1)
        y1 = np.clip(y1, 0, height - 1)
        values = (
            array[y0, x0] * (1.0 - wx) * (1.0 - wy)
            + array[y0, x1] * wx * (1.0 - wy)
            + array[y1, x0] * (1.0 - wx) * wy
            + array[y1, x1] * wx * wy
        )
        return values, valid

    def _dt_score(self, gx, gy, dt, cx, cy, angles, scales, use_distance=True,
                  model_mask=None, model_scale=1.0) -> np.ndarray:
        angles = np.asarray(angles, dtype=np.float32).reshape(-1)
        scales = np.asarray(scales, dtype=np.float32).reshape(-1)
        poses = angles.size
        if poses == 0 or scales.size != poses:
            return np.empty(0, dtype=np.float64)

        cx = np.asarray(cx, dtype=np.float32).reshape(-1)
        cy = np.asarray(cy, dtype=np.float32).reshape(-1)
        if cx.size == 1:
            cx = np.full(poses, cx[0], dtype=np.float32)
        if cy.size == 1:
            cy = np.full(poses, cy[0], dtype=np.float32)
        if cx.size != poses or cy.size != poses:
            return np.empty(0, dtype=np.float64)

        if model_mask is None:
            model_x = self._model_x
            model_y = self._model_y
            model_theta = self._model_theta
            model_mag = self._model_mag
        else:
            model_x = self._model_x[model_mask]
            model_y = self._model_y[model_mask]
            model_theta = self._model_theta[model_mask]
            model_mag = self._model_mag[model_mask]
            total = float(model_mag.sum())
            if model_x.size == 0 or total <= 0:
                return np.zeros(poses, dtype=np.float64)
            model_mag = model_mag / total
        if model_scale != 1.0:
            model_x = model_x / model_scale
            model_y = model_y / model_scale

        if _kernel_score is not None:
            dt_arg = dt
            if not use_distance or dt_arg is None:
                dt_arg = np.zeros((1, 1), dtype=np.float32)
            return np.asarray(
                _kernel_score(
                    np.ascontiguousarray(gx, dtype=np.float32),
                    np.ascontiguousarray(gy, dtype=np.float32),
                    np.ascontiguousarray(dt_arg, dtype=np.float32),
                    np.ascontiguousarray(model_x, dtype=np.float32),
                    np.ascontiguousarray(model_y, dtype=np.float32),
                    np.ascontiguousarray(model_theta, dtype=np.float32),
                    np.ascontiguousarray(model_mag, dtype=np.float32),
                    np.ascontiguousarray(cx, dtype=np.float32),
                    np.ascontiguousarray(cy, dtype=np.float32),
                    np.ascontiguousarray(angles, dtype=np.float32),
                    np.ascontiguousarray(scales, dtype=np.float32),
                    bool(use_distance),
                    float(_DISTANCE_SIGMA),
                    float(_ORIENTATION_EXPONENT),
                    5.0,
                ),
                dtype=np.float64,
            )

        radians = np.radians(angles)[:, None]
        cos_a = np.cos(radians)
        sin_a = np.sin(radians)
        scale_col = scales[:, None]
        model_x = model_x[None, :]
        model_y = model_y[None, :]
        xs = cx[:, None] + scale_col * (cos_a * model_x + sin_a * model_y)
        ys = cy[:, None] + scale_col * (-sin_a * model_x + cos_a * model_y)

        gx_values, valid = self._bilinear(gx, xs, ys)
        gy_values, _ = self._bilinear(gy, xs, ys)
        magnitude = np.sqrt(gx_values * gx_values + gy_values * gy_values)
        expected = model_theta[None, :] - radians
        dot = (
            np.cos(expected) * gx_values + np.sin(expected) * gy_values
        ) / np.maximum(magnitude, 1e-6)
        orientation = np.maximum(dot, 0.0) ** float(_ORIENTATION_EXPONENT)
        if use_distance:
            distance, _ = self._bilinear(dt, xs, ys)
            near = np.exp(
                -(distance * distance) / (2.0 * _DISTANCE_SIGMA * _DISTANCE_SIGMA)
            )
        else:
            near = 1.0
        weight = (
            model_mag[None, :]
            * orientation
            * near
            * (magnitude > 5.0)
            * valid
        )
        return np.clip(weight.sum(axis=1), 0.0, 1.0)

    def _dt_scan_axis(self, gx, gy, dt, x, y, angle, scale, axis, low, high, step,
                      model_mask=None):
        values = np.arange(low, high + step * 0.5, step, dtype=np.float32)
        if values.size == 0:
            return None
        if axis == "angle":
            scores = self._dt_score(
                gx, gy, dt, np.full(values.size, x), np.full(values.size, y),
                values, np.full(values.size, scale), model_mask=model_mask,
            )
        else:
            scores = self._dt_score(
                gx, gy, dt, np.full(values.size, x), np.full(values.size, y),
                np.full(values.size, angle), values, model_mask=model_mask,
            )
        if scores.size == 0:
            return None
        index = int(np.argmax(scores))
        best_value = float(values[index])
        best_score = float(scores[index])
        if 0 < index < values.size - 1:
            left = float(scores[index - 1])
            center = float(scores[index])
            right = float(scores[index + 1])
            denominator = left - 2.0 * center + right
            if abs(denominator) > 1e-12:
                best_value += 0.5 * (left - right) / denominator * float(values[1] - values[0])
        return best_value, best_score

    def _dt_scan_xy(self, gx, gy, dt, x, y, angle, scale, span, step,
                    model_mask=None):
        offsets = np.arange(-span, span + step * 0.5, step, dtype=np.float32)
        dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
        dx = dx.ravel()
        dy = dy.ravel()
        scores = self._dt_score(
            gx, gy, dt, x + dx, y + dy,
            np.full(dx.size, angle), np.full(dx.size, scale), model_mask=model_mask,
        )
        if scores.size == 0:
            return None
        index = int(np.argmax(scores))
        return float(x + dx[index]), float(y + dy[index]), float(scores[index])

    # ------------------------------------------------------------------
    # Robust ICP for occlusion
    # ------------------------------------------------------------------
    @staticmethod
    def _umeyama(points, targets, weights):
        total = float(weights.sum())
        if total <= 0:
            return None
        mu_p = (points * weights[:, None]).sum(axis=0) / total
        mu_q = (targets * weights[:, None]).sum(axis=0) / total
        p0 = points - mu_p
        q0 = targets - mu_q
        covariance = (q0 * weights[:, None]).T @ p0 / total
        u, singular, vt = np.linalg.svd(covariance)
        rotation = u @ np.diag([1.0, np.linalg.det(u @ vt)]) @ vt
        variance = float((p0 * p0 * weights[:, None]).sum() / total)
        if variance <= 1e-12:
            return None
        scale = float(singular.sum() / variance)
        translation = mu_q - scale * (rotation @ mu_p)
        angle = -math.degrees(math.atan2(rotation[1, 0], rotation[0, 0]))
        return float(translation[0]), float(translation[1]), angle, scale

    def _icp_refine(self, gx, gy, dt, x, y, angle, scale, iterations=2,
                    max_points=400):
        ddx = cv2.Sobel(dt, cv2.CV_32F, 1, 0, ksize=3)
        ddy = cv2.Sobel(dt, cv2.CV_32F, 0, 1, ksize=3)
        if _kernel_icp is not None:
            try:
                result = _kernel_icp(
                    np.ascontiguousarray(gx, dtype=np.float32),
                    np.ascontiguousarray(gy, dtype=np.float32),
                    np.ascontiguousarray(dt, dtype=np.float32),
                    np.ascontiguousarray(ddx, dtype=np.float32),
                    np.ascontiguousarray(ddy, dtype=np.float32),
                    np.ascontiguousarray(self._model_x, dtype=np.float32),
                    np.ascontiguousarray(self._model_y, dtype=np.float32),
                    np.ascontiguousarray(self._model_theta, dtype=np.float32),
                    np.ascontiguousarray(self._model_mag, dtype=np.float32),
                    float(x),
                    float(y),
                    float(angle),
                    float(scale),
                    int(iterations),
                    int(max_points or 0),
                )
                return tuple(float(value) for value in result)
            except Exception:
                pass

        points = np.stack(
            [self._model_x.astype(np.float64), self._model_y.astype(np.float64)], axis=1
        )
        model_theta = self._model_theta.astype(np.float64)
        model_mag = self._model_mag.astype(np.float64)
        if max_points and points.shape[0] > max_points:
            stride = int(math.ceil(points.shape[0] / float(max_points)))
            points = points[::stride]
            model_theta = model_theta[::stride]
            model_mag = model_mag[::stride]
            total = float(model_mag.sum())
            if total > 0:
                model_mag = model_mag / total
        for _ in range(iterations):
            radians = math.radians(angle)
            cos_a = math.cos(radians)
            sin_a = math.sin(radians)
            model_x = points[:, 0]
            model_y = points[:, 1]
            xs = x + scale * (cos_a * model_x + sin_a * model_y)
            ys = y + scale * (-sin_a * model_x + cos_a * model_y)
            distance, valid = self._bilinear(dt, xs, ys)
            gx_values, _ = self._bilinear(gx, xs, ys)
            gy_values, _ = self._bilinear(gy, xs, ys)
            ddx_values, _ = self._bilinear(ddx, xs, ys)
            ddy_values, _ = self._bilinear(ddy, xs, ys)
            gradient = np.sqrt(ddx_values * ddx_values + ddy_values * ddy_values)
            targets = np.stack(
                [
                    xs - distance * ddx_values / np.maximum(gradient, 1e-6),
                    ys - distance * ddy_values / np.maximum(gradient, 1e-6),
                ],
                axis=1,
            )
            magnitude = np.sqrt(gx_values * gx_values + gy_values * gy_values)
            dot = (
                np.cos(model_theta - radians) * gx_values
                + np.sin(model_theta - radians) * gy_values
            ) / np.maximum(magnitude, 1e-6)
            orientation = np.maximum(dot, 0.0) ** float(_ORIENTATION_EXPONENT)
            near = np.exp(-(distance * distance) / 2.0)
            weights = (
                model_mag
                * orientation
                * near
                * (distance < 2.5)
                * (magnitude > 5.0)
                * valid
            )
            if float(weights.sum()) < 0.1:
                break
            estimate = self._umeyama(points, targets, weights)
            if estimate is None:
                break
            next_x, next_y, next_angle, next_scale = estimate
            if (
                abs(next_x - x) > 2.0
                or abs(next_y - y) > 2.0
                or abs(next_angle - angle) > 1.0
                or abs(next_scale / scale - 1.0) > 0.05
            ):
                break
            x, y, angle, scale = next_x, next_y, next_angle, next_scale
        return x, y, angle, scale

    # ------------------------------------------------------------------
    # Pose refinement
    # ------------------------------------------------------------------
    def _visible_mask(self, scene_gray, x, y, angle, scale):
        """Model points that are not covered by an occluder at this pose."""
        if self._template_gray is None or scene_gray is None:
            return None
        height, width = self._template_gray.shape[:2]
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, scale)
        matrix[0, 2] += x - width / 2.0
        matrix[1, 2] += y - height / 2.0
        inverse = cv2.invertAffineTransform(matrix)
        patch = cv2.warpAffine(
            scene_gray, inverse, (width, height), flags=cv2.INTER_LINEAR, borderValue=0
        )
        if patch.dtype == np.uint8 and self._template_gray.dtype == np.uint8:
            # The difference is integer-valued 0..255.  A 256-bin histogram
            # gives the exact median and MAD without sorting the template.
            difference = cv2.absdiff(patch, self._template_gray)
            total = int(difference.size)
            hist = np.bincount(difference.ravel(), minlength=256)
            cumulative = np.cumsum(hist)
            lo = int(np.searchsorted(cumulative, (total - 1) // 2, side="right"))
            hi = int(np.searchsorted(cumulative, total // 2, side="right"))
            median2 = lo + hi
            median = 0.5 * median2
            # Keep the deviations in doubled units so half-integer medians
            # remain exact; the final MAD is divided by two.
            dev2 = np.abs(difference.astype(np.int16) * 2 - median2)
            hist2 = np.bincount(dev2.ravel(), minlength=511)
            cumulative2 = np.cumsum(hist2)
            lo2 = int(np.searchsorted(cumulative2, (total - 1) // 2, side="right"))
            hi2 = int(np.searchsorted(cumulative2, total // 2, side="right"))
            mad = 0.25 * (lo2 + hi2)
            threshold = min(45.0, max(30.0, median + 3.0 * 1.4826 * mad))
            visible = difference < threshold
        else:
            difference = np.abs(
                patch.astype(np.float32) - self._template_gray.astype(np.float32)
            )
            median = float(np.median(difference))
            mad = float(np.median(np.abs(difference - median)))
            threshold = min(45.0, max(30.0, median + 3.0 * 1.4826 * mad))
            visible = difference < threshold
        px = np.rint(self._model_x + width / 2.0).astype(np.int32)
        py = np.rint(self._model_y + height / 2.0).astype(np.int32)
        valid = (px >= 0) & (px < width) & (py >= 0) & (py < height)
        keep = np.zeros(self._model_x.size, dtype=bool)
        keep[valid] = visible[py[valid], px[valid]]
        if int(keep.sum()) < max(40, int(0.2 * keep.size)):
            return None
        return keep

    def _alignment_score(self, gx, gy, x, y, angle, scale, coordinate_scale=1.0):
        return float(
            self._dt_score(
                gx,
                gy,
                None,
                np.array([x / coordinate_scale]),
                np.array([y / coordinate_scale]),
                np.array([angle]),
                np.array([scale]),
                use_distance=False,
                model_scale=coordinate_scale,
            )[0]
        )

    def _local_alignment(self, gray, x, y, angle, scale):
        if self._template_size[0] <= 0:
            return 0.0
        width, height = self._template_size
        radius = int(math.ceil(0.5 * math.hypot(width, height) * scale)) + 32
        left = max(0, int(round(x)) - radius)
        top = max(0, int(round(y)) - radius)
        right = min(gray.shape[1], int(round(x)) + radius + 1)
        bottom = min(gray.shape[0], int(round(y)) + radius + 1)
        patch = gray[top:bottom, left:right]
        if patch.shape[0] < 16 or patch.shape[1] < 16:
            return 0.0
        patch_gx = cv2.Sobel(patch, cv2.CV_32F, 1, 0, ksize=3)
        patch_gy = cv2.Sobel(patch, cv2.CV_32F, 0, 1, ksize=3)
        return self._alignment_score(
            patch_gx,
            patch_gy,
            x - left,
            y - top,
            angle,
            scale,
        )

    def _candidate_ncc(self, candidate, search_gray, search_map,
                       theta_start, theta_end, scale_start, scale_end):
        """Single-pose NCC for a vote proposal (cheap pre-gate)."""
        if search_gray is None or search_map is None:
            return None
        angle = float(np.clip(candidate["angle"], theta_start, theta_end))
        scale = float(np.clip(candidate["scale"], scale_start, scale_end))
        rendered = self._render_search(angle, scale, self._search_template_gray)
        if rendered is None:
            return None
        scale_x, scale_y, origin_x, origin_y = search_map
        match = self._ncc_match(
            search_gray,
            rendered,
            candidate["x"] * scale_x + origin_x,
            candidate["y"] * scale_y + origin_y,
        )
        if match is None:
            return None
        return float(match[2])

    def _coarse_pose(self, gray, gx, gy, x, y, angle, scale,
                     angle_min, angle_max, scale_min, scale_max,
                     alignment_gx=None, alignment_gy=None, alignment_scale=1.0,
                     search_gray=None, search_map=None):
        if search_gray is not None and search_map is not None:
            scale_x, scale_y, origin_x, origin_y = search_map
            scene = search_gray
            template = self._search_template_gray
            sx = x * scale_x + origin_x
            sy = y * scale_y + origin_y
        else:
            scale_x = scale_y = 1.0
            origin_x = origin_y = 0.0
            scene = gray
            template = self._template_gray
            sx = x
            sy = y

        angle_result = self._ncc_angle_scan(
            scene, sx, sy, angle, scale, angle_min, angle_max, template
        )
        if angle_result is None:
            return None
        ncc_score, angle, sx, sy = angle_result
        scale_result = self._ncc_scale_scan(
            scene, sx, sy, angle, scale, scale_min, scale_max, template
        )
        if scale_result is not None:
            ncc_score, scale, sx, sy = scale_result
        x = (sx - origin_x) / scale_x
        y = (sy - origin_y) / scale_y

        # Cheap rejection: a real match keeps a healthy NCC even on the reduced
        # search view, so do not pay for full-resolution alignment or a solver
        # when the coarse score is already poor.
        if ncc_score < _MIN_NCC:
            return x, y, angle, scale, ncc_score, None

        if alignment_gx is not None:
            alignment = self._alignment_score(
                alignment_gx,
                alignment_gy,
                x,
                y,
                angle,
                scale,
                coordinate_scale=alignment_scale,
            )
        else:
            alignment = self._local_alignment(gray, x, y, angle, scale)
        return x, y, angle, scale, ncc_score, alignment

    @staticmethod
    def _reduced_gray(gray, factor, cached=None, cached_factor=1):
        if cached is not None and int(cached_factor) == int(factor):
            return cached
        width = max(8, int(round(gray.shape[1] / factor)))
        height = max(8, int(round(gray.shape[0] / factor)))
        return cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)

    @staticmethod
    def _pyramid_views(frame_gray, ax, ay, aw, ah, factors):
        """Build ROI views on a grid anchored to the frame origin.

        The crop starts at a multiple of the factor, so the downscaled pixels
        are the same blocks a full-frame resize would use.  A larger ROI then
        only adds blocks; it cannot change the sampling phase of the part, and
        the cost stays proportional to the ROI.
        """
        height, width = frame_gray.shape[:2]
        views = {}
        for factor in sorted({max(1, int(value)) for value in factors}):
            if factor <= 1:
                view = frame_gray[ay:ay + ah, ax:ax + aw]
                views[factor] = (view, (1.0, 1.0, 0.0, 0.0))
                continue
            left = (int(ax) // factor) * factor
            top = (int(ay) // factor) * factor
            right = int(math.ceil((ax + aw) / float(factor))) * factor
            bottom = int(math.ceil((ay + ah) / float(factor))) * factor
            left = max(0, left)
            top = max(0, top)
            right = min(width, right)
            bottom = min(height, bottom)
            crop = frame_gray[top:bottom, left:right]
            small = cv2.resize(
                crop,
                (
                    max(1, crop.shape[1] // factor),
                    max(1, crop.shape[0] // factor),
                ),
                interpolation=cv2.INTER_AREA,
            )
            scale_x = small.shape[1] / float(crop.shape[1])
            scale_y = small.shape[0] / float(crop.shape[0])
            views[factor] = (
                small,
                (
                    scale_x,
                    scale_y,
                    (ax - left) * scale_x,
                    (ay - top) * scale_y,
                ),
            )
        return views

    @staticmethod
    def _resized_view(gray, factor):
        """Resize a crop and return the view plus its ROI-to-view map."""
        if factor <= 1:
            return gray, (1.0, 1.0, 0.0, 0.0)
        width = max(8, int(round(gray.shape[1] / factor)))
        height = max(8, int(round(gray.shape[0] / factor)))
        view = cv2.resize(gray, (width, height), interpolation=cv2.INTER_AREA)
        return view, (
            view.shape[1] / float(gray.shape[1]),
            view.shape[0] / float(gray.shape[0]),
            0.0,
            0.0,
        )

    def _ecc_solver(self, gray, x, y, angle, scale,
                    angle_min, angle_max, scale_min, scale_max,
                    ecc_gray=None, ecc_map=None):
        factor = max(1, int(self._ecc_factor))
        fixed_scale = abs(float(scale_max) - float(scale_min)) < 1e-9
        ecc_motion = cv2.MOTION_EUCLIDEAN if fixed_scale else cv2.MOTION_AFFINE
        pre_refined = False
        pre_coefficient = 0.0
        if factor > 1 and self._ecc_template_gray is not None and ecc_map is not None:
            scene = ecc_gray
            scale_x, scale_y, origin_x, origin_y = ecc_map
            start = self._ecc_refine(
                scene,
                x * scale_x + origin_x,
                y * scale_y + origin_y,
                angle,
                scale,
                template_gray=self._ecc_template_gray,
                motion=ecc_motion,
                template_scale=(
                    self._ecc_template_gray.shape[1] / float(self._template_size[0]),
                    self._ecc_template_gray.shape[0] / float(self._template_size[1]),
                ),
                scene_scale=(scale_x, scale_y),
                scene_offset=(origin_x, origin_y),
            )
            if start is not None:
                sx, sy, start_angle, start_scale, start_cc = start
                candidate = (sx, sy, start_angle, start_scale)
                if (
                    abs(start_angle - angle) <= 5.0
                    and abs(start_scale / scale - 1.0) <= 0.10
                    and math.hypot(
                        candidate[0] - x, candidate[1] - y
                    ) <= 8.0
                ):
                    x, y, angle, scale = candidate
                    pre_refined = True
                    pre_coefficient = float(start_cc)

        if pre_refined:
            # The coarse affine pass already refined angle/scale; at full
            # resolution only the translation needs a sub-pixel correction,
            # and a rendered NCC match is far cheaper than another ECC pass.
            x, y = self._ncc_translate(gray, x, y, angle, scale)
            if (
                not (angle_min - 0.05 <= angle <= angle_max + 0.05)
                or not (scale_min - 0.05 <= scale <= scale_max + 0.05)
            ):
                return None, 0.0
            return (x, y, angle, scale), pre_coefficient

        result = self._ecc_refine(
            gray,
            x,
            y,
            angle,
            scale,
            motion=ecc_motion,
        )
        if result is None:
            return None, 0.0
        ex, ey, ea, es, coefficient = result
        if (
            coefficient < _ECC_MIN_CC
            or not (angle_min - 0.05 <= ea <= angle_max + 0.05)
            or not (scale_min - 0.05 <= es <= scale_max + 0.05)
            or abs(ea - angle) > 5.0
            or abs(es / scale - 1.0) > 0.10
            or math.hypot(ex - x, ey - y) > 8.0
        ):
            return None, coefficient
        x, y, angle, scale = ex, ey, ea, es
        if pre_refined:
            return (x, y, angle, scale), coefficient
        # A translation-only polish is cheap, but when the affine pass is
        # already essentially perfect it cannot move the pose enough to matter.
        if coefficient >= _ECC_POLISH_MIN_CC:
            return (x, y, angle, scale), coefficient
        polish = self._ecc_refine(
            gray, x, y, angle, scale, motion=cv2.MOTION_TRANSLATION
        )
        if polish is not None:
            px, py, pa, ps, polish_cc = polish
            if (
                polish_cc >= _ECC_MIN_CC
                and abs(pa - angle) <= 1.0
                and abs(ps / scale - 1.0) <= 0.02
                and math.hypot(px - x, py - y) <= 2.0
            ):
                x, y, angle, scale = px, py, pa, ps
                coefficient = polish_cc
        return (x, y, angle, scale), coefficient

    @staticmethod
    def _distance_map(cache, scene_edges):
        if cache[0] is None:
            cache[0] = cv2.distanceTransform(255 - scene_edges, cv2.DIST_L2, 3)
        return cache[0]

    def _occlusion_refine(self, gx, gy, dt, gray, x, y, angle, scale,
                          angle_min, angle_max, scale_min, scale_max):
        result = self._dt_scan_axis(
            gx, gy, dt, x, y, angle, scale, "angle",
            max(angle_min, angle - 1.5), min(angle_max, angle + 1.5), 0.75,
        )
        if result:
            angle = result[0]
        x, y = self._ncc_translate(gray, x, y, angle, scale)

        result = self._dt_scan_axis(
            gx, gy, dt, x, y, angle, scale, "scale",
            max(scale_min, scale - 0.020), min(scale_max, scale + 0.020), 0.002,
        )
        if result:
            scale = result[0]

        visible_fraction = None
        alignment = self._alignment_score(gx, gy, x, y, angle, scale)
        # The masked angle scan below is the stronger refinement for heavily
        # occluded parts; running ICP first only duplicates that work.  Keep
        # ICP for the milder cases where the masked scan is not triggered.
        if alignment > 0.70:
            _, _, icp_angle, _ = self._icp_refine(gx, gy, dt, x, y, angle, scale)
            if abs(icp_angle - angle) <= 0.2:
                angle = icp_angle
        if alignment <= 0.70:
            mask = self._visible_mask(gray, x, y, angle, scale)
            if mask is None:
                return None
            masked = self._dt_scan_axis(
                gx, gy, dt, x, y, angle, scale, "angle",
                max(angle_min, angle - 0.45), min(angle_max, angle + 0.45), 0.15,
                model_mask=mask,
            )
            if masked is not None:
                angle = masked[0]

        result = self._dt_scan_axis(
            gx, gy, dt, x, y, angle, scale, "scale",
            max(scale_min, scale - 0.006), min(scale_max, scale + 0.006), 0.003,
        )
        if result:
            scale = result[0]

        x, y = self._ncc_translate(gray, x, y, angle, scale)
        if alignment <= 0.70:
            final_mask = self._visible_mask(gray, x, y, angle, scale)
            if final_mask is None:
                return None
            visible_fraction = float(final_mask.mean())
        return x, y, angle, scale, visible_fraction

    @staticmethod
    def _full_gradients(gray, cache):
        if cache[0] is None:
            cache[0] = (
                cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3),
                cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3),
            )
        return cache[0]

    def _refine_candidate(self, gray, gx, gy, edge_cache, dt_cache, gradient_cache,
                          candidate, theta_start, theta_end, scale_start, scale_end,
                          alignment_gx=None, alignment_gy=None, alignment_scale=1.0,
                          search_gray=None, search_map=None,
                          ecc_gray=None, ecc_map=None, solve=True):
        # ``theta_*`` and ``angle`` below are internal OpenCV angles
        # (positive = counter-clockwise); the caller converts the final pose.
        x = float(candidate["x"])
        y = float(candidate["y"])
        angle = float(np.clip(candidate["angle"], theta_start, theta_end))
        scale = float(np.clip(candidate["scale"], scale_start, scale_end))

        coarse = self._coarse_pose(
            gray, gx, gy, x, y, angle, scale,
            theta_start, theta_end, scale_start, scale_end,
            alignment_gx=alignment_gx,
            alignment_gy=alignment_gy,
            alignment_scale=alignment_scale,
            search_gray=search_gray,
            search_map=search_map,
        )
        if coarse is None:
            return None, {"warning": "ncc_refinement_failed"}
        x, y, angle, scale, ncc_score, alignment = coarse
        if alignment is None:
            return None, {"warning": "low_ncc", "ncc_score": ncc_score}
        if ncc_score < _MIN_NCC and alignment < 0.5:
            return None, {
                "warning": "low_ncc",
                "ncc_score": ncc_score,
                "alignment": alignment,
            }

        if not solve:
            return (x, y, angle, scale, ncc_score, alignment), None
        return self._solve_pose(
            gray, gx, gy, edge_cache, dt_cache, gradient_cache,
            x, y, angle, scale, ncc_score, alignment,
            theta_start, theta_end, scale_start, scale_end,
            ecc_gray=ecc_gray, ecc_map=ecc_map,
        )

    def _solve_pose(self, gray, gx, gy, edge_cache, dt_cache, gradient_cache,
                    x, y, angle, scale, ncc_score, alignment,
                    theta_start, theta_end, scale_start, scale_end,
                    ecc_gray=None, ecc_map=None):
        solver = None
        ecc_cc = None
        visible_fraction = None
        warnings = []

        if alignment >= _ECC_ALIGNMENT:
            pose, ecc_cc = self._ecc_solver(
                gray, x, y, angle, scale,
                theta_start, theta_end, scale_start, scale_end,
                ecc_gray=ecc_gray,
                ecc_map=ecc_map,
            )
            if pose is not None:
                x, y, angle, scale = pose
                solver = "ecc"
            elif alignment < _ECC_REJECT_ALIGNMENT:
                if gx is None:
                    gx, gy = self._full_gradients(gray, gradient_cache)
                dt = self._distance_map(
                    dt_cache, self._scene_edges(gray, edge_cache)
                )
                occlusion = self._occlusion_refine(
                    gx, gy, dt, gray, x, y, angle, scale,
                    theta_start, theta_end, scale_start, scale_end,
                )
                if occlusion is None:
                    return None, {
                        "warning": "occlusion_solver_failed",
                        "ncc_score": ncc_score,
                        "ecc_cc": ecc_cc,
                    }
                x, y, angle, scale, visible_fraction = occlusion
                solver = "occlusion"
                warnings.append("ecc_failed_reclassified")
            else:
                return None, {
                    "warning": "ecc_failed_visible_part",
                    "ncc_score": ncc_score,
                    "ecc_cc": ecc_cc,
                }
        else:
            if gx is None:
                gx, gy = self._full_gradients(gray, gradient_cache)
            dt = self._distance_map(
                dt_cache, self._scene_edges(gray, edge_cache)
            )
            occlusion = self._occlusion_refine(
                gx, gy, dt, gray, x, y, angle, scale,
                theta_start, theta_end, scale_start, scale_end,
            )
            if occlusion is None:
                return None, {
                    "warning": "occlusion_solver_failed",
                    "ncc_score": ncc_score,
                }
            x, y, angle, scale, visible_fraction = occlusion
            solver = "occlusion"

        if gx is None:
            score = self._local_alignment(gray, x, y, angle, scale)
        else:
            score = self._alignment_score(gx, gy, x, y, angle, scale)
        return (
            x,
            y,
            angle,
            scale,
            score,
            solver,
            ncc_score,
            visible_fraction,
            ecc_cc,
            warnings,
        ), None

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, advance=False) -> None:
        self.outputs["output image"] = None
        self.outputs["count"] = 0
        self.outputs["results"] = ToolResults()
        self.outputs["warnings"] = []
        self.outputs["trained"] = self.is_trained

        source = self.inputs.get("input image")
        if source is None:
            self.data_changed.emit()
            return

        self._ensure_state()
        self.outputs["trained"] = self.is_trained

        v_img = source if isinstance(source, VisionImage) else VisionImage.from_array(source)
        vision_img = VisionImage(
            v_img.data,
            x=v_img.x,
            y=v_img.y,
            angle=v_img.angle,
            scale_x=v_img.scale_x,
            scale_y=v_img.scale_y,
            scale_z=v_img.scale_z,
            offset_z=v_img.offset_z,
        )
        vision_img.inherit_overlays_from(v_img)
        propagate_3d(v_img, vision_img, regenerate=False)
        layer = vision_img.get_overlay_layer("default", source_tool=self.name)
        # The layer may have been created by an upstream tool, in which case
        # get_overlay_layer() returns it unchanged and add_roi()/add_contour()
        # would tag the new elements with the upstream tool's name, so
        # _copy_overlays_to_input() would drop them. Stamp our own name here,
        # exactly like PatternMatcher does.
        layer.source_tool = self.name
        layer.elements = [e for e in layer.elements if e.source_tool != self.name]
        self.outputs["output image"] = vision_img

        roi = extract_roi(v_img, self.params["search_roi"])
        ax, ay = roi["ax"], roi["ay"]
        roi_angle = float(roi["abs_angle"])
        roi_m_inv = roi["M_inv"]

        # Draw the search ROI even when untrained (same as PatternMatcher) so
        # the ROI editor always has visual feedback.  Use the unclamped
        # absolute rect so a rotated ROI is shown with its real angle.
        if self.params["show_search_roi"]:
            layer.add_roi(
                [roi["abs_x"], roi["abs_y"], roi["abs_w"], roi["abs_h"], roi_angle],
                color=(255, 128, 0),
                thickness=2,
            )

        if not self.is_trained:
            self.data_changed.emit()
            return

        roi_src = roi["roi_src"]
        raw = roi_src.data if isinstance(roi_src, VisionImage) else roi_src
        if raw is None or raw.size == 0:
            self.data_changed.emit()
            return
        gray = self._gray(raw)
        if gray.dtype == np.uint8:
            gray_for_match = gray
        else:
            gray_for_match = cv2.normalize(
                gray, None, 0, 255, cv2.NORM_MINMAX
            ).astype(np.uint8)

        vote_factor = self._vote_factor()
        search_factor = int(self._search_factor)
        ecc_factor = int(self._ecc_factor)
        vote_map = search_map = ecc_map = None
        if roi_m_inv is None:
            # Every coarse view is a crop of one fixed-phase pyramid built from
            # the whole frame.  A larger search ROI can then only add scene
            # pixels: the part is voted and scanned on exactly the same pixels
            # regardless of where the ROI starts.
            frame = v_img.data
            frame_gray = self._gray(frame)
            if frame_gray.dtype != np.uint8:
                frame_gray = cv2.normalize(
                    frame_gray, None, 0, 255, cv2.NORM_MINMAX
                ).astype(np.uint8)
            gray_for_match = frame_gray[
                ay:ay + roi["ah"], ax:ax + roi["aw"]
            ]
            views = self._pyramid_views(
                frame_gray, ax, ay, roi["aw"], roi["ah"],
                (vote_factor, search_factor, ecc_factor),
            )
            vote_gray, vote_map = views[vote_factor]
            search_gray, search_map = views[search_factor]
            ecc_gray, ecc_map = views[ecc_factor]
        else:
            # Rotated ROIs are already resampled by ``extract_roi``; keep the
            # previous crop-resize path for them.
            vote_gray, vote_map = self._resized_view(gray_for_match, vote_factor)
            search_gray, search_map = self._resized_view(
                gray_for_match, search_factor
            )
            ecc_gray, ecc_map = self._resized_view(gray_for_match, ecc_factor)
        # ``spatialGradient`` is the same 3x3 Sobel kernel as two separate
        # ``Sobel`` calls but computes both axes in one pass.  Magnitude and
        # orientation are only needed on Canny edge pixels, so avoid the full
        # ``cv2.magnitude`` pass on large ROIs.
        vote_gx, vote_gy = cv2.spatialGradient(vote_gray)
        vote_edges = cv2.Canny(vote_gray, _EDGE_LOW, _EDGE_HIGH)
        edge_y, edge_x = np.nonzero(vote_edges)
        vote_mag = np.zeros(vote_gray.shape, dtype=np.float32)
        if edge_y.size:
            vote_mag[edge_y, edge_x] = np.hypot(
                vote_gx[edge_y, edge_x].astype(np.float32),
                vote_gy[edge_y, edge_x].astype(np.float32),
            )
        vote_edges = self._cap_scene_edges(vote_mag, vote_edges)
        edge_y, edge_x = np.nonzero(vote_edges)
        vote_theta = np.zeros(vote_gray.shape, dtype=np.float32)
        if edge_y.size:
            vote_theta[edge_y, edge_x] = np.arctan2(
                vote_gy[edge_y, edge_x], vote_gx[edge_y, edge_x]
            )

        # Public angle range: positive = clockwise in the image frame.  The
        # crop is straightened by ``roi_angle``, so the crop-relative range is
        # shifted by it.  Search stages work with OpenCV's counter-clockwise
        # ``theta``, hence ``theta = roi_angle - angle``.
        angle_start = float(self.params["angle_start"])
        angle_end = float(self.params["angle_end"])
        if angle_start > angle_end:
            angle_start, angle_end = angle_end, angle_start
        theta_start = roi_angle - angle_end
        theta_end = roi_angle - angle_start
        scale_start = float(self.params["scale_start"])
        scale_end = float(self.params["scale_end"])
        if scale_start > scale_end:
            scale_start, scale_end = scale_end, scale_start

        span = theta_end - theta_start
        coarse_angle = min(10.0, max(2.0, span / 4.0)) if span > 0 else 0.0
        if coarse_angle > 0:
            angles = list(
                np.arange(theta_start, theta_end + coarse_angle * 0.5, coarse_angle)
            )
        else:
            angles = [theta_start]

        scale_span = scale_end - scale_start
        if scale_span < 1e-9:
            scales = [scale_start]
        else:
            count = min(5, max(3, int(round(scale_span / 0.10)) + 1))
            scales = list(np.linspace(scale_start, scale_end, count))

        factor_x = 1.0 / vote_map[0]
        factor_y = 1.0 / vote_map[1]
        coarse = self._kernel_coarse(
            vote_mag,
            vote_theta,
            vote_edges,
            angles,
            scales,
            model_scale_x=factor_x,
            model_scale_y=factor_y,
        )
        vx_scale, vy_scale, vx_origin, vy_origin = vote_map
        for candidate in coarse:
            candidate["x"] = (candidate["x"] - vx_origin) / vx_scale
            candidate["y"] = (candidate["y"] - vy_origin) / vy_scale
        coarse.sort(key=lambda item: item["peak"], reverse=True)
        if coarse:
            peak_floor = max(_VOTE_MIN_PEAK, coarse[0]["peak"] * 0.10)
            coarse = [item for item in coarse if item["peak"] >= peak_floor]
        min_distance = max(10.0, min(self._template_size) * 0.4)
        max_matches = int(self.params["max_matches"])
        limit = _MAX_MATCHES if max_matches <= 0 else min(max_matches, _MAX_MATCHES)
        # Merge only near-identical hypotheses.  A bigger ROI adds clutter
        # peaks, and an aggressive distance here used to drop the true
        # candidate before it was ever refined.
        coarse = self._dedupe_candidates(
            coarse,
            max(8.0, min(self._template_size) * 0.15),
            angle_tolerance=max(1.0, coarse_angle * 0.5),
            scale_tolerance=0.05,
        )
        # Two-stage NCC: one cheap correlation ranks every proposal; the full
        # angle/scale scan runs only on the promising ones plus the strongest
        # vote peaks, so a proposal whose kernel angle/scale is still far off
        # is not dropped before it has been measured.
        if search_gray is not None and search_map is not None and len(coarse) > 4:
            pre_gate = []
            for candidate in coarse[:_MAX_PRE_GATE]:
                score = self._candidate_ncc(
                    candidate,
                    search_gray,
                    search_map,
                    theta_start,
                    theta_end,
                    scale_start,
                    scale_end,
                )
                pre_gate.append((score if score is not None else -1.0, candidate))
            pre_gate.sort(key=lambda item: item[0], reverse=True)
            ranked = [candidate for _, candidate in pre_gate]
            chosen = []
            seen = set()
            for candidate in ranked[:_MAX_NCC_SCANS] + coarse[:4]:
                if id(candidate) in seen:
                    continue
                seen.add(id(candidate))
                chosen.append(candidate)
            coarse = chosen[:_MAX_NCC_CANDIDATES]
        else:
            coarse = coarse[:_MAX_NCC_CANDIDATES]

        candidates = []
        refined_centers = []
        warnings = []
        dt_cache = [None]
        edge_cache = [None]
        gradient_cache = [None]
        alignment_scale = 1.0
        if vote_factor == 1:
            # The vote path only needs the int16 gradients; the alignment
            # score bilinearly samples them and wants float32.
            gx = vote_gx.astype(np.float32)
            gy = vote_gy.astype(np.float32)
            alignment_gx, alignment_gy = gx, gy
        else:
            # One full-resolution gradient pair for the whole ROI is far
            # cheaper than a Sobel patch per proposal, and it is cached for
            # the occlusion solver too.
            gx, gy = self._full_gradients(gray_for_match, gradient_cache)
            alignment_gx, alignment_gy = gx, gy
        prepared = []
        short_distance = max(16.0, min(self._template_size) * 0.20)
        for coarse_candidate in coarse:
            pose, diagnostics = self._refine_candidate(
                gray_for_match,
                gx,
                gy,
                edge_cache,
                dt_cache,
                gradient_cache,
                coarse_candidate,
                theta_start,
                theta_end,
                scale_start,
                scale_end,
                alignment_gx=alignment_gx,
                alignment_gy=alignment_gy,
                alignment_scale=alignment_scale,
                search_gray=search_gray,
                search_map=search_map,
                solve=False,
            )
            if pose is None:
                if diagnostics:
                    warnings.append(diagnostics)
                continue
            x, y, angle, scale, ncc_score, alignment = pose
            prepared.append((ncc_score, alignment, x, y, angle, scale))

        # Keep the best NCC per physical location before paying for
        # ECC/occlusion.  This preserves the extra hypotheses a bigger ROI
        # produces without running the expensive solver on near-duplicates.
        prepared.sort(key=lambda item: (item[0], item[1]), reverse=True)
        shortlist = []
        for item in prepared:
            if item[0] < 0.5:
                continue
            if any(
                math.hypot(item[2] - other[2], item[3] - other[3]) < short_distance
                for other in shortlist
            ):
                continue
            shortlist.append(item)
            if len(shortlist) >= min(_MAX_SOLVE, limit + 2):
                break
        if not shortlist and prepared:
            shortlist = prepared[:1]

        for ncc_score, _alignment_pre, x, y, angle, scale in shortlist:
            refined, diagnostics = self._solve_pose(
                gray_for_match,
                gx,
                gy,
                edge_cache,
                dt_cache,
                gradient_cache,
                x,
                y,
                angle,
                scale,
                ncc_score,
                _alignment_pre,
                theta_start,
                theta_end,
                scale_start,
                scale_end,
                ecc_gray=ecc_gray,
                ecc_map=ecc_map,
            )
            if refined is None:
                if diagnostics:
                    warnings.append(diagnostics)
                continue
            (
                refined_x,
                refined_y,
                refined_angle,
                refined_scale,
                score,
                solver,
                ncc_score,
                visible_fraction,
                ecc_cc,
                refine_warnings,
            ) = refined
            warnings.extend(refine_warnings)
            if score < float(self.params["min_score"]):
                continue
            refined_centers.append((refined_x, refined_y))
            if roi_m_inv is not None:
                mapped = roi_m_inv @ np.array([refined_x, refined_y, 1.0])
                center_x, center_y = float(mapped[0]), float(mapped[1])
            else:
                center_x = refined_x + ax
                center_y = refined_y + ay
            # Refinement works in OpenCV theta; convert to the framework
            # angle in the image frame (undo the crop rotation).
            angle = -refined_angle + roi_angle
            origin_x, origin_y = self._origin_point(
                center_x, center_y, angle, refined_scale
            )
            candidates.append(
                {
                    "x": origin_x,
                    "y": origin_y,
                    "center_x": center_x,
                    "center_y": center_y,
                    "angle": angle,
                    "scale": refined_scale,
                    "score": score,
                    "solver": solver,
                    "ncc_score": ncc_score,
                    "visible_fraction": visible_fraction,
                    "ecc_cc": ecc_cc,
                    "warnings": list(refine_warnings),
                }
            )

        candidates.sort(key=lambda item: item["score"], reverse=True)
        kept = []
        for candidate in candidates:
            if all(
                math.hypot(candidate["x"] - other["x"], candidate["y"] - other["y"])
                >= min_distance
                for other in kept
            ):
                kept.append(candidate)
                if len(kept) >= limit:
                    break

        results = ToolResults()
        for index, candidate in enumerate(kept):
            results.append(
                ShapeMatchResult(
                    id=index,
                    x=candidate["x"],
                    y=candidate["y"],
                    angle=candidate["angle"],
                    scale=candidate["scale"],
                    score=candidate["score"],
                    solver=candidate["solver"],
                    ncc_score=candidate["ncc_score"],
                    visible_fraction=candidate["visible_fraction"],
                    ecc_cc=candidate["ecc_cc"],
                    warnings=candidate["warnings"],
                )
            )
            if self.params["show_found_region"]:
                angle = math.radians(candidate["angle"])
                cos_a, sin_a = math.cos(angle), math.sin(angle)
                width, height = self._template_size
                half_w = candidate["scale"] * width / 2.0
                half_h = candidate["scale"] * height / 2.0
                corners = [
                    (-half_w, -half_h),
                    (half_w, -half_h),
                    (half_w, half_h),
                    (-half_w, half_h),
                ]
                # Public angle: positive = clockwise in image coordinates.
                points = [
                    (
                        candidate["center_x"] + cos_a * px - sin_a * py,
                        candidate["center_y"] + sin_a * px + cos_a * py,
                    )
                    for px, py in corners
                ]
                layer.add_contour(
                    points, color=(0, 255, 0), thickness=2, ref_id=f"match_{index}"
                )
            if self.params["show_found_origin"]:
                layer.add_axes(
                    (int(candidate["x"]), int(candidate["y"])),
                    angle=candidate["angle"],
                    length=30,
                    color_x=(0, 0, 255),
                    color_y=(0, 255, 0),
                    thickness=2,
                    ref_id=f"match_{index}",
                )
            if self.params["show_labels"]:
                layer.add_text(
                    f"#{index} {candidate['score']:.2f}",
                    (candidate["x"] + 10, candidate["y"] - 10),
                    color=(0, 255, 255),
                    font_scale=0.5,
                )

        if self.params.get("act_as_fixture") and results:
            best = results[0]
            vision_img.x = best.x
            vision_img.y = best.y
            vision_img.angle = best.angle

        self.outputs["output image"] = vision_img
        self.outputs["count"] = len(results)
        self.outputs["results"] = results
        self.outputs["warnings"] = warnings
        self._copy_overlays_to_input(vision_img, v_img)
        self.data_changed.emit()

    def get_config_widget(self):
        from .config import ShapeMatchConfig
        return ShapeMatchConfig(self)
