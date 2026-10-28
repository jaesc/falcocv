"""PatchCore anomaly detection engine.

Uses ResNet18-Layer3 ONNX backbone for spatial feature extraction,
greedy coreset subsampling, per-patch 1-NN scoring, template-matching
alignment, and heatmap generation.  Follows the spec at
specifications/anomaly_detector_spec.md.
"""

__all__ = ["PatchCoreDetector", "AnomalyResult", "greedy_coreset_subsampling"]

import base64
import io
import time
import numpy as np
import cv2
from dataclasses import dataclass
from typing import Optional, Tuple, List

from falcocv.core.anomaly.patch_backbone import PatchBackbone
from falcocv.core.anomaly.backbone_effnet import EfficientNetBackbone
from falcocv.core.anomaly.backbone_layer2 import Layer2Backbone


@dataclass
class AnomalyResult:
    """Result of a single anomaly detection inference."""
    is_anomaly: bool
    score: float                    # max patch distance to nearest neighbor
    score_normalized: float         # score / threshold (>1.0 = anomaly)
    threshold: float
    heatmap: np.ndarray             # [H, W, 3] BGR colormap
    heatmap_overlay: np.ndarray     # [H, W, 3] blended overlay
    patch_scores: np.ndarray        # [H_feat, W_feat] raw distances
    latency_ms: float
    aligned_image: Optional[np.ndarray] = None


def greedy_coreset_subsampling(
    patches: np.ndarray,
    target_size: int = 200,
    seed: int = 42,
) -> np.ndarray:
    """Greedy farthest-point sampling (spec §3.2).

    Args:
        patches: (N, D) float32, L2-normalized patch descriptors.
        target_size: number of coreset patches to select.

    Returns:
        (target_size, D) coreset patches.
    """
    N, D = patches.shape
    if N <= target_size:
        return patches

    rng = np.random.default_rng(seed)
    coreset_indices = [int(rng.integers(0, N))]
    min_dists = np.full(N, fill_value=np.inf, dtype=np.float64)

    for _ in range(target_size - 1):
        last = patches[coreset_indices[-1]]
        dists_to_last = 1.0 - (patches @ last)
        np.minimum(min_dists, dists_to_last, out=min_dists)
        next_idx = int(np.argmax(min_dists))
        coreset_indices.append(next_idx)

    return patches[np.array(coreset_indices)]


class PatchCoreDetector:
    """PatchCore anomaly detector.

    Uses ResNet18-Layer3 ONNX backbone → (256, 28, 28) spatial features.
    Memory bank stores a greedy coreset of ~200 L2-normalized patch descriptors.
    Anomaly score = max per-patch distance to nearest neighbor in coreset.
    """

    BACKBONE_OPTIONS = {
        "fast":     (Layer2Backbone, 224),
        "balanced": (EfficientNetBackbone, 224),
        "quality":  (PatchBackbone, 448),
        "detail":   (Layer2Backbone, 448),
        "hd":       (PatchBackbone, 896),
        "effnet":   (EfficientNetBackbone, 224),
        "patch":    (PatchBackbone, 448),
    }

    def __init__(self, backbone_type: str = "balanced"):
        self.backbone_type = backbone_type
        backbone_class, target_size = self.BACKBONE_OPTIONS[backbone_type]
        self.backbone = backbone_class(target_size=target_size)
        self._target_size = target_size

        self.memory_bank: Optional[np.ndarray] = None      # [N_coreset, D]
        self.threshold: Optional[float] = None
        self.train_scores: Optional[np.ndarray] = None      # for recalibration
        self.template: Optional[np.ndarray] = None          # gray template for alignment
        self.spatial_shape: Optional[Tuple[int, int]] = None
        self.is_fitted: bool = False

        # Config
        self.coreset_size: int = 200
        self.alignment_enabled: bool = True
        self.alignment_max_shift_px: int = 30
        self.heatmap_blur_kernel: int = 21

    # ─────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────

    def fit(self, good_images: List[np.ndarray]) -> dict:
        """Train detector on good (nominal) images.

        Args:
            good_images: list of BGR uint8 arrays, any size.

        Returns:
            dict of training metrics.
        """
        t0 = time.perf_counter()

        if len(good_images) < 5:
            raise ValueError(
                f"Minimum 5 images required, got {len(good_images)}."
            )

        # 1. Resize all to 448x448 and build alignment template
        images_448 = [
            cv2.resize(img, (self._target_size, self._target_size),
                       interpolation=cv2.INTER_LINEAR)
            for img in good_images
        ]

        if self.alignment_enabled:
            self.template = cv2.cvtColor(images_448[0], cv2.COLOR_BGR2GRAY)

        # 2. Extract patch descriptors from each image
        all_patches: List[np.ndarray] = []
        for img_448 in images_448:
            if self.alignment_enabled and self.template is not None:
                aligned = self._align(img_448)
            else:
                aligned = img_448
            feature_map = self.backbone.extract(aligned)        # (256, 28, 28)
            patches, spatial_shape = self._extract_patches(feature_map)
            all_patches.append(patches)

        self.spatial_shape = spatial_shape
        all_patches_np = np.vstack(all_patches).astype(np.float32)

        # 3. Coreset subsampling
        self.memory_bank = greedy_coreset_subsampling(
            all_patches_np, target_size=self.coreset_size,
        )

        # 4. Compute per-image training scores (max patch distance to coreset)
        self.train_scores = self._compute_training_scores(all_patches)

        # 5. Threshold = percentile 99 of training max-dist scores
        self.threshold = float(np.percentile(self.train_scores, 99))
        self.is_fitted = True

        t1 = time.perf_counter()
        return {
            "n_images": len(good_images),
            "n_patches_total": all_patches_np.shape[0],
            "n_patches_coreset": self.memory_bank.shape[0],
            "threshold": self.threshold,
            "train_score_mean": float(self.train_scores.mean()),
            "train_score_p99": float(np.percentile(self.train_scores, 99)),
            "train_score_p100": float(self.train_scores.max()),
            "fit_time_s": t1 - t0,
            "memory_bank_mb": self.memory_bank.nbytes / 1e6,
        }

    def _extract_patches(
        self, feature_map: np.ndarray,
    ) -> Tuple[np.ndarray, Tuple[int, int]]:
        """Convert spatial feature map to L2-normalized patch descriptors.

        Returns:
            patches: (H*W, C) float32 L2-normalized.
            spatial_shape: (H, W).
        """
        C, H, W = feature_map.shape
        patches = feature_map.reshape(C, H * W).T   # (H*W, C)
        norms = np.linalg.norm(patches, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        patches = patches / norms
        return patches.astype(np.float32), (H, W)

    def _compute_training_scores(
        self, all_patches: List[np.ndarray],
    ) -> np.ndarray:
        """Compute max patch distance to coreset for each training image."""
        scores = []
        for patches in all_patches:
            similarities = patches @ self.memory_bank.T   # (N, N_coreset)
            max_sim = similarities.max(axis=1)             # (N,)
            dists = 1.0 - max_sim
            scores.append(float(dists.max()))
        return np.array(scores, dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────────────────────────

    def predict(self, image_bgr: np.ndarray, build_heatmap: bool = True) -> AnomalyResult:
        """Detect anomalies in a single image.

        Args:
            image_bgr: BGR uint8 array, any size.
            build_heatmap: If False, skip pixel-level heatmap to save latency.
                           The returned heatmap/heatmap_overlay will be zeros.

        Returns:
            AnomalyResult with score, decision, and (optionally) heatmap.
        """
        if not self.is_fitted or self.memory_bank is None:
            raise RuntimeError("Call fit() before predict().")

        t0 = time.perf_counter()
        orig_h, orig_w = image_bgr.shape[:2]

        # 1. Resize to 448x448
        img_448 = cv2.resize(image_bgr, (self._target_size, self._target_size),
                             interpolation=cv2.INTER_LINEAR)

        # 2. Alignment
        if self.alignment_enabled and self.template is not None:
            aligned = self._align(img_448)
        else:
            aligned = img_448

        # 3. Feature extraction → scoring
        feature_map = self.backbone.extract(aligned)     # (256, 28, 28)
        score_data = self._score(feature_map)

        max_dist = score_data["max_dist"]
        patch_scores = score_data["patch_scores"]       # (28, 28)

        # 4. Heatmap (optional)
        if build_heatmap:
            heatmap, overlay = self._build_heatmap(patch_scores, image_bgr)
        else:
            heatmap = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            overlay = image_bgr.copy()

        # 5. Normalized score (1.0 = at threshold)
        score_normalized = max_dist / (self.threshold + 1e-8)

        t1 = time.perf_counter()

        return AnomalyResult(
            is_anomaly=max_dist > self.threshold,
            score=max_dist,
            score_normalized=score_normalized,
            threshold=self.threshold,
            heatmap=heatmap,
            heatmap_overlay=overlay,
            patch_scores=patch_scores,
            latency_ms=(t1 - t0) * 1000.0,
            aligned_image=aligned,
        )

    def _score(self, feature_map: np.ndarray) -> dict:
        """Compute per-patch distances to memory bank.

        Args:
            feature_map: (256, 28, 28) float32.

        Returns:
            dict with max_dist, mean_dist, patch_scores.
        """
        patches, (H, W) = self._extract_patches(feature_map)   # (784, 256)

        # Distances = 1.0 - max similarity to coreset
        similarities = patches @ self.memory_bank.T             # (784, N_coreset)
        max_sim = similarities.max(axis=1)                     # (784,)
        distances = 1.0 - max_sim                               # (784,)
        patch_scores = distances.reshape(H, W)                  # (28, 28)

        return {
            "max_dist": float(distances.max()),
            "mean_dist": float(distances.mean()),
            "patch_scores": patch_scores,
        }

    # ─────────────────────────────────────────────────────────────────
    # HEATMAP
    # ─────────────────────────────────────────────────────────────────

    def _build_heatmap(
        self,
        patch_scores: np.ndarray,       # (H_feat, W_feat)
        original_image: np.ndarray,     # (H, W, 3) BGR
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build colormap and blended overlay from patch-level scores.

        Returns:
            (heatmap_colored, heatmap_overlay) both at original size.
        """
        H_orig, W_orig = original_image.shape[:2]

        # 1. Interpolate to original size (INTER_LINEAR provides built-in smoothing)
        heatmap_float = cv2.resize(
            patch_scores.astype(np.float32),
            (W_orig, H_orig),
            interpolation=cv2.INTER_LINEAR,
        )

        # 2. Normalize: 0 = score 0, 1 = threshold*2
        upper_bound = max(self.threshold * 2.0, heatmap_float.max() + 1e-8)
        heatmap_norm = np.clip(heatmap_float / upper_bound, 0.0, 1.0)
        heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

        # 3. Colormap
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)

        # 4. Alpha blend (max alpha 0.7) using in-place arithmetic
        alpha_map = np.clip(heatmap_norm * 1.5, 0.0, 0.7)[..., np.newaxis]
        orig_float = original_image.astype(np.float32)
        heatmap_float_c = heatmap_colored.astype(np.float32)

        # of + am * (hc - of) — 1 temp instead of 3-4
        tmp = np.empty_like(orig_float)
        np.subtract(heatmap_float_c, orig_float, out=tmp)
        np.multiply(tmp, alpha_map, out=tmp)
        np.add(orig_float, tmp, out=tmp)
        overlay = tmp.astype(np.uint8)

        return heatmap_colored, overlay

    # ─────────────────────────────────────────────────────────────────
    # ALIGNMENT
    # ─────────────────────────────────────────────────────────────────

    def _build_template(self, images: List[np.ndarray]) -> np.ndarray:
        """Build gray template from the first training image."""
        img_448 = cv2.resize(
            images[0], (self._target_size, self._target_size),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.cvtColor(img_448, cv2.COLOR_BGR2GRAY)

    def _align(self, img_448: np.ndarray) -> np.ndarray:
        """Template-matching alignment on a 448x448 image.

        Compensates ±self.alignment_max_shift_px translation.
        Returns warped image (448x448).
        """
        if self.template is None:
            return img_448

        img_gray = cv2.cvtColor(img_448, cv2.COLOR_BGR2GRAY)

        # Central 50 % patch
        cx = cy = self._target_size // 2
        patch_size = self._target_size // 2
        half = patch_size // 2

        template_patch = self.template[
            cy - half : cy + half,
            cx - half : cx + half,
        ]

        result = cv2.matchTemplate(img_gray, template_patch, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        expected_loc = (cx - half, cy - half)
        dx = max_loc[0] - expected_loc[0]
        dy = max_loc[1] - expected_loc[1]

        dx = int(np.clip(dx, -self.alignment_max_shift_px, self.alignment_max_shift_px))
        dy = int(np.clip(dy, -self.alignment_max_shift_px, self.alignment_max_shift_px))

        if abs(dx) > 2 or abs(dy) > 2:
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            img_448 = cv2.warpAffine(
                img_448, M, (self._target_size, self._target_size),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101,
            )

        return img_448

    # ─────────────────────────────────────────────────────────────────
    # THRESHOLD MANAGEMENT
    # ─────────────────────────────────────────────────────────────────

    def set_threshold(self, new_threshold: float):
        """Manual threshold adjustment (no refit needed)."""
        if new_threshold <= 0:
            raise ValueError("Threshold must be > 0")
        self.threshold = float(new_threshold)

    def recalibrate_threshold(self, percentile: float = 99.0) -> float:
        """Recalibrate threshold from training scores."""
        if self.train_scores is None:
            raise RuntimeError("Model not fitted.")
        self.threshold = float(np.percentile(self.train_scores, percentile))
        return self.threshold

    # ─────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        """Serialize detector state for .cvj embedding.

        The backbone is not saved (comes from ONNX model file).
        """
        state: dict = {
            "backbone_type": self.backbone_type,
            "threshold": self.threshold,
            "spatial_shape": list(self.spatial_shape) if self.spatial_shape else None,
            "coreset_size": self.coreset_size,
            "alignment_enabled": self.alignment_enabled,
            "alignment_max_shift_px": self.alignment_max_shift_px,
            "heatmap_blur_kernel": self.heatmap_blur_kernel,
            "is_fitted": self.is_fitted,
        }

        # Memory bank
        if self.memory_bank is not None:
            buf = io.BytesIO()
            np.save(buf, self.memory_bank)
            state["memory_bank_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")

        # Train scores
        if self.train_scores is not None:
            buf = io.BytesIO()
            np.save(buf, self.train_scores)
            state["train_scores_b64"] = base64.b64encode(buf.getvalue()).decode("ascii")

        # Template
        if self.template is not None:
            _, buf = cv2.imencode(".png", self.template)
            state["template_b64"] = base64.b64encode(buf).decode("ascii")

        return state

    @classmethod
    def from_dict(cls, data: dict) -> "PatchCoreDetector":
        """Restore detector from serialized dict."""
        backbone_type = data.get("backbone_type", "patch")
        detector = cls(backbone_type=backbone_type)
        detector.coreset_size = data.get("coreset_size", 200)
        detector.alignment_enabled = data.get("alignment_enabled", True)
        detector.alignment_max_shift_px = data.get("alignment_max_shift_px", 30)
        detector.heatmap_blur_kernel = data.get("heatmap_blur_kernel", 21)
        detector.is_fitted = data.get("is_fitted", False)

        # Memory bank
        mem_b64 = data.get("memory_bank_b64")
        if mem_b64:
            buf = base64.b64decode(mem_b64)
            detector.memory_bank = np.load(io.BytesIO(buf))

        # Threshold
        detector.threshold = data.get("threshold")

        # Spatial shape
        sp = data.get("spatial_shape")
        if sp:
            detector.spatial_shape = tuple(sp)

        # Train scores
        ts_b64 = data.get("train_scores_b64")
        if ts_b64:
            buf = base64.b64decode(ts_b64)
            detector.train_scores = np.load(io.BytesIO(buf))

        # Template
        tmpl_b64 = data.get("template_b64")
        if tmpl_b64:
            buf = base64.b64decode(tmpl_b64)
            arr = np.frombuffer(buf, dtype=np.uint8)
            detector.template = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)

        return detector
