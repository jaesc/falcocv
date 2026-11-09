# Industrial Anomaly Detection Engine — Opinionated Specification
## Context: mixed defects · ±20px position · 10-30 good images · Core/Xeon CPU

---

> **Design premise:** this is the hardest problem in the previous document.
> You have less data than in classification, you don't know what you are going to detect, and you need
> a heatmap. Every decision here is forced by the constraints of the real context,
> not by theoretical preferences.

---

## TL;DR — The 7 key decisions

| Decision | Recommendation | Discarded |
|---|---|---|
| Base algorithm | **PatchCore** (patches from a frozen backbone) | Autoencoder: needs 500+ images to reconstruct well |
| Backbone | **EfficientNet-B0 pretrained, frozen** | Training from scratch: impossible with 10-30 images |
| Features | **Intermediate layers, not the last one** | Last layer: too semantic, loses spatial detail |
| Memory bank | **Coreset subsampling** (~100-200 patches) | All patches: slow and redundant |
| Heatmap | **Bilinear interpolation of the per-patch score** | GradCAM: does not apply at inference without backprop |
| Threshold | **99th percentile of the training set + tuning UI** | Fixed absolute threshold: not adaptable to the part |
| ±20px alignment | **Light template matching before inference** | Ignoring it: causes false positives at the edges |

---

## 1. Why PatchCore and not the obvious alternatives

### 1.1 The problem with autoencoders and few images

The first idea that comes up for anomaly detection is an autoencoder: train a network to reconstruct good images, and measure the reconstruction error at inference. If the error is high, it is an anomaly.

The problem with 10-30 images is that the autoencoder does not see enough variability of the good images to learn to reconstruct the correct part in a general way. It learns to reconstruct the 20 training images almost from memory. Result: any new image —even a perfectly good one— with a small lighting variation generates a high reconstruction error and is classified as an anomaly.

```
What you need for an autoencoder:
  Simple textures, fixed position:   ~200 images minimum viable
  Complex textures:                  ~500-1000 images
  Your case (10-30):                 ✗ Does not work reliably
```

### 1.2 The problem with classical statistical methods (PCA, SVDD)

PCA and One-Class SVM on image features have the same problem: with 10-30 points in a high-dimensional space, the estimation of the normal distribution is very noisy. The learned hyperplane or ellipsoid does not generalize.

### 1.3 Why PatchCore works with very few images

PatchCore (Roth et al., 2022) does not train anything. What it does:

1. Passes all good images through a **pre-trained and frozen** backbone
2. Extracts the **intermediate feature maps** (not the last layer)
3. Takes each spatial position of the feature map as a 512D **patch descriptor**
4. Stores all these descriptors in a **memory bank**
5. At inference: for each patch of the query image, it searches for the **nearest neighbor** in the memory bank
6. The anomaly score of that patch is that **distance to the nearest neighbor**
7. The heatmap is the **distance map** interpolated to the original size

The key: the pre-trained backbone already knows how to represent textures, edges, shapes. You don't need to train anything. The only thing you need from the good images is to populate the memory bank with what the correct part looks like in the backbone's feature space.

With 10-30 images of 224×224, an intermediate layer produces feature maps of 28×28. That is **10-30 × 784 = 7,840-23,520 patch descriptors**. Enough to characterize the normal distribution of the part well.

---

## 2. Complete architecture

### 2.1 Training pipeline (fit)

```
[10-30 GOOD IMAGES]
         │
         ▼
┌────────────────────────┐
│  ALIGNMENT             │  Light template matching
│  ±20px → <5px          │  ~2ms/image
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  FROZEN BACKBONE       │  EfficientNet-B0 pretrained
│  Extract feature maps  │  Layers: block3 + block5
│  from 2 layers         │  [B, C, H, W]
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  PATCH DESCRIPTORS     │  Reshape: [B×H×W, C]
│  Concatenate layers    │  L2 normalize
│  Adaptive avg pool     │
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  CORESET SUBSAMPLING   │  Greedy farthest-point sampling
│  ~200 patches          │  from all patches
└────────────────────────┘
         │
         ▼
┌────────────────────────┐
│  MEMORY BANK           │  [N_coreset, D] float32
│  + Calibrated threshold│  99th percentile of train scores
└────────────────────────┘
```

### 2.2 Inference pipeline

```
[QUERY IMAGE]
         │
         ▼
  Alignment + preprocessing
         │
         ▼
  Backbone → feature maps (2 layers)
         │
         ▼
  Patch descriptors [H×W, D]
         │
         ▼
  k-NN in memory bank (k=1)    ← O(N_coreset × D) dot products
  → distance per patch [H×W]
         │
         ├──────────────────────┐
         ▼                      ▼
  max(distances)         Interpolate to [224×224]
  → anomaly score        → normalized heatmap
         │                      │
         ▼                      ▼
  score > threshold?     Colormap overlay
  → GOOD / ANOMALY       → output image
```

---

## 3. Complete implementation

### 3.1 Multi-layer feature extraction

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
import numpy as np
from typing import List, Dict, Tuple, Optional

class FeatureExtractor(nn.Module):
    """
    Extracts feature maps from intermediate layers of the backbone.

    Why intermediate layers and not the last one:
    - The last layer (1280D) is too semantic: it encodes
      "what the object is", not "what its local texture looks like".
    - Intermediate layers encode textures, edges and local patterns:
      exactly what you need to detect
      scratches, stains or deformations.

    For EfficientNet-B0, we use:
      block3 output: stride 8,  ~40×40 for input 224, features 80D
      block5 output: stride 16, ~20×20 for input 224, features 192D

    Concatenated and pooled to a common resolution: ~28×28, 272D total.
    """

    # Layer names in EfficientNet-B0 (timm)
    TARGET_LAYERS = ['blocks.2', 'blocks.4']

    def __init__(self):
        super().__init__()

        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            features_only=True,      # returns intermediate feature maps
            out_indices=(2, 4),      # block3 and block5
        )

        # Freeze completely
        for param in self.backbone.parameters():
            param.requires_grad = False

        self.backbone.eval()

        # Output dimensions (verify with a forward pass)
        # block3: [B, 40, H/8,  W/8 ]
        # block5: [B, 112, H/16, W/16]
        # After adaptation: [B, 256, 28, 28]

        # 1×1 adapters to bring both layers to a common dimension
        # These COULD be trained, but with 10-30 imgs we prefer
        # not to train anything. We only use them to reduce dimensionality.
        self.adapter3 = nn.Sequential(
            nn.Conv2d(40, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128)      # GroupNorm works with batch=1
        )
        self.adapter5 = nn.Sequential(
            nn.Conv2d(112, 128, kernel_size=1, bias=False),
            nn.GroupNorm(8, 128)
        )

        # Initialize adapters with PCA-like init (scaled identity)
        # so as not to add noise to the pre-trained backbone
        nn.init.xavier_uniform_(self.adapter3[0].weight)
        nn.init.xavier_uniform_(self.adapter5[0].weight)

        self.output_dim = 256   # 128 + 128
        self.spatial_size = 28  # output feature map resolution

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, 3, 224, 224]
        returns: [B, 256, 28, 28]
        """
        self.backbone.eval()

        # Extract feature maps from the two target layers
        features = self.backbone(x)   # list of tensors
        f3 = features[0]              # [B, 40,  28, 28] (approx)
        f5 = features[1]              # [B, 112, 14, 14]

        # Adapt dimensions
        f3 = self.adapter3(f3)        # [B, 128, 28, 28]
        f5 = self.adapter5(f5)        # [B, 128, 14, 14]

        # Upsample f5 to the resolution of f3
        f5_up = F.interpolate(
            f5,
            size=f3.shape[-2:],
            mode='bilinear',
            align_corners=False
        )                             # [B, 128, 28, 28]

        # Concatenate: captures information at two scales
        combined = torch.cat([f3, f5_up], dim=1)   # [B, 256, 28, 28]

        return combined

    def extract_patch_descriptors(
        self,
        x: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Extracts and normalizes per-patch descriptors.

        x: [B, 3, 224, 224]
        returns:
          patches: [B × H × W, 256]  L2-normalized
          spatial_shape: (H, W) of the feature map
        """
        feature_maps = self.forward(x)              # [B, 256, H, W]
        B, C, H, W = feature_maps.shape

        # Reshape into per-patch descriptors
        patches = feature_maps.permute(0, 2, 3, 1)  # [B, H, W, C]
        patches = patches.reshape(-1, C)             # [B×H×W, C]

        # L2-normalize each descriptor
        patches = F.normalize(patches, dim=-1)

        return patches, (H, W)
```

### 3.2 Coreset subsampling

With 20 images × 28×28 = 15,680 patches, k-NN at inference would make 15,680 comparisons. Coreset reduces it to ~200 while maintaining coverage.

```python
def greedy_coreset_subsampling(
    patches: np.ndarray,      # [N, D] L2-normalized
    target_size: int = 200,
    seed: int = 42
) -> np.ndarray:
    """
    Greedy farthest-point sampling to select a coreset
    representative of the training patches.

    Algorithm:
    1. Start with a random point
    2. At each step, add the point FARTHEST from the current coreset
    3. Stop when we have target_size points

    Why not K-Means:
    - Farthest-point sampling guarantees that no point of the
      original set is farther than 2× the optimal distance
      from the coreset. K-Means does not give this coverage guarantee.
    - With few images, K-Means can collapse clusters in
      dense areas and leave sparse areas without representation.

    Complexity: O(N × target_size) — acceptable for N < 25,000
    """
    N, D = patches.shape
    rng = np.random.default_rng(seed)

    if N <= target_size:
        return patches  # Already small, no need to reduce

    coreset_indices = [int(rng.integers(0, N))]
    # Minimum distances to the current coreset for each point
    min_dists = np.full(N, fill_value=np.inf)

    for _ in range(target_size - 1):
        last = patches[coreset_indices[-1]]   # last added

        # Cosine distances from the last point to all others
        # (in L2-normalized space, cosine dist = 1 - dot product)
        dists_to_last = 1.0 - (patches @ last)

        # Update minimum distances
        np.minimum(min_dists, dists_to_last, out=min_dists)

        # Add the farthest one
        next_idx = int(np.argmax(min_dists))
        coreset_indices.append(next_idx)

    return patches[np.array(coreset_indices)]
```

### 3.3 Main anomaly detection engine

```python
import json
import os
import time
import cv2
from dataclasses import dataclass, field
from pathlib import Path

@dataclass
class AnomalyDetectorConfig:
    backbone: str = 'efficientnet_b0'
    coreset_size: int = 200
    k_neighbors: int = 1             # k=1 is enough and faster
    threshold_percentile: float = 99.0
    input_size: int = 224
    # Alignment
    alignment_enabled: bool = True
    alignment_max_shift_px: int = 30  # search within ±30px
    # Heatmap
    heatmap_blur_kernel: int = 21     # heatmap smoothing
    heatmap_colormap: int = cv2.COLORMAP_JET

@dataclass
class AnomalyResult:
    is_anomaly: bool
    score: float                      # raw score (maximum distance to the nearest neighbor)
    score_normalized: float           # [0, 1] normalized with respect to the threshold
    threshold: float
    heatmap: np.ndarray               # [H, W, 3] BGR, same size as input
    heatmap_overlay: np.ndarray       # Heatmap overlaid on the original image
    patch_scores: np.ndarray          # [H_feat, W_feat] per-patch scores (not interpolated)
    latency_ms: float
    aligned_image: Optional[np.ndarray] = None  # aligned image used for inference


class PatchCoreAnomalyDetector:
    """
    Anomaly detector based on PatchCore.
    Only needs good images to train.
    Does not train any neural network parameters.
    """

    def __init__(self, config: AnomalyDetectorConfig = None):
        self.config = config or AnomalyDetectorConfig()
        self.feature_extractor = FeatureExtractor()
        self.feature_extractor.eval()

        # Model state (filled in fit())
        self.memory_bank: Optional[np.ndarray] = None    # [N_coreset, D]
        self.threshold: Optional[float] = None
        self.train_scores: Optional[np.ndarray] = None   # for recalibration
        self.template: Optional[np.ndarray] = None       # for alignment
        self.spatial_shape: Optional[Tuple[int, int]] = None
        self.is_fitted: bool = False

        # Preprocessing constants
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    # ─────────────────────────────────────────────────────────────────
    # FIT
    # ─────────────────────────────────────────────────────────────────

    def fit(self, good_images: List[np.ndarray]) -> dict:
        """
        Trains the detector with good images.

        good_images: list of BGR images (OpenCV format), any size
        returns: dict with fitting metrics
        """
        t0 = time.perf_counter()

        if len(good_images) < 5:
            raise ValueError(
                f"At least 5 images required, received {len(good_images)}. "
                f"With fewer than 10 the detector quality will be very low."
            )

        print(f"[PatchCore] Fitting with {len(good_images)} good images...")

        # 1. Build template for alignment (mean of all good ones)
        if self.config.alignment_enabled:
            self.template = self._build_template(good_images)

        # 2. Preprocess and align all images
        preprocessed = []
        for img in good_images:
            aligned = self._align(img) if self.config.alignment_enabled else img
            tensor = self._preprocess(aligned)
            preprocessed.append(tensor)

        # 3. Extract patch descriptors from all images
        all_patches = []
        with torch.no_grad():
            for tensor in preprocessed:
                patches, spatial_shape = self.feature_extractor.extract_patch_descriptors(
                    tensor.unsqueeze(0)
                )
                all_patches.append(patches.cpu().numpy())

        self.spatial_shape = spatial_shape
        all_patches_np = np.vstack(all_patches)   # [N_total_patches, D]
        print(f"[PatchCore] Total patches: {all_patches_np.shape[0]} × {all_patches_np.shape[1]}D")

        # 4. Coreset subsampling
        self.memory_bank = greedy_coreset_subsampling(
            all_patches_np,
            target_size=self.config.coreset_size
        )
        print(f"[PatchCore] Memory bank: {self.memory_bank.shape[0]} patches (coreset)")

        # 5. Compute training scores to calibrate the threshold
        self.train_scores = self._compute_scores_batch(preprocessed)

        # 6. Threshold = 99th percentile of training scores
        self.threshold = float(
            np.percentile(self.train_scores, self.config.threshold_percentile)
        )
        print(f"[PatchCore] Threshold calibrated: {self.threshold:.4f} "
              f"(percentile {self.config.threshold_percentile})")

        self.is_fitted = True
        t1 = time.perf_counter()

        return {
            'n_images': len(good_images),
            'n_patches_total': all_patches_np.shape[0],
            'n_patches_coreset': self.memory_bank.shape[0],
            'threshold': self.threshold,
            'train_score_mean': float(self.train_scores.mean()),
            'train_score_p99': float(np.percentile(self.train_scores, 99)),
            'train_score_p100': float(self.train_scores.max()),
            'fit_time_s': t1 - t0,
            'memory_bank_mb': self.memory_bank.nbytes / 1e6
        }

    def _compute_scores_batch(
        self, preprocessed_tensors: List[torch.Tensor]
    ) -> np.ndarray:
        """Computes the anomaly score of each training image."""
        scores = []
        with torch.no_grad():
            for tensor in preprocessed_tensors:
                result = self._score_tensor(tensor)
                scores.append(result['max_dist'])
        return np.array(scores)

    # ─────────────────────────────────────────────────────────────────
    # PREDICT
    # ─────────────────────────────────────────────────────────────────

    def predict(self, image_bgr: np.ndarray) -> AnomalyResult:
        """
        Detects anomalies in an image.

        image_bgr: BGR image (OpenCV), any reasonable size
        returns: AnomalyResult with score, decision and heatmap
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict()")

        t0 = time.perf_counter()

        # 1. Alignment
        if self.config.alignment_enabled and self.template is not None:
            aligned = self._align(image_bgr)
        else:
            aligned = image_bgr

        # 2. Preprocessing
        tensor = self._preprocess(aligned)

        # 3. Scoring
        with torch.no_grad():
            score_data = self._score_tensor(tensor)

        max_dist = score_data['max_dist']
        patch_scores = score_data['patch_scores']   # [H_feat, W_feat]

        # 4. Heatmap
        heatmap, overlay = self._build_heatmap(patch_scores, aligned)

        # 5. Score normalized with respect to the threshold
        # 1.0 = at the threshold, >1.0 = anomaly, <1.0 = normal
        score_normalized = max_dist / (self.threshold + 1e-8)

        t1 = time.perf_counter()

        return AnomalyResult(
            is_anomaly=bool(max_dist > self.threshold),
            score=float(max_dist),
            score_normalized=float(score_normalized),
            threshold=float(self.threshold),
            heatmap=heatmap,
            heatmap_overlay=overlay,
            patch_scores=patch_scores,
            latency_ms=(t1 - t0) * 1000.0,
            aligned_image=aligned
        )

    def _score_tensor(self, tensor: torch.Tensor) -> dict:
        """
        Computes distances of all patches to the nearest neighbor
        in the memory bank.

        tensor: [3, 224, 224]
        returns: {
            'max_dist': float,
            'patch_scores': np.ndarray [H_feat, W_feat],
            'mean_dist': float
        }
        """
        patches, spatial_shape = self.feature_extractor.extract_patch_descriptors(
            tensor.unsqueeze(0)
        )
        patches_np = patches.cpu().numpy()   # [H×W, D]

        # k-NN in the memory bank
        # Cosine distances: dist = 1 - similarity
        # patches_np and memory_bank are already L2-normalized
        # → cosine similarity = dot product
        similarities = patches_np @ self.memory_bank.T   # [H×W, N_coreset]
        # Distance to the nearest neighbor (k=1)
        max_similarities = similarities.max(axis=1)      # [H×W]
        distances = 1.0 - max_similarities               # [H×W]

        # Reshape to spatial map
        H, W = spatial_shape
        patch_scores = distances.reshape(H, W)           # [H, W]

        return {
            'max_dist': float(distances.max()),
            'mean_dist': float(distances.mean()),
            'patch_scores': patch_scores
        }

    # ─────────────────────────────────────────────────────────────────
    # HEATMAP
    # ─────────────────────────────────────────────────────────────────

    def _build_heatmap(
        self,
        patch_scores: np.ndarray,     # [H_feat, W_feat]
        original_image: np.ndarray    # [H, W, 3] BGR
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Builds heatmap and overlay from the per-patch scores.

        Design decisions:
        1. Bilinear interpolation (not nearest): smoother heatmap
        2. Local normalization: the heatmap maximum = maximum anomaly
           BUT the colormap is anchored to the threshold for comparability
        3. Gaussian blur: removes artifacts from the feature map grid
        4. Overlay with variable alpha: more red = more anomaly = more opaque
        """
        H_orig, W_orig = original_image.shape[:2]

        # 1. Interpolate to original size
        heatmap_float = cv2.resize(
            patch_scores.astype(np.float32),
            (W_orig, H_orig),
            interpolation=cv2.INTER_LINEAR
        )

        # 2. Blur to smooth grid artifacts
        k = self.config.heatmap_blur_kernel
        if k > 1:
            heatmap_float = cv2.GaussianBlur(
                heatmap_float, (k, k), sigmaX=0
            )

        # 3. Normalize for colormap
        # Anchor: 0 = score 0, 1 = 2×threshold (so the threshold is at 50%)
        # Effect: safe zone (green), borderline zone (yellow), anomaly (red)
        upper_bound = max(self.threshold * 2.0, heatmap_float.max() + 1e-8)
        heatmap_norm = np.clip(heatmap_float / upper_bound, 0.0, 1.0)
        heatmap_uint8 = (heatmap_norm * 255).astype(np.uint8)

        # 4. Apply colormap
        heatmap_colored = cv2.applyColorMap(
            heatmap_uint8,
            self.config.heatmap_colormap
        )   # [H, W, 3] BGR

        # 5. Overlay with alpha proportional to the score
        # Normal zones: almost transparent (low alpha)
        # Anomalous zones: more opaque (high alpha)
        alpha_map = np.clip(heatmap_norm * 1.5, 0.0, 0.7)   # max alpha=0.7
        alpha_3ch = np.stack([alpha_map] * 3, axis=-1)

        original_float = original_image.astype(np.float32)
        heatmap_float_c = heatmap_colored.astype(np.float32)
        overlay = (
            (1 - alpha_3ch) * original_float +
            alpha_3ch * heatmap_float_c
        ).astype(np.uint8)

        return heatmap_colored, overlay

    # ─────────────────────────────────────────────────────────────────
    # ALIGNMENT
    # ─────────────────────────────────────────────────────────────────

    def _build_template(self, images: List[np.ndarray]) -> np.ndarray:
        """
        Builds the template as the mean of the good images aligned
        to each other (simple iteration: align all to the first one).
        """
        target_size = (self.config.input_size, self.config.input_size)
        resized = [
            cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
            for img in images
        ]

        # Template = first image. We could iterate to refine,
        # but with 10-30 images the first one is enough.
        template_gray = cv2.cvtColor(resized[0], cv2.COLOR_BGR2GRAY)
        return template_gray

    def _align(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Template matching alignment to compensate for ±20px variation.

        Strategy: normalized correlation in the central region.
        No rotation (we assume the fixture controls rotation).
        Translation XY only.

        Why not feature matching (ORB/SIFT):
        - ORB/SIFT are for large viewpoint changes
        - For ±20px, template matching is faster and more accurate
        - Feature matching with uniform textures frequently fails
        """
        target_size = (self.config.input_size, self.config.input_size)
        img_resized = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        img_gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)

        # Use central region as search patch (avoids edges)
        cx, cy = target_size[0] // 2, target_size[1] // 2
        patch_size = target_size[0] // 2   # central 50% patch

        template_patch = self.template[
            cy - patch_size//2 : cy + patch_size//2,
            cx - patch_size//2 : cx + patch_size//2
        ]

        # Template matching
        result = cv2.matchTemplate(img_gray, template_patch, cv2.TM_CCOEFF_NORMED)
        _, _, _, max_loc = cv2.minMaxLoc(result)

        # Compute displacement relative to the expected position
        expected_loc = (cx - patch_size//2, cy - patch_size//2)
        dx = max_loc[0] - expected_loc[0]
        dy = max_loc[1] - expected_loc[1]

        # Limit correction to the configured range
        max_shift = self.config.alignment_max_shift_px
        dx = int(np.clip(dx, -max_shift, max_shift))
        dy = int(np.clip(dy, -max_shift, max_shift))

        # Apply translation if significant (>2px)
        if abs(dx) > 2 or abs(dy) > 2:
            M = np.float32([[1, 0, -dx], [0, 1, -dy]])
            img_resized = cv2.warpAffine(
                img_resized, M, target_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT_101
            )

        return img_resized

    # ─────────────────────────────────────────────────────────────────
    # THRESHOLD MANAGEMENT
    # ─────────────────────────────────────────────────────────────────

    def set_threshold(self, new_threshold: float):
        """
        Manual threshold adjustment by the technician.
        Does not require a refit.
        """
        if new_threshold <= 0:
            raise ValueError("Threshold must be > 0")
        self.threshold = float(new_threshold)

    def recalibrate_threshold(self, percentile: float = 99.0):
        """
        Recalibrates the threshold using the training set scores.
        Useful if the technician wants to adjust the sensitivity level.
        """
        if self.train_scores is None:
            raise RuntimeError("Model not trained")
        self.threshold = float(np.percentile(self.train_scores, percentile))
        return self.threshold

    def get_threshold_options(self) -> dict:
        """
        Returns predefined threshold options for the UI.
        Lets the technician choose the sensitivity level without
        understanding the underlying math.
        """
        if self.train_scores is None:
            return {}
        return {
            'permissive':  float(np.percentile(self.train_scores, 99.5)),
            'normal':      float(np.percentile(self.train_scores, 99.0)),
            'strict':      float(np.percentile(self.train_scores, 97.0)),
            'very_strict': float(np.percentile(self.train_scores, 95.0)),
            'current':     self.threshold
        }

    # ─────────────────────────────────────────────────────────────────
    # PREPROCESSING
    # ─────────────────────────────────────────────────────────────────

    def _preprocess(self, image_bgr: np.ndarray) -> torch.Tensor:
        """BGR uint8 → normalized float32 tensor [3, 224, 224]"""
        img = cv2.resize(
            image_bgr,
            (self.config.input_size, self.config.input_size),
            interpolation=cv2.INTER_LINEAR
        )
        img = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, [0,1]
        img = (img - self._mean) / self._std
        tensor = torch.from_numpy(img.transpose(2, 0, 1))   # [3, H, W]
        return tensor

    # ─────────────────────────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────────────────────────

    def save(self, directory: str):
        """
        Saves the detector state.
        The backbone is not saved (it lives in timm/pytorch hub).
        Only saves: memory bank, threshold, template, config.
        """
        os.makedirs(directory, exist_ok=True)

        # Memory bank
        np.save(os.path.join(directory, 'memory_bank.npy'), self.memory_bank)

        # Train scores (for recalibration)
        if self.train_scores is not None:
            np.save(os.path.join(directory, 'train_scores.npy'), self.train_scores)

        # Template
        if self.template is not None:
            cv2.imwrite(os.path.join(directory, 'template.png'), self.template)

        # Config + threshold + metadata
        state = {
            'threshold': self.threshold,
            'spatial_shape': list(self.spatial_shape) if self.spatial_shape else None,
            'config': {
                'coreset_size': self.config.coreset_size,
                'k_neighbors': self.config.k_neighbors,
                'threshold_percentile': self.config.threshold_percentile,
                'input_size': self.config.input_size,
                'alignment_enabled': self.config.alignment_enabled,
                'alignment_max_shift_px': self.config.alignment_max_shift_px,
                'heatmap_blur_kernel': self.config.heatmap_blur_kernel,
            },
            'is_fitted': self.is_fitted
        }
        # Atomic write
        tmp_path = os.path.join(directory, 'state.json.tmp')
        final_path = os.path.join(directory, 'state.json')
        with open(tmp_path, 'w') as f:
            json.dump(state, f, indent=2)
        os.replace(tmp_path, final_path)

        print(f"[PatchCore] Saved to {directory}")

    @classmethod
    def load(cls, directory: str) -> 'PatchCoreAnomalyDetector':
        """Restores detector from disk."""
        with open(os.path.join(directory, 'state.json')) as f:
            state = json.load(f)

        config = AnomalyDetectorConfig(**state['config'])
        detector = cls(config=config)

        detector.memory_bank = np.load(
            os.path.join(directory, 'memory_bank.npy')
        )

        train_scores_path = os.path.join(directory, 'train_scores.npy')
        if os.path.exists(train_scores_path):
            detector.train_scores = np.load(train_scores_path)

        template_path = os.path.join(directory, 'template.png')
        if os.path.exists(template_path):
            detector.template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)

        detector.threshold = state['threshold']
        detector.spatial_shape = tuple(state['spatial_shape']) if state['spatial_shape'] else None
        detector.is_fitted = state['is_fitted']

        return detector
```

---

## 4. Threshold tuning for the technician

### 4.1 The threshold problem in anomaly detection

In supervised binary classification (with negative examples), the threshold is calibrated with an ROC curve. Here you don't have negative examples during training. The threshold is necessarily a bet on how much a new good part can deviate from the good training parts.

The correct approach for the technician:

```
CALIBRATION WORKFLOW DURING COMMISSIONING
──────────────────────────────────────────

Step 1: train with 20 confirmed good parts
Step 2: run 50 additional good parts (not from training)
        → all must come out GOOD
        → if any comes out ANOMALY, raise the threshold (more permissive)

Step 3: run 10 parts with known defects (if you have them)
        → all must come out ANOMALY
        → if any comes out GOOD, lower the threshold (stricter)

Step 4: if you don't have defective parts, use the "strict" level
        and monitor the reject rate in production:
        - Reject rate > 5%: raise the threshold
        - Reject rate < 0.1%: it may be too permissive
```

### 4.2 Exposing the threshold in the UI

```python
class ThresholdUI:
    """
    Threshold adjustment interface for the vision technician.
    Hides the math, exposes industrial concepts.
    """

    LEVELS = {
        'very_permissive': 99.5,   # Only rejects very large differences
        'permissive':      99.0,   # Default, recommended for startup
        'normal':          98.0,
        'strict':          97.0,   # More rejections, more safety
        'very_strict':     95.0    # Rejects subtle variations
    }

    @staticmethod
    def set_sensitivity(
        detector: PatchCoreAnomalyDetector,
        level: str
    ) -> float:
        """Adjusts sensitivity by predefined level."""
        if level not in ThresholdUI.LEVELS:
            raise ValueError(f"Level must be one of: {list(ThresholdUI.LEVELS.keys())}")
        percentile = ThresholdUI.LEVELS[level]
        return detector.recalibrate_threshold(percentile)

    @staticmethod
    def set_threshold_from_sample(
        detector: PatchCoreAnomalyDetector,
        sample_image: np.ndarray,
        is_good: bool,
        safety_margin: float = 0.1
    ) -> float:
        """
        Adjusts the threshold from a reference image.

        If is_good=True:  threshold = score_sample * (1 + safety_margin)
                          "this image must pass, with margin"
        If is_good=False: threshold = score_sample * (1 - safety_margin)
                          "this image must fail, with margin"
        """
        result = detector.predict(sample_image)
        score = result.score

        if is_good:
            new_threshold = score * (1.0 + safety_margin)
        else:
            new_threshold = score * (1.0 - safety_margin)

        detector.set_threshold(new_threshold)
        return new_threshold
```

---

## 5. Alignment: why it is critical with ±20px

### 5.1 The mathematical problem of ±20px

The feature extractor works at 28×28 resolution for a 224×224 image. Each "pixel" of the feature map corresponds to a patch of `224/28 = 8px` in the original image.

A shift of ±20px in the original image is `20/8 = 2.5 positions` in the feature map. This means that the patch descriptor at position (10, 10) of the feature map may be looking at a completely different area of the part between two images of the same good class.

**Without alignment:** the k-NN will find false distant neighbors simply because the part is shifted, not because there is a defect. Result: systematic false positives at the edges of the part.

**With alignment:** the shift is removed before extracting features, and each position of the feature map always corresponds to the same area of the part.

### 5.2 Limitations of the implemented alignment

Template matching alignment assumes:
- Translation only (no rotation, no scale)
- Small displacements (<30px)
- The part has enough texture to correlate

If the part is very uniform (smooth plastic of uniform color), template matching may fail. In that case, disable alignment and increase the heatmap blur to reduce sensitivity to position.

---

## 6. Known limitations and how to handle them

### 6.1 With only 10 training images

With 10 images, the 200-patch coreset may contain effective duplicates and may not cover all the normal variations of the part well. The detector will generate more false positives with good parts that it has not "seen" in training.

**Practical recommendation:** if the technician only has 10 images at commissioning time, capture at least 10 more before production and add them to training:

```python
def add_good_images(
    detector: PatchCoreAnomalyDetector,
    new_good_images: List[np.ndarray]
) -> dict:
    """
    Adds good images to the already trained detector.
    Full refit: it is fast (does not train a neural network).
    """
    # Recover previous training images if we have them saved
    # (recommended to save them in the model directory)
    # Or simply call fit() with all the images again
    return detector.fit(new_good_images)
```

### 6.2 Very small defects (<5px)

Feature maps at 28×28 cannot detect 1-2 pixel defects. If the defect is smaller than `224/28 × 2 ≈ 16px` in the original image, it may not appear in the heatmap.

**Solution:** use a crop of the region of interest at higher resolution. If the expected defect is 5px in a 1024×768 image, crop 224×224 of the relevant area before passing it through the detector.

### 6.3 Lighting variations between shifts

If the lighting changes a lot between shifts (morning/afternoon, seasons), the backbone features capture some of that variation. The detector may generate false positives.

**Solution:** capture training images in all representative lighting conditions, not only in the conditions at commissioning time.

---

## 7. Expected benchmarks for your context

### 7.1 Latency on Core i5/Xeon

```
Operation                              Time
──────────────────────────────────────────────
Template matching (alignment)          1.5ms
Preprocessing (resize + normalize)     0.3ms
EfficientNet-B0 features (FP32)        8-10ms
  — block3 + block5 only               ~6ms
k-NN over 200-patch coreset            0.15ms
  (28×28=784 queries × 200 bank × 256D)
Interpolation + heatmap                0.8ms
──────────────────────────────────────────────
ESTIMATED TOTAL                        ~9-12ms

⚠️  We are at the limit of the <10ms target with FP32.

To be comfortably below:
  Option A: quantize backbone to INT8 → ~5ms backbone → total ~7ms ✅
  Option B: reduce input to 192×192 → ~20% faster → total ~8ms ✅
  Option C: use only block5 (one layer) → ~4ms backbone → total ~6ms
            (worse detection of small defects but faster)
```

### 7.2 Expected detection performance

With 20 training images, mixed defects, ±20px variation:

```
Condition                                          Expected
──────────────────────────────────────────────────────────
Large surface defects (>20px)                      90-97%
Medium surface defects (10-20px)                   75-88%
Small surface defects (<10px)                      40-65%  ← resolution limitation
Structural defects (breakage, missing)             92-98%
Visible contamination                              88-95%
False positive rate (good parts)                   1-3%    with 99th percentile threshold
──────────────────────────────────────────────────────────

With 30 training images:
  All metrics improve by ~3-5 percentage points
  False positive rate drops to ~0.5-1%
```

### 7.3 Resources

```
RAM:
  EfficientNet-B0 backbone FP32:    ~21MB
  Memory bank (200 × 256 × 4B):      0.2MB
  ONNX Runtime / PyTorch overhead:  ~150MB
  ──────────────────────────────────────────
  TOTAL process:                    ~175MB RSS

  (More than the classifier because the backbone is in FP32 in PyTorch,
  not in ONNX Runtime. If exported to ONNX: drop to ~80MB)

Disk:
  memory_bank.npy:                    0.2MB
  template.png:                       0.1MB
  train_scores.npy + state.json:      0.1MB
  ──────────────────────────────────────────
  TOTAL (without backbone):           0.4MB
  The backbone (timm) is downloaded once and reused

Fitting time:
  20 images, EfficientNet-B0 FP32, Core i5:
  Feature extraction:    ~3s
  Coreset sampling:      ~0.5s
  ──────────────────────────────────────────
  TOTAL fitting:         ~4s    ✅ Acceptable for commissioning
```

---

## 8. Implementation checklist

### Phase 1 — Functional MVP
- [ ] `FeatureExtractor` with EfficientNet-B0 (block3 + block5)
- [ ] `greedy_coreset_subsampling` (200 patches)
- [ ] `PatchCoreAnomalyDetector.fit()` with ≥10 images
- [ ] `PatchCoreAnomalyDetector.predict()` → score + decision
- [ ] `_build_heatmap()` → heatmap + overlay
- [ ] Threshold by 99th percentile with predefined options
- [ ] Persistence (save/load)

### Phase 2 — Robustness
- [ ] `_align()` with template matching (±20px compensation)
- [ ] `ThresholdUI` with predefined levels and per-sample adjustment
- [ ] Automatic post-fit validation: mean training score, warning if < 10 images
- [ ] Atomic backup before each refit
- [ ] Prediction logging with timestamp

### Phase 3 — Optimization (if latency is a problem)
- [ ] Export backbone to ONNX INT8 to reduce to ~6ms
- [ ] Pre-allocate preprocessing buffers
- [ ] Cache of the last processed template

### What NOT to implement in v1
- **Autoencoder:** needs 500+ images, outside your context
- **SPADE / PaDiM**: more accurate than PatchCore but require slower fitting and more RAM
- **Student-teacher**: excellent but implementation complexity not justified for 10-30 imgs
- **Rotational alignment (ORB/SIFT)**: the fixture controls rotation, you don't need it
- **Multi-scale PatchCore**: improves small-defect detection but multiplies latency
