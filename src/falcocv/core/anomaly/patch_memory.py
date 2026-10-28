"""Patch-level memory bank for PatchCore anomaly detection."""

import base64
import io
import numpy as np


class PatchMemoryBank:
    """PatchCore-style memory bank.

    Stores L2-normalized local patch features from nominal (OK) images.
    Anomaly score is computed per-patch as the distance to the nearest
    stored patch, then aggregated (mean or max) across the spatial map.

    To keep memory and compute manageable, the bank randomly subsamples
    patches when the total exceeds *max_patches* (default 50 000).
    """

    DEFAULT_MAX_PATCHES = 10000

    def __init__(self):
        self._patches: np.ndarray | None = None  # (N, C) float32, L2 normalized
        self.norm_factor: float = 1.0
        self._feature_shape: tuple | None = None  # (C, H, W) of training maps

    def fit(self, feature_maps: list[np.ndarray], max_patches: int = DEFAULT_MAX_PATCHES):
        """Build the memory bank from a list of training feature maps.

        Args:
            feature_maps: List of (C, H, W) float32 arrays.
            max_patches: Maximum number of patches to store.  If the total
                         exceeds this, random subsampling is performed.
        """
        all_patches = []
        for fm in feature_maps:
            c, h, w = fm.shape
            patches = fm.reshape(c, h * w).T  # (H*W, C)
            # L2-normalize each patch
            norms = np.linalg.norm(patches, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1.0, norms)
            patches = patches / norms
            all_patches.append(patches)

        raw = np.concatenate(all_patches, axis=0).astype(np.float32)

        # Subsample if too many patches
        if raw.shape[0] > max_patches:
            rng = np.random.default_rng(42)
            indices = rng.choice(raw.shape[0], max_patches, replace=False)
            raw = raw[indices]

        self._patches = raw
        self._feature_shape = feature_maps[0].shape if feature_maps else None

        # Estimate norm_factor from a subset of queries (leave-one-out-ish)
        N = self._patches.shape[0]
        if N <= 1:
            self.norm_factor = 1.0
            return

        subset_size = min(5000, N)
        rng = np.random.default_rng(43)
        query_idx = rng.choice(N, subset_size, replace=False)
        queries = self._patches[query_idx]  # (Q, C)

        # Compute 1-NN distances in batches to avoid huge matrices
        batch_size = 4096
        all_min_dists = []
        for i in range(0, N, batch_size):
            bank_batch = self._patches[i:i + batch_size]  # (B, C)
            dots = queries @ bank_batch.T  # (Q, B)
            np.clip(dots, -1.0, 1.0, out=dots)
            dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))  # (Q, B)
            all_min_dists.append(dists)

        full_dists = np.concatenate(all_min_dists, axis=1)  # (Q, N)
        # Exclude self-distance: set to large value where query_idx matches column
        large = full_dists.max() + 1.0
        for row, qidx in enumerate(query_idx):
            full_dists[row, qidx] = large

        nn_dists = np.partition(full_dists, 0, axis=1)[:, 0]
        self.norm_factor = float(np.percentile(nn_dists, 99.5))
        self.norm_factor = max(self.norm_factor, 1e-6)

    def score(self, feature_map: np.ndarray, agg: str = "mean",
              test_stride: int = 2) -> tuple[float, np.ndarray]:
        """Compute anomaly score for a single feature map.

        Args:
            feature_map: (C, H, W) float32 array.
            agg: Aggregation over spatial patches — "mean" or "max".
            test_stride: Subsample test patches by this factor (e.g. 2
                         evaluates every other patch, giving 4x speedup).

        Returns:
            (image_score, patch_score_map) where patch_score_map is (H, W).
        """
        if self._patches is None or self._patches.shape[0] == 0:
            return 0.0, np.zeros(feature_map.shape[1:], dtype=np.float32)

        c, h, w = feature_map.shape

        # Subsample patches on a grid
        if test_stride > 1:
            fm_sub = feature_map[:, ::test_stride, ::test_stride]
            c, h_sub, w_sub = fm_sub.shape
            patches = fm_sub.reshape(c, h_sub * w_sub).T  # (H_sub*W_sub, C)
        else:
            patches = feature_map.reshape(c, h * w).T  # (H*W, C)

        # L2-normalize each patch
        norms = np.linalg.norm(patches, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        patches = patches / norms

        # Distance to nearest neighbor in memory bank (batched)
        batch_size = 4096
        all_mins = []
        for i in range(0, self._patches.shape[0], batch_size):
            bank_batch = self._patches[i:i + batch_size]  # (B, C)
            dots = patches @ bank_batch.T  # (Q, B)
            np.clip(dots, -1.0, 1.0, out=dots)
            dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))
            all_mins.append(dists.min(axis=1))

        nn_dists = np.stack(all_mins, axis=0).min(axis=0)  # (Q,)

        # Normalize
        scores = nn_dists / self.norm_factor
        scores = np.clip(scores, 0.0, 1.0)

        if test_stride > 1:
            # Upsample score map back to original resolution
            import cv2
            score_map_sub = scores.reshape(h_sub, w_sub)
            score_map = cv2.resize(
                score_map_sub.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR
            )
        else:
            score_map = scores.reshape(h, w)

        if agg == "max":
            image_score = float(score_map.max())
        else:
            image_score = float(score_map.mean())

        return image_score, score_map

    def to_b64(self) -> str:
        """Serialize memory bank to base64 string."""
        if self._patches is None:
            return ""
        buf = io.BytesIO()
        np.save(buf, self._patches)
        header = f"{self.norm_factor:.8f}|"
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return header + payload

    @classmethod
    def from_b64(cls, data: str):
        """Deserialize from base64."""
        bank = cls()
        if not data:
            return bank
        try:
            parts = data.split("|", 1)
            if len(parts) == 2:
                bank.norm_factor = float(parts[0])
                payload = parts[1]
            else:
                payload = data
                bank.norm_factor = 1.0
            buf = io.BytesIO(base64.b64decode(payload))
            bank._patches = np.load(buf)
        except Exception:
            bank._patches = None
        return bank
