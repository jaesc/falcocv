"""Simple global-average + 1-NN memory bank for anomaly detection."""

import base64
import io
import numpy as np


class SimpleMemoryBank:
    """Global-average-pooling feature + 1-NN distance anomaly detector.

    Extracts a global vector by spatially averaging the feature map,
    L2-normalizes it, and stores it.  Inference computes the Euclidean
    distance to the nearest stored vector.  Very fast and robust to small
    fixture shifts because the spatial averaging acts as a natural
    translation-invariant aggregator.
    """

    def __init__(self):
        self._embeddings: np.ndarray | None = None  # (N, C) float32, L2 normalized
        self.norm_factor: float = 1.0

    def fit(self, feature_maps: list[np.ndarray]):
        """Build memory bank from training feature maps.

        Args:
            feature_maps: List of (C, H, W) float32 arrays.
        """
        embs = []
        for fm in feature_maps:
            # Global Average Pooling over spatial dims
            vec = fm.mean(axis=(1, 2))  # (C,)
            # L2 normalize
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embs.append(vec)

        self._embeddings = np.stack(embs).astype(np.float32)

        # Compute norm_factor from leave-one-out 1-NN distances
        N = self._embeddings.shape[0]
        if N <= 1:
            self.norm_factor = 1.0
            return

        dots = self._embeddings @ self._embeddings.T  # (N, N)
        np.fill_diagonal(dots, -1.0)  # exclude self
        dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))
        nn_dists = dists.min(axis=1)
        self.norm_factor = float(nn_dists.max())
        self.norm_factor = max(self.norm_factor, 1e-6)

    def score(self, feature_map: np.ndarray) -> float:
        """Compute anomaly score for a single feature map.

        Args:
            feature_map: (C, H, W) float32 array.

        Returns:
            float in [0, 1].
        """
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            return 0.0

        # Global Average Pooling
        vec = feature_map.mean(axis=(1, 2))
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm

        # 1-NN distance
        dots = vec @ self._embeddings.T
        np.clip(dots, -1.0, 1.0, out=dots)
        dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))
        nn = float(dists.min())

        raw = nn / self.norm_factor
        return float(np.clip(raw, 0.0, 1.0))

    def to_b64(self) -> str:
        """Serialize memory bank to base64 string."""
        if self._embeddings is None:
            return ""
        buf = io.BytesIO()
        np.save(buf, self._embeddings)
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
            bank._embeddings = np.load(buf)
        except Exception:
            bank._embeddings = None
        return bank
