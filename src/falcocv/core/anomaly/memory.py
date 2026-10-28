"""Memory bank for anomaly detection — stores OK embeddings and K-NN distance scoring."""

import base64
import io
import numpy as np


def embeddings_to_b64(embeddings: np.ndarray) -> str:
    """Serialize (N, D) embedding array to base64 string."""
    buf = io.BytesIO()
    np.save(buf, embeddings)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def b64_to_embeddings(data: str) -> np.ndarray:
    """Deserialize base64 string back to (N, D) array."""
    buf = io.BytesIO(base64.b64decode(data))
    return np.load(buf)


class MemoryBank:
    """K-NN based memory bank for anomaly scoring.

    Stores L2-normalized embeddings and computes the mean distance to the
    k nearest training samples.  Since embeddings are unit vectors, cosine
    distance and Euclidean distance are monotonically related; we use the
    Euclidean formulation for intuitive scale.
    """

    def __init__(self, embeddings: np.ndarray | None = None, k: int = 1):
        self._embeddings: np.ndarray | None = None  # (N, D) float32, L2 normalized
        self.k: int = max(k, 1)
        self.norm_factor: float = 1.0
        if embeddings is not None and len(embeddings) > 0:
            self.fit(embeddings)

    def fit(self, embeddings: np.ndarray):
        """Store embeddings and compute the 99.5th-percentile k-NN distance
        on the training set (leave-one-out) for later score normalization."""
        arr = np.asarray(embeddings, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        # Re-normalize to unit length (safety net)
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        self._embeddings = arr / norms

        N = self._embeddings.shape[0]
        if N <= 1:
            self.norm_factor = 1.0
            return

        # Pairwise Euclidean distances for unit vectors:
        # ||a - b||^2 = 2 - 2*a·b
        dots = self._embeddings @ self._embeddings.T  # (N, N)
        # Clamp because of numerical drift
        np.clip(dots, -1.0, 1.0, out=dots)
        dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))  # (N, N)

        # Exclude self-distance from nearest-neighbour search
        large = dists.max() + 1.0
        np.fill_diagonal(dists, large)

        k_eff = min(self.k, N - 1)
        # Partial sort: k smallest distances per row
        k_nearest = np.partition(dists, k_eff - 1, axis=1)[:, :k_eff]
        knn_means = k_nearest.mean(axis=1)

        self.norm_factor = float(np.percentile(knn_means, 99.5))
        self.norm_factor = max(self.norm_factor, 1e-6)

    def knn_distance(self, query: np.ndarray) -> float:
        """Distance to the k nearest stored embeddings (mean when k > 1)."""
        if self._embeddings is None or self._embeddings.shape[0] == 0:
            return 0.0

        q = np.asarray(query, dtype=np.float32).reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        dots = (q @ self._embeddings.T).flatten()  # (N,)
        np.clip(dots, -1.0, 1.0, out=dots)
        dists = np.sqrt(np.maximum(0.0, 2.0 - 2.0 * dots))

        k_eff = min(self.k, len(dists))
        k_nearest = np.partition(dists, k_eff - 1)[:k_eff]
        return float(k_nearest.mean())

    def score(self, query: np.ndarray) -> float:
        """Normalized anomaly score in [0, 1].

        0  = very similar to training samples (low k-NN distance).
        1  = very different (high k-NN distance).

        The 99.5th-percentile k-NN distance of the training set is used as
        a reference, giving generous margin for natural fixture variation.
        """
        dist = self.knn_distance(query)
        raw = dist / self.norm_factor
        return float(np.clip(raw, 0.0, 1.0))

    def to_b64(self) -> str:
        """Serialize memory bank (embeddings + k + norm_factor) to a base64 string."""
        if self._embeddings is None:
            return ""
        buf = io.BytesIO()
        np.save(buf, self._embeddings)
        header = f"{self.k}|{self.norm_factor:.8f}|"
        payload = base64.b64encode(buf.getvalue()).decode("ascii")
        return header + payload

    @classmethod
    def from_b64(cls, data: str):
        """Deserialize from base64."""
        bank = cls()
        if not data:
            return bank
        try:
            parts = data.split("|", 2)
            if len(parts) == 3:
                bank.k = int(parts[0])
                bank.norm_factor = float(parts[1])
                payload = parts[2]
            else:
                # Legacy fallback: treat entire string as plain embeddings
                payload = data
                bank.k = 5
                bank.norm_factor = 1.0
            buf = io.BytesIO(base64.b64decode(payload))
            bank._embeddings = np.load(buf)
        except Exception:
            bank._embeddings = None
        return bank
