import numpy as np
import base64
import io


class ProjectionHead:
    def __init__(self, in_dim: int = 1280, out_dim: int = 128):
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W: np.ndarray | None = None

    def is_trained(self) -> bool:
        return self.W is not None

    def apply(self, features: np.ndarray) -> np.ndarray:
        if self.W is None:
            raise RuntimeError("ProjectionHead not trained")
        projected = features @ self.W
        return projected / (np.linalg.norm(projected, axis=-1, keepdims=True) + 1e-12)

    def from_torch_state(self, W: np.ndarray):
        self.W = W.astype(np.float32)
        assert self.W.shape == (self.in_dim, self.out_dim), (
            f"Expected ({self.in_dim}, {self.out_dim}), got {self.W.shape}"
        )

    def to_b64(self) -> str:
        if self.W is None:
            return ""
        buf = io.BytesIO()
        np.save(buf, self.W)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    @classmethod
    def from_b64(cls, data: str) -> "ProjectionHead":
        if not data:
            return cls()
        buf = io.BytesIO(base64.b64decode(data))
        W = np.load(buf)
        ph = cls(in_dim=W.shape[0], out_dim=W.shape[1])
        ph.W = W
        return ph
