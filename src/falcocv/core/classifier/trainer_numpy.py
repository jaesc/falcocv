"""Pure-numpy reimplementation of the projection head trainer.

Numerically validated against the torch version in
tests/test_trainer_equivalence.py.

Replaces torch.optim.Adam, torch.nn.functional, DataLoader, and triplet loss
with equivalent numpy operations, producing identical results for the same
seed and inputs.
"""

import numpy as np
from .projection import ProjectionHead


# ── Adam optimiser (torch defaults) ────────────────────────────────────────

def _adam(W: np.ndarray, grad: np.ndarray, m: np.ndarray, v: np.ndarray,
          t: int, lr: float = 1e-3, beta1: float = 0.9,
          beta2: float = 0.999, eps: float = 1e-8,
          weight_decay: float = 1e-4) -> tuple:
    grad = grad + weight_decay * W
    np.multiply(beta1, m, out=m)
    m += (1.0 - beta1) * grad
    np.multiply(beta2, v, out=v)
    v += (1.0 - beta2) * (grad * grad)
    m_hat = m / (1.0 - beta1 ** t)
    v_hat = v / (1.0 - beta2 ** t)
    step = lr * m_hat / (np.sqrt(v_hat) + eps)
    W -= step
    return W, m, v


# ── Triplet Loss with hard mining ─────────────────────────────────────────

def _triplet_loss(embeddings: np.ndarray, labels: np.ndarray,
                  margin: float = 0.2) -> float:
    B = embeddings.shape[0]
    sim = embeddings @ embeddings.T
    dist = 1.0 - sim
    labels_eq = labels[:, None] == labels[None, :]
    labels_ne = ~labels_eq
    eye = np.eye(B, dtype=bool)
    pos_mask = labels_eq & ~eye
    pos_dists = np.where(pos_mask, dist, 0.0)
    hardest_pos = pos_dists.max(axis=1)
    neg_dists = np.where(labels_ne, dist, np.inf)
    hardest_neg = neg_dists.min(axis=1)
    valid = pos_mask.any(axis=1)
    if not valid.any():
        return 0.0
    loss = np.maximum(0, hardest_pos[valid] - hardest_neg[valid] + margin)
    return float(loss.mean())


# ── Gradient of triplet loss w.r.t. normalised embeddings ─────────────────

def _triplet_grad(embeddings: np.ndarray, labels: np.ndarray,
                  margin: float = 0.2) -> np.ndarray:
    B = embeddings.shape[0]
    sim = embeddings @ embeddings.T
    dist = 1.0 - sim
    labels_eq = labels[:, None] == labels[None, :]
    labels_ne = ~labels_eq
    eye = np.eye(B, dtype=bool)
    pos_mask = labels_eq & ~eye
    neg_mask = labels_ne
    pos_dists = np.where(pos_mask, dist, 0.0)
    hardest_pos = pos_dists.max(axis=1, keepdims=True)
    pos_max_mask = (pos_dists == hardest_pos) & pos_mask
    neg_dists = np.where(neg_mask, dist, np.inf)
    hardest_neg = neg_dists.min(axis=1, keepdims=True)
    neg_min_mask = (neg_dists == hardest_neg) & neg_mask
    valid = pos_mask.any(axis=1)
    margin_ok = (hardest_pos.squeeze() - hardest_neg.squeeze() + margin > 0) & valid
    n_valid = int(margin_ok.sum())
    if n_valid == 0:
        return np.zeros_like(sim)
    dloss = np.zeros_like(sim)
    for i in range(B):
        if not margin_ok[i]:
            continue
        dloss[i, :] -= pos_max_mask[i].astype(np.float32)
        dloss[i, :] += neg_min_mask[i].astype(np.float32)
    dloss /= n_valid
    return dloss


def _normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norms + 1e-12)


def _normalize_grad(x: np.ndarray, grad_out: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    x_norm = x / norms
    return (grad_out - x_norm * (grad_out * x_norm).sum(axis=-1, keepdims=True)) / norms


def _xavier_uniform(shape: tuple, rng: np.random.RandomState) -> np.ndarray:
    fan_in, fan_out = shape[0], shape[1]
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return rng.uniform(-limit, limit, size=shape).astype(np.float32)


def _balanced_batch_sampler(
    labels: np.ndarray,
    n_classes: int,
    n_per_class: int,
    n_batches: int,
    rng: np.random.RandomState,
):
    unique = np.unique(labels).tolist()
    indices = {c: np.where(labels == c)[0].tolist() for c in unique}
    batch_size = n_classes * n_per_class
    for _ in range(n_batches):
        batch = []
        for c in list(indices.keys()):
            pool = indices[c]
            k = min(n_per_class, len(pool))
            chosen = rng.choice(pool, size=k, replace=False).tolist()
            batch.extend(chosen)
        if len(batch) < 2:
            continue
        yield batch[:batch_size]


def train_projection_head(
    features_by_class: dict[str, np.ndarray],
    in_dim: int = 1280,
    out_dim: int = 128,
    n_epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    margin: float = 0.2,
) -> ProjectionHead:
    """Train a linear projection head W with triplet loss + Adam.

    Pure numpy equivalent of core.classifier.trainer.train_projection_head.
    Behaviour matches the torch version identically for the same seed and data.
    """
    all_features, all_labels, label_map = [], [], {}
    for label_idx, (class_id, feats) in enumerate(features_by_class.items()):
        label_map[class_id] = label_idx
        all_features.append(feats)
        n = len(feats)
        all_labels.append(np.full(n, label_idx, dtype=np.int64))
    all_features_np = np.concatenate(all_features, axis=0).astype(np.float32)
    all_labels_np = np.concatenate(all_labels, axis=0)

    rng = np.random.RandomState(42)
    W = _xavier_uniform((in_dim, out_dim), rng)
    m = np.zeros_like(W)
    v = np.zeros_like(W)

    best_loss = float("inf")
    best_W = W.copy()
    global_step = 0
    n_classes = len(features_by_class)
    n_per_class = max(2, batch_size // n_classes)

    in_scheduler = lr
    scheduler_patience = 5
    scheduler_factor = 0.5
    min_lr = 1e-5
    scheduler_counter = 0

    for epoch in range(n_epochs):
        epoch_losses = []
        sampler = _balanced_batch_sampler(
            all_labels_np, n_classes, n_per_class, n_batches=50, rng=rng,
        )
        for batch_idx in sampler:
            global_step += 1
            batch_feats = all_features_np[batch_idx]
            batch_labels = all_labels_np[batch_idx]

            projected = batch_feats @ W
            normalized = _normalize(projected)

            loss_val = _triplet_loss(normalized, batch_labels, margin)
            epoch_losses.append(loss_val)

            if loss_val > 0:
                dloss = _triplet_grad(normalized, batch_labels, margin)
                d_embed = (dloss + dloss.T) @ normalized
                dnorm = _normalize_grad(projected, d_embed)
                dW = batch_feats.T @ dnorm
                norm_dW = np.linalg.norm(dW)
                if norm_dW > 5.0:
                    dW = dW * (5.0 / norm_dW)
                W, m, v = _adam(W, dW, m, v, global_step, lr=in_scheduler)

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_W = W.copy()
            scheduler_counter = 0
        else:
            scheduler_counter += 1
            if scheduler_counter >= scheduler_patience:
                in_scheduler = max(in_scheduler * scheduler_factor, min_lr)
                scheduler_counter = 0

    ph = ProjectionHead(in_dim=in_dim, out_dim=out_dim)
    ph.from_torch_state(best_W)
    return ph
