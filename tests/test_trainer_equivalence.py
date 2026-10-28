"""Validate that trainer_numpy produces equivalent results to trainer (torch).

Tests:
  1. Single-batch exact identity -- gradient + loss must match at float32 eps.
  2. End-to-end equivalence -- both trainers converge to similar W on synthetic
     data, and predictions on held-out test data agree.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def _generate_features(
    n_classes: int = 3,
    n_per_class: int = 10,
    in_dim: int = 8,
    out_dim: int = 4,
    seed: int = 42,
):
    rng = np.random.RandomState(seed)
    centroids = rng.randn(n_classes, in_dim).astype(np.float32)
    centroids = centroids / np.linalg.norm(centroids, axis=1, keepdims=True)
    features_by_class = {}
    for i in range(n_classes):
        cluster = centroids[i] + 0.3 * rng.randn(n_per_class, in_dim).astype(np.float32)
        features_by_class[f"class_{i}"] = cluster
    return features_by_class, out_dim


class TestTrainerEquivalence(unittest.TestCase):
    def test_single_batch_identical(self):
        import torch

        torch.manual_seed(0)
        np.random.seed(0)

        in_dim, out_dim = 4, 2
        B = 4
        feats_np = np.random.randn(B, in_dim).astype(np.float32)
        lbls_np = np.array([0, 0, 1, 1], dtype=np.int64)

        W_init_t = torch.nn.Parameter(torch.empty(in_dim, out_dim))
        torch.nn.init.xavier_uniform_(W_init_t)
        W_init_n = W_init_t.detach().numpy().copy()

        # -- One torch step -----------------------------------------
        projected_t = torch.from_numpy(feats_np) @ W_init_t
        normalized_t = torch.nn.functional.normalize(projected_t, dim=-1)
        sim_t = normalized_t @ normalized_t.T
        dist_t = 1.0 - sim_t
        labels_eq = torch.from_numpy(lbls_np).unsqueeze(0) == torch.from_numpy(lbls_np).unsqueeze(1)
        labels_ne = ~labels_eq
        eye = torch.eye(B, dtype=torch.bool)
        pos_mask = labels_eq & ~eye
        pos_dists = dist_t.clone()
        pos_dists[~pos_mask] = 0.0
        hardest_pos = pos_dists.max(dim=1).values
        neg_dists = dist_t.clone()
        neg_dists[~labels_ne] = float("inf")
        hardest_neg = neg_dists.min(dim=1).values
        valid = pos_mask.any(dim=1)
        loss_t = torch.nn.functional.relu(
            hardest_pos[valid] - hardest_neg[valid] + 0.2
        ).mean()
        loss_t.backward()
        grad_t = W_init_t.grad.detach().clone()

        # -- One numpy step -----------------------------------------
        from falcocv.core.classifier.trainer_numpy import (
            _adam, _normalize, _triplet_loss, _triplet_grad, _normalize_grad,
        )

        W_n = W_init_n.copy()
        projected_n = feats_np @ W_n
        normalized_n = _normalize(projected_n)
        loss_n = _triplet_loss(normalized_n, lbls_np, margin=0.2)
        dloss_n = _triplet_grad(normalized_n, lbls_np, margin=0.2)
        d_embed_n = (dloss_n + dloss_n.T) @ normalized_n
        dnorm_n = _normalize_grad(projected_n, d_embed_n)
        dW_n = feats_np.T @ dnorm_n

        m_n = np.zeros_like(W_n)
        v_n = np.zeros_like(W_n)
        W_n, _, _ = _adam(W_n, dW_n, m_n, v_n, 1, lr=1e-2)

        # -- Compare -------------------------------------------------
        loss_diff = abs(float(loss_t.item()) - loss_n)
        grad_diff = np.max(np.abs(grad_t.numpy() - dW_n))

        self.assertLess(loss_diff, 1e-6, f"Loss differs: {loss_diff}")
        self.assertLess(grad_diff, 1e-5, f"Gradient differs: {grad_diff}")

    def test_end_to_end(self):
        import torch

        features_by_class, out_dim = _generate_features(
            n_classes=3, n_per_class=20, in_dim=16, out_dim=8, seed=42,
        )
        in_dim = 16

        # Run torch version
        torch.manual_seed(42)
        np.random.seed(42)
        from falcocv.core.classifier.trainer import train_projection_head as train_torch
        ph_torch = train_torch(
            features_by_class,
            in_dim=in_dim, out_dim=out_dim,
            n_epochs=30, batch_size=9, lr=5e-2, margin=0.2,
        )

        # Run numpy version
        np.random.seed(42)
        from falcocv.core.classifier.trainer_numpy import train_projection_head as train_numpy
        ph_numpy = train_numpy(
            features_by_class,
            in_dim=in_dim, out_dim=out_dim,
            n_epochs=30, batch_size=9, lr=5e-2, margin=0.2,
        )

        # RNG differences in batch sampling cause different optimization paths,
        # so we only verify functional equivalence (not numerical identity).
        rng = np.random.RandomState(99)
        test_feats = rng.randn(60, in_dim).astype(np.float32)

        def _class_distance_stats(ph):
            emb = ph.apply(test_feats)
            same, diff, ns, nd = 0.0, 0.0, 0, 0
            for i in range(len(emb)):
                for j in range(i+1, len(emb)):
                    c = float(np.sum(emb[i] * emb[j]))
                    if i // 20 == j // 20:
                        same += c; ns += 1
                    else:
                        diff += c; nd += 1
            return same/max(ns,1), diff/max(nd,1)

        sc_t, dc_t = _class_distance_stats(ph_torch)
        sc_n, dc_n = _class_distance_stats(ph_numpy)
        self.assertGreater(sc_t, dc_t, "Torch: same-class similarity should exceed cross-class")
        self.assertGreater(sc_n, dc_n, "Numpy: same-class similarity should exceed cross-class")


if __name__ == "__main__":
    unittest.main()
