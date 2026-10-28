import numpy as np
import json
from collections import Counter
from typing import Optional


class ClassifierMemoryBank:
    def __init__(self):
        self._classes: dict[str, dict] = {}
        self._all_embeddings: Optional[np.ndarray] = None
        self._all_labels: Optional[np.ndarray] = None
        self._all_class_ids: Optional[np.ndarray] = None
        self._cache_valid = False

    def add_class(self, class_id: str, label: str, embeddings: np.ndarray):
        centroid = embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        dists = 1.0 - (embeddings @ centroid)
        std_dev = float(dists.std())
        self._classes[class_id] = {
            "label": label,
            "centroid": centroid,
            "std_dev": std_dev,
            "n_samples": len(embeddings),
            "raw_embeddings": embeddings,
        }
        self._cache_valid = False

    def get_class_labels(self) -> dict[str, str]:
        return {cid: c["label"] for cid, c in self._classes.items()}

    def num_classes(self) -> int:
        return len(self._classes)

    def num_samples(self) -> int:
        return sum(c["n_samples"] for c in self._classes.values())

    def _rebuild_cache(self):
        all_embs, all_labels, all_cids = [], [], []
        for cid, entry in self._classes.items():
            for emb in entry["raw_embeddings"]:
                all_embs.append(emb)
                all_labels.append(entry["label"])
                all_cids.append(cid)
        if all_embs:
            self._all_embeddings = np.stack(all_embs)
            self._all_labels = np.array(all_labels)
            self._all_class_ids = np.array(all_cids)
        else:
            self._all_embeddings = None
            self._all_labels = None
            self._all_class_ids = None
        self._cache_valid = True

    def predict(self, embedding: np.ndarray, k: int = 3) -> dict:
        if not self._cache_valid:
            self._rebuild_cache()
        if self._all_embeddings is None:
            return {
                "class_id": None,
                "label": "",
                "similarity": 0.0,
                "confidence": 0.0,
                "vote_fraction": 0.0,
                "status": "NO_CLASSES",
            }
        k = min(k, len(self._all_embeddings))
        sims = self._all_embeddings @ embedding
        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_cids = self._all_class_ids[top_k_idx]
        top_k_sims = sims[top_k_idx]
        vote_counts = Counter(top_k_cids)
        best_cid = vote_counts.most_common(1)[0][0]
        vote_frac = vote_counts[best_cid] / k
        mask = top_k_cids == best_cid
        mean_sim = float(top_k_sims[mask].mean())
        confidence = float(vote_frac * mean_sim)
        return {
            "class_id": best_cid,
            "label": self._classes[best_cid]["label"],
            "similarity": mean_sim,
            "confidence": confidence,
            "vote_fraction": vote_frac,
            "status": "KNOWN",
        }

    def calibrate_thresholds(self, rejection_percentile: float = 5.0) -> dict[str, float]:
        thresholds = {}
        for cid, entry in self._classes.items():
            sims = entry["raw_embeddings"] @ entry["centroid"]
            th = float(np.percentile(sims, rejection_percentile))
            thresholds[cid] = float(np.clip(th, 0.50, 0.95))
        return thresholds

    def to_dict(self) -> dict:
        data = {}
        for cid, entry in self._classes.items():
            data[cid] = {
                "label": entry["label"],
                "centroid": entry["centroid"].tolist(),
                "std_dev": entry["std_dev"],
                "n_samples": entry["n_samples"],
                "raw_embeddings": entry["raw_embeddings"].tolist(),
            }
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "ClassifierMemoryBank":
        bank = cls()
        for cid, entry in data.items():
            raw = np.array(entry["raw_embeddings"], dtype=np.float32)
            bank._classes[cid] = {
                "label": entry["label"],
                "centroid": np.array(entry["centroid"], dtype=np.float32),
                "std_dev": entry["std_dev"],
                "n_samples": entry["n_samples"],
                "raw_embeddings": raw,
            }
        return bank
