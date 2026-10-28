"""PP-OCRv5 ONNX Runtime engine — pure recognition (no detection).

Pipeline:
   1. Cropped ROI → SVTR text recognition → string

No external dependencies beyond onnxruntime, numpy, opencv.
"""

__all__ = [
    "PPOCRv5Engine", "OCRConfig", "OCRResult",
    "preload_engine",
]

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np
import cv2

try:
    from Levenshtein import distance as _lev_distance
except ImportError:
    _lev_distance = None

from falcocv.core.models import models_dir

_MODELS_DIR = models_dir()

_MODEL_NAMES = {
    "rec_mobile": "ch_PP-OCRv5_rec_mobile.onnx",
    "rec_server": "ch_PP-OCRv5_rec_server.onnx",
}

_DICT_PATH = _MODELS_DIR / "ppocrv5_dict.txt"


@dataclass
class OCRConfig:
    model_type: str = "server"
    min_region_confidence: float = 0.5
    split_lines: bool = False


@dataclass
class OCRResult:
    text: str = ""
    confidence: float = 0.0
    regions: List[dict] = field(default_factory=list)
    latency_ms: float = 0.0


def _load_char_dict(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def _normalize_rec(image, target_w=None):
    """Resize height to 48, normalize to [-1, 1], return CHW.
    When target_w is given the output width is ALWAYS exactly target_w
    (zero-padded if narrower, resized if wider)."""
    h, w = image.shape[:2]
    scale = 48.0 / h
    new_w = max(int(w * scale), 16)
    img = cv2.resize(image, (new_w, 48), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32)
    np.subtract(img, 127.5, out=img)
    np.divide(img, 127.5, out=img)
    img = img.transpose(2, 0, 1)
    if target_w is not None and target_w != new_w:
        if target_w > new_w:
            padded = np.zeros((3, 48, target_w), dtype=np.float32)
            padded[:, :, :new_w] = img
            return padded
        else:
            img_hwc = img.transpose(1, 2, 0)
            img_hwc = cv2.resize(img_hwc, (target_w, 48), interpolation=cv2.INTER_LINEAR)
            return img_hwc.transpose(2, 0, 1)
    return img


def _crop_line_to_text(line_img: np.ndarray, margin: int = 20) -> np.ndarray:
    """Crop a full-width line image to just the text region + margin horizontally.
    Uses adaptive threshold (same as _split_lines) for robust binarization."""
    gray = cv2.cvtColor(line_img, cv2.COLOR_BGR2GRAY)
    h_line = gray.shape[0]
    block_size = max(15, (h_line // 16) | 1)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, 5)
    profile = thresh.mean(axis=0)
    threshold = float(profile.max()) * 0.05
    in_text = profile > threshold
    if not np.any(in_text):
        return line_img
    left = int(np.argmax(in_text))
    right = int(len(in_text) - np.argmax(in_text[::-1]) - 1)
    left = max(0, left - margin)
    right = min(line_img.shape[1] - 1, right + margin)
    cropped = line_img[:, left:right + 1]
    min_w = min(line_img.shape[1], 40)
    if cropped.shape[1] < min_w:
        center = (left + right) // 2
        left = max(0, center - min_w // 2)
        right = left + min_w
        if right > line_img.shape[1]:
            right = line_img.shape[1]
            left = max(0, right - min_w)
        return line_img[:, left:right]
    return cropped


def _split_lines(image: np.ndarray, min_gap_ratio: float = 0.3) -> List[Tuple[int, int]]:
    """Split image into horizontal text lines by projection profile vectorized."""
    h, w = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    block_size = max(15, (h // 16) | 1)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, block_size, 5)
    profile = thresh.mean(axis=1).astype(np.float32)

    p_min, p_max = profile.min(), profile.max()
    if p_max - p_min < 1:
        return [(0, h)]
    profile = (profile - p_min) / (p_max - p_min)

    in_text = profile > 0.15
    diff = np.diff(in_text.astype(np.int8))
    starts = np.where(diff == 1)[0] + 1
    ends = np.where(diff == -1)[0] + 1
    if in_text[0]:
        starts = np.insert(starts, 0, 0)
    if in_text[-1]:
        ends = np.append(ends, h)
    lines = [(s, e) for s, e in zip(starts, ends) if e - s >= 5]

    if not lines:
        return [(0, h)]

    heights = [e - s for s, e in lines]
    median_h = float(np.median(heights)) if heights else h
    min_gap = max(2, int(median_h * min_gap_ratio))

    merged = [lines[0]]
    for y1, y2 in lines[1:]:
        if y1 - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], y2)
        else:
            merged.append((y1, y2))

    return merged


def _decode_ctc(logits: np.ndarray, char_dict: List[str]) -> Tuple[str, float]:
    """Vectorized CTC greedy decode from logits."""
    preds = logits.argmax(axis=1)
    mask = np.ones_like(preds, dtype=bool)
    mask[1:] = preds[1:] != preds[:-1]
    mask &= preds != 0
    if not np.any(mask):
        return "", 0.0
    char_indices = preds[mask] - 1
    char_confs = logits[mask, preds[mask]]
    text = "".join(
        char_dict[c] if 0 <= c < len(char_dict) else "" for c in char_indices
    )
    confidence = float(np.mean(char_confs)) if len(char_confs) > 0 else 0.0
    return text, confidence


class PPOCRv5Engine:
    """PP-OCRv5 text recognition via ONNX Runtime (no detection)."""

    def __init__(self, model_type: str = "server", config: Optional[OCRConfig] = None):
        self.config = config or OCRConfig(model_type=model_type)
        self.model_type = model_type
        self.char_dict: Optional[List[str]] = None
        self.rec_session = None
        self._is_ready = False
        self._rec_input_name = None

    @property
    def is_ready(self) -> bool:
        return self._is_ready and self.rec_session is not None

    def load(self):
        import onnxruntime as ort

        rec_path = _MODELS_DIR / _MODEL_NAMES[f"rec_{self.model_type}"]

        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        opts.intra_op_num_threads = max(1, os.cpu_count() or 4)
        priority = [
            "DmlExecutionProvider",
            "OpenVINOExecutionProvider",
            "CPUExecutionProvider",
        ]
        available = ort.get_available_providers()
        providers = [p for p in priority if p in available]
        if not providers:
            providers = ["CPUExecutionProvider"]
        try:
            self.rec_session = ort.InferenceSession(
                str(rec_path), opts, providers=providers
            )
        except Exception:
            providers = ["CPUExecutionProvider"]
            self.rec_session = ort.InferenceSession(
                str(rec_path), opts, providers=providers
            )
        self._rec_input_name = self.rec_session.get_inputs()[0].name

        if _DICT_PATH.exists():
            self.char_dict = _load_char_dict(_DICT_PATH)
        else:
            self.char_dict = list("0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")

        self._is_ready = True

    def _ensure_loaded(self):
        if not self._is_ready:
            self.load()

    def _recognize(self, image: np.ndarray, target_w=None) -> Tuple[str, float]:
        if image is None or image.size == 0:
            return "", 0.0
        inp = _normalize_rec(image, target_w=target_w)
        inp_batch = np.expand_dims(inp, 0)
        outputs = self.rec_session.run(None, {self._rec_input_name: inp_batch})
        return _decode_ctc(outputs[0][0], self.char_dict)

    def _recognize_batch(self, images: List[np.ndarray]) -> List[Tuple[str, float]]:
        """Batch multiple line images into a single ONNX inference."""
        if not images:
            return []

        norms = []
        widths = []
        for img in images:
            if img is None or img.size == 0:
                norms.append(None)
                widths.append(0)
                continue
            h, w = img.shape[:2]
            scale = 48.0 / h
            new_w = max(int(w * scale), 16)
            widths.append(new_w)
            norms.append(_normalize_rec(img))

        valid = [(i, n, w_) for i, (n, w_) in enumerate(zip(norms, widths)) if n is not None]
        if not valid:
            return [("", 0.0)] * len(images)

        max_w = max(w_ for _, _, w_ in valid)
        batch = np.zeros((len(valid), 3, 48, max_w), dtype=np.float32)
        for bi, (_, n, w_) in enumerate(valid):
            batch[bi, :, :, :w_] = n

        outputs = self.rec_session.run(None, {self._rec_input_name: batch})

        results = [("", 0.0)] * len(images)
        for bi, (orig_i, _, _) in enumerate(valid):
            results[orig_i] = _decode_ctc(outputs[0][bi], self.char_dict)
        return results

    def read(self, image: np.ndarray) -> OCRResult:
        self._ensure_loaded()
        t0 = time.perf_counter()
        h, w = image.shape[:2]

        regions = []
        all_text = []
        all_conf = []

        if self.config.split_lines:
            lines = _split_lines(image)
            if len(lines) > 1:
                # Two-pass: crop all lines, compute max width using clamped height
                cropped = []
                max_new_w = 0
                for y1, y2 in lines:
                    line_img = _crop_line_to_text(image[y1:y2, :])
                    if line_img.shape[1] == image.shape[1]:
                        continue
                    h_line = line_img.shape[0]
                    h_scale = max(h_line, 20)
                    new_w = min(max(int(line_img.shape[1] * 48.0 / h_scale), 16), 800)
                    if new_w > max_new_w:
                        max_new_w = new_w
                    cropped.append((y1, y2, line_img))
                target_w = max_new_w
                # Parallel inference with threads: ONNX Runtime releases the GIL
                # during session.run(), so each thread runs in true parallel on
                # available CPU cores.
                results = [("", 0.0)] * len(cropped)
                with ThreadPoolExecutor(max_workers=min(3, len(cropped))) as pool:
                    fut_map = {}
                    for bi, (y1, y2, line_img) in enumerate(cropped):
                        fut = pool.submit(self._recognize, line_img, target_w)
                        fut_map[fut] = (bi, y1, y2)
                    for f in as_completed(fut_map):
                        bi, y1, y2 = fut_map[f]
                        results[bi] = f.result()
                for bi, (y1, y2, _) in enumerate(cropped):
                    text, conf = results[bi]
                    if text and conf >= self.config.min_region_confidence:
                        regions.append({
                            "text": text,
                            "confidence": conf,
                            "bbox": (0, y1, w, y2 - y1),
                            "centroid": (w / 2.0, (y1 + y2) / 2.0),
                            "points": [[0, y1], [w, y1], [w, y2], [0, y2]],
                        })
                        all_text.append(text)
                        all_conf.append(conf)
            else:
                for y1, y2 in lines:
                    text, conf = self._recognize(image[y1:y2, :])
                    if text and conf >= self.config.min_region_confidence:
                        regions.append({
                            "text": text,
                            "confidence": conf,
                            "bbox": (0, y1, w, y2 - y1),
                            "centroid": (w / 2.0, (y1 + y2) / 2.0),
                            "points": [[0, y1], [w, y1], [w, y2], [0, y2]],
                        })
                        all_text.append(text)
                        all_conf.append(conf)
        else:
            text, conf = self._recognize(image)
            if text and conf >= self.config.min_region_confidence:
                regions.append({
                    "text": text,
                    "confidence": conf,
                    "bbox": (0, 0, w, h),
                    "centroid": (w / 2.0, h / 2.0),
                    "points": [[0, 0], [w, 0], [w, h], [0, h]],
                })
                all_text.append(text)
                all_conf.append(conf)

        full_text = "\n".join(all_text)
        avg_conf = float(np.mean(all_conf)) if all_conf else 0.0
        elapsed = (time.perf_counter() - t0) * 1000.0
        return OCRResult(
            text=full_text,
            confidence=avg_conf,
            regions=regions,
            latency_ms=elapsed,
        )

    def verify(self, image: np.ndarray, expected: str) -> OCRResult:
        result = self.read(image)
        found = result.text.strip()
        exp = expected.strip()
        if not found or not exp:
            return result
        if _lev_distance is not None:
            dist = _lev_distance(found, exp)
            max_len = max(len(found), len(exp), 1)
            result.confidence = 1.0 - (dist / max_len)
        else:
            result.confidence = float(found == exp)
        return result

    def locate(self, image: np.ndarray) -> OCRResult:
        result = self.read(image)
        for region in result.regions:
            bx, by, bw, bh = region["bbox"]
            text = region["text"]
            if not text or bw <= 0 or bh <= 0:
                continue
            char_w = bw / len(text)
            region["char_boxes"] = [
                {"char": c, "bbox": (int(bx + i * char_w), by, int(char_w), bh)}
                for i, c in enumerate(text)
            ]
        return result


# ── Global preload ────────────────────────────────────────────────────────

_PRELOADED_ENGINES: Dict[str, PPOCRv5Engine] = {}


def preload_engine(model_type: str = "server") -> PPOCRv5Engine:
    if model_type not in _PRELOADED_ENGINES:
        eng = PPOCRv5Engine(model_type=model_type)
        eng.load()
        _PRELOADED_ENGINES[model_type] = eng
    return _PRELOADED_ENGINES[model_type]
