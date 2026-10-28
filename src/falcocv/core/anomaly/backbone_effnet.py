"""ONNX backbone for multi-layer EfficientNet-B0 feature extraction.

FeatureExtractor: 224×224 input → (256, 28, 28) spatial feature map.
"""

import os
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

_MODEL_PATH = None

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_TARGET_SIZE = 224


class EfficientNetBackbone:
    """EfficientNet-B0 multi-layer feature extractor (ONNX).

    Lazy-loads the ONNX model on first use.
    Input:  (H, W, 3) uint8 BGR → resize+pad to 224×224, RGB, ImageNet norm
    Output: (256, 28, 28) float32 spatial feature map, L2-normalized per channel.
    """

    def __init__(self, target_size: int = 224):
        self._session = None
        self._input_name = None
        self._output_name = None
        self.target_size = target_size

    def _ensure_model(self):
        global _MODEL_PATH
        if self._session is not None:
            return
        if _MODEL_PATH is None:
            from falcocv.core.models import model_path
            _MODEL_PATH = str(model_path("EfficientNet-B0-MultiLayer.onnx"))
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"ONNX model not found at {_MODEL_PATH}. "
                "Run scripts/export_effnet_multilayer.py to generate it."
            )
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(
            _MODEL_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Resize to 224×224 with letterbox, RGB, ImageNet normalise."""
        h, w = image.shape[:2]
        scale = _TARGET_SIZE / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        if (new_h, new_w) != (h, w):
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        else:
            resized = image

        pad_h = _TARGET_SIZE - new_h
        pad_w = _TARGET_SIZE - new_w
        top = pad_h // 2
        left = pad_w // 2
        canvas = np.full((_TARGET_SIZE, _TARGET_SIZE, 3), 128, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized

        # BGR → RGB, uint8 → float32, normalize
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        img = np.transpose(rgb, (2, 0, 1))      # (3, 224, 224)
        img = np.expand_dims(img, 0)             # (1, 3, 224, 224)
        return img.astype(np.float32)

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract spatial feature map.

        Args:
            image: (H, W, 3) uint8 BGR array.

        Returns:
            (256, 28, 28) float32 feature map, L2-normalized per channel.
        """
        self._ensure_model()
        inp = self.preprocess(image)
        outputs = self._session.run([self._output_name], {self._input_name: inp})
        return outputs[0][0].astype(np.float32, copy=False)
