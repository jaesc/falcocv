"""ONNX backbone for feature extraction (ResNet18-Layer2, stride 8).

Layer2Backbone: 224/448/896 input → (128, H/8, W/8) spatial feature map.
"""

import os
import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

_MODEL_PATH = None

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


class Layer2Backbone:
    """ResNet18-Layer2 feature extractor for lightweight PatchCore.

    Lazy-loads the ONNX model on first use.
    Input:  (H, W, 3) uint8 BGR → resize+pad to target_size, RGB, ImageNet norm
    Output: (128, H, W) float32 spatial feature map where H=W=target_size//8
            because the backbone trunk has stride 8.
    """

    def __init__(self, target_size: int = 448):
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
            _MODEL_PATH = str(model_path("ResNet18-Layer2.onnx"))
        if not os.path.exists(_MODEL_PATH):
            raise FileNotFoundError(
                f"ONNX model not found at {_MODEL_PATH}. "
                "Run scripts/export_resnet_patchcore.py to generate it."
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
        """Resize to target size with letterbox and ImageNet-normalize."""
        h, w = image.shape[:2]
        scale = self.target_size / max(h, w)
        new_h, new_w = int(round(h * scale)), int(round(w * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        pad_h = self.target_size - new_h
        pad_w = self.target_size - new_w
        top = pad_h // 2
        left = pad_w // 2
        canvas = np.full((self.target_size, self.target_size, 3), 128, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized

        img = canvas.astype(np.float32) / 255.0
        img = (img - _MEAN.squeeze((0, 2, 3))) / _STD.squeeze((0, 2, 3))
        img = np.transpose(img, (2, 0, 1))
        img = np.expand_dims(img, 0)
        return img

    def extract(self, image: np.ndarray) -> np.ndarray:
        """Extract spatial feature map from an image.

        Args:
            image: (H, W, 3) uint8 BGR or RGB array.

        Returns:
            (128, H, W) float32 feature map where H=W=target_size//8.
        """
        self._ensure_model()
        if image.shape[2] == 3:
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb = image
        inp = self.preprocess(rgb)
        outputs = self._session.run([self._output_name], {self._input_name: inp})
        feature_map = outputs[0][0]
        return feature_map.astype(np.float32, copy=False)
