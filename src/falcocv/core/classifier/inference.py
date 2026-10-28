import os
import numpy as np
import cv2
import onnxruntime as ort
from .projection import ProjectionHead
from .classifier_memory import ClassifierMemoryBank


_MODEL_PATH = None
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_TARGET_SIZE = 224


def _get_model_path() -> str:
    global _MODEL_PATH
    if _MODEL_PATH is None:
        from falcocv.core.models import model_path
        _MODEL_PATH = str(model_path("EfficientNet-B0.onnx"))
    return _MODEL_PATH


def preload_classifier_model() -> bool:
    """Preload the classifier backbone ONNX model at startup.
    
    Safe to call early — returns False if model file doesn't exist.
    """
    path = _get_model_path()
    if not os.path.exists(path):
        return False
    opts = ort.SessionOptions()
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    try:
        ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        return True
    except Exception:
        return False


class ClassifierInferenceEngine:
    def __init__(self, projection: ProjectionHead | None = None,
                 memory_bank: ClassifierMemoryBank | None = None,
                 thresholds: dict[str, float] | None = None,
                 k_nn: int = 3):
        self._session = None
        self._input_name = None
        self._output_name = None
        self.projection = projection
        self.memory_bank = memory_bank
        self.thresholds = thresholds or {}
        self.k_nn = k_nn

    def _ensure_model(self):
        if self._session is not None:
            return
        path = _get_model_path()
        if not os.path.exists(path):
            raise FileNotFoundError(f"ONNX model not found: {path}")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 4
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self._session = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        scale = _TARGET_SIZE / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top = (_TARGET_SIZE - new_h) // 2
        left = (_TARGET_SIZE - new_w) // 2
        canvas = np.full((_TARGET_SIZE, _TARGET_SIZE, 3), 128, dtype=np.uint8)
        canvas[top:top + new_h, left:left + new_w] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        rgb = rgb.astype(np.float32) / 255.0
        rgb = (rgb - _MEAN) / _STD
        return np.transpose(rgb, (2, 0, 1))[np.newaxis, ...].astype(np.float32)

    def extract_features(self, image_bgr: np.ndarray) -> np.ndarray:
        self._ensure_model()
        tensor = self._preprocess(image_bgr)
        features = self._session.run([self._output_name], {self._input_name: tensor})[0][0]
        return features.astype(np.float32)

    def predict(self, image_bgr: np.ndarray) -> dict:
        if self.projection is None or not self.projection.is_trained():
            return {"class_id": "", "label": "", "score": 0.0}
        if self.memory_bank is None or self.memory_bank.num_classes() == 0:
            return {"class_id": "", "label": "", "score": 0.0}
        features = self.extract_features(image_bgr)
        embedding = self.projection.apply(features.reshape(1, -1))[0]
        result = self.memory_bank.predict(embedding, k=self.k_nn)
        if result["class_id"] is None:
            return {"class_id": "", "label": "", "score": 0.0}
        centroid = self.memory_bank._classes[result["class_id"]]["centroid"]
        score = float(embedding @ centroid)
        return {"class_id": result["class_id"], "label": result["label"], "score": score}
