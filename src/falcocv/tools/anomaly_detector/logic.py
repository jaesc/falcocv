# Logic module for AnomalyDetector.

"""AnomalyDetector — PatchCore-based unsupervised defect detection.

Recording mode captures ROI images from the pipeline for training.
Uses ResNet18-Layer3 ONNX backbone + greedy coreset + per-patch k-NN.
"""

__all__ = ["AnomalyDetector"]

import base64
import io
import numpy as np
import cv2


from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi, create_roi_only_output
from falcocv.core.pointcloud_utils import propagate_3d
from falcocv.core.anomaly.detector import PatchCoreDetector, AnomalyResult


def _img_to_b64(img: np.ndarray) -> str:
    ret, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("ascii")


def _b64_to_img(data: str) -> np.ndarray:
    buf = base64.b64decode(data)
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


@register_tool(prefix="[ANOM]", default_name="Anomaly Detector", group="Machine Learning")
class AnomalyDetector(BaseTool):
    def __init__(self, name="AnomalyDetector"):
        super().__init__(name)
        self.params = {
            "threshold": 0.5,
            "roi": [0, 0, 0, 0, 0],
            "output_mode": "Full Image",
            "is_trained": False,
            "show_labels": True,
            "show_roi_overlay": True,
            "show_heatmap": True,
            "alignment_enabled": True,
            "max_train_samples": 20,
            "save_train_images": True,
            "copy_overlays_to_input": True,
            "backbone_type": "balanced",
        }
        self.inputs = {"input image": None}
        self.outputs = {
            "output image": None,
            "score": 0.0,
            "pass": False,
        }
        self._detector: PatchCoreDetector | None = None
        self._trained = False
        self._train_images: list[np.ndarray] = []
        self._recording = False
        self._recording_count = 0

    @property
    def is_trained(self):
        return self._trained

    @property
    def recording(self):
        return self._recording

    @property
    def train_count(self):
        return len(self._train_images)

    def start_recording(self):
        self._recording = True
        self._recording_count = 0
        if not self.params.get("save_train_images", True):
            self._train_images.clear()

    def stop_recording(self):
        self._recording = False
        self._recording_count = 0

    def clear_training(self):
        self._trained = False
        self._recording = False
        self._recording_count = 0
        self._detector = None
        self.params["is_trained"] = False
        self.params.pop("detector_state", None)

    def run(self, advance=False):
        img_input = self.inputs.get("input image")
        if img_input is None:
            self.outputs["output image"] = None
            self.outputs["score"] = 0.0
            self.outputs["pass"] = False
            self.data_changed.emit()
            return

        img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)

        roi = extract_roi(img, self.params["roi"])
        roi_src = roi["roi_src"]
        is_full_img = roi["is_full_img"]
        ax, ay, aw, ah = roi["ax"], roi["ay"], roi["aw"], roi["ah"]

        # Recording mode: capture ROI image
        if self._recording:
            max_s = self.params.get("max_train_samples", 20)
            if self._recording_count < max_s:
                self._train_images.append(roi_src.copy())
                self._recording_count += 1
                if self._recording_count >= max_s:
                    self._recording = False

        # Sync alignment flag and check coherence
        if self._detector is not None:
            self._detector.alignment_enabled = self.params.get("alignment_enabled", True)
            if self._detector.alignment_enabled and self._detector.template is None:
                self.clear_training()

        # Inference
        score = 0.0
        passed = False
        result: AnomalyResult | None = None
        if self._trained and self._detector is not None:
            try:
                result = self._detector.predict(
                    roi_src, build_heatmap=self.params.get("show_heatmap", True),
                )
                score = result.score
                threshold = self.params.get("threshold", 0.5)
                passed = score < threshold
            except Exception:
                pass

        # Build output base
        show_hm = self.params.get("show_heatmap", True) and result is not None

        if self.params["output_mode"] == "ROI Only" and show_hm:
            base_data = result.heatmap_overlay
        elif self.params["output_mode"] != "ROI Only" and show_hm:
            base_data = img.data.copy()
            h_hm, w_hm = result.heatmap_overlay.shape[:2]
            y2 = min(ay + h_hm, base_data.shape[0])
            x2 = min(ax + w_hm, base_data.shape[1])
            base_data[ay:y2, ax:x2] = result.heatmap_overlay[:y2 - ay, :x2 - ax]
        else:
            base_data = img.data

        if self.params["output_mode"] == "ROI Only":
            vision_img = VisionImage(img.data, x=img.x, y=img.y, angle=img.angle)
        else:
            vision_img = VisionImage(base_data, x=img.x, y=img.y, angle=img.angle)

        vision_img.inherit_overlays_from(img)
        propagate_3d(img, vision_img, regenerate=True)
        vision_img.get_overlay_layer("default", source_tool=self.name)
        layer = vision_img.get_overlay_layer("default")
        layer.elements = [e for e in layer.elements if e.source_tool != self.name]

        # Compute absolute ROI positions (needed for overlay + text)
        M_inv_roi = roi.get("M_inv", None)
        if M_inv_roi is not None:
            cx_w = aw / 2.0
            cy_w = ah / 2.0
            abs_cx = M_inv_roi[0, 0] * cx_w + M_inv_roi[0, 1] * cy_w + M_inv_roi[0, 2]
            abs_cy = M_inv_roi[1, 0] * cx_w + M_inv_roi[1, 1] * cy_w + M_inv_roi[1, 2]
            rx = abs_cx - aw / 2.0
            ry = abs_cy - ah / 2.0
        else:
            rx, ry = float(ax), float(ay)

        # Draw ROI region overlay
        if not is_full_img and self.params.get("show_roi_overlay", True):
            layer.add_roi([rx, ry, float(aw), float(ah), float(roi.get("abs_angle", 0.0))],
                          color=(255, 128, 0), thickness=2)

        # Text overlay positioning
        if not is_full_img:
            text_x = int(rx)
            text_y = max(int(ry - 30), 5)
        else:
            text_x = 10
            text_y = 30

        if not self._trained:
            layer.add_text("NOT TRAINED",
                           position=(text_x, text_y),
                           color=(128, 128, 128), font_scale=1.0, thickness=2)
        elif self.params.get("show_labels", True):
            color = (0, 255, 0) if passed else (255, 0, 0)
            status_str = "PASS" if passed else "FAIL"
            layer.add_text(status_str, position=(text_x, text_y),
                           color=color, font_scale=1.0, thickness=2)
            layer.add_text(f"Score: {score:.2f}", position=(text_x, text_y + 25),
                           color=color, font_scale=0.8, thickness=1)

        # Output mode
        if self.params["output_mode"] == "ROI Only":
            working = result.heatmap_overlay if show_hm else roi_src
            self.outputs["output image"] = create_roi_only_output(
                roi, working, vision_img, self.name, ax, ay,
            )
        elif show_hm:
            self.outputs["output image"] = vision_img
        else:
            self.outputs["output image"] = vision_img

        self.outputs["score"] = score
        self.outputs["pass"] = passed
        self._copy_overlays_to_input(vision_img, img)
        self.data_changed.emit()

    def serialize(self, job_manager=None):
        if self.params.get("save_train_images", True) and self._train_images:
            self.params["train_images_b64"] = [_img_to_b64(img) for img in self._train_images]
        else:
            self.params.pop("train_images_b64", None)
        if self._detector is not None:
            self.params["detector_state"] = self._detector.to_dict()
        return super().serialize(job_manager)

    def deserialize(self, data: dict, job_manager=None):
        super().deserialize(data, job_manager)
        self._train_images.clear()
        raw = self.params.get("train_images_b64")
        if raw:
            try:
                self._train_images = [_b64_to_img(s) for s in raw]
            except Exception:
                self._train_images = []
        ds = self.params.get("detector_state")
        if ds:
            try:
                self._detector = PatchCoreDetector.from_dict(ds)
                self._trained = self._detector.is_fitted
                bb = self._detector.backbone_type
                self.params["backbone_type"] = {"effnet": "balanced", "patch": "quality"}.get(bb, bb)
            except Exception:
                self._trained = False
                self._detector = None

    def get_config_widget(self):
        from .config import AnomalyDetectorConfig
        return AnomalyDetectorConfig(self)
