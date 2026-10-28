# Logic module for Classifier.

"""Classifier — multi-class classification via embedding similarity + k-NN.

Record mode captures ROI images assigned to the currently active class tab.
Training: backbone ONNX → triplet loss projection head → k-NN memory bank.
"""

__all__ = ["Classifier"]

import base64
import io
import os
import uuid
import numpy as np
import cv2


from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi, create_roi_only_output
from falcocv.core.pointcloud_utils import propagate_3d
from falcocv.core.classifier.inference import ClassifierInferenceEngine
from falcocv.core.classifier.projection import ProjectionHead
from falcocv.core.classifier.classifier_memory import ClassifierMemoryBank
def _img_to_b64(img: np.ndarray) -> str:
    ret, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("ascii")


def _b64_to_img(data: str) -> np.ndarray:
    buf = base64.b64decode(data)
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


_DEFAULT_THRESHOLD = 0.65


@register_tool(prefix="[CLAS]", default_name="Classifier", group="Machine Learning")
class Classifier(BaseTool):
    def __init__(self, name="Classifier"):
        super().__init__(name)
        self.params = {
            "threshold": _DEFAULT_THRESHOLD,
            "roi": [0, 0, 0, 0, 0],
            "output_mode": "Full Image",
            "is_trained": False,
            "show_labels": True,
            "show_roi_overlay": True,
            "max_samples_per_class": 50,
            "save_train_images": True,
            "k_nn": 3,
            "class_data": {},
            "class_labels": {},
            "projection_b64": None,
            "memory_bank_data": None,
            "thresholds": {},
            "copy_overlays_to_input": True,
        }
        self.inputs = {"input image": None}
        self.params["expected_class"] = ""
        self.outputs = {
            "output image": None,
            "class_label": "",
            "score": 0.0,
            "pass": False,
        }
        self._inference: ClassifierInferenceEngine | None = None
        self._trained = False
        self._class_data: dict[str, list[np.ndarray]] = {}
        self._class_labels: dict[str, str] = {}
        self._active_class: str | None = None
        self._recording = False
        self._recording_count = 0

    @property
    def is_trained(self):
        return self._trained

    @property
    def recording(self):
        return self._recording

    def get_class_list(self) -> list[tuple[str, str, int]]:
        return [(cid, self._class_labels.get(cid, cid), len(self._class_data.get(cid, [])))
                for cid in self._class_data]

    def num_classes(self) -> int:
        return len(self._class_data)

    def _new_class_id(self) -> str:
        return f"class_{uuid.uuid4().hex[:8]}"

    def add_class(self, label: str) -> str:
        cid = self._new_class_id()
        self._class_data[cid] = []
        self._class_labels[cid] = label
        return cid

    def remove_class(self, class_id: str):
        self._class_data.pop(class_id, None)
        self._class_labels.pop(class_id, None)
        if self._active_class == class_id:
            self._active_class = next(iter(self._class_data.keys())) if self._class_data else None

    def get_class_images(self, class_id: str) -> list[np.ndarray]:
        return self._class_data.get(class_id, [])

    def start_recording(self, class_id: str):
        self._active_class = class_id
        self._recording = True
        self._recording_count = 0
        if not self.params.get("save_train_images", True):
            if class_id in self._class_data:
                self._class_data[class_id].clear()

    def stop_recording(self):
        self._recording = False
        self._active_class = None
        self._recording_count = 0

    def train(self) -> str:
        # Validate: at least 2 classes with >= 5 images each
        valid_classes = {cid: imgs for cid, imgs in self._class_data.items() if len(imgs) >= 5}
        if len(valid_classes) < 2:
            return "Need >=2 classes with >=5 images each."

        engine = ClassifierInferenceEngine()
        features_by_class: dict[str, np.ndarray] = {}

        total_imgs = 0
        for cid, imgs in valid_classes.items():
            feats = []
            for img in imgs:
                try:
                    f = engine.extract_features(img)
                    feats.append(f)
                except Exception:
                    continue
            if len(feats) < 5:
                continue
            features_by_class[cid] = np.stack(feats)
            total_imgs += len(feats)

        if len(features_by_class) < 2:
            return "Need >=2 classes with valid feature extraction."

        # Train projection head
        from falcocv.core.classifier.trainer import train_projection_head
        projection = train_projection_head(features_by_class)

        # Build memory bank
        memory_bank = ClassifierMemoryBank()
        for cid, feats in features_by_class.items():
            projected = projection.apply(feats)
            label = self._class_labels.get(cid, cid)
            memory_bank.add_class(cid, label, projected)

        # Store
        self._inference = engine
        self._inference.projection = projection
        self._inference.memory_bank = memory_bank
        self._inference.k_nn = self.params.get("k_nn", 3)
        self._trained = True

        self.params["is_trained"] = True
        self.params["projection_b64"] = projection.to_b64()
        self.params["memory_bank_data"] = memory_bank.to_dict()

        class_summary = ", ".join(
            f"{self._class_labels.get(c, c)} ({len(fs)})"
            for c, fs in features_by_class.items()
        )
        return f"Trained {len(features_by_class)} classes: {class_summary}"

    def clear_training(self):
        self._trained = False
        self._recording = False
        self._recording_count = 0
        self._inference = None
        self.params["is_trained"] = False
        self.params["projection_b64"] = None
        self.params["memory_bank_data"] = None

    def run(self, advance=False):
        img_input = self.inputs.get("input image")
        if img_input is None:
            self.outputs["output image"] = None
            self.outputs["class_label"] = ""
            self.outputs["score"] = 0.0
            self.outputs["pass"] = False
            self.data_changed.emit()
            return

        img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)

        roi = extract_roi(img, self.params["roi"])
        roi_src = roi["roi_src"]
        is_full_img = roi["is_full_img"]
        ax, ay, aw, ah = roi["ax"], roi["ay"], roi["aw"], roi["ah"]

        # Recording mode: capture ROI image for active class
        if self._recording and self._active_class and self._active_class in self._class_data:
            max_s = self.params.get("max_samples_per_class", 50)
            if self._recording_count < max_s:
                self._class_data[self._active_class].append(roi_src.copy())
                self._recording_count += 1
                if self._recording_count >= max_s:
                    self._recording = False

        # Build output image
        vision_img = VisionImage(img.data, x=img.x, y=img.y, angle=img.angle)
        vision_img.inherit_overlays_from(img)
        propagate_3d(img, vision_img, regenerate=False)
        vision_img.get_overlay_layer("default", source_tool=self.name)
        layer = vision_img.get_overlay_layer("default")
        layer.elements = [e for e in layer.elements if e.source_tool != self.name]

        # Draw ROI region overlay
        if not is_full_img and self.params.get("show_roi_overlay", True):
            abs_angle = roi.get("abs_angle", 0.0)
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
            layer.add_roi([rx, ry, float(aw), float(ah), float(abs_angle)],
                          color=(255, 128, 0), thickness=2)

        # Inference
        label_text = ""
        scr = 0.0
        passed = False
        if self._trained and self._inference is not None:
            try:
                result = self._inference.predict(roi_src)
                label_text = result.get("label", "")
                scr = result.get("score", 0.0)
            except Exception:
                pass

        if label_text:
            expected = self.params.get("expected_class", "")
            threshold = self.params.get("threshold", 0.65)
            passed = (not expected or label_text == expected) and scr >= threshold

        # Position text
        if not is_full_img:
            if M_inv_roi is not None:
                text_x = int(rx)
                text_y = max(int(ry - 30), 5)
            else:
                text_x = ax
                text_y = max(ay - 30, 5)
        else:
            text_x = 10
            text_y = 30

        if not label_text:
            layer.add_text("NOT TRAINED" if not self._trained else "ERROR",
                           position=(text_x, text_y),
                           color=(128, 128, 128), font_scale=1.0, thickness=2)
        elif self.params.get("show_labels", True):
            color = (0, 255, 0) if passed else (255, 0, 0)
            status_str = "PASS" if passed else "FAIL"
            layer.add_text(
                f"{label_text} {status_str}", position=(text_x, text_y),
                color=color, font_scale=1.0, thickness=2,
            )
            layer.add_text(
                f"Score: {scr:.2f}", position=(text_x, text_y + 25),
                color=color, font_scale=0.8, thickness=1,
            )

        # Output mode
        if self.params["output_mode"] == "ROI Only":
            self.outputs["output image"] = create_roi_only_output(
                roi, roi_src, vision_img, self.name, ax, ay,
            )
        else:
            self.outputs["output image"] = vision_img

        self.outputs["class_label"] = label_text
        self.outputs["score"] = scr
        self.outputs["pass"] = passed
        self._copy_overlays_to_input(vision_img, img)
        self.data_changed.emit()

    def serialize(self, job_manager=None):
        if self.params.get("save_train_images", True) and self._class_data:
            cd = {}
            for cid, imgs in self._class_data.items():
                cd[cid] = [_img_to_b64(img) for img in imgs]
            self.params["class_data"] = cd
            self.params["class_labels"] = dict(self._class_labels)
        else:
            self.params["class_data"] = {}
        if self._inference and self._inference.projection is not None:
            self.params["projection_b64"] = self._inference.projection.to_b64()
        if self._inference and self._inference.memory_bank is not None:
            self.params["memory_bank_data"] = self._inference.memory_bank.to_dict()
        return super().serialize(job_manager)

    def deserialize(self, data: dict, job_manager=None):
        super().deserialize(data, job_manager)
        self.params.setdefault("expected_class", "")
        self._class_data.clear()
        self._class_labels.clear()
        raw_cd = self.params.get("class_data", {})
        raw_cl = self.params.get("class_labels", {})
        if raw_cd:
            for cid, imgs_b64 in raw_cd.items():
                try:
                    self._class_data[cid] = [_b64_to_img(s) for s in imgs_b64]
                except Exception:
                    self._class_data[cid] = []
                self._class_labels[cid] = raw_cl.get(cid, cid)
        if self.params.get("is_trained"):
            self._restore_from_params()

    def _restore_from_params(self):
        try:
            proj_b64 = self.params.get("projection_b64")
            mem_data = self.params.get("memory_bank_data")
            thresholds = self.params.get("thresholds", {})
            if proj_b64 and mem_data:
                projection = ProjectionHead.from_b64(proj_b64)
                memory_bank = ClassifierMemoryBank.from_dict(mem_data)
                engine = ClassifierInferenceEngine(
                    projection=projection,
                    memory_bank=memory_bank,
                    thresholds=thresholds,
                    k_nn=self.params.get("k_nn", 3),
                )
                self._inference = engine
                self._trained = True
        except Exception:
            self._trained = False
            self._inference = None

    def get_config_widget(self):
        from .config import ClassifierConfig
        return ClassifierConfig(self)
