# Logic module for ImageCalculatorTool.

import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.config import ConfigManager
from falcocv.core.roi_utils import extract_roi, create_roi_only_output, composite_roi_to_full
from falcocv.core.pointcloud_utils import propagate_3d

OPERATION_DEFS = {
    "add": {
        "label": "Add (A + B)",
        "inputs": 2,
        "params": [],
    },
    "subtract": {
        "label": "Subtract (A − B)",
        "inputs": 2,
        "params": [
            {"key": "reverse", "label": "Reverse (B - A)", "type": "bool", "default": False},
        ],
    },
    "abs_diff": {
        "label": "Abs Diff (A, B)",
        "inputs": 2,
        "params": [],
    },
    "multiply": {
        "label": "Multiply (A * B)",
        "inputs": 2,
        "params": [],
    },
    "divide": {
        "label": "Divide (A / B)",
        "inputs": 2,
        "params": [
            {"key": "reverse", "label": "Reverse (B / A)", "type": "bool", "default": False},
            {"key": "scale", "label": "Scale", "type": "float", "default": 1.0, "min": 0.001},
        ],
    },
    "max": {
        "label": "Max (A, B)",
        "inputs": 2,
        "params": [],
    },
    "min": {
        "label": "Min (A, B)",
        "inputs": 2,
        "params": [],
    },
    "weighted_add": {
        "label": "Weighted Add (A, B)",
        "inputs": 2,
        "params": [
            {"key": "alpha", "label": "Alpha (A weight)", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            {"key": "beta", "label": "Beta (B weight)", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            {"key": "gamma", "label": "Gamma (offset)", "type": "float", "default": 0.0},
        ],
    },
    "bitwise_and": {
        "label": "AND (A, B)",
        "inputs": 2,
        "params": [],
    },
    "bitwise_or": {
        "label": "OR (A, B)",
        "inputs": 2,
        "params": [],
    },
    "bitwise_xor": {
        "label": "XOR (A, B)",
        "inputs": 2,
        "params": [],
    },
    "bitwise_not": {
        "label": "NOT (A)",
        "inputs": 1,
        "params": [],
    },
    "add_constant": {
        "label": "Add Const (A)",
        "inputs": 1,
        "params": [
            {"key": "value", "label": "Value", "type": "int", "default": 0, "min": 0, "max": 255},
        ],
    },
    "multiply_constant": {
        "label": "Mul Const (A)",
        "inputs": 1,
        "params": [
            {"key": "value", "label": "Value", "type": "float", "default": 1.0, "min": 0.0},
        ],
    },
}

OPERATION_ORDER = [
    "add", "subtract", "abs_diff", "multiply", "divide",
    "max", "min", "weighted_add",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "add_constant", "multiply_constant",
]


def _make_default_op(op_type):
    op = {"enabled": True, "type": op_type}
    defs = OPERATION_DEFS.get(op_type)
    if defs:
        for p in defs["params"]:
            op[p["key"]] = p["default"]
    return op


def _op_summary(op):
    op_type = op.get("type", "")
    defs = OPERATION_DEFS.get(op_type)
    if not defs:
        return op_type
    parts = [defs["label"]]
    for p in defs["params"]:
        parts.append(f"{p['label']}={op.get(p['key'], p['default'])}")
    return " | ".join(parts)


def _apply_op(img_a, img_b, op_type, params):
    if img_a is None:
        return None

    a = img_a if isinstance(img_a, np.ndarray) else img_a.data if hasattr(img_a, "data") else img_a
    if not isinstance(a, np.ndarray):
        return None

    a = a.astype(np.uint8)

    if op_type == "bitwise_not":
        return cv2.bitwise_not(a)

    if op_type == "add_constant":
        val = int(params.get("value", 0))
        return np.clip(a.astype(np.int32) + val, 0, 255).astype(np.uint8)

    if op_type == "multiply_constant":
        val = params.get("value", 1.0)
        result = a.astype(np.float64) * val
        return np.clip(result, 0, 255).astype(np.uint8)

    if img_b is None:
        return a

    b = img_b if isinstance(img_b, np.ndarray) else img_b.data if hasattr(img_b, "data") else img_b
    if not isinstance(b, np.ndarray):
        return a
    b = b.astype(np.uint8)

    if a.shape[:2] != b.shape[:2]:
        b = cv2.resize(b, (a.shape[1], a.shape[0]))

    if a.ndim == 2 and b.ndim == 3:
        b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    elif a.ndim == 3 and b.ndim == 2:
        b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
    elif a.ndim == 3 and b.ndim == 3 and a.shape[2] != b.shape[2]:
        if a.shape[2] > b.shape[2]:
            b = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR) if b.shape[2] == 1 else b[:, :, :3]
        else:
            a = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR) if a.shape[2] == 1 else a[:, :, :3]

    if op_type == "add":
        return np.clip(a.astype(np.float64) + b.astype(np.float64), 0, 255).astype(np.uint8)

    if op_type == "subtract":
        reverse = params.get("reverse", False)
        if reverse:
            return np.clip(b.astype(np.float64) - a.astype(np.float64), 0, 255).astype(np.uint8)
        return np.clip(a.astype(np.float64) - b.astype(np.float64), 0, 255).astype(np.uint8)

    if op_type == "abs_diff":
        return np.clip(np.abs(a.astype(np.float64) - b.astype(np.float64)), 0, 255).astype(np.uint8)

    if op_type == "multiply":
        return np.clip(a.astype(np.float64) * b.astype(np.float64) / 255.0, 0, 255).astype(np.uint8)

    if op_type == "divide":
        scale = params.get("scale", 1.0)
        reverse = params.get("reverse", False)
        epsilon = 1e-6
        a_f = a.astype(np.float64)
        b_f = b.astype(np.float64)
        if reverse:
            d = np.divide(b_f, a_f + epsilon)
        else:
            d = np.divide(a_f, b_f + epsilon)
        return np.clip(d * scale, 0, 255).astype(np.uint8)

    if op_type == "max":
        return np.maximum(a, b)

    if op_type == "min":
        return np.minimum(a, b)

    if op_type == "weighted_add":
        alpha = params.get("alpha", 0.5)
        beta = params.get("beta", 0.5)
        gamma = params.get("gamma", 0.0)
        result = a.astype(np.float64) * alpha + b.astype(np.float64) * beta + gamma
        return np.clip(result, 0, 255).astype(np.uint8)

    if op_type == "bitwise_and":
        return cv2.bitwise_and(a, b)

    if op_type == "bitwise_or":
        return cv2.bitwise_or(a, b)

    if op_type == "bitwise_xor":
        return cv2.bitwise_xor(a, b)

    return a


@register_tool(prefix="[CALC]", default_name="Image Calculator", group="Preprocess")
class ImageCalculatorTool(BaseTool):
    def __init__(self, name="ImageCalc"):
        super().__init__(name)
        self.inputs = {"image_A": None, "image_B": None}
        self.outputs = {"output image": None}
        self.params = {
            "ops": [],
            "roi": [0, 0, 0, 0, 0],
            "output_mode": "Full Image",
            "show_overlays": True,
        }

    def run(self, advance=False):
        img_a = self.inputs.get("image_A")
        img_b = self.inputs.get("image_B")

        if img_a is not None and isinstance(img_a, VisionImage):
            v_img = img_a
            raw_a = v_img.data
        elif isinstance(img_a, np.ndarray):
            v_img = VisionImage.from_array(img_a)
            raw_a = img_a
        else:
            self.outputs["output image"] = None
            self.data_changed.emit()
            return

        if img_b is not None and isinstance(img_b, VisionImage):
            raw_b = img_b.data
        elif isinstance(img_b, np.ndarray):
            raw_b = img_b
        else:
            raw_b = None

        roi_p = self.params["roi"]
        x, y, w, h = roi_p[0], roi_p[1], roi_p[2], roi_p[3]
        a = roi_p[4] if len(roi_p) > 4 else 0

        roi = extract_roi(v_img, self.params["roi"])
        working = roi["roi_src"]

        if w <= 0 or h <= 0:
            h_img, w_img = raw_a.shape[:2]
            x, y, w, h = 0, 0, w_img, h_img
            a = 0
            working = raw_a.copy()

        # If raw_b exists, resize it to match working (ROI) size
        if raw_b is not None:
            if raw_b.shape[:2] != working.shape[:2]:
                raw_b = cv2.resize(raw_b, (working.shape[1], working.shape[0]))

        for op in self.params.get("ops", []):
            if not op.get("enabled", True):
                continue
            op_type = op.get("type")
            op_params = {k: v for k, v in op.items() if k not in ("enabled", "type")}
            result = _apply_op(working, raw_b, op_type, op_params)
            if result is not None:
                working = result

        vision_img = VisionImage(v_img.data, x=v_img.x, y=v_img.y, angle=v_img.angle)
        vision_img.inherit_overlays_from(v_img)
        layer = vision_img.get_overlay_layer("default")
        layer.source_tool = self.name
        layer.elements = [e for e in layer.elements if e.source_tool != self.name]

        if self.params.get("output_mode") == "ROI Only":
            out_vimg = create_roi_only_output(
                roi, working, vision_img, self.name, roi["ax"], roi["ay"],
            )
            propagate_3d(v_img, out_vimg, regenerate=True)
            self.outputs["output image"] = out_vimg
        else:
            raw_img = raw_a
            is_full_img = (w >= raw_img.shape[1] and h >= raw_img.shape[0])

            if is_full_img:
                out_vimg = VisionImage(working, v_img.x, v_img.y, v_img.angle,
                                       v_img.scale_x, v_img.scale_y, v_img.scale_z, v_img.offset_z)
                out_vimg.inherit_overlays_from(vision_img)
                propagate_3d(v_img, out_vimg, regenerate=True)
                self.outputs["output image"] = out_vimg
            else:
                show_roi = self.params.get("show_overlays", True)
                out_vimg = composite_roi_to_full(working, vision_img, self.params["roi"], draw_roi=show_roi, source_tool=self.name)
                propagate_3d(v_img, out_vimg, regenerate=True)
                self.outputs["output image"] = out_vimg

        self._copy_overlays_to_input(vision_img, v_img)
        self.data_changed.emit()

    def get_config_widget(self):
        from .config import ImageCalculatorConfig
        return ImageCalculatorConfig(self)

    def serialize(self, job_manager=None):
        data = super().serialize(job_manager)
        data["type"] = "ImageCalculatorTool"
        return data
