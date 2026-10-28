# Logic module for OCR.

"""OCR tool using PP-OCRv5 (ONNX Runtime) — pure recognition."""

import numpy as np
import cv2


from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi
from falcocv.core.pointcloud_utils import propagate_3d
from falcocv.core.ocr.industrial_ocr_engine import (
    PPOCRv5Engine, OCRResult, preload_engine,
)


@register_tool(prefix="[IOCR]", default_name="OCR", group="Vision Tools")
class OCR(BaseTool):
    def __init__(self, name="OCR"):
        super().__init__(name)
        self.params = {
            "mode": "Read",
            "model_type": "server",
            "expected_text": "",
            "roi": [0, 0, 0, 0, 0],
            "show_overlays": True,
            "show_search_roi": True,
            "show_regions": True,
            "show_labels": True,
            "show_found_origin": True,
            "copy_overlays_to_input": True,
            "min_region_confidence": 0.5,
            "split_lines": False,
        }
        self.inputs = {"input image": None}
        self.outputs = {
            "output image": None,
            "text": "",
            "confidence": 0.0,
            "count": 0,
        }
        self._engine: PPOCRv5Engine | None = None

    def _get_engine(self) -> PPOCRv5Engine:
        mt = self.params.get("model_type", "server")
        if self._engine is None or self._engine.model_type != mt:
            self._engine = preload_engine(mt)
        self._engine.config.min_region_confidence = self.params.get("min_region_confidence", 0.5)
        self._engine.config.split_lines = self.params.get("split_lines", False)
        return self._engine

    def run(self, advance=False):
        img_input = self.inputs.get("input image")
        if img_input is None:
            self.outputs["output image"] = None
            self.outputs["text"] = ""
            self.outputs["confidence"] = 0.0
            self.outputs["count"] = 0
            self.data_changed.emit()
            return

        img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)

        roi = extract_roi(img, self.params["roi"])
        roi_src = roi["roi_src"]
        is_full_img = roi["is_full_img"]
        ax, ay, aw, ah = roi["ax"], roi["ay"], roi["aw"], roi["ah"]
        M_inv = roi.get("M_inv")
        has_affine = M_inv is not None

        def _to_image(x, y):
            if has_affine:
                return int(M_inv[0, 0] * x + M_inv[0, 1] * y + M_inv[0, 2]), \
                       int(M_inv[1, 0] * x + M_inv[1, 1] * y + M_inv[1, 2])
            return int(x + ax), int(y + ay)

        engine = self._get_engine()
        mode = self.params.get("mode", "Read")

        result: OCRResult | None = None
        try:
            if mode == "Read":
                result = engine.read(roi_src)
            elif mode == "Verify":
                expected = self.params.get("expected_text", "")
                result = engine.verify(roi_src, expected)
            elif mode == "Locate":
                result = engine.locate(roi_src)
        except Exception:
            import traceback
            traceback.print_exc()

        if result is not None:
            self.outputs["text"] = result.text
            self.outputs["confidence"] = result.confidence
            self.outputs["count"] = len(result.regions)
        else:
            self.outputs["text"] = ""
            self.outputs["confidence"] = 0.0
            self.outputs["count"] = 0

        out_img = VisionImage(img.data, x=img.x, y=img.y, angle=img.angle)
        out_img.inherit_overlays_from(img)
        propagate_3d(img, out_img, regenerate=False)
        layer = out_img.get_overlay_layer("default", source_tool=self.name)
        layer.elements = [e for e in layer.elements if e.source_tool != self.name]

        if self.params.get("show_search_roi", True) and not is_full_img:
            rx, ry = _to_image(0, 0)
            abs_angle = roi.get("abs_angle", 0.0)
            layer.add_roi(
                [float(rx), float(ry), float(aw), float(ah), float(abs_angle)],
                color=(255, 128, 0), thickness=2,
            )

        if result is not None and result.regions:
            for i, region in enumerate(result.regions):
                if self.params.get("show_regions", True):
                    pts = region.get("points", [])
                    if pts and has_affine:
                        pts_arr = np.array(pts, dtype=np.float64)
                        pts_img = (M_inv[:, :2] @ pts_arr.T + M_inv[:, 2:]).T.astype(int).tolist()
                    elif pts:
                        pts_img = [(int(x + ax), int(y + ay)) for x, y in pts]
                    else:
                        pts_img = []
                    if pts_img:
                        layer.add_contour(pts_img, color=(0, 255, 0), thickness=2)

                if self.params.get("show_found_origin", True):
                    cx, cy = region.get("centroid", (0, 0))
                    cx_abs, cy_abs = _to_image(cx, cy)
                    layer.add_point((cx_abs, cy_abs), color=(0, 0, 255), radius=4)

                if self.params.get("show_labels", True):
                    label = f"#{i} {region['text']} ({region['confidence']:.2f})"
                    bx, by = region["bbox"][:2]
                    bx_abs, by_abs = _to_image(bx, by)
                    layer.add_text(label, position=(bx_abs, by_abs - 5),
                                   color=(255, 255, 0), font_scale=0.5, thickness=1)

                if self.params.get("show_labels", True) and mode == "Locate":
                    for ch in region.get("char_boxes", []):
                        cx, cy, cw, ch_h = ch["bbox"]
                        cx_abs, cy_abs = _to_image(cx, cy)
                        layer.add_text(
                            ch["char"], position=(cx_abs, cy_abs - 2),
                            color=(0, 255, 255), font_scale=0.3, thickness=1,
                        )

        if mode == "Verify" and result is not None:
            label = f"Verify: {result.confidence:.2f}"
            color = (0, 255, 0) if result.confidence > 0.8 else (0, 0, 255)
            layer.add_text(label, position=(10, 30), color=color, font_scale=0.8, thickness=2)

        self.outputs["output image"] = out_img
        self._copy_overlays_to_input(out_img, img)
        self.data_changed.emit()

    def get_config_widget(self):
        from .config import OCRConfigWidget
        return OCRConfigWidget(self)
