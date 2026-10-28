# Logic module for FixtureOverlay.

import math
import numpy as np
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.overlay_layer import OverlayType
from falcocv.core.pointcloud_utils import propagate_3d
from falcocv.core.config import ConfigManager


def _make_label(elem, idx):
    src = elem.source_tool or "?"
    rid = elem.ref_id
    tname = elem.element_type.name
    if rid:
        return f"{src} / {rid} ({tname})"
    return f"{src} #{idx} ({tname})"


def _get_elem_key(elem, idx):
    return (elem.source_tool, elem.ref_id, elem.element_type.name, idx)


def _enumerate_elements(v_img):
    items = []
    if not isinstance(v_img, VisionImage):
        return items
    for layer_name, layer in v_img.overlay_layers.items():
        for idx, elem in enumerate(layer.elements):
            items.append({
                "key": _get_elem_key(elem, idx),
                "label": _make_label(elem, idx),
                "source_tool": elem.source_tool,
                "ref_id": elem.ref_id,
                "type": elem.element_type.name,
                "elem_idx": idx,
                "layer": layer_name,
                "elem": elem,
            })
    return items


def _compute_fixture(elem):
    t = elem.element_type
    d = elem.data
    if t == OverlayType.LINE:
        sx, sy = d["start"]
        ex, ey = d["end"]
        cx = (sx + ex) / 2.0
        cy = (sy + ey) / 2.0
        angle = math.degrees(math.atan2(ey - sy, ex - sx))
    elif t == OverlayType.POINT:
        cx, cy = d["center"]
        angle = 0.0
    elif t == OverlayType.CIRCLE:
        cx, cy = d["center"]
        angle = 0.0
    elif t == OverlayType.ROI:
        roi = d["roi"]
        cx = roi[0] + roi[2] / 2.0
        cy = roi[1] + roi[3] / 2.0
        angle = float(roi[4]) if len(roi) > 4 else 0.0
    elif t in (OverlayType.CONTOUR, OverlayType.POLYGON):
        raw = d.get("contour") or d.get("points", [])
        if raw:
            arr = np.array(raw, dtype=np.float64).reshape(-1, 2)
            cx = float(arr[:, 0].mean())
            cy = float(arr[:, 1].mean())
        else:
            cx, cy = 0.0, 0.0
        angle = 0.0
    elif t == OverlayType.TEXT:
        cx, cy = d["position"]
        angle = 0.0
    elif t == OverlayType.AXES:
        cx, cy = d["center"]
        angle = float(d.get("angle", 0.0))
    else:
        cx, cy = 0.0, 0.0
        angle = 0.0
    return (cx, cy, angle)


@register_tool(prefix="[FIXT]", default_name="Fixture Overlay", group="Vision Tools")
class FixtureOverlay(BaseTool):
    def __init__(self, name="Fixture Overlay"):
        super().__init__(name)
        self.inputs = {"input image": None}
        self.outputs = {"output image": None}
        self.params = {
            "selected_source_tool": None,
            "selected_ref_id": None,
            "selected_type": None,
        }

    def run(self, advance=False):
        self.outputs["output image"] = None

        img = self.inputs.get("input image")
        if img is None:
            self.data_changed.emit()
            return

        v_img = img if isinstance(img, VisionImage) else VisionImage.from_array(img)

        out = VisionImage(v_img.data, x=v_img.x, y=v_img.y, angle=v_img.angle,
                          scale_x=v_img.scale_x, scale_y=v_img.scale_y)
        out.inherit_overlays_from(v_img)
        propagate_3d(v_img, out, regenerate=False)

        st = self.params.get("selected_source_tool")
        rid = self.params.get("selected_ref_id")
        tp = self.params.get("selected_type")
        if st is not None and rid is not None:
            items = _enumerate_elements(v_img)
            found = next((it for it in items
                          if it["source_tool"] == st
                          and it["ref_id"] == rid
                          and it["type"] == tp), None)
            if found:
                cx, cy, angle = _compute_fixture(found["elem"])
                out.x = cx
                out.y = cy
                out.angle = angle

        self.outputs["output image"] = out
        self.data_changed.emit()

    def get_config_widget(self):
        from .config import FixtureOverlayConfig
        return FixtureOverlayConfig(self)
