import cv2
import numpy as np
import math
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_base import ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.overlay_layer import OverlayType
from falcocv.core.pointcloud_utils import propagate_3d


MEASURABLE_TYPES = {
    OverlayType.POINT, OverlayType.LINE, OverlayType.CIRCLE,
    OverlayType.CONTOUR, OverlayType.AXES, OverlayType.ROI,
    OverlayType.POLYGON,
}

STRATEGIES = {
    "POINT": ["center"],
    "AXES": ["center"],
    "LINE": ["center", "closest", "farthest"],
    "CIRCLE": ["center", "closest"],
    "ROI": ["center", "perimeter"],
    "CONTOUR": ["centroid", "closest"],
    "POLYGON": ["centroid", "closest"],
}

COLOR_PRESETS = {
    "Yellow": [0, 255, 255],
    "Green": [0, 255, 0],
    "Red": [0, 0, 255],
    "Cyan": [255, 255, 0],
    "Magenta": [255, 0, 255],
    "Orange": [0, 128, 255],
    "White": [255, 255, 255],
}

COLOR_NAMES = list(COLOR_PRESETS.keys())

MODE_NAMES = ["distance", "angle", "area", "perimeter", "radius", "diameter", "width", "height"]

SINGLE_ELEM_MODES = {"area", "perimeter", "radius", "diameter", "width", "height"}

TWO_ELEM_MODES = {"distance", "angle"}

MODE_VALID_TYPES = {
    "distance": {"POINT", "AXES", "LINE", "CIRCLE", "ROI", "CONTOUR", "POLYGON"},
    "angle": {"LINE"},
    "area": {"ROI", "CONTOUR", "POLYGON", "CIRCLE"},
    "perimeter": {"ROI", "CONTOUR", "POLYGON", "CIRCLE"},
    "radius": {"CIRCLE"},
    "diameter": {"CIRCLE"},
    "width": {"ROI"},
    "height": {"ROI"},
}

MODE_UNIT = {
    "distance": "mm",
    "angle": "deg",
    "area": "mm2",
    "perimeter": "mm",
    "radius": "mm",
    "diameter": "mm",
    "width": "mm",
    "height": "mm",
}

DEFAULT_ROW = {
    "enabled": True,
    "mode": "distance",
    "source1": "", "ref_id1": "", "type1": "POINT", "strategy1": "center",
    "source2": "", "ref_id2": "", "type2": "POINT", "strategy2": "center",
    "color": [0, 255, 255], "thickness": 2,
}


def _make_label(elem, idx):
    src = elem.source_tool or "?"
    rid = elem.ref_id
    tname = elem.element_type.name
    if rid:
        return f"{src} / {rid} ({tname})"
    return f"{src} #{idx} ({tname})"


def _get_elem_key(elem, idx):
    return (elem.source_tool, elem.ref_id, elem.element_type.name, idx)


def _closest_on_segment(s, e, p):
    sx, sy = s; ex, ey = e; px, py = p
    dx, dy = ex - sx, ey - sy
    if dx == 0 and dy == 0:
        return s
    t = ((px - sx) * dx + (py - sy) * dy) / (dx * dx + dy * dy)
    t = max(0, min(1, t))
    return (sx + t * dx, sy + t * dy)


def _reference_point(elem, strategy, other_pt=None):
    t = elem.element_type
    d = elem.data

    if t in (OverlayType.POINT, OverlayType.AXES):
        return tuple(map(float, d["center"]))

    elif t == OverlayType.LINE:
        s = tuple(map(float, d["start"]))
        e = tuple(map(float, d["end"]))
        if strategy == "center":
            return ((s[0] + e[0]) / 2, (s[1] + e[1]) / 2)
        elif strategy == "closest" and other_pt:
            return _closest_on_segment(s, e, other_pt)
        elif strategy == "farthest" and other_pt:
            dx1 = other_pt[0] - s[0]; dy1 = other_pt[1] - s[1]
            dx2 = other_pt[0] - e[0]; dy2 = other_pt[1] - e[1]
            return e if (dx1*dx1 + dy1*dy1) >= (dx2*dx2 + dy2*dy2) else s
        return s

    elif t == OverlayType.CIRCLE:
        cx, cy = map(float, d["center"])
        radius = float(d["radius"])
        if strategy == "center":
            return (cx, cy)
        elif strategy == "closest" and other_pt:
            dx = other_pt[0] - cx; dy = other_pt[1] - cy
            dist = math.hypot(dx, dy)
            if dist < 1e-9:
                return (cx + radius, cy)
            return (cx + dx / dist * radius, cy + dy / dist * radius)
        return (cx, cy)

    elif t == OverlayType.ROI:
        roi = d["roi"]
        rcx = float(roi[0]) + float(roi[2]) / 2
        rcy = float(roi[1]) + float(roi[3]) / 2
        rw = float(roi[2])
        rh = float(roi[3])
        if strategy == "center":
            return (rcx, rcy)
        elif strategy in ("closest", "perimeter") and other_pt:
            l = rcx - rw / 2; r = rcx + rw / 2
            t_ = rcy - rh / 2; b_ = rcy + rh / 2
            px = max(l, min(r, other_pt[0]))
            py = max(t_, min(b_, other_pt[1]))
            if px == l or px == r or py == t_ or py == b_:
                return (px, py)
            dl = px - l; dr = r - px; dt = py - t_; db = b_ - py
            min_d = min(dl, dr, dt, db)
            if min_d == dl:  return (l, py)
            if min_d == dr:  return (r, py)
            if min_d == dt:  return (px, t_)
            return (px, b_)
        return (rcx, rcy)

    elif t in (OverlayType.CONTOUR, OverlayType.POLYGON):
        raw = d.get("contour") or d.get("points", [])
        if not raw:
            return (0, 0)
        pts = np.array(raw, dtype=np.int32).reshape(-1, 2)
        if len(pts) < 3:
            return tuple(pts.mean(axis=0).astype(np.float64))
        if strategy in ("centroid", "center"):
            M = cv2.moments(pts.reshape(-1, 1, 2))
            if M["m00"] != 0:
                return (M["m10"] / M["m00"], M["m01"] / M["m00"])
            return tuple(pts.mean(axis=0).astype(np.float64))
        elif strategy == "closest" and other_pt:
            idx = np.argmin(np.sum((pts - np.array(other_pt, dtype=np.float64)) ** 2, axis=1))
            return tuple(pts[idx].astype(np.float64))
        return tuple(pts.mean(axis=0).astype(np.float64))
    return (0, 0)


def _color_name_from_bgr(bgr):
    for name, rgb in COLOR_PRESETS.items():
        if list(bgr) == rgb:
            return name
    return "Yellow"


def _enumerate_elements(v_img):
    items = []
    if not isinstance(v_img, VisionImage):
        return items
    for layer_name, layer in v_img.overlay_layers.items():
        for idx, elem in enumerate(layer.elements):
            if elem.element_type not in MEASURABLE_TYPES:
                continue
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


def _find_element(items, source, ref_id, type_name, fallback_idx):
    if ref_id:
        for it in items:
            if it["source_tool"] == source and it["ref_id"] == ref_id and it["type"] == type_name:
                return it
        return None
    idx = 0
    for it in items:
        if it["source_tool"] == source and it["type"] == type_name:
            if idx == fallback_idx:
                return it
            idx += 1
    return None


class MeasurementResult(ToolResult):
    def __init__(self, id, type, value, unit, x1, y1, x2, y2, angle, label):
        super().__init__(id=id, type=type, value=value, unit=unit,
                         x1=x1, y1=y1, x2=x2, y2=y2, angle=angle, label=label)

    def items(self):
        common = [("id", self.id), ("type", self.type), ("value", self.value), ("unit", self.unit), ("label", self.label)]
        if self.type == "distance":
            for item in common + [("x1", self.x1), ("y1", self.y1), ("x2", self.x2), ("y2", self.y2), ("angle", self.angle)]:
                yield item
        else:
            for item in common:
                yield item


@register_tool(prefix="[MEAS]", default_name="Measurement", group="Vision Tools")
class MeasurementTool(BaseTool):
    def __init__(self, name="Measurement"):
        super().__init__(name)
        self.params = {
            "pixels_per_mm": 1.0,
            "show_measurement_lines": True,
            "show_labels": True,
            "overlay_measurements": [],
            "copy_overlays_to_input": True,
        }
        self.inputs = {"input image": None}
        self.outputs = {
            "output image": None,
            "count": 0,
            "results": ToolResults(),
        }

    # ── Single-element measurement helpers ────────────────────────────

    def _measure_area(self, elem, px_per_mm):
        t = elem.element_type
        d = elem.data
        if t == OverlayType.ROI:
            w = float(d["roi"][2]); h = float(d["roi"][3])
            area_px = w * h
        elif t == OverlayType.CIRCLE:
            r = float(d["radius"])
            area_px = math.pi * r * r
        elif t in (OverlayType.CONTOUR, OverlayType.POLYGON):
            raw = d.get("contour") or d.get("points", [])
            if not raw: return 0.0
            pts = np.array(raw, dtype=np.int32).reshape(-1, 1, 2)
            area_px = abs(cv2.contourArea(pts))
        else:
            return 0.0
        return area_px / (px_per_mm * px_per_mm) if px_per_mm > 0 else area_px

    def _measure_perimeter(self, elem, px_per_mm):
        t = elem.element_type
        d = elem.data
        if t == OverlayType.ROI:
            w = float(d["roi"][2]); h = float(d["roi"][3])
            perim_px = 2 * (w + h)
        elif t == OverlayType.CIRCLE:
            r = float(d["radius"])
            perim_px = 2 * math.pi * r
        elif t in (OverlayType.CONTOUR, OverlayType.POLYGON):
            raw = d.get("contour") or d.get("points", [])
            if not raw: return 0.0
            pts = np.array(raw, dtype=np.int32).reshape(-1, 1, 2)
            perim_px = cv2.arcLength(pts, True)
        else:
            return 0.0
        return perim_px / px_per_mm if px_per_mm > 0 else perim_px

    def _measure_radius(self, elem, px_per_mm):
        if elem.element_type != OverlayType.CIRCLE:
            return 0.0
        r = float(elem.data["radius"])
        return r / px_per_mm if px_per_mm > 0 else r

    def _measure_diameter(self, elem, px_per_mm):
        t = elem.element_type
        d = elem.data
        if t == OverlayType.CIRCLE:
            diam = 2 * float(d["radius"])
        elif t == OverlayType.ROI:
            diam = min(float(d["roi"][2]), float(d["roi"][3]))
        else:
            return 0.0
        return diam / px_per_mm if px_per_mm > 0 else diam

    def _measure_width(self, elem, px_per_mm):
        if elem.element_type != OverlayType.ROI:
            return 0.0
        w = float(elem.data["roi"][2])
        return w / px_per_mm if px_per_mm > 0 else w

    def _measure_height(self, elem, px_per_mm):
        if elem.element_type != OverlayType.ROI:
            return 0.0
        h = float(elem.data["roi"][3])
        return h / px_per_mm if px_per_mm > 0 else h

    # ── Angle between two lines ───────────────────────────────────────

    def _angle_between_lines(self, e1, e2):
        s1, e1p = e1.data["start"], e1.data["end"]
        s2, e2p = e2.data["start"], e2.data["end"]
        dx1 = e1p[0] - s1[0]; dy1 = e1p[1] - s1[1]
        dx2 = e2p[0] - s2[0]; dy2 = e2p[1] - s2[1]
        dot = dx1 * dx2 + dy1 * dy2
        n1 = math.hypot(dx1, dy1); n2 = math.hypot(dx2, dy2)
        if n1 < 1e-9 or n2 < 1e-9:
            return 0.0
        cos_a = max(-1, min(1, dot / (n1 * n2)))
        angle = math.degrees(math.acos(cos_a))
        # Return acute angle (0-90) for undirected lines
        return min(angle, 180.0 - angle)

    def run(self, advance=False):
        self.outputs["output image"] = None
        self.outputs["count"] = 0
        self.outputs["results"] = ToolResults()

        img_input = self.inputs.get("input image")
        if img_input is None:
            self.data_changed.emit()
            return

        try:
            v_img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)
            vision_img = VisionImage(v_img.data, x=v_img.x, y=v_img.y, angle=v_img.angle,
                                      scale_x=v_img.scale_x, scale_y=v_img.scale_y,
                                      scale_z=v_img.scale_z, offset_z=v_img.offset_z)
            vision_img.inherit_overlays_from(v_img)
            propagate_3d(v_img, vision_img, regenerate=False)
            vision_img.get_overlay_layer("default", source_tool=self.name)

            measurements = self.params.get("overlay_measurements", [])
            px_per_mm = float(self.params.get("pixels_per_mm", 1.0) or 1.0)
            results = ToolResults()
            layer = vision_img.get_overlay_layer("default")
            layer.elements = [e for e in layer.elements if e.source_tool != self.name]

            for row_idx, row in enumerate(measurements):
                if not row.get("enabled", True):
                    continue

                mode = row.get("mode", "distance")
                items = _enumerate_elements(vision_img)
                e1 = _find_element(items, row.get("source1", ""), row.get("ref_id1", ""),
                                   row.get("type1", "POINT"), row_idx)
                if e1 is None:
                    continue

                color = tuple(row.get("color", [0, 255, 255]))
                thickness = row.get("thickness", 2)
                result_value = 0.0
                result_angle = 0.0
                result_unit = "mm"
                result_type = mode
                p1 = (0, 0); p2 = (0, 0)

                if mode in SINGLE_ELEM_MODES:
                    elem = e1["elem"]
                    if mode == "area":
                        result_value = self._measure_area(elem, px_per_mm)
                        result_unit = "mm2"
                    elif mode == "perimeter":
                        result_value = self._measure_perimeter(elem, px_per_mm)
                    elif mode == "radius":
                        result_value = self._measure_radius(elem, px_per_mm)
                    elif mode == "diameter":
                        result_value = self._measure_diameter(elem, px_per_mm)
                    elif mode == "width":
                        result_value = self._measure_width(elem, px_per_mm)
                    elif mode == "height":
                        result_value = self._measure_height(elem, px_per_mm)

                    if self.params["show_measurement_lines"]:
                        if elem.element_type in (OverlayType.CONTOUR, OverlayType.POLYGON):
                            pts = np.array(elem.data.get("contour") or elem.data.get("points", []), dtype=np.int32).reshape(-1, 2)
                            for i in range(len(pts)-1):
                                layer.add_line(tuple(pts[i]), tuple(pts[i+1]), color=color, thickness=1)
                            layer.add_line(tuple(pts[-1]), tuple(pts[0]), color=color, thickness=1)
                        elif elem.element_type == OverlayType.CIRCLE:
                            cx, cy = map(float, elem.data["center"])
                            r = float(elem.data["radius"])
                            layer.add_circle((cx, cy), r, color=color, thickness=thickness)

                else:
                    e2 = _find_element(items, row.get("source2", ""), row.get("ref_id2", ""),
                                       row.get("type2", "POINT"), row_idx)
                    if e2 is None:
                        continue

                    if mode == "angle":
                        e1p = _reference_point(e1["elem"], "center")
                        e2p = _reference_point(e2["elem"], "center")
                        p1, p2 = e1p, e2p
                        result_value = self._angle_between_lines(e1["elem"], e2["elem"])
                        result_unit = MODE_UNIT["angle"]
                    else:
                        strat1 = row.get("strategy1", "center")
                        strat2 = row.get("strategy2", "center")
                        needs_other = ("closest", "farthest", "perimeter")
                        p1_init = _reference_point(e1["elem"],
                            "center" if strat1 in needs_other else strat1)
                        p2_init = _reference_point(e2["elem"],
                            "center" if strat2 in needs_other else strat2)
                        p1 = _reference_point(e1["elem"], strat1, p2_init) if strat1 in needs_other else p1_init
                        p2 = _reference_point(e2["elem"], strat2, p1) if strat2 in needs_other else p2_init
                        dist_px = math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)
                        dist_mm = dist_px / px_per_mm if px_per_mm > 0 else dist_px
                        result_value = dist_mm
                        result_unit = MODE_UNIT["distance"]
                        result_angle = math.degrees(math.atan2(-(p2[1] - p1[1]), p2[0] - p1[0]))

                    if self.params["show_measurement_lines"] and mode == "distance":
                        layer.add_line(p1, p2, color=color, thickness=thickness, ref_id=f"line_{row_idx}")
                    elif self.params["show_measurement_lines"] and mode == "angle":
                        s1, e_ = e1["elem"].data["start"], e1["elem"].data["end"]
                        s2, e2_ = e2["elem"].data["start"], e2["elem"].data["end"]
                        layer.add_line(tuple(s1), tuple(e_), color=color, thickness=max(1, thickness-1))
                        layer.add_line(tuple(s2), tuple(e2_), color=color, thickness=max(1, thickness-1))

                if self.params["show_labels"]:
                    if mode in SINGLE_ELEM_MODES:
                        cx, cy = _reference_point(e1["elem"], "center")
                        mx, my = cx, cy - 12
                    elif mode == "angle":
                        mx = (p1[0] + p2[0]) / 2; my = (p1[1] + p2[1]) / 2 - 12
                    else:
                        mx = (p1[0] + p2[0]) / 2; my = (p1[1] + p2[1]) / 2 - 12
                    label_txt = f"{result_value:.2f}{result_unit}"
                    layer.add_text(label_txt, (mx, my), color=color, font_scale=0.5, thickness=thickness)

                results.append(MeasurementResult(
                    id=len(results), type=result_type, value=result_value, unit=result_unit,
                    x1=float(p1[0]), y1=float(p1[1]),
                    x2=float(p2[0]), y2=float(p2[1]),
                    angle=result_angle, label=f"M{row_idx}"))

            self.outputs["output image"] = vision_img
            self.outputs["count"] = len(results)
            self.outputs["results"] = results
            self._copy_overlays_to_input(vision_img, v_img)
        except Exception:
            import traceback
            traceback.print_exc()
        finally:
            self.data_changed.emit()

    def get_config_widget(self):
        from .config import MeasurementConfig
        return MeasurementConfig(self)


