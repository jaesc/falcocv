import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any
from enum import Enum


class OverlayType(Enum):
    ROI = "roi"
    CONTOUR = "contour"
    CIRCLE = "circle"
    TEXT = "text"
    LINE = "line"
    AXES = "axes"
    POINT = "point"
    POLYGON = "polygon"
    WORLD_POLYGON = "world_polygon"
    WORLD_CONTOUR = "world_contour"
    WORLD_LINE = "world_line"
    WORLD_POINTS = "world_points"
    WORLD_ROI = "world_roi"
    WORLD_BBOX = "world_bbox"


@dataclass
class OverlayElement:
    element_type: OverlayType
    data: dict
    visible: bool = True
    priority: int = 0
    style: dict = field(default_factory=dict)
    source_tool: str = ""
    ref_id: str = ""

    def to_dict(self) -> dict:
        return {
            "element_type": self.element_type.value,
            "data": self.data,
            "visible": self.visible,
            "priority": self.priority,
            "style": self.style,
            "source_tool": self.source_tool,
            "ref_id": self.ref_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OverlayElement":
        return cls(
            element_type=OverlayType(d["element_type"]),
            data=d["data"],
            visible=d.get("visible", True),
            priority=d.get("priority", 0),
            style=d.get("style", {}),
            source_tool=d.get("source_tool", ""),
            ref_id=d.get("ref_id", ""),
        )

    def copy(self) -> "OverlayElement":
        """Create a copy of this overlay element."""
        return OverlayElement(
            element_type=self.element_type,
            data=self.data.copy(),
            visible=self.visible,
            priority=self.priority,
            style=self.style.copy(),
            source_tool=self.source_tool,
            ref_id=self.ref_id,
        )

    def transform(self, x: float, y: float, angle: float, scale_x: float = 1.0, scale_y: float = 1.0) -> "OverlayElement":
        """Create a new element with coordinates transformed by VisionImage corrections.
        
        Transforms overlay coordinates from the source coordinate system to the 
        current VisionImage's transformed coordinate system.
        """
        new_data = self.data.copy()
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        
        if self.element_type == OverlayType.ROI:
            roi = list(new_data.get("roi", [0, 0, 0, 0, 0]))
            cx = roi[0] + roi[2] / 2
            cy = roi[1] + roi[3] / 2
            rel_cx = cx * cos_a - cy * sin_a + x
            rel_cy = cx * sin_a + cy * cos_a + y
            new_w = roi[2] * scale_x
            new_h = roi[3] * scale_y
            new_angle = roi[4] + angle if len(roi) > 4 else angle
            new_data["roi"] = [rel_cx - new_w/2, rel_cy - new_h/2, new_w, new_h, new_angle]
            
        elif self.element_type == OverlayType.CONTOUR:
            pts = np.array(new_data.get("contour", []), dtype=np.float32).reshape(-1, 2)
            if len(pts) > 0:
                rel_pts = np.dot(pts, np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T) + [x, y]
                rel_pts[:, 0] *= scale_x
                rel_pts[:, 1] *= scale_y
                new_data["contour"] = rel_pts.reshape(-1, 2).tolist()
                
        elif self.element_type == OverlayType.CIRCLE:
            cx, cy = new_data.get("center", (0, 0))
            new_cx = x + (cx * cos_a - cy * sin_a) * scale_x
            new_cy = y + (cx * sin_a + cy * cos_a) * scale_y
            new_data["center"] = (new_cx, new_cy)
            new_data["radius"] = new_data.get("radius", 1) * scale_x
            
        elif self.element_type == OverlayType.TEXT:
            tx, ty = new_data.get("position", (0, 0))
            new_tx = x + (tx * cos_a - ty * sin_a) * scale_x
            new_ty = y + (tx * sin_a + ty * cos_a) * scale_y
            new_data["position"] = (new_tx, new_ty)
            
        elif self.element_type == OverlayType.LINE:
            sx, sy = new_data.get("start", (0, 0))
            ex, ey = new_data.get("end", (0, 0))
            new_sx = x + (sx * cos_a - sy * sin_a) * scale_x
            new_sy = y + (sx * sin_a + sy * cos_a) * scale_y
            new_ex = x + (ex * cos_a - ey * sin_a) * scale_x
            new_ey = y + (ex * sin_a + ey * cos_a) * scale_y
            new_data["start"] = (new_sx, new_sy)
            new_data["end"] = (new_ex, new_ey)
            
        elif self.element_type == OverlayType.AXES:
            cx, cy = new_data.get("center", (0, 0))
            new_cx = x + (cx * cos_a - cy * sin_a) * scale_x
            new_cy = y + (cx * sin_a + cy * cos_a) * scale_y
            new_data["center"] = (new_cx, new_cy)
            new_data["angle"] = new_data.get("angle", 0) + angle
            
        elif self.element_type == OverlayType.POINT:
            px, py = new_data.get("center", (0, 0))
            new_px = x + (px * cos_a - py * sin_a) * scale_x
            new_py = y + (px * sin_a + py * cos_a) * scale_y
            new_data["center"] = (new_px, new_py)
            
        elif self.element_type == OverlayType.POLYGON:
            pts = np.array(new_data.get("points", []), dtype=np.float32).reshape(-1, 2)
            if len(pts) > 0:
                rel_pts = np.dot(pts, np.array([[cos_a, -sin_a], [sin_a, cos_a]]).T) + [x, y]
                rel_pts[:, 0] *= scale_x
                rel_pts[:, 1] *= scale_y
                new_data["points"] = rel_pts.reshape(-1, 2).tolist()
        
        return OverlayElement(
            element_type=self.element_type,
            data=new_data,
            visible=self.visible,
            priority=self.priority,
            style=self.style.copy(),
            source_tool=self.source_tool,
            ref_id=self.ref_id,
        )

    def offset_rotate(self, dx: float, dy: float, angle: float) -> "OverlayElement":
        """Offset by (-dx, -dy) then rotate by angle (for ROI crop mapping).

        Unlike transform() which rotates first then translates, this method
        translates first then rotates:
            P_new = R(angle) * (P_old - (dx, dy))
        """
        rad = np.deg2rad(angle)
        cos_a, sin_a = np.cos(rad), np.sin(rad)
        tx = -(dx * cos_a - dy * sin_a)
        ty = -(dx * sin_a + dy * cos_a)
        return self.transform(tx, ty, angle)

    def affine_transform(self, M: np.ndarray) -> "OverlayElement":
        """Apply a 2x3 affine matrix M to all coordinates.

        M maps from absolute image space to warped (ROI) space.
        Each coordinate (x, y) is transformed as:
            x' = M[0,0]*x + M[0,1]*y + M[0,2]
            y' = M[1,0]*x + M[1,1]*y + M[1,2]
        """
        new_data = self.data.copy()
        m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
        m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]

        def _apply(px, py):
            return (m00 * px + m01 * py + m02, m10 * px + m11 * py + m12)

        if self.element_type == OverlayType.ROI:
            roi = list(new_data.get("roi", [0, 0, 0, 0, 0]))
            x1, y1 = _apply(roi[0], roi[1])
            x2, y2 = _apply(roi[0] + roi[2], roi[1] + roi[3])
            nx = min(x1, x2)
            ny = min(y1, y2)
            nw = abs(x2 - x1)
            nh = abs(y2 - y1)
            new_angle = roi[4] if len(roi) > 4 else 0.0
            new_data["roi"] = [nx, ny, nw, nh, new_angle]

        elif self.element_type == OverlayType.CONTOUR:
            pts = np.array(new_data.get("contour", []), dtype=np.float32).reshape(-1, 2)
            if len(pts) > 0:
                xf = pts[:, 0] * m00 + pts[:, 1] * m01 + m02
                yf = pts[:, 0] * m10 + pts[:, 1] * m11 + m12
                new_data["contour"] = np.stack([xf, yf], axis=1).reshape(-1, 2).tolist()

        elif self.element_type == OverlayType.CIRCLE:
            cx, cy = new_data.get("center", (0, 0))
            nc = _apply(cx, cy)
            new_data["center"] = nc
            new_data["radius"] = new_data.get("radius", 1) * (abs(m00) + abs(m01))

        elif self.element_type == OverlayType.TEXT:
            tx, ty = new_data.get("position", (0, 0))
            new_data["position"] = _apply(tx, ty)

        elif self.element_type == OverlayType.LINE:
            sx, sy = new_data.get("start", (0, 0))
            ex, ey = new_data.get("end", (0, 0))
            new_data["start"] = _apply(sx, sy)
            new_data["end"] = _apply(ex, ey)

        elif self.element_type == OverlayType.AXES:
            cx, cy = new_data.get("center", (0, 0))
            new_data["center"] = _apply(cx, cy)

        elif self.element_type == OverlayType.POINT:
            px, py = new_data.get("center", (0, 0))
            new_data["center"] = _apply(px, py)

        elif self.element_type == OverlayType.POLYGON:
            pts = np.array(new_data.get("points", []), dtype=np.float32).reshape(-1, 2)
            if len(pts) > 0:
                xf = pts[:, 0] * m00 + pts[:, 1] * m01 + m02
                yf = pts[:, 0] * m10 + pts[:, 1] * m11 + m12
                new_data["points"] = np.stack([xf, yf], axis=1).tolist()

        return OverlayElement(
            element_type=self.element_type,
            data=new_data,
            visible=self.visible,
            priority=self.priority,
            style=self.style.copy(),
            source_tool=self.source_tool,
            ref_id=self.ref_id,
        )


class OverlayLayer:
    def __init__(self, name: str, source_tool: str = ""):
        self.name = name
        self.source_tool = source_tool
        self.elements: List[OverlayElement] = []
        self.visible: bool = True
        self.z_order: int = 0

    def add_element(self, element: OverlayElement) -> None:
        self.elements.append(element)

    def add_elements(self, elements) -> None:
        for element in elements:
            self.elements.append(element)
        self.elements.sort(key=lambda e: e.priority)

    def add_roi(self, roi_data: List[float], color: Tuple[int, int, int] = (255, 128, 0),
                thickness: int = 2, visible: bool = True, priority: int = 0, fill: bool = False,
                fill_image: Optional[np.ndarray] = None, fill_alpha: float = 0.6,
                ref_id: str = "") -> None:
        data = {"roi": roi_data, "color": color, "thickness": thickness, "fill": fill}
        if fill_image is not None:
            data["fill_image"] = fill_image
            data["fill_alpha"] = fill_alpha
        self.add_element(OverlayElement(
            element_type=OverlayType.ROI,
            data=data,
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_contour(self, contour: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 2, visible: bool = True, priority: int = 0, fill: bool = False,
                    ref_id: str = "") -> None:
        contour_list = contour.tolist() if hasattr(contour, 'tolist') else list(contour)
        self.add_element(OverlayElement(
            element_type=OverlayType.CONTOUR,
            data={"contour": contour_list, "color": color, "thickness": thickness, "fill": fill},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_circle(self, center: Tuple[float, float], radius: float,
                   color: Tuple[int, int, int] = (0, 255, 0),
                   thickness: int = 2, visible: bool = True, priority: int = 0, fill: bool = False,
                   ref_id: str = "") -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.CIRCLE,
            data={"center": center, "radius": radius, "color": color, "thickness": thickness, "fill": fill},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_text(self, text: str, position: Tuple[float, float],
                 color: Tuple[int, int, int] = (255, 255, 0),
                 font_scale: float = 0.5, thickness: int = 1,
                 visible: bool = True, priority: int = 0) -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.TEXT,
            data={"text": text, "position": position, "color": color,
                  "font_scale": font_scale, "thickness": thickness},
            visible=visible, priority=priority,
            source_tool=self.source_tool
        ))

    def add_line(self, start: Tuple[float, float], end: Tuple[float, float],
               color: Tuple[int, int, int] = (0, 0, 255),
               thickness: int = 2, visible: bool = True, priority: int = 0,
               ref_id: str = "") -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.LINE,
            data={"start": start, "end": end, "color": color, "thickness": thickness},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_axes(self, center: Tuple[float, float], angle: float, length: float = 30,
                 color_x: Tuple[int, int, int] = (0, 0, 255),
                 color_y: Tuple[int, int, int] = (0, 255, 0),
                 thickness: int = 2, visible: bool = True, priority: int = 0,
                 ref_id: str = "") -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.AXES,
            data={"center": center, "angle": angle, "length": length,
                  "color_x": color_x, "color_y": color_y, "thickness": thickness},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_point(self, center: Tuple[float, float], color: Tuple[int, int, int] = (0, 0, 255),
                  radius: int = 4, visible: bool = True, priority: int = 0,
                  ref_id: str = "") -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.POINT,
            data={"center": center, "color": color, "radius": radius},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_polygon(self, points: List[Tuple[float, float]],
                   color: Tuple[int, int, int] = (255, 0, 0),
                   thickness: int = 1, visible: bool = True, priority: int = 0,
                   fill: bool = False, fill_alpha: float = 0.6,
                   ref_id: str = "") -> None:
        pts_list = list(points)
        self.add_element(OverlayElement(
            element_type=OverlayType.POLYGON,
            data={"points": pts_list, "color": color, "thickness": thickness,
                  "fill": fill, "fill_alpha": fill_alpha},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def clear(self) -> None:
        self.elements.clear()

    # ── 3D world-space overlays (rendered by Viewer3D) ──────────────
    def add_world_polygon(self, world_vertices, color=(0, 200, 255),
                          fill_alpha=0.35, thickness=2, visible=True,
                          priority=0, ref_id="") -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.WORLD_POLYGON,
            data={"world_vertices": list(world_vertices), "color": color,
                  "fill_alpha": fill_alpha, "thickness": thickness},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_world_contour(self, world_vertices, color=(0, 255, 100),
                          thickness=2, visible=True, priority=0, ref_id="") -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.WORLD_CONTOUR,
            data={"world_vertices": list(world_vertices), "color": color,
                  "thickness": thickness},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_world_line(self, world_start, world_end, color=(255, 50, 50),
                       thickness=3, visible=True, priority=0, ref_id="") -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.WORLD_LINE,
            data={"world_start": list(world_start), "world_end": list(world_end),
                  "color": color, "thickness": thickness},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_world_points(self, world_points, color=(255, 255, 0),
                         point_size=5.0, visible=True, priority=0, ref_id="") -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.WORLD_POINTS,
            data={"world_points": list(world_points), "color": color,
                  "point_size": point_size},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_world_roi(self, world_center, world_size, color=(255, 128, 0),
                      visible=True, priority=0, ref_id="") -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.WORLD_ROI,
            data={"world_center": list(world_center), "world_size": list(world_size),
                  "color": color},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def add_world_bbox(self, world_center, world_size, world_rotation,
                       color=(255, 128, 0), visible=True, priority=0,
                       ref_id="") -> None:
        self.add_element(OverlayElement(
            element_type=OverlayType.WORLD_BBOX,
            data={"world_center": list(world_center), "world_size": list(world_size),
                  "world_rotation": list(world_rotation), "color": color},
            visible=visible, priority=priority,
            source_tool=self.source_tool, ref_id=ref_id,
        ))

    def get_visible_elements(self) -> List[OverlayElement]:
        return [e for e in self.elements if e.visible]

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "source_tool": self.source_tool,
            "visible": self.visible,
            "z_order": self.z_order,
            "elements": [e.to_dict() for e in self.elements]
        }

    @classmethod
    def from_dict(cls, d: dict) -> "OverlayLayer":
        layer = cls(name=d["name"], source_tool=d.get("source_tool", ""))
        layer.visible = d.get("visible", True)
        layer.z_order = d.get("z_order", 0)
        for elem_dict in d.get("elements", []):
            layer.elements.append(OverlayElement.from_dict(elem_dict))
        return layer