"""
Edge Detection Tool — Affine 1D caliper for industrial subpixel edge detection.

Algorithm: samples a rotated ROI band directly from the source image using
cv2.remap with bilinear interpolation, projects into a 1D intensity profile
by averaging along the longitudinal axis, applies a derivative-of-gaussian,
and detects peaks by polarity. Subpixel refinement via parabolic interpolation.

Reconstruction: P = center + v * edge_position

Inspired by Cognex VisionPro caliper tools.
"""

import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_base import ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.pointcloud_utils import propagate_3d


class EdgePairResult(ToolResult):
    def __init__(self, id, x1, y1, x2, y2, width, score1, score2):
        super().__init__(id=id, x1=x1, y1=y1, x2=x2, y2=y2,
                         width=width, score1=score1, score2=score2)


class SingleEdgeResult(ToolResult):
    def __init__(self, id, x, y, score, polarity):
        super().__init__(id=id, x=x, y=y, score=score, polarity=polarity)


@register_tool(prefix="[EDGE]", default_name="Edge Detection", group="Vision Tools")
class EdgeDetection(BaseTool):
    def __init__(self, name="Edge Detector"):
        super().__init__(name)
        self.params = {
            "detect_mode": "Single Edge",
            "edge_selector": "First",
            "edge1_polarity": "Dark\u2192Light",
            "edge2_polarity": "Light\u2192Dark",
            "single_polarity": "Any",
            "edge_threshold": 5,
            "min_edge_width": 3,
            "max_edge_width": 9999,
            "gaussian_sigma": 1.0,
            "subpixel": True,
            "roi": [0, 0, 0, 0, 0],
            "show_search_roi": True,
            "show_direction_arrow": True,
            "show_edge_lines": True,
            "show_edge_points": True,
            "show_labels": True,
            "show_width_line": True,
            "copy_overlays_to_input": True,
        }
        self.inputs = {"input image": None}
        self.outputs = {
            "output image": None,
            "count": 0,
            "results": ToolResults(),
        }

    def run(self, advance=False):
        img_input = self.inputs.get("input image")
        if img_input is None:
            self.outputs["output image"] = None
            self.outputs["count"] = 0
            self.outputs["results"] = ToolResults()
            return

        img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)

        abs_roi = img.fixture_to_image(self.params["roi"])
        ax, ay, aw, ah, abs_angle = abs_roi
        h_img, w_img = img.data.shape[:2]

        if aw <= 0 or ah <= 0:
            self.outputs["output image"] = img
            self.outputs["count"] = 0
            self.outputs["results"] = ToolResults()
            self.data_changed.emit()
            return

        cx = ax + aw / 2.0
        cy = ay + ah / 2.0
        theta = np.deg2rad(abs_angle)
        u = (np.cos(theta), np.sin(theta))
        v = (-np.sin(theta), np.cos(theta))

        num_t = max(1, int(aw))
        num_s = max(1, int(ah))
        t = np.linspace(-aw / 2.0, aw / 2.0, num_t)
        s = np.linspace(-ah / 2.0, ah / 2.0, num_s)

        T, S = np.meshgrid(t, s)
        X = cx + u[0] * T + v[0] * S
        Y = cy + u[1] * T + v[1] * S

        map1 = X.astype(np.float32)
        map2 = Y.astype(np.float32)

        gray = cv2.cvtColor(img.data, cv2.COLOR_BGR2GRAY) if len(img.data.shape) == 3 else img.data
        samples = cv2.remap(gray, map1, map2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        profile = np.mean(samples, axis=1, dtype=np.float32)

        sigma = float(self.params.get("gaussian_sigma", 1.0))
        grad = self._derivative_of_gaussian(profile, sigma)
        threshold = max(1, self.params["edge_threshold"])
        subpixel = self.params["subpixel"]
        ds = ah / max(1, num_s - 1) if num_s > 1 else 0.0

        if self.params["detect_mode"] == "Edge Pair":
            raw = self._find_edge_pair(grad, threshold, subpixel)
            if raw is not None:
                e1_pos, e2_pos, width, score1, score2 = raw
                s1 = (-ah / 2.0) + e1_pos * ds
                s2 = (-ah / 2.0) + e2_pos * ds
                p1 = (cx + v[0] * s1, cy + v[1] * s1)
                p2 = (cx + v[0] * s2, cy + v[1] * s2)
                result = EdgePairResult(
                    id=0, x1=p1[0], y1=p1[1], x2=p2[0], y2=p2[1],
                    width=width, score1=score1, score2=score2
                )
            else:
                result = None
        else:
            raw = self._find_single_edge(grad, threshold, subpixel)
            if raw is not None:
                e_pos, score, polarity = raw
                s_off = (-ah / 2.0) + e_pos * ds
                px = cx + v[0] * s_off
                py = cy + v[1] * s_off
                result = SingleEdgeResult(id=0, x=px, y=py, score=score, polarity=polarity)
            else:
                result = None

        vision_img = VisionImage(img.data, x=img.x, y=img.y, angle=img.angle)
        vision_img.inherit_overlays_from(img)
        propagate_3d(img, vision_img, regenerate=False)
        vision_img.get_overlay_layer("default", source_tool=self.name)
        layer = vision_img.get_overlay_layer("default")
        layer.source_tool = self.name
        layer.elements = [e for e in layer.elements if e.source_tool != self.name]

        ax_c = max(0, min(int(ax), w_img - 1))
        ay_c = max(0, min(int(ay), h_img - 1))
        aw_c = max(1, min(int(aw), w_img - ax_c))
        ah_c = max(1, min(int(ah), h_img - ay_c))
        is_full_img = (ax_c == 0 and ay_c == 0 and aw_c >= w_img and ah_c >= h_img)

        if self.params.get("show_search_roi", True) and not is_full_img:
            layer.add_roi([float(ax_c), float(ay_c), float(aw_c), float(ah_c), float(abs_angle)],
                          color=(255, 128, 0), thickness=2)

        if self.params.get("show_direction_arrow", True):
            self._draw_direction_arrow(layer, cx, cy, v, ah_c, (255, 200, 0))

        self._draw_overlays(layer, result, u, v)

        self.outputs["output image"] = vision_img
        if result is not None:
            self.outputs["count"] = 1
            self.outputs["results"] = ToolResults([result])
        else:
            self.outputs["count"] = 0
            self.outputs["results"] = ToolResults()
        self._copy_overlays_to_input(vision_img, img)
        self.data_changed.emit()

    def _derivative_of_gaussian(self, profile, sigma):
        if sigma > 0.1:
            size = int(6 * sigma) | 1
            x = np.arange(size) - size // 2
            kernel = np.exp(-x**2 / (2 * sigma**2))
            kernel /= kernel.sum()
            pad = size // 2
            padded = np.pad(profile, pad, mode='edge')
            smoothed = np.convolve(padded, kernel, mode='valid')
        else:
            smoothed = profile
        grad = np.zeros_like(smoothed)
        grad[1:-1] = (smoothed[2:] - smoothed[:-2]) * 0.5
        return grad

    def _find_single_edge(self, grad, threshold, subpixel):
        polarity = self.params["single_polarity"]
        selector = self.params["edge_selector"]

        if polarity == "Dark\u2192Light":
            peaks = (grad[1:-1] > grad[:-2]) & (grad[1:-1] > grad[2:])
            peak_indices = np.where(peaks)[0] + 1
            peak_values = grad[peak_indices]
        elif polarity == "Light\u2192Dark":
            peaks = (grad[1:-1] < grad[:-2]) & (grad[1:-1] < grad[2:])
            peak_indices = np.where(peaks)[0] + 1
            peak_values = -grad[peak_indices]
        else:
            abs_grad = np.abs(grad)
            peaks = (abs_grad[1:-1] > abs_grad[:-2]) & (abs_grad[1:-1] > abs_grad[2:])
            peak_indices = np.where(peaks)[0] + 1
            peak_values = abs_grad[peak_indices]

        valid = peak_values >= threshold
        peak_indices = peak_indices[valid]
        peak_values = peak_values[valid]

        if len(peak_indices) == 0:
            return None

        if selector == "First":
            best_idx = peak_indices[0]
        elif selector == "Last":
            best_idx = peak_indices[-1]
        else:
            best_idx = peak_indices[np.argmax(peak_values)]

        edge_pos = self._subpixel_peak(grad, best_idx) if subpixel else float(best_idx)
        return edge_pos, float(abs(grad[best_idx])), polarity

    def _find_edge_pair(self, grad, threshold, subpixel):
        pol1 = self.params["edge1_polarity"]
        pol2 = self.params["edge2_polarity"]
        min_w = self.params["min_edge_width"]
        max_w = self.params["max_edge_width"]
        abs_grad = np.abs(grad)

        def _get_peaks(pol):
            if pol == "Dark\u2192Light":
                peaks = (grad[1:-1] > grad[:-2]) & (grad[1:-1] > grad[2:])
                idx = np.where(peaks)[0] + 1
                val = grad[idx]
            elif pol == "Light\u2192Dark":
                peaks = (grad[1:-1] < grad[:-2]) & (grad[1:-1] < grad[2:])
                idx = np.where(peaks)[0] + 1
                val = -grad[idx]
            else:
                peaks = (abs_grad[1:-1] > abs_grad[:-2]) & (abs_grad[1:-1] > abs_grad[2:])
                idx = np.where(peaks)[0] + 1
                val = abs_grad[idx]
            return idx, val

        idx1, val1 = _get_peaks(pol1)
        idx2, val2 = _get_peaks(pol2)

        valid1 = val1 >= threshold
        idx1 = idx1[valid1]
        val1 = val1[valid1]
        valid2 = val2 >= threshold
        idx2 = idx2[valid2]
        val2 = val2[valid2]

        if len(idx1) == 0 or len(idx2) == 0:
            return None

        if subpixel:
            sub1 = [self._subpixel_peak(grad, i) for i in idx1]
            sub2 = [self._subpixel_peak(grad, i) for i in idx2]
        else:
            sub1 = [float(i) for i in idx1]
            sub2 = [float(i) for i in idx2]

        # Vectorized: broadcasting N×M matrix
        diff = idx2[None, :] - idx1[:, None]
        mask = (diff >= min_w) & (diff <= max_w) & (idx2[None, :] > idx1[:, None])

        scores = (val1[:, None] + val2[None, :]) * 0.5
        scores = np.where(mask, scores, -1.0)

        if not np.any(scores >= 0):
            return None

        best_flat = np.argmax(scores)
        i, j = np.unravel_index(best_flat, scores.shape)

        e1_pos = sub1[i]
        e2_pos = sub2[j]
        w_subpixel = e2_pos - e1_pos

        return (e1_pos, e2_pos, w_subpixel if subpixel else int(idx2[j] - idx1[i]),
                float(abs_grad[idx1[i]]), float(abs_grad[idx2[j]]))

    def _subpixel_peak(self, grad, idx):
        if idx <= 0 or idx >= len(grad) - 1:
            return float(idx)
        g_left = grad[idx - 1]
        g_mid = grad[idx]
        g_right = grad[idx + 1]
        denom = 2 * (g_left + g_right - 2 * g_mid)
        if abs(denom) < 1e-6:
            return float(idx)
        delta = (g_left - g_right) / denom
        return float(idx) + delta

    def _draw_direction_arrow(self, layer, cx, cy, v, ah, color):
        margin = 5
        base_offset = ah / 2 + margin
        tip_extend = 8
        half_w = 3
        perp_x = -v[1]
        perp_y = v[0]
        pts = [
            (cx + v[0] * (base_offset + tip_extend), cy + v[1] * (base_offset + tip_extend)),
            (cx + v[0] * base_offset + perp_x * half_w, cy + v[1] * base_offset + perp_y * half_w),
            (cx + v[0] * base_offset - perp_x * half_w, cy + v[1] * base_offset - perp_y * half_w),
        ]
        layer.add_polygon(pts, color=color, thickness=1, fill=True)

    def _draw_overlays(self, layer, result, u, v):
        if result is None:
            return

        edge1_color = (0, 255, 255)
        edge2_color = (255, 100, 255)
        edge1_color_single = (0, 255, 255)

        if isinstance(result, EdgePairResult):
            if self.params.get("show_edge_lines", True):
                self._draw_edge_line(layer, result.x1, result.y1, u, edge1_color, 2, ref_id="edge1")
                self._draw_edge_line(layer, result.x2, result.y2, u, edge2_color, 2, ref_id="edge2")
            if self.params.get("show_edge_points", True):
                layer.add_point((result.x1, result.y1), color=edge1_color, radius=4, ref_id="edge1")
                layer.add_point((result.x2, result.y2), color=edge2_color, radius=4, ref_id="edge2")
            if self.params.get("show_width_line", True):
                mid_x = (result.x1 + result.x2) * 0.5
                mid_y = (result.y1 + result.y2) * 0.5
                layer.add_line((result.x1, result.y1), (result.x2, result.y2),
                               color=(255, 255, 0), thickness=1, ref_id="width")
                layer.add_text(f"w:{result.width:.1f}", (mid_x + 5, mid_y - 5),
                               color=(255, 255, 0), font_scale=0.6, thickness=2)
        elif isinstance(result, SingleEdgeResult):
            if self.params.get("show_edge_lines", True):
                self._draw_edge_line(layer, result.x, result.y, u, edge1_color_single, 2, ref_id="edge1")
            if self.params.get("show_edge_points", True):
                layer.add_point((result.x, result.y), color=edge1_color_single, radius=4, ref_id="edge1")
            if self.params.get("show_labels", True):
                pol = result.polarity.replace("\u2192", "")
                layer.add_text(f"E:({result.x:.1f},{result.y:.1f}) {pol}",
                               (result.x + 5, result.y - 15),
                               color=edge1_color_single, font_scale=0.5, thickness=1)

    def _draw_edge_line(self, layer, ex, ey, u, color, thickness, ref_id=""):
        margin = 5
        x1, y1 = ex - u[0] * margin, ey - u[1] * margin
        x2, y2 = ex + u[0] * margin, ey + u[1] * margin
        layer.add_line((x1, y1), (x2, y2), color=color, thickness=thickness, ref_id=ref_id)

    def get_config_schema(self) -> dict:
        return {
            "type": "object",
            "properties": {
                "detect_mode": {"type": "string", "enum": ["Single Edge", "Edge Pair"], "default": "Single Edge"},
                "edge_selector": {"type": "string", "enum": ["First", "Last", "Best"], "default": "First"},
                "edge1_polarity": {"type": "string", "enum": ["Dark→Light", "Light→Dark", "Any"], "default": "Dark→Light"},
                "edge2_polarity": {"type": "string", "enum": ["Dark→Light", "Light→Dark", "Any"], "default": "Light→Dark"},
                "single_polarity": {"type": "string", "enum": ["Dark→Light", "Light→Dark", "Any"], "default": "Any"},
                "edge_threshold": {"type": "integer", "minimum": 1, "maximum": 255, "default": 5},
                "min_edge_width": {"type": "integer", "minimum": 1, "maximum": 9999, "default": 3},
                "max_edge_width": {"type": "integer", "minimum": 1, "maximum": 9999, "default": 9999},
                "gaussian_sigma": {"type": "number", "minimum": 0.1, "maximum": 10.0, "default": 1.0},
                "subpixel": {"type": "boolean", "default": True},
                "show_search_roi": {"type": "boolean", "default": True},
                "show_direction_arrow": {"type": "boolean", "default": True},
                "show_edge_lines": {"type": "boolean", "default": True},
                "show_edge_points": {"type": "boolean", "default": True},
                "show_labels": {"type": "boolean", "default": True},
                "show_width_line": {"type": "boolean", "default": True},
                "copy_overlays_to_input": {"type": "boolean", "default": True},
            },
        }

    def get_config_widget(self):
        from .config import EdgeDetectionConfig
        return EdgeDetectionConfig(self)


