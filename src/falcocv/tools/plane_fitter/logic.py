import numpy as np
from falcocv.core.tool_base import BaseTool, ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_3d import calib_from_vimg, get_3d_data
from falcocv.core.pointcloud_utils import propagate_3d
from falcocv.core.plane_fit import plane_fit as _plane_fit


class PlaneFitResult(ToolResult):
    def __init__(self, normal, center, distance, inlier_count, inlier_ratio, rms_error, total_points, polygon_2d):
        super().__init__(
            normal=normal.tolist() if hasattr(normal, "tolist") else list(normal),
            center=center.tolist() if hasattr(center, "tolist") else list(center),
            distance=distance,
            inlier_count=inlier_count,
            inlier_ratio=inlier_ratio,
            rms_error=rms_error,
            total_points=total_points,
        )
        if polygon_2d is not None:
            self.polygon_2d = polygon_2d


@register_tool(prefix="[PLAN]", default_name="Plane Fitter", group="3D")
class PlaneFitter(BaseTool):
    def __init__(self, name="Plane Fitter"):
        super().__init__(name)
        self.params = {
            "rois": [],
            "method": "ransac",
            "distance_threshold": 1.0,
            "max_iterations": 100,
            "min_points": 10,
            "max_fit_points": 50000,
            "copy_overlays_to_input": True,
            "show_overlays": True,
        }
        self.inputs = {"input image": None}
        self.outputs = {
            "output image": None,
            "results": ToolResults(),
        }

    def run(self, advance=False):
        img_input = self.inputs.get("input image")
        if img_input is None:
            self._clear_outputs()
            return

        img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)
        depth = img.data if hasattr(img, "data") else img
        if depth is None:
            self._clear_outputs()
            return

        cloud = get_3d_data(img)
        if cloud is None or cloud.get("points") is None or len(cloud["points"]) == 0:
            self._clear_outputs()
            return

        points = cloud["points"]
        calib = calib_from_vimg(img)
        rois = self.params.get("rois", [])

        # Convert fixture ROIs to absolute image coordinates
        if hasattr(img, "fixture_to_image"):
            rois_abs = []
            for r in rois:
                r_copy = dict(r)
                r_copy["roi"] = list(img.fixture_to_image(r["roi"]))
                rois_abs.append(r_copy)
        else:
            rois_abs = rois

        output = _plane_fit({
            "image": depth,
            "points": points,
            "rois": rois_abs,
            "calib": calib,
            "params": {
                "method": self.params.get("method", "ransac"),
                "distance_threshold": float(self.params.get("distance_threshold", 1.0)),
                "max_iterations": int(self.params.get("max_iterations", 100)),
                "min_points": int(self.params.get("min_points", 10)),
                "max_fit_points": int(self.params.get("max_fit_points", 50000)),
            }
        })

        if "error" in output:
            self.last_error = str(output["error"])
            self._clear_outputs()
            return

        normal = np.asarray(output["normal"], dtype=np.float32)
        center = np.asarray(output["center"], dtype=np.float32)
        distance = float(output["distance"])
        inlier_count = int(output["inlier_count"])
        inlier_ratio = float(output["inlier_ratio"])
        rms_error = float(output["rms_error"])
        total = int(output["total_points"])

        selected_indices = np.asarray(output["selected_indices"], dtype=np.int64)
        inlier_mask_sel = np.asarray(output["inlier_mask"], dtype=bool)
        full_inlier_mask = np.zeros(len(points), dtype=bool)
        if len(selected_indices) == len(inlier_mask_sel):
            full_inlier_mask[selected_indices] = inlier_mask_sel

        polygon_2d = output.get("polygon_2d")

        # Validate polygon coordinates — NaN/Inf would crash overlay rendering.
        if polygon_2d is not None:
            clean = []
            for p in polygon_2d:
                x, y = float(p[0]), float(p[1])
                if np.isfinite(x) and np.isfinite(y):
                    clean.append([x, y])
            if len(clean) < 3:
                polygon_2d = None
            else:
                polygon_2d = clean

        scale_x, scale_y = calib[0], calib[1]

        # Build output image sharing the input data
        out = VisionImage(
            img.data,
            x=img.x, y=img.y, angle=img.angle,
            scale_x=img.scale_x, scale_y=img.scale_y,
            scale_z=img.scale_z, offset_z=img.offset_z,
        )
        out.inherit_overlays_from(img)
        propagate_3d(img, out, regenerate=False)

        # Clean old overlays from this tool
        layer = out.get_overlay_layer("default")
        layer.source_tool = self.name
        layer.elements = [e for e in layer.elements if e.source_tool != self.name]

        show_overlays = bool(self.params.get("show_overlays", True))

        # Plane polygon overlay (semi-transparent filled quadrilateral)
        if show_overlays and polygon_2d is not None and len(polygon_2d) >= 3:
            pts = [[float(p[0]), float(p[1])] for p in polygon_2d]
            layer.add_polygon(
                pts,
                color=(0, 200, 255),
                thickness=2,
                fill=True,
                fill_alpha=0.15,
                ref_id="plane_polygon"
            )

            # 3D world polygon for Viewer3D
            h_img, w_img = depth.shape[:2]
            nx, ny, nz = normal
            verts_3d = []
            for col, row in polygon_2d:
                wx = (w_img - 1 - col) * scale_x
                wy = row * scale_y
                if abs(nz) > 1e-12:
                    wz = (distance - nx * wx - ny * wy) / nz
                else:
                    wz = center[2]
                verts_3d.append([wx, wy, wz])
            layer.add_world_polygon(verts_3d,
                                    color=(0, 200, 255), fill_alpha=0.15,
                                    ref_id="plane_polygon_3d")

        # Normal arrow projected to 2D
        if show_overlays and abs(scale_x) > 1e-12 and abs(scale_y) > 1e-12 and depth is not None:
            h, w = depth.shape[:2]
            arrow_len = min(w * scale_x, h * scale_y) * 0.25
            # Undo the world-X mirroring used by the point cloud:
            # image_col = (img_w - 1) - world_x / scale_x.
            def _world_x_to_col(wx):
                return float(w - 1) - float(wx) / scale_x
            cx_2d = _world_x_to_col(center[0])
            cy_2d = float(center[1]) / scale_y
            tip_x = _world_x_to_col(center[0] + normal[0] * arrow_len)
            tip_y = float(center[1] + normal[1] * arrow_len) / scale_y
            layer.add_line(
                (cx_2d, cy_2d), (tip_x, tip_y),
                color=(255, 50, 50), thickness=3, ref_id="plane_normal"
            )
            layer.add_text(
                "normal",
                position=(tip_x + 5, tip_y),
                color=(255, 50, 50),
                font_scale=0.5,
                thickness=2,
            )

        # ROI overlays — must use absolute image coords so they align with the
        # displayed range image, even when the stored ROIs are in fixture coords.
        if show_overlays:
            for roi_entry in rois_abs:
                roi_2d = roi_entry.get("roi", [0, 0, 0, 0, 0])
                if roi_2d[2] > 0 and roi_2d[3] > 0:
                    layer.add_roi(
                        [float(roi_2d[0]), float(roi_2d[1]),
                         float(roi_2d[2]), float(roi_2d[3]),
                         float(roi_2d[4]) if len(roi_2d) >= 5 else 0.0],
                        color=(0, 255, 100),
                        thickness=2,
                        ref_id="plane_roi"
                    )

        self.outputs["output image"] = out
        self.outputs["results"] = ToolResults([PlaneFitResult(
            normal, center, distance, inlier_count, inlier_ratio, rms_error, total, polygon_2d
        )])

        self._copy_overlays_to_input(out, img)

    def _clear_outputs(self):
        self.outputs["output image"] = None
        self.outputs["results"] = ToolResults()

    def serialize(self, job_manager=None):
        data = super().serialize(job_manager)
        data["type"] = "PlaneFitter"
        return data

    def get_config_widget(self):
        from .config import PlaneFitterConfig
        return PlaneFitterConfig(self)
