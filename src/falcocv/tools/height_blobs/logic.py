"""3D blob measurement above a reference plane.

Each connected region of the height mask is one feature: its tip is the
centroid of the band just below the highest point, expressed in the frame of
the reference plane so tilt does not bias the measurement.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from falcocv.core.features import FeaturePoint, FeatureSet
from falcocv.core.plane_fit import PlaneModel
from falcocv.core.port_types import FEATURES, IMAGE, PLANE
from falcocv.core.pointcloud_utils import _is_depth_like
from falcocv.core.roi_3d import calib_from_vimg
from falcocv.core.tool_base import BaseTool, ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage

SORT_MODES = ("none", "u", "v", "h", "area", "grid")


class HeightBlobResult(ToolResult):
    def __init__(self, feature: FeaturePoint):
        super().__init__(
            id=feature.id,
            index=feature.index,
            u=feature.u,
            v=feature.v,
            h=feature.h,
            depth=feature.depth,
            area_px=feature.area_px,
            area_mm2=feature.area_mm2,
            volume_mm3=feature.volume_mm3,
            height_max=feature.height_max,
            height_p95=feature.height_p95,
            height_mean=feature.height_mean,
            points=feature.points,
            touches_border=feature.touches_border,
        )


@register_tool(prefix="[H3D]", default_name="Height Blobs 3D", group="3D")
class HeightBlobs3D(BaseTool):
    input_port_types = {"input image": IMAGE, "reference plane": PLANE}
    output_port_types = {
        "output image": IMAGE,
        "binary mask": IMAGE,
        "features": FEATURES,
    }

    def __init__(self, name="Height Blobs 3D"):
        super().__init__(name)
        self.inputs = {"input image": None, "reference plane": None}
        self.outputs = {
            "output image": None,
            "binary mask": None,
            "results": ToolResults(),
            "warnings": [],
            "features": None,
        }
        self.params = {
            "rois": [],
            "plane_offset": 0.0,
            "direction": "above",
            "min_height": 0.2,
            "seed_height": 0.5,
            "median_px": 0,
            "close_px": 1,
            "open_px": 1,
            "connectivity": 8,
            "min_points": 3,
            "max_blobs": 100,
            "watershed": False,
            "watershed_erode_px": 2,
            "tip_band": 0.05,
            "tip_stat": "mean",
            "sort_mode": "u",
            "sort_descending": False,
            "row_tolerance": 0.5,
            "show_overlays": True,
            "copy_overlays_to_input": True,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _kernel(radius: int):
        size = max(1, 2 * int(radius) + 1)
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))

    @staticmethod
    def _roi_corners(roi):
        x, y, w, h = (float(roi[0]), float(roi[1]), float(roi[2]), float(roi[3]))
        angle = float(roi[4]) if len(roi) >= 5 else 0.0
        return cv2.boxPoints(((x + w * 0.5, y + h * 0.5), (w, h), angle))

    def _roi_entries(self, img):
        rois = self.params.get("rois") or []
        entries = []
        for entry in rois:
            roi = entry.get("roi") if isinstance(entry, dict) else None
            if not roi or len(roi) < 4 or roi[2] <= 0 or roi[3] <= 0:
                continue
            if hasattr(img, "fixture_to_image"):
                roi_abs = list(img.fixture_to_image(roi))
            else:
                roi_abs = list(roi)
            z_min = float(entry.get("z_min", 0.0))
            z_max = float(entry.get("z_max", 1.0))
            entries.append((roi_abs, z_min, z_max))
        return entries

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    def run(self, advance=False) -> None:
        self.outputs["output image"] = None
        self.outputs["binary mask"] = None
        self.outputs["results"] = ToolResults()
        self.outputs["warnings"] = []
        self.outputs["features"] = None

        source = self.inputs.get("input image")
        if source is None:
            self.set_error("Input image is missing")
            self.data_changed.emit()
            return

        img = source if isinstance(source, VisionImage) else VisionImage.from_array(source)
        if not _is_depth_like(img):
            self.set_error("Input image has no depth data")
            self.data_changed.emit()
            return

        plane = self.inputs.get("reference plane")
        if not isinstance(plane, PlaneModel):
            self.set_error(
                "Reference plane is missing; connect a Plane Fitter output."
            )
            self.data_changed.emit()
            return

        depth = img.data
        depth_h, depth_w = depth.shape[:2]
        scale_x, scale_y, scale_z, offset_z = calib_from_vimg(img)
        plane = plane.offset(float(self.params.get("plane_offset", 0.0)))

        entries = self._roi_entries(img)
        if not entries:
            entries = [([0.0, 0.0, float(depth_w), float(depth_h), 0.0],
                        -math.inf, math.inf)]

        corners = np.concatenate([self._roi_corners(entry[0]) for entry in entries])
        x0 = max(0, int(math.floor(float(corners[:, 0].min()))))
        y0 = max(0, int(math.floor(float(corners[:, 1].min()))))
        x1 = min(depth_w, int(math.ceil(float(corners[:, 0].max()))) + 1)
        y1 = min(depth_h, int(math.ceil(float(corners[:, 1].max()))) + 1)
        if x1 <= x0 or y1 <= y0:
            self.set_error("ROI is outside the image")
            self.data_changed.emit()
            return

        crop = np.ascontiguousarray(depth[y0:y1, x0:x1], dtype=np.float32)
        crop_h, crop_w = crop.shape[:2]

        footprint = np.zeros((crop_h, crop_w), dtype=np.uint8)
        inside = np.zeros((crop_h, crop_w), dtype=bool)
        z_values = crop * np.float32(scale_z) + np.float32(offset_z)
        valid = np.isfinite(crop) & (crop != 0.0)
        for roi_abs, z_min, z_max in entries:
            polygon = np.round(self._roi_corners(roi_abs) - [x0, y0]).astype(np.int32)
            single = np.zeros((crop_h, crop_w), dtype=np.uint8)
            cv2.fillPoly(single, [polygon], 1)
            footprint |= single
            band = valid & (single > 0)
            auto_z = z_min == 0.0 and z_max == 1.0
            if not auto_z:
                band &= (z_values >= z_min) & (z_values <= z_max)
            inside |= band

        if not inside.any():
            self._set_warning(
                "No valid depth inside the ROI; check the ROI Z range."
            )
            self._publish_empty(img, plane)
            return

        normal = np.asarray(plane.normal, dtype=np.float64)
        abs_cols = np.arange(x0, x1, dtype=np.float32)
        abs_rows = np.arange(y0, y1, dtype=np.float32)
        world_x = ((depth_w - 1) - abs_cols) * np.float32(scale_x)
        world_y = abs_rows * np.float32(scale_y)
        height = (
            world_x[None, :] * np.float32(normal[0])
            + world_y[:, None] * np.float32(normal[1])
            + z_values * np.float32(normal[2])
            - np.float32(plane.distance)
        )
        height = height.astype(np.float32, copy=False)

        median_px = int(self.params.get("median_px", 0))
        if median_px >= 3:
            original = height.copy()
            smoothed = cv2.medianBlur(height.astype(np.float32), median_px)
            height = np.where(valid, smoothed, original).astype(np.float32)

        direction = str(self.params.get("direction", "above")).lower()
        sign = -1.0 if direction == "below" else 1.0
        measure = height if sign > 0 else -height

        min_height = float(self.params.get("min_height"))
        seed_height = float(self.params.get("seed_height"))
        grow = inside & (measure >= min_height)
        if not grow.any():
            lowest = float(np.nanmin(height[inside]))
            highest = float(np.nanmax(height[inside]))
            hint = (
                "apply a negative plane_offset or set direction to below"
                if direction != "below"
                else "apply a positive plane_offset or set direction to above"
            )
            self._set_warning(
                f"No features on the {direction} side of the plane. Heights "
                f"inside the ROI range from {lowest:.2f} to {highest:.2f}. If "
                f"the part is on the other side, {hint}."
            )
            self._publish_empty(img, plane)
            return

        if seed_height > min_height:
            seeds = inside & (measure >= seed_height)
            if not seeds.any():
                self._set_warning(
                    "No pixel reaches seed_height; lower it or raise the "
                    "plane_offset."
                )
                self._publish_empty(img, plane)
                return
            count, labels = cv2.connectedComponents(
                grow.astype(np.uint8),
                connectivity=int(self.params.get("connectivity", 8)),
            )
            keep = np.unique(labels[seeds])
            keep = keep[keep > 0]
            keep_lut = np.zeros(count, dtype=bool)
            keep_lut[keep] = True
            grow = keep_lut[labels]

        mask = grow.astype(np.uint8)
        close_px = int(self.params.get("close_px", 0))
        if close_px > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self._kernel(close_px))
        open_px = int(self.params.get("open_px", 0))
        if open_px > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel(open_px))
        mask &= footprint

        connectivity = int(self.params.get("connectivity", 8))
        if self.params.get("watershed") and mask.any():
            mask = self._split_watershed(
                mask,
                int(self.params.get("watershed_erode_px", 2)),
                connectivity,
            )

        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            mask, connectivity=connectivity
        )
        min_points = max(1, int(self.params.get("min_points", 3)))
        labels_keep = [
            label for label in range(1, count)
            if int(stats[label, cv2.CC_STAT_AREA]) >= min_points
        ]
        max_blobs = int(self.params.get("max_blobs", 100))
        if max_blobs > 0 and len(labels_keep) > max_blobs:
            labels_keep.sort(
                key=lambda label: int(stats[label, cv2.CC_STAT_AREA]),
                reverse=True,
            )
            labels_keep = labels_keep[:max_blobs]

        if not labels_keep:
            self._set_warning("Mask is empty after filtering; lower min_points.")
            self._publish_empty(img, plane)
            return

        angle_hint = None
        if entries:
            roi_angle = float(entries[0][0][4]) if len(entries[0][0]) >= 5 else 0.0
            radians = math.radians(roi_angle)
            angle_hint = (math.cos(radians), -math.sin(radians), 0.0)
        frame = plane.frame(angle_hint)

        pixel_area = abs(scale_x * scale_y)
        eroded = cv2.erode(footprint, np.ones((3, 3), np.uint8))
        tip_band = max(0.0, float(self.params.get("tip_band", 0.05)))
        tip_stat = str(self.params.get("tip_stat", "mean"))

        features = []
        for label in labels_keep:
            left = int(stats[label, cv2.CC_STAT_LEFT])
            top = int(stats[label, cv2.CC_STAT_TOP])
            box_w = int(stats[label, cv2.CC_STAT_WIDTH])
            box_h = int(stats[label, cv2.CC_STAT_HEIGHT])
            region = labels[top:top + box_h, left:left + box_w] == label
            if not region.any():
                continue
            region_height = height[top:top + box_h, left:left + box_w]
            region_measure = measure[top:top + box_h, left:left + box_w]
            values = region_measure[region]
            height_max = float(values.max())
            height_p95 = float(np.percentile(values, 95))
            height_mean = float(values.mean())

            top_mask = region & (region_measure >= height_max - tip_band)
            if not top_mask.any():
                top_mask = region
            local_rows, local_cols = np.nonzero(top_mask)
            abs_rows = local_rows + top + y0
            abs_cols = local_cols + left + x0
            world_points = np.stack(
                [
                    ((depth_w - 1) - abs_cols).astype(np.float64) * scale_x,
                    abs_rows.astype(np.float64) * scale_y,
                    crop[local_rows + top, local_cols + left].astype(np.float64)
                    * scale_z + offset_z,
                ],
                axis=1,
            )
            u_values, v_values, h_values = frame.to_plane(world_points)
            if tip_stat == "median":
                u = float(np.median(u_values))
                v = float(np.median(v_values))
                h = float(np.median(h_values))
            else:
                u = float(np.mean(u_values))
                v = float(np.mean(v_values))
                h = float(np.mean(h_values))

            area_px = int(np.count_nonzero(region))
            contour = ()
            contours, _ = cv2.findContours(
                region.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            if contours:
                largest = max(contours, key=cv2.contourArea)
                contour = tuple(
                    (float(point[0][0] + left + x0),
                     float(point[0][1] + top + y0))
                    for point in largest
                )

            border_region = eroded[top:top + box_h, left:left + box_w]
            features.append(FeaturePoint(
                id=int(label),
                index=0,
                u=u,
                v=v,
                h=h,
                depth=sign * h,
                area_px=area_px,
                area_mm2=float(area_px) * pixel_area,
                volume_mm3=float(np.sum(np.clip(values, 0.0, None))) * pixel_area,
                height_max=height_max,
                height_p95=height_p95,
                height_mean=height_mean,
                points=area_px,
                touches_border=bool(np.any(region & (border_region == 0))),
                contour_2d=contour,
            ))

        features = self._sort(features)
        feature_set = FeatureSet(frame=frame, points=tuple(features))

        self.outputs["features"] = feature_set
        self.outputs["results"] = ToolResults(
            [HeightBlobResult(feature) for feature in features]
        )
        self.outputs["binary mask"] = self._mask_image(mask, img)
        self.outputs["output image"] = self._draw_overlays(
            img, features, frame
        )
        self.data_changed.emit()

    # ------------------------------------------------------------------
    # Output helpers
    # ------------------------------------------------------------------
    def _set_warning(self, message: str) -> None:
        warnings = list(self.outputs.get("warnings") or [])
        warnings.append(message)
        self.outputs["warnings"] = warnings

    def _publish_empty(self, img, plane) -> None:
        self.outputs["results"] = ToolResults()
        angle_hint = (1.0, 0.0, 0.0)
        self.outputs["features"] = FeatureSet(
            frame=plane.frame(angle_hint), points=()
        )
        self.outputs["output image"] = self._draw_overlays(img, [], None)
        self.data_changed.emit()

    @staticmethod
    def _mask_image(mask, img) -> VisionImage:
        display = (mask * 255).astype(np.uint8)
        return VisionImage(display)

    def _draw_overlays(self, img, features, frame):
        out = VisionImage(
            img.data,
            x=img.x,
            y=img.y,
            angle=img.angle,
            scale_x=img.scale_x,
            scale_y=img.scale_y,
            scale_z=img.scale_z,
            offset_z=img.offset_z,
        )
        out.inherit_overlays_from(img)
        layer = out.get_overlay_layer("default")
        layer.source_tool = self.name
        layer.elements = [
            element for element in layer.elements
            if element.source_tool != self.name
        ]
        if bool(self.params.get("show_overlays", True)):
            for feature in features:
                if feature.contour_2d:
                    layer.add_contour(
                        list(feature.contour_2d),
                        color=(0, 200, 255),
                        thickness=2,
                        ref_id=f"blob_{feature.index}",
                    )
                if feature.contour_2d:
                    x, y = feature.contour_2d[0]
                else:
                    x = y = 0.0
                layer.add_text(
                    f"{feature.index}: {feature.h:.2f}",
                    position=(x + 4, y - 4),
                    color=(0, 200, 255),
                    font_scale=0.5,
                    thickness=1,
                )
            if frame is not None and features:
                u_values = np.array([feature.u for feature in features])
                v_values = np.array([feature.v for feature in features])
                h_values = np.array([feature.h for feature in features])
                world_points = frame.to_world(u_values, v_values, h_values)
                layer.add_world_points(
                    [
                        tuple(float(value) for value in point)
                        for point in world_points
                    ],
                    color=(0, 220, 255),
                    point_size=6.0,
                    ref_id="blob_tips",
                )
        self._copy_overlays_to_input(out, img)
        return out

    def _sort(self, features):
        mode = str(self.params.get("sort_mode", "u"))
        if mode not in SORT_MODES:
            mode = "u"
        descending = bool(self.params.get("sort_descending", False))
        tolerance = max(1e-6, float(self.params.get("row_tolerance", 0.5)))

        if mode == "none":
            ordered = list(features)
        elif mode == "grid":
            ordered = sorted(
                features,
                key=lambda f: (round(f.v / tolerance), f.u, f.id),
            )
            if descending:
                ordered.reverse()
        else:
            def key(feature):
                if mode == "u":
                    return (feature.u, feature.id)
                if mode == "v":
                    return (feature.v, feature.id)
                if mode == "h":
                    return (feature.depth, feature.id)
                return (feature.area_px, feature.id)

            ordered = sorted(features, key=key, reverse=descending)

        return [
            FeaturePoint(**{**feature.__dict__, "index": index})
            for index, feature in enumerate(ordered)
        ]

    def _split_watershed(self, mask, erode_px, connectivity):
        radius = max(1, int(erode_px))
        seeds = cv2.erode(mask, self._kernel(radius))
        count, seed_labels = cv2.connectedComponents(seeds, connectivity=connectivity)
        if count <= 2:
            return mask
        markers = seed_labels.astype(np.int32) + 1
        full_count, full_labels = cv2.connectedComponents(mask, connectivity=connectivity)
        for label in range(1, full_count):
            if not np.any(markers[full_labels == label]):
                rows, cols = np.nonzero(full_labels == label)
                markers[rows[len(rows) // 2], cols[len(cols) // 2]] = int(markers.max()) + 1
        image = cv2.cvtColor((mask * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
        cv2.watershed(image, markers)
        # markers: 1 = background, 2..N = seeds, -1 = watershed boundary.
        split = ((markers > 1).astype(np.uint8)) & mask
        return split

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    def get_config_widget(self):
        from .config import HeightBlobs3DConfig
        return HeightBlobs3DConfig(self)
