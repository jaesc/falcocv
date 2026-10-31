import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_base import ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi, create_roi_only_output
from falcocv.core.pointcloud_utils import propagate_3d


class BlobResult(ToolResult):
    """Specific result for a Blob detection."""
    def __init__(self, id, x, y, area):
        super().__init__(id=id, x=x, y=y, area=area)


@register_tool(prefix="[BLOB]", default_name="Blob Detector", group="Vision Tools")
class BlobDetector(BaseTool):
    def __init__(self, name="Blob Detector"):
        super().__init__(name)
        self.params = {
            "threshold_mode": "Fixed", 
            "threshold_value": 128,
            "adaptive_block_size": 11,
            "adaptive_const": 2,
            "polarity": "Bright Blobs", 
            "blob_mode": "Outer",
            "area_min": 100,
            "area_max": 1000000,
            "show_contours": True,
            "show_centroids": True,
            "show_labels": True,
            "show_search_roi": True,
            "sort_mode": "Area Descending",
            "output_mode": "Full Image",
            "copy_overlays_to_input": True,
            "roi": [0, 0, 0, 0, 0]
        }
        self.inputs = {"input image": None}
        self.outputs = {
            "output image": None, 
            "binary mask": None, 
            "count": 0, 
            "results": ToolResults() 
        }

    @staticmethod
    def _composite_binary_mask(binary, roi):
        """Place the ROI binary crop back into the full image.

        Axis-aligned crops can be copied directly, using the real crop shape
        (the clamped ROI may be smaller than the requested one).  Rotated
        crops must be warped back with the ROI transform; pasting them into
        an axis-aligned slice is both wrong and shape-dependent.
        """
        h_img = int(roi["h_img"])
        w_img = int(roi["w_img"])
        mask_full = np.zeros((h_img, w_img), dtype=np.uint8)
        if binary is None or binary.size == 0:
            return mask_full

        M_inv = roi.get("M_inv")
        if M_inv is None:
            ax = max(0, min(int(roi["ax"]), w_img))
            ay = max(0, min(int(roi["ay"]), h_img))
            bh, bw = binary.shape[:2]
            bw = min(bw, w_img - ax)
            bh = min(bh, h_img - ay)
            if bw > 0 and bh > 0:
                mask_full[ay:ay + bh, ax:ax + bw] = binary[:bh, :bw]
            return mask_full

        # Rotated ROI: M_inv maps ROI coordinates to image coordinates.
        rh, rw = binary.shape[:2]
        corners = np.array(
            [[0.0, 0.0], [rw, 0.0], [rw, rh], [0.0, rh]],
            dtype=np.float64,
        )
        pts = corners @ M_inv[:, :2].T + M_inv[:, 2]
        x0 = max(0, int(np.floor(pts[:, 0].min())))
        y0 = max(0, int(np.floor(pts[:, 1].min())))
        x1 = min(w_img, int(np.ceil(pts[:, 0].max())) + 1)
        y1 = min(h_img, int(np.ceil(pts[:, 1].max())) + 1)
        if x1 <= x0 or y1 <= y0:
            return mask_full

        M = M_inv.copy()
        M[0, 2] -= x0
        M[1, 2] -= y0
        warped = cv2.warpAffine(
            binary,
            M,
            (x1 - x0, y1 - y0),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        mask_full[y0:y1, x0:x1] = warped
        return mask_full

    def run(self, advance=False):
        img_input = self.inputs.get("input image")
        if img_input is None:
            self.outputs["output image"] = None
            self.outputs["binary mask"] = None
            self.outputs["count"] = 0
            self.outputs["results"] = ToolResults()
            return
            
        img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)

        # Extract ROI using shared utility
        roi = extract_roi(img, self.params["roi"])
        roi_src = roi["roi_src"]
        ax, ay, aw, ah = roi["ax"], roi["ay"], roi["aw"], roi["ah"]

        # 1. Pre-processing
        gray = cv2.cvtColor(roi_src, cv2.COLOR_BGR2GRAY) if len(roi_src.shape) == 3 else roi_src.copy()

        # 2. Thresholding
        thresh_type = cv2.THRESH_BINARY
        if self.params["polarity"] == "Dark Blobs":
            thresh_type = cv2.THRESH_BINARY_INV

        if self.params["threshold_mode"] == "Fixed":
            _, binary = cv2.threshold(gray, self.params["threshold_value"], 255, thresh_type)
        else:
            bs = self.params["adaptive_block_size"]
            if bs % 2 == 0: bs += 1
            binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          thresh_type, bs, self.params["adaptive_const"])
        
        # Save Binary Mask Output.  The ROI crop can be larger than the
        # clamped image box (rotated ROIs, or ROIs crossing an image edge),
        # so compositing it with a plain slice raises a shape mismatch.
        mask_full = self._composite_binary_mask(binary, roi)
        self.outputs["binary mask"] = VisionImage(mask_full)

        # 3. Find Contours
        # "Outer" keeps the historical RETR_EXTERNAL behaviour.  "Nested"
        # uses CCOMP and keeps only level-0 contours: one contour per
        # foreground group, including groups nested inside a hole, while
        # dropping the hole boundaries themselves.
        if self.params.get("blob_mode", "Outer") == "Nested":
            contours, hierarchy = cv2.findContours(
                binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
            )
            if hierarchy is not None:
                contours = [
                    cnt for index, cnt in enumerate(contours)
                    if hierarchy[0][index][3] == -1
                ]
        else:
            contours, _ = cv2.findContours(
                binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

        # 4. Filter and Create Results
        raw_results = []
        M_inv = roi.get("M_inv", None)

        for i, cnt in enumerate(contours):
            area = cv2.contourArea(cnt)
            if self.params["area_min"] <= area <= self.params["area_max"]:
                if M_inv is not None:
                    pts = cnt.reshape(-1, 2).astype(np.float64)
                    pts = pts @ M_inv[:, :2].T + M_inv[:, 2]
                    cnt_offset = pts.reshape(-1, 1, 2).astype(np.int32)
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = M_inv[0, 0] * M["m10"] / M["m00"] + M_inv[0, 1] * M["m01"] / M["m00"] + M_inv[0, 2]
                        cy = M_inv[1, 0] * M["m10"] / M["m00"] + M_inv[1, 1] * M["m01"] / M["m00"] + M_inv[1, 2]
                    else:
                        cx, cy = float(ax), float(ay)
                else:
                    cnt_offset = cnt + [ax, ay]
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        cx = M["m10"] / M["m00"] + ax
                        cy = M["m01"] / M["m00"] + ay
                    else:
                        cx, cy = float(ax), float(ay)
                
                raw_results.append({
                    "id": 0,
                    "x": cx,
                    "y": cy,
                    "area": area,
                    "contour": cnt_offset
                })

        # 5. Sorting
        mode = self.params["sort_mode"]
        if mode == "Area Ascending":
            raw_results.sort(key=lambda x: x["area"])
        elif mode == "Area Descending":
            raw_results.sort(key=lambda x: x["area"], reverse=True)
        elif mode == "X Ascending":
            raw_results.sort(key=lambda x: x["x"])
        elif mode == "Y Ascending":
            raw_results.sort(key=lambda x: x["y"])

        # Create final ToolResults
        results = ToolResults()
        valid_contours = []
        for i, res_dict in enumerate(raw_results):
            res = BlobResult(id=i, x=res_dict["x"], y=res_dict["y"], area=res_dict["area"])
            results.append(res)
            valid_contours.append(res_dict["contour"])

        # 6. Drawing Output
        is_roi_only = self.params["output_mode"] == "ROI Only"
        vision_img = VisionImage(img.data, x=img.x, y=img.y, angle=img.angle)

        # Inherit overlays from input image (preserves their original source_tool/ref_id)
        vision_img.inherit_overlays_from(img)
        propagate_3d(img, vision_img, regenerate=False)

        # Use default overlay layer for all drawing — ensure source_tool is ours
        layer = vision_img.get_overlay_layer("default")
        layer.source_tool = self.name
        # Clear inherited elements from this tool to prevent accumulation
        layer.elements = [e for e in layer.elements if e.source_tool != self.name]

        # Draw search ROI overlay if enabled and ROI is smaller than full image
        is_full_img = roi["is_full_img"]
        if self.params.get("show_search_roi", True) and aw > 0 and ah > 0 and not is_full_img:
            abs_angle = roi.get("abs_angle", 0.0)
            M_inv = roi.get("M_inv", None)
            if M_inv is not None:
                cx_w = aw / 2.0
                cy_w = ah / 2.0
                abs_cx = M_inv[0, 0] * cx_w + M_inv[0, 1] * cy_w + M_inv[0, 2]
                abs_cy = M_inv[1, 0] * cx_w + M_inv[1, 1] * cy_w + M_inv[1, 2]
                rx = abs_cx - aw / 2.0
                ry = abs_cy - ah / 2.0
            else:
                rx, ry = float(ax), float(ay)
            layer.add_roi([float(rx), float(ry), float(aw), float(ah), float(abs_angle)], color=(255, 128, 0), thickness=2)
        
        # Add overlays based on checkbox states
        if len(valid_contours) > 0:
            for i, res in enumerate(results):
                cnt = valid_contours[i]
                if cnt is None or cnt.size == 0:
                    continue

                ref = f"blob_{res.id}"
                # Show Boundaries - contours
                if self.params["show_contours"]:
                    layer.add_contour(cnt.tolist(), color=(0, 255, 0), thickness=3, ref_id=ref)

                # Show Centroids - points
                if self.params["show_centroids"]:
                    layer.add_point((res.x, res.y), color=(0, 0, 255), radius=5, ref_id=ref)

                # Show Labels - text
                if self.params["show_labels"]:
                    layer.add_text(f"#{res.id} A:{int(res.area)}", position=(res.x + 10, res.y), color=(255, 255, 0), font_scale=0.6, thickness=2)

        if is_roi_only:
            self.outputs["output image"] = create_roi_only_output(
                roi, roi_src, vision_img, self.name, ax, ay,
            )
        else:
            self.outputs["output image"] = vision_img
        self.outputs["binary mask"] = VisionImage(mask_full)
        self.outputs["count"] = results.Count
        self.outputs["results"] = results
        self._copy_overlays_to_input(vision_img, img)
        self.data_changed.emit()

    def serialize(self, job_manager=None):
        data = super().serialize(job_manager)
        data["type"] = "BlobDetector"
        return data

    def get_config_widget(self):
        from .config import BlobDetectorConfig
        return BlobDetectorConfig(self)
