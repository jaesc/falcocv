"""
ROI extraction utilities shared across vision tools.

Encapsulates the common pattern of extracting fixture-aware ROI data
from a VisionImage that is duplicated in every ROI-enabled tool.

Returns a dict with all variables needed by tool run() methods:
  roi_src, ax, ay, aw, ah, h_img, w_img, is_full_img, abs_angle, M_inv, M_fwd
"""

import cv2
import numpy as np
from falcocv.core.vision_image import VisionImage


def extract_roi(vision_img, roi_params):
    """
    Extract ROI image data and absolute coordinates from fixture-relative ROI params.

    Args:
        vision_img: VisionImage instance (input image with optional fixture)
        roi_params: list of 4 [x, y, w, h] or 5 [x, y, w, h, angle] elements
                    in fixture coordinates. Zero w/h means full image.

    Returns:
        dict with keys:
          roi_src     - np.ndarray of the ROI pixel data (or full image)
          ax, ay      - int, absolute top-left corner in image coords (clamped)
          aw, ah      - int, absolute width/height in image coords (clamped)
          h_img, w_img - int, full image dimensions
          is_full_img  - bool, True if ROI covers the entire image
          abs_angle    - float, absolute rotation angle after fixture transform
          M_inv        - 2x3 inverse affine transform (ROI-space → image-space),
                         or None when no rotation / full image
    """
    if len(roi_params) == 4:
        x, y, w, h = roi_params
        a = 0.0
    else:
        x, y, w, h, a = roi_params

    raw_img = vision_img.data
    h_img, w_img = raw_img.shape[:2]

    if w <= 0 or h <= 0:
        return {
            "roi_src": raw_img,
            "ax": 0, "ay": 0, "aw": w_img, "ah": h_img,
            "h_img": h_img, "w_img": w_img,
            "is_full_img": True,
            "abs_angle": 0.0,
            "M_inv": None,
        }

    roi_vimg = vision_img.get_roi_data([x, y, w, h, a])
    if roi_vimg is None or roi_vimg.size == 0:
        raise ValueError("Failed to extract ROI data.")

    roi_src = roi_vimg.data if isinstance(roi_vimg, VisionImage) else roi_vimg
    abs_x, abs_y, abs_w, abs_h, abs_angle = vision_img.fixture_to_image([x, y, w, h, a])

    ax = max(0, min(int(abs_x), w_img - 1))
    ay = max(0, min(int(abs_y), h_img - 1))
    aw = max(1, min(int(abs_w), w_img - ax))
    ah = max(1, min(int(abs_h), h_img - ay))

    # Compute forward and inverse affine transforms for rotated ROIs.
    # warpAffine uses M (dst→src), so M_fwd maps warped→image.
    # We want M_fwd such that image center (abs_cx, abs_cy) maps to warped center (rw/2, rh/2).
    M_fwd = None
    M_inv = None
    if abs(abs_angle) > 0.001:
        rx, ry, rw, rh, ra = x, y, w, h, a
        cx_f = rx + rw / 2.0
        cy_f = ry + rh / 2.0
        cos_a, sin_a = vision_img._cos_angle, vision_img._sin_angle
        abs_cx = vision_img.x + (cx_f * cos_a - cy_f * sin_a)
        abs_cy = vision_img.y + (cx_f * sin_a + cy_f * cos_a)

        M_rot = cv2.getRotationMatrix2D((abs_cx, abs_cy), abs_angle, 1.0)

        # Compute where image center maps to in warped space with current M_rot
        current = M_rot @ np.array([abs_cx, abs_cy, 1.0])

        # Adjust translation so that image center maps to warped center
        M_fwd = M_rot.copy()
        M_fwd[0, 2] += ((rw / 2.0) - current[0])
        M_fwd[1, 2] += ((rh / 2.0) - current[1])
        M_fwd = M_fwd.astype(np.float64)
        M_inv = cv2.invertAffineTransform(M_fwd).astype(np.float64)

    return {
        "roi_src": roi_src,
        "ax": ax, "ay": ay, "aw": aw, "ah": ah,
        "h_img": h_img, "w_img": w_img,
        "is_full_img": (ax == 0 and ay == 0 and aw >= w_img and ah >= h_img),
        "abs_angle": float(abs_angle),
        "M_fwd": M_fwd,
        "M_inv": M_inv,
    }


def create_roi_only_output(roi, working, source_vimg, tool_name, dx, dy):
    """Create ROI Only VisionImage with tool's own overlays transformed.

    Args:
        roi: dict from extract_roi().
        working: np.ndarray — the cropped/processed ROI pixels.
        source_vimg: VisionImage — source containing tool overlays.
        tool_name: str — only overlays with this source_tool are copied.
        dx, dy: float — fallback translation when no M_fwd.

    Returns:
        VisionImage with transformed overlays (ROI rectangles excluded).
    """
    from falcocv.core.overlay_layer import OverlayType

    M_fwd = roi.get("M_fwd", None)
    result = VisionImage(working)

    for ln, lyr in source_vimg.overlay_layers.items():
        new_layer = result.get_overlay_layer(ln)
        for elem in lyr.elements:
            if elem.source_tool == tool_name and elem.element_type != OverlayType.ROI:
                if M_fwd is not None:
                    new_layer.elements.append(elem.affine_transform(M_fwd))
                else:
                    new_layer.elements.append(elem.transform(-dx, -dy, 0))

    return result


def composite_roi_to_full(
    working: np.ndarray,
    v_img: VisionImage,
    roi_params: list,
    draw_roi: bool = True,
    source_tool: str = "",
) -> VisionImage:
    """Composite a processed ROI back into the full image with fixture.

    Args:
        working: processed ROI pixel data
        v_img: source VisionImage (data, fixture, overlays)
        roi_params: [x, y, w, h, a] in fixture-relative coordinates
        draw_roi: draw ROI border overlay on output
        source_tool: tool claiming the ROI overlay; without it the border would
            inherit whatever owner the layer already had

    Returns:
        VisionImage with composited result + inherited overlays
    """
    raw_img = v_img.data
    x, y, w, h = roi_params[:4]
    a = roi_params[4] if len(roi_params) > 4 else 0.0
    total_angle = v_img.angle + a

    cx_rel, cy_rel = x + w / 2.0, y + h / 2.0
    cos_a, sin_a = v_img._cos_angle, v_img._sin_angle
    abs_cx = v_img.x + (cx_rel * cos_a - cy_rel * sin_a)
    abs_cy = v_img.y + (cx_rel * sin_a + cy_rel * cos_a)

    if abs(total_angle) < 1e-9:
        # Axis-aligned in image space: paste the crop at the exact same
        # integer origin get_roi_data() used, so the round trip is lossless.
        # A rotated fixture still moves the ROI, so the origin comes from
        # fixture_to_image() and never from the fixture-relative x/y.
        abs_x, abs_y, _, _, _ = v_img.fixture_to_image([x, y, w, h, a])
        result = raw_img.copy()
        h_img, w_img = result.shape[:2]
        ix = max(0, min(int(abs_x), w_img - 1))
        iy = max(0, min(int(abs_y), h_img - 1))
        ch = min(int(working.shape[0]), h_img - iy)
        cw = min(int(working.shape[1]), w_img - ix)
        if ch > 0 and cw > 0:
            patch = working[:ch, :cw]
            if patch.ndim == 2 and result.ndim == 3:
                patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
            result[iy:iy + ch, ix:ix + cw] = patch
    else:
        M_inv = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -total_angle, 1.0)
        M_inv[0, 2] += (abs_cx - w / 2.0)
        M_inv[1, 2] += (abs_cy - h / 2.0)

        result = raw_img.copy()
        if len(result.shape) == 3 and len(working.shape) == 2:
            working = cv2.cvtColor(working, cv2.COLOR_GRAY2BGR)
        cv2.warpAffine(working, M_inv, (result.shape[1], result.shape[0]),
                       dst=result, borderMode=cv2.BORDER_TRANSPARENT)

    out = VisionImage(result, v_img.x, v_img.y, v_img.angle,
                      v_img.scale_x, v_img.scale_y,
                      v_img.scale_z, v_img.offset_z)
    out.inherit_overlays_from(v_img)

    if source_tool:
        out.get_overlay_layer("default", source_tool=source_tool)

    if draw_roi:
        out.draw_roi(
            [float(abs_cx - w / 2.0), float(abs_cy - h / 2.0),
             float(w), float(h), float(total_angle)],
            color=(255, 128, 0), thickness=2)

    return out