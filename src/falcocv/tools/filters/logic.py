# Logic module for FiltersTool.

import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi, create_roi_only_output, composite_roi_to_full
from falcocv.core.pointcloud_utils import propagate_3d
from falcocv.core.config import ConfigManager

# ---------------------------------------------------------------------------
# Filter definitions
# ---------------------------------------------------------------------------

FILTER_DEFS = {
    "gaussian_blur": {
        "label": "Gaussian Blur",
        "params": [
            {"key": "kernel_size", "label": "Kernel Size", "type": "int", "default": 5, "min": 3, "step": 2},
            {"key": "sigma", "label": "Sigma", "type": "float", "default": 1.0, "min": 0.001},
        ],
    },
    "median_blur": {
        "label": "Median Blur",
        "params": [
            {"key": "kernel_size", "label": "Kernel Size", "type": "int", "default": 5, "min": 3, "step": 2},
        ],
    },
    "bilateral": {
        "label": "Bilateral Filter",
        "params": [
            {"key": "d", "label": "Diameter", "type": "int", "default": 9, "min": 1, "step": 2},
            {"key": "sigma_color", "label": "Sigma Color", "type": "float", "default": 75.0, "min": 1.0},
            {"key": "sigma_space", "label": "Sigma Space", "type": "float", "default": 75.0, "min": 1.0},
        ],
    },
    "sharpen": {
        "label": "Sharpen",
        "params": [
            {"key": "amount", "label": "Amount", "type": "float", "default": 1.0, "min": 0.0},
        ],
    },
    "sobel": {
        "label": "Sobel",
        "params": [
            {"key": "mode", "label": "Mode", "type": "choice", "default": "magnitude",
             "choices": [("x", "X"), ("y", "Y"), ("magnitude", "Magnitude"), ("angle", "Angle")]},
            {"key": "kernel_size", "label": "Kernel Size", "type": "int", "default": 3, "min": 1, "step": 2},
        ],
    },
    "canny": {
        "label": "Canny Edge",
        "params": [
            {"key": "threshold1", "label": "Low Threshold", "type": "int", "default": 50, "min": 0, "max": 255, "step": 1},
            {"key": "threshold2", "label": "High Threshold", "type": "int", "default": 150, "min": 0, "max": 255, "step": 1},
            {"key": "aperture_size", "label": "Aperture", "type": "int", "default": 3, "min": 3, "max": 7, "step": 2},
        ],
    },
    "morphology": {
        "label": "Morphology",
        "params": [
            {"key": "operation", "label": "Operation", "type": "choice", "default": "dilate",
             "choices": [("erode", "Erode"), ("dilate", "Dilate"), ("open", "Open"), ("close", "Close"), ("gradient", "Gradient")]},
            {"key": "shape", "label": "Kernel Shape", "type": "choice", "default": "rect",
             "choices": [("rect", "Rectangle"), ("ellipse", "Ellipse"), ("cross", "Cross")]},
            {"key": "kernel_size", "label": "Kernel Size", "type": "int", "default": 3, "min": 3, "step": 2},
            {"key": "iterations", "label": "Iterations", "type": "int", "default": 1, "min": 1},
        ],
    },
    "threshold": {
        "label": "Threshold",
        "params": [
            {"key": "value", "label": "Value", "type": "int", "default": 128, "min": 0, "max": 255, "step": 1},
            {"key": "inverted", "label": "Inverted", "type": "bool", "default": False},
        ],
    },
    "clahe": {
        "label": "CLAHE",
        "params": [
            {"key": "clip_limit", "label": "Clip Limit", "type": "float", "default": 2.0, "min": 0.1},
            {"key": "grid_size", "label": "Grid Size", "type": "int", "default": 8, "min": 2},
        ],
    },
    "normalize": {
        "label": "Normalize",
        "params": [
            {"key": "min", "label": "Min", "type": "int", "default": 0, "min": 0, "max": 255, "step": 1},
            {"key": "max", "label": "Max", "type": "int", "default": 255, "min": 0, "max": 255, "step": 1},
        ],
    },
    "equalize": {
        "label": "Equalize Histogram",
        "params": [],
    },
    "color_conversion": {
        "label": "Color Conversion",
        "params": [
            {"key": "mode", "label": "Mode", "type": "choice", "default": "gray",
             "choices": [
                 ("gray", "Grayscale"),
                 ("red", "Red Channel"),
                 ("green", "Green Channel"),
                 ("blue", "Blue Channel"),
                 ("hue", "Hue (HSV)"),
                 ("sat", "Saturation (HSV)"),
                 ("val", "Value (HSV)"),
                 ("lightness", "Lightness (HLS)"),
             ]},
        ],
    },
    "invert": {
        "label": "Invert",
        "params": [],
    },
    "resize": {
        "label": "Resize",
        "params": [
            {"key": "mode", "label": "Mode", "type": "choice", "default": "factor",
             "choices": [("factor", "Factor"), ("width", "Width"), ("height", "Height"), ("longest", "Longest Side")]},
            {"key": "value", "label": "Value", "type": "float", "default": 0.5, "min": 0.01, "step": 0.01},
            {"key": "interpolation", "label": "Interpolation", "type": "choice", "default": "linear",
             "choices": [("nearest", "Nearest"), ("linear", "Linear"), ("cubic", "Cubic"), ("lanczos", "Lanczos")]},
        ],
    },
    "rotate_flip": {
        "label": "Rotate / Flip",
        "params": [
            {"key": "mode", "label": "Mode", "type": "choice", "default": "90cw",
             "choices": [
                 ("90cw", "Rotate 90° CW"),
                 ("90ccw", "Rotate 90° CCW"),
                 ("180", "Rotate 180°"),
                 ("flip_h", "Flip Horizontal"),
                 ("flip_v", "Flip Vertical"),
                 ("flip_hv", "Flip Both"),
             ]},
        ],
    },
}

FILTER_TYPE_ORDER = ["gaussian_blur", "median_blur", "bilateral", "sharpen", "sobel", "canny", "morphology", "threshold", "clahe", "normalize", "equalize", "color_conversion", "invert", "resize", "rotate_flip"]


def _normalize_to_u8(img):
    """Normalize a float image to uint8 [0, 255] (C++ parity for depth maps)."""
    out = cv2.normalize(img, None, 0.0, 255.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    return out.astype(np.uint8)



def _make_default_filter(ftype):
    f = {"enabled": True, "type": ftype}
    defs = FILTER_DEFS.get(ftype)
    if defs:
        for p in defs["params"]:
            f[p["key"]] = p["default"]
    return f


def _filter_summary(f):
    ftype = f.get("type", "")
    defs = FILTER_DEFS.get(ftype)
    if not defs:
        return ftype
    parts = []
    for p in defs["params"]:
        val = f.get(p["key"], p["default"])
        if p["type"] == "float":
            parts.append(f"{val:.1f}")
        elif p["type"] == "bool":
            parts.append("INV" if val else "NORM")
        elif p["type"] == "choice":
            choices = dict(p.get("choices", []))
            parts.append(choices.get(val, str(val)))
        else:
            parts.append(str(val))
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# Standalone filter application
# ---------------------------------------------------------------------------

def apply_filter(img, ftype, params):
    if ftype == "gaussian_blur":
        ks = int(params.get("kernel_size", 5))
        if ks % 2 == 0:
            ks += 1
        return cv2.GaussianBlur(img, (ks, ks), params.get("sigma", 1.0))

    if ftype == "median_blur":
        ks = int(params.get("kernel_size", 5))
        if ks % 2 == 0:
            ks += 1
        return cv2.medianBlur(img, ks)

    if ftype == "bilateral":
        return cv2.bilateralFilter(
            img,
            int(params.get("d", 9)),
            params.get("sigma_color", 75.0),
            params.get("sigma_space", 75.0),
        )

    if ftype == "sharpen":
        amount = params.get("amount", 1.0)
        kernel = np.array([
            [0, -amount, 0],
            [-amount, 1 + 4 * amount, -amount],
            [0, -amount, 0],
        ], dtype=np.float32)
        return cv2.filter2D(img, -1, kernel)

    if ftype == "sobel":
        ks = max(1, int(params.get("kernel_size", 3)) | 1)
        ks = min(ks, 31)
        mode = params.get("mode", "magnitude")
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if mode == "x":
            g = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ks)
        elif mode == "y":
            g = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ks)
        elif mode == "angle":
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ks)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ks)
            g = np.degrees(np.arctan2(gy, gx))
        else:
            gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ks)
            gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ks)
            g = cv2.magnitude(gx, gy)

        if mode == "angle":
            g = cv2.normalize(g + 180, None, 0, 255, cv2.NORM_MINMAX)
        else:
            g = cv2.normalize(np.abs(g), None, 0, 255, cv2.NORM_MINMAX)
        g = g.astype(np.uint8)
        if len(img.shape) == 3:
            g = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        return g

    if ftype == "canny":
        t1 = int(params.get("threshold1", 50))
        t2 = int(params.get("threshold2", 150))
        ap = max(3, int(params.get("aperture_size", 3)) | 1)
        ap = min(ap, 7)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if gray.dtype == np.float32:
            gray = _normalize_to_u8(gray)
        edges = cv2.Canny(gray, t1, t2, apertureSize=ap)
        if len(img.shape) == 3:
            edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges

    if ftype == "morphology":
        op_map = {
            "erode": cv2.MORPH_ERODE, "dilate": cv2.MORPH_DILATE,
            "open": cv2.MORPH_OPEN, "close": cv2.MORPH_CLOSE,
            "gradient": cv2.MORPH_GRADIENT,
        }
        shape_map = {
            "rect": cv2.MORPH_RECT, "ellipse": cv2.MORPH_ELLIPSE, "cross": cv2.MORPH_CROSS,
        }
        ks = int(params.get("kernel_size", 3))
        if ks % 2 == 0:
            ks += 1
        op = op_map.get(params.get("operation", "dilate"), cv2.MORPH_DILATE)
        shape = shape_map.get(params.get("shape", "rect"), cv2.MORPH_RECT)
        kernel = cv2.getStructuringElement(shape, (ks, ks))
        iterations = max(1, int(params.get("iterations", 1)))
        return cv2.morphologyEx(img, op, kernel, iterations=iterations)

    if ftype == "threshold":
        val = int(params.get("value", 128))
        inv = params.get("inverted", False)
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        _, result = cv2.threshold(gray, val, 255, cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY)
        if len(img.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result

    if ftype == "clahe":
        clip = params.get("clip_limit", 2.0)
        grid = max(2, int(params.get("grid_size", 8)))
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if gray.dtype == np.float32:
            gray = _normalize_to_u8(gray)
        clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
        result = clahe.apply(gray)
        if len(img.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result

    if ftype == "equalize":
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img
        if gray.dtype == np.float32:
            gray = _normalize_to_u8(gray)
        result = cv2.equalizeHist(gray)
        if len(img.shape) == 3:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        return result

    if ftype == "normalize":
        min_val = int(params.get("min", 0))
        max_val = int(params.get("max", 255))
        return cv2.normalize(img, None, min_val, max_val, cv2.NORM_MINMAX)

    if ftype == "color_conversion":
        mode = params.get("mode", "gray")
        if mode == "gray":
            if len(img.shape) == 3:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            else:
                gray = img
            if len(img.shape) == 3:
                return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            return gray
        if len(img.shape) < 3:
            return img
        b, g, r = cv2.split(img)
        if mode == "red":
            return cv2.cvtColor(r, cv2.COLOR_GRAY2BGR)
        if mode == "green":
            return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
        if mode == "blue":
            return cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        if mode == "hue":
            return cv2.cvtColor(h, cv2.COLOR_GRAY2BGR)
        if mode == "sat":
            return cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
        if mode == "val":
            return cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)
        if mode == "lightness":
            hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
            l_ch = hls[:, :, 1]
            return cv2.cvtColor(l_ch, cv2.COLOR_GRAY2BGR)
        return img

    if ftype == "invert":
        if img.dtype == np.float32:
            min_v = float(np.nanmin(img))
            max_v = float(np.nanmax(img))
            return (max_v + min_v) - img
        return cv2.bitwise_not(img)

    if ftype == "resize":
        mode = params.get("mode", "factor")
        value = params.get("value", 0.5)
        interp_map = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
                      "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}
        interp = interp_map.get(params.get("interpolation", "linear"), cv2.INTER_LINEAR)
        h, w = img.shape[:2]
        if mode == "factor":
            nw, nh = int(w * value), int(h * value)
        elif mode == "width":
            nw = int(value)
            nh = int(h * nw / w) if w else h
        elif mode == "height":
            nh = int(value)
            nw = int(w * nh / h) if h else w
        else:  # longest side
            scale = value / max(w, h)
            nw, nh = int(w * scale), int(h * scale)
        nw, nh = max(nw, 1), max(nh, 1)
        return cv2.resize(img, (nw, nh), interpolation=interp)

    if ftype == "rotate_flip":
        mode = params.get("mode", "90cw")
        if mode == "90cw":
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if mode == "90ccw":
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if mode == "180":
            return cv2.rotate(img, cv2.ROTATE_180)
        if mode == "flip_h":
            return cv2.flip(img, 1)
        if mode == "flip_v":
            return cv2.flip(img, 0)
        if mode == "flip_hv":
            return cv2.flip(img, -1)
        return img

    return img


# ---------------------------------------------------------------------------
# FiltersTool
# ---------------------------------------------------------------------------

@register_tool(prefix="[FILT]", default_name="Filters", group="Preprocess")
class FiltersTool(BaseTool):
    def __init__(self, name="Filters"):
        super().__init__(name)
        self.inputs = {"input image": None}
        self.outputs = {"output image": None}
        self.params = {
            "filters": [],
            "roi": [0, 0, 0, 0, 0],
            "output_mode": "Full Image",
            "show_overlays": True,
        }

    def run(self, advance=False):
        img = self.inputs.get("input image")
        if img is None:
            self.outputs["output image"] = None
            self.data_changed.emit()
            return

        v_img = img if isinstance(img, VisionImage) else VisionImage.from_array(img)

        roi_p = self.params["roi"]
        x, y, w, h = roi_p[0], roi_p[1], roi_p[2], roi_p[3]
        a = roi_p[4] if len(roi_p) > 4 else 0

        roi = extract_roi(v_img, self.params["roi"])
        working = roi["roi_src"]
        input_was_float = working.dtype.kind == "f"
        if input_was_float:
            working = np.nan_to_num(working, nan=0.0)

        if w <= 0 or h <= 0:
            h_img, w_img = v_img.data.shape[:2]
            x, y, w, h = 0, 0, w_img, h_img
            a = 0

        for f in self.params.get("filters", []):
            if not f.get("enabled", True):
                continue
            ftype = f.get("type")
            fparams = {k: v for k, v in f.items() if k not in ("enabled", "type")}
            working = apply_filter(working, ftype, fparams)

        # Preserve float32 (depth -> depth) as the C++ compute did.
        if input_was_float and working.dtype != np.float32:
            working = working.astype(np.float32)

        if self.params.get("output_mode") == "ROI Only":
            out_vimg = create_roi_only_output(
                roi, working, v_img, self.name, roi["ax"], roi["ay"],
            )
            propagate_3d(v_img, out_vimg, regenerate=True)
            self.outputs["output image"] = out_vimg
        else:
            raw_img = v_img.data
            is_full_img = (w >= raw_img.shape[1] and h >= raw_img.shape[0])

            if is_full_img:
                out_vimg = VisionImage(working, v_img.x, v_img.y, v_img.angle,
                                        v_img.scale_x, v_img.scale_y, v_img.scale_z, v_img.offset_z)
                out_vimg.inherit_overlays_from(v_img)
                propagate_3d(v_img, out_vimg, regenerate=True)
                self.outputs["output image"] = out_vimg
            else:
                show_roi = self.params.get("show_overlays", True)
                out_vimg = composite_roi_to_full(working, v_img, self.params["roi"], draw_roi=show_roi, source_tool=self.name)
                propagate_3d(v_img, out_vimg, regenerate=True)
                self.outputs["output image"] = out_vimg

        self.data_changed.emit()

    def get_config_widget(self):
        from .config import FiltersConfig
        return FiltersConfig(self)

    def serialize(self, job_manager=None):
        data = super().serialize(job_manager)
        data["type"] = "FiltersTool"
        return data
