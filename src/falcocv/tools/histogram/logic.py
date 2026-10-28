import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_base import ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi


class HistogramResult(ToolResult):
    def __init__(self, mean, std_dev, min_val, max_val, median_val, bins):
        super().__init__(
            mean=float(mean), std_dev=float(std_dev),
            min_val=int(min_val), max_val=int(max_val),
            median_val=float(median_val), bins=int(bins)
        )


@register_tool(prefix="[HIST]", default_name="Histogram", group="Vision Tools")
class HistogramTool(BaseTool):
    def __init__(self, name="Histogram"):
        super().__init__(name)
        self.params = {
            "bin_mode": "Auto",
            "num_bins": 256,
            "roi": [0, 0, 0, 0, 0],
            "output_width": 800,
            "output_height": 600,
            "enable_output_image": True,
            "copy_overlays_to_input": True,
        }
        self.inputs = {"input image": None}
        self.outputs = {
            "output image": None, 
            "results": ToolResults(),
        }

    def run(self, advance=False):
        img_input = self.inputs.get("input image")
        if img_input is None:
            self.outputs["output image"] = None
            self.outputs["results"] = ToolResults()
            self.data_changed.emit()
            return

        v_img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)

        # Extract ROI using shared utility
        roi = extract_roi(v_img, self.params["roi"])
        roi_src = roi["roi_src"]

        gray = cv2.cvtColor(roi_src, cv2.COLOR_BGR2GRAY) if roi_src.ndim == 3 else roi_src

        # Determine number of bins and range
        float_range = None
        if self.params["bin_mode"] == "Auto":
            if gray.dtype.kind == "f":
                finite = gray[np.isfinite(gray)]
                lo = float(finite.min()) if finite.size else 0.0
                hi = float(finite.max()) if finite.size else 1.0
                if hi <= lo:
                    hi = lo + 1.0
                bins = 256
                hist_range = [lo, hi]
                float_range = (lo, hi)
            elif gray.dtype == np.uint8:
                bins = 256
            elif gray.dtype == np.uint16:
                max_val = np.iinfo(gray.dtype).max
                if max_val == 1023:  # 10-bit
                    bins = 1024
                elif max_val == 4095:  # 12-bit
                    bins = 4096
                elif max_val == 16383:  # 14-bit
                    bins = 16384
                elif max_val == 65535:  # 16-bit
                    bins = 65536
                else:
                    bins = 256  # fallback
            else:
                bins = 256
            if gray.dtype.kind != "f":
                hist_range = [0, np.iinfo(gray.dtype).max + 1]
        else:
            bins = self.params["num_bins"]
            hist_range = [0, 256]  # Assuming 8-bit for fixed mode

        # Compute histogram first so we can derive median from it
        hist_src = np.nan_to_num(gray, nan=0.0) if gray.dtype.kind == "f" else gray
        hist = cv2.calcHist([hist_src], [0], None, [bins], hist_range)
        hist = np.squeeze(hist)

        # Compute statistics
        stats_src = gray[np.isfinite(gray)] if gray.dtype.kind == "f" else gray
        if stats_src.size == 0:
            stats_src = np.zeros(1, dtype=gray.dtype)
        mean_val = np.mean(stats_src)
        std_val = np.std(stats_src)
        min_val = np.min(stats_src)
        max_val = np.max(stats_src)
        # Fast median from cumulative histogram (O(bins)) instead of sorting O(n log n)
        cumsum = np.cumsum(hist.astype(np.int64))
        median_idx = np.searchsorted(cumsum, int(cumsum[-1]) // 2)
        if float_range is not None:
            lo, hi = float_range
            median_val = lo + (float(median_idx) + 0.5) / bins * (hi - lo)
        else:
            median_val = float(median_idx)

        results = ToolResults()
        results.append(HistogramResult(mean_val, std_val, min_val, max_val, median_val, bins))

        if self.params.get("enable_output_image", True):
            hist_img = self._draw_histogram(hist, mean_val, std_val, min_val, max_val, median_val,
                                            self.params["output_width"], self.params["output_height"])
            self.outputs["output image"] = VisionImage(hist_img)
        else:
            self.outputs["output image"] = None
        self.outputs["results"] = results
        out_vimg = self.outputs.get("output image")
        if out_vimg is not None:
            self._copy_overlays_to_input(out_vimg, v_img)
        self.data_changed.emit()

    def _draw_histogram(self, hist, mean_val, std_val, min_val, max_val, median_val, width=800, height=600):
        """Render a professional histogram using OpenCV primitives."""
        sf = min(width / 800, height / 600)
        margin_l, margin_r, margin_t, margin_b = int(70 * sf), int(30 * sf), int(40 * sf), int(80 * sf)
        plot_w = width - margin_l - margin_r
        plot_h = height - margin_t - margin_b

        _BG = (30, 30, 30)
        _BAR = (214, 156, 86)
        _BORDER = (60, 60, 60)
        _SUCCESS = (176, 201, 78)
        _STAT = (120, 145, 206)
        _TEXT = (212, 212, 212)
        _TEXT_MUTED = (128, 128, 128)
        _PANEL_BG = (37, 37, 38)

        img = np.full((height, width, 3), _BG, dtype=np.uint8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        title_font = cv2.FONT_HERSHEY_DUPLEX
        fscale = max(0.3, 0.4 * sf)
        tscale = max(0.4, 0.6 * sf)
        thick = 1

        # ── Grid lines ──
        for i in range(5):
            gy = int(margin_t + plot_h * i / 4)
            cv2.line(img, (margin_l, gy), (margin_l + plot_w, gy), _BORDER, 1)

        # ── Downsample bins if needed ──
        bins = len(hist)
        if bins > plot_w:
            downsampled = np.zeros(plot_w, dtype=hist.dtype)
            for i in range(plot_w):
                start = int(i * bins / plot_w)
                end = int((i + 1) * bins / plot_w)
                downsampled[i] = hist[start:end].max()
            hist = downsampled

        # ── Normalize ──
        h_max = hist.max() if hist.max() > 0 else 1
        hist_norm = hist / h_max * plot_h

        # ── Draw histogram bars ──
        bar_w = max(1, plot_w // len(hist_norm))
        for i, val in enumerate(hist_norm):
            x = margin_l + i * bar_w
            y = int(margin_t + plot_h - val)
            h_bar = max(1, int(val))
            cv2.rectangle(img, (x, y), (x + bar_w, y + h_bar), _BAR, -1)

        # ── Mean line ──
        data_range = max_val - min_val if max_val > min_val else 256
        mean_x = int(margin_l + (mean_val - min_val) / data_range * plot_w)
        cv2.line(img, (mean_x, margin_t), (mean_x, margin_t + plot_h), _SUCCESS, max(1, int(2 * sf)))

        # ── Median line ──
        median_x = int(margin_l + (median_val - min_val) / data_range * plot_w)
        cv2.line(img, (median_x, margin_t), (median_x, margin_t + plot_h), _STAT, max(1, int(2 * sf)))

        # ── Axis labels ──
        min_lbl = f"{int(min_val)}"
        max_lbl = f"{int(max_val)}"
        label_y = margin_t + plot_h + int(20 * sf)
        cv2.putText(img, min_lbl, (margin_l, label_y), font, fscale, _TEXT_MUTED, thick)
        max_size = cv2.getTextSize(max_lbl, font, fscale, thick)[0]
        cv2.putText(img, max_lbl, (margin_l + plot_w - max_size[0], label_y), font, fscale, _TEXT_MUTED, thick)

        # ── Title ──
        title_size = cv2.getTextSize("Histogram", title_font, tscale, thick)[0]
        cv2.putText(img, "Histogram", ((width - title_size[0]) // 2, int(24 * sf)),
                    title_font, tscale, _TEXT, thick)

        # ── Statistics panel ──
        pw = int(180 * sf)
        ph = int(160 * sf)
        px = int(8 * sf)
        py = int(40 * sf)
        cv2.rectangle(img, (px, py), (px + pw, py + ph), _PANEL_BG, -1)

        stat_lines = [
            ("Mean",   f"{mean_val:.1f}",  _SUCCESS),
            ("Median", f"{median_val:.1f}", _STAT),
            ("StdDev", f"{std_val:.1f}",   _TEXT_MUTED),
            ("Min",    f"{int(min_val)}",   _TEXT_MUTED),
            ("Max",    f"{int(max_val)}",   _TEXT_MUTED),
            ("Bins",   f"{bins}",           _TEXT_MUTED),
        ]
        sx = int(18 * sf)
        sy = int(58 * sf) + 10
        spacing = int(22 * sf)
        for label, value, clr in stat_lines:
            cv2.putText(img, label, (sx, sy), font, fscale, _TEXT, thick)
            vsize = cv2.getTextSize(value, font, fscale, thick)[0]
            cv2.putText(img, value, (px + pw - vsize[0] - int(8 * sf), sy), font, fscale, clr, thick)
            sy += spacing

        # ── Legend ──
        leg_x = width - int(150 * sf)
        leg_y = margin_t + int(12 * sf)
        leg_h = int(16 * sf)
        cv2.line(img, (leg_x, int(leg_y + leg_h / 2)), (leg_x + int(20 * sf), int(leg_y + leg_h / 2)),
                 _SUCCESS, max(1, int(2 * sf)))
        cv2.putText(img, "Mean", (leg_x + int(26 * sf), int(leg_y + leg_h / 2 + 6)),
                    font, fscale, _TEXT_MUTED, thick)

        leg_y += int(24 * sf)
        cv2.line(img, (leg_x, int(leg_y + leg_h / 2)), (leg_x + int(20 * sf), int(leg_y + leg_h / 2)),
                 _STAT, max(1, int(2 * sf)))
        cv2.putText(img, "Median", (leg_x + int(26 * sf), int(leg_y + leg_h / 2 + 6)),
                    font, fscale, _TEXT_MUTED, thick)

        return img

    def get_config_widget(self):
        from .config import HistogramConfig
        return HistogramConfig(self)


