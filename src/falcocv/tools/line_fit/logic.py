"""Line Fit — caliper-array or point-based robust line fitting."""

from falcocv.core.fitting import fit_line
from falcocv.core.port_types import IMAGE, RESULTS
from falcocv.core.tool_base import BaseTool, ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.tools._fit_support import (
    collect_overlay_points,
    drawable_roi,
    filter_by_roi,
    prepare_output,
    roi_to_image,
    scan_line_calipers,
)


class LineFitResult(ToolResult):
    def __init__(
        self,
        id,
        x1,
        y1,
        x2,
        y2,
        angle,
        length,
        rms,
        inliers,
        score,
        calipers=0,
        edge_points=0,
        outliers=0,
    ):
        super().__init__(
            id=id,
            type="line",
            value=angle,
            unit="deg",
            x1=x1,
            y1=y1,
            x2=x2,
            y2=y2,
            angle=angle,
            length=length,
            rms=rms,
            inliers=inliers,
            score=score,
            calipers=calipers,
            edge_points=edge_points,
            outliers=outliers,
        )


@register_tool(prefix="[LFIT]", default_name="Line Fit", group="Inspection")
class LineFit(BaseTool):
    input_port_types = {"input image": IMAGE}
    output_port_types = {"output image": IMAGE, "results": RESULTS}

    def __init__(self, name="Line Fit"):
        super().__init__(name)
        self.params = {
            "roi": [0, 0, 0, 0, 0],
            "point_source": "calipers",
            "caliper_count": 9,
            "caliper_width": 5.0,
            "caliper_length": 0.0,
            "caliper_edge_polarity": "Any",
            "caliper_selector": "Best",
            "caliper_threshold": 5.0,
            "caliper_sigma": 1.0,
            "caliper_subpixel": True,
            "use_overlay_points": True,
            "use_contours": False,
            "robust": True,
            "max_residual_px": 0.0,
            "max_outlier_percent": 20.0,
            "iterations": 200,
            "min_points": 2,
            "min_results": 1,
            "show_search_roi": True,
            "show_calipers": True,
            "show_fit_line": True,
            "show_inliers": True,
            "show_outliers": False,
            "show_labels": True,
            "copy_overlays_to_input": True,
        }
        self.inputs = {"input image": None}
        self.outputs = {
            "output image": None,
            "count": 0,
            "results": ToolResults(),
        }

    def run(self, advance=False):
        try:
            self._run()
        except Exception as exc:  # noqa: BLE001 - surface the failure in the tool
            self.outputs["count"] = 0
            self.outputs["results"] = ToolResults()
            self.set_error(f"Line fit failed: {type(exc).__name__}: {exc}")
        finally:
            self.data_changed.emit()

    def _caliper_kwargs(self):
        return {
            "count": int(self.params.get("caliper_count", 9) or 9),
            "length": float(self.params.get("caliper_length", 0.0) or 0.0),
            "width": float(self.params.get("caliper_width", 5.0) or 5.0),
            "polarity": str(self.params.get("caliper_edge_polarity", "Any")),
            "threshold": float(self.params.get("caliper_threshold", 5.0) or 0.0),
            "sigma": float(self.params.get("caliper_sigma", 1.0) or 0.0),
            "selector": str(self.params.get("caliper_selector", "Best")),
            "subpixel": bool(self.params.get("caliper_subpixel", True)),
        }

    def _outlier_fraction(self):
        percent = float(self.params.get("max_outlier_percent", 20.0) or 0.0)
        return min(max(percent, 0.0), 50.0) / 100.0

    def _run(self):
        self.outputs["output image"] = None
        self.outputs["count"] = 0
        self.outputs["results"] = ToolResults()

        image_input = self.inputs.get("input image")
        if image_input is None:
            self.set_error("Input image is missing")
            return

        img, vision_img, layer = prepare_output(image_input, self.name)
        roi_image = roi_to_image(img, self.params.get("roi"))

        source = str(self.params.get("point_source", "calipers") or "calipers")
        scan = None
        total_calipers = 0
        if source == "calipers":
            if roi_image is None:
                self.outputs["output image"] = vision_img
                self._copy_overlays_to_input(vision_img, img)
                self.set_error(
                    "Line Fit needs a search ROI: place it over the edge, with "
                    "its long side along the line, so the calipers can scan it"
                )
                return
            scan = scan_line_calipers(img.data, roi_image, **self._caliper_kwargs())
            points = scan.points
            total_calipers = scan.total
        else:
            points = collect_overlay_points(
                img,
                use_points=bool(self.params.get("use_overlay_points", True)),
                use_contours=bool(self.params.get("use_contours", False)),
                exclude_source=self.name,
            )
            points = filter_by_roi(points, roi_image)

        minimum = max(2, int(self.params.get("min_points", 2) or 2))
        if len(points) < minimum:
            self._draw_search_roi(layer, roi_image, vision_img.data.shape)
            self._draw_calipers(layer, scan)
            self.outputs["output image"] = vision_img
            self._copy_overlays_to_input(vision_img, img)
            if scan is not None:
                self.set_error(
                    f"Caliper array found {len(points)} edge points in "
                    f"{total_calipers} calipers; line fit needs at least "
                    f"{minimum}. Check the ROI, polarity and edge threshold"
                )
            else:
                self.set_error(
                    f"Line fit needs at least {minimum} points; found {len(points)}"
                )
            return

        fit = fit_line(
            points,
            robust=bool(self.params.get("robust", True)),
            threshold=float(self.params.get("max_residual_px", 0.0) or 0.0),
            iterations=int(self.params.get("iterations", 200) or 200),
            max_outlier_fraction=self._outlier_fraction(),
        )
        if fit is None:
            self._draw_search_roi(layer, roi_image, vision_img.data.shape)
            self._draw_calipers(layer, scan)
            self.outputs["output image"] = vision_img
            self._copy_overlays_to_input(vision_img, img)
            self.set_error("Line fit is degenerate: the points are coincident")
            return

        if fit.cap_exceeded:
            percent = float(self.params.get("max_outlier_percent", 20.0) or 0.0)
            allowed = int(round(len(points) * self._outlier_fraction()))
            self._draw_search_roi(layer, roi_image, vision_img.data.shape)
            self._draw_calipers(layer, scan)
            self._draw_rejected(layer, points, fit)
            self.outputs["output image"] = vision_img
            self._copy_overlays_to_input(vision_img, img)
            self.set_error(
                f"Line fit rejected {fit.outliers} of {len(points)} points, "
                f"above the {percent:.0f}% outlier limit ({allowed} allowed); "
                "check the ROI, polarity and edge threshold, or raise Max outliers"
            )
            return

        spread = (points - fit.point) @ fit.direction
        start = fit.point + fit.direction * float(spread.min())
        end = fit.point + fit.direction * float(spread.max())
        length = float(spread.max() - spread.min())

        self._draw_search_roi(layer, roi_image, vision_img.data.shape)
        self._draw_calipers(layer, scan)
        self._draw_fit(layer, start, end, points, fit)
        self.outputs["output image"] = vision_img
        self.outputs["count"] = 1
        self.outputs["results"] = ToolResults(
            [
                LineFitResult(
                    id=0,
                    x1=float(start[0]),
                    y1=float(start[1]),
                    x2=float(end[0]),
                    y2=float(end[1]),
                    angle=float(fit.angle_deg),
                    length=length,
                    rms=float(fit.rms),
                    inliers=int(fit.count),
                    score=float(fit.inlier_ratio),
                    calipers=total_calipers,
                    edge_points=int(len(points)),
                    outliers=int(len(points) - fit.count),
                )
            ]
        )
        self.fail_if_below_min_results(self.outputs["count"], "line", "lines")
        self._copy_overlays_to_input(vision_img, img)

    def _draw_search_roi(self, layer, roi_image, shape):
        if not self.params.get("show_search_roi", True):
            return
        rect = drawable_roi(roi_image, shape)
        if rect is not None:
            layer.add_roi(rect, color=(255, 128, 0), thickness=2)

    def _draw_calipers(self, layer, scan):
        if scan is None or not self.params.get("show_calipers", True):
            return
        for start, end in scan.segments:
            layer.add_line(start, end, color=(120, 120, 120), thickness=1)

    def _draw_rejected(self, layer, points, fit):
        for point in points[~fit.inliers]:
            layer.add_point(tuple(point), color=(0, 0, 255), radius=4)

    def _draw_fit(self, layer, start, end, points, fit):
        if self.params.get("show_fit_line", True):
            layer.add_line(
                tuple(start), tuple(end), color=(0, 255, 255), thickness=2, ref_id="fit"
            )
            layer.add_point(
                tuple(fit.point), color=(0, 255, 255), radius=3, ref_id="fit_center"
            )

        inliers = points[fit.inliers]
        if self.params.get("show_inliers", True) and len(inliers):
            for point in inliers:
                layer.add_point(tuple(point), color=(0, 255, 0), radius=3)
        if self.params.get("show_outliers", False):
            for point in points[~fit.inliers]:
                layer.add_point(tuple(point), color=(0, 0, 255), radius=4)

        if self.params.get("show_labels", True):
            mid = (start + end) / 2.0
            layer.add_text(
                f"{fit.angle_deg:.2f} deg  rms {fit.rms:.2f} px",
                (mid[0] + 6, mid[1] - 10),
                color=(0, 255, 255),
                font_scale=0.5,
                thickness=1,
            )

    def get_config_widget(self):
        from .config import LineFitConfig

        return LineFitConfig(self)
