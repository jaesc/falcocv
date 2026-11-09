"""Circle Fit — radial caliper-array or point-based robust circle fitting."""

from falcocv.core.fitting import fit_circle
from falcocv.core.port_types import IMAGE, RESULTS
from falcocv.core.tool_base import BaseTool, ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.tools._fit_support import (
    collect_overlay_points,
    drawable_roi,
    filter_by_roi,
    prepare_output,
    roi_to_image,
    scan_radial_calipers,
)


class CircleFitResult(ToolResult):
    def __init__(
        self,
        id,
        cx,
        cy,
        radius,
        rms,
        inliers,
        score,
        calipers=0,
        edge_points=0,
        outliers=0,
    ):
        super().__init__(
            id=id,
            type="circle",
            value=radius,
            unit="px",
            cx=cx,
            cy=cy,
            center=(cx, cy),
            radius=radius,
            diameter=2.0 * radius,
            rms=rms,
            inliers=inliers,
            score=score,
            calipers=calipers,
            edge_points=edge_points,
            outliers=outliers,
        )


@register_tool(prefix="[CFIT]", default_name="Circle Fit", group="Inspection")
class CircleFit(BaseTool):
    input_port_types = {"input image": IMAGE}
    output_port_types = {"output image": IMAGE, "results": RESULTS}

    def __init__(self, name="Circle Fit"):
        super().__init__(name)
        self.params = {
            "roi": [0, 0, 0, 0, 0],
            "point_source": "calipers",
            "caliper_count": 16,
            "caliper_width": 3.0,
            "caliper_length": 0.0,
            "caliper_radius": 0.0,
            "caliper_direction": "inward",
            "caliper_edge_polarity": "Any",
            "caliper_selector": "Best",
            "caliper_threshold": 5.0,
            "caliper_sigma": 1.0,
            "caliper_subpixel": True,
            "use_overlay_points": True,
            "use_contours": True,
            "robust": True,
            "max_residual_px": 0.0,
            "max_outlier_percent": 20.0,
            "iterations": 200,
            "min_points": 3,
            "min_results": 1,
            "show_search_roi": True,
            "show_calipers": True,
            "show_fit_circle": True,
            "show_center": True,
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
            self.set_error(f"Circle fit failed: {type(exc).__name__}: {exc}")
        finally:
            self.data_changed.emit()

    def _caliper_kwargs(self):
        return {
            "count": int(self.params.get("caliper_count", 16) or 16),
            "radius": float(self.params.get("caliper_radius", 0.0) or 0.0),
            "length": float(self.params.get("caliper_length", 0.0) or 0.0),
            "direction": str(self.params.get("caliper_direction", "inward")),
            "width": float(self.params.get("caliper_width", 3.0) or 3.0),
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
                    "Circle Fit needs a search ROI: draw it around the circle, "
                    "centred on it, so the radial calipers can scan it"
                )
                return
            scan = scan_radial_calipers(img.data, roi_image, **self._caliper_kwargs())
            points = scan.points
            total_calipers = scan.total
        else:
            points = collect_overlay_points(
                img,
                use_points=bool(self.params.get("use_overlay_points", True)),
                use_contours=bool(self.params.get("use_contours", True)),
                exclude_source=self.name,
            )
            points = filter_by_roi(points, roi_image)

        minimum = max(3, int(self.params.get("min_points", 3) or 3))
        if len(points) < minimum:
            self._draw_search_roi(layer, roi_image, vision_img.data.shape)
            self._draw_calipers(layer, scan)
            self.outputs["output image"] = vision_img
            self._copy_overlays_to_input(vision_img, img)
            if scan is not None:
                self.set_error(
                    f"Caliper array found {len(points)} edge points in "
                    f"{total_calipers} calipers; circle fit needs at least "
                    f"{minimum}. Check the ROI, caliper radius/direction/length, "
                    "polarity and edge threshold"
                )
            else:
                self.set_error(
                    f"Circle fit needs at least {minimum} points; found {len(points)}"
                )
            return

        fit = fit_circle(
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
            self.set_error("Circle fit is degenerate: the points are collinear")
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
                f"Circle fit rejected {fit.outliers} of {len(points)} points, "
                f"above the {percent:.0f}% outlier limit ({allowed} allowed); "
                "check the ROI, caliper radius/direction/length, polarity "
                "and edge threshold, or raise Max outliers"
            )
            return

        self._draw_search_roi(layer, roi_image, vision_img.data.shape)
        self._draw_calipers(layer, scan)
        self._draw_fit(layer, points, fit)
        self.outputs["output image"] = vision_img
        self.outputs["count"] = 1
        self.outputs["results"] = ToolResults(
            [
                CircleFitResult(
                    id=0,
                    cx=float(fit.center[0]),
                    cy=float(fit.center[1]),
                    radius=float(fit.radius),
                    rms=float(fit.rms),
                    inliers=int(fit.count),
                    score=float(fit.inlier_ratio),
                    calipers=total_calipers,
                    edge_points=int(len(points)),
                    outliers=int(len(points) - fit.count),
                )
            ]
        )
        self.fail_if_below_min_results(self.outputs["count"], "circle", "circles")
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

    def _draw_fit(self, layer, points, fit):
        if self.params.get("show_fit_circle", True):
            layer.add_circle(
                tuple(fit.center),
                float(fit.radius),
                color=(0, 255, 255),
                thickness=2,
                ref_id="fit",
            )
        if self.params.get("show_center", True):
            layer.add_point(
                tuple(fit.center), color=(0, 255, 255), radius=4, ref_id="fit_center"
            )

        inliers = points[fit.inliers]
        if self.params.get("show_inliers", True) and len(inliers):
            for point in inliers:
                layer.add_point(tuple(point), color=(0, 255, 0), radius=3)
        if self.params.get("show_outliers", False):
            for point in points[~fit.inliers]:
                layer.add_point(tuple(point), color=(0, 0, 255), radius=4)

        if self.params.get("show_labels", True):
            cx, cy = fit.center
            layer.add_text(
                f"R {fit.radius:.2f} px  rms {fit.rms:.2f} px",
                (cx + fit.radius + 6, cy - 10),
                color=(0, 255, 255),
                font_scale=0.5,
                thickness=1,
            )

    def get_config_widget(self):
        from .config import CircleFitConfig

        return CircleFitConfig(self)
