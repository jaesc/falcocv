"""Regression tests for fixes that landed after the tools were migrated to C++.

These guard against reintroducing old bugs during the hybrid Python port:
- float32 depth must survive the filter pipeline,
- the histogram must not crash on float32/NaN input,
- tools must not accumulate their own overlays across repeated runs.
"""
import unittest

import cv2
import numpy as np

import falcocv.core  # noqa: F401
from falcocv.core.vision_image import VisionImage
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.edge_detection.logic import EdgeDetection
from falcocv.tools.filters.logic import FiltersTool
from falcocv.tools.histogram.logic import HistogramTool
from falcocv.tools.image_calculator.logic import ImageCalculatorTool
from falcocv.tools.measurement.logic import MeasurementTool


def _depth(h=120, w=160):
    depth = np.linspace(0.0, 1000.0, h * w, dtype=np.float32).reshape(h, w)
    depth[::17, ::17] = np.nan
    return depth


def _bgr(h=300, w=400):
    img = np.zeros((h, w, 3), dtype=np.uint8)
    img[:, : w // 2] = (40, 120, 220)
    img[:, w // 2:] = (200, 80, 30)
    cv2.circle(img, (w // 2, h // 2), 40, (255, 255, 255), -1)
    return img


class TestFloat32Depth(unittest.TestCase):
    def test_filters_preserve_float32(self):
        depth = _depth()
        cases = [
            ("gaussian_blur", {"kernel_size": 5, "sigma": 1.0}),
            ("median_blur", {"kernel_size": 3}),
            ("canny", {"threshold1": 50, "threshold2": 150, "aperture_size": 3}),
            ("clahe", {"clip_limit": 2.0, "grid_size": 8}),
            ("equalize", {}),
            ("invert", {}),
            ("threshold", {"value": 128, "inverted": False}),
            ("sobel", {"mode": "magnitude", "kernel_size": 3}),
        ]
        for ftype, params in cases:
            tool = FiltersTool()
            tool.inputs["input image"] = VisionImage(depth)
            tool.params["filters"] = [{"enabled": True, "type": ftype, **params}]
            tool.run()
            out = tool.outputs["output image"].data
            self.assertEqual(out.ndim, 2, ftype)
            self.assertEqual(out.dtype, np.float32, ftype)
            self.assertTrue(np.isfinite(out).all(), ftype)

    def test_histogram_accepts_float32_with_nan(self):
        tool = HistogramTool()
        tool.inputs["input image"] = VisionImage(_depth())
        tool.params["enable_output_image"] = False
        tool.run()
        result = tool.outputs["results"][0]
        self.assertTrue(np.isfinite(result.mean))
        self.assertTrue(np.isfinite(result.median_val))
        self.assertGreaterEqual(result.max_val, result.min_val)


class TestNoOverlayAccumulation(unittest.TestCase):
    def test_blob_detector(self):
        img = VisionImage(_bgr())
        tool = BlobDetector()
        tool.inputs["input image"] = img
        tool.params["threshold_mode"] = "Otsu"
        tool.params["polarity"] = "Bright Blobs"
        counts = []
        for _ in range(3):
            tool.run()
            counts.append(len(img.get_overlay_layer("default").elements))
        self.assertEqual(len(set(counts)), 1, counts)

    def test_edge_detection(self):
        img = VisionImage(_bgr())
        tool = EdgeDetection()
        tool.inputs["input image"] = img
        tool.params["roi"] = [0, 0, img.data.shape[1], img.data.shape[0], 0]
        counts = []
        for _ in range(3):
            tool.run()
            counts.append(len(img.get_overlay_layer("default").elements))
        self.assertEqual(len(set(counts)), 1, counts)

    def test_image_calculator(self):
        img = VisionImage(_bgr())
        tool = ImageCalculatorTool()
        tool.inputs["image_A"] = img
        tool.inputs["image_B"] = VisionImage(img.data.copy())
        tool.params["ops"] = [{"enabled": True, "type": "add"}]
        counts = []
        for _ in range(3):
            tool.run()
            counts.append(len(img.get_overlay_layer("default").elements))
        self.assertEqual(len(set(counts)), 1, counts)


class TestOverlayOwnership(unittest.TestCase):
    """Tools must tag their overlays with their own name.

    ``OverlayLayer.add_*()`` copies ``layer.source_tool`` into every element and
    ``BaseTool._copy_overlays_to_input()`` filters by ``element.source_tool``, so
    a tool that inherits a "default" layer created by an upstream tool must
    re-tag it before drawing, otherwise its overlays are silently dropped.
    """

    def test_get_overlay_layer_retags_inherited_layer(self):
        img = VisionImage(_bgr())
        img.get_overlay_layer("default", source_tool="Upstream").add_contour(
            [[0, 0], [10, 0], [10, 10]]
        )

        layer = img.get_overlay_layer("default", source_tool="Mine")
        self.assertEqual(layer.source_tool, "Mine")

        layer.add_text("x", (5, 5))
        self.assertEqual(layer.elements[-1].source_tool, "Mine")
        # Already existing elements keep their original owner.
        self.assertEqual(layer.elements[0].source_tool, "Upstream")

    def test_get_overlay_layer_without_source_tool_keeps_owner(self):
        img = VisionImage(_bgr())
        img.get_overlay_layer("default", source_tool="Upstream")
        self.assertEqual(img.get_overlay_layer("default").source_tool, "Upstream")

    def test_filters_roi_overlay_is_owned_by_filters(self):
        img = VisionImage(_bgr())
        img.get_overlay_layer("default", source_tool="Upstream").add_contour(
            [[0, 0], [10, 0], [10, 10]]
        )

        tool = FiltersTool()
        tool.inputs["input image"] = img
        tool.params["roi"] = [50, 50, 120, 80, 0]
        tool.params["filters"] = [
            {"enabled": True, "type": "gaussian_blur", "kernel_size": 5, "sigma": 1.0}
        ]
        tool.run()

        layer = tool.outputs["output image"].get_overlay_layer("default")
        self.assertEqual(layer.source_tool, tool.name)
        rois = [e for e in layer.elements if e.element_type.name == "ROI"]
        self.assertEqual(len(rois), 1)
        self.assertEqual(rois[0].source_tool, tool.name)

    def test_measurement_overlays_reach_input_with_upstream_layer(self):
        img = VisionImage(_bgr())
        img.get_overlay_layer("default", source_tool="Upstream").add_roi(
            [20, 20, 60, 40, 0]
        )

        tool = MeasurementTool()
        tool.inputs["input image"] = img
        tool.params["overlay_measurements"] = [
            {
                "enabled": True,
                "mode": "area",
                "source1": "Upstream",
                "type1": "ROI",
                "ref_id1": "",
            }
        ]

        counts = []
        for _ in range(3):
            tool.run()
            self.assertEqual(tool.outputs["count"], 1)
            layer = img.get_overlay_layer("default")
            mine = [e for e in layer.elements if e.source_tool == tool.name]
            self.assertTrue(mine, "measurement overlays were not copied to the input")
            counts.append(len(mine))
            # Upstream overlay survives untouched.
            self.assertTrue(
                any(e.source_tool == "Upstream" for e in layer.elements)
            )
        self.assertEqual(len(set(counts)), 1, counts)


if __name__ == "__main__":
    unittest.main()
