"""Regression tests for migrated C++ tools (Histogram, ImageCalculator, PatternMatcher)."""

import numpy as np
import cv2
import unittest

from falcocv.core.vision_image import VisionImage


def _gradient_image(w=128, h=128):
    grad = np.tile(np.linspace(0, 255, w, dtype=np.uint8), (h, 1))
    return cv2.cvtColor(grad, cv2.COLOR_GRAY2BGR)


class TestHistogramTool(unittest.TestCase):

    def test_histogram_run_returns_stats_and_image(self):
        from falcocv.tools.histogram.logic import HistogramTool
        tool = HistogramTool()
        tool.inputs["input image"] = VisionImage(_gradient_image())
        tool.run()
        self.assertIsNotNone(tool.outputs["output image"])
        self.assertGreater(len(tool.outputs["results"]), 0)
        stats = tool.outputs["results"][0]
        self.assertIn("mean", stats)
        self.assertIn("std_dev", stats)


class TestImageCalculatorTool(unittest.TestCase):

    def test_add_operation(self):
        from falcocv.tools.image_calculator.logic import ImageCalculatorTool
        tool = ImageCalculatorTool()
        a = VisionImage(np.full((50, 50, 3), 100, dtype=np.uint8))
        b = VisionImage(np.full((50, 50, 3), 50, dtype=np.uint8))
        tool.inputs["image_A"] = a
        tool.inputs["image_B"] = b
        tool.params["ops"] = [{"enabled": True, "type": "add"}]
        tool.run()
        out = tool.outputs["output image"]
        self.assertIsNotNone(out)
        self.assertEqual(out.data.shape, (50, 50, 3))


class TestPatternMatcherTool(unittest.TestCase):

    def test_pattern_match_finds_self(self):
        from falcocv.tools.pattern_matcher.logic import PatternMatcher
        tool = PatternMatcher()
        # Small template in the center of a larger image
        img = np.zeros((80, 80, 3), dtype=np.uint8)
        img[30:50, 30:50] = 255
        tool.inputs["input image"] = VisionImage(img)
        tool.save_train_image()
        tool.train_from_roi(30, 30, 20, 20)
        # Limit search space to keep the test fast
        tool.params["angle_start"] = 0
        tool.params["angle_end"] = 0
        tool.params["angle_step"] = 1
        tool.params["scale_start"] = 1.0
        tool.params["scale_end"] = 1.0
        tool.params["scale_step"] = 0.1
        tool.params["pyramid_levels"] = 1
        tool.params["score_threshold"] = 0.7
        tool.run()
        self.assertTrue(tool.outputs.get("trained", False))
        self.assertGreaterEqual(tool.outputs.get("count", 0), 1)
        self.assertGreaterEqual(len(tool.outputs["results"]), 1)


if __name__ == "__main__":
    unittest.main()
