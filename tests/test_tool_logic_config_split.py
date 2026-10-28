"""Regression tests for the logic / config separation refactor (Fase C)."""

import numpy as np
import cv2
import unittest

from falcocv.core.vision_image import VisionImage
from falcocv.core.overlay_layer import OverlayType


class TestMeasurementLogic(unittest.TestCase):
    """Measurement helpers now live in tools.measurement.logic."""

    def test_constants_and_helpers_import(self):
        from falcocv.tools.measurement import logic
        self.assertIn("distance", logic.MODE_NAMES)
        self.assertIn("POINT", logic.STRATEGIES)
        self.assertEqual(logic.MODE_UNIT["angle"], "deg")

    def test_closest_on_segment(self):
        from falcocv.tools.measurement.logic import _closest_on_segment
        s, e = (0, 0), (10, 0)
        self.assertEqual(_closest_on_segment(s, e, (-5, 5)), (0, 0))
        self.assertEqual(_closest_on_segment(s, e, (5, 5)), (5, 0))
        self.assertEqual(_closest_on_segment(s, e, (15, 5)), (10, 0))

    def test_reference_point_point(self):
        from falcocv.tools.measurement.logic import _reference_point
        img = VisionImage(np.zeros((100, 100, 3), dtype=np.uint8))
        layer = img.get_overlay_layer("test")
        layer.add_point((10, 20), color=(0, 255, 0))
        elem = layer.elements[0]
        pt = _reference_point(elem, "center")
        self.assertEqual(pt, (10.0, 20.0))


class TestFiltersLogic(unittest.TestCase):
    """Filter definitions and preview helper live in tools.filters.logic."""

    def test_filter_defs_have_all_types(self):
        from falcocv.tools.filters.logic import FILTER_DEFS, FILTER_TYPE_ORDER
        self.assertEqual(set(FILTER_DEFS.keys()), set(FILTER_TYPE_ORDER))

    def test_make_default_filter(self):
        from falcocv.tools.filters.logic import _make_default_filter, FILTER_DEFS
        f = _make_default_filter("gaussian_blur")
        self.assertEqual(f["type"], "gaussian_blur")
        for p in FILTER_DEFS["gaussian_blur"]["params"]:
            self.assertIn(p["key"], f)

    def test_filter_summary(self):
        from falcocv.tools.filters.logic import _filter_summary
        f = {"type": "threshold", "value": 128, "inverted": False}
        summary = _filter_summary(f)
        self.assertIn("128", summary)

    def test_apply_filter_invert(self):
        from falcocv.tools.filters.logic import apply_filter
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        out = apply_filter(img, "invert", {})
        np.testing.assert_array_equal(out, 255 - img)

    def test_apply_filter_blur_preserves_shape(self):
        from falcocv.tools.filters.logic import apply_filter
        img = np.zeros((20, 30, 3), dtype=np.uint8)
        out = apply_filter(img, "gaussian_blur", {"kernel_size": 5, "sigma": 1.0})
        self.assertEqual(out.shape, img.shape)
        self.assertEqual(out.dtype, img.dtype)


class TestPythonScriptPortCRUD(unittest.TestCase):
    """Port CRUD methods moved from config.py into PythonScriptTool."""

    def setUp(self):
        from falcocv.tools.python_script.logic import PythonScriptTool
        self.tool = PythonScriptTool()

    def test_add_input_port(self):
        name = self.tool.add_input_port()
        self.assertEqual(name, "in_1")
        self.assertIn(name, self.tool.inputs)

    def test_add_output_port_with_name(self):
        name = self.tool.add_output_port("custom_out")
        self.assertEqual(name, "custom_out")
        self.assertIn(name, self.tool.outputs)

    def test_remove_input_port(self):
        self.tool.add_input_port()
        self.tool.remove_input_port(0)
        self.assertEqual(len(self.tool.params["input_ports"]), 1)
        self.assertEqual(list(self.tool.inputs.keys()), ["in_1"])

    def test_set_ports_filters_empty_names(self):
        self.tool.set_input_ports(["a", "", "b", "   "])
        self.assertEqual(self.tool.params["input_ports"], ["a", "b"])


class TestGigeCameraLogic(unittest.TestCase):
    """Camera pool / snap / preview helpers live in tools.gige_camera.logic."""

    def setUp(self):
        from falcocv.tools.gige_camera.logic import GigeCameraLoader
        self.tool = GigeCameraLoader()
        self.tool.params["camera_type"] = "Simulation"

    def test_simulation_connect_and_snap(self):
        self.tool.connect()
        self.assertTrue(self.tool.is_connected())
        result = self.tool.snap()
        self.assertIsNotNone(result)
        self.assertIsInstance(result, VisionImage)
        self.assertIsNotNone(self.tool.outputs.get("output image"))

    def test_preview_grab(self):
        self.tool.connect()
        self.tool.start_preview()
        frame = self.tool.grab_preview_frame()
        self.assertIsNotNone(frame)
        self.assertIsInstance(frame, np.ndarray)
        self.tool.stop_preview()
        self.tool.disconnect()
        self.assertFalse(self.tool.is_connected())


class TestClassifierTrainWorker(unittest.TestCase):
    """TrainWorker lives in tools.classifier.train_worker."""

    def test_train_worker_import_and_signals(self):
        from falcocv.tools.classifier.train_worker import TrainWorker
        from PyQt6.QtCore import pyqtSignal
        self.assertIsInstance(TrainWorker.progress, pyqtSignal)
        self.assertIsInstance(TrainWorker.finished, pyqtSignal)
        self.assertIsInstance(TrainWorker.error, pyqtSignal)
        worker = TrainWorker({}, {}, {}, 3)
        self.assertTrue(hasattr(worker, "result_engine"))


if __name__ == "__main__":
    unittest.main()
