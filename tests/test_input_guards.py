"""Tools must fail when a required input is missing or unusable."""

import numpy as np
import pytest

from falcocv.core.vision_image import VisionImage
from falcocv.tools.filters.logic import FiltersTool
from falcocv.tools.fixture_overlay.logic import FixtureOverlay
from falcocv.tools.histogram.logic import HistogramTool
from falcocv.tools.image_calculator.logic import ImageCalculatorTool
from falcocv.tools.python_script.logic import PythonScriptTool

Classifier = pytest.importorskip("falcocv.tools.classifier.logic").Classifier
AnomalyDetector = pytest.importorskip(
    "falcocv.tools.anomaly_detector.logic"
).AnomalyDetector


def _image():
    return VisionImage(np.zeros((80, 120, 3), np.uint8))


def test_filters_missing_input_is_an_error():
    tool = FiltersTool()
    tool.run()
    assert tool.last_error == "Input image is missing"
    assert tool.outputs["output image"] is None


def test_filters_reject_non_image_input():
    tool = FiltersTool()
    tool.inputs["input image"] = "not an image"
    tool.run()
    assert tool.last_error == "Input image is not a valid image"
    assert tool.outputs["output image"] is None


def test_histogram_missing_input_is_an_error():
    tool = HistogramTool()
    tool.run()
    assert tool.last_error == "Input image is missing"
    assert tool.outputs["output image"] is None
    assert len(tool.outputs["results"]) == 0


def test_image_calculator_missing_a_is_an_error():
    tool = ImageCalculatorTool()
    tool.run()
    assert tool.last_error == "Input image A is missing"
    assert tool.outputs["output image"] is None


def test_image_calculator_two_input_operation_requires_b():
    tool = ImageCalculatorTool()
    tool.inputs["image_A"] = _image()
    tool.params["ops"] = [{"enabled": True, "type": "add"}]
    tool.run()
    assert tool.last_error == "Input image B is missing"
    assert tool.outputs["output image"] is None


def test_image_calculator_unary_operation_does_not_require_b():
    tool = ImageCalculatorTool()
    tool.inputs["image_A"] = _image()
    tool.params["ops"] = [{"enabled": True, "type": "invert"}]
    tool.run()
    assert tool.last_error is None
    assert tool.outputs["output image"] is not None


def test_image_calculator_disabled_two_input_operation_does_not_require_b():
    tool = ImageCalculatorTool()
    tool.inputs["image_A"] = _image()
    tool.params["ops"] = [{"enabled": False, "type": "add"}]
    tool.run()
    assert tool.last_error is None
    assert tool.outputs["output image"] is not None


def test_fixture_overlay_missing_input_is_an_error():
    tool = FixtureOverlay()
    tool.run()
    assert tool.last_error == "Input image is missing"
    assert tool.outputs["output image"] is None


def test_fixture_overlay_without_source_is_valid():
    tool = FixtureOverlay()
    tool.inputs["input image"] = _image()
    tool.run()
    assert tool.last_error is None
    assert tool.outputs["output image"] is not None


def test_fixture_overlay_missing_source_element_is_an_error():
    image = _image()
    layer = image.get_overlay_layer("default", source_tool="Source")
    layer.add_point((50, 60), ref_id="point_1")
    tool = FixtureOverlay()
    tool.inputs["input image"] = image
    tool.params.update(
        {
            "selected_source_tool": "Source",
            "selected_ref_id": "point_1",
            "selected_type": "POINT",
        }
    )
    tool.run()
    assert tool.last_error is None
    assert tool.outputs["output image"].x == 50
    assert tool.outputs["output image"].y == 60

    tool.params["selected_ref_id"] = "missing"
    tool.run()
    assert tool.last_error == "Source element not found: Source.missing (POINT)"
    assert tool.outputs["output image"] is None


def test_fixture_overlay_accepts_elements_without_ref_id():
    image = _image()
    layer = image.get_overlay_layer("default", source_tool="Source")
    layer.add_point((30, 40), ref_id="")
    tool = FixtureOverlay()
    tool.inputs["input image"] = image
    tool.params.update(
        {
            "selected_source_tool": "Source",
            "selected_ref_id": "",
            "selected_type": "POINT",
        }
    )
    tool.run()
    assert tool.last_error is None
    assert tool.outputs["output image"].x == 30
    assert tool.outputs["output image"].y == 40


def test_classifier_missing_input_is_an_error():
    tool = Classifier()
    tool.run()
    assert tool.last_error == "Input image is missing"
    assert tool.outputs["output image"] is None


def test_classifier_without_training_is_an_error():
    tool = Classifier()
    tool.inputs["input image"] = _image()
    tool.run()
    assert tool.last_error == "Classifier is not trained"
    assert tool.outputs["output image"] is not None


def test_classifier_inference_error_is_reported():
    class BoomInference:
        def predict(self, _image):
            raise RuntimeError("boom")

    tool = Classifier()
    tool.inputs["input image"] = _image()
    tool._trained = True
    tool._inference = BoomInference()
    tool.run()
    assert tool.last_error == "Classifier inference failed: RuntimeError: boom"


def test_classifier_recording_without_training_is_not_an_error():
    tool = Classifier()
    tool.inputs["input image"] = _image()
    tool._recording = True
    tool._active_class = "class_1"
    tool._class_data["class_1"] = []
    tool.run()
    assert tool.last_error is None


def test_anomaly_detector_missing_input_is_an_error():
    tool = AnomalyDetector()
    tool.run()
    assert tool.last_error == "Input image is missing"
    assert tool.outputs["output image"] is None


def test_anomaly_detector_without_training_is_an_error():
    tool = AnomalyDetector()
    tool.inputs["input image"] = _image()
    tool.run()
    assert tool.last_error == "Anomaly detector is not trained"
    assert tool.outputs["output image"] is not None


def test_anomaly_detector_inference_error_is_reported():
    class BoomDetector:
        alignment_enabled = False
        template = object()

        def predict(self, _image, build_heatmap=True):
            raise RuntimeError("boom")

    tool = AnomalyDetector()
    tool.inputs["input image"] = _image()
    tool._trained = True
    tool._detector = BoomDetector()
    tool.run()
    assert tool.last_error == "Anomaly inference failed: RuntimeError: boom"


def test_anomaly_detector_recording_without_training_is_not_an_error():
    tool = AnomalyDetector()
    tool.inputs["input image"] = _image()
    tool._recording = True
    tool.run()
    assert tool.last_error is None


def test_python_script_without_script_is_an_error():
    tool = PythonScriptTool()
    tool.run()
    assert tool.last_error == "No script configured"
    assert tool.outputs["image_out"] is None
