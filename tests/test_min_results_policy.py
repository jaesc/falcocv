"""Per-tool minimum-result error policy."""

import cv2
import numpy as np

from falcocv.accuracy.synthetic import make_template
from falcocv.core.vision_image import VisionImage
from falcocv.tools.barcode_decoder.logic import BarcodeDecoder
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.edge_detection.logic import EdgeDetection
from falcocv.tools.ocr.logic import OCR
from falcocv.tools.pattern_matcher.logic import PatternMatcher
from falcocv.tools.shape_match.logic import ShapeMatch


def _blank_image():
    return np.zeros((240, 360, 3), np.uint8)


def _one_blob_image():
    image = np.zeros((240, 360, 3), np.uint8)
    cv2.circle(image, (180, 120), 22, (255, 255, 255), -1)
    return image


def _blob_tool(**params):
    tool = BlobDetector("Blob Detector")
    tool.params.update(
        {
            "threshold_mode": "Fixed",
            "threshold_value": 128,
            "polarity": "Bright Blobs",
            "area_min": 100,
            "area_max": 5000,
            "roi": [0, 0, 0, 0, 0],
            "show_contours": False,
            "show_centroids": False,
            "show_labels": False,
            "show_search_roi": False,
            "copy_overlays_to_input": False,
        }
    )
    tool.params.update(params)
    return tool


def _trained_shape():
    template = make_template()
    tool = ShapeMatch("Shape Match")
    tool._static_train_image = template.copy()
    tool.params.update(
        {
            "train_roi": [0, 0, template.shape[1], template.shape[0], 0],
            "angle_start": -20.0,
            "angle_end": 20.0,
            "min_score": 0.5,
            "show_search_roi": False,
            "show_found_region": False,
            "show_found_origin": False,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool.train_pattern()
    return tool


def _trained_pattern():
    template = np.zeros((80, 60, 3), np.uint8)
    cv2.rectangle(template, (15, 10), (25, 65), (255, 255, 255), -1)
    cv2.rectangle(template, (15, 55), (55, 65), (255, 255, 255), -1)
    tool = PatternMatcher("Pattern Matcher")
    tool._static_train_image = template.copy()
    tool.params.update(
        {
            "train_roi": [0, 0, 0, 0, 0],
            "search_roi": [0, 0, 0, 0, 0],
            "angle_start": -10.0,
            "angle_end": 10.0,
            "angle_step": 2.0,
            "scale_start": 1.0,
            "scale_end": 1.0,
            "scale_step": 0.1,
            "pyramid_levels": 1,
            "score_threshold": 0.9,
            "max_matches": 1,
            "show_search_roi": False,
            "show_found_region": False,
            "show_found_origin": False,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool.train_pattern()
    return tool


def test_supported_tools_default_to_one_required_result():
    assert ShapeMatch().params["min_results"] == 1
    assert PatternMatcher().params["min_results"] == 1
    assert BlobDetector().params["min_results"] == 1
    assert EdgeDetection().params["min_results"] == 1
    assert BarcodeDecoder().params["min_results"] == 1
    assert OCR().params["min_results"] == 1


def test_shape_match_errors_below_min_results():
    tool = _trained_shape()
    tool.inputs["input image"] = VisionImage(_blank_image())
    tool.params["min_results"] = 1
    tool.execute()
    assert tool.outputs["count"] == 0
    assert tool.last_error == "Found 0 matches; minimum is 1"

    tool.params["min_results"] = 0
    tool.execute()
    assert tool.last_error is None


def test_pattern_matcher_errors_below_min_results():
    tool = _trained_pattern()
    tool.inputs["input image"] = VisionImage(_blank_image())
    tool.params["min_results"] = 1
    tool.execute()
    assert tool.outputs["count"] == 0
    assert tool.last_error == "Found 0 matches; minimum is 1"

    tool.params["min_results"] = 0
    tool.execute()
    assert tool.last_error is None


def test_blob_detector_min_results():
    tool = _blob_tool(min_results=1)
    tool.inputs["input image"] = VisionImage(_blank_image())
    tool.execute()
    assert tool.outputs["count"] == 0
    assert tool.last_error == "Found 0 blobs; minimum is 1"

    tool.params["min_results"] = 0
    tool.execute()
    assert tool.last_error is None

    tool.inputs["input image"] = VisionImage(_one_blob_image())
    tool.params["min_results"] = 2
    tool.execute()
    assert tool.outputs["count"] == 1
    assert tool.last_error == "Found 1 blob; minimum is 2"

    tool.params["min_results"] = 1
    tool.execute()
    assert tool.last_error is None


def test_edge_detection_min_results():
    tool = EdgeDetection("Edge Detection")
    tool.params.update(
        {
            "min_results": 1,
            "roi": [0, 0, 300, 200, 0],
            "show_search_roi": False,
            "copy_overlays_to_input": False,
        }
    )
    tool.inputs["input image"] = VisionImage(_blank_image())
    tool.execute()
    assert tool.outputs["count"] == 0
    assert tool.last_error == "Found 0 edges; minimum is 1"

    tool.params["min_results"] = 0
    tool.execute()
    assert tool.last_error is None


def test_barcode_decoder_min_results():
    tool = BarcodeDecoder("Barcode Decoder")
    tool.params["min_results"] = 1
    tool.inputs["input image"] = VisionImage(_blank_image())
    tool.execute()
    assert tool.outputs["count"] == 0
    assert tool.last_error == "Found 0 codes; minimum is 1"

    tool.params["min_results"] = 0
    tool.execute()
    assert tool.last_error is None


def test_missing_input_is_always_an_error():
    for tool in (
        BlobDetector("Blob"),
        EdgeDetection("Edge"),
        BarcodeDecoder("Code"),
    ):
        tool.execute()
        assert tool.last_error == "Input image is missing"


def test_old_jobs_keep_the_tool_default():
    tool = _blob_tool()
    data = tool.serialize()
    data["params"].pop("min_results", None)

    restored = BlobDetector("restored")
    restored.deserialize(data)

    assert restored.params["min_results"] == 1
