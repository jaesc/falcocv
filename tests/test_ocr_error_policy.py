"""OCR minimum-result and Verify comparison policy."""

import numpy as np

from falcocv.core.ocr.industrial_ocr_engine import OCRResult
from falcocv.core.vision_image import VisionImage
from falcocv.tools.ocr.logic import OCR


class FakeEngine:
    def __init__(self, text="ABC", confidence=1.0):
        self.text = text
        self.confidence = confidence

    def _result(self):
        regions = []
        if self.text:
            regions.append(
                {
                    "text": self.text,
                    "confidence": self.confidence,
                    "bbox": (0, 0, 20, 10),
                    "centroid": (10, 5),
                    "points": [[0, 0], [20, 0], [20, 10], [0, 10]],
                }
            )
        return OCRResult(
            text=self.text,
            confidence=self.confidence,
            regions=regions,
        )

    def read(self, image):
        return self._result()

    def verify(self, image, expected):
        return self._result()

    def locate(self, image):
        return self._result()


def _tool(engine, **params):
    tool = OCR("OCR")
    tool._get_engine = lambda: engine
    tool.params.update(
        {
            "mode": "Read",
            "roi": [0, 0, 0, 0, 0],
            "show_overlays": False,
            "show_search_roi": False,
            "show_regions": False,
            "show_labels": False,
            "show_found_origin": False,
            "copy_overlays_to_input": False,
            "min_results": 1,
        }
    )
    tool.params.update(params)
    tool.inputs["input image"] = VisionImage(np.zeros((80, 160, 3), np.uint8))
    return tool


def test_read_mode_min_results():
    tool = _tool(FakeEngine(text=""), min_results=1)
    tool.execute()
    assert tool.outputs["count"] == 0
    assert tool.last_error == "Found 0 text regions; minimum is 1"

    tool.params["min_results"] = 0
    tool.execute()
    assert tool.last_error is None


def test_read_mode_returns_the_recognized_text():
    tool = _tool(FakeEngine(text="ABC"), min_results=1)
    tool.execute()
    assert tool.outputs["text"] == "ABC"
    assert tool.outputs["count"] == 1
    assert tool.last_error is None


def test_verify_match_and_mismatch():
    tool = _tool(FakeEngine(text="ABC"), mode="Verify", expected_text="ABC")
    tool.execute()
    assert tool.last_error is None

    tool.params["expected_text"] = "XYZ"
    tool.execute()
    assert tool.last_error.startswith("Verification failed: expected 'XYZ'")


def test_verify_empty_expected_is_a_setup_error():
    tool = _tool(FakeEngine(text="ABC"), mode="Verify", expected_text="")
    tool.execute()
    assert tool.last_error == "Expected text is empty"


def test_verify_without_text_reports_no_text():
    tool = _tool(FakeEngine(text=""), mode="Verify", expected_text="ABC", min_results=0)
    tool.execute()
    assert tool.last_error == "Verification failed: no text found"


def test_verify_comparison_options():
    tool = _tool(FakeEngine(text="ABC"), mode="Verify", expected_text="abc")
    tool.execute()
    assert tool.last_error.startswith("Verification failed")

    tool.params["verify_ignore_case"] = True
    tool.execute()
    assert tool.last_error is None

    tool.params["expected_text"] = "  ABC  "
    tool.execute()
    assert tool.last_error is None

    tool.params["expected_text"] = "A B C"
    tool.params["verify_ignore_whitespace"] = True
    tool.execute()
    assert tool.last_error is None

    tool.params["expected_text"] = "A B C"
    tool.params["verify_ignore_whitespace"] = False
    tool.execute()
    assert tool.last_error.startswith("Verification failed")


def test_verify_ignores_accents_when_requested():
    tool = _tool(FakeEngine(text="ABC"), mode="Verify", expected_text="ÁBC")
    tool.execute()
    assert tool.last_error.startswith("Verification failed")

    tool.params["verify_ignore_accents"] = True
    tool.execute()
    assert tool.last_error is None


def test_min_results_and_verify_are_independent():
    tool = _tool(
        FakeEngine(text="ABC"),
        mode="Verify",
        expected_text="XYZ",
        min_results=0,
    )
    tool.execute()
    assert tool.last_error.startswith("Verification failed")
