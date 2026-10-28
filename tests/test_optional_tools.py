"""Smoke import tests for optional ML tools."""

import unittest


class TestOptionalMLTools(unittest.TestCase):
    """OCR and AnomalyDetector need optional ML/DNN dependencies."""

    def test_ocr_logic_import(self):
        try:
            from falcocv.tools.ocr.logic import OCR
        except ImportError as e:
            self.skipTest(f"OCR dependencies not available: {e}")
        self.assertTrue(callable(OCR))

    def test_anomaly_detector_logic_import(self):
        try:
            from falcocv.tools.anomaly_detector.logic import AnomalyDetector
        except ImportError as e:
            self.skipTest(f"AnomalyDetector dependencies not available: {e}")
        self.assertTrue(callable(AnomalyDetector))


if __name__ == "__main__":
    unittest.main()
