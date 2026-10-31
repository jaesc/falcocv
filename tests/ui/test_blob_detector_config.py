"""Blob Detector config: blob-mode selector wiring."""

import pytest

pytestmark = pytest.mark.ui

from falcocv.tools.blob_detector.config import BlobDetectorConfig
from falcocv.tools.blob_detector.logic import BlobDetector


def test_blob_mode_combo_updates_the_tool(qtbot):
    detector = BlobDetector("Blob Detector")
    config = BlobDetectorConfig(detector)
    qtbot.addWidget(config)

    config.combo_blob_mode.setCurrentText("Nested blobs")
    assert detector.params["blob_mode"] == "Nested"

    config.combo_blob_mode.setCurrentText("Outer blobs")
    assert detector.params["blob_mode"] == "Outer"
