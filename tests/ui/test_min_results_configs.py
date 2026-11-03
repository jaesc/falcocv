"""The shared Minimum results control is wired in every supported tool."""

import pytest

pytestmark = pytest.mark.ui

from falcocv.tools.barcode_decoder.config import BarcodeDecoderConfig
from falcocv.tools.barcode_decoder.logic import BarcodeDecoder
from falcocv.tools.blob_detector.config import BlobDetectorConfig
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.edge_detection.config import EdgeDetectionConfig
from falcocv.tools.edge_detection.logic import EdgeDetection
from falcocv.tools.ocr.config import OCRConfigWidget
from falcocv.tools.ocr.logic import OCR
from falcocv.tools.template_match.config import TemplateMatchConfig
from falcocv.tools.template_match.logic import TemplateMatch
from falcocv.tools.shape_match.config import ShapeMatchConfig
from falcocv.tools.shape_match.logic import ShapeMatch


@pytest.mark.parametrize(
    "tool_cls, config_cls, maximum",
    [
        (ShapeMatch, ShapeMatchConfig, 100),
        (TemplateMatch, TemplateMatchConfig, 100),
        (BlobDetector, BlobDetectorConfig, 1000),
        (EdgeDetection, EdgeDetectionConfig, 1),
        (BarcodeDecoder, BarcodeDecoderConfig, 100),
        (OCR, OCRConfigWidget, 100),
    ],
    ids=["shape", "pattern", "blob", "edge", "barcode", "ocr"],
)
def test_min_results_control_updates_the_tool(qtbot, tool_cls, config_cls, maximum):
    tool = tool_cls()
    config = config_cls(tool)
    qtbot.addWidget(config)

    assert config.spin_min_results.value() == 1
    target = min(3, maximum)
    config.spin_min_results.setValue(target)
    assert tool.params["min_results"] == target

    config.spin_min_results.setValue(0)
    assert tool.params["min_results"] == 0
