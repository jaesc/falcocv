"""OCR config: minimum results and Verify comparison options."""

import pytest

pytestmark = pytest.mark.ui

from falcocv.tools.ocr.config import OCRConfigWidget
from falcocv.tools.ocr.logic import OCR


def test_ocr_error_and_verify_options(qtbot):
    tool = OCR("OCR")
    config = OCRConfigWidget(tool)
    qtbot.addWidget(config)

    assert config.spin_min_results.value() == 1
    config.spin_min_results.setValue(3)
    assert tool.params["min_results"] == 3

    config.chk_verify_trim.setChecked(False)
    assert tool.params["verify_trim"] is False

    config.chk_verify_case.setChecked(True)
    assert tool.params["verify_ignore_case"] is True

    config.chk_verify_whitespace.setChecked(True)
    assert tool.params["verify_ignore_whitespace"] is True

    config.chk_verify_accents.setChecked(True)
    assert tool.params["verify_ignore_accents"] is True
