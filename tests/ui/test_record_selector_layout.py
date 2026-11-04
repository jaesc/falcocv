"""Record-selector placement across processing tool configs."""

import pytest

pytestmark = pytest.mark.ui

from falcocv.core.tool_registry import create_tool
from falcocv.ui.widgets.tool_config_dialog import ToolConfigDialog
from falcocv.ui.widgets.tool_run_controls import ToolRunControls


PROCESSING_TOOLS = [
    "AnomalyDetector",
    "BarcodeDecoder",
    "BlobDetector",
    "Classifier",
    "EdgeDetection",
    "FiltersTool",
    "FixtureOverlay",
    "HeightBlobs3D",
    "HistogramTool",
    "ImageCalculatorTool",
    "MeasurementTool",
    "OCR",
    "PlaneFitter",
    "ShapeMatch",
    "TemplateMatch",
]

NO_SELECTOR_TOOLS = [
    "ImageLoader",
    "DepthRangeLoader",
    "GigeCameraLoader",
    "PythonScriptTool",
    "Block",
]


@pytest.mark.parametrize("tool_type", PROCESSING_TOOLS)
def test_processing_tool_record_selector_is_standard(qtbot, tool_type):
    tool = create_tool(tool_type)
    dialog = ToolConfigDialog(tool)
    qtbot.addWidget(dialog)
    dialog.resize(1200, 800)
    dialog.show()
    qtbot.wait(80)

    widget = dialog._config_widget
    selector = getattr(widget, "record_selector", None)
    assert selector is not None, f"{tool_type} has no record selector"
    controls = widget.findChild(ToolRunControls)
    assert controls is not None, f"{tool_type} has no Run/Auto controls"
    assert selector.width() == 285
    assert selector.y() == 0
    assert controls.x() == 8
    assert controls.y() == 0
    assert selector.x() == controls.x() + controls.width() + 8

    dialog.close()


@pytest.mark.parametrize("tool_type", NO_SELECTOR_TOOLS)
def test_source_and_special_tools_have_no_record_selector(qtbot, tool_type):
    tool = create_tool(tool_type)
    dialog = ToolConfigDialog(tool)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    assert not hasattr(dialog._config_widget, "record_selector")
    assert dialog._config_widget.findChild(ToolRunControls) is None

    dialog.close()
