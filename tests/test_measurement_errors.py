"""Measurement fails when a configured source element is not available."""

import numpy as np

from falcocv.core.vision_image import VisionImage
from falcocv.tools.measurement.logic import MeasurementTool


def _image_with_points():
    image = VisionImage(np.zeros((120, 200, 3), np.uint8))
    layer = image.get_overlay_layer("default", source_tool="Source")
    layer.add_point((20, 30), color=(0, 255, 0), radius=3, ref_id="a")
    layer.add_point((100, 30), color=(0, 255, 0), radius=3, ref_id="b")
    return image


def _row(**overrides):
    row = {
        "enabled": True,
        "mode": "distance",
        "source1": "Source",
        "ref_id1": "a",
        "type1": "POINT",
        "strategy1": "center",
        "source2": "Source",
        "ref_id2": "b",
        "type2": "POINT",
        "strategy2": "center",
        "color": [0, 255, 255],
        "thickness": 2,
    }
    row.update(overrides)
    return row


def _tool(image, rows):
    tool = MeasurementTool("Measurement")
    tool.params.update(
        {
            "overlay_measurements": rows,
            "show_measurement_lines": False,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool.inputs["input image"] = image
    return tool


def test_valid_measurement_has_no_error():
    tool = _tool(_image_with_points(), [_row()])
    tool.execute()
    assert tool.outputs["count"] == 1
    assert tool.last_error is None


def test_missing_second_source_is_an_error():
    tool = _tool(_image_with_points(), [_row(ref_id2="missing")])
    tool.execute()
    assert tool.outputs["count"] == 0
    assert "source2" in tool.last_error
    assert "missing" in tool.last_error


def test_missing_first_source_is_an_error():
    tool = _tool(_image_with_points(), [_row(ref_id1="missing")])
    tool.execute()
    assert tool.outputs["count"] == 0
    assert "source1" in tool.last_error
    assert "missing" in tool.last_error


def test_disabled_row_with_missing_source_is_ignored():
    tool = _tool(_image_with_points(), [_row(enabled=False, ref_id2="missing")])
    tool.execute()
    assert tool.outputs["count"] == 0
    assert tool.last_error is None


def test_valid_and_invalid_rows_report_partial_results():
    tool = _tool(
        _image_with_points(),
        [_row(ref_id2="b"), _row(ref_id2="missing")],
    )
    tool.execute()
    assert tool.outputs["count"] == 1
    assert "M1" in tool.last_error
    assert "source2" in tool.last_error


def test_incompatible_type_is_an_error():
    tool = _tool(_image_with_points(), [_row(mode="angle")])
    tool.execute()
    assert tool.outputs["count"] == 0
    assert "not valid for angle" in tool.last_error


def test_no_active_rows_is_not_an_error():
    tool = _tool(_image_with_points(), [])
    tool.execute()
    assert tool.outputs["count"] == 0
    assert tool.last_error is None


def test_missing_input_is_an_error():
    tool = MeasurementTool("Measurement")
    tool.execute()
    assert tool.last_error == "Input image is missing"
