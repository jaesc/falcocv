"""LineFit / CircleFit tool behavior (headless, no Qt)."""

import math

import numpy as np
import pytest

from falcocv.core.vision_image import VisionImage
from falcocv.tools.circle_fit import CircleFit
from falcocv.tools.line_fit import LineFit


def _blank_image(size=300):
    return VisionImage(np.zeros((size, size, 3), dtype=np.uint8))


def _line_image(slope=0.25, offset=60.0, size=300):
    """White below ``y = slope * x + offset``; the edge is the line."""
    yy, xx = np.mgrid[0:size, 0:size]
    data = np.zeros((size, size, 3), dtype=np.uint8)
    data[yy > (slope * xx + offset)] = 255
    return VisionImage(data)


def _circle_image(center=(150.0, 150.0), radius=50.0, size=300):
    yy, xx = np.mgrid[0:size, 0:size]
    data = np.zeros((size, size, 3), dtype=np.uint8)
    data[(xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2] = 255
    return VisionImage(data)


def _two_edge_image(size=300):
    """Top edge for x >= 100; a parallel edge 35 px lower everywhere else."""
    yy, xx = np.mgrid[0:size, 0:size]
    top = 0.25 * xx + 60.0
    bottom = top + 35.0
    data = np.zeros((size, size, 3), dtype=np.uint8)
    data[(yy > top) & (xx >= 100)] = 255
    data[yy > bottom] = 255
    return VisionImage(data)


def _run_line(image, **params):
    tool = LineFit()
    tool.inputs["input image"] = image
    tool.params.update(params)
    tool.run()
    return tool


def test_line_fit_uses_the_caliper_array_by_default():
    tool = _run_line(_line_image(), roi=[40, 55, 220, 90, 0])

    assert not tool.last_error, tool.last_error
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    assert result.type == "line"
    assert result.angle == pytest.approx(-14.0362, abs=0.3)
    assert result.unit == "deg"
    assert result.calipers == 9
    assert result.edge_points == 9
    assert result.inliers == 9
    assert result.outliers == 0
    assert result.rms < 0.5
    # Caliper centres are inset by half a station from the ROI ends.
    station_span = 220.0 - 220.0 / 9.0
    assert result.length == pytest.approx(
        math.hypot(station_span, station_span * 0.25), abs=2.0
    )
    # The caliper segments are drawn as well as the fit.
    output = tool.outputs["output image"]
    own = [
        element
        for element in output.get_overlay_layer("default").elements
        if element.source_tool == tool.name
    ]
    assert any(element.element_type.name == "LINE" for element in own)
    assert any(element.ref_id == "fit" for element in own)
    assert len(own) >= 1 + 9 + 9  # fit line, caliper segments, inlier points


def test_line_fit_caliper_mode_requires_a_roi():
    tool = _run_line(_line_image())

    assert tool.outputs["count"] == 0
    assert tool.last_error
    assert "ROI" in tool.last_error


def test_line_fit_calipers_respect_the_polarity():
    image = _line_image()
    tool = _run_line(
        image, roi=[40, 55, 220, 90, 0], caliper_edge_polarity="Light→Dark"
    )

    # The synthetic edge brightens towards +V, so the opposite polarity must
    # not produce a full array of points (the replicate border can still give
    # a spurious edge, so only the fitted result is asserted).
    if tool.outputs["count"]:
        assert tool.outputs["results"][0].edge_points < 9


def test_line_fit_trims_calipers_on_the_wrong_edge():
    # The top edge only exists for x >= 100; the two leftmost calipers find a
    # parallel edge 35 px lower.  The capped trimming must keep the real line.
    tool = _run_line(_two_edge_image(), roi=[40, 55, 220, 90, 0])

    assert not tool.last_error, tool.last_error
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    assert result.angle == pytest.approx(-14.0362, abs=1.0)
    assert result.outliers == 2
    assert result.inliers == result.edge_points - 2
    assert result.rms < 1.0


def test_line_fit_errors_when_more_calipers_are_outliers_than_allowed():
    tool = _run_line(
        _two_edge_image(), roi=[40, 55, 220, 90, 0], max_outlier_percent=10.0
    )

    assert tool.outputs["count"] == 0
    assert tool.last_error
    assert "outlier" in tool.last_error.lower()


def test_fit_tools_do_not_expose_an_input_points_port():
    assert "input points" not in LineFit().inputs
    assert "input points" not in CircleFit().inputs


def test_line_fit_tool_reports_too_few_points():
    tool = LineFit()
    tool.inputs["input image"] = _blank_image()
    tool.params["roi"] = [40, 40, 220, 120, 0]
    tool.params["min_points"] = 3
    tool.run()

    assert tool.outputs["count"] == 0
    assert tool.outputs["results"] == []
    assert tool.last_error
    assert "found 0" in tool.last_error


def test_circle_fit_uses_the_radial_caliper_array_by_default():
    tool = CircleFit()
    tool.inputs["input image"] = _circle_image()
    tool.params["roi"] = [80, 80, 140, 140, 0]
    tool.run()

    assert not tool.last_error, tool.last_error
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    assert result.type == "circle"
    assert result.cx == pytest.approx(150.0, abs=1.0)
    assert result.cy == pytest.approx(150.0, abs=1.0)
    assert result.radius == pytest.approx(50.0, abs=1.0)
    assert result.diameter == pytest.approx(2.0 * result.radius)
    assert result.unit == "px"
    assert result.calipers == 16
    assert result.edge_points == 16
    assert result.inliers == 16
    assert result.outliers == 0
    assert result.rms < 1.0


def test_circle_fit_scans_outwards_from_the_caliper_ring():
    # The edge sits at radius 50; the ring is inside it, so the band has to
    # extend away from the centre to reach it.
    tool = CircleFit()
    tool.inputs["input image"] = _circle_image()
    tool.params["roi"] = [80, 80, 140, 140, 0]
    tool.params["caliper_direction"] = "outward"
    tool.params["caliper_radius"] = 25.0
    tool.params["caliper_length"] = 40.0
    tool.run()

    assert not tool.last_error, tool.last_error
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    assert result.cx == pytest.approx(150.0, abs=1.0)
    assert result.cy == pytest.approx(150.0, abs=1.0)
    assert result.radius == pytest.approx(50.0, abs=1.5)
    assert result.edge_points == 16

    # The same ring and band scanning inwards stays inside the white disc and
    # cannot reach the edge, which proves the direction is what changed.
    inward = CircleFit()
    inward.inputs["input image"] = _circle_image()
    inward.params["roi"] = [80, 80, 140, 140, 0]
    inward.params["caliper_radius"] = 25.0
    inward.params["caliper_length"] = 40.0
    inward.run()
    assert inward.outputs["count"] == 0


def test_circle_fit_caliper_mode_requires_a_roi():
    tool = CircleFit()
    tool.inputs["input image"] = _circle_image()
    tool.run()

    assert tool.outputs["count"] == 0
    assert tool.last_error
    assert "ROI" in tool.last_error


def test_circle_fit_tool_reports_too_few_points():
    tool = CircleFit()
    tool.inputs["input image"] = _blank_image()
    tool.params["roi"] = [60, 60, 180, 180, 0]
    tool.run()

    assert tool.outputs["count"] == 0
    assert tool.last_error
    assert "found 0" in tool.last_error


def test_circle_fit_rejects_a_radius_above_the_max():
    tool = CircleFit()
    tool.inputs["input image"] = _circle_image()
    tool.params["roi"] = [80, 80, 140, 140, 0]
    tool.params["max_radius"] = 40.0
    tool.run()

    assert tool.outputs["count"] == 0
    assert tool.outputs["results"] == []
    assert tool.last_error
    assert "above" in tool.last_error.lower()


def test_circle_fit_rejects_a_radius_below_the_min():
    tool = CircleFit()
    tool.inputs["input image"] = _circle_image()
    tool.params["roi"] = [80, 80, 140, 140, 0]
    tool.params["min_radius"] = 60.0
    tool.run()

    assert tool.outputs["count"] == 0
    assert tool.last_error
    assert "below" in tool.last_error.lower()


def test_circle_fit_accepts_a_radius_inside_the_limits():
    tool = CircleFit()
    tool.inputs["input image"] = _circle_image()
    tool.params["roi"] = [80, 80, 140, 140, 0]
    tool.params["min_radius"] = 40.0
    tool.params["max_radius"] = 60.0
    tool.run()

    assert not tool.last_error, tool.last_error
    assert tool.outputs["count"] == 1
    assert tool.outputs["results"][0].radius == pytest.approx(50.0, abs=1.0)


def test_circle_fit_limits_are_disabled_at_zero():
    tool = CircleFit()
    tool.inputs["input image"] = _circle_image()
    tool.params["roi"] = [80, 80, 140, 140, 0]
    tool.params["min_radius"] = 0.0
    tool.params["max_radius"] = 0.0
    tool.run()

    assert not tool.last_error, tool.last_error
    assert tool.outputs["count"] == 1


def test_circle_fit_rejects_inverted_limits():
    tool = CircleFit()
    tool.inputs["input image"] = _circle_image()
    tool.params["roi"] = [80, 80, 140, 140, 0]
    tool.params["min_radius"] = 80.0
    tool.params["max_radius"] = 40.0
    tool.run()

    assert tool.outputs["count"] == 0
    assert tool.last_error
    assert "inverted" in tool.last_error.lower()


def test_circle_fit_rejects_a_straight_edge_with_a_max_radius():
    tool = CircleFit()
    tool.inputs["input image"] = _line_image()
    tool.params["roi"] = [60, 60, 180, 180, 0]
    tool.params["max_radius"] = 500.0
    tool.run()

    assert tool.outputs["count"] == 0
    assert tool.last_error
    assert "above" in tool.last_error.lower()
