"""Fixture composition across tools.

A tool that acts as a fixture must replace only the components the user
selected and keep the rest of the input fixture.  This keeps a chain of
fixture tools composable: a tool that only sets X must not reset Y or the
angle.
"""

import itertools

import cv2
import numpy as np
import pytest

from falcocv.core.vision_image import VisionImage
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.edge_detection.logic import EdgeDetection
from falcocv.tools.filters.logic import FiltersTool
from falcocv.tools.fixture_overlay.logic import (
    FixtureOverlay,
    _enumerate_elements,
)
from falcocv.tools.measurement.logic import MeasurementTool
from falcocv.tools.shape_match.logic import ShapeMatch
from falcocv.tools.template_match.logic import TemplateMatch

FIXTURE = (5.0, 6.0, 7.0)
POINT = (100.0, 120.0)
SOURCE = "Blob Detector"


def _image(fixture=FIXTURE):
    return VisionImage(
        np.zeros((200, 200), dtype=np.uint8),
        x=fixture[0],
        y=fixture[1],
        angle=fixture[2],
    )


def _image_with_point(fixture=FIXTURE, point=POINT, ref_id="blob_0"):
    image = _image(fixture)
    layer = image.get_overlay_layer("default", source_tool=SOURCE)
    layer.add_point(point, ref_id=ref_id)
    return image


def _fixture_tool(image, ref_id="blob_0", type_name="POINT", source=SOURCE):
    tool = FixtureOverlay()
    tool.inputs["input image"] = image
    tool.params.update(
        {
            "selected_source_tool": source,
            "selected_ref_id": ref_id,
            "selected_type": type_name,
        }
    )
    return tool


@pytest.mark.parametrize(
    "use_x, use_y, use_angle",
    list(itertools.product([True, False], repeat=3)),
)
def test_fixture_overlay_replaces_only_selected_components(
    use_x, use_y, use_angle
):
    tool = _fixture_tool(_image_with_point())
    tool.params.update(
        {
            "fixture_use_x": use_x,
            "fixture_use_y": use_y,
            "fixture_use_angle": use_angle,
        }
    )
    tool.run()

    out = tool.outputs["output image"]
    assert out is not None
    assert out.x == pytest.approx(POINT[0] if use_x else FIXTURE[0])
    assert out.y == pytest.approx(POINT[1] if use_y else FIXTURE[1])
    assert out.angle == pytest.approx(0.0 if use_angle else FIXTURE[2])


def test_fixture_overlay_keeps_y_and_angle_when_only_x_is_used():
    tool = _fixture_tool(_image_with_point())
    tool.params.update(
        {"fixture_use_x": True, "fixture_use_y": False, "fixture_use_angle": False}
    )
    tool.run()

    out = tool.outputs["output image"]
    assert out.x == pytest.approx(POINT[0])
    assert out.y == pytest.approx(FIXTURE[1])
    assert out.angle == pytest.approx(FIXTURE[2])


def test_fixture_overlay_roi_uses_centre_and_angle():
    image = _image()
    image.get_overlay_layer("default", source_tool=SOURCE).add_roi(
        [40.0, 50.0, 20.0, 30.0, 12.5], ref_id="roi_1"
    )
    tool = _fixture_tool(image, ref_id="roi_1", type_name="ROI")
    tool.run()

    out = tool.outputs["output image"]
    assert out.x == pytest.approx(50.0)
    assert out.y == pytest.approx(65.0)
    assert out.angle == pytest.approx(12.5)


def test_fixture_overlay_normalises_the_angle_like_the_match_tools():
    image = _image()
    image.get_overlay_layer("default", source_tool=SOURCE).add_roi(
        [0.0, 0.0, 100.0, 100.0, 270.0], ref_id="roi_1"
    )
    tool = _fixture_tool(image, ref_id="roi_1", type_name="ROI")
    tool.run()

    assert tool.outputs["output image"].angle == pytest.approx(-90.0)


def test_fixture_overlay_without_selection_keeps_the_whole_fixture():
    tool = FixtureOverlay()
    tool.inputs["input image"] = _image_with_point()
    tool.run()

    out = tool.outputs["output image"]
    assert (out.x, out.y, out.angle) == pytest.approx(FIXTURE)


def test_fixture_overlay_hides_world_space_elements():
    image = _image()
    layer = image.get_overlay_layer("default", source_tool="Plane Fitter")
    layer.add_world_points(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)], ref_id="plane_points"
    )

    assert _enumerate_elements(image) == []


def test_fixture_overlay_missing_element_reports_an_error():
    tool = _fixture_tool(_image_with_point(), ref_id="missing")
    tool.run()

    assert tool.outputs["output image"] is None
    assert "not found" in (tool.last_error or "")


# ── Pass-through tools keep the input fixture ─────────────────────────


def _filters():
    tool = FiltersTool()
    tool.params["filters"] = []
    return tool


def _blob():
    tool = BlobDetector()
    tool.params["min_results"] = 0
    return tool


def _edge():
    tool = EdgeDetection()
    tool.params.update({"roi": [10.0, 10.0, 50.0, 50.0, 0.0], "min_results": 0})
    return tool


def _measurement():
    return MeasurementTool()


@pytest.mark.parametrize(
    "factory",
    [_filters, _blob, _edge, _measurement],
    ids=["filters", "blob", "edge", "measurement"],
)
def test_processing_tools_preserve_the_input_fixture(factory):
    tool = factory()
    tool.inputs["input image"] = _image()
    tool.run()

    out = tool.outputs["output image"]
    assert out is not None
    assert (out.x, out.y, out.angle) == pytest.approx(FIXTURE)


# ── TemplateMatch composes with an existing fixture ─────────────────────


def _template():
    template = np.zeros((80, 60, 3), dtype=np.uint8)
    cv2.rectangle(template, (15, 10), (25, 65), (255, 255, 255), -1)
    cv2.rectangle(template, (15, 55), (55, 65), (255, 255, 255), -1)
    return template


def _scene(template, angle=30.0, cx=200.0, cy=150.0):
    height, width = template.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), -angle, 1.0)
    matrix[0, 2] += cx - width / 2.0
    matrix[1, 2] += cy - height / 2.0
    return cv2.warpAffine(
        template,
        matrix,
        (400, 300),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )


def _template_tool(template):
    tool = TemplateMatch()
    tool._static_train_image = template.copy()
    tool.params.update(
        {
            "train_roi": [0, 0, 0, 0, 0],
            "angle_start": 20.0,
            "angle_end": 40.0,
            "angle_step": 1.0,
            "scale_start": 1.0,
            "scale_end": 1.0,
            "scale_step": 0.1,
            "pyramid_levels": 1,
            "score_threshold": 0.6,
            "min_results": 0,
            "show_search_roi": False,
            "show_found_region": False,
            "show_found_origin": False,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool.train_pattern()
    return tool


def test_template_match_partial_fixture_keeps_the_input_y():
    template = _template()
    image = VisionImage(
        _scene(template), x=0.0, y=12.5, angle=30.0
    )
    roi = image.image_to_fixture([170.0, 110.0, 60.0, 80.0, 30.0])
    roi[4] = 0.0  # ROI aligned with the fixture

    tool = _template_tool(template)
    tool.params.update(
        {
            "search_roi": roi,
            "act_as_fixture": True,
            "fixture_use_x": True,
            "fixture_use_y": False,
            "fixture_use_angle": True,
        }
    )
    tool.inputs["input image"] = image
    tool.run()

    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    out = tool.outputs["output image"]
    assert out.y == pytest.approx(12.5)
    assert out.x == pytest.approx(result.x)
    assert out.angle == pytest.approx(result.angle)


# ── ShapeMatch composes with an existing fixture ────────────────────────


def _shape_tool(template):
    tool = ShapeMatch()
    tool.params.update(
        {
            "angle_start": 20.0,
            "angle_end": 40.0,
            "scale_start": 1.0,
            "scale_end": 1.0,
            "min_score": 0.25,
            "max_matches": 1,
            "max_features": 800,
            "show_search_roi": False,
            "show_found_region": False,
            "show_found_origin": False,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool._static_train_image = template.copy()
    tool.params["train_roi"] = [0, 0, template.shape[1], template.shape[0], 0]
    tool.train_pattern()
    return tool


def test_shape_match_partial_fixture_keeps_the_input_y():
    from falcocv.accuracy.synthetic import make_template, render_scene

    template = make_template()
    scene = render_scene(template, cx=300.0, cy=190.0, angle=30.0, scale=1.0)
    image = VisionImage(scene, x=0.0, y=12.5, angle=30.0)
    roi = image.image_to_fixture([200.0, 90.0, 200.0, 200.0, 30.0])
    roi[4] = 0.0  # ROI aligned with the fixture

    tool = _shape_tool(template)
    tool.params.update(
        {
            "search_roi": roi,
            "act_as_fixture": True,
            "fixture_use_x": True,
            "fixture_use_y": False,
            "fixture_use_angle": True,
        }
    )
    tool.inputs["input image"] = image
    tool.run()

    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    out = tool.outputs["output image"]
    assert out.y == pytest.approx(12.5)
    assert out.x == pytest.approx(result.x)
    assert out.angle == pytest.approx(result.angle)
