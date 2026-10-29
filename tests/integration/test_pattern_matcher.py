"""PatternMatcher angle-convention regression tests."""

import math

import cv2
import numpy as np
import pytest

from falcocv.core.overlay_layer import OverlayType
from falcocv.core.vision_image import VisionImage
from falcocv.tools.pattern_matcher.logic import PatternMatcher

pytestmark = pytest.mark.integration


def _template():
    """Asymmetric L on a black background.

    The black border makes the auto-expanded rotated template blend with the
    scene instead of adding bright corners that dominate ``matchTemplate``.
    """
    template = np.zeros((80, 60, 3), dtype=np.uint8)
    cv2.rectangle(template, (15, 10), (25, 65), (255, 255, 255), -1)
    cv2.rectangle(template, (15, 55), (55, 65), (255, 255, 255), -1)
    return template


def _scene(template, angle=30.0, cx=200.0, cy=150.0):
    height, width = template.shape[:2]
    # Positive angle is clockwise (FalcoCV convention).
    matrix = cv2.getRotationMatrix2D(
        (width / 2.0, height / 2.0), -angle, 1.0
    )
    matrix[0, 2] += cx - width / 2.0
    matrix[1, 2] += cy - height / 2.0
    return cv2.warpAffine(
        template,
        matrix,
        (400, 300),
        flags=cv2.INTER_LINEAR,
        borderValue=(0, 0, 0),
    )


def _tool(template):
    tool = PatternMatcher()
    tool._static_train_image = template.copy()
    tool.params.update(
        {
            "train_roi": [0, 0, 0, 0, 0],
            "search_roi": [0, 0, 0, 0, 0],
            "angle_start": -45.0,
            "angle_end": 45.0,
            "angle_step": 1.0,
            "scale_start": 1.0,
            "scale_end": 1.0,
            "scale_step": 0.1,
            "pyramid_levels": 1,
            "score_threshold": 0.3,
            "max_matches": 1,
            "show_search_roi": False,
            "show_found_region": True,
            "show_found_origin": True,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool.train_pattern()
    return tool


def test_pattern_matcher_reports_clockwise_angle():
    template = _template()
    tool = _tool(template)
    tool.inputs["input image"] = VisionImage(_scene(template, angle=30.0))
    tool.run()

    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    assert abs(result.angle - 30.0) <= 1.5
    assert result.score >= 0.8


def test_pattern_matcher_angle_range_is_clockwise():
    template = _template()
    tool = _tool(template)
    tool.params.update({"angle_start": 20.0, "angle_end": 40.0})
    tool.inputs["input image"] = VisionImage(_scene(template, angle=30.0))
    tool.run()

    assert tool.outputs["count"] == 1
    assert abs(tool.outputs["results"][0].angle - 30.0) <= 1.5

    # The same clockwise +30 part must not match a negative-only range.
    tool.params.update(
        {"angle_start": -40.0, "angle_end": -20.0, "score_threshold": 0.6}
    )
    tool.run()
    assert tool.outputs["count"] == 0


def test_pattern_matcher_axes_follow_found_region():
    template = _template()
    tool = _tool(template)
    tool.inputs["input image"] = VisionImage(_scene(template, angle=30.0))
    tool.run()

    layer = tool.outputs["output image"].get_overlay_layer("default")
    contour = next(
        element
        for element in layer.elements
        if element.element_type == OverlayType.CONTOUR
        and element.ref_id == "match_0"
    )
    axes = next(
        element
        for element in layer.elements
        if element.element_type == OverlayType.AXES and element.ref_id == "match_0"
    )

    p0, p1 = np.asarray(contour.data["contour"][:2], dtype=np.float64)
    edge = p1 - p0
    edge /= np.linalg.norm(edge)
    radians = math.radians(axes.data["angle"])
    axis = np.array([math.cos(radians), math.sin(radians)])
    assert float(np.dot(edge, axis)) > 0.999
    assert abs(float(edge[0] * axis[1] - edge[1] * axis[0])) < 1e-3


def test_pattern_matcher_rotated_search_roi_pose():
    template = _template()
    tool = _tool(template)

    for cx, cy, angle, roi_angle in (
        (260.0, 190.0, 20.0, 30.0),
        (245.0, 175.0, -15.0, -25.0),
    ):
        tool.params.update(
            {
                "search_roi": [250.0 - 70.0, 180.0 - 70.0, 140.0, 140.0, roi_angle],
                "angle_start": angle - 10.0,
                "angle_end": angle + 10.0,
            }
        )
        tool.inputs["input image"] = VisionImage(
            _scene(template, angle=angle, cx=cx, cy=cy)
        )
        tool.run()
        assert tool.outputs["count"] == 1
        result = tool.outputs["results"][0]
        assert abs(result.x - cx) <= 1.0
        assert abs(result.y - cy) <= 1.0
        assert abs(result.angle - angle) <= 1.5


def test_pattern_matcher_rotation_keeps_center():
    template = _template()
    tool = _tool(template)
    for angle in (0.0, 30.0, 90.0):
        tool.params.update(
            {"angle_start": angle - 15.0, "angle_end": angle + 15.0}
        )
        tool.inputs["input image"] = VisionImage(
            _scene(template, angle=angle, cx=200.0, cy=150.0)
        )
        tool.run()
        assert tool.outputs["count"] == 1
        result = tool.outputs["results"][0]
        assert abs(result.x - 200.0) <= 1.0
        assert abs(result.y - 150.0) <= 1.0


def test_pattern_matcher_act_as_fixture_uses_absolute_angle():
    template = _template()
    tool = _tool(template)
    scene = _scene(template, angle=30.0, cx=200.0, cy=150.0)
    image = VisionImage(scene, x=0.0, y=0.0, angle=30.0)
    roi = image.image_to_fixture([170.0, 110.0, 60.0, 80.0, 30.0])
    roi[4] = 0.0  # ROI aligned with the fixture

    tool.params.update(
        {
            "search_roi": roi,
            "angle_start": 20.0,
            "angle_end": 40.0,
            "act_as_fixture": True,
        }
    )
    tool.inputs["input image"] = image
    tool.run()
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    output = tool.outputs["output image"]
    assert abs(result.angle - 30.0) <= 1.5
    assert abs(output.angle - result.angle) <= 1e-6
