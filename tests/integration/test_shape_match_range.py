"""ShapeMatch range-image training and fixture propagation."""

import numpy as np

from falcocv.core.vision_image import VisionImage
from falcocv.tools.shape_match.logic import ShapeMatch
from falcocv.tools.template_match.logic import TemplateMatch


def _template() -> np.ndarray:
    depth = np.full((100, 120), 1000.0, dtype=np.float32)
    depth[20:80, 20:35] = 900.0
    depth[65:80, 20:90] = 900.0
    depth[35:65, 20:90] = 930.0
    return depth


def _scene(template: np.ndarray, cx=220, cy=170) -> np.ndarray:
    depth = np.full((320, 420), 1000.0, dtype=np.float32)
    h, w = template.shape
    left = cx - w // 2
    top = cy - h // 2
    depth[top:top + h, left:left + w] = template
    return depth


def _tool(template: np.ndarray) -> ShapeMatch:
    tool = ShapeMatch()
    tool.params.update(
        {
            "angle_start": -5.0,
            "angle_end": 5.0,
            "scale_start": 1.0,
            "scale_end": 1.0,
            "min_score": 0.2,
            "max_matches": 1,
            "act_as_fixture": True,
            "show_search_roi": False,
            "show_found_region": False,
            "show_found_origin": False,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool.inputs["input image"] = VisionImage(template)
    tool.save_train_image()
    tool.params["train_roi"] = [0, 0, template.shape[1], template.shape[0], 0]
    tool.train_pattern()
    return tool


def test_shape_match_range_training_uses_full_image_limits():
    template = _template()
    tool = _tool(template)

    assert tool.params["range_vmin"] == 900.0
    assert tool.params["range_vmax"] == 1000.0
    assert tool._static_train_image.dtype == np.uint8


def test_shape_match_range_output_keeps_source_and_sets_fixture():
    template = _template()
    scene = _scene(template)
    tool = _tool(template)

    tool.inputs["input image"] = VisionImage(scene)
    tool.run()

    assert tool.last_error is None
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    output = tool.outputs["output image"]

    assert abs(result.x - 220.0) < 2.0
    assert abs(result.y - 170.0) < 2.0
    assert output.data is scene
    assert output.data.dtype == np.float32
    assert np.isclose(output.x, result.x)
    assert np.isclose(output.y, result.y)
    assert np.isclose(output.angle, result.angle)


def test_shape_match_range_uses_normalized_roi():
    template = _template()
    scene = _scene(template)
    tool = _tool(template)
    tool.inputs["input image"] = VisionImage(scene)
    tool.run()

    assert tool.last_error is None
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    assert result.solver == "geometry"
    assert abs(result.x - 220.0) < 2.0
    assert abs(result.y - 170.0) < 2.0


def test_shape_match_range_uses_training_limits_on_out_of_range_scene():
    template = _template()
    scene = np.full((320, 420), 5000.0, dtype=np.float32)
    h, w = template.shape
    left = 220 - w // 2
    top = 170 - h // 2
    scene[top:top + h, left:left + w] = template

    tool = _tool(template)
    tool.inputs["input image"] = VisionImage(scene)
    tool.run()

    assert tool.last_error is None
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    assert abs(result.x - 220.0) < 2.0
    assert abs(result.y - 170.0) < 2.0


def test_template_match_range_uses_training_limits_on_out_of_range_scene():
    template = _template()
    scene = np.full((320, 420), 5000.0, dtype=np.float32)
    h, w = template.shape
    left = 220 - w // 2
    top = 170 - h // 2
    scene[top:top + h, left:left + w] = template

    tool = TemplateMatch()
    tool.params.update(
        {
            "angle_start": -5.0,
            "angle_end": 5.0,
            "angle_step": 1.0,
            "scale_start": 1.0,
            "scale_end": 1.0,
            "scale_step": 0.1,
            "pyramid_levels": 1,
            "score_threshold": 0.3,
            "max_matches": 1,
            "act_as_fixture": True,
            "show_search_roi": False,
            "show_found_region": False,
            "show_found_origin": False,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool.inputs["input image"] = VisionImage(template)
    tool.save_train_image()
    tool.params["train_roi"] = [0, 0, w, h, 0]
    tool.train_pattern()
    tool.inputs["input image"] = VisionImage(scene)
    tool.run()

    assert tool.last_error is None
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    assert abs(result.x - 220.0) < 2.0
    assert abs(result.y - 170.0) < 2.0


def test_template_match_range_training_and_fixture():
    template = _template()
    scene = _scene(template)
    tool = TemplateMatch()
    tool.params.update(
        {
            "angle_start": -5.0,
            "angle_end": 5.0,
            "angle_step": 1.0,
            "scale_start": 1.0,
            "scale_end": 1.0,
            "scale_step": 0.1,
            "pyramid_levels": 1,
            "score_threshold": 0.3,
            "max_matches": 1,
            "act_as_fixture": True,
            "show_search_roi": False,
            "show_found_region": False,
            "show_found_origin": False,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    tool.inputs["input image"] = VisionImage(template)
    tool.save_train_image()
    tool.params["train_roi"] = [
        0,
        0,
        template.shape[1],
        template.shape[0],
        0,
    ]
    tool.train_pattern()

    assert tool.params["range_vmin"] == 900.0
    assert tool.params["range_vmax"] == 1000.0

    tool.inputs["input image"] = VisionImage(scene)
    tool.run()

    assert tool.last_error is None
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    output = tool.outputs["output image"]
    assert abs(result.x - 220.0) < 2.0
    assert abs(result.y - 170.0) < 2.0
    assert output.data is scene
    assert np.isclose(output.x, result.x)
    assert np.isclose(output.y, result.y)

def test_edge_proposals_keep_range_geometry_detection():
    template = _template()
    scene = _scene(template)
    tool = _tool(template)
    tool.params["edge_proposals"] = True
    tool.inputs["input image"] = VisionImage(scene)
    tool.run()

    assert tool.last_error is None
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    assert abs(result.x - 220.0) < 2.0
    assert abs(result.y - 170.0) < 2.0
