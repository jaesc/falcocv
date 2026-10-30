"""ShapeMatch v2 integration tests against synthetic ground truth."""

import math

import cv2
import numpy as np
import pytest

from falcocv.accuracy.synthetic import BACKGROUND, make_template, render_scene, scene_suite
from falcocv.core.job_manager import JobManager
from falcocv.core.overlay_layer import OverlayType
from falcocv.core.vision_image import VisionImage
from falcocv.core.workflow import Workflow
from falcocv.tools.image_loader.logic import ImageLoader
from falcocv.tools.shape_match.logic import ShapeMatch

pytestmark = pytest.mark.integration


def _tool(template, max_matches=1):
    tool = ShapeMatch()
    tool.params.update(
        {
            "angle_start": -30.0,
            "angle_end": 30.0,
            "min_score": 0.25,
            "max_matches": max_matches,
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


def _run(tool, image):
    tool.inputs["input image"] = VisionImage(image)
    tool.run()
    assert tool.outputs["count"] >= 1
    return tool.outputs["results"][0]


def test_shape_match_baseline_rotation_and_scale():
    scenes = {scene.name: scene for scene in scene_suite()}
    tool = _tool(scenes["baseline"].template)

    for name in ("baseline", "translation", "rotation+15", "rotation-20"):
        scene = scenes[name]
        result = _run(tool, scene.image)
        assert abs(result.x - scene.cx) <= 0.15, name
        assert abs(result.y - scene.cy) <= 0.15, name
        assert abs(result.angle - scene.angle) <= 0.08, name

    for name in ("scale0.9", "scale1.1"):
        scene = scenes[name]
        result = _run(tool, scene.image)
        assert abs(result.x - scene.cx) <= 0.15, name
        assert abs(result.y - scene.cy) <= 0.15, name
        assert abs(result.angle - scene.angle) <= 0.08, name
        assert abs(result.scale - scene.scale) / scene.scale <= 0.002, name

    combo = scenes["combo"]
    result = _run(tool, combo.image)
    assert abs(result.x - combo.cx) <= 0.15
    assert abs(result.y - combo.cy) <= 0.15
    assert abs(result.angle - combo.angle) <= 0.08
    assert abs(result.scale - combo.scale) / combo.scale <= 0.002


def test_shape_match_occlusion_and_illumination():
    scenes = {scene.name: scene for scene in scene_suite()}
    tool = _tool(scenes["baseline"].template)

    for name in ("gamma0.7", "gamma1.4", "noise8", "blur1.5"):
        scene = scenes[name]
        result = _run(tool, scene.image)
        assert abs(result.x - scene.cx) <= 0.15, name
        assert abs(result.y - scene.cy) <= 0.15, name
        assert abs(result.angle - scene.angle) <= 0.08, name
        assert abs(result.scale - scene.scale) / scene.scale <= 0.002, name

    for name in ("occlusion20", "occlusion30"):
        scene = scenes[name]
        result = _run(tool, scene.image)
        assert result.score >= 0.5, name
        assert abs(result.x - scene.cx) <= 0.15, name
        assert abs(result.y - scene.cy) <= 0.15, name
        assert abs(result.angle - scene.angle) <= 0.08, name
        assert abs(result.scale - scene.scale) / scene.scale <= 0.002, name


def _two_part_scene(template):
    scene = np.full((360, 480, 3), BACKGROUND, dtype=np.uint8)
    height, width = template.shape[:2]
    for cx, cy, angle in ((180.0, 150.0, 0.0), (360.0, 230.0, 15.0)):
        # Positive angle is clockwise (FalcoCV convention).
        matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -angle, 1.0)
        matrix[0, 2] += cx - width / 2
        matrix[1, 2] += cy - height / 2
        warped = cv2.warpAffine(template, matrix, (480, 360), borderValue=(BACKGROUND,) * 3)
        mask = cv2.warpAffine(
            np.full((height, width), 255, np.uint8),
            matrix,
            (480, 360),
            flags=cv2.INTER_NEAREST,
            borderValue=0,
        )
        scene = np.where(mask[..., None] > 0, warped, scene)
    return scene


def test_shape_match_multiple_instances():
    template = make_template()
    tool = _tool(template, max_matches=2)
    tool.inputs["input image"] = VisionImage(_two_part_scene(template))
    tool.run()

    assert tool.outputs["count"] == 2
    results = sorted(tool.outputs["results"], key=lambda result: result.x)
    assert abs(results[0].x - 180.0) <= 1.0
    assert abs(results[0].y - 150.0) <= 1.0
    assert abs(results[0].angle) <= 0.1
    assert abs(results[1].x - 360.0) <= 1.0
    assert abs(results[1].y - 230.0) <= 1.0
    assert abs(results[1].angle - 15.0) <= 0.1
    assert abs(results[1].scale - 1.0) <= 0.002


def test_shape_match_subpixel_repeatability():
    template = make_template()
    tool = _tool(template)
    offsets = (-0.4, -0.2, 0.0, 0.2, 0.4)
    xs, ys, angles, scales = [], [], [], []
    for dx in offsets:
        for dy in offsets:
            scene = render_scene(
                template,
                cx=300.0 + dx,
                cy=190.0 + dy,
                angle=12.0,
                scale=1.05,
                gamma=0.9,
                noise=3.0,
                seed=7,
            )
            result = _run(tool, scene)
            xs.append(result.x - (300.0 + dx))
            ys.append(result.y - (190.0 + dy))
            angles.append(result.angle - 12.0)
            scales.append(result.scale - 1.05)

    assert np.std(xs) <= 0.02
    assert np.std(ys) <= 0.02
    assert np.std(angles) <= 0.02
    assert np.std(scales) <= 0.0005
    assert abs(float(np.mean(xs))) <= 0.05
    assert abs(float(np.mean(ys))) <= 0.05
    assert abs(float(np.mean(angles))) <= 0.05
    assert abs(float(np.mean(scales))) <= 0.001


def test_shape_match_requires_kernel(monkeypatch):
    import falcocv.tools.shape_match.logic as shape_logic

    scenes = {scene.name: scene for scene in scene_suite()}
    combo = scenes["combo"]
    tool = _tool(combo.template)
    monkeypatch.setattr(shape_logic, "_kernel_vote", None)
    tool.inputs["input image"] = VisionImage(combo.image)
    with pytest.raises(RuntimeError, match="native _falco_kernel"):
        tool.run()


def test_shape_match_origin_offset():
    template = make_template()
    scene = render_scene(template, cx=300, cy=190, angle=0, scale=1.0)
    tool = _tool(template)
    tool.params["origin_x"] = 10.0
    tool.params["origin_y"] = 5.0
    result = _run(tool, scene)
    assert abs(result.x - 310.0) <= 0.2
    assert abs(result.y - 195.0) <= 0.2


def test_shape_match_all_matches():
    template = make_template()
    tool = _tool(template, max_matches=0)
    tool.inputs["input image"] = VisionImage(_two_part_scene(template))
    tool.run()
    assert tool.outputs["count"] == 2


def test_shape_match_solver_diagnostics():
    scenes = {scene.name: scene for scene in scene_suite()}
    tool = _tool(scenes["baseline"].template)

    baseline = _run(tool, scenes["baseline"].image)
    assert baseline.solver == "ecc"
    assert baseline.ncc_score is not None

    occluded = _run(tool, scenes["occlusion30"].image)
    assert occluded.solver == "occlusion"
    assert occluded.visible_fraction is not None
    assert occluded.visible_fraction < 1.0


def test_shape_match_native_score_matches_python():
    scenes = {scene.name: scene for scene in scene_suite()}
    scene = scenes["occlusion30"]
    tool = _tool(scene.template)
    gray = cv2.cvtColor(scene.image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.Canny(gray, 50, 150)
    dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)
    angles = np.linspace(-1.5, 1.5, 5, dtype=np.float32)
    scales = np.ones(5, dtype=np.float32)
    cx = np.full(5, scene.cx, dtype=np.float32)
    cy = np.full(5, scene.cy, dtype=np.float32)

    native = tool._dt_score(gx, gy, dt, cx, cy, angles, scales)

    import falcocv.tools.shape_match.logic as shape_logic

    saved = shape_logic._kernel_score
    shape_logic._kernel_score = None
    try:
        reference = tool._dt_score(gx, gy, dt, cx, cy, angles, scales)
    finally:
        shape_logic._kernel_score = saved
    assert np.allclose(native, reference, rtol=0.0, atol=1e-6)


def test_shape_match_rotated_training_roi():
    template = make_template()
    scene = render_scene(template, cx=300, cy=190, angle=15, scale=1.0)
    tool = _tool(template)
    tool._static_train_image = scene.copy()
    tool.params["train_roi"] = [240, 145, 120, 90, 15.0]
    tool.train_pattern()
    assert tool.is_trained
    assert tool._template_color is not None
    assert tool._model_x.size > 0
    result = _run(tool, scene)
    assert abs(result.x - 300.0) <= 1.0
    assert abs(result.y - 190.0) <= 1.0


def test_shape_match_fixture_training_roi():
    template = make_template()
    scene = render_scene(template, cx=300, cy=190, angle=0, scale=1.0)
    image = VisionImage(scene, x=25.0, y=-15.0, angle=7.0)
    tool = _tool(template)
    tool.params["train_roi"] = image.image_to_fixture([240, 145, 120, 90, 0])
    tool.inputs["input image"] = image
    tool.save_train_image()
    tool.train_pattern()
    assert tool.is_trained
    assert tool._template_size == (120, 90)


def test_shape_match_job_roundtrip(tmp_path):
    template = make_template()
    scene = render_scene(template, cx=300, cy=190, angle=12, scale=1.0, gamma=0.9)
    image_path = tmp_path / "scene.png"
    assert cv2.imwrite(str(image_path), scene)

    workflow = Workflow()
    loader = ImageLoader("Loader")
    matcher = _tool(template)
    matcher.inputs["input image"] = VisionImage(scene)
    matcher.params["train_roi"] = [240, 145, 120, 90, 0]
    matcher.save_train_image()
    loader.params.update({"mode": "File", "file_path": str(image_path)})
    workflow.add_tool(loader)
    workflow.add_tool(matcher)
    workflow.connect(loader, "output image", matcher, "input image")

    path = tmp_path / "shape.cvj"
    JobManager().save_job(workflow, str(path))

    loaded = Workflow()
    JobManager().load_job(loaded, str(path))
    loaded_matcher = next(t for t in loaded.tools if isinstance(t, ShapeMatch))
    assert loaded_matcher._static_train_image is not None
    assert loaded_matcher._template_color is not None
    loaded.run()
    assert loaded.last_error is None
    assert loaded_matcher.outputs["count"] == 1
    result = loaded_matcher.outputs["results"][0]
    assert abs(result.x - 300) <= 0.3
    assert abs(result.y - 190) <= 0.3
    assert abs(result.angle - 12) <= 0.1

def test_shape_match_drops_legacy_params():
    tool = ShapeMatch()
    tool.deserialize(
        {
            "uuid": tool.uuid,
            "name": tool.name,
            "params": {
                "angle_step": 2.0,
                "scale_step": 0.1,
                "use_kernel": False,
                "use_pyramid": True,
                "is_trained": False,
            },
        }
    )
    for key in ("angle_step", "scale_step", "use_kernel", "use_pyramid"):
        assert key not in tool.params

def test_shape_match_vote_scaling_helpers():
    template = make_template()
    tool = _tool(template)
    assert 0 < tool._vote_model_x.size <= 500
    assert tool._vote_factor(480, 360) == 1
    tool._template_size = (300, 300)
    assert tool._vote_factor(1000, 1000) == 2
    assert tool._vote_factor(2588, 1940) == 4

    tool._template_size = (30, 30)
    assert tool._vote_factor(2588, 1940) == 1

    mag = np.zeros((32, 32), dtype=np.float32)
    edges = np.zeros((32, 32), dtype=np.uint8)
    for index in range(40):
        x = index % 20 + 10
        y = index // 20 + 10
        mag[y, x] = float(index + 1)
        edges[y, x] = 255
    capped = tool._cap_scene_edges(mag, edges, target=10)
    assert int((capped > 0).sum()) == 10


def test_shape_match_large_scene_downscaled_vote():
    template = make_template()
    tool = _tool(template)
    scene = render_scene(
        template,
        cx=640.0,
        cy=480.0,
        angle=12.0,
        scale=1.05,
        gamma=0.9,
        noise=3.0,
        scene_size=(960, 1280),
        seed=7,
    )
    result = _run(tool, scene)
    assert abs(result.x - 640.0) <= 0.5
    assert abs(result.y - 480.0) <= 0.5
    assert abs(result.angle - 12.0) <= 0.2
    assert abs(result.scale - 1.05) / 1.05 <= 0.003

def test_shape_match_bright_border_template():
    template = np.full((120, 160, 3), 190, dtype=np.uint8)
    cv2.rectangle(template, (20, 20), (140, 100), (40, 40, 40), -1)
    cv2.circle(template, (80, 60), 14, (190, 190, 190), -1)
    cv2.line(template, (20, 100), (140, 20), (90, 90, 90), 4)
    cv2.rectangle(template, (28, 30), (42, 42), (190, 190, 190), -1)

    scene = np.full((360, 480, 3), 190, dtype=np.uint8)
    matrix = cv2.getRotationMatrix2D((80, 60), -12.0, 1.05)
    matrix[0, 2] += 300.0 - 80.0
    matrix[1, 2] += 190.0 - 60.0
    warped = cv2.warpAffine(
        template, matrix, (480, 360), flags=cv2.INTER_LINEAR,
        borderValue=(190, 190, 190),
    )

    tool = _tool(template)
    assert tool._template_border >= 150
    result = _run(tool, warped)
    assert abs(result.x - 300.0) <= 0.5
    assert abs(result.y - 190.0) <= 0.5
    assert abs(result.angle - 12.0) <= 0.2
    assert abs(result.scale - 1.05) / 1.05 <= 0.01


def test_shape_match_spatial_edge_cap_keeps_weak_regions():
    mag = np.zeros((200, 200), dtype=np.float32)
    edges = np.zeros((200, 200), dtype=np.uint8)
    for index in range(300):
        x = index % 20 + 10
        y = index // 20 + 10
        mag[y, x] = 10.0
        edges[y, x] = 255
    for index in range(300):
        x = index % 20 + 150
        y = index // 20 + 150
        mag[y, x] = 1.0
        edges[y, x] = 255

    capped = ShapeMatch._cap_scene_edges(mag, edges, target=200)
    ys, xs = np.nonzero(capped)
    assert np.any((xs < 100) & (ys < 100))
    assert np.any((xs > 100) & (ys > 100))


def _overlay_types(image, tool_name):
    layer = image.get_overlay_layer("default", create=False)
    if layer is None:
        return []
    return [
        element.element_type.name
        for element in layer.elements
        if element.source_tool == tool_name
    ]


def test_shape_match_overlays_reach_input_after_upstream_tool():
    """ShapeMatch overlays must be tagged with its own name and copied back.

    Regression: get_overlay_layer("default", source_tool=...) is a no-op when an
    upstream tool already created the layer, so the elements inherited the
    upstream name and _copy_overlays_to_input() dropped them.
    """
    template = make_template()
    scene = render_scene(template, cx=300, cy=190, angle=0, scale=1.0)
    tool = _tool(template)
    tool.params.update(
        {
            "show_search_roi": True,
            "show_found_region": True,
            "show_found_origin": True,
            "show_labels": True,
            "copy_overlays_to_input": True,
        }
    )

    image = VisionImage(scene)
    image.get_overlay_layer("default", source_tool="Upstream").add_contour(
        [[2, 2], [20, 2], [20, 20]]
    )
    tool.inputs["input image"] = image
    tool.run()

    assert tool.outputs["count"] == 1
    copied = _overlay_types(image, tool.name)
    assert {"ROI", "CONTOUR", "AXES", "TEXT"} <= set(copied)
    # The upstream overlay is preserved untouched.
    assert "CONTOUR" in _overlay_types(image, "Upstream")

    # Re-running must replace its own overlays, never accumulate them.
    tool.run()
    assert _overlay_types(image, tool.name) == copied


def test_shape_match_untrained_draws_search_roi():
    """Parity with PatternMatcher: the search ROI shows before training."""
    rng = np.random.default_rng(3)
    image = rng.integers(0, 80, (160, 160, 3), dtype=np.uint8)
    tool = ShapeMatch()
    tool.inputs["input image"] = VisionImage(image)
    tool.params["search_roi"] = [20, 20, 80, 80, 0]
    tool.run()

    assert tool.outputs["count"] == 0
    assert "ROI" in _overlay_types(tool.outputs["output image"], tool.name)


def test_shape_match_angle_range_is_clockwise():
    """Public angle ranges use the framework convention (positive = clockwise)."""
    template = make_template()
    tool = _tool(template)

    scene = render_scene(template, cx=300, cy=190, angle=20.0, scale=1.0)
    tool.params.update({"angle_start": 10.0, "angle_end": 30.0})
    result = _run(tool, scene)
    assert abs(result.angle - 20.0) <= 0.2

    scene = render_scene(template, cx=300, cy=190, angle=-20.0, scale=1.0)
    tool.params.update({"angle_start": -30.0, "angle_end": -10.0})
    result = _run(tool, scene)
    assert abs(result.angle + 20.0) <= 0.2

    # The same positive part must not match a negative-only range.
    scene = render_scene(template, cx=300, cy=190, angle=20.0, scale=1.0)
    tool.params.update({"angle_start": -30.0, "angle_end": -10.0})
    tool.inputs["input image"] = VisionImage(scene)
    tool.run()
    assert tool.outputs["count"] == 0


def test_shape_match_axes_follow_found_region():
    """The AXES X direction must be parallel to the found region's X edge."""
    template = make_template()
    tool = _tool(template)
    tool.params.update(
        {
            "show_found_region": True,
            "show_found_origin": True,
            "show_labels": False,
            "copy_overlays_to_input": False,
        }
    )
    scene = render_scene(template, cx=300, cy=190, angle=25.0, scale=1.0)
    result = _run(tool, scene)

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
    assert abs(result.angle - axes.data["angle"]) <= 1e-6
    assert float(np.dot(edge, axis)) > 0.999
    assert abs(float(edge[0] * axis[1] - edge[1] * axis[0])) < 1e-3


def test_shape_match_act_as_fixture_uses_framework_angle():
    """act_as_fixture must rotate downstream ROIs in the same sense as the part."""
    template = make_template()
    tool = _tool(template)
    tool.params["act_as_fixture"] = True
    scene = render_scene(template, cx=300, cy=190, angle=25.0, scale=1.0)
    result = _run(tool, scene)
    output = tool.outputs["output image"]

    assert abs(result.angle - 25.0) <= 0.2
    assert abs(output.angle - result.angle) <= 1e-6

    # Local +X/+Y offsets must move down-right for a positive (clockwise)
    # angle; the old OpenCV sign moved them up-right.
    abs_roi = output.fixture_to_image([0.0, 0.0, 10.0, 10.0, 0.0])
    centre_x = abs_roi[0] + 5.0
    centre_y = abs_roi[1] + 5.0
    assert centre_x - output.x > 0
    assert centre_y - output.y > 0


def test_shape_match_rotated_search_roi_pose():
    template = make_template()
    tool = _tool(template)
    for cx, cy, angle, roi_angle in (
        (260.0, 190.0, 20.0, 30.0),
        (245.0, 175.0, -15.0, -25.0),
    ):
        tool.params.update(
            {
                "search_roi": [250.0 - 90.0, 180.0 - 90.0, 180.0, 180.0, roi_angle],
                "angle_start": angle - 10.0,
                "angle_end": angle + 10.0,
            }
        )
        scene = render_scene(template, cx=cx, cy=cy, angle=angle, scale=1.0)
        result = _run(tool, scene)
        assert abs(result.x - cx) <= 1.0
        assert abs(result.y - cy) <= 1.0
        assert abs(result.angle - angle) <= 0.3


def test_shape_match_act_as_fixture_with_fixtured_input():
    template = make_template()
    tool = _tool(template)
    scene = render_scene(template, cx=300.0, cy=190.0, angle=30.0, scale=1.0)
    image = VisionImage(scene, x=0.0, y=0.0, angle=30.0)
    roi = image.image_to_fixture([200.0, 90.0, 200.0, 200.0, 30.0])
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
    assert abs(result.angle - 30.0) <= 0.3
    assert abs(output.angle - result.angle) <= 1e-6
