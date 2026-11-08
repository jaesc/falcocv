"""ShapeMatch v4 integration tests against synthetic ground truth."""

import base64
import io
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
from falcocv.tools.shape_match.logic import MODEL_VERSION, ShapeMatch

pytestmark = pytest.mark.integration


def _tool(template, max_matches=1):
    tool = ShapeMatch()
    tool.params.update(
        {
            "angle_start": -30.0,
            "angle_end": 30.0,
            "scale_start": 0.9,
            "scale_end": 1.1,
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
        # Edge positions on a resampled rotated part carry a small synthetic
        # bias; TemplateMatch shows the same offset on these scenes.
        tolerance = 0.15 if name in ("baseline", "translation") else 0.5
        assert abs(result.x - scene.cx) <= tolerance, name
        assert abs(result.y - scene.cy) <= tolerance, name
        assert abs(result.angle - scene.angle) <= 0.15, name

    for name in ("scale0.9", "scale1.1"):
        scene = scenes[name]
        result = _run(tool, scene.image)
        assert abs(result.x - scene.cx) <= 0.5, name
        assert abs(result.y - scene.cy) <= 0.5, name
        assert abs(result.angle - scene.angle) <= 0.15, name
        assert abs(result.scale - scene.scale) / scene.scale <= 0.002, name

    combo = scenes["combo"]
    result = _run(tool, combo.image)
    assert abs(result.x - combo.cx) <= 0.50
    assert abs(result.y - combo.cy) <= 0.50
    assert abs(result.angle - combo.angle) <= 0.20
    assert abs(result.scale - combo.scale) / combo.scale <= 0.002


def test_proposal_factor_reduces_the_search_view(monkeypatch):
    tool = ShapeMatch()
    tool._template_size = (120, 90)
    assert tool._proposal_factor(1) == 2
    assert tool._proposal_factor(3) == 3

    tool._template_size = (800, 600)
    assert tool._proposal_factor(1) == 8

    monkeypatch.setenv("FALCOCV_SHAPE_MATCH_PROPOSAL_FACTOR", "search")
    assert tool._proposal_factor(4) == 4
    monkeypatch.setenv("FALCOCV_SHAPE_MATCH_PROPOSAL_FACTOR", "1")
    assert tool._proposal_factor(4) == 1


def test_low_confidence_geometry_uses_the_best_edge_proposal(monkeypatch):
    scenes = {scene.name: scene for scene in scene_suite()}
    scene = scenes["baseline"]
    tool = _tool(scene.template)
    edge_x = scene.cx + 40.0
    edge_y = scene.cy + 25.0

    def fake_edge_proposals(*args, **kwargs):
        return [
            {
                "x": edge_x,
                "y": edge_y,
                "angle": 0.0,
                "scale": 1.0,
                "peak": 0.9,
            }
        ]

    def fake_solve(*args, **kwargs):
        x = float(args[6])
        y = float(args[7])
        angle = float(args[8])
        scale = float(args[9])
        near_edge = abs(x - edge_x) < 1.0 and abs(y - edge_y) < 1.0
        score = 0.80 if near_edge else 0.20
        return (
            (
                x,
                y,
                angle,
                scale,
                score,
                "geometry",
                0.5 if near_edge else 0.2,
                [],
            ),
            None,
        )

    monkeypatch.setattr(tool, "_edge_proposals", fake_edge_proposals)
    monkeypatch.setattr(tool, "_solve_pose", fake_solve)
    tool.inputs["input image"] = VisionImage(scene.image)
    tool.run()

    assert tool.last_error is None
    assert tool.outputs["count"] == 1
    result = tool.outputs["results"][0]
    assert result.score == pytest.approx(0.80)
    assert abs(result.x - edge_x) < 1.0
    assert abs(result.y - edge_y) < 1.0


def test_shape_match_occlusion_and_illumination():
    scenes = {scene.name: scene for scene in scene_suite()}
    tool = _tool(scenes["baseline"].template)

    for name in ("gamma0.7", "gamma1.4", "noise8", "blur1.5"):
        scene = scenes[name]
        result = _run(tool, scene.image)
        assert abs(result.x - scene.cx) <= 0.5, name
        assert abs(result.y - scene.cy) <= 0.5, name
        assert abs(result.angle - scene.angle) <= 0.15, name
        assert abs(result.scale - scene.scale) / scene.scale <= 0.006, name

    for name in ("occlusion20", "occlusion30"):
        scene = scenes[name]
        result = _run(tool, scene.image)
        assert result.score >= 0.5, name
        assert abs(result.x - scene.cx) <= 0.5, name
        assert abs(result.y - scene.cy) <= 0.5, name
        assert abs(result.angle - scene.angle) <= 0.15, name
        assert abs(result.scale - scene.scale) / scene.scale <= 0.006, name


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
    assert np.std(angles) <= 0.05
    assert np.std(scales) <= 0.0005
    # Geometry matches edge positions, so the perceived centre can differ
    # slightly from the synthetic warp centre; repeatability is the gate.
    assert abs(float(np.mean(xs))) <= 0.5
    assert abs(float(np.mean(ys))) <= 0.5
    assert abs(float(np.mean(angles))) <= 0.05
    assert abs(float(np.mean(scales))) <= 0.002


def test_shape_match_requires_kernel(monkeypatch):
    import falcocv.tools.shape_match.logic as shape_logic

    scenes = {scene.name: scene for scene in scene_suite()}
    combo = scenes["combo"]
    tool = _tool(combo.template)
    monkeypatch.setattr(shape_logic, "_kernel_vote_fused", None)
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
    assert baseline.solver == "geometry"
    assert baseline.score >= 0.9

    occluded = _run(tool, scenes["occlusion30"].image)
    assert occluded.solver == "geometry"
    assert occluded.score < baseline.score
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


def test_shape_match_native_icp_matches_python():
    scenes = {scene.name: scene for scene in scene_suite()}
    scene = scenes["occlusion30"]
    tool = _tool(scene.template)
    gray = cv2.cvtColor(scene.image, cv2.COLOR_BGR2GRAY)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    edges = cv2.Canny(gray, 50, 150)
    dt = cv2.distanceTransform(255 - edges, cv2.DIST_L2, 3)

    native = tool._icp_refine(gx, gy, dt, scene.cx, scene.cy, 0.0, 1.0)

    import falcocv.tools.shape_match.logic as shape_logic

    saved = shape_logic._kernel_icp
    shape_logic._kernel_icp = None
    try:
        reference = tool._icp_refine(
            gx, gy, dt, scene.cx, scene.cy, 0.0, 1.0
        )
    finally:
        shape_logic._kernel_icp = saved
    assert np.allclose(native, reference, rtol=0.0, atol=1e-3)


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
    matcher.inputs["input image"] = VisionImage(template)
    matcher.save_train_image()
    matcher.train_pattern()
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
    assert abs(result.x - 300) <= 0.5
    assert abs(result.y - 190) <= 0.5
    assert abs(result.angle - 12) <= 0.15

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
    # Hybrid matching was removed; no legacy mode switch is kept around.
    assert "match_mode" not in tool.params
    assert "pre_gate_scorer" not in tool.params

def test_shape_match_model_v4_drops_gray_and_color():
    tool = _tool(make_template())
    raw = base64.b64decode(tool.params["model_b64"])
    with np.load(io.BytesIO(raw), allow_pickle=False) as data:
        fields = {name: np.asarray(data[name]) for name in data.files}

    assert MODEL_VERSION == 4
    assert int(fields["version"][0]) == MODEL_VERSION
    assert "gray" not in fields
    assert "color" not in fields
    assert "edges" in fields

    # A v3 payload is rejected instead of being read with obsolete fields.
    fields["version"] = np.array([3], dtype=np.int32)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **fields)
    broken = ShapeMatch()
    broken.params.update(
        {
            "is_trained": True,
            "model_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
    )
    with pytest.raises(ValueError, match="model version"):
        broken._load_model()


def test_shape_match_vote_scaling_helpers():
    template = make_template()
    tool = _tool(template)
    assert 0 < tool._vote_model_x.size <= 500
    # The vote resolution depends on the template only, never on the ROI.
    assert tool._vote_factor(480, 360) == 1
    tool._template_size = (300, 300)
    assert tool._vote_factor(1000, 1000) == 3
    assert tool._vote_factor(2588, 1940) == 3

    tool._template_size = (30, 30)
    assert tool._vote_factor(2588, 1940) == 1


def test_adaptive_pre_gate_keeps_close_candidates():
    first = object()
    second = object()
    far = object()
    chosen = ShapeMatch._select_pre_gate_candidates(
        [(0.95, first), (0.94, second), (0.80, far)],
        [],
    )
    assert chosen == [first, second]


def test_adaptive_pre_gate_keeps_floor_and_vote_peaks():
    first = object()
    second = object()
    vote_peaks = [object() for _ in range(4)]
    chosen = ShapeMatch._select_pre_gate_candidates(
        [(0.95, first), (0.20, second)],
        vote_peaks,
    )
    assert chosen[:2] == [first, second]
    assert all(candidate in chosen for candidate in vote_peaks)


def test_adaptive_pre_gate_caps_close_candidates():
    candidates = [object() for _ in range(6)]
    pre_gate = [(0.95 - index * 0.001, candidate)
                for index, candidate in enumerate(candidates)]
    vote_peaks = [object() for _ in range(4)]
    chosen = ShapeMatch._select_pre_gate_candidates(
        pre_gate,
        vote_peaks,
    )
    assert len(chosen) == 8
    assert all(candidate in chosen for candidate in candidates[:4])
    assert all(candidate in chosen for candidate in vote_peaks)


def test_geometric_alignment_pre_gate_detects():
    template = make_template()
    scene = render_scene(
        template,
        cx=240.0,
        cy=180.0,
        angle=12.0,
        scale=1.05,
        gamma=0.9,
        noise=3.0,
        scene_size=(360, 480),
        seed=7,
    )
    tool = _tool(template)
    result = _run(tool, scene)
    assert abs(result.x - 240.0) <= 0.5
    assert abs(result.y - 180.0) <= 0.5
    assert abs(result.angle - 12.0) <= 0.1


def test_geometry_is_the_only_match_mode():
    tool = ShapeMatch()
    assert "match_mode" not in tool.params
    assert "pre_gate_scorer" not in tool.params


def test_geometry_only_mode_detects_baseline():
    template = make_template()
    scene = render_scene(
        template,
        cx=240.0,
        cy=180.0,
        angle=12.0,
        scale=1.05,
        gamma=0.9,
        noise=3.0,
        scene_size=(360, 480),
        seed=7,
    )
    tool = _tool(template)
    result = _run(tool, scene)
    assert result.solver == "geometry"
    assert abs(result.x - 240.0) <= 0.5
    assert abs(result.y - 180.0) <= 0.5
    assert abs(result.angle - 12.0) <= 0.2
    assert abs(result.scale - 1.05) / 1.05 <= 0.01


def _quadratic_scan(self, gx, gy, dt, x, y, angle, scale, span, step,
                    model_mask=None, subpixel=True):
    offsets = np.arange(-span, span + step * 0.5, step, dtype=np.float64)
    dx, dy = np.meshgrid(offsets, offsets, indexing="ij")
    score = np.exp(-(((dx - 2.6) ** 2 + (dy + 1.4) ** 2) / 72.0))
    index = int(np.argmax(score))
    return (
        float(x + dx.ravel()[index]),
        float(y + dy.ravel()[index]),
        float(score.ravel()[index]),
    )


def test_staged_translation_grid_matches_a_single_grid(monkeypatch):
    monkeypatch.setattr(ShapeMatch, "_dt_scan_xy", _quadratic_scan)
    tool = ShapeMatch()
    for start in ((0.0, 0.0), (100.0, 200.0), (37.5, 88.25)):
        single = tool._dt_scan_xy(
            None, None, None, *start, 0.0, 1.0, 10.0, 2.0
        )
        staged = tool._dt_scan_xy_staged(
            None, None, None, *start, 0.0, 1.0, 10.0, 2.0
        )
        assert staged == pytest.approx(single)


def test_staged_translation_grid_keeps_small_spans(monkeypatch):
    calls = []

    def record(self, gx, gy, dt, x, y, angle, scale, span, step,
               model_mask=None, subpixel=True):
        calls.append((span, step))
        return (x, y, 1.0)

    monkeypatch.setattr(ShapeMatch, "_dt_scan_xy", record)
    ShapeMatch()._dt_scan_xy_staged(
        None, None, None, 5.0, 6.0, 0.0, 1.0, 2.0, 1.0
    )
    assert calls == [(2.0, 1.0)]


def test_staged_translation_grid_uses_coarse_step_multiples(monkeypatch):
    calls = []

    def record(self, gx, gy, dt, x, y, angle, scale, span, step,
               model_mask=None, subpixel=True):
        calls.append((span, step))
        return (x, y, 1.0)

    monkeypatch.setattr(ShapeMatch, "_dt_scan_xy", record)
    ShapeMatch()._dt_scan_xy_staged(
        None, None, None, 0.0, 0.0, 0.0, 1.0, 10.0, 1.5
    )
    assert calls == [(10.0, 4.5), (4.5, 1.5)]


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
    result = _run(tool, warped)
    assert abs(result.x - 300.0) <= 0.5
    assert abs(result.y - 190.0) <= 0.5
    assert abs(result.angle - 12.0) <= 0.2
    assert abs(result.scale - 1.05) / 1.05 <= 0.01


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
    """Parity with TemplateMatch: the search ROI shows before training."""
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

def test_shape_match_larger_roi_keeps_the_match():
    """A bigger search ROI must never lose a match found in a smaller one."""
    template = make_template()
    scene = render_scene(
        template,
        cx=300.0,
        cy=190.0,
        angle=12.0,
        scale=1.05,
        gamma=0.9,
        noise=3.0,
        seed=7,
    )
    rng = np.random.default_rng(11)
    for _ in range(60):
        x0 = int(rng.integers(0, 460))
        y0 = int(rng.integers(0, 340))
        w = int(rng.integers(8, 44))
        h = int(rng.integers(8, 44))
        if abs(x0 - 300) < 110 and abs(y0 - 190) < 110:
            continue
        shade = int(rng.integers(20, 240))
        cv2.rectangle(scene, (x0, y0), (x0 + w, y0 + h), (shade,) * 3, -1)

    tool = _tool(template)
    for roi in (
        [200.0, 90.0, 200.0, 200.0, 0.0],
        [100.0, 0.0, 380.0, 360.0, 0.0],
        [0, 0, 0, 0, 0],
    ):
        tool.params["search_roi"] = roi
        result = _run(tool, scene)
        assert abs(result.x - 300.0) <= 0.5, roi
        assert abs(result.y - 190.0) <= 0.5, roi
        assert abs(result.angle - 12.0) <= 0.2, roi
        assert abs(result.scale - 1.05) / 1.05 <= 0.01, roi

def test_pyramid_views_from_roi_block_match_full_frame():
    tool = ShapeMatch()
    tool._template_size = (120, 90)
    rng = np.random.default_rng(11)
    frame = rng.integers(0, 255, (640, 860, 3), dtype=np.uint8)
    ax, ay, aw, ah = 203, 151, 307, 241

    for vote_factor, search_factor, proposal_factor in (
        (1, 1, 1),
        (2, 4, 4),
        (3, 7, 8),
    ):
        full_gray = tool._gray(frame)
        old_views = tool._pyramid_views(
            full_gray,
            ax,
            ay,
            aw,
            ah,
            (vote_factor, search_factor, proposal_factor),
        )
        block, left, top = tool._aligned_roi_gray_block(
            frame,
            ax,
            ay,
            aw,
            ah,
            vote_factor,
            search_factor,
            proposal_factor,
        )
        new_views = tool._pyramid_views(
            block,
            ax - left,
            ay - top,
            aw,
            ah,
            (vote_factor, search_factor, proposal_factor),
        )
        for factor in (vote_factor, search_factor, proposal_factor):
            old_gray, old_map = old_views[factor]
            new_gray, new_map = new_views[factor]
            assert np.array_equal(old_gray, new_gray), (factor, "pixels")
            assert np.allclose(old_map, new_map), (factor, "map")


def test_explicit_roi_does_not_gray_the_complete_frame(monkeypatch):
    template = make_template()
    scene = render_scene(
        template,
        cx=1294,
        cy=970,
        angle=12.0,
        scale=1.05,
        gamma=0.9,
        noise=3.0,
        scene_size=(1940, 2588),
        seed=7,
    )
    tool = _tool(template)
    tool.params["search_roi"] = [1094, 770, 400, 400, 0]
    seen = []
    original_gray = tool._gray

    def record_gray(image):
        seen.append(image.shape[:2])
        return original_gray(image)

    monkeypatch.setattr(tool, "_gray", record_gray)
    tool.inputs["input image"] = VisionImage(scene)
    tool.run()

    assert seen
    assert max(height * width for height, width in seen) < 1940 * 2588
    result = tool.outputs["results"][0]
    assert abs(result.x - 1294.0) <= 0.5
    assert abs(result.y - 970.0) <= 0.5
    assert abs(result.angle - 12.0) <= 0.1


def test_shape_match_spatial_dedupe_matches_bruteforce():
    rng = np.random.default_rng(5)
    candidates = [
        {
            "x": float(rng.uniform(0, 800)),
            "y": float(rng.uniform(0, 600)),
            "angle": float(rng.choice([-20.0, -10.0, 0.0, 10.0, 20.0])),
            "scale": float(rng.choice([0.9, 1.0, 1.1])),
            "peak": float(rng.uniform(0, 1)),
        }
        for _ in range(500)
    ]

    def bruteforce(items, min_distance, angle_tolerance, scale_tolerance):
        kept = []
        for candidate in items:
            duplicate = False
            for other in kept:
                if np.hypot(
                    candidate["x"] - other["x"], candidate["y"] - other["y"]
                ) >= min_distance:
                    continue
                if abs(candidate["angle"] - other["angle"]) > angle_tolerance:
                    continue
                if abs(candidate["scale"] / other["scale"] - 1.0) > scale_tolerance:
                    continue
                duplicate = True
                break
            if not duplicate:
                kept.append(candidate)
        return kept

    fast = ShapeMatch._dedupe_candidates(
        candidates, 60.0, angle_tolerance=5.0, scale_tolerance=0.05
    )
    slow = bruteforce(candidates, 60.0, 5.0, 0.05)
    assert [(c["x"], c["y"]) for c in fast] == [(c["x"], c["y"]) for c in slow]

def test_edge_noise_filter_auto_and_manual():
    mask = np.zeros((100, 100), dtype=np.uint8)
    cv2.rectangle(mask, (10, 10), (60, 60), 255, 1)
    mask[80, 20] = 255
    mask[85, 30] = 255

    filtered, area, removed, total = ShapeMatch._filter_training_edges(
        mask, 100, 100, "auto", 0
    )
    assert total >= 3
    assert area > 0
    assert removed >= 2
    assert filtered[80, 20] == 0
    assert filtered[10, 10] > 0

    unfiltered, area0, removed0, _ = ShapeMatch._filter_training_edges(
        mask, 100, 100, "manual", 0
    )
    assert area0 == 0
    assert removed0 == 0
    assert np.array_equal(unfiltered, mask)


def test_noise_filter_is_saved_and_marks_model_stale():
    rng = np.random.default_rng(0)
    image = np.zeros((120, 240), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (220, 90), 255, 2)
    image[rng.random(image.shape) > 0.985] = 255

    tool = ShapeMatch()
    tool.params.update(
        {
            "edge_low": 50,
            "edge_high": 150,
            "min_gradient": 0.0,
            "max_features": 200,
            "edge_filter_mode": "auto",
        }
    )
    tool._static_train_image = image
    tool.params["train_roi"] = [0, 0, 240, 120, 0]
    tool.train_pattern()

    assert tool._model_edge_filter_area > 0
    assert tool._model_edge_filter_removed > 0
    assert tool._model_edge_filter_total >= tool._model_edge_filter_removed
    assert tool.params.get("model_b64")

    restored = ShapeMatch()
    restored.params.update(tool.params)
    restored._restore_state_from_params()
    assert restored._model_edge_filter_mode == "auto"
    assert restored._model_edge_filter_area == tool._model_edge_filter_area
    assert restored._model_edge_filter_removed == tool._model_edge_filter_removed
    assert not restored.refresh_stale_state()

    restored.params["edge_filter_mode"] = "manual"
    restored.params["edge_filter_min_area"] = 30
    assert restored.refresh_stale_state()
