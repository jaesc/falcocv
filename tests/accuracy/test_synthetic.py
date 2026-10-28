"""Sanity checks for the synthetic ground-truth generator."""

import numpy as np
import pytest

from falcocv.accuracy.synthetic import (
    make_template,
    pose_error,
    render_scene,
    scene_suite,
)

pytestmark = pytest.mark.accuracy


def test_scene_suite_is_deterministic_and_covers_effects():
    scenes = scene_suite()
    names = [scene.name for scene in scenes]
    assert names[0] == "baseline"
    assert "rotation+15" in names
    assert "occlusion30" in names
    assert "combo" in names
    assert len(scenes) == len(set(names))
    assert all(scene.image.shape == (360, 480, 3) for scene in scenes)
    assert all(scene.template is scenes[0].template for scene in scenes)


def test_ground_truth_pose_error_is_zero():
    scene = scene_suite()[0]
    error = pose_error(scene, scene.cx, scene.cy, scene.angle, scene.scale)
    assert error.translation_px == pytest.approx(0.0)
    assert error.rotation_deg == pytest.approx(0.0)
    assert error.scale_rel == pytest.approx(0.0)


def test_render_scene_places_part_at_known_center():
    template = make_template()
    image = render_scene(template, cx=200.0, cy=150.0, angle=0.0, scale=1.0)
    # The template's bright body is 20 px left of its center; the dark hole is
    # at the center itself.
    assert image[150, 180].mean() > 150
    # Sample inside the hole but away from the diagonal rib that crosses it.
    assert image[142, 200].mean() < 100
    assert image.shape == (360, 480, 3)
    assert image.dtype == np.uint8


def test_rotation_is_reported_with_shortest_angle():
    scene = scene_suite()[0]
    error = pose_error(scene, scene.cx, scene.cy, 179.0, scene.scale)
    assert error.rotation_deg == pytest.approx(179.0)
    error = pose_error(scene, scene.cx, scene.cy, 181.0, scene.scale)
    assert error.rotation_deg == pytest.approx(179.0)
