"""ShapeMatch training controls, model versioning and stale-state tests."""

import base64
import io

import cv2
import numpy as np
import pytest

from falcocv.accuracy.synthetic import make_template, render_scene
from falcocv.core.vision_image import VisionImage
from falcocv.tools.shape_match.logic import MODEL_VERSION, ShapeMatch


def _trained():
    template = make_template()
    tool = ShapeMatch()
    tool.params["angle_start"] = -30.0
    tool.params["angle_end"] = 30.0
    tool._static_train_image = template.copy()
    tool.params["train_roi"] = [
        0,
        0,
        template.shape[1],
        template.shape[0],
        0,
    ]
    tool.train_pattern()
    return tool, template


def test_model_roundtrip_keeps_mask_thresholds_and_version():
    tool, template = _trained()
    mask = np.full(template.shape[:2], 255, dtype=np.uint8)
    mask[:20, :30] = 0
    tool.set_train_mask(mask)
    tool.params["edge_low"] = 40
    tool.params["edge_high"] = 120
    tool.params["min_gradient"] = 3.0
    tool.params["max_features"] = 700
    tool.train_pattern()

    data = tool.serialize()
    loaded = ShapeMatch()
    loaded.deserialize(data)

    assert loaded.is_trained
    assert not loaded.is_model_stale
    assert loaded.params["edge_low"] == 40
    assert loaded.params["edge_high"] == 120
    assert loaded.params["min_gradient"] == 3.0
    assert loaded.params["max_features"] == 700
    assert np.array_equal(loaded.get_train_mask(), mask)

    raw = base64.b64decode(loaded.params["model_b64"])
    with np.load(io.BytesIO(raw), allow_pickle=False) as model:
        assert int(model["version"][0]) == MODEL_VERSION == 4
        assert "gray" not in model.files
        assert "color" not in model.files
        assert "edges" in model.files
        assert "mask" in model.files
        assert "train_image_token" in model.files


def test_old_model_version_is_rejected_and_cleared():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, version=np.array([2], dtype=np.int32))
    tool = ShapeMatch()
    tool.deserialize(
        {
            "type": "ShapeMatch",
            "uuid": tool.uuid,
            "name": tool.name,
            "params": {
                "model_b64": base64.b64encode(buffer.getvalue()).decode("ascii"),
                "is_trained": True,
            },
        }
    )

    assert "Unsupported ShapeMatch model version (2)" in tool.last_error
    assert tool.model_load_error is not None
    assert not tool.is_trained
    assert tool.params["model_b64"] is None
    assert tool.params["is_trained"] is False


def test_training_mask_excludes_ignored_features():
    height, width = 120, 160
    image = np.zeros((height, width, 3), dtype=np.uint8)
    cv2.rectangle(image, (50, 35), (110, 85), (255, 255, 255), -1)
    cv2.line(image, (5, 5), (40, 30), (255, 255, 255), 3)

    tool = ShapeMatch()
    tool._static_train_image = image
    tool.params["train_roi"] = [0, 0, width, height, 0]
    mask = np.full((height, width), 255, dtype=np.uint8)
    mask[:40, :45] = 0
    tool.set_train_mask(mask)
    tool.train_pattern()

    px = np.rint(tool._model_x + width / 2.0).astype(int)
    py = np.rint(tool._model_y + height / 2.0).astype(int)
    assert tool._model_x.size >= 20
    assert np.all(mask[py, px] > 0)
    assert tool.train_mask_ignored_fraction() > 0.0


def test_canny_fallback_is_visible():
    image = np.tile(np.linspace(0, 25, 100, dtype=np.uint8), (100, 1))
    tool = ShapeMatch()
    tool._static_train_image = image
    tool.params["train_roi"] = [0, 0, 100, 100, 0]
    tool.train_pattern()

    assert tool._model_canny_count < 20
    assert tool._model_canny_fallback is True
    assert tool._model_x.size >= 20


def test_training_parameter_change_marks_model_stale_and_blocks_run():
    tool, template = _trained()
    scene = render_scene(template, cx=300, cy=190, angle=12, scale=1.0)
    tool.inputs["input image"] = VisionImage(scene)

    tool.params["edge_low"] = 60
    assert tool.refresh_stale_state() is True
    tool.execute()
    assert tool.outputs["count"] == 0
    assert tool.last_error == (
        "Training parameters changed; retrain ShapeMatch."
    )

    tool.train_pattern()
    assert tool.refresh_stale_state() is False
    tool.execute()
    assert tool.last_error is None
    assert tool.outputs["count"] == 1


def test_new_training_image_marks_model_stale():
    tool, template = _trained()
    tool.inputs["input image"] = VisionImage(
        np.zeros_like(template)
    )
    tool.save_train_image()
    assert tool.refresh_stale_state() is True


def test_invalid_training_parameters_are_rejected():
    tool, _template = _trained()
    tool.params["max_features"] = 10
    with pytest.raises(ValueError, match="max_features"):
        tool.train_pattern()

    tool.params["max_features"] = 800
    tool.params["edge_low"] = 150
    tool.params["edge_high"] = 50
    with pytest.raises(ValueError, match="edge_low/edge_high"):
        tool.train_pattern()

    tool.params["edge_low"] = 50
    tool.params["edge_high"] = 150
    tool.params["min_gradient"] = -1.0
    with pytest.raises(ValueError, match="min_gradient"):
        tool.train_pattern()


def test_scene_edges_use_the_model_canny_thresholds():
    tool, _template = _trained()
    tool._model_edge_low = 120
    tool._model_edge_high = 200
    gray = np.tile(np.linspace(0, 255, 80, dtype=np.uint8), (60, 1))

    edges = tool._scene_edges(gray, [None])
    expected = cv2.Canny(gray, 120, 200)
    assert np.array_equal(edges, expected)


def test_train_roi_float32_rounding_does_not_mark_stale():
    tool, _template = _trained()
    tool.params["train_roi"][0] += 1.4e-5
    tool.params["train_roi"][1] -= 1.4e-5
    assert tool.refresh_stale_state() is False

    tool.params["train_roi"][0] += 0.1
    assert tool.refresh_stale_state() is True


def _manual_tool():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    for y in range(30, 370, 35):
        for x in range(30, 570, 35):
            cv2.rectangle(image, (x, y), (x + 18, y + 18), (255, 255, 255), -1)
    tool = ShapeMatch()
    tool.inputs["input image"] = VisionImage(image)
    tool.save_train_image()
    tool.params["train_roi"] = [0, 0, 600, 400, 0]
    tool.params["max_features"] = 800
    tool.train_pattern()
    return tool


def test_manual_feature_selection_roundtrip_and_model_presence():
    tool = _manual_tool()
    candidates = tool.training_candidates()
    selected = set(tool.selected_candidate_indices(candidates))
    assert len(selected) == tool._model_x.size == 800

    added = next(
        index for index in range(len(candidates["mag"]))
        if index not in selected
    )
    removed = next(iter(selected))
    tool.apply_manual_selection(
        candidates,
        (selected - {removed}) | {added},
        manual={added},
        edited=True,
    )

    assert tool._model_x.size == 800
    assert tool.is_manual_edit is True
    assert int(tool._model_manual.sum()) == 1

    width, height = tool._template_size
    expected_x = float(candidates["x"][added]) - width / 2.0
    expected_y = float(candidates["y"][added]) - height / 2.0
    in_model = (
        np.isclose(tool._model_x, expected_x)
        & np.isclose(tool._model_y, expected_y)
    )
    assert np.any(in_model)
    assert 0 < tool._vote_model_x.size <= 500

    data = tool.serialize()
    loaded = ShapeMatch()
    loaded.deserialize(data)
    assert loaded.is_manual_edit is True
    assert int(loaded._model_manual.sum()) == 1
    assert loaded.is_model_stale is False

    loaded.reset_auto_selection()
    assert loaded.is_manual_edit is False
    assert int(loaded._model_manual.sum()) == 0
    assert loaded._model_x.size == 800


def test_manual_flags_do_not_change_the_vote_subset():
    tool = _manual_tool()
    candidates = tool.training_candidates()
    selected = list(tool.selected_candidate_indices(candidates))

    tool.apply_manual_selection(
        candidates, selected, manual=(), edited=False
    )
    automatic_vote = (
        tool._vote_model_x.copy(),
        tool._vote_model_y.copy(),
        tool._vote_model_theta.copy(),
        tool._vote_model_mag.copy(),
    )

    tool.apply_manual_selection(
        candidates, selected, manual={selected[0]}, edited=True
    )
    manual_vote = (
        tool._vote_model_x,
        tool._vote_model_y,
        tool._vote_model_theta,
        tool._vote_model_mag,
    )

    for automatic, manual in zip(automatic_vote, manual_vote):
        assert np.array_equal(automatic, manual)


def test_spatial_feature_selection_covers_weak_region():
    strong_x, strong_y = np.meshgrid(
        np.arange(0, 48), np.arange(0, 100, 10)
    )
    weak_x, weak_y = np.meshgrid(
        np.arange(152, 200), np.arange(0, 100, 10)
    )
    xs = np.concatenate([strong_x.ravel(), weak_x.ravel()])
    ys = np.concatenate([strong_y.ravel(), weak_y.ravel()])
    mag = np.concatenate([
        np.full(strong_x.size, 10.0),
        np.full(weak_x.size, 1.0),
    ])

    selected = ShapeMatch._select_spatial_features(
        xs, ys, mag, count=80
    )

    assert len(selected) == 80
    assert np.any(xs[selected] < 100)
    assert np.any(xs[selected] >= 100)
    assert selected == ShapeMatch._select_spatial_features(
        xs, ys, mag, count=80
    )


def test_spatial_feature_selection_keeps_both_rectangle_axes():
    image = np.zeros((300, 300, 3), dtype=np.uint8)
    cv2.line(image, (60, 60), (240, 60), (255, 255, 255), 3)
    cv2.line(image, (60, 240), (240, 240), (255, 255, 255), 3)
    cv2.line(image, (60, 60), (60, 240), (60, 60, 60), 3)
    cv2.line(image, (240, 60), (240, 240), (60, 60, 60), 3)

    tool = ShapeMatch()
    tool._static_train_image = image
    tool.params["train_roi"] = [0, 0, 300, 300, 0]
    tool.params["max_features"] = 80
    tool.params["edge_low"] = 10
    tool.params["edge_high"] = 30
    tool.train_pattern()

    px = np.rint(tool._model_x + 150.0).astype(int)
    py = np.rint(tool._model_y + 150.0).astype(int)
    assert tool._model_x.size == 80
    assert np.any(px < 100)
    assert np.any(px > 200)
    assert np.any(py < 100)
    assert np.any(py > 200)


def test_manual_selection_enforces_budget_and_minimum():
    tool = _manual_tool()
    candidates = tool.training_candidates()
    selected = list(tool.selected_candidate_indices(candidates))
    added = next(
        index for index in range(len(candidates["mag"]))
        if index not in set(selected)
    )

    with pytest.raises(ValueError, match="Select between"):
        tool.apply_manual_selection(
            candidates,
            selected + [added],
            manual={added},
        )
    with pytest.raises(ValueError, match="Select between"):
        tool.apply_manual_selection(candidates, selected[:19])


def test_training_candidates_rejects_a_stale_model():
    tool = _manual_tool()
    tool.params["edge_low"] = 60
    with pytest.raises(ValueError, match="retrain"):
        tool.training_candidates()


def test_mask_must_match_the_training_crop():
    tool = ShapeMatch()
    tool._static_train_image = np.zeros((40, 60, 3), dtype=np.uint8)
    tool.params["train_roi"] = [0, 0, 60, 40, 0]
    with pytest.raises(ValueError, match="mask shape"):
        tool.set_train_mask(np.zeros((10, 10), dtype=np.uint8))
