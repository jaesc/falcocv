"""ShapeMatch training UI: controls, stale banner and mask editor."""

import cv2
import numpy as np
from PyQt6.QtCore import QPoint
from PyQt6.QtWidgets import QDialog

from falcocv.accuracy.synthetic import make_template
from falcocv.tools.shape_match.config import ShapeMatchConfig
from falcocv.tools.shape_match.feature_editor import (
    _AUTO_BGR,
    _CANDIDATE_BGR,
    _CandidateGrid,
    ShapeMatchFeatureEditorDialog,
)
from falcocv.tools.shape_match.mask_editor import ShapeMatchMaskDialog
from falcocv.tools.shape_match.logic import ShapeMatch


def _manual_tool():
    image = np.zeros((400, 600, 3), dtype=np.uint8)
    for y in range(30, 370, 35):
        for x in range(30, 570, 35):
            cv2.rectangle(image, (x, y), (x + 18, y + 18), (255, 255, 255), -1)
    tool = ShapeMatch()
    tool._static_train_image = image
    tool.params["train_roi"] = [0, 0, 600, 400, 0]
    tool.params["max_features"] = 800
    tool.train_pattern()
    return tool


def test_training_controls_write_params_and_show_status(qtbot):
    tool = ShapeMatch()
    config = ShapeMatchConfig(tool)
    qtbot.addWidget(config)

    assert config.spin_edge_low.value() == 50
    assert config.spin_edge_high.value() == 150
    assert config.spin_min_gradient.value() == 0.0
    assert config.spin_max_features.value() == 800

    config.spin_edge_low.setValue(40)
    config.spin_edge_high.setValue(120)
    config.spin_min_gradient.setValue(4.0)
    config.spin_max_features.setValue(700)

    assert tool.params["edge_low"] == 40
    assert tool.params["edge_high"] == 120
    assert tool.params["min_gradient"] == 4.0
    assert tool.params["max_features"] == 700

    template = make_template()
    tool._static_train_image = template.copy()
    tool.params["train_roi"] = [
        0,
        0,
        template.shape[1],
        template.shape[0],
        0,
    ]
    tool.train_pattern()
    config.refresh_ui()

    assert "Trained:" in config.label_state.text()
    assert f"{tool._model_x.size} features" in config.label_features.text()
    assert config.btn_edit_mask.isEnabled()
    assert config.template_viewer._current_vision_image is not None

    config.spin_edge_low.setValue(60)
    assert "STALE" in config.label_state.text()
    assert tool.is_model_stale


def test_pre_gate_selector_writes_tool_param(qtbot):
    tool = ShapeMatch()
    config = ShapeMatchConfig(tool)
    qtbot.addWidget(config)

    assert config.combo_pre_gate.currentData() == "ncc"
    index = config.combo_pre_gate.findData("alignment")
    assert index >= 0
    config.combo_pre_gate.setCurrentIndex(index)
    assert tool.params["pre_gate_scorer"] == "alignment"

    assert config.combo_match_mode.currentData() == "geometry"
    assert not config.combo_pre_gate.isEnabled()
    assert config.pre_gate_group.isHidden()

    hybrid_index = config.combo_match_mode.findData("hybrid")
    assert hybrid_index >= 0
    config.combo_match_mode.setCurrentIndex(hybrid_index)
    assert tool.params["match_mode"] == "hybrid"
    assert config.combo_pre_gate.isEnabled()
    assert not config.pre_gate_group.isHidden()

    geometry_index = config.combo_match_mode.findData("geometry")
    assert geometry_index >= 0
    config.combo_match_mode.setCurrentIndex(geometry_index)
    assert tool.params["match_mode"] == "geometry"
    assert not config.combo_pre_gate.isEnabled()
    assert config.pre_gate_group.isHidden()


def test_preview_renders_features_at_display_resolution(qtbot):
    import cv2
    import numpy as np

    image = np.zeros((400, 600, 3), dtype=np.uint8)
    cv2.rectangle(image, (80, 60), (520, 340), (255, 255, 255), -1)
    cv2.putText(
        image,
        "FALCO",
        (120, 220),
        cv2.FONT_HERSHEY_SIMPLEX,
        3.0,
        (0, 0, 0),
        8,
    )

    tool = ShapeMatch()
    config = ShapeMatchConfig(tool)
    qtbot.addWidget(config)
    tool._static_train_image = image
    tool.params["train_roi"] = [0, 0, 600, 400, 0]
    tool.train_pattern()
    config.refresh_ui()

    preview = config.template_viewer._current_vision_image
    assert preview is not None
    assert max(preview.data.shape[:2]) <= 250
    assert config._preview_scale_x > 1.0
    yellow = np.count_nonzero(
        np.all(
            preview.data == np.array([0, 255, 255], dtype=np.uint8),
            axis=2,
        )
    )
    assert yellow > 0


def test_mask_dialog_starts_with_full_mask_and_can_paint(qtbot):
    crop = make_template()
    dialog = ShapeMatchMaskDialog(crop, None)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    assert dialog.mask.shape == crop.shape[:2]
    assert np.all(dialog.mask == 255)

    dialog._set_mode(0)
    dialog.canvas.mask[10:20, 10:20] = 0
    dialog.canvas.update()
    assert np.any(dialog.mask == 0)

    dialog._set_mode(255)
    dialog.canvas.mask[:] = 255
    assert np.all(dialog.mask == 255)


def test_mask_canvas_paints_hover_brush_cursor(qtbot):
    crop = make_template()
    dialog = ShapeMatchMaskDialog(crop, None)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    canvas = dialog.canvas
    canvas._hover_pos = QPoint(
        canvas.width() // 2, canvas.height() // 2
    )
    canvas.update()
    qtbot.wait(20)

    assert not canvas.grab().isNull()


def test_feature_editor_removes_adds_undoes_and_applies(qtbot):
    tool = _manual_tool()
    dialog = ShapeMatchFeatureEditorDialog(tool)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    assert len(dialog._selected) == 800
    assert "800 / 800" in dialog.label_status.text()
    assert dialog.spin_brush.value() == 6

    selected = next(iter(dialog._selected))
    dialog._set_mode("remove")
    dialog._on_click(int(dialog._xs[selected]), int(dialog._ys[selected]))
    after_remove = len(dialog._selected)
    assert 20 < after_remove < 800

    unselected = next(
        index for index in range(len(dialog._xs))
        if index not in dialog._selected
    )
    dialog._set_mode("add")
    dialog._on_click(int(dialog._xs[unselected]), int(dialog._ys[unselected]))
    assert len(dialog._selected) > after_remove
    assert dialog._undo_stack

    dialog._undo()
    assert len(dialog._selected) == after_remove

    unselected_after_undo = next(
        index for index in range(len(dialog._xs))
        if index not in dialog._selected and index != unselected
    )
    x = int(dialog._xs[unselected_after_undo])
    y = int(dialog._ys[unselected_after_undo])
    dialog.viewer.stroke_started.emit(x, y)
    dialog.viewer.stroke_finished.emit(x, y)
    assert len(dialog._selected) > after_remove
    dialog._apply()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert tool.is_manual_edit is True
    assert tool._model_x.size == len(dialog._selected)
    assert int(tool._model_manual.sum()) >= 1


def _find_brush_center(dialog, radius=20, minimum=2):
    for index in sorted(dialog._selected):
        x = int(dialog._xs[index])
        y = int(dialog._ys[index])
        hits = [
            candidate
            for candidate in dialog._grid.query(x, y, radius)
            if candidate in dialog._selected
        ]
        if len(hits) >= minimum:
            return x, y, len(hits)
    raise AssertionError("No suitable brush centre found")


def test_feature_editor_brush_removes_and_adds_with_one_undo_step(qtbot):
    tool = _manual_tool()
    dialog = ShapeMatchFeatureEditorDialog(tool)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    dialog.spin_brush.setValue(20)
    x, y, _hits = _find_brush_center(dialog)
    before = set(dialog._selected)
    undo_before = len(dialog._undo_stack)

    dialog._set_mode("remove")
    dialog._begin_stroke(x, y)
    dialog._end_stroke(x, y)
    after_remove = set(dialog._selected)

    assert len(before) - len(after_remove) >= 2
    assert len(dialog._undo_stack) == undo_before + 1

    dialog._set_mode("add")
    dialog._begin_stroke(x, y)
    dialog._end_stroke(x, y)

    assert len(dialog._selected) > len(after_remove)
    assert len(dialog._undo_stack) == undo_before + 2

    dialog._undo()
    assert dialog._selected == after_remove
    dialog._undo()
    assert dialog._selected == before


def test_feature_editor_brush_respects_feature_budget(qtbot):
    tool = _manual_tool()
    dialog = ShapeMatchFeatureEditorDialog(tool)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    assert len(dialog._selected) == 800
    unselected = next(
        index
        for index in range(len(dialog._xs))
        if index not in dialog._selected
    )
    x = int(dialog._xs[unselected])
    y = int(dialog._ys[unselected])

    dialog._set_mode("add")
    dialog._begin_stroke(x, y)
    dialog._end_stroke(x, y)

    assert len(dialog._selected) == 800
    assert "Budget full" in dialog.label_status.text()


def test_brush_interpolation_covers_the_stroke_path(qtbot):
    tool = _manual_tool()
    dialog = ShapeMatchFeatureEditorDialog(tool)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    dialog.spin_brush.setValue(20)
    points = list(dialog._interpolated_points(10, 10, 210, 10))
    assert points[-1] == (210, 10)
    gaps = [
        abs(points[index + 1][0] - points[index][0])
        for index in range(len(points) - 1)
    ]
    assert max(gaps) <= 11
    assert len(points) >= 2


def test_candidate_grid_matches_bruteforce():
    rng = np.random.default_rng(7)
    xs = rng.integers(-20, 140, size=300)
    ys = rng.integers(-20, 140, size=300)
    grid = _CandidateGrid(xs, ys, cell_size=7)

    for x, y, radius in (
        (0, 0, 1),
        (35, 47, 8),
        (119, 121, 15),
        (-10, 60, 12),
    ):
        expected = {
            index
            for index, (px, py) in enumerate(zip(xs, ys))
            if (int(px) - x) ** 2 + (int(py) - y) ** 2 <= radius**2
        }
        assert set(grid.query(x, y, radius)) == expected


def test_feature_editor_uses_candidate_and_feature_colors(qtbot):
    tool = _manual_tool()
    dialog = ShapeMatchFeatureEditorDialog(tool)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    assert np.all(
        dialog._display.data[dialog._ys, dialog._xs]
        == np.asarray(_CANDIDATE_BGR, dtype=np.uint8)
    )
    colors = {
        tuple(element.data["color"])
        for element in dialog._layer.elements
    }
    assert _AUTO_BGR in colors

    index = next(iter(dialog._selected))
    x = int(dialog._xs[index])
    y = int(dialog._ys[index])
    dialog._on_mouse_moved(x, y, None)
    assert dialog.viewer._brush_cursor is not None

    dialog._set_mode("pan")
    assert dialog.viewer._brush_cursor is None

def test_noise_filter_controls_write_params(qtbot):
    tool = ShapeMatch()
    config = ShapeMatchConfig(tool)
    qtbot.addWidget(config)

    assert config.combo_edge_filter.currentData() == "auto"
    assert not config.spin_edge_filter_area.isEnabled()

    config.combo_edge_filter.setCurrentIndex(
        config.combo_edge_filter.findData("manual")
    )
    assert config.spin_edge_filter_area.isEnabled()
    config.spin_edge_filter_area.setValue(54)
    assert tool.params["edge_filter_mode"] == "manual"
    assert tool.params["edge_filter_min_area"] == 54

    config.combo_edge_filter.setCurrentIndex(
        config.combo_edge_filter.findData("auto")
    )
    assert not config.spin_edge_filter_area.isEnabled()
    assert tool.params["edge_filter_mode"] == "auto"

def test_edge_proposals_checkbox_writes_param(qtbot):
    tool = ShapeMatch()
    config = ShapeMatchConfig(tool)
    qtbot.addWidget(config)

    assert config.chk_edge_proposals.isChecked()
    config.chk_edge_proposals.setChecked(False)
    assert tool.params["edge_proposals"] is False
    config.chk_edge_proposals.setChecked(True)
    assert tool.params["edge_proposals"] is True


def test_settings_panel_is_compact_and_fits_saved_width(qtbot):
    tool = ShapeMatch()
    config = ShapeMatchConfig(tool)
    qtbot.addWidget(config)
    config.resize(1200, 800)
    config.show()
    qtbot.wait(50)

    # Regression guard: the old layout demanded ~670 px and forced a wide
    # panel with a horizontal scrollbar.  HeightBlobs 3D is the reference.
    assert config.left_panel.minimumSizeHint().width() <= 300

    config.left_scroll.setFixedWidth(293)
    qtbot.wait(20)
    assert config.left_scroll.horizontalScrollBar().maximum() == 0
