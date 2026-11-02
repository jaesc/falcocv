"""ShapeMatch training UI: controls, stale banner and mask editor."""

import cv2
import numpy as np
from PyQt6.QtWidgets import QDialog

from falcocv.accuracy.synthetic import make_template
from falcocv.tools.shape_match.config import ShapeMatchConfig
from falcocv.tools.shape_match.feature_editor import (
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


def test_feature_editor_removes_adds_undoes_and_applies(qtbot):
    tool = _manual_tool()
    dialog = ShapeMatchFeatureEditorDialog(tool)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    assert len(dialog._selected) == 800
    assert "800 / 800" in dialog.label_status.text()

    selected = next(iter(dialog._selected))
    dialog._set_mode("remove")
    dialog._on_click(int(dialog._xs[selected]), int(dialog._ys[selected]))
    assert len(dialog._selected) == 799

    unselected = next(
        index for index in range(len(dialog._xs))
        if index not in dialog._selected
    )
    dialog._set_mode("add")
    dialog._on_click(int(dialog._xs[unselected]), int(dialog._ys[unselected]))
    assert len(dialog._selected) == 800
    assert dialog._undo_stack

    dialog._undo()
    assert len(dialog._selected) == 799

    unselected_after_undo = next(
        index for index in range(len(dialog._xs))
        if index not in dialog._selected and index != unselected
    )
    x = int(dialog._xs[unselected_after_undo])
    y = int(dialog._ys[unselected_after_undo])
    dialog.viewer.mouse_clicked.emit(x, y)
    dialog.viewer.mouse_clicked.emit(x, y)
    assert len(dialog._selected) == 800
    dialog._apply()
    assert dialog.result() == QDialog.DialogCode.Accepted
    assert tool.is_manual_edit is True
    assert tool._model_x.size == 800
    assert int(tool._model_manual.sum()) >= 1
