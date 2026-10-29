"""Every registered tool must expose a working configuration dialog.

This is the regression net for refactors that move/import widget modules: the
``docs.tool_help`` bug made every config dialog abort the process because the
import only happened on double-click.
"""

import pytest

pytestmark = pytest.mark.ui

from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QSpinBox,
)

import falcocv.tools  # noqa: F401  (populate the registry)
from falcocv.core.tool_registry import create_tool, get_all_tools
from falcocv.ui.widgets.tool_config_dialog import ToolConfigDialog

TOOLS = get_all_tools()


@pytest.mark.parametrize("reg", TOOLS, ids=lambda r: r.tool_type)
def test_config_dialog_opens_and_closes(qtbot, reg):
    tool = create_tool(reg.tool_type)
    dialog = ToolConfigDialog(tool)
    qtbot.addWidget(dialog)

    dialog.show()
    qtbot.wait(100)

    assert dialog._config_widget is not None
    assert tool._config_window is dialog

    dialog.close()
    assert tool._config_window is None


@pytest.mark.parametrize("reg", TOOLS, ids=lambda r: r.tool_type)
def test_config_widgets_are_safe_to_edit(qtbot, reg):
    """Cycle combos / checkboxes / spin boxes without raising."""
    tool = create_tool(reg.tool_type)
    dialog = ToolConfigDialog(tool)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    widget = dialog._config_widget

    for combo in widget.findChildren(QComboBox):
        for index in range(combo.count()):
            combo.setCurrentIndex(index)
            qtbot.wait(1)

    for check in widget.findChildren(QCheckBox):
        check.setChecked(not check.isChecked())
        qtbot.wait(1)

    for spin in widget.findChildren(QSpinBox):
        spin.setValue(spin.minimum())
        spin.setValue(spin.maximum())
        qtbot.wait(1)

    for spin in widget.findChildren(QDoubleSpinBox):
        spin.setValue(spin.minimum())
        spin.setValue(spin.maximum())
        qtbot.wait(1)

    dialog.close()
    assert tool._config_window is None


def test_help_dialog_opens_for_a_tool(qtbot):
    from PyQt6.QtCore import QTimer
    from PyQt6.QtWidgets import QApplication

    from falcocv.ui.widgets.help_dialog import ToolHelpDialog

    tool = create_tool("ImageLoader")

    def close_dialog():
        for widget in QApplication.topLevelWidgets():
            if isinstance(widget, ToolHelpDialog):
                widget.close()

    QTimer.singleShot(100, close_dialog)
    # show_for_tool is modal; the timer closes it so the call returns.
    ToolHelpDialog.show_for_tool(tool)

def test_shape_match_config_has_training_ux(qtbot):
    from falcocv.accuracy.synthetic import make_template
    from falcocv.tools.shape_match.logic import ShapeMatch

    tool = ShapeMatch()
    dialog = ToolConfigDialog(tool)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)

    widget = dialog._config_widget
    assert widget.tabs.count() == 2
    keys = [widget.record_selector.itemData(i) for i in range(widget.record_selector.count())]
    assert keys == ["input image", "train image", "output image"]
    assert widget.train_roi_editor.param_key == "train_roi"
    assert widget.search_roi_editor.param_key == "search_roi"
    assert widget.template_viewer is not None

    tool._static_train_image = make_template()
    tool.params["train_roi"] = [0, 0, 120, 90, 0]
    tool.train_pattern()
    widget.refresh_ui()
    assert tool._template_color is not None
    assert widget.template_viewer._current_vision_image is not None

    dialog.close()

def test_shape_match_preview_has_fixed_square_size(qtbot):
    from falcocv.tools.shape_match.logic import ShapeMatch

    tool = ShapeMatch()
    dialog = ToolConfigDialog(tool)
    qtbot.addWidget(dialog)
    dialog.resize(1200, 800)
    dialog.show()
    widget = dialog._config_widget
    widget.tabs.setCurrentIndex(1)
    qtbot.wait(150)

    viewer = widget.template_viewer
    assert viewer.width() == 250
    assert viewer.height() == 250

    widget.content_layout.setSizes([700, 500])
    qtbot.wait(150)
    assert viewer.width() == 250
    assert viewer.height() == 250

    dialog.close()


def test_pattern_matcher_preview_has_fixed_square_size(qtbot):
    from falcocv.tools.pattern_matcher.logic import PatternMatcher

    tool = PatternMatcher()
    dialog = ToolConfigDialog(tool)
    qtbot.addWidget(dialog)
    dialog.resize(1200, 800)
    dialog.show()
    widget = dialog._config_widget
    widget.tabs.setCurrentIndex(1)
    qtbot.wait(150)

    viewer = widget.template_viewer
    assert viewer.width() == 250
    assert viewer.height() == 250

    widget.content_layout.setSizes([700, 500])
    qtbot.wait(150)
    assert viewer.width() == 250
    assert viewer.height() == 250

    dialog.close()


def test_shape_match_capture_roi_train_uses_roi(qtbot):
    from falcocv.accuracy.synthetic import make_template, render_scene
    from falcocv.core.vision_image import VisionImage
    from falcocv.tools.shape_match.logic import ShapeMatch

    scene = render_scene(make_template(), cx=300, cy=190, angle=0, scale=1.0)
    tool = ShapeMatch()
    tool.inputs["input image"] = VisionImage(scene)
    dialog = ToolConfigDialog(tool)
    qtbot.addWidget(dialog)
    dialog.show()
    qtbot.wait(50)
    widget = dialog._config_widget

    widget._on_capture()
    qtbot.wait(10)
    assert widget.record_selector.currentData() == "train image"

    # Simulate drawing a ROI in the train-image viewer.
    widget.viewer.roi_changed.emit([240.0, 145.0, 120.0, 90.0, 0.0])
    qtbot.wait(10)
    roi = list(tool.params["train_roi"])
    assert len(roi) >= 4
    assert roi[2] > 100.0
    assert roi[3] > 70.0

    widget._on_train()
    assert tool.is_trained
    assert abs(tool._template_size[0] - 120) <= 2
    assert abs(tool._template_size[1] - 90) <= 2

    dialog.close()
