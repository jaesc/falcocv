"""Workflow editor footer pointer readout (2D and 3D)."""

import math

import pytest

from falcocv.core.workflow import Workflow
from falcocv.tools.image_loader.logic import ImageLoader
from falcocv.ui.widgets.workflow_editor import WorkflowEditor


pytestmark = pytest.mark.ui


def _editor(qtbot):
    workflow = Workflow()
    workflow.add_tool(ImageLoader("image"))
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)
    return editor


def test_footer_updates_from_2d_scalar(qtbot):
    editor = _editor(qtbot)
    editor.viewer_2d.mouse_moved.emit(12, 34, 128)

    assert editor.label_coords.text() == "X,Y: 12, 34"
    assert editor.label_value.text() == "VALUE: 128"


def test_footer_formats_2d_float(qtbot):
    editor = _editor(qtbot)
    editor.viewer_2d.mouse_moved.emit(3, 4, 10.123456)

    assert editor.label_value.text() == "VALUE: 10.1235"


def test_footer_formats_2d_bgr_as_rgb(qtbot):
    editor = _editor(qtbot)
    # ImageViewer keeps OpenCV's BGR order.
    editor.viewer_2d.mouse_moved.emit(3, 4, (30, 20, 10))

    assert editor.label_value.text() == "VALUE: (10,20,30)"


def test_footer_clears_when_leaving_2d(qtbot):
    editor = _editor(qtbot)
    editor.viewer_2d.mouse_moved.emit(1, 2, 5)
    editor.viewer_2d.mouse_left.emit()

    assert editor.label_coords.text() == "X,Y: -, -"
    assert editor.label_value.text() == "VALUE: -"


def test_footer_updates_from_3d(qtbot):
    editor = _editor(qtbot)
    if editor.viewer_3d is None:
        pytest.skip("Viewer3D is not available")

    editor.viewer_3d.mouse_moved.emit(1.0, 2.0, 3.5)

    assert editor.label_coords.text() == "X,Y: 1.0, 2.0"
    assert editor.label_value.text() == "VALUE: Z 3.5"


def test_footer_clears_for_3d_nan_and_leave(qtbot):
    editor = _editor(qtbot)
    if editor.viewer_3d is None:
        pytest.skip("Viewer3D is not available")

    editor.viewer_3d.mouse_moved.emit(math.nan, math.nan, math.nan)
    assert editor.label_coords.text() == "X,Y: -, -"
    assert editor.label_value.text() == "VALUE: -"

    editor.viewer_3d.mouse_moved.emit(1.0, 2.0, 3.0)
    editor.viewer_3d.mouse_left.emit()
    assert editor.label_coords.text() == "X,Y: -, -"
    assert editor.label_value.text() == "VALUE: -"
