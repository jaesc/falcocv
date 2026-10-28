"""Tests for workflow_tree array shape formatting (prevents 1D array crashes)."""

import numpy as np


def test_format_array_shape_various_dimensions():
    from falcocv.ui.widgets.workflow_tree import _format_array_shape

    assert _format_array_shape(np.array(42)) == "[scalar]"
    assert _format_array_shape(np.array([1, 2, 3])) == "[3]"
    assert _format_array_shape(np.zeros((10, 20))) == "[10x20]"
    assert _format_array_shape(np.zeros((10, 20, 3))) == "[10x20x3]"
    assert _format_array_shape(np.zeros((2, 3, 4, 5))) == "[2x3x4x5]"
