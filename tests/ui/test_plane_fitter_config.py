"""Plane Fitter UI controls for ROI Z editing and overlays."""

import pytest

from falcocv.tools.plane_fitter.config import PlaneFitterConfig
from falcocv.tools.plane_fitter.logic import PlaneFitter


pytestmark = pytest.mark.ui


def test_overlay_section_and_toolbar_stay_in_sync(qtbot):
    tool = PlaneFitter()
    config = PlaneFitterConfig(tool)
    qtbot.addWidget(config)

    config.chk_draw_overlays.setChecked(False)
    assert tool.params["show_overlays"] is False
    assert config.chk_overlays.isChecked() is False

    config.chk_overlays.setChecked(True)
    assert tool.params["show_overlays"] is True
    assert config.chk_draw_overlays.isChecked() is True

    config.chk_copy_overlays.setChecked(False)
    assert tool.params["copy_overlays_to_input"] is False


def test_roi_z_edit_is_not_overwritten_by_refresh(qtbot):
    tool = PlaneFitter()
    tool.params["rois"] = [
        {"roi": [0.0, 0.0, 10.0, 10.0, 0.0],
         "z_min": 0.0, "z_max": 1.0}
    ]
    config = PlaneFitterConfig(tool)
    qtbot.addWidget(config)

    editor = config.roi_editor
    editor._active_index = 0
    editor._refresh()

    editor.spin_z_min.setValue(10.0)
    editor.spin_z_max.setValue(20.0)

    entry = tool.params["rois"][0]
    assert entry["z_min"] == 10.0
    assert entry["z_max"] == 20.0

    config.refresh_ui()

    assert editor.spin_z_min.value() == 10.0
    assert editor.spin_z_max.value() == 20.0
