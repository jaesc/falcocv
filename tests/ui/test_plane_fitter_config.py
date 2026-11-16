"""Plane Fitter UI controls for ROI Z editing and overlays."""

import numpy as np
import pytest

from falcocv.core.pointcloud_utils import ensure_pointcloud_data
from falcocv.core.vision_image import VisionImage
from falcocv.tools.plane_fitter.config import PlaneFitterConfig
from falcocv.tools.plane_fitter.logic import PlaneFitter


pytestmark = pytest.mark.ui


def _make_3d_image():
    h = w = 200
    y, x = np.mgrid[0:h, 0:w]
    depth = (10.0 + 0.01 * x + 0.005 * y).astype(np.float32)
    image = VisionImage(
        depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0
    )
    ensure_pointcloud_data(image, target_points=50_000)
    return image


def _make_rois():
    return [
        {"roi": [20.0, 20.0, 60.0, 60.0, 0.0], "z_min": 9.0, "z_max": 12.0},
        {"roi": [100.0, 100.0, 60.0, 60.0, 0.0], "z_min": 9.0, "z_max": 12.0},
    ]


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


def test_plane_and_roi_overlay_checkboxes_update_tool(qtbot):
    tool = PlaneFitter()
    config = PlaneFitterConfig(tool)
    qtbot.addWidget(config)

    assert config.chk_plane.isChecked() is True
    assert config.chk_search_roi.isChecked() is True

    config.chk_plane.setChecked(False)
    assert tool.params["show_plane"] is False

    config.chk_search_roi.setChecked(False)
    assert tool.params["show_search_roi"] is False

    tool.params["show_plane"] = True
    tool.params["show_search_roi"] = True
    config.refresh_ui()

    assert config.chk_plane.isChecked() is True
    assert config.chk_search_roi.isChecked() is True


def test_show_3d_pushes_active_and_secondary_rois(qtbot):
    tool = PlaneFitter()
    tool.set_input("input image", _make_3d_image())
    tool.params["rois"] = _make_rois()

    config = PlaneFitterConfig(tool)
    qtbot.addWidget(config)
    config.resize(1000, 700)
    config.show()
    qtbot.wait(20)

    config.chk_show3d.setChecked(True)
    qtbot.wait(20)

    viewer = config.viewer3d
    assert viewer._roi_visible is True
    assert len(viewer._roi_boxes) == 1
    assert viewer._roi_boxes_visible is True
    active_center = viewer._roi_center.copy()

    config.roi_editor.roi_list.setCurrentRow(1)
    qtbot.wait(20)

    assert viewer._roi_visible is True
    assert len(viewer._roi_boxes) == 1
    assert not np.allclose(viewer._roi_center, active_center)
    assert not np.allclose(viewer._roi_boxes[0][0], viewer._roi_center)


def test_auto_z_margin_spin_updates_tool(qtbot):
    tool = PlaneFitter()
    config = PlaneFitterConfig(tool)
    qtbot.addWidget(config)

    assert config.spin_auto_margin.value() == 3.0
    assert config.roi_editor._auto_z_margin() == 3.0

    config.spin_auto_margin.setValue(5.5)
    assert tool.params["auto_z_margin"] == 5.5
    assert config.roi_editor._auto_z_margin() == 5.5


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
