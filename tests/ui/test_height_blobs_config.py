"""Height Blobs 3D config: tabs, mask view and shared record selector."""

import numpy as np

from falcocv.core.pointcloud_utils import ensure_pointcloud_data
from falcocv.core.vision_image import VisionImage
from falcocv.tools.height_blobs.config import HeightBlobs3DConfig
from falcocv.tools.height_blobs.logic import HeightBlobs3D


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


def test_config_has_parameters_and_results_tabs(qtbot):
    tool = HeightBlobs3D()
    config = HeightBlobs3DConfig(tool)
    qtbot.addWidget(config)

    assert config.tabs.count() == 2
    assert config.tabs.tabText(0) == "Parameters"
    assert config.tabs.tabText(1) == "Results"


def test_binary_mask_hides_the_roi(qtbot):
    tool = HeightBlobs3D()
    config = HeightBlobs3DConfig(tool)
    qtbot.addWidget(config)

    index = config.record_selector.findData("binary mask")
    assert index >= 0
    config.record_selector.setCurrentIndex(index)

    assert config.viewer._roi_data is None
    assert not config.roi_editor.btn_draw.isChecked()


def test_record_selector_is_wide_enough(qtbot):
    tool = HeightBlobs3D()
    config = HeightBlobs3DConfig(tool)
    qtbot.addWidget(config)

    assert config.record_selector.minimumWidth() >= 285


def test_parameter_page_fits_a_narrow_panel(qtbot):
    tool = HeightBlobs3D()
    config = HeightBlobs3DConfig(tool)
    qtbot.addWidget(config)
    config.resize(1200, 800)
    config.show()
    qtbot.wait(20)

    assert config.params_page.minimumSizeHint().width() <= 240
    page = config.tabs.widget(0)
    assert page.horizontalScrollBar().isVisible() is False


def test_visibility_toggles_write_params(qtbot):
    tool = HeightBlobs3D()
    config = HeightBlobs3DConfig(tool)
    qtbot.addWidget(config)

    cases = (
        ("chk_draw_overlays", "show_overlays"),
        ("chk_search_roi", "show_search_roi"),
        ("chk_contours", "show_contours"),
        ("chk_centroids", "show_centroids"),
        ("chk_labels", "show_labels"),
        ("chk_copy_overlays", "copy_overlays_to_input"),
    )
    for attribute, key in cases:
        checkbox = getattr(config, attribute)
        assert checkbox.isChecked()
        checkbox.setChecked(False)
        assert tool.params[key] is False
        checkbox.setChecked(True)
        assert tool.params[key] is True


def test_view_bar_overlay_checkbox_syncs_the_master(qtbot):
    tool = HeightBlobs3D()
    config = HeightBlobs3DConfig(tool)
    qtbot.addWidget(config)

    config.chk_draw_overlays.setChecked(False)
    assert tool.params["show_overlays"] is False
    assert not config.chk_overlays.isChecked()

    config.chk_overlays.setChecked(True)
    assert tool.params["show_overlays"] is True
    assert config.chk_draw_overlays.isChecked()


def test_show_3d_pushes_active_and_secondary_rois(qtbot):
    tool = HeightBlobs3D()
    tool.set_input("input image", _make_3d_image())
    tool.params["rois"] = _make_rois()

    config = HeightBlobs3DConfig(tool)
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

    config.roi_editor.roi_list.setCurrentRow(1)
    qtbot.wait(20)

    assert viewer._roi_visible is True
    assert len(viewer._roi_boxes) == 1
    assert not np.allclose(viewer._roi_boxes[0][0], viewer._roi_center)
