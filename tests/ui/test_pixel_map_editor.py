"""Pixel-map editor: seeding, ranges, presets and mouse editing."""

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtTest import QTest
from PyQt6.QtWidgets import QApplication, QDialog, QWidget

from falcocv.core.vision_image import VisionImage
from falcocv.tools.filters.config import FiltersConfig
from falcocv.tools.filters.logic import FiltersTool
from falcocv.tools.filters.pixel_map_editor import PixelMapDialog

pytestmark = pytest.mark.ui


def _image_uint8():
    return VisionImage(
        np.arange(256, dtype=np.uint8).reshape(16, 16).copy()
    )


def _image_depth():
    rng = np.random.default_rng(0)
    depth = rng.uniform(1000, 3000, (60, 80)).astype(np.float32)
    depth[::9, ::9] = 0.0
    return VisionImage(
        depth, scale_x=0.05, scale_y=0.05, scale_z=0.01, offset_z=-20.0
    )


def _filter():
    return {
        "enabled": True,
        "type": "pixel_map",
        "domain": [0.0, 255.0],
        "points": [[0.0, 0.0], [255.0, 255.0]],
        "preserve_invalid": True,
    }


def _dialog(qtbot, image=None):
    image = image or _image_uint8()
    dialog = PixelMapDialog(None, _filter(), image.data)
    qtbot.addWidget(dialog)
    dialog.resize(760, 680)
    dialog.show()
    qtbot.wait(20)
    return dialog


def test_dialog_is_an_independent_normal_window(qtbot):
    dialog = PixelMapDialog(None, _filter(), _image_uint8().data)
    qtbot.addWidget(dialog)

    assert dialog.isWindow()
    assert dialog.parentWidget() is None
    flags = dialog.windowFlags()
    assert (flags & Qt.WindowType.WindowType_Mask) == Qt.WindowType.Window


def test_dialog_does_not_become_a_transient_of_the_host(qtbot):
    host = QWidget()
    host.setWindowTitle("host")
    qtbot.addWidget(host)
    host.show()
    qtbot.wait(10)

    dialog = PixelMapDialog(host, _filter(), _image_uint8().data)
    qtbot.addWidget(dialog)
    assert dialog.isWindow()
    assert dialog.parentWidget() is None
    assert dialog._host_window is host


def test_dialog_header_shows_the_input_depth(qtbot):
    dialog = _dialog(qtbot)
    assert "uint8" in dialog.label_input.text()
    assert "8-bit" in dialog.label_input.text()

    depth = PixelMapDialog(
        None, _filter(), _image_depth().data, scale_z=0.01, offset_z=-20.0
    )
    qtbot.addWidget(depth)
    text = depth.label_input.text()
    assert "float32" in text
    assert "mm" in text


def test_apply_range_updates_the_curve(qtbot):
    dialog = _dialog(qtbot)
    dialog.spin_in0.setValue(0)
    dialog.spin_in1.setValue(100)
    dialog.spin_out0.setValue(0)
    dialog.spin_out1.setValue(0)
    dialog._on_apply_range()

    points = dialog.plot.points()
    assert [0.0, 0.0] in points
    assert [100.0, 0.0] in points
    assert [100.0, 100.0] in points

    dialog._on_accept()
    assert dialog.result() == QDialog.DialogCode.Accepted
    result = dialog.result_filter()
    assert result["type"] == "pixel_map"
    assert result["points"] == points


def test_presets(qtbot):
    dialog = _dialog(qtbot)

    dialog.combo_preset.setCurrentIndex(
        dialog.combo_preset.findData("invert")
    )
    dialog._on_apply_preset()
    assert dialog.plot.points() == [[0.0, 255.0], [255.0, 0.0]]

    dialog.combo_preset.setCurrentIndex(
        dialog.combo_preset.findData("threshold")
    )
    dialog.spin_threshold.setValue(128)
    dialog._on_apply_preset()
    assert dialog.plot.points() == [
        [0.0, 0.0], [128.0, 0.0], [128.0, 255.0], [255.0, 255.0]
    ]

    dialog.combo_preset.setCurrentIndex(
        dialog.combo_preset.findData("identity")
    )
    dialog._on_apply_preset()
    assert dialog.plot.points() == [[0.0, 0.0], [255.0, 255.0]]


def test_double_click_adds_and_right_click_removes(qtbot):
    dialog = _dialog(qtbot)
    plot = dialog.plot

    middle = plot._to_px(128, 128)
    QTest.mouseDClick(plot, Qt.MouseButton.LeftButton, pos=middle.toPoint())
    qtbot.wait(10)
    assert any(
        abs(point[0] - 128.0) < 1.0 for point in plot.points()
    )

    target = next(
        point for point in plot.points() if abs(point[0] - 128.0) < 1.0
    )
    pixel = plot._to_px(*target)
    QTest.mousePress(
        plot, Qt.MouseButton.RightButton, pos=pixel.toPoint()
    )
    qtbot.wait(10)
    assert not any(
        abs(point[0] - 128.0) < 1.0 for point in plot.points()
    )


def test_drag_moves_a_control_point(qtbot):
    dialog = _dialog(qtbot)
    plot = dialog.plot
    middle = plot._to_px(128, 128)
    QTest.mouseDClick(plot, Qt.MouseButton.LeftButton, pos=middle.toPoint())
    qtbot.wait(10)
    before = next(
        point for point in plot.points() if abs(point[0] - 128.0) < 1.0
    )

    pixel = plot._to_px(*before)
    QTest.mousePress(plot, Qt.MouseButton.LeftButton, pos=pixel.toPoint())
    QTest.mouseMove(plot, QPoint(int(pixel.x()), int(pixel.y()) - 30))
    qtbot.wait(10)
    QTest.mouseRelease(
        plot,
        Qt.MouseButton.LeftButton,
        pos=QPoint(int(pixel.x()), int(pixel.y()) - 30),
    )
    qtbot.wait(10)

    after = next(
        point for point in plot.points() if abs(point[0] - 128.0) < 1.0
    )
    assert after[1] > before[1]


def test_config_seeds_the_domain_and_applies_the_dialog_result(
    qtbot, monkeypatch
):
    tool = FiltersTool()
    tool.inputs["input image"] = _image_depth()
    config = FiltersConfig(tool)
    qtbot.addWidget(config)

    config._add_filter()
    combo = config.filter_table.cellWidget(0, 1)
    combo.setCurrentIndex(combo.findData("pixel_map"))

    seeded = tool.params["filters"][0]
    assert seeded["domain"][0] == 0.0
    assert seeded["domain"][1] > 2000.0
    assert seeded["points"] == [
        [0.0, 0.0],
        [seeded["domain"][1], seeded["domain"][1]],
    ]

    class _FakeDialog:
        def __init__(self, *args, **kwargs):
            pass

        def exec(self):
            return QDialog.DialogCode.Accepted

        def result_filter(self):
            return {
                "enabled": True,
                "type": "pixel_map",
                "domain": [0.0, 3000.0],
                "points": [[0.0, 0.0], [3000.0, 0.0]],
                "preserve_invalid": True,
            }

    monkeypatch.setattr(
        "falcocv.tools.filters.config.PixelMapDialog", _FakeDialog
    )
    config._edit_pixel_map()

    applied = tool.params["filters"][0]
    assert applied["points"] == [[0.0, 0.0], [3000.0, 0.0]]
    assert config.filter_table.item(0, 2).text() == "2 points"


def test_output_dtype_combo_writes_the_param(qtbot):
    tool = FiltersTool()
    tool.inputs["input image"] = _image_depth()
    config = FiltersConfig(tool)
    qtbot.addWidget(config)

    index = config.combo_output_dtype.findData("uint8")
    config.combo_output_dtype.setCurrentIndex(index)
    assert tool.params["output_dtype"] == "uint8"
    assert config.combo_output_dtype.currentData() == "uint8"
