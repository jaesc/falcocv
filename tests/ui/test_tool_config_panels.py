"""Splitter limits for tool configuration panels."""

import pytest

from falcocv.tools.blob_detector.config import BlobDetectorConfig
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.block.logic import Block
from falcocv.ui.widgets.tool_config_dialog import ToolConfigDialog

pytestmark = pytest.mark.ui


def _block_config(qtbot):
    block = Block("Block")
    dialog = ToolConfigDialog(block)
    qtbot.addWidget(dialog)
    dialog.resize(1000, 700)
    dialog.show()
    qtbot.wait(30)
    return dialog, dialog._config_widget


def _blob_config(qtbot):
    config = BlobDetectorConfig(BlobDetector("blob"))
    qtbot.addWidget(config)
    config.resize(1000, 700)
    config.show()
    qtbot.wait(120)
    return config


def test_block_record_selector_is_long_enough(qtbot):
    dialog, config = _block_config(qtbot)
    selector = config.editor.record_selector

    assert selector.minimumWidth() >= 360
    assert selector.width() >= 360
    dialog.close()


def test_block_left_panel_can_be_widened(qtbot):
    dialog, config = _block_config(qtbot)
    splitter = config.editor.splitter

    splitter.setSizes([900, 100])
    qtbot.wait(20)
    left, right = splitter.sizes()

    assert left >= 700
    assert right >= 200
    dialog.close()


def test_tool_left_panel_can_be_widened(qtbot):
    config = _blob_config(qtbot)
    splitter = config.content_layout

    splitter.setSizes([900, 100])
    qtbot.wait(20)
    left, right = splitter.sizes()

    assert left >= 700
    assert right >= 200
    config.close()


def test_block_viewer_can_be_collapsed(qtbot):
    dialog, config = _block_config(qtbot)
    splitter = config.editor.splitter

    splitter.setSizes([1000, 0])
    qtbot.wait(20)

    assert splitter.sizes()[1] == 0
    dialog.close()


def test_tool_viewer_can_be_collapsed(qtbot):
    config = _blob_config(qtbot)
    splitter = config.content_layout

    splitter.setSizes([1000, 0])
    qtbot.wait(20)

    assert splitter.sizes()[1] == 0
    config.close()


def test_block_left_panel_keeps_width_when_window_grows(qtbot):
    dialog, config = _block_config(qtbot)
    splitter = config.editor.splitter
    splitter.setSizes([320, 600])
    qtbot.wait(20)
    before = splitter.sizes()[0]

    dialog.resize(1400, 700)
    qtbot.wait(50)

    assert abs(splitter.sizes()[0] - before) <= 2
    dialog.close()


def test_tool_left_panel_keeps_width_when_window_grows(qtbot):
    config = _blob_config(qtbot)
    splitter = config.content_layout
    splitter.setSizes([320, 600])
    qtbot.wait(20)
    before = splitter.sizes()[0]

    config.resize(1400, 700)
    qtbot.wait(50)

    assert abs(splitter.sizes()[0] - before) <= 2
    config.close()
