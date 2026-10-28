"""Tests that ImageViewer keeps ROIs in original image pixels even when downsampled."""

import numpy as np
import pytest

pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def _qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_roi_converted_between_original_and_display(_qapp):
    from falcocv.ui.widgets.viewer import ImageViewer
    from PyQt6.QtCore import QRectF

    viewer = ImageViewer()
    # Image larger than the 4096 display limit triggers downsampling.
    img = np.zeros((8192, 4096), dtype=np.uint8)
    viewer.set_image(img)

    assert viewer._display_scale_x > 1.0
    assert viewer._display_scale_y > 1.0

    # Original-image ROI.
    orig_roi = [1000.0, 2000.0, 500.0, 800.0]
    viewer.set_roi(orig_roi)

    item = viewer.roi_rect_item
    display_rect = item.rect()
    display_pos = item.pos()

    # The on-screen rect must be smaller than the original ROI.
    assert abs(display_rect.width() - orig_roi[2] / viewer._display_scale_x) < 1.0
    assert abs(display_rect.height() - orig_roi[3] / viewer._display_scale_y) < 1.0
    assert abs(display_pos.x() - orig_roi[0] / viewer._display_scale_x) < 1.0
    assert abs(display_pos.y() - orig_roi[1] / viewer._display_scale_y) < 1.0

    # Simulate the item reporting a change in display coords -> original coords.
    reported = QRectF(
        display_pos.x() + display_rect.x(),
        display_pos.y() + display_rect.y(),
        display_rect.width(),
        display_rect.height(),
    )
    captured = []
    viewer.roi_changed.connect(lambda r: captured.append(r))
    viewer.on_roi_item_changed(reported)

    assert len(captured) == 1
    out = captured[0]
    assert isinstance(out, list) and len(out) == 5
    assert abs(out[0] - orig_roi[0]) < 1.0
    assert abs(out[1] - orig_roi[1]) < 1.0
    assert abs(out[2] - orig_roi[2]) < 1.0
    assert abs(out[3] - orig_roi[3]) < 1.0


def test_qrectf_roi_scaled(_qapp):
    from falcocv.ui.widgets.viewer import ImageViewer
    from PyQt6.QtCore import QRectF

    viewer = ImageViewer()
    img = np.zeros((8192, 4096), dtype=np.uint8)
    viewer.set_image(img)

    orig_rect = QRectF(1000.0, 2000.0, 500.0, 800.0)
    viewer.set_roi(orig_rect)

    item = viewer.roi_rect_item
    assert abs(item.rect().x() - 1000.0 / viewer._display_scale_x) < 1.0
    assert abs(item.rect().y() - 2000.0 / viewer._display_scale_y) < 1.0
    assert abs(item.rect().width() - 500.0 / viewer._display_scale_x) < 1.0
    assert abs(item.rect().height() - 800.0 / viewer._display_scale_y) < 1.0


def test_no_downsample_no_conversion(_qapp):
    from falcocv.ui.widgets.viewer import ImageViewer

    viewer = ImageViewer()
    img = np.zeros((1000, 1000), dtype=np.uint8)
    viewer.set_image(img)

    assert viewer._display_scale_x == 1.0
    assert viewer._display_scale_y == 1.0

    viewer.set_roi([10.0, 20.0, 30.0, 40.0])
    item = viewer.roi_rect_item
    assert abs(item.rect().width() - 30.0) < 1e-3
    assert abs(item.pos().x() - 10.0) < 1e-3
