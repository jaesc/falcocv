"""ROI geometry spin boxes must keep a readable text area."""

import pytest
from PyQt6.QtWidgets import QLabel, QVBoxLayout, QWidget

from falcocv.ui.style import TEXT_MUTED, disabled_css
from falcocv.ui.widgets.roi_editor import RoiEditorWidget
from falcocv.ui.widgets.roi_editor_3d import RoiEditor3DWidget

pytestmark = pytest.mark.ui

# The application QSS triggers the Qt 6.12 NoButtons layout regression.
_HOST_QSS = (
    "QDoubleSpinBox { color: #e0e0e0; background-color: #1f1f1f;"
    " border: 1px solid #333333; }"
)


def _show_editor(qtbot, editor, width=520, height=420):
    host = QWidget()
    host.setStyleSheet(_HOST_QSS)
    layout = QVBoxLayout(host)
    layout.addWidget(editor)
    qtbot.addWidget(host)
    host.resize(width, height)
    host.show()
    qtbot.wait(50)
    return host


def _assert_spins_visible(spins):
    for spin in spins:
        assert "font-size: 9px" not in spin.styleSheet()
        assert spin.text()
        assert spin.width() >= 90
        assert spin.lineEdit().width() > 10


def _assert_labels_use_default_font(editor):
    for label in editor.findChildren(QLabel):
        assert "font-size: 9px" not in label.styleSheet()


def test_roi_editor_spins_keep_text_area(qtbot):
    editor = RoiEditorWidget("ROI")
    host = _show_editor(qtbot, editor)
    assert host is not None
    _assert_spins_visible(editor.spins.values())
    _assert_labels_use_default_font(editor)


def test_roi_editor_3d_spins_keep_text_area(qtbot):
    editor = RoiEditor3DWidget()
    host = _show_editor(qtbot, editor, width=520, height=520)
    assert host is not None
    _assert_spins_visible(editor._spins_2d.values())
    _assert_labels_use_default_font(editor)


def test_disabled_roi_values_use_readable_color():
    assert f"color: {TEXT_MUTED}" in disabled_css()
