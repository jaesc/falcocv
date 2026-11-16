"""Tests for 3D ROI handle interaction in Viewer3D."""

import numpy as np
import pytest
from PyQt6.QtCore import QPoint, QPointF

pytestmark = pytest.mark.ui


@pytest.fixture(scope="module")
def _qapp():
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


class _NoGLViewer3D:
    """Mixin that disables GL calls so we can test logic without a context."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._gl_initialized = True

    def _build_roi_buffers(self):
        pass

    def _upload_point_data(self):
        self._pending_upload = False

    def update(self, *args, **kwargs):
        pass

    def initializeGL(self):
        pass


def _make_viewer(_qapp):
    from falcocv.ui.widgets.viewer3d import Viewer3D

    class TestableViewer3D(_NoGLViewer3D, Viewer3D):
        pass

    v = TestableViewer3D()
    v.resize(400, 400)
    points = np.array(
        [[0, 0, 0], [20, 0, 0], [0, 20, 0], [0, 0, 20]], dtype=np.float32
    )
    colors = np.array([[255, 0, 0]] * 4, dtype=np.uint8)
    v.set_data(points=points, colors=colors)

    # Reset camera to a known position so projection is deterministic.
    v.camera.azimuth = 0.0
    v.camera.elevation = 30.0
    v.camera.distance = 100.0
    v.camera.target = np.array([10.0, 10.0, 10.0], dtype=np.float32)
    v.camera.aspect = 1.0

    # ROI centered at origin, 10 units per side.
    v.set_roi([0.0, 0.0, 0.0], [10.0, 10.0, 10.0])
    return v


def _screen_pt(v, world_pt):
    sp = v._project_world_to_screen(world_pt)
    assert sp is not None
    return QPointF(sp.x(), sp.y())



def test_set_roi_applies_the_rotation(_qapp):
    import math

    v = _make_viewer(_qapp)
    v.set_roi([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], rotation=-25.0)
    assert v._roi_rotation == -25.0
    ex, ey, _ez = v._roi_axes()
    angle = math.radians(-25.0)
    assert ex[0] == pytest.approx(math.cos(angle))
    assert ex[1] == pytest.approx(math.sin(angle))
    assert ey[0] == pytest.approx(-math.sin(angle))
    assert ey[1] == pytest.approx(math.cos(angle))


def test_rotated_face_handles_follow_the_rotated_axes(_qapp):
    v = _make_viewer(_qapp)
    v.set_roi_edit_mode(True)
    v.set_roi([0.0, 0.0, 0.0], [10.0, 10.0, 10.0], rotation=90.0)
    h = v._roi_size * 0.5
    c = v._roi_center
    ex, ey, ez = v._roi_axes()
    faces = [
        (c + ex * h[0], 0),
        (c + ey * h[1], 1),
        (c + ez * h[2], 2),
        (c - ex * h[0], 3),
        (c - ey * h[1], 4),
        (c - ez * h[2], 5),
    ]
    for fc, axis in faces:
        pt = _screen_pt(v, fc)
        assert v._hit_roi_handle(pt) == axis, f"failed for handle {axis}"


def test_shift_drag_on_center_handle_moves_the_z_band(_qapp):
    v = _make_viewer(_qapp)
    v.set_roi_edit_mode(True)
    c0 = v._roi_center.copy()
    v._roi_dragging = True
    v._roi_drag_axis = 6
    v._roi_drag_shift = True
    v._roi_drag_start_mouse = _screen_pt(v, c0)
    v._roi_drag_start_center = c0.copy()
    v._roi_drag_start_size = v._roi_size.copy()

    ez = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v._handle_roi_drag(_screen_pt(v, c0 + ez * 10.0))

    assert v._roi_center[2] > c0[2] + 1.0
    assert abs(v._roi_center[0] - c0[0]) < 1e-3
    assert abs(v._roi_center[1] - c0[1]) < 1e-3


def test_hit_center_handle(_qapp):
    v = _make_viewer(_qapp)
    v.set_roi_edit_mode(True)
    pt = _screen_pt(v, v._roi_center)
    assert v._hit_roi_handle(pt) == 6


def test_hit_face_handles(_qapp):
    v = _make_viewer(_qapp)
    v.set_roi_edit_mode(True)
    h = v._roi_size * 0.5
    c = v._roi_center
    expected = [0, 1, 2, 3, 4, 5]
    face_centers = [
        c + [h[0], 0, 0],   # +X
        c + [0, h[1], 0],   # +Y
        c + [0, 0, h[2]],   # +Z
        c - [h[0], 0, 0],   # -X
        c - [0, h[1], 0],   # -Y
        c - [0, 0, h[2]],   # -Z
    ]
    for fc, axis in zip(face_centers, expected):
        pt = _screen_pt(v, fc)
        assert v._hit_roi_handle(pt) == axis, f"failed for handle {axis}"


def test_face_handle_resizes_from_opposite_face(_qapp):
    v = _make_viewer(_qapp)
    v.set_roi_edit_mode(True)

    # Drag the +X face further in +X.
    h = v._roi_size * 0.5
    c = v._roi_center
    old_min = c[0] - h[0]
    drag_start = c + [h[0], 0, 0]
    drag_end = drag_start + [5.0, 0, 0]

    v._roi_dragging = True
    v._roi_drag_axis = 0
    v._roi_drag_start_mouse = _screen_pt(v, drag_start)
    v._roi_drag_start_center = v._roi_center.copy()
    v._roi_drag_start_size = v._roi_size.copy()

    v._handle_roi_drag(_screen_pt(v, drag_end))

    new_c = v._roi_center
    new_s = v._roi_size
    # Opposite face (-X) must stay fixed.
    assert abs((new_c[0] - new_s[0] * 0.5) - old_min) < 1e-3
    # Center and size moved in +X.
    assert new_c[0] > c[0]
    assert new_s[0] > 10.0


def test_center_handle_translates_xy(_qapp):
    v = _make_viewer(_qapp)
    v.set_roi_edit_mode(True)

    start = v._roi_center.copy()
    end = start + [3.0, -2.0, 0.0]

    v._roi_dragging = True
    v._roi_drag_axis = 6
    v._roi_drag_start_mouse = _screen_pt(v, start)
    v._roi_drag_start_center = v._roi_center.copy()
    v._roi_drag_start_size = v._roi_size.copy()
    v._roi_drag_start_hit = v._screen_to_plane(_screen_pt(v, start), start[2])

    v._handle_roi_drag(_screen_pt(v, end))

    assert abs(v._roi_center[0] - (start[0] + 3.0)) < 1e-3
    assert abs(v._roi_center[1] - (start[1] - 2.0)) < 1e-3
    assert abs(v._roi_center[2] - start[2]) < 1e-3
    assert np.allclose(v._roi_size, v._roi_drag_start_size)


def test_edit_mode_does_not_move_camera_target(_qapp):
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import QEvent, QPointF, Qt

    v = _make_viewer(_qapp)
    v.set_roi_edit_mode(True)

    # Place a point directly under the cursor and remember the target.
    target_before = v.camera.target.copy()

    # Use a position well away from handles (bottom-right corner).
    event = QMouseEvent(
        QEvent.Type.MouseButtonPress,
        QPointF(380, 380),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )
    v.mousePressEvent(event)

    assert np.allclose(v.camera.target, target_before)


def test_mouse_move_does_not_crash(_qapp):
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import QEvent, QPointF, Qt

    v = _make_viewer(_qapp)
    event = QMouseEvent(
        QEvent.Type.MouseMove,
        QPointF(100, 100),
        Qt.MouseButton.NoButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    # Should not raise (previously called self.viewport() which does not exist
    # on QOpenGLWidget).
    v.mouseMoveEvent(event)


def _mouse_event(kind, pos, button, buttons):
    from PyQt6.QtGui import QMouseEvent
    from PyQt6.QtCore import QPointF, Qt

    return QMouseEvent(
        kind,
        QPointF(*pos),
        button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )


def _right_press_move_release(v, pressed, moved, released):
    from PyQt6.QtCore import QEvent, Qt

    v.mousePressEvent(_mouse_event(
        QEvent.Type.MouseButtonPress, pressed,
        Qt.MouseButton.RightButton, Qt.MouseButton.RightButton,
    ))
    v.mouseMoveEvent(_mouse_event(
        QEvent.Type.MouseMove, moved,
        Qt.MouseButton.NoButton, Qt.MouseButton.RightButton,
    ))
    v.mouseReleaseEvent(_mouse_event(
        QEvent.Type.MouseButtonRelease, released,
        Qt.MouseButton.RightButton, Qt.MouseButton.NoButton,
    ))


def test_right_drag_pans_without_context_menu(_qapp, monkeypatch):
    v = _make_viewer(_qapp)
    opened = []
    monkeypatch.setattr(v, "_show_context_menu", lambda pos: opened.append(pos))
    target_before = v.camera.target.copy()

    _right_press_move_release(v, (200, 200), (260, 230), (260, 230))

    assert opened == []
    assert not np.allclose(v.camera.target, target_before)
    assert v._mouse_pressed is False


def test_right_drag_and_return_does_not_open_context_menu(_qapp, monkeypatch):
    v = _make_viewer(_qapp)
    opened = []
    monkeypatch.setattr(v, "_show_context_menu", lambda pos: opened.append(pos))

    _right_press_move_release(v, (200, 200), (260, 230), (200, 200))

    assert opened == []
    assert v._mouse_pressed is False


def test_right_click_without_drag_opens_context_menu(_qapp, monkeypatch):
    v = _make_viewer(_qapp)
    opened = []
    monkeypatch.setattr(v, "_show_context_menu", lambda pos: opened.append(pos))
    target_before = v.camera.target.copy()

    _right_press_move_release(v, (200, 200), (202, 201), (202, 201))

    assert len(opened) == 1
    # Sub-threshold jitter must not leave the camera panned.
    assert np.allclose(v.camera.target, target_before)
    assert v._mouse_pressed is False


def test_set_roi_boxes_stores_only_inactive_boxes(_qapp):
    v = _make_viewer(_qapp)
    boxes = [
        {"center": [0.0, 0.0, 0.0], "size": [10.0, 10.0, 10.0], "rotation": 0.0},
        {"center": [40.0, 0.0, 0.0], "size": [10.0, 10.0, 10.0], "rotation": 15.0},
    ]

    v.set_roi_boxes(boxes, active_index=0)

    assert len(v._roi_boxes) == 1
    assert np.allclose(v._roi_boxes[0][0], [40.0, 0.0, 0.0])
    assert v._roi_boxes[0][2] == pytest.approx(15.0)
    assert v._roi_boxes_visible is True

    v.show_roi(False)
    assert v._roi_boxes_visible is False


def test_hover_points_are_capped_for_large_clouds(_qapp):
    v = _make_viewer(_qapp)
    points = np.zeros((250_001, 3), dtype=np.float32)

    v.set_data(points)

    assert 0 < len(v._hover_points) <= 200_000
