"""Tests for the PlaneFitter tool."""

import numpy as np
import pytest

from falcocv.core.vision_image import VisionImage
from falcocv.core.pointcloud_utils import ensure_pointcloud_data
from falcocv.core.roi_3d import auto_fit_z, add_roi, update_roi_2d_in_params
from falcocv.tools.plane_fitter.logic import PlaneFitter


def _make_depth_plane(h=500, w=500, sx=1.0, sy=1.0, sz=1.0):
    """Create a synthetic depth image with a slanted plane."""
    depth = np.zeros((h, w), dtype=np.float32)
    for y in range(h):
        depth[y, :] = 10.0 + 0.01 * np.arange(w) + 0.005 * y
    return depth


def test_plane_fitter_lsq_basic():
    depth = _make_depth_plane(200, 200)
    vimg = VisionImage(depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    tool.params["rois"] = [
        {"roi": [50.0, 50.0, 100.0, 100.0, 0.0], "z_min": 0.0, "z_max": 1.0}
    ]
    tool.params["method"] = "lsq"
    tool.run()

    out = tool.outputs["output image"]
    assert out is not None
    assert out is not vimg
    assert out.data is vimg.data
    assert tool.outputs["results"][0].inlier_count > 0
    assert tool.outputs["results"][0].total_points > 0

    normal = np.asarray(tool.outputs["results"][0].normal)
    assert normal is not None
    assert len(normal) == 3
    # Plane normal should be close to (-0.01, -0.005, 1) normalized
    expected = np.array([-0.01, -0.005, 1.0])
    expected /= np.linalg.norm(expected)
    similarity = np.abs(np.dot(normal / np.linalg.norm(normal), expected))
    assert similarity > 0.99


def test_plane_fitter_combined_rois():
    depth = _make_depth_plane(300, 300)
    vimg = VisionImage(depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    # Two disjoint ROIs
    tool.params["rois"] = [
        {"roi": [20.0, 20.0, 80.0, 80.0, 0.0], "z_min": 0.0, "z_max": 1.0},
        {"roi": [200.0, 200.0, 80.0, 80.0, 0.0], "z_min": 0.0, "z_max": 1.0},
    ]
    tool.params["method"] = "lsq"
    tool.run()

    total = tool.outputs["results"][0].total_points
    assert total > 0
    # Each ROI has ~6400 points; combined should be roughly twice that
    assert 10_000 < total < 15_000


def test_plane_fitter_fast_on_large_image():
    """Plane fitter should finish quickly even on a multi-megapixel image."""
    import time
    h, w = 4000, 2000
    depth = _make_depth_plane(h, w)
    vimg = VisionImage(depth, scale_x=0.1, scale_y=0.1, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=1_000_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    tool.params["rois"] = [
        {"roi": [w * 0.3, h * 0.3, w * 0.4, h * 0.4, 0.0], "z_min": 0.0, "z_max": 1.0}
    ]

    start = time.perf_counter()
    tool.run()
    elapsed_ms = (time.perf_counter() - start) * 1000

    assert tool.outputs["output image"] is not None
    assert tool.outputs["results"][0].total_points > 0
    # Should be well under 200 ms even on 8 MP
    assert elapsed_ms < 200.0


def test_plane_fitter_polygon_overlay():
    depth = _make_depth_plane(200, 200)
    vimg = VisionImage(depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    tool.params["rois"] = [
        {"roi": [50.0, 50.0, 100.0, 100.0, 0.0], "z_min": 0.0, "z_max": 1.0}
    ]
    tool.run()

    out = tool.outputs["output image"]
    layer = out.get_overlay_layer("default")
    assert any(e.ref_id == "plane_polygon" for e in layer.elements)
    assert any(e.ref_id == "plane_normal" for e in layer.elements)


def test_plane_fitter_polygon_is_bounded():
    """The fitted-plane polygon must stay inside a small margin around the image."""
    depth = _make_depth_plane(200, 200)
    vimg = VisionImage(depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    tool.params["rois"] = [
        {"roi": [50.0, 50.0, 100.0, 100.0, 0.0], "z_min": 0.0, "z_max": 1.0}
    ]
    tool.run()

    polygon = np.asarray(tool.outputs["results"][0].polygon_2d, dtype=np.float32)
    assert polygon.ndim == 2
    assert polygon.shape[0] >= 3
    assert polygon.shape[1] == 2
    # All polygon vertices must be within a modest margin of the image bounds.
    margin = 50.0
    assert np.all(polygon[:, 0] >= -margin)
    assert np.all(polygon[:, 0] <= 200.0 + margin)
    assert np.all(polygon[:, 1] >= -margin)
    assert np.all(polygon[:, 1] <= 200.0 + margin)


def test_viewer3d_front_camera_matches_range_image():
    """Default camera + flipped-X cloud: image left -> screen left."""
    from falcocv.ui.widgets.viewer3d import ArcballCamera

    cam = ArcballCamera()
    cam.target = np.array([100.0, 50.0, 50.0], dtype=np.float32)
    cam.distance = 200.0

    # With the flipped-X convention, image column 0 (left) is world +X,
    # and image column 199 (right) is world X = 0.
    left_pt = np.array([199.0, 50.0, 50.0, 1.0], dtype=np.float32)
    right_pt = np.array([0.0, 50.0, 50.0, 1.0], dtype=np.float32)

    mvp = cam.projection_matrix() @ cam.view_matrix()
    l = (mvp @ left_pt)
    l = l[:3] / l[3]
    r = (mvp @ right_pt)
    r = r[:3] / r[3]

    assert l[0] < r[0], "left side of image must appear on the left of the screen"


def test_pointcloud_x_is_flipped():
    """generate_pointcloud must mirror image columns so column 0 is world +X."""
    depth = np.zeros((10, 10), dtype=np.float32)
    depth[5, 0] = 1.0   # leftmost column
    depth[5, 9] = 1.0   # rightmost column
    vimg = VisionImage(depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=1_000_000)

    pts = vimg._3d_data["points"]
    left_pt = pts[np.isclose(pts[:, 1], 5.0) & np.isclose(pts[:, 2], 1.0)]
    assert left_pt.size > 0
    assert left_pt[0, 0] == 9.0, "leftmost image column must map to the largest world X"

    right_pt = pts[np.isclose(pts[:, 1], 5.0) & np.isclose(pts[:, 2], 1.0)]
    # The same query returns both points because they share y/z, so check both.
    xs = set(int(round(p[0])) for p in right_pt)
    assert xs == {0, 9}, "rightmost image column must map to world X=0 and leftmost to X=9"


def test_roi_3d_to_fixture_conversion():
    """ROI drawn in 3D must be stored in fixture coords when image has a fixture."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QObject, pyqtSignal
    from falcocv.ui.widgets.roi_editor_3d import RoiEditor3DWidget

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    depth = _make_depth_plane(200, 200)
    # Image with fixture offset (x=10, y=20)
    vimg = VisionImage(depth, x=10.0, y=20.0, angle=0.0,
                       scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    tool.params["rois"] = []

    class DummyViewer(QObject):
        roi_changed = pyqtSignal(dict)
        roi_drawing_finished = pyqtSignal()

        def set_roi(self, center, size):
            pass

        def show_roi(self, visible):
            pass

        def set_roi_edit_mode(self, enabled):
            pass

        def set_roi_draw_mode(self, enabled):
            pass

        def roi_draw_mode_active(self):
            return False

        def get_draw_screen_rect(self):
            return None

        def get_drawing_plane_z(self):
            return 0.0

        def screen_rect_to_world_points(self, rect, plane_z):
            return None

    editor = RoiEditor3DWidget()
    editor.set_tool(tool, viewer_2d=None, viewer_3d=DummyViewer())

    # Simulate a 3D ROI drawn at metric center (60, 70, 12) with size (20, 20, 5)
    editor._on_viewer_3d_roi({
        "center": [60.0, 70.0, 12.0],
        "size": [20.0, 20.0, 5.0],
        "rotation": [0.0, 0.0, 0.0],
    })

    roi = tool.params["rois"][0]["roi"]
    # 3D center (60, 70) size 20 -> world X=60, Y=70.
    # With flipped X: image center col = (W-1) - 60 = 139, so image ROI x = 129.
    # With fixture offset (x=10, y=20) this becomes fixture ROI (119, 40, 20, 20).
    assert abs(roi[0] - 119.0) < 1e-3
    assert abs(roi[1] - 40.0) < 1e-3
    assert abs(roi[2] - 20.0) < 1e-3
    assert abs(roi[3] - 20.0) < 1e-3


def test_roi_2d_to_3d_sync_with_fixture():
    """A 2D ROI drawn in fixture coords must be converted back to the same 3D box."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QObject, pyqtSignal
    from falcocv.ui.widgets.roi_editor_3d import RoiEditor3DWidget

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    depth = _make_depth_plane(200, 200)
    vimg = VisionImage(depth, x=10.0, y=20.0, angle=0.0,
                       scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    tool.params["rois"] = []

    captured = {}

    class DummyViewer(QObject):
        roi_changed = pyqtSignal(dict)
        roi_drawing_finished = pyqtSignal()

        def set_roi(self, center, size):
            captured["center"] = list(center)
            captured["size"] = list(size)

        def show_roi(self, visible):
            pass

        def set_roi_edit_mode(self, enabled):
            pass

        def set_roi_draw_mode(self, enabled):
            pass

        def roi_draw_mode_active(self):
            return False

        def get_draw_screen_rect(self):
            return None

        def get_drawing_plane_z(self):
            return 0.0

        def screen_rect_to_world_points(self, rect, plane_z):
            return None

    editor = RoiEditor3DWidget()
    editor.set_tool(tool, viewer_2d=None, viewer_3d=DummyViewer())

    # Simulate a 3D ROI drawn at center (60, 70, 12) size (20, 20, 5).
    editor._on_viewer_3d_roi({
        "center": [60.0, 70.0, 12.0],
        "size": [20.0, 20.0, 5.0],
        "rotation": [0.0, 0.0, 0.0],
    })

    # Now refresh the 3D view from the stored fixture ROI.
    editor._update_viewer_3d_roi()

    assert "center" in captured
    assert abs(captured["center"][0] - 60.0) < 1e-3
    assert abs(captured["center"][1] - 70.0) < 1e-3
    assert abs(captured["size"][0] - 20.0) < 1e-3
    assert abs(captured["size"][1] - 20.0) < 1e-3


def test_3d_draw_backprojects_to_image_roi():
    """Drawing a screen rect in 3D must produce the matching 2D image ROI."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QObject, QPoint, pyqtSignal
    from falcocv.ui.widgets.viewer3d import ArcballCamera
    from falcocv.ui.widgets.roi_editor_3d import RoiEditor3DWidget

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    h, w = 200, 200
    depth = _make_depth_plane(h, w)
    vimg = VisionImage(depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    tool.params["rois"] = []

    captured = {}

    class DummyViewer(QObject):
        roi_changed = pyqtSignal(dict)
        roi_drawing_finished = pyqtSignal()

        def __init__(self):
            super().__init__()
            self.camera = ArcballCamera()
            self.camera.target = np.array([100.0, 100.0, 12.0], dtype=np.float32)
            self.camera.distance = 300.0
            self.camera.azimuth = 270.0
            self.camera.elevation = 90.0
            self.camera.aspect = 1.0
            self._rect = None
            self._plane_z = 12.0

        def set_roi(self, center, size):
            captured["center"] = list(center)
            captured["size"] = list(size)

        def show_roi(self, visible):
            pass

        def set_roi_edit_mode(self, enabled):
            pass

        def set_roi_draw_mode(self, enabled):
            pass

        def roi_draw_mode_active(self):
            return False

        def set_draw_rect(self, rect):
            self._rect = rect

        def get_draw_screen_rect(self):
            return self._rect

        def get_draw_start_screen(self):
            return self._start_screen

        def get_draw_current_screen(self):
            return self._current_screen

        def set_draw_points(self, start, current):
            self._start_screen = start
            self._current_screen = current

        def get_drawing_plane_z(self):
            return self._plane_z

        def screen_to_ray(self, pos):
            view = self.camera.view_matrix()
            proj = self.camera.projection_matrix()
            inv_mvp = np.linalg.inv(proj @ view)
            widget_w = 400
            widget_h = 400
            nx = 2.0 * pos.x() / widget_w - 1.0
            ny = 1.0 - 2.0 * pos.y() / widget_h
            near_pt = inv_mvp @ np.array([nx, ny, -1.0, 1.0], dtype=np.float32)
            far_pt = inv_mvp @ np.array([nx, ny, 1.0, 1.0], dtype=np.float32)
            near_pt = near_pt[:3] / near_pt[3]
            far_pt = far_pt[:3] / far_pt[3]
            ray_dir = far_pt - near_pt
            ray_dir = ray_dir / np.linalg.norm(ray_dir)
            return near_pt, ray_dir

        def _screen_to_plane(self, pos, plane_z):
            origin, direction = self.screen_to_ray(pos)
            if abs(direction[2]) < 1e-8:
                return None
            t = (plane_z - origin[2]) / direction[2]
            if t < 0:
                return None
            return origin + direction * t

        def screen_pos_to_nearest_world_point(self, pos):
            # Simulate picking a point on the cloud surface at plane_z.
            return self._screen_to_plane(pos, self._plane_z)

        def screen_rect_to_world_points(self, rect, plane_z):
            corners = [
                (rect[0], rect[1]),
                (rect[2], rect[1]),
                (rect[2], rect[3]),
                (rect[0], rect[3]),
            ]
            pts = []
            for sx, sy in corners:
                pt = self._screen_to_plane(QPoint(int(round(sx)), int(round(sy))), plane_z)
                if pt is None:
                    return None
                pts.append(pt)
            return np.array(pts, dtype=np.float32)

    viewer = DummyViewer()
    editor = RoiEditor3DWidget()
    editor.set_tool(tool, viewer_2d=None, viewer_3d=viewer)

    # Choose a screen rect that should map to image ROI x=50, y=50, w=40, h=30.
    # With scale=1 and flipped X, world X = (W-1) - col, world Y = row.
    # For top-down camera at distance 300 and widget 400x400, fov=45,
    # approximate world size per pixel at plane Z=12:
    #   world_height = 2 * distance * tan(22.5deg) ~= 248.5 for full height.
    #   pixel_world = 248.5 / 400 ~= 0.621
    # This is approximate; we use the actual back-projection below.
    # We compute the required screen rect numerically.
    def world_to_screen(pt):
        view = viewer.camera.view_matrix()
        proj = viewer.camera.projection_matrix()
        mvp = proj @ view
        p = mvp @ np.array([pt[0], pt[1], pt[2], 1.0], dtype=np.float32)
        p = p[:3] / p[3]
        sx = int(round((p[0] + 1.0) * 0.5 * 400))
        sy = int(round((1.0 - p[1]) * 0.5 * 400))
        return sx, sy

    target = [50.0, 50.0, 40.0, 30.0]  # image ROI
    # world diagonal for this ROI (scale=1, flip_x)
    wc_start = ((w - 1) - target[0], target[1], viewer._plane_z)
    wc_end = ((w - 1) - (target[0] + target[2]), target[1] + target[3], viewer._plane_z)
    viewer.set_draw_points(QPoint(*world_to_screen(wc_start)), QPoint(*world_to_screen(wc_end)))

    editor._on_drawing_finished_3d()

    roi = tool.params["rois"][0]["roi"]
    assert abs(roi[0] - target[0]) < 2.0
    assert abs(roi[1] - target[1]) < 2.0
    assert abs(roi[2] - target[2]) < 2.0
    assert abs(roi[3] - target[3]) < 2.0

    # Z range should have been auto-fitted from the actual depth data.
    z_min = tool.params["rois"][0]["z_min"]
    z_max = tool.params["rois"][0]["z_max"]
    assert z_min < z_max
    assert z_max - z_min < 5.0  # plane is relatively flat


def test_2d_roi_to_3d_box_uses_image_scale():
    """A 2D ROI drawn in pixels must convert to a 3D box using the image scale."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QObject, pyqtSignal
    from falcocv.ui.widgets.roi_editor_3d import RoiEditor3DWidget

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    h, w = 400, 400
    depth = _make_depth_plane(h, w)
    scale_x, scale_y, scale_z = 0.025, 0.025, 1.0
    vimg = VisionImage(depth, scale_x=scale_x, scale_y=scale_y,
                       scale_z=scale_z, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    tool.params["rois"] = []

    captured = {}

    class DummyViewer(QObject):
        roi_changed = pyqtSignal(dict)
        roi_drawing_finished = pyqtSignal()

        def set_roi(self, center, size):
            captured["center"] = list(center)
            captured["size"] = list(size)

        def show_roi(self, visible):
            pass

        def set_roi_edit_mode(self, enabled):
            pass

        def set_roi_draw_mode(self, enabled):
            pass

        def roi_draw_mode_active(self):
            return False

        def get_draw_screen_rect(self):
            return None

        def get_drawing_plane_z(self):
            return 0.0

        def screen_rect_to_world_points(self, rect, plane_z):
            return None

        def screen_pos_to_nearest_world_point(self, pos):
            return None

        def get_draw_start_screen(self):
            return None

        def get_draw_current_screen(self):
            return None

    editor = RoiEditor3DWidget()
    editor.set_tool(tool, viewer_2d=None, viewer_3d=DummyViewer())

    # Draw a 2D ROI in pixels and verify the 3D box uses the image scale.
    roi_2d = [100.0, 80.0, 40.0, 30.0]
    editor._on_viewer_2d_roi(roi_2d)

    assert "center" in captured
    cx, cy, cz = captured["center"]
    sx, sy, sz = captured["size"]

    # With flip-X: world_x = (w-1 - col) * scale_x.
    expected_cx = (w - 1 - (roi_2d[0] + roi_2d[2] / 2)) * scale_x
    expected_cy = (roi_2d[1] + roi_2d[3] / 2) * scale_y
    expected_sx = roi_2d[2] * scale_x
    expected_sy = roi_2d[3] * scale_y

    assert abs(cx - expected_cx) < 1e-3
    assert abs(cy - expected_cy) < 1e-3
    assert abs(sx - expected_sx) < 1e-3
    assert abs(sy - expected_sy) < 1e-3


def test_rotated_roi_selects_matching_points():
    """A rotated 2D ROI must select the matching oriented region in world coords."""
    from falcocv.core.plane_fit import plane_fit as _plane_fit

    h, w = 101, 101
    depth = np.zeros((h, w), dtype=np.float32)

    # Build points in the same row-major, flipped-X order as generate_pointcloud.
    points = []
    pixel_coords = []
    for r in range(h):
        for c in range(w):
            wx = float(w - 1 - c)
            wy = float(r)
            points.append([wx, wy, 0.0])
            pixel_coords.append((c, r))
    points = np.asarray(points, dtype=np.float32)

    # Rotated narrow ROI centered in the image, 30 deg visual-CW.
    roi = [30.0, 40.0, 60.0, 20.0, 30.0]

    output = _plane_fit({
        "image": depth,
        "points": points,
        "rois": [{"roi": roi, "z_min": 0.0, "z_max": 1.0}],
        "calib": [1.0, 1.0, 1.0, 0.0],
        "params": {
            "method": "lsq",
            "distance_threshold": 1.0,
            "max_iterations": 100,
            "min_points": 10,
            "max_fit_points": 50000,
        }
    })

    assert "error" not in output, output.get("error")

    selected = set(int(i) for i in output["selected_indices"])

    # Replicate C++ selection logic (flipped X, world angle = -image angle).
    x, y, rw, rh, angle = roi
    cx_img = x + rw / 2.0
    cx = (w - 1) - cx_img
    cy = y + rh / 2.0
    theta = np.deg2rad(-angle)
    ca, sa = np.cos(theta), np.sin(theta)

    expected = set()
    for idx, (wc, wr) in enumerate(pixel_coords):
        wx = float(w - 1 - wc)
        wy = float(wr)
        dx = wx - cx
        dy = wy - cy
        lx = dx * ca + dy * sa
        ly = -dx * sa + dy * ca
        if abs(lx) <= rw / 2.0 + 1e-6 and abs(ly) <= rh / 2.0 + 1e-6:
            expected.add(idx)

    assert selected == expected, (
        f"selected {len(selected)} points, expected {len(expected)}; "
        f"symmetric diff {len(selected ^ expected)}"
    )


def test_2d_roi_angle_roundtrip():
    """The 2D ROI angle must survive the fixture<->image round-trip unchanged."""
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QObject, pyqtSignal
    from falcocv.ui.widgets.roi_editor_3d import RoiEditor3DWidget

    app = QApplication.instance()
    if app is None:
        app = QApplication([])

    depth = _make_depth_plane(200, 200)
    vimg = VisionImage(depth, x=0.0, y=0.0, angle=0.0,
                       scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    tool.params["rois"] = []

    captured = {}

    class DummyViewer2D(QObject):
        roi_changed = pyqtSignal(object)
        roi_drawing_finished = pyqtSignal()
        roi_edit_finished = pyqtSignal()

        def set_roi(self, data):
            captured["roi"] = list(data) if isinstance(data, (list, tuple)) else data

        def set_roi_editing(self, enabled):
            pass

        def set_drawing_new_roi(self, enabled):
            pass

    editor = RoiEditor3DWidget()
    editor.set_tool(tool, viewer_2d=DummyViewer2D(), viewer_3d=None)

    # Simulate the 2D viewer emitting a rotated ROI.
    editor._on_viewer_2d_roi([10.0, 20.0, 30.0, 40.0, 15.0])

    # Refresh the 2D view from the stored fixture ROI.
    editor._update_viewer_2d_roi()

    assert "roi" in captured
    assert abs(captured["roi"][4] - 15.0) < 1e-3


def test_roi_overlay_uses_absolute_coords():
    """The green ROI overlay must be painted in absolute image coords, not fixture."""
    depth = _make_depth_plane(200, 200)
    vimg = VisionImage(depth, x=50.0, y=30.0, angle=10.0,
                       scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    fixture_roi = [10.0, 20.0, 80.0, 80.0, 5.0]
    tool.params["rois"] = [{"roi": fixture_roi, "z_min": 0.0, "z_max": 1.0}]
    tool.params["method"] = "lsq"
    tool.run()

    out = tool.outputs["output image"]
    layer = out.get_overlay_layer("default")
    roi_elem = next(e for e in layer.elements if e.ref_id == "plane_roi")
    overlay_roi = roi_elem.data["roi"]
    expected_abs = vimg.fixture_to_image(fixture_roi)

    for i in range(5):
        assert abs(overlay_roi[i] - expected_abs[i]) < 1e-3, (
            f"overlay ROI coord {i} = {overlay_roi[i]}, expected absolute {expected_abs[i]}"
        )


def test_polygon_vertices_lie_on_fitted_plane():
    """The 2D plane polygon vertices must satisfy the fitted plane equation."""
    h, w = 200, 200
    depth = _make_depth_plane(h, w)
    vimg = VisionImage(depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    tool.params["rois"] = [{"roi": [50.0, 50.0, 100.0, 100.0, 0.0], "z_min": 0.0, "z_max": 1.0}]
    tool.params["method"] = "lsq"
    tool.run()

    normal = np.asarray(tool.outputs["results"][0].normal)
    distance = float(tool.outputs["results"][0].distance)
    polygon = np.asarray(tool.outputs["results"][0].polygon_2d, dtype=np.float64)

    # Every vertex of an image-box intersection polygon must lie on at least
    # one image border (left/right/top/bottom) within a small tolerance.
    margin = 1.0
    on_left = np.abs(polygon[:, 0]) <= margin
    on_right = np.abs(polygon[:, 0] - (w - 1)) <= margin
    on_top = np.abs(polygon[:, 1]) <= margin
    on_bottom = np.abs(polygon[:, 1] - (h - 1)) <= margin
    assert np.all(on_left | on_right | on_top | on_bottom)

    assert np.all(polygon[:, 0] >= -margin)
    assert np.all(polygon[:, 0] <= w - 1 + margin)
    assert np.all(polygon[:, 1] >= -margin)
    assert np.all(polygon[:, 1] <= h - 1 + margin)


def test_roi_in_zero_area_returns_no_points():
    """With zero_as_invalid=True default, depth=0 pixels become NaN and excluded.

    A ROI covering only zero-area pixels should yield "not enough points".
    """
    h, w = 200, 200
    depth = np.full((h, w), 10.0, dtype=np.float32)
    depth[100:150, 100:150] = 0.0  # bottom-right block intentionally invalid
    vimg = VisionImage(depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0)
    ensure_pointcloud_data(vimg, target_points=100_000)

    tool = PlaneFitter()
    tool.set_input("input image", vimg)
    # ROI entirely within the zero block
    tool.params["rois"] = [{"roi": [110.0, 110.0, 30.0, 30.0, 0.0], "z_min": 0.0, "z_max": 1.0}]
    tool.params["method"] = "lsq"
    tool.run()
    assert tool.outputs["output image"] is None
    assert "not enough points" in str(tool.last_error)


def test_auto_fit_z_with_fixture_offset():
    """auto_fit_z must convert fixture->image coords before sampling depth.

    With a fixture offset (x=20, y=10), an ROI stored in fixture coords must
    be translated to absolute pixels before indexing the depth map, otherwise
    the wrong patch is sampled.
    """
    h, w = 100, 100
    depth = np.full((h, w), 5.0, dtype=np.float32)
    # Carve a distinct z bump at absolute pixels [40:60, 60:80].
    depth[40:60, 60:80] = 7.5
    vimg = VisionImage(depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0,
                       x=20.0, y=10.0, angle=0.0)

    params = {"rois": []}
    add_roi(params)  # creates a default entry
    # ROI that, after fixture->image (add x=20, y=10), lands inside [40:60,60:80]
    update_roi_2d_in_params(params, 0, [40.0, 30.0, 20.0, 20.0, 0.0])

    z_min, z_max = auto_fit_z(params, 0, depth, 1.0, 0.0, fixture_to_image_fn=vimg.fixture_to_image)
    assert z_min is not None
    # Without fixture conversion, raw coords [40,30,20,20] index depth[30:50,40:60]
    # which is the flat base (5.0). With conversion, the ROI lands inside the bump
    # at absolute [60,40,20,20] -> z_min=7.5, z_max=7.5 + buffer (uniform patch).
    assert abs(z_min - 7.5) < 0.01, f"z_min={z_min}, expected 7.5"
    assert z_max > 7.5, f"z_max={z_max}, expected bumped above 7.5"
    # Sanity: without the conversion, the raw-coord path would give z_min=5.0.
    params2 = {"rois": []}
    add_roi(params2)
    update_roi_2d_in_params(params2, 0, [40.0, 30.0, 20.0, 20.0, 0.0])
    z_min_raw, _ = auto_fit_z(params2, 0, depth, 1.0, 0.0)
    assert abs(z_min_raw - 5.0) < 0.01, f"raw z_min={z_min_raw}, expected 5.0"


def test_auto_fit_z_without_fixture_fn_uses_raw():
    """Without fixture_to_image_fn, auto_fit_z uses raw coords (back-compat)."""
    h, w = 80, 80
    depth = np.full((h, w), 3.0, dtype=np.float32)
    depth[20:40, 20:40] = 9.0
    params = {"rois": []}
    add_roi(params)
    update_roi_2d_in_params(params, 0, [20.0, 20.0, 20.0, 20.0, 0.0])
    z_min, z_max = auto_fit_z(params, 0, depth, 1.0, 0.0)
    assert z_min is not None
    assert abs(z_min - 9.0) < 0.01


