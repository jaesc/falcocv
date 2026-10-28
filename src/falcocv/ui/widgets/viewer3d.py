# Viewer3D — Interactive 3D point cloud viewer using PyOpenGL + QOpenGLWidget.

import numpy as np
import math
from PyQt6.QtOpenGLWidgets import QOpenGLWidget
from PyQt6.QtCore import Qt, QPoint, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import QMouseEvent, QWheelEvent, QSurfaceFormat, QAction
from PyQt6.QtWidgets import QMenu

try:
    from OpenGL import GL
except ImportError:
    GL = None

from falcocv.ui.style import HANDLE_FILL, HANDLE_STROKE


# ── GLSL shaders (OpenGL 3.3 core) ──────────────────────────────────

_POINT_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
uniform mat4 u_mvp;
uniform float u_point_size;
out vec3 v_color;
void main() {
    gl_Position = u_mvp * vec4(position, 1.0);
    gl_PointSize = u_point_size;
    v_color = color;
}
"""

_POINT_FRAGMENT_SHADER = """
#version 330 core
in vec3 v_color;
out vec4 frag_color;
void main() {
    frag_color = vec4(v_color, 1.0);
}
"""

_LINE_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
uniform mat4 u_mvp;
out vec4 v_color;
void main() {
    gl_Position = u_mvp * vec4(position, 1.0);
    v_color = color;
}
"""

_LINE_FRAGMENT_SHADER = """
#version 330 core
in vec4 v_color;
out vec4 frag_color;
void main() {
    frag_color = v_color;
}
"""


# ── Camera ──────────────────────────────────────────────────────────

class ArcballCamera:
    """Simple arcball camera — orbit around a target point."""

    def __init__(self):
        # Isometric view that shows the 3D shape clearly while keeping
        # the world X axis consistent with the flipped point-cloud convention.
        self.azimuth = 45.0          # degrees
        self.elevation = 30.0        # degrees
        self.distance = 200.0
        self.target = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.fov = 45.0              # degrees
        self.z_near = 1.0
        self.z_far = 100000.0
        self.aspect = 1.0

    def view_matrix(self):
        az = math.radians(self.azimuth)
        el = math.radians(self.elevation)
        cx = self.distance * math.cos(el) * math.cos(az)
        cy = self.distance * math.cos(el) * math.sin(az)
        cz = self.distance * math.sin(el)
        eye = self.target + np.array([cx, cy, cz])
        return self._look_at(eye, self.target, np.array([0.0, 0.0, 1.0]))

    def projection_matrix(self):
        return self._perspective(math.radians(self.fov), self.aspect, self.z_near, self.z_far)

    def mvp_matrix(self):
        return self.projection_matrix() @ self.view_matrix()

    def _look_at(self, eye, center, up):
        f = center - eye
        fnorm = np.linalg.norm(f)
        if fnorm < 1e-12:
            f = np.array([0.0, 0.0, -1.0])
        else:
            f = f / fnorm
        s = np.cross(f, up)
        snorm = np.linalg.norm(s)
        if snorm < 1e-12:
            s = np.array([1.0, 0.0, 0.0])
        else:
            s = s / snorm
        u = np.cross(s, f)
        u = u / np.linalg.norm(u)

        m = np.eye(4, dtype=np.float32)
        m[0, :3] = s
        m[1, :3] = u
        m[2, :3] = -f
        m[:3, 3] = -m[:3, :3] @ eye
        return m

    def _perspective(self, fov_y, aspect, zn, zf):
        f = 1.0 / math.tan(fov_y * 0.5)
        m = np.zeros((4, 4), dtype=np.float32)
        m[0, 0] = f / aspect
        m[1, 1] = f
        m[2, 2] = (zf + zn) / (zn - zf)
        m[2, 3] = 2.0 * zf * zn / (zn - zf)
        m[3, 2] = -1.0
        return m


# ── Shader helper ───────────────────────────────────────────────────

def _compile_shader(src, shader_type):
    shader = GL.glCreateShader(shader_type)
    GL.glShaderSource(shader, src)
    GL.glCompileShader(shader)
    if not GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS):
        log = GL.glGetShaderInfoLog(shader)
        GL.glDeleteShader(shader)
        raise RuntimeError(f"Shader compile error: {log}")
    return shader


def _link_program(vertex_src, fragment_src):
    vs = _compile_shader(vertex_src, GL.GL_VERTEX_SHADER)
    fs = _compile_shader(fragment_src, GL.GL_FRAGMENT_SHADER)
    prog = GL.glCreateProgram()
    GL.glAttachShader(prog, vs)
    GL.glAttachShader(prog, fs)
    GL.glLinkProgram(prog)
    if not GL.glGetProgramiv(prog, GL.GL_LINK_STATUS):
        log = GL.glGetProgramInfoLog(prog)
        GL.glDeleteProgram(prog)
        raise RuntimeError(f"Program link error: {log}")
    GL.glDeleteShader(vs)
    GL.glDeleteShader(fs)
    return prog


# ── Viewer3D ────────────────────────────────────────────────────────

class Viewer3D(QOpenGLWidget):
    """Interactive 3D viewer for point clouds and meshes."""

    # Signals
    mouse_moved = pyqtSignal(float, float, float)     # (x, y, z) or NaN if no hit
    mouse_clicked = pyqtSignal(float, float, float)    # (x, y, z) or NaN
    roi_changed = pyqtSignal(object)                   # dict with center, size, rotation
    roi_drawing_finished = pyqtSignal()                # emitted when a 3D ROI draw completes
    point_picked = pyqtSignal(int, float, float, float)  # (index, x, y, z)
    gl_state_changed = pyqtSignal(bool, str)           # (initialized, error)

    def __init__(self, parent=None):
        super().__init__(parent)
        try:
            fmt = QSurfaceFormat()
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CompatibilityProfile)
            self.setFormat(fmt)
        except Exception:
            pass
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._show_context_menu)

        self.camera = ArcballCamera()
        self._point_program = None
        self._line_program = None
        self._point_vao = 0
        self._point_vbo_pos = 0
        self._point_vbo_col = 0
        self._point_count = 0

        self._grid_vao = 0
        self._grid_vbo = 0
        self._grid_count = 0

        self._roi_vao = 0
        self._roi_vbo = 0
        self._roi_count = 0

        # 3D overlays (generic: WORLD_POLYGON, WORLD_LINE, etc.)
        self._overlay_vao = 0
        self._overlay_vbo = 0
        self._overlay_draws = []  # list of (mode, first, count)
        self._overlay_visible = False

        # Point cloud data (full resolution)
        self._points = np.zeros((0, 3), dtype=np.float32)
        self._colors = np.zeros((0, 3), dtype=np.uint8)
        self._point_size = 3.0
        self._bbox_center = np.zeros(3, dtype=np.float32)
        self._bbox_size = np.ones(3, dtype=np.float32)

        # LOD data uploaded to GPU — keeps the viewer responsive with huge clouds.
        self._RENDER_BUDGET = 3_000_000
        self._lod_points = np.zeros((0, 3), dtype=np.float32)
        self._lod_colors = np.zeros((0, 3), dtype=np.uint8)
        self._lod_count = 0
        self._lod_rng = np.random.default_rng(0)
        self._lod_focus_timer = QTimer(self)
        self._lod_focus_timer.setSingleShot(True)
        self._lod_focus_timer.timeout.connect(self._update_lod)

        # ROI box
        self._roi_center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self._roi_size = np.array([50.0, 50.0, 50.0], dtype=np.float32)
        self._roi_visible = False
        self._roi_edit_mode = False
        self._roi_dragging = False
        self._roi_drag_axis = -1  # 0=x, 1=y, 2=z (face handles), 6=center translate
        self._roi_hover_axis = -1
        self._roi_drag_start_mouse = None
        self._roi_drag_start_center = None
        self._roi_drag_start_size = None
        self._roi_drag_start_hit = None  # 3D point hit on press (translation only)

        # ROI drawing from scratch
        self._roi_draw_mode = False
        self._roi_drawing_new = False
        self._roi_draw_start = None
        self._roi_draw_current = None
        self._roi_draw_start_screen = None
        self._roi_draw_current_screen = None

        # Camera interaction
        self._mouse_pressed = False
        self._mouse_button = Qt.MouseButton.NoButton
        self._last_mouse_pos = QPoint()
        self._drag_start_mouse = QPoint()
        self._drag_start_azimuth = 0.0
        self._drag_start_elevation = 0.0
        self._drag_start_target = None

        # Hover
        self._pending_upload = False
        self._hovered_point = None

        # OpenGL init state
        self._gl_initialized = False
        self._gl_error = None

    # ── OpenGL lifecycle ───────────────────────────────────────────

    def initializeGL(self):
        if GL is None:
            self._gl_error = "PyOpenGL not available"
            return
        try:
            self.makeCurrent()

            # Force PyOpenGL to resolve functions against the current context.
            version_bytes = GL.glGetString(GL.GL_VERSION)
            version_str = version_bytes.decode() if version_bytes else ""
            version_parts = version_str.split()[0].split(".") if version_str else ["0", "0"]
            major = int(version_parts[0])
            minor = int(version_parts[1]) if len(version_parts) > 1 else 0
            if major < 3:
                raise RuntimeError(f"OpenGL {major}.{minor} found; 3.0+ required")

            GL.glClearColor(0.12, 0.12, 0.14, 1.0)
            GL.glEnable(GL.GL_DEPTH_TEST)
            GL.glEnable(GL.GL_PROGRAM_POINT_SIZE)

            # Clear any stale error flag before creating VAOs.
            while GL.glGetError() != GL.GL_NO_ERROR:
                pass

            self._point_program = _link_program(_POINT_VERTEX_SHADER, _POINT_FRAGMENT_SHADER)
            self._line_program = _link_program(_LINE_VERTEX_SHADER, _LINE_FRAGMENT_SHADER)

            self._point_vao = int(GL.glGenVertexArrays(1))
            self._point_vbo_pos = int(GL.glGenBuffers(1))
            self._point_vbo_col = int(GL.glGenBuffers(1))

            self._grid_vao = int(GL.glGenVertexArrays(1))
            self._grid_vbo = int(GL.glGenBuffers(1))
            self._build_grid()

            self._roi_vao = int(GL.glGenVertexArrays(1))
            self._roi_vbo = int(GL.glGenBuffers(1))

            self._overlay_vao = int(GL.glGenVertexArrays(1))
            self._overlay_vbo = int(GL.glGenBuffers(1))

            if self._roi_visible:
                self._build_roi_buffers()
            if self._pending_upload:
                self._upload_point_data()

            # Smoke-test a function that is only used later in paintGL.
            GL.glUseProgram(0)

            self._gl_initialized = True
            self.gl_state_changed.emit(True, "")
        except Exception as e:
            self._gl_initialized = False
            self._gl_error = str(e)
            self.gl_state_changed.emit(False, str(e))

    def resizeGL(self, w, h):
        if GL is None or not self._gl_initialized:
            return
        self.camera.aspect = w / max(h, 1)
        if self.camera.aspect <= 0.0:
            self.camera.aspect = 1.0

    def paintGL(self):
        if GL is None or not self._gl_initialized:
            if GL is not None:
                GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            return
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)

        mvp = self.camera.mvp_matrix()

        # Draw grid
        if self._grid_count > 0:
            GL.glUseProgram(self._line_program)
            loc_mvp = GL.glGetUniformLocation(self._line_program, "u_mvp")
            GL.glUniformMatrix4fv(loc_mvp, 1, GL.GL_TRUE, mvp)
            GL.glBindVertexArray(self._grid_vao)
            GL.glDrawArrays(GL.GL_LINES, 0, self._grid_count)
            GL.glBindVertexArray(0)

        # Upload pending data if needed
        if self._pending_upload:
            self._upload_point_data()
        # Draw point cloud (LOD)
        if self._lod_count > 0:
            GL.glUseProgram(self._point_program)
            loc_mvp = GL.glGetUniformLocation(self._point_program, "u_mvp")
            loc_ps = GL.glGetUniformLocation(self._point_program, "u_point_size")
            GL.glUniformMatrix4fv(loc_mvp, 1, GL.GL_TRUE, mvp)
            GL.glUniform1f(loc_ps, self._point_size)
            GL.glBindVertexArray(self._point_vao)
            GL.glDrawArrays(GL.GL_POINTS, 0, self._lod_count)
            GL.glBindVertexArray(0)

        # Draw ROI box
        if self._roi_visible:
            self._draw_roi(mvp)

        # Draw 3D overlays
        if self._overlay_visible:
            self._draw_overlays(mvp)

        GL.glUseProgram(0)

    # ── Data upload ────────────────────────────────────────────────

    def _upload_point_data(self):
        if GL is None or not self._gl_initialized:
            return
        pos = self._lod_points
        col = self._lod_colors

        GL.glBindVertexArray(self._point_vao)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._point_vbo_pos)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, pos.nbytes, pos, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 0, GL.GLvoidp(0))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._point_vbo_col)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, col.nbytes, col, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 3, GL.GL_UNSIGNED_BYTE, GL.GL_TRUE, 0, GL.GLvoidp(0))

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
        GL.glBindVertexArray(0)
        self._pending_upload = False

    def _compute_lod_stride(self):
        """Return a stride that keeps the rendered point count near the budget.

        The stride is reduced automatically when the camera zooms in, so regions
        seen close-up get more detail. The result is always >= 1.
        """
        n = len(self._points)
        if n <= self._RENDER_BUDGET:
            return 1

        # Stride needed to hit the render budget across the whole cloud.
        base_stride = max(1, int(np.ceil(n / self._RENDER_BUDGET)))

        # Hard cap: never render more than ~6M points, even when zoomed in.
        # This protects the GPU from choking on extreme close-ups.
        MAX_RENDER_POINTS = 6_000_000
        min_stride = max(1, int(np.ceil(n / MAX_RENDER_POINTS)))

        # Adapt to camera distance: closer -> smaller stride (more detail).
        cloud_radius = float(np.max(self._bbox_size)) * 0.5
        if cloud_radius < 1e-6:
            return max(base_stride, min_stride)

        # distance ~ 1.5 * radius frames the whole object.
        ratio = self.camera.distance / cloud_radius
        if not np.isfinite(ratio):
            return max(base_stride, min_stride)
        # Clamp to reasonable range and invert: small ratio (zoomed in) -> small scale.
        scale = max(0.05, min(1.5, ratio)) / 1.5
        if not np.isfinite(scale):
            return max(base_stride, min_stride)
        stride = max(min_stride, int(np.round(base_stride * scale)))
        return stride

    def _compute_frustum_aabb(self):
        """Return the world-space axis-aligned box that contains the view frustum."""
        mvp = self.camera.projection_matrix() @ self.camera.view_matrix()
        try:
            inv_mvp = np.linalg.inv(mvp)
        except np.linalg.LinAlgError:
            return None, None

        # NDC cube corners.
        ndc = np.array([
            [-1, -1, -1], [-1, -1,  1], [-1,  1, -1], [-1,  1,  1],
            [ 1, -1, -1], [ 1, -1,  1], [ 1,  1, -1], [ 1,  1,  1],
        ], dtype=np.float32)
        corners = np.concatenate([ndc, np.ones((8, 1), dtype=np.float32)], axis=1)
        world = (inv_mvp @ corners.T).T
        world = world[:, :3] / world[:, 3:4]
        return world.min(axis=0), world.max(axis=0)

    def _get_visible_indices(self, sample_only=False):
        """Return indices of points inside the current view frustum AABB.

        For huge clouds a sampled scan is performed first. If the estimated
        visible count is small enough, a full scan is done so close-ups keep
        full detail.
        """
        lo, hi = self._compute_frustum_aabb()
        if lo is None:
            return np.array([], dtype=np.int64)
        # Add a small margin to avoid clipping points at the edges.
        margin = np.max(self._bbox_size) * 0.01
        lo -= margin
        hi += margin

        n = len(self._points)
        MAX_RENDER_POINTS = 6_000_000

        def _in_box(pts):
            return np.all((pts >= lo) & (pts <= hi), axis=1)

        # Fast sampled scan.
        sample_step = max(1, n // 2_000_000)
        sample_idx = np.arange(0, n, sample_step)
        sample_mask = _in_box(self._points[sample_idx])
        sample_visible = sample_idx[sample_mask]

        if sample_only:
            return sample_visible

        if len(sample_visible) == 0:
            return sample_visible

        # Estimate full visible count from the sample.
        estimated_visible = int(len(sample_visible) * (n / len(sample_idx)))

        # If the view is small enough, pay for a full scan to get every point.
        if estimated_visible <= MAX_RENDER_POINTS * 2:
            full_mask = _in_box(self._points)
            return np.where(full_mask)[0]

        return sample_visible

    def _build_lod(self):
        """Build the LOD array uploaded to the GPU.

        Uses a uniform stride for distant views and a frustum-culled subset
        when the camera is zoomed in, so close-up detail is preserved without
        sending the whole cloud to the GPU.
        """
        n = len(self._points)
        if n == 0:
            self._lod_points = self._points
            self._lod_colors = self._colors
            self._lod_count = 0
            return

        if n <= self._RENDER_BUDGET:
            self._lod_points = self._points
            self._lod_colors = self._colors
            self._lod_count = n
            return

        MAX_RENDER_POINTS = 6_000_000
        cloud_diameter = float(np.max(self._bbox_size))

        # If zoomed in, try to render only the visible points.
        if self.camera.distance < cloud_diameter * 1.0:
            visible_idx = self._get_visible_indices()
            if len(visible_idx) > 0:
                if len(visible_idx) > MAX_RENDER_POINTS:
                    stride = max(1, int(np.ceil(len(visible_idx) / MAX_RENDER_POINTS)))
                    visible_idx = visible_idx[::stride]
                self._lod_points = np.ascontiguousarray(self._points[visible_idx])
                self._lod_colors = np.ascontiguousarray(self._colors[visible_idx])
                self._lod_count = len(self._lod_points)
                return

        # Distant view: uniform stride over the whole cloud.
        stride = self._compute_lod_stride()
        self._lod_points = np.ascontiguousarray(self._points[::stride])
        self._lod_colors = np.ascontiguousarray(self._colors[::stride])
        self._lod_count = len(self._lod_points)

    def _update_lod(self):
        """Rebuild LOD after camera changes and request GPU upload."""
        n = len(self._points)
        if n == 0 or n <= self._RENDER_BUDGET:
            return
        self._build_lod()
        self._pending_upload = True
        if self._gl_initialized:
            self.update()

    def set_data(self, points, colors=None, bbox_center=None, bbox_size=None):
        if GL is None:
            return
        self._points = np.ascontiguousarray(points, dtype=np.float32)
        n = len(self._points)
        if colors is None or len(colors) != n:
            colors = np.full((n, 3), 200, dtype=np.uint8)
        self._colors = np.ascontiguousarray(colors, dtype=np.uint8)
        self._point_count = n
        if n > 0:
            if bbox_center is not None and bbox_size is not None:
                self._bbox_center = np.asarray(bbox_center, dtype=np.float32).copy()
                self._bbox_size = np.asarray(bbox_size, dtype=np.float32).copy()
            else:
                self._bbox_center, self._bbox_size = _compute_bbox(self._points)
            self.camera.target = self._bbox_center.copy()
            self.camera.distance = max(self._bbox_size) * 1.5
            self.camera.distance = max(self.camera.distance, 1.0)
        self._build_lod()
        self._pending_upload = True
        if self._gl_initialized:
            self.update()

    def set_roi(self, data, size=None):
        """Set the 3D ROI box position and size.

        Accepts either:
          - set_roi(None)                 -> hide ROI
          - set_roi(center, size)         -> explicit arrays
          - set_roi({"center": c, "size": s}) or set_roi([c, s])
        """
        if data is None:
            self._roi_visible = False
            if self._gl_initialized:
                self.update()
            return

        if isinstance(data, dict):
            center = np.asarray(data.get("center", [0.0, 0.0, 0.0]), dtype=np.float32)
            size_arr = np.asarray(data.get("size", [1.0, 1.0, 1.0]), dtype=np.float32)
        elif isinstance(data, (list, tuple)) and size is None and len(data) == 2:
            center = np.asarray(data[0], dtype=np.float32)
            size_arr = np.asarray(data[1], dtype=np.float32)
        else:
            center = np.asarray(data, dtype=np.float32)
            size_arr = np.asarray(size, dtype=np.float32)

        self._roi_center = center
        self._roi_size = size_arr
        self._roi_size[self._roi_size < 1.0] = 1.0
        self._roi_visible = True
        if self._gl_initialized:
            self._build_roi_buffers()
            self.update()

    def show_roi(self, visible=True):
        self._roi_visible = visible
        if self._gl_initialized:
            self.update()

    def set_point_size(self, size):
        self._point_size = max(1.0, float(size))
        self.update()

    def set_roi_edit_mode(self, enabled):
        """Enable/disable ROI manipulation via mouse handles."""
        self._roi_edit_mode = bool(enabled)
        if not self._roi_edit_mode:
            self._roi_hover_axis = -1
            if self._roi_dragging:
                self._roi_dragging = False
                self._roi_drag_axis = -1
                if self.mouseGrabber() is self:
                    self.releaseMouse()
                if self._roi_drag_start_center is not None and self._roi_drag_start_size is not None:
                    self.set_roi(self._roi_drag_start_center, self._roi_drag_start_size)
        self.update()
        self.update()

    def set_roi_draw_mode(self, enabled):
        """Enable/disable drawing a new ROI box directly in the 3D view."""
        self._roi_draw_mode = bool(enabled)
        self._roi_drawing_new = False
        if not enabled:
            self._roi_draw_start_screen = None
            self._roi_draw_current_screen = None
        if not enabled and self._roi_visible:
            # Cancel any in-progress draw
            self.update()

    def roi_draw_mode_active(self):
        """Return True if the user is currently in ROI draw mode."""
        return self._roi_draw_mode

    def _default_roi_height(self):
        return float(max(np.max(self._bbox_size) * 0.1, 1.0))

    def _drawing_plane_z(self):
        """Return the Z coordinate of the plane used for drawing a 3D ROI.

        We draw on the plane that passes through the current ROI (if any) or
        through the centre of the point cloud, so that the drawn rectangle
        aligns with the visible data instead of being projected onto Z=0.
        """
        if self._roi_visible and np.isfinite(self._roi_center[2]):
            return float(self._roi_center[2])
        if np.isfinite(self._bbox_center[2]):
            return float(self._bbox_center[2])
        return float(self.camera.target[2])

    def get_drawing_plane_z(self):
        """Public accessor for the current drawing-plane Z."""
        return self._drawing_plane_z()

    def get_draw_screen_rect(self):
        """Return the screen rectangle of the last 3D ROI draw, or None.

        Returns a tuple (x0, y0, x1, y1) in widget coordinates, or None if
        no draw gesture is in progress / has just finished.
        """
        if self._roi_draw_start_screen is None or self._roi_draw_current_screen is None:
            return None
        x0 = min(self._roi_draw_start_screen.x(), self._roi_draw_current_screen.x())
        y0 = min(self._roi_draw_start_screen.y(), self._roi_draw_current_screen.y())
        x1 = max(self._roi_draw_start_screen.x(), self._roi_draw_current_screen.x())
        y1 = max(self._roi_draw_start_screen.y(), self._roi_draw_current_screen.y())
        return (x0, y0, x1, y1)

    def get_draw_start_screen(self):
        """Return the screen position where the current draw gesture started."""
        return self._roi_draw_start_screen

    def get_draw_current_screen(self):
        """Return the current/last screen position of the draw gesture."""
        return self._roi_draw_current_screen

    def screen_pos_to_nearest_world_point(self, pos):
        """Return the world coordinates of the nearest displayed point under pos.

        The ray is cast from the camera through the screen pixel. The nearest
        point among the displayed cloud that is within a distance tolerance of
        the ray is returned, or None if no point is close enough.
        """
        if len(self._points) == 0:
            return None
        ray_origin, ray_dir = self.screen_to_ray(pos)
        to_points = self._points - ray_origin
        t = np.dot(to_points, ray_dir)
        near_mask = t > 0
        if not near_mask.any():
            return None
        closest = self._points[near_mask]
        t_near = t[near_mask]
        proj = ray_origin.reshape(1, 3) + t_near.reshape(-1, 1) * ray_dir.reshape(1, 3)
        dist = np.linalg.norm(closest - proj, axis=1)
        max_dist = max(self.camera.distance * 0.01, 1.0)
        close_enough = dist < max_dist
        if not close_enough.any():
            return None
        best = np.argmin(t_near[close_enough])
        return closest[close_enough][best].copy()

    def screen_rect_to_world_points(self, rect, plane_z):
        """Back-project the four corners of a screen rect onto a Z plane.

        Returns a (4, 3) float32 array of world points, or None if any corner
        cannot be projected.
        """
        corners_screen = [
            (rect[0], rect[1]),
            (rect[2], rect[1]),
            (rect[2], rect[3]),
            (rect[0], rect[3]),
        ]
        pts = []
        for sx, sy in corners_screen:
            # Use QPoint-compatible interface
            ray_origin, ray_dir = self.screen_to_ray(QPoint(int(round(sx)), int(round(sy))))
            if abs(ray_dir[2]) < 1e-8:
                return None
            t = (plane_z - ray_origin[2]) / ray_dir[2]
            if t < 0:
                return None
            pts.append(ray_origin + ray_dir * t)
        return np.array(pts, dtype=np.float32)

    def _screen_to_plane(self, pos, plane_z):
        """Intersect a screen ray with the horizontal plane Z=plane_z."""
        ray_origin, ray_dir = self.screen_to_ray(pos)
        if abs(ray_dir[2]) < 1e-8:
            return None
        t = (plane_z - ray_origin[2]) / ray_dir[2]
        if t < 0:
            return None
        return ray_origin + ray_dir * t

    # ── Grid ────────────────────────────────────────────────────────

    def _build_grid(self):
        """Build a grid on the XY plane (Z=0)."""
        if GL is None or not self._gl_initialized:
            return
        n_lines = 20
        extent = 100.0
        verts = []
        step = 2.0 * extent / n_lines
        for i in range(n_lines + 1):
            v = -extent + i * step
            verts.extend([v, -extent, 0.0, 0.5, 0.5, 0.5,
                          v, extent, 0.0, 0.5, 0.5, 0.5])
            verts.extend([-extent, v, 0.0, 0.5, 0.5, 0.5,
                          extent, v, 0.0, 0.5, 0.5, 0.5])

        # Axes: X=red, Y=green, Z=blue
        axis_len = extent * 0.4
        verts.extend([0, 0, 0, 1, 0, 0, axis_len, 0, 0, 1, 0, 0,
                      0, 0, 0, 0, 1, 0, 0, axis_len, 0, 0, 1, 0,
                      0, 0, 0, 0, 0, 1, 0, 0, axis_len, 0, 0, 1])

        arr = np.array(verts, dtype=np.float32)
        self._grid_count = len(verts) // 6

        try:
            GL.glBindVertexArray(self._grid_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._grid_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, GL.GLvoidp(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, GL.GLvoidp(12))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)
        except Exception:
            pass

    # ── ROI box ─────────────────────────────────────────────────────

    def _build_roi_buffers(self):
        """Build ROI box wireframe + face handles + center handle."""
        if GL is None or not self._gl_initialized:
            return

        h = self._roi_size * 0.5
        c = self._roi_center

        corners = np.array([
            [-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1],
            [-1, 1, -1], [1, 1, -1], [1, 1, 1], [-1, 1, 1],
        ], dtype=np.float32)
        corners = corners * h + c

        # 12 edges
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]

        verts = []
        roi_color = (1.0, 0.5, 0.0)  # orange
        for i, j in edges:
            verts.extend([*corners[i], *roi_color, *corners[j], *roi_color])

        # Face handle quads (center of each face, large enough to grab)
        handle_color = (0.0, 0.8, 1.0)
        hs = float(0.15 * np.max(self._roi_size))  # handle size scalar
        face_centers = [
            (c + [h[0], 0, 0], [1, 0, 0]),   # +X  (axis idx 0)
            (c - [h[0], 0, 0], [-1, 0, 0]),  # -X  (axis idx 3)
            (c + [0, h[1], 0], [0, 1, 0]),   # +Y  (axis idx 1)
            (c - [0, h[1], 0], [0, -1, 0]),  # -Y  (axis idx 4)
            (c + [0, 0, h[2]], [0, 0, 1]),   # +Z  (axis idx 2)
            (c - [0, 0, h[2]], [0, 0, -1]),  # -Z  (axis idx 5)
        ]

        for fc, axis in face_centers:
            # Build a small quad centered at fc, perpendicular to axis
            if abs(axis[0]) > 0:
                pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=np.float32)
            elif abs(axis[1]) > 0:
                pts = np.array([[-1, 0, -1], [1, 0, -1], [1, 0, 1], [-1, 0, 1]], dtype=np.float32) * 0.5
            else:
                pts = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32) * 0.5

            pts = pts * hs
            # Rotate quad to face direction
            if abs(axis[1]) > 0:
                pts_3d = np.column_stack([pts[:, 0], np.full(4, fc[1]), pts[:, 2]])
            elif abs(axis[2]) > 0:
                pts_3d = pts  # already on XY plane, just offset by fc below
            else:
                pts_3d = np.column_stack([np.full(4, fc[0]), pts[:, 0], pts[:, 1]])

            pts_3d = pts_3d + fc
            for p in pts_3d:
                verts.extend([*p, *handle_color])

        # Center translation handle (yellow square on XY plane)
        center_handle_color = (1.0, 0.9, 0.0)
        chs = float(0.12 * np.max(self._roi_size))
        center_pts = np.array([[-1, -1, 0], [1, -1, 0], [1, 1, 0], [-1, 1, 0]], dtype=np.float32) * chs + c
        for p in center_pts:
            verts.extend([*p, *center_handle_color])

        # 6 face handles + 1 center handle
        self._roi_handle_count = len(face_centers) + 1
        arr = np.array(verts, dtype=np.float32)
        self._roi_count = len(verts) // 6

        try:
            self.makeCurrent()
            GL.glBindVertexArray(self._roi_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._roi_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, GL.GLvoidp(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, GL.GLvoidp(12))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)
        except Exception:
            pass

    def _draw_roi(self, mvp):
        if self._roi_count <= 0:
            return

        GL.glUseProgram(self._line_program)
        loc_mvp = GL.glGetUniformLocation(self._line_program, "u_mvp")
        GL.glUniformMatrix4fv(loc_mvp, 1, GL.GL_TRUE, mvp)

        # Make sure the ROI is visible even if it is inside the point cloud.
        GL.glDisable(GL.GL_DEPTH_TEST)

        GL.glBindVertexArray(self._roi_vao)

        # Draw wireframe (edges). Keep line width at 1.0; wider lines are not
        # reliably supported across OpenGL core-profile drivers and can crash
        # some Intel/older GPUs.
        edge_count = 12 * 2  # 12 edges, 2 verts each
        GL.glDrawArrays(GL.GL_LINES, 0, edge_count)

        # Draw face handles as filled quads + outline, only in edit mode.
        if self._roi_edit_mode and self._roi_handle_count > 0:
            handle_start = edge_count
            for h_idx in range(self._roi_handle_count):
                quad_start = handle_start + h_idx * 4
                is_active = (h_idx == self._roi_drag_axis or h_idx == self._roi_hover_axis)
                # Filled quad (brighter when hovered/active)
                GL.glDrawArrays(GL.GL_TRIANGLE_FAN, quad_start, 4)
                # Outline
                GL.glDrawArrays(GL.GL_LINE_LOOP, quad_start, 4)

        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glUseProgram(0)

    # ── 3D Overlays (generic) ─────────────────────────────────────

    def set_overlays_from_image(self, image):
        """Read 3D overlays from a VisionImage and build GPU buffers."""
        if image is None:
            self._overlay_draws = []
            self.update()
            return
        from falcocv.core.overlay_layer import OverlayType as OT
        overlays = image.get_overlays_3d()
        self._set_overlays(overlays, OT)

    def _set_overlays(self, overlays, OT):
        """Build GPU buffers for a list of 3D overlay elements."""
        self._overlay_draws = []
        all_verts = []

        for elem in overlays:
            d = elem.data
            t = elem.element_type
            try:
                color = tuple(c / 255.0 for c in d["color"])
            except Exception:
                color = (0.0, 0.8, 1.0)
            alpha = d.get("fill_alpha", 1.0)

            if t == OT.WORLD_POLYGON:
                verts = np.asarray(d["world_vertices"], dtype=np.float32)
                if len(verts) < 3:
                    continue
                centroid = verts.mean(axis=0)
                first = len(all_verts) // 7
                for v in [centroid] + list(verts):
                    all_verts.extend([float(v[0]), float(v[1]), float(v[2]), *color, alpha])
                self._overlay_draws.append((GL.GL_TRIANGLE_FAN, first, len(verts) + 1))
                first = len(all_verts) // 7
                for v in list(verts) + [verts[0]]:
                    all_verts.extend([float(v[0]), float(v[1]), float(v[2]), *color, alpha])
                self._overlay_draws.append((GL.GL_LINE_LOOP, first, len(verts) + 1))

            elif t == OT.WORLD_CONTOUR:
                verts = np.asarray(d["world_vertices"], dtype=np.float32)
                if len(verts) < 2:
                    continue
                first = len(all_verts) // 7
                for v in verts:
                    all_verts.extend([float(v[0]), float(v[1]), float(v[2]), *color, alpha])
                self._overlay_draws.append((GL.GL_LINE_LOOP, first, len(verts)))

            elif t == OT.WORLD_LINE:
                start = d["world_start"]
                end = d["world_end"]
                first = len(all_verts) // 7
                all_verts.extend([float(start[0]), float(start[1]), float(start[2]), *color, alpha])
                all_verts.extend([float(end[0]), float(end[1]), float(end[2]), *color, alpha])
                self._overlay_draws.append((GL.GL_LINES, first, 2))

            elif t == OT.WORLD_POINTS:
                pts = np.asarray(d["world_points"], dtype=np.float32)
                if len(pts) == 0:
                    continue
                first = len(all_verts) // 7
                for v in pts:
                    all_verts.extend([float(v[0]), float(v[1]), float(v[2]), *color, alpha])
                self._overlay_draws.append((GL.GL_POINTS, first, len(pts)))

        if not all_verts:
            self._overlay_visible = False
            if self._gl_initialized:
                self.update()
            return

        arr = np.array(all_verts, dtype=np.float32)
        self._overlay_visible = True

        if not self._gl_initialized:
            return

        try:
            self.makeCurrent()
            GL.glBindVertexArray(self._overlay_vao)
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._overlay_vbo)
            GL.glBufferData(GL.GL_ARRAY_BUFFER, arr.nbytes, arr, GL.GL_STATIC_DRAW)
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 28, GL.GLvoidp(0))
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 4, GL.GL_FLOAT, GL.GL_FALSE, 28, GL.GLvoidp(12))
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, 0)
            GL.glBindVertexArray(0)
        except Exception:
            pass

        self.update()

    def _draw_overlays(self, mvp):
        if not self._overlay_draws:
            return

        GL.glUseProgram(self._line_program)
        loc_mvp = GL.glGetUniformLocation(self._line_program, "u_mvp")
        GL.glUniformMatrix4fv(loc_mvp, 1, GL.GL_TRUE, mvp)

        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_FALSE)

        GL.glBindVertexArray(self._overlay_vao)

        for mode, first, count in self._overlay_draws:
            if count > 0:
                GL.glDrawArrays(mode, first, count)

        GL.glBindVertexArray(0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        GL.glUseProgram(0)

    # ── Mouse interaction ──────────────────────────────────────────

    def mousePressEvent(self, event: QMouseEvent):
        self._mouse_pressed = True
        self._mouse_button = event.button()
        self._drag_start_mouse = event.pos()
        self._last_mouse_pos = event.pos()
        self.setFocus()

        if event.button() == Qt.MouseButton.LeftButton:
            # ROI edit mode: check handle hit
            if self._roi_edit_mode and self._roi_visible:
                axis = self._hit_roi_handle(event.pos())
                if axis >= 0:
                    self._roi_dragging = True
                    self._roi_drag_axis = axis
                    self._roi_drag_start_mouse = event.pos()
                    self._roi_drag_start_center = self._roi_center.copy()
                    self._roi_drag_start_size = self._roi_size.copy()
                    if axis == 6:
                        # Translation: ray-cast to the ROI center XY plane (Z constant).
                        self._roi_drag_start_hit = self._screen_to_plane(
                            event.pos(), self._roi_center[2]
                        )
                    else:
                        self._roi_drag_start_hit = None
                    self.grabMouse()
                    event.accept()
                    return

            # ROI draw mode: start new ROI
            if self._roi_draw_mode:
                plane_z = self._drawing_plane_z()
                pt = self._screen_to_plane(event.pos(), plane_z)
                if pt is not None:
                    self._roi_drawing_new = True
                    self._roi_draw_start = pt.copy()
                    self._roi_draw_current = pt.copy()
                    self._roi_draw_start_screen = event.pos()
                    self._roi_draw_current_screen = event.pos()
                    self.grabMouse()
                event.accept()
                return

            # Default: orbit camera
            self._drag_start_azimuth = self.camera.azimuth
            self._drag_start_elevation = self.camera.elevation
            # In ROI edit mode we never want point-click to jump the camera target.
            if not self._roi_edit_mode:
                idx = self._hit_test_point(event.pos())
                if idx >= 0:
                    self.camera.target = self._points[idx].copy()
            self._drag_start_target = self.camera.target.copy()
        elif event.button() == Qt.MouseButton.RightButton:
            self._drag_start_target = self.camera.target.copy()

        event.accept()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._roi_dragging:
            self._handle_roi_drag(event.pos())
            event.accept()
            return

        if self._roi_drawing_new:
            plane_z = self._drawing_plane_z()
            pt = self._screen_to_plane(event.pos(), plane_z)
            if pt is not None and self._roi_draw_start is not None:
                self._roi_draw_current = pt.copy()
                self._roi_draw_current_screen = event.pos()
                c = (self._roi_draw_start + self._roi_draw_current) * 0.5
                s = np.abs(self._roi_draw_current - self._roi_draw_start)
                s[2] = self._default_roi_height()
                self.set_roi(c, s)
            event.accept()
            return

        if not self._mouse_pressed:
            if self._roi_edit_mode and self._roi_visible:
                axis = self._hit_roi_handle(event.pos())
                if axis != self._roi_hover_axis:
                    self._roi_hover_axis = axis
                    self.update()
                if axis >= 0:
                    self.setCursor(Qt.CursorShape.OpenHandCursor)
                else:
                    self.setCursor(Qt.CursorShape.CrossCursor)
            else:
                self._update_hover(event.pos())
                self.setCursor(Qt.CursorShape.CrossCursor)
            super().mouseMoveEvent(event)
            return

        dx = event.pos().x() - self._drag_start_mouse.x()
        dy = event.pos().y() - self._drag_start_mouse.y()

        if self._mouse_button == Qt.MouseButton.LeftButton:
            # Orbit
            self.camera.azimuth = self._drag_start_azimuth - dx * 0.3
            self.camera.elevation = self._drag_start_elevation + dy * 0.3
            self.camera.elevation = max(-89.0, min(89.0, self.camera.elevation))
        elif self._mouse_button == Qt.MouseButton.RightButton:
            # Pan — correct screen-space vectors
            sens = self.camera.distance * 0.002
            az_r = math.radians(self.camera.azimuth)
            el_r = math.radians(self.camera.elevation)
            s = np.array([-math.sin(az_r), math.cos(az_r), 0.0])
            u = np.array([-math.cos(az_r)*math.sin(el_r),
                          -math.sin(az_r)*math.sin(el_r),
                          math.cos(el_r)])
            self.camera.target = self._drag_start_target - s * dx * sens + u * dy * sens

        self._last_mouse_pos = event.pos()
        self._lod_focus_timer.start(150)
        self.update()
        event.accept()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if self._roi_dragging and event.button() == Qt.MouseButton.LeftButton:
            self._roi_dragging = False
            self._roi_drag_axis = -1
            if self.mouseGrabber() is self:
                self.releaseMouse()
            self.roi_changed.emit({
                "center": self._roi_center.tolist(),
                "size": self._roi_size.tolist(),
                "rotation": [0.0, 0.0, 0.0],
            })
            self.update()
            event.accept()
            self._mouse_pressed = False
            self._mouse_button = Qt.MouseButton.NoButton
            return

        if self._roi_drawing_new and event.button() == Qt.MouseButton.LeftButton:
            self._roi_drawing_new = False
            if self.mouseGrabber() is self:
                self.releaseMouse()
            self._roi_draw_current_screen = event.pos()
            plane_z = self._drawing_plane_z()
            pt = self._screen_to_plane(event.pos(), plane_z)
            roi_created = False
            if pt is not None and self._roi_draw_start is not None:
                c = (self._roi_draw_start + pt) * 0.5
                s = np.abs(pt - self._roi_draw_start)
                s[2] = self._default_roi_height()
                self.set_roi(c, s)
                self.roi_changed.emit({
                    "center": c.tolist(),
                    "size": s.tolist(),
                    "rotation": [0.0, 0.0, 0.0],
                })
                roi_created = True
            # Auto-finish draw mode: one gesture = one ROI.
            self._roi_draw_mode = False
            if roi_created:
                self.roi_drawing_finished.emit()
            event.accept()
            self._mouse_pressed = False
            self._mouse_button = Qt.MouseButton.NoButton
            return

        self._mouse_pressed = False
        self._mouse_button = Qt.MouseButton.NoButton
        event.accept()
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        """Cancel any in-progress ROI operation if the mouse leaves the widget."""
        if self._roi_dragging:
            self._roi_dragging = False
            self._roi_drag_axis = -1
            if self.mouseGrabber() is self:
                self.releaseMouse()
            # Restore original ROI
            if self._roi_drag_start_center is not None and self._roi_drag_start_size is not None:
                self.set_roi(self._roi_drag_start_center, self._roi_drag_start_size)
            self.update()
        if self._roi_drawing_new:
            self._roi_drawing_new = False
            if self.mouseGrabber() is self:
                self.releaseMouse()
            self._roi_draw_start_screen = None
            self._roi_draw_current_screen = None
            self.update()
        if self._roi_hover_axis >= 0:
            self._roi_hover_axis = -1
            self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        delta = event.angleDelta().y()
        self.camera.distance *= (1.0 - delta * 0.001)
        self.camera.distance = max(1.0, min(100000.0, self.camera.distance))
        self._lod_focus_timer.start(150)
        self.update()
        super().wheelEvent(event)

    # ── ROI handle hit testing ─────────────────────────────────────

    def _project_world_to_screen(self, world_pt):
        """Project a 3D world point to widget pixels. Returns None if behind camera."""
        w, h = self.width(), max(self.height(), 1)
        if w <= 0 or h <= 0:
            return None
        mvp = self.camera.mvp_matrix()
        pt = np.array([world_pt[0], world_pt[1], world_pt[2], 1.0], dtype=np.float32)
        clip = mvp @ pt
        if abs(clip[3]) < 1e-8 or clip[3] <= 0:
            return None
        ndc = clip[:3] / clip[3]
        x = (ndc[0] + 1.0) * 0.5 * w
        y = (1.0 - ndc[1]) * 0.5 * h
        return QPointF(float(x), float(y))

    def _world_to_screen_scale(self, world_pt):
        """Approximate pixels per world unit at world_pt."""
        w, h = self.width(), max(self.height(), 1)
        if w <= 0 or h <= 0:
            return 0.0
        mvp = self.camera.mvp_matrix()
        p0 = np.array([world_pt[0], world_pt[1], world_pt[2], 1.0], dtype=np.float32)
        p1 = np.array([world_pt[0] + 1.0, world_pt[1], world_pt[2], 1.0], dtype=np.float32)
        c0 = mvp @ p0
        c1 = mvp @ p1
        if c0[3] <= 0 or c1[3] <= 0:
            return 0.0
        ndc0 = c0[:3] / c0[3]
        ndc1 = c1[:3] / c1[3]
        x0 = (ndc0[0] + 1.0) * 0.5 * w
        x1 = (ndc1[0] + 1.0) * 0.5 * w
        # Use Y scale too and average
        p2 = np.array([world_pt[0], world_pt[1] + 1.0, world_pt[2], 1.0], dtype=np.float32)
        c2 = mvp @ p2
        if c2[3] <= 0:
            return abs(x1 - x0)
        ndc2 = c2[:3] / c2[3]
        y0 = (1.0 - ndc0[1]) * 0.5 * h
        y2 = (1.0 - ndc2[1]) * 0.5 * h
        return (abs(x1 - x0) + abs(y2 - y0)) * 0.5

    def _hit_roi_handle(self, mouse_pos):
        """Screen-space handle hit test. Much more reliable than ray-casting tiny
        world-space quads, especially for small or distant ROIs.

        Returns:
            0-2   : +X, +Y, +Z face handles
            3-5   : -X, -Y, -Z face handles
            6     : center translation handle
            -1    : no hit
        """
        if not self._roi_visible:
            return -1

        h = self._roi_size * 0.5
        c = self._roi_center
        handle_centers = [
            (c + [h[0], 0, 0], 0),    # +X
            (c + [0, h[1], 0], 1),    # +Y
            (c + [0, 0, h[2]], 2),    # +Z
            (c - [h[0], 0, 0], 3),    # -X
            (c - [0, h[1], 0], 4),    # -Y
            (c - [0, 0, h[2]], 5),    # -Z
            (c, 6),                   # center XY translate
        ]

        scale = self._world_to_screen_scale(c)
        handle_world_size = 0.15 * float(np.max(self._roi_size))
        # Handle radius in pixels: at least 10 px, larger if ROI is big on screen.
        hit_radius_px = max(10.0, handle_world_size * scale)

        mx, my = float(mouse_pos.x()), float(mouse_pos.y())
        best_axis = -1
        best_z = float("inf")

        for center, axis in handle_centers:
            sp = self._project_world_to_screen(center)
            if sp is None:
                continue
            dx = sp.x() - mx
            dy = sp.y() - my
            dist = math.hypot(dx, dy)
            if dist <= hit_radius_px:
                # If multiple handles overlap (small ROI), prefer the closest to camera.
                # Use the projected depth (clip Z/W) as tie-breaker.
                if dist < best_z:
                    best_z = dist
                    best_axis = axis

        return best_axis

    def _handle_roi_drag(self, mouse_pos):
        """Handle ROI drag: resize (face handles) or translate (center handle)."""
        if self._roi_drag_axis == 6:
            # Center translation: drag parallel to the ROI center XY plane.
            start_hit = self._roi_drag_start_hit
            if start_hit is None:
                return
            current_hit = self._screen_to_plane(mouse_pos, self._roi_drag_start_center[2])
            if current_hit is None:
                return
            delta = current_hit - start_hit
            self._roi_center = self._roi_drag_start_center + delta
            self._build_roi_buffers()
            self.update()
            return

        axis = self._roi_drag_axis % 3
        sign = 1 if self._roi_drag_axis < 3 else -1
        base = np.zeros(3, dtype=np.float32)
        base[axis] = 1.0

        # Drag the face while keeping the opposite face fixed. We use a
        # screen-space projection so the handle follows the mouse one-to-one
        # regardless of camera angle or perspective distortion.
        h0 = self._roi_drag_start_size * 0.5
        c0 = self._roi_drag_start_center
        old_min = c0[axis] - h0[axis]
        old_max = c0[axis] + h0[axis]
        fixed = old_max if sign < 0 else old_min
        original_face = old_max if sign > 0 else old_min

        # Project the drag axis to screen to know which screen direction
        # corresponds to pulling/pushing this face.
        start_sp = self._project_world_to_screen(c0)
        axis_sp = self._project_world_to_screen(c0 + base * sign * max(h0[axis], 1.0))
        if start_sp is None or axis_sp is None:
            return
        axis_screen_x = float(axis_sp.x() - start_sp.x())
        axis_screen_y = float(axis_sp.y() - start_sp.y())
        axis_screen_len = math.hypot(axis_screen_x, axis_screen_y)
        if axis_screen_len < 1e-3:
            return

        dx = float(mouse_pos.x()) - float(self._roi_drag_start_mouse.x())
        dy = float(mouse_pos.y()) - float(self._roi_drag_start_mouse.y())
        delta_px = (dx * axis_screen_x + dy * axis_screen_y) / axis_screen_len

        scale = self._world_to_screen_scale(c0)
        if scale <= 0:
            return
        delta_world = delta_px / scale

        new_face = original_face + sign * delta_world
        # Keep the dragged face on the correct side of the fixed face.
        if sign > 0:
            new_face = max(new_face, fixed + 0.1)
        else:
            new_face = min(new_face, fixed - 0.1)

        new_center = self._roi_drag_start_center.copy()
        new_center[axis] = (fixed + new_face) * 0.5
        new_size = self._roi_drag_start_size.copy()
        new_size[axis] = max(abs(new_face - fixed), 0.1)

        self._roi_center = new_center
        self._roi_size = new_size
        self._build_roi_buffers()
        self.update()

    # ── Pick / hover ───────────────────────────────────────────────

    def screen_to_ray(self, pos):
        """Convert screen (x, y) to a 3D ray in world space."""
        view = self.camera.view_matrix()
        proj = self.camera.projection_matrix()
        inv_mvp = np.linalg.inv(proj @ view)

        w, h = self.width(), max(self.height(), 1)
        nx = 2.0 * pos.x() / w - 1.0
        ny = 1.0 - 2.0 * pos.y() / h

        near_pt = inv_mvp @ np.array([nx, ny, -1.0, 1.0], dtype=np.float32)
        far_pt = inv_mvp @ np.array([nx, ny, 1.0, 1.0], dtype=np.float32)
        near_pt = near_pt[:3] / near_pt[3]
        far_pt = far_pt[:3] / far_pt[3]

        ray_origin = near_pt
        ray_dir = far_pt - near_pt
        nlen = np.linalg.norm(ray_dir)
        if nlen > 0:
            ray_dir = ray_dir / nlen
        return ray_origin, ray_dir

    def _update_hover(self, mouse_pos):
        """Find nearest point under the mouse cursor."""
        if self._roi_edit_mode or len(self._points) == 0 or len(self._points) > 200000:
            return
        ray_origin, ray_dir = self.screen_to_ray(mouse_pos)

        # Brute-force nearest point along ray (for small clouds)
        # For large clouds, this should use a spatial index
        points = self._points
        to_points = points - ray_origin
        t = np.dot(to_points, ray_dir)
        near_mask = (t > 0)
        if not near_mask.any():
            self.mouse_moved.emit(float("nan"), float("nan"), float("nan"))
            self._hovered_point = None
            return

        # Closest approach distance from each point to the ray
        closest = points[near_mask]
        t_near = t[near_mask]
        proj = ray_origin.reshape(1, 3) + t_near.reshape(-1, 1) * ray_dir.reshape(1, 3)
        dist = np.linalg.norm(closest - proj, axis=1)

        max_dist = self.camera.distance * 0.01  # 1% of camera distance
        close_enough = dist < max_dist
        if not close_enough.any():
            self.mouse_moved.emit(float("nan"), float("nan"), float("nan"))
            self._hovered_point = None
            return

        # Nearest point along ray
        best = np.argmin(t_near[close_enough])
        global_idx = np.where(near_mask)[0][close_enough][best]
        pt = points[global_idx]
        self.mouse_moved.emit(float(pt[0]), float(pt[1]), float(pt[2]))
        self._hovered_point = global_idx

    def _hit_test_point(self, mouse_pos):
        """Find the index of the point under the mouse cursor, or -1."""
        if len(self._points) == 0 or len(self._points) > 200000:
            return -1
        ray_origin, ray_dir = self.screen_to_ray(mouse_pos)
        points = self._points
        to_points = points - ray_origin
        t = np.dot(to_points, ray_dir)

        near_mask = (t > 0)
        if not near_mask.any():
            return -1

        closest = points[near_mask]
        t_near = t[near_mask]
        proj = ray_origin.reshape(1, 3) + t_near.reshape(-1, 1) * ray_dir.reshape(1, 3)
        dist = np.linalg.norm(closest - proj, axis=1)
        max_dist = self.camera.distance * 0.01
        close_enough = dist < max_dist
        if not close_enough.any():
            return -1

        best = np.argmin(t_near[close_enough])
        return int(np.where(near_mask)[0][close_enough][best])

    # ── Public helpers ─────────────────────────────────────────────

    def get_roi(self):
        """Returns dict with current ROI state."""
        return {
            "center": self._roi_center.tolist(),
            "size": self._roi_size.tolist(),
            "rotation": [0.0, 0.0, 0.0],
        }

    def fit_view(self):
        """Reset camera to frame the entire point cloud."""
        if len(self._points) == 0:
            return
        self._bbox_center, self._bbox_size = _compute_bbox(self._points)
        self.camera.target = self._bbox_center.copy()
        self.camera.distance = max(self._bbox_size) * 1.5
        self.camera.distance = max(self.camera.distance, 1.0)
        self.update()

    def set_view_preset(self, name):
        """Set camera to a standard orientation.

        Default camera is azimuth=45, elevation=30 (isometric). With the
        flipped-X point-cloud convention this places the left side of the
        range image on the left of the screen.
        """
        presets = {
            "front": (270, 0),
            "back": (90, 0),
            "left": (180, 0),
            "right": (0, 0),
            "top": (270, 90),
            "bottom": (270, -90),
        }
        if name in presets:
            self.camera.azimuth, self.camera.elevation = presets[name]
            self.update()

    def _show_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("Fit View", self.fit_view)
        sub = menu.addMenu("View Presets")
        for pname in ["top", "bottom", "front", "back", "left", "right"]:
            sub.addAction(pname.capitalize(), lambda n=pname: self.set_view_preset(n))
        menu.exec(self.mapToGlobal(pos))


def _compute_bbox(points):
    finite = np.isfinite(points).all(axis=1)
    if not finite.any():
        return np.zeros(3, dtype=np.float32), np.ones(3, dtype=np.float32) * 1e-6
    pts = points[finite]
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    center = (lo + hi) * 0.5
    size = hi - lo
    size[size < 1e-6] = 1e-6
    return center, size