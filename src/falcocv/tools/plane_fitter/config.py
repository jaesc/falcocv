import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGroupBox, QStackedWidget, QSplitter, QComboBox,
    QDoubleSpinBox, QSpinBox, QCheckBox, QSizePolicy,
)
from PyQt6.QtCore import Qt
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)
from falcocv.ui.widgets.viewer import ImageViewer
from falcocv.ui.widgets.roi_editor_3d import RoiEditor3DWidget
from falcocv.core.roi_3d import calib_from_vimg, get_3d_data
from .logic import PlaneFitter


class PlaneFitterConfig(QWidget):
    def __init__(self, tool: PlaneFitter):
        super().__init__()
        self.tool = tool
        self._updating = False
        self._current_img_id = None

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── LEFT: Controls ──────────────────────────────────────────
        self.left_panel = QWidget()
        cl = QVBoxLayout(self.left_panel)
        cl.setContentsMargins(0, 0, 0, 0)

        # ROI Editor 3D
        self.roi_editor = RoiEditor3DWidget()
        cl.addWidget(self.roi_editor)

        # Fitting parameters
        self._build_fit_params(cl)

        # Results display
        self._build_results(cl)

        cl.addStretch()
        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_splitter.addWidget(self.left_scroll)

        # ── RIGHT: Viewer ───────────────────────────────────────────
        self.right_panel = QWidget()
        rl = QVBoxLayout(self.right_panel)
        rl.setContentsMargins(0, 0, 0, 0)

        # View toggle bar
        self.view_bar = QWidget()
        vb = QHBoxLayout(self.view_bar)
        vb.setContentsMargins(5, 2, 5, 2)
        vb.setSpacing(4)

        self.record_selector = QComboBox()
        self.record_selector.addItem("input image", "input image")
        self.record_selector.addItem("output image", "output image")
        self.record_selector.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self.record_selector.currentIndexChanged.connect(self._on_record_changed)
        vb.addWidget(self.record_selector)

        self.chk_show3d = QCheckBox("Show 3D")
        self.chk_show3d.setVisible(False)
        self.chk_show3d.toggled.connect(self._on_show3d_toggled)
        vb.addWidget(self.chk_show3d)

        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(bool(self.tool.params.get("show_overlays", True)))
        self.chk_overlays.toggled.connect(self._on_overlays_toggled)
        vb.addWidget(self.chk_overlays)

        vb.addStretch()
        rl.addWidget(self.view_bar)

        self.viewer_stack = QStackedWidget()
        self.viewer = ImageViewer()
        self.viewer_stack.addWidget(self.viewer)  # index 0

        try:
            from falcocv.ui.widgets.viewer3d import Viewer3D
            self.viewer3d = Viewer3D()
            self.viewer3d.gl_state_changed.connect(self._on_gl_state_changed)
            self.viewer_stack.addWidget(self.viewer3d)  # index 1
        except ImportError:
            self.viewer3d = None

        rl.addWidget(self.viewer_stack)

        # Connect ROI editor to both viewers
        self.roi_editor.set_tool(self.tool, self.viewer, self.viewer3d)
        self.roi_editor.roi_list_changed.connect(self._on_roi_list_changed)

        self.content_splitter.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_splitter, self.left_scroll, self.right_panel, "PlaneFitter"
        )
        self.main_layout.addWidget(self.content_splitter, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.label_result = QLabel("")
        self.label_result.setStyleSheet("font-size: 11px;")
        footer_layout = self.footer.layout()
        if isinstance(footer_layout, QHBoxLayout):
            footer_layout.insertWidget(1, self.label_result, 1)
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)
        self.update_ui_state()
        self.refresh_ui()

    def _build_fit_params(self, parent_layout):
        grp = QGroupBox("Fitting Parameters")
        gl = QVBoxLayout(grp)

        method_row = QHBoxLayout()
        method_row.addWidget(QLabel("Method:"))
        self.combo_method = QComboBox()
        self.combo_method.addItems(["ransac", "lsq"])
        self.combo_method.setCurrentText(str(self.tool.params.get("method", "ransac")))
        self.combo_method.currentTextChanged.connect(self._on_method)
        method_row.addWidget(self.combo_method)
        method_row.addStretch()
        gl.addLayout(method_row)

        thr_row = QHBoxLayout()
        thr_row.addWidget(QLabel("Threshold:"))
        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.001, 1e6)
        self.spin_threshold.setDecimals(3)
        self.spin_threshold.setValue(float(self.tool.params.get("distance_threshold", 1.0)))
        self.spin_threshold.valueChanged.connect(self._on_threshold)
        thr_row.addWidget(self.spin_threshold)
        thr_row.addStretch()
        gl.addLayout(thr_row)

        iter_row = QHBoxLayout()
        iter_row.addWidget(QLabel("Max Iter:"))
        self.spin_iter = QSpinBox()
        self.spin_iter.setRange(10, 100000)
        self.spin_iter.setValue(int(self.tool.params.get("max_iterations", 100)))
        self.spin_iter.valueChanged.connect(self._on_iterations)
        iter_row.addWidget(self.spin_iter)
        iter_row.addStretch()
        gl.addLayout(iter_row)

        pts_row = QHBoxLayout()
        pts_row.addWidget(QLabel("Fit Points:"))
        self.spin_max_pts = QSpinBox()
        self.spin_max_pts.setRange(100, 10_000_000)
        self.spin_max_pts.setSingleStep(1000)
        self.spin_max_pts.setValue(int(self.tool.params.get("max_fit_points", 50000)))
        self.spin_max_pts.valueChanged.connect(self._on_max_fit_points)
        pts_row.addWidget(self.spin_max_pts)
        pts_row.addStretch()
        gl.addLayout(pts_row)

        min_row = QHBoxLayout()
        min_row.addWidget(QLabel("Min Points:"))
        self.spin_min_pts = QSpinBox()
        self.spin_min_pts.setRange(3, 1000000)
        self.spin_min_pts.setValue(int(self.tool.params.get("min_points", 10)))
        self.spin_min_pts.valueChanged.connect(self._on_min_points)
        min_row.addWidget(self.spin_min_pts)
        min_row.addStretch()
        gl.addLayout(min_row)

        parent_layout.addWidget(grp)

    def _build_results(self, parent_layout):
        grp = QGroupBox("Results")
        gl = QVBoxLayout(grp)

        self.label_normal = QLabel("Normal: —")
        self.label_normal.setStyleSheet("font-size: 10px;")
        gl.addWidget(self.label_normal)

        self.label_center = QLabel("Center: —")
        self.label_center.setStyleSheet("font-size: 10px;")
        gl.addWidget(self.label_center)

        self.label_distance = QLabel("Distance: —")
        self.label_distance.setStyleSheet("font-size: 10px;")
        gl.addWidget(self.label_distance)

        self.label_inliers = QLabel("Inliers: —")
        self.label_inliers.setStyleSheet("font-size: 10px;")
        gl.addWidget(self.label_inliers)

        self.label_rms = QLabel("RMS Error: —")
        self.label_rms.setStyleSheet("font-size: 10px;")
        gl.addWidget(self.label_rms)

        self.label_polygon = QLabel("Polygon: —")
        self.label_polygon.setStyleSheet("font-size: 10px;")
        self.label_polygon.setWordWrap(True)
        gl.addWidget(self.label_polygon)

        parent_layout.addWidget(grp)

    # ── Slots for parameter changes ─────────────────────────────────

    def _on_method(self, text):
        self.tool.params["method"] = text
        self.tool.mark_dirty()

    def _on_threshold(self, val):
        self.tool.params["distance_threshold"] = float(val)
        self.tool.mark_dirty()

    def _on_iterations(self, val):
        self.tool.params["max_iterations"] = int(val)
        self.tool.mark_dirty()

    def _on_max_fit_points(self, val):
        self.tool.params["max_fit_points"] = int(val)
        self.tool.mark_dirty()

    def _on_min_points(self, val):
        self.tool.params["min_points"] = int(val)
        self.tool.mark_dirty()

    def _on_roi_list_changed(self):
        self.tool.mark_dirty()

    # ── 3D view toggle ──────────────────────────────────────────────

    def _on_show3d_toggled(self, checked):
        self.viewer_stack.setCurrentIndex(1 if checked else 0)
        if checked:
            self._update_3d_viewer(self._get_display_image())

    def _on_overlays_toggled(self, checked):
        self.tool.params["show_overlays"] = bool(checked)
        self.tool.mark_dirty()
        self.update_viewer()

    def _on_gl_state_changed(self, initialized, error):
        pass

    # ── Viewer update ───────────────────────────────────────────────

    def _update_3d_viewer(self, img_obj):
        has_3d = False
        if img_obj is not None:
            cloud = get_3d_data(img_obj)
            has_3d = cloud is not None and cloud.get("points") is not None and len(cloud["points"]) > 0

        self.chk_show3d.setVisible(has_3d)

        if self.viewer3d is None:
            if self.chk_show3d.isChecked():
                self.chk_show3d.setChecked(False)
            return

        if not has_3d:
            self.viewer3d.set_overlays_from_image(None)
            self.viewer3d.set_data(np.zeros((0, 3), dtype=np.float32))
            if self.chk_show3d.isChecked():
                self.chk_show3d.setChecked(False)
            return

        # Only upload/build LOD if the 3D view is actually shown
        if not self.chk_show3d.isChecked():
            return

        # ── Generic 3D overlays (WORLD_POLYGON, etc. from the image) ──
        if hasattr(self, "chk_overlays") and not self.chk_overlays.isChecked():
            self.viewer3d.set_overlays_from_image(None)
        else:
            self.viewer3d.set_overlays_from_image(img_obj)

        cloud = get_3d_data(img_obj)
        pts = cloud.get("points")
        colors = cloud.get("colors")
        if colors is None or len(colors) != len(pts):
            colors = np.full((len(pts), 3), 180, dtype=np.uint8)
        bbox_center = cloud.get("bbox_center")
        bbox_size = cloud.get("bbox_size")

        # Preserve camera when the point cloud has not changed (e.g. ROI edit)
        pts_id = id(pts)
        reset_cam = pts_id != getattr(self, "_last_cloud_id", None)
        self._last_cloud_id = pts_id

        if not reset_cam:
            saved_target = self.viewer3d.camera.target.copy()
            saved_distance = self.viewer3d.camera.distance

        self.viewer3d.set_data(pts, colors, bbox_center=bbox_center, bbox_size=bbox_size)

        if not reset_cam:
            self.viewer3d.camera.target = saved_target
            self.viewer3d.camera.distance = saved_distance

        self.roi_editor._update_viewer_3d_roi()

    # ── Image source switching ────────────────────────────────────

    def _on_record_changed(self):
        key = self.record_selector.currentData()
        if key != "input image":
            self.roi_editor.btn_draw.setChecked(False)
            self.roi_editor.btn_edit_3d.setChecked(False)
            self.roi_editor.btn_draw_3d.setChecked(False)
        self.update_viewer()

    def _get_display_image(self):
        key = self.record_selector.currentData()
        if key == "input image":
            return self.tool.inputs.get("input image")
        return self.tool.outputs.get(key)

    def update_viewer(self):
        img_obj = self._get_display_image()
        if img_obj is not None:
            raw = img_obj.data if hasattr(img_obj, "data") else img_obj
            if isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.dtype.kind == "f":
                self._current_img_id = id(img_obj)
                if hasattr(img_obj, "overlay_layers"):
                    self.viewer.set_image(
                        img_obj, auto_fit=True, depth_display={"use_colormap": True},
                        render_overlays_flag=self.chk_overlays.isChecked() if hasattr(self, "chk_overlays") else True,
                    )
                else:
                    self.viewer.set_image(
                        raw, auto_fit=True, depth_display={"use_colormap": True},
                    )
            else:
                self.viewer.set_image(
                    img_obj if hasattr(img_obj, "overlay_layers") else raw,
                    auto_fit=True,
                )
            # Respect the Show 3D toggle; only force 2D when it is off.
            if not self.chk_show3d.isChecked() and self.viewer_stack.currentIndex() != 0:
                self.viewer_stack.setCurrentIndex(0)
        else:
            self._current_img_id = None
            self.viewer.set_image(None)
            self.viewer_stack.setCurrentIndex(0)

        self._update_3d_viewer(img_obj)
        self.roi_editor._update_viewer_2d_roi()

    # ── Refresh ─────────────────────────────────────────────────────

    @safe_refresh
    def refresh_ui(self):
        self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")
        self._sync_controls()
        self._update_results()
        self.update_viewer()

    def _sync_controls(self):
        self.combo_method.blockSignals(True)
        self.combo_method.setCurrentText(str(self.tool.params.get("method", "ransac")))
        self.combo_method.blockSignals(False)

        self.spin_threshold.blockSignals(True)
        self.spin_threshold.setValue(float(self.tool.params.get("distance_threshold", 1.0)))
        self.spin_threshold.blockSignals(False)

        self.spin_iter.blockSignals(True)
        self.spin_iter.setValue(int(self.tool.params.get("max_iterations", 100)))
        self.spin_iter.blockSignals(False)

        self.spin_max_pts.blockSignals(True)
        self.spin_max_pts.setValue(int(self.tool.params.get("max_fit_points", 50000)))
        self.spin_max_pts.blockSignals(False)

        self.spin_min_pts.blockSignals(True)
        self.spin_min_pts.setValue(int(self.tool.params.get("min_points", 10)))
        self.spin_min_pts.blockSignals(False)

        if hasattr(self, "chk_overlays"):
            self.chk_overlays.blockSignals(True)
            self.chk_overlays.setChecked(bool(self.tool.params.get("show_overlays", True)))
            self.chk_overlays.blockSignals(False)

        self.roi_editor._refresh()

    def _update_results(self):
        results = self.tool.outputs.get("results")
        if results and results.Count > 0:
            r = results[0]
            normal = r.normal if hasattr(r, "normal") else None
            center = r.center if hasattr(r, "center") else None
            distance = float(r.distance) if hasattr(r, "distance") else 0.0
            inlier_count = int(r.inlier_count) if hasattr(r, "inlier_count") else 0
            inlier_ratio = float(r.inlier_ratio) if hasattr(r, "inlier_ratio") else 0.0
            rms = float(r.rms_error) if hasattr(r, "rms_error") else 0.0
            total = int(r.total_points) if hasattr(r, "total_points") else 0
        else:
            normal = center = None
            distance = inlier_count = 0
            inlier_ratio = rms = total = 0

        if normal is not None and hasattr(normal, '__getitem__') and len(normal) >= 3:
            n_str = f"({float(normal[0]):.4f}, {float(normal[1]):.4f}, {float(normal[2]):.4f})"
            self.label_normal.setText(f"Normal: {n_str}")
        else:
            self.label_normal.setText("Normal: —")

        if center is not None and hasattr(center, '__getitem__') and len(center) >= 3:
            c_str = f"({float(center[0]):.2f}, {float(center[1]):.2f}, {float(center[2]):.2f})"
            self.label_center.setText(f"Center: {c_str}")
        else:
            self.label_center.setText("Center: —")

        self.label_distance.setText(f"Distance: {distance:.4f}")
        self.label_inliers.setText(f"Inliers: {inlier_count}/{total} ({inlier_ratio*100:.1f}%)")
        self.label_rms.setText(f"RMS Error: {rms:.4f}")

        polygon = results[0].polygon_2d if (results and results.Count > 0 and hasattr(results[0], "polygon_2d")) else None
        if polygon is not None and len(polygon) >= 3:
            pts_str = ", ".join(f"({float(p[0]):.1f},{float(p[1]):.1f})" for p in polygon[:4])
            if len(polygon) > 4:
                pts_str += f", ... ({len(polygon)} pts)"
            self.label_polygon.setText(f"Polygon: {pts_str}")
        else:
            self.label_polygon.setText("Polygon: —")

        if inlier_count > 0:
            self.label_result.setText(f"{inlier_count} inliers, RMS {rms:.3f}")
            self.label_result.setStyleSheet("color: #4caf50; font-size: 11px;")
        elif self.tool.last_error:
            self.label_result.setText(f"Error: {str(self.tool.last_error)[:60]}")
            self.label_result.setStyleSheet("color: #f44336; font-size: 11px;")
        else:
            self.label_result.setText("Run the tool to fit a plane")
            self.label_result.setStyleSheet("color: #aaaaaa; font-size: 11px;")

    def update_ui_state(self):
        pass

    def on_config_closed(self):
        save_params_splitter(self.content_splitter, "PlaneFitter")
