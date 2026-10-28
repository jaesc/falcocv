import os
import numpy as np
from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QFileDialog, QGroupBox, QDoubleSpinBox, QCheckBox, QStackedWidget,
    QSplitter, QComboBox,
    QRadioButton, QSpinBox,
)
from PyQt6.QtCore import Qt
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)
from falcocv.ui.widgets.viewer import ImageViewer
from falcocv.core.depth_viz import COLORMAP_NAMES
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.core.pointcloud_utils import generate_pointcloud
from .logic import DepthRangeLoader


def _ensure_3d(tool: DepthRangeLoader, img_obj):
    depth = img_obj.data if hasattr(img_obj, "data") else img_obj
    if depth is None or not hasattr(depth, "ndim") or depth.ndim != 2:
        return

    if not bool(tool.params.get("generate_3d", False)):
        img_obj._3d_data = None
        img_obj._3d_cache_key = None
        return

    calib = (
        float(getattr(img_obj, "scale_x", 1.0)),
        float(getattr(img_obj, "scale_y", 1.0)),
        float(getattr(img_obj, "scale_z", 1.0)),
        float(getattr(img_obj, "offset_z", 0.0)),
    )

    show_all = bool(tool.params.get("show_all_points", False))
    target_points = int(tool.params.get("target_points", 1_000_000))
    effective_target_points = 0 if show_all else target_points
    cache_key = (*calib, effective_target_points)

    current_key = getattr(img_obj, "_3d_cache_key", None)
    if current_key == cache_key and getattr(img_obj, "_3d_data", None) is not None:
        return

    x_res, y_res, z_res, z_offset = calib
    zero_invalid = bool(tool.params.get("zero_as_invalid", False))
    cloud = generate_pointcloud(
        depth, x_res, y_res, z_res, z_offset, effective_target_points,
        zero_invalid, "TURBO", True,
    )
    img_obj._3d_data = {
        "points": np.asarray(cloud["points"], dtype=np.float32),
        "intensity": np.asarray(cloud["intensity"], dtype=np.float32),
        "colors": np.asarray(cloud["colors"], dtype=np.uint8) if "colors" in cloud else None,
        "bbox_center": np.asarray(cloud["bbox_center"], dtype=np.float32) if "bbox_center" in cloud else None,
        "bbox_size": np.asarray(cloud["bbox_size"], dtype=np.float32) if "bbox_size" in cloud else None,
    }
    img_obj._3d_cache_key = cache_key
    img_obj.metadata["_3d_target_points"] = effective_target_points


class DepthRangeLoaderConfig(QWidget):
    def __init__(self, tool: DepthRangeLoader):
        super().__init__()
        self.tool = tool
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)

        # ── LEFT: Controls ──────────────────────────────────────────
        self.left_panel = QWidget()
        cl = QVBoxLayout(self.left_panel)
        cl.setContentsMargins(0, 0, 0, 0)

        # Mode selection
        mode_grp = QGroupBox("Mode")
        mode_lyt = QVBoxLayout(mode_grp)
        self.radio_file = QRadioButton("Single File")
        self.radio_folder = QRadioButton("Folder Stack")
        if tool.params.get("mode", "File") == "File":
            self.radio_file.setChecked(True)
        else:
            self.radio_folder.setChecked(True)
        self.radio_file.toggled.connect(self._on_mode_changed)
        mode_lyt.addWidget(self.radio_file)
        mode_lyt.addWidget(self.radio_folder)
        cl.addWidget(mode_grp)

        # Path selection
        self.label_path = QLabel()
        self.label_path.setWordWrap(True)
        self.label_path.setStyleSheet("font-size: 10px;")
        cl.addWidget(self.label_path)

        self.btn_select = QPushButton()
        self.btn_select.clicked.connect(self._on_select_path)
        cl.addWidget(self.btn_select)

        # Folder-specific controls
        self.folder_controls = QWidget()
        fc = QVBoxLayout(self.folder_controls)
        fc.setContentsMargins(0, 0, 0, 0)
        fc.addWidget(QLabel("Sort By:"))
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["Name", "Date"])
        self.combo_sort.setCurrentText(tool.params.get("sort_by", "Name"))
        self.combo_sort.currentTextChanged.connect(self._on_sort_changed)
        fc.addWidget(self.combo_sort)
        fc.addWidget(QLabel("Loop Delay (ms):"))
        self.spin_delay = QSpinBox()
        self.spin_delay.setRange(0, 60000)
        self.spin_delay.setValue(int(tool.params.get("delay", 0)))
        self.spin_delay.valueChanged.connect(self._on_delay_changed)
        fc.addWidget(self.spin_delay)
        self.label_count = QLabel()
        fc.addWidget(self.label_count)
        cl.addWidget(self.folder_controls)

        # Conversion group
        grp = QGroupBox("Conversion")
        gl = QVBoxLayout(grp)
        scale_row = QHBoxLayout()
        scale_row.addWidget(QLabel("Scale (x raw):"))
        self.spin_scale = QDoubleSpinBox()
        self.spin_scale.setRange(1e-9, 1e9)
        self.spin_scale.setDecimals(4)
        self.spin_scale.setFixedWidth(110)
        self.spin_scale.setValue(float(tool.params.get("scale", 1.0)))
        self.spin_scale.valueChanged.connect(self._on_scale)
        scale_row.addWidget(self.spin_scale)
        scale_row.addStretch()
        gl.addLayout(scale_row)
        self.chk_zero = QCheckBox("Treat 0 as invalid (NaN)")
        self.chk_zero.setChecked(bool(tool.params.get("zero_as_invalid", True)))
        self.chk_zero.toggled.connect(self._on_zero)
        gl.addWidget(self.chk_zero)
        cl.addWidget(grp)

        # 3D Resolution group
        res_grp = QGroupBox("3D Resolution")
        res_layout = QVBoxLayout(res_grp)
        self.chk_gen3d = QCheckBox("Generate 3D point cloud")
        self.chk_gen3d.setChecked(bool(tool.params.get("generate_3d", False)))
        self.chk_gen3d.toggled.connect(self._on_gen3d)
        res_layout.addWidget(self.chk_gen3d)

        def _add_double(label, param, default=1.0, rng=(-1e9, 1e9)):
            r = QHBoxLayout()
            lbl = QLabel(label)
            lbl.setFixedWidth(50)
            r.addWidget(lbl)
            sp = QDoubleSpinBox()
            sp.setRange(*rng)
            sp.setDecimals(6)
            sp.setFixedWidth(110)
            sp.setValue(float(tool.params.get(param, default)))
            sp.valueChanged.connect(lambda v, p=param: self._on_param(p, v))
            r.addWidget(sp)
            r.addStretch()
            res_layout.addLayout(r)
            return sp

        self.spin_xr = _add_double("X res:", "x_res")
        self.spin_yr = _add_double("Y res:", "y_res")
        self.spin_zr = _add_double("Z res:", "z_res")
        self.spin_zo = _add_double("Z off:", "z_offset", 0.0)

        tp_row = QHBoxLayout()
        tp_row.addWidget(QLabel("Max points:"))
        self.spin_target_points = QSpinBox()
        self.spin_target_points.setRange(100_000, 10_000_000)
        self.spin_target_points.setSingleStep(100_000)
        self.spin_target_points.setValue(int(tool.params.get("target_points", 1_000_000)))
        self.spin_target_points.valueChanged.connect(self._on_target_points)
        tp_row.addWidget(self.spin_target_points)
        tp_row.addStretch()
        res_layout.addLayout(tp_row)

        self.chk_show_all_points = QCheckBox("Show all points")
        self.chk_show_all_points.setChecked(bool(tool.params.get("show_all_points", False)))
        self.chk_show_all_points.toggled.connect(self._on_show_all_points)
        res_layout.addWidget(self.chk_show_all_points)

        self.spin_target_points.setEnabled(not self.chk_show_all_points.isChecked())

        cl.addWidget(res_grp)
        cl.addStretch()

        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_splitter.addWidget(self.left_scroll)

        # ── RIGHT: Viewer ───────────────────────────────────────────
        self.right_panel = QWidget()
        rl = QVBoxLayout(self.right_panel)
        rl.setContentsMargins(0, 0, 0, 0)

        # Colormap bar (for 2D depth display)
        self.depth_bar = QWidget()
        db = QVBoxLayout(self.depth_bar)
        db.setContentsMargins(5, 2, 5, 2)
        db.setSpacing(2)

        top_row = QHBoxLayout()
        top_row.setSpacing(4)
        self.chk_cmap = QCheckBox("Colormap")
        self.chk_cmap.setChecked(True)
        self.chk_cmap.toggled.connect(self._on_cmap_changed)
        top_row.addWidget(self.chk_cmap)
        self.combo_cmap = QComboBox()
        self.combo_cmap.addItems(COLORMAP_NAMES)
        self.combo_cmap.setCurrentText("TURBO")
        self.combo_cmap.currentTextChanged.connect(self._on_cmap_changed)
        top_row.addWidget(self.combo_cmap)
        self.chk_show3d = QCheckBox("Show 3D")
        self.chk_show3d.setVisible(False)
        self.chk_show3d.toggled.connect(self._on_show3d_toggled)
        top_row.addWidget(self.chk_show3d)
        top_row.addStretch()
        db.addLayout(top_row)

        self._updating_viewer = False
        self._current_img_id = None

        rl.addWidget(self.depth_bar)

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

        self.content_splitter.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_splitter, self.left_scroll, self.right_panel, "DepthRangeLoader"
        )
        self.main_layout.addWidget(self.content_splitter, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.label_status = QLabel("")
        self.label_status.setStyleSheet("font-size: 11px;")
        footer_layout = self.footer.layout()
        if isinstance(footer_layout, QHBoxLayout):
            footer_layout.insertWidget(1, self.label_status, 1)
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)
        self.update_ui_state()
        self.refresh_ui()

    # ── Mode / folder ───────────────────────────────────────────────

    def update_ui_state(self):
        mode = self.tool.params.get("mode", "File")
        if mode == "File":
            self.folder_controls.hide()
            self.btn_select.setText("Select Image File")
            self.label_path.setText(f"File: {self.tool.params.get('file_path') or 'None'}")
        else:
            self.folder_controls.show()
            self.btn_select.setText("Select Directory")
            self.label_path.setText(f"Folder: {self.tool.params.get('folder_path') or 'None'}")
            total = len(self.tool._image_files)
            current = self.tool.params.get("current_index", 0) + 1 if total > 0 else 0
            self.label_count.setText(f"Images: {current} / {total}")

    def _on_mode_changed(self):
        mode = "File" if self.radio_file.isChecked() else "Folder"
        self.tool.params["mode"] = mode
        self.update_ui_state()
        self.tool.mark_dirty()

    def _on_select_path(self):
        if self.radio_file.isChecked():
            path, _ = QFileDialog.getOpenFileName(
                self, "Open range / depth image", "",
                "Images (*.tif *.tiff *.png *.bmp *.jpg *.jpeg);;All files (*)",
            )
            if path:
                self.tool.params["file_path"] = path
                self.tool.mark_dirty()
        else:
            path = QFileDialog.getExistingDirectory(self, "Select Folder")
            if path:
                self.tool.params["folder_path"] = path
                self.tool.params["current_index"] = 0
                self.tool._refresh_folder()
                self.tool.mark_dirty()
        self.update_ui_state()

    def _on_sort_changed(self, text):
        self.tool.params["sort_by"] = text
        if self.tool.params.get("mode") == "Folder":
            self.tool._refresh_folder()
            self.update_ui_state()
        self.tool.mark_dirty()

    def _on_delay_changed(self, value):
        self.tool.params["delay"] = value

    # ── Params ──────────────────────────────────────────────────────

    def _on_scale(self, v):
        self.tool.params["scale"] = float(v)

    def _on_zero(self, checked):
        self.tool.params["zero_as_invalid"] = bool(checked)
        self.tool.mark_dirty()

    def _on_gen3d(self, checked):
        self.tool.params["generate_3d"] = bool(checked)
        self.tool.mark_dirty()

    def _on_param(self, param, v):
        self.tool.params[param] = float(v)

    def _on_target_points(self, value):
        self.tool.params["target_points"] = int(value)
        self.tool.mark_dirty()

    def _on_show_all_points(self, checked):
        self.tool.params["show_all_points"] = bool(checked)
        self.spin_target_points.setEnabled(not bool(checked))
        self.tool.mark_dirty()

    # ── Display ─────────────────────────────────────────────────────

    def _save_display_params(self):
        self.tool.params["_display_cmap"] = self.combo_cmap.currentText()

    def _on_cmap_changed(self):
        if self._updating_viewer:
            return
        self._save_display_params()
        self.update_viewer()

    def _depth_display(self):
        use_cmap = self.chk_cmap.isChecked()
        cmap = self.combo_cmap.currentText()
        return {
            "use_colormap": use_cmap,
            "colormap": cmap,
            "vmin": None,
            "vmax": None,
        }

    def _on_show3d_toggled(self, checked):
        if checked:
            img_obj = self.tool.outputs.get("output image")
            if img_obj is not None:
                _ensure_3d(self.tool, img_obj)
        self.viewer_stack.setCurrentIndex(1 if checked else 0)

    # ── Viewer ──────────────────────────────────────────────────────

    def _update_3d_viewer(self, img_obj):
        has_3d = hasattr(img_obj, "_3d_data") and bool(img_obj._3d_data)
        self.chk_show3d.setVisible(has_3d)
        if has_3d:
            d3 = img_obj._3d_data
            pts = d3.get("points")
            colors = d3.get("colors")
            if colors is None:
                colors = np.zeros((0, 3), dtype=np.uint8)
            bbox_center = d3.get("bbox_center")
            bbox_size = d3.get("bbox_size")
            self.viewer3d.set_data(pts, colors, bbox_center=bbox_center, bbox_size=bbox_size)
            if bbox_center is not None and bbox_size is not None:
                self.viewer3d.set_roi(bbox_center, bbox_size)
                self.viewer3d.show_roi(True)
            elif pts is not None and len(pts) > 0:
                from falcocv.core.pointcloud_utils import axis_aligned_bounding_box
                center, size = axis_aligned_bounding_box(pts)
                self.viewer3d.set_roi(center, size)
                self.viewer3d.show_roi(True)
        else:
            self.chk_show3d.setChecked(False)
            if self.viewer3d is not None:
                self.viewer3d.set_data(np.zeros((0, 3), dtype=np.float32))

        self.viewer_stack.setCurrentIndex(1 if (has_3d and self.chk_show3d.isChecked()) else 0)

    def update_viewer(self):
        img_obj = self.tool.outputs.get("output image")
        if img_obj is not None:
            raw = img_obj.data if hasattr(img_obj, "data") else img_obj
            if isinstance(raw, np.ndarray) and raw.ndim == 2 and raw.dtype.kind == "f":
                self._current_img_id = id(img_obj)
                dd = self._depth_display()
                self.depth_bar.show()
                if hasattr(img_obj, "overlay_layers"):
                    self.viewer.set_image(
                        img_obj, auto_fit=True, depth_display=dd,
                        render_overlays_flag=True,
                    )
                else:
                    self.viewer.set_image(
                        raw, auto_fit=True, depth_display=dd,
                    )
            else:
                self.depth_bar.hide()
                self.viewer.set_image(
                    img_obj if hasattr(img_obj, "overlay_layers") else raw,
                    auto_fit=True,
                )

            # 3D generation — synchronous with cache key
            if bool(self.tool.params.get("generate_3d", False)):
                _ensure_3d(self.tool, img_obj)

            self._update_3d_viewer(img_obj)
        else:
            self._current_img_id = None
            self.viewer.set_image(None)
            self.depth_bar.hide()
            self._update_3d_viewer(None)
            self.viewer_stack.setCurrentIndex(0)

    # ── Refresh ────────────────────────────────────────────────────

    @safe_refresh
    def refresh_ui(self):
        self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")
        self._sync_controls()
        self._update_status()
        self.update_viewer()

    def _sync_controls(self):
        widgets = [
            (self.spin_scale, "scale"),
            (self.spin_xr, "x_res"),
            (self.spin_yr, "y_res"),
            (self.spin_zr, "z_res"),
            (self.spin_zo, "z_offset"),
        ]
        for sp, key in widgets:
            sp.blockSignals(True)
            sp.setValue(float(self.tool.params.get(key, 1.0)))
            sp.blockSignals(False)
        self.chk_zero.blockSignals(True)
        self.chk_zero.setChecked(bool(self.tool.params.get("zero_as_invalid", False)))
        self.chk_zero.blockSignals(False)
        self.chk_gen3d.blockSignals(True)
        self.chk_gen3d.setChecked(bool(self.tool.params.get("generate_3d", False)))
        self.chk_gen3d.blockSignals(False)
        self.spin_target_points.blockSignals(True)
        self.spin_target_points.setValue(int(self.tool.params.get("target_points", 1_000_000)))
        self.spin_target_points.blockSignals(False)
        self.chk_show_all_points.blockSignals(True)
        self.chk_show_all_points.setChecked(bool(self.tool.params.get("show_all_points", False)))
        self.chk_show_all_points.blockSignals(False)
        self.spin_target_points.setEnabled(not self.chk_show_all_points.isChecked())

    def _update_status(self):
        if self.tool.last_error:
            msg = str(self.tool.last_error).splitlines()[0]
            self.label_status.setText(f"Error: {msg}")
            self.label_status.setStyleSheet("color: #f44336; font-size: 11px;")
            return

        img_obj = self.tool.outputs.get("output image")
        if img_obj is None:
            self.label_status.setText("No image loaded")
            self.label_status.setStyleSheet("color: #aaaaaa; font-size: 11px;")
            return

        raw = img_obj.data if hasattr(img_obj, "data") else img_obj
        if isinstance(raw, np.ndarray):
            h, w = raw.shape[:2]
            text = f"{w}x{h} output image"
        else:
            text = "output image"

        if self.viewer3d is None:
            text += " | 3D viewer not available"
        elif not getattr(self.viewer3d, "_gl_initialized", False):
            err = getattr(self.viewer3d, "_gl_error", None)
            if err:
                text += f" | 3D viewer error: {err}"
            else:
                text += " | 3D viewer not ready"
        elif hasattr(img_obj, "_3d_data") and img_obj._3d_data:
            pts = img_obj._3d_data.get("points")
            n = len(pts) if pts is not None else 0
            text += f" | Point cloud: {n:,} points"

        self.label_status.setText(text)
        self.label_status.setStyleSheet("color: #aaaaaa; font-size: 11px;")

    def _on_gl_state_changed(self, initialized, error):
        self._update_status()

    def on_config_closed(self):
        save_params_splitter(self.content_splitter, "DepthRangeLoader")
