"""3D ROI editor widget — manages multiple 3D ROIs with Z-range controls.

Bridges the 2D ImageViewer (XY only) and the 3D Viewer3D (full 3D box).
"""

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QDoubleSpinBox, QGroupBox, QListWidget,
    QListWidgetItem, QGridLayout, QSplitter, QSizePolicy,
    QToolTip,
)
from PyQt6.QtCore import Qt, pyqtSignal
from falcocv.core.roi_3d import (
    roi_2d_to_3d, roi_3d_to_2d,
    add_roi, remove_roi, update_roi_2d_in_params,
    update_z_range_in_params, auto_fit_z, calib_from_vimg,
    z_range_from_params,
)
from falcocv.core.pointcloud_utils import axis_aligned_bounding_box
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.style import CAUTION, WHITE


class RoiEditor3DWidget(QGroupBox):
    """Widget for managing multiple 3D ROIs with Z-range and sync."""

    roi_list_changed = pyqtSignal()
    active_roi_changed = pyqtSignal(int)

    def __init__(self, title="3D ROIs", parent=None):
        super().__init__(title, parent)
        self.tool = None
        self.viewer_2d = None
        self.viewer_3d = None
        self._updating = False
        self._active_index = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # ── ROI list ────────────────────────────────────────────────
        list_row = QHBoxLayout()
        self.roi_list = QListWidget()
        self.roi_list.setMaximumHeight(100)
        self.roi_list.currentRowChanged.connect(self._on_selection_changed)
        list_row.addWidget(self.roi_list, 1)

        btn_col = QVBoxLayout()
        self.btn_add = QPushButton("+")
        self.btn_add.setFixedWidth(28)
        self.btn_add.clicked.connect(self._add_roi)
        btn_col.addWidget(self.btn_add)

        self.btn_remove = QPushButton("-")
        self.btn_remove.setFixedWidth(28)
        self.btn_remove.clicked.connect(self._remove_roi)
        btn_col.addWidget(self.btn_remove)
        btn_col.addStretch()
        list_row.addLayout(btn_col)
        layout.addLayout(list_row)

        # ── 2D ROI controls ─────────────────────────────────────────
        self._build_2d_controls(layout)

        # ── Z-range controls ────────────────────────────────────────
        self._build_z_controls(layout)

        # ── Draw buttons ─────────────────────────────────────────────
        self.btn_draw = QPushButton("Select Region in 2D View")
        self.btn_draw.setCheckable(True)
        self.btn_draw.toggled.connect(self._toggle_draw)
        self.btn_draw.setStyleSheet(f"""
            QPushButton:checked {{
                background-color: {CAUTION};
                color: {WHITE};
                font-weight: bold;
            }}
        """)
        layout.addWidget(self.btn_draw)

        draw3d_row = QHBoxLayout()
        self.btn_edit_3d = QPushButton("Edit ROI in 3D")
        self.btn_edit_3d.setCheckable(True)
        self.btn_edit_3d.setToolTip("Drag face handles to resize/translate the ROI")
        self.btn_edit_3d.toggled.connect(self._toggle_edit_3d)
        draw3d_row.addWidget(self.btn_edit_3d)

        self.btn_draw_3d = QPushButton("Draw ROI in 3D")
        self.btn_draw_3d.setCheckable(True)
        self.btn_draw_3d.setToolTip("Click and drag on the 3D view to draw a new ROI")
        self.btn_draw_3d.toggled.connect(self._toggle_draw_3d)
        draw3d_row.addWidget(self.btn_draw_3d)
        layout.addLayout(draw3d_row)

    def _build_2d_controls(self, parent_layout):
        grp = QGroupBox("2D Region (XY)")
        grid = QGridLayout(grp)
        grid.setSpacing(2)
        self._spins_2d = {}
        labels = {"X": (0, 0), "Y": (1, 0), "A": (2, 0), "W": (0, 1), "H": (1, 1)}
        for label, (col, row) in labels.items():
            lbl = QLabel(label)
            lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            lbl.setStyleSheet("font-size: 9px;")
            grid.addWidget(lbl, row * 2, col)

            if label == "A":
                spin = QDoubleSpinBox()
                spin.setRange(-360, 360)
                spin.setSuffix("°")
                spin.setDecimals(2)
            else:
                spin = QDoubleSpinBox()
                spin.setRange(-100000, 100000)
                spin.setDecimals(1)
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            spin.setMinimumWidth(50)
            spin.setMaximumWidth(65)
            spin.setAlignment(Qt.AlignmentFlag.AlignRight)
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            spin.setStyleSheet("font-size: 9px;")
            spin.valueChanged.connect(self._on_spin_2d_changed)
            grid.addWidget(spin, row * 2 + 1, col)
            self._spins_2d[label] = spin
        parent_layout.addWidget(grp)

    def _build_z_controls(self, parent_layout):
        grp = QGroupBox("Z Range (3D)")
        z_layout = QVBoxLayout(grp)

        row = QHBoxLayout()
        row.addWidget(QLabel("Z min:"))
        self.spin_z_min = QDoubleSpinBox()
        self.spin_z_min.setRange(-1e9, 1e9)
        self.spin_z_min.setDecimals(3)
        self.spin_z_min.valueChanged.connect(self._on_z_changed)
        row.addWidget(self.spin_z_min)

        row.addWidget(QLabel("Z max:"))
        self.spin_z_max = QDoubleSpinBox()
        self.spin_z_max.setRange(-1e9, 1e9)
        self.spin_z_max.setDecimals(3)
        self.spin_z_max.valueChanged.connect(self._on_z_changed)
        row.addWidget(self.spin_z_max)

        self.btn_auto_z = QPushButton("Auto-fit Z")
        self.btn_auto_z.clicked.connect(self._on_auto_z)
        row.addWidget(self.btn_auto_z)
        row.addStretch()
        z_layout.addLayout(row)
        parent_layout.addWidget(grp)

    def set_tool(self, tool, viewer_2d=None, viewer_3d=None):
        self.tool = tool
        self.viewer_2d = viewer_2d
        self.viewer_3d = viewer_3d

        if viewer_2d is not None:
            viewer_2d.roi_changed.connect(self._on_viewer_2d_roi)
            viewer_2d.roi_drawing_finished.connect(self._on_drawing_finished)

        if viewer_3d is not None:
            viewer_3d.roi_changed.connect(self._on_viewer_3d_roi)
            viewer_3d.roi_drawing_finished.connect(self._on_drawing_finished_3d)

        self._refresh()

    def _ensure_rois(self):
        if self.tool is None:
            return
        if "rois" not in self.tool.params or not self.tool.params["rois"]:
            add_roi(self.tool.params)

    def _refresh(self):
        if self.tool is None:
            return
        self._updating = True
        self._ensure_rois()
        rois = self.tool.params.get("rois", [])

        # Update list
        self.roi_list.blockSignals(True)
        self.roi_list.clear()
        for i, entry in enumerate(rois):
            roi = entry.get("roi", [0, 0, 0, 0, 0])
            label = f"ROI {i+1}: ({roi[0]:.0f}, {roi[1]:.0f}) {roi[2]:.0f}x{roi[3]:.0f}"
            item = QListWidgetItem(label)
            self.roi_list.addItem(item)
        if self.roi_list.count() > 0 and self._active_index < self.roi_list.count():
            self.roi_list.setCurrentRow(self._active_index)
        self.roi_list.blockSignals(False)

        self._update_spins()
        self._update_viewers()
        self._updating = False

    def _update_spins(self):
        rois = self.tool.params.get("rois", [])
        if not rois or self._active_index >= len(rois):
            return
        entry = rois[self._active_index]
        roi_2d = entry.get("roi", [0, 0, 0, 0, 0])
        z_min = float(entry.get("z_min", 0.0))
        z_max = float(entry.get("z_max", 1.0))

        for label, spin in self._spins_2d.items():
            idx = {"X": 0, "Y": 1, "W": 2, "H": 3, "A": 4}.get(label, 0)
            val = float(roi_2d[idx]) if idx < len(roi_2d) else 0.0
            spin.blockSignals(True)
            spin.setValue(val)
            spin.blockSignals(False)

        self.spin_z_min.blockSignals(True)
        self.spin_z_min.setValue(z_min)
        self.spin_z_min.blockSignals(False)
        self.spin_z_max.blockSignals(True)
        self.spin_z_max.setValue(z_max)
        self.spin_z_max.blockSignals(False)

    def _update_viewer_3d_roi(self):
        if self.viewer_3d is None or self.tool is None:
            return
        if hasattr(self.viewer_3d, 'isVisible') and not self.viewer_3d.isVisible():
            return
        rois = self.tool.params.get("rois", [])
        if not rois or self._active_index >= len(rois):
            self.viewer_3d.show_roi(False)
            return

        entry = rois[self._active_index]
        roi_fixture = entry.get("roi", [0, 0, 0, 0, 0])
        z_min = float(entry.get("z_min", 0.0))
        z_max = float(entry.get("z_max", 1.0))

        img_obj = self.tool.inputs.get("input image") if hasattr(self.tool, "inputs") else None
        depth = None
        if img_obj is not None:
            depth = img_obj.data if hasattr(img_obj, "data") else (img_obj if hasattr(img_obj, "ndim") else None)

        calib = calib_from_vimg(img_obj) if img_obj is not None else (1.0, 1.0, 1.0, 0.0)

        # ROIs are stored in fixture coordinates; convert to absolute image
        # coordinates before projecting to 3D (which uses the flipped-X cloud).
        if img_obj is not None and hasattr(img_obj, "fixture_to_image"):
            roi_2d = img_obj.fixture_to_image(roi_fixture)
        else:
            roi_2d = roi_fixture

        # Auto-fit Z for display only — never modify params
        if z_min == 0.0 and z_max == 1.0 and roi_2d[2] > 0 and roi_2d[3] > 0 and depth is not None:
            from falcocv.core.roi_3d import _auto_z_range
            display_z_min, display_z_max = _auto_z_range(
                depth, roi_2d[0], roi_2d[1], roi_2d[2], roi_2d[3], calib[2], calib[3]
            )
        else:
            display_z_min, display_z_max = z_min, z_max

        center_3d, size_3d, _, _ = roi_2d_to_3d(roi_2d, depth, *calib)

        cz = (display_z_min + display_z_max) * 0.5
        sz = max(display_z_max - display_z_min, 1e-6)
        center_3d[2] = cz
        size_3d[2] = sz

        self.viewer_3d.set_roi(center_3d.tolist(), size_3d.tolist())
        self.viewer_3d.show_roi(True)

    def _update_viewer_2d_roi(self):
        if self.viewer_2d is None or self.tool is None:
            return
        rois = self.tool.params.get("rois", [])
        if not rois or self._active_index >= len(rois):
            return

        entry = rois[self._active_index]
        roi_2d = entry.get("roi", [0, 0, 0, 0, 0])

        img_obj = self.tool.inputs.get("input image") if hasattr(self.tool, "inputs") else None
        if img_obj is not None:
            from falcocv.core.vision_image import VisionImage
            v_img = VisionImage.from_array(img_obj) if not hasattr(img_obj, "fixture_to_image") else img_obj
            roi_abs = v_img.fixture_to_image(roi_2d)
            self.viewer_2d.set_roi(roi_abs)
        else:
            self.viewer_2d.set_roi(roi_2d)

    def _update_viewers(self):
        self._update_viewer_2d_roi()
        self._update_viewer_3d_roi()

    # ── Slots ──────────────────────────────────────────────────────────

    def _on_selection_changed(self, row):
        if row < 0 or self._updating:
            return
        self._active_index = row
        self._update_spins()
        self._update_viewers()
        self.active_roi_changed.emit(row)

    def _on_spin_2d_changed(self):
        if self._updating or self.tool is None:
            return
        rois = self.tool.params.get("rois", [])
        if not rois or self._active_index >= len(rois):
            return
        roi_2d = [
            self._spins_2d["X"].value(),
            self._spins_2d["Y"].value(),
            self._spins_2d["W"].value(),
            self._spins_2d["H"].value(),
            self._spins_2d["A"].value(),
        ]
        update_roi_2d_in_params(self.tool.params, self._active_index, roi_2d)
        self._update_viewers()
        self._update_list_label()
        self.roi_list_changed.emit()

    def _on_z_changed(self):
        if self._updating or self.tool is None:
            return
        z_min = self.spin_z_min.value()
        z_max = self.spin_z_max.value()
        if z_max < z_min:
            z_max = z_min
            self.spin_z_max.blockSignals(True)
            self.spin_z_max.setValue(z_max)
            self.spin_z_max.blockSignals(False)
        update_z_range_in_params(self.tool.params, self._active_index, z_min, z_max)
        self._update_viewer_3d_roi()
        self.roi_list_changed.emit()

    def _on_auto_z(self):
        if self.tool is None:
            return
        img_obj = self.tool.inputs.get("input image") if hasattr(self.tool, "inputs") else None
        if img_obj is None:
            return
        depth = img_obj.data if hasattr(img_obj, "data") else img_obj
        if depth is None or not hasattr(depth, "ndim") or depth.ndim != 2:
            return
        calib = calib_from_vimg(img_obj)
        fix_to_img = img_obj.fixture_to_image if hasattr(img_obj, "fixture_to_image") else None
        result = auto_fit_z(self.tool.params, self._active_index, depth, calib[2], calib[3],
                            fixture_to_image_fn=fix_to_img)
        if result is not None:
            self._update_spins()
            self._update_viewer_3d_roi()
            self.roi_list_changed.emit()

    def _toggle_draw(self, checked):
        if self.viewer_2d:
            self.viewer_2d.set_drawing_new_roi(checked)
        if checked:
            self.btn_draw.setText("DRAWING - Click & drag in 2D view")
            # Disable 3D modes
            self.btn_edit_3d.setChecked(False)
            self.btn_draw_3d.setChecked(False)
        else:
            self.btn_draw.setText("Select Region in 2D View")

    def _toggle_edit_3d(self, checked):
        if self.viewer_3d is not None:
            self.viewer_3d.set_roi_edit_mode(checked)
        if checked:
            self.btn_edit_3d.setText("EDITING ROI - Use handles")
            self.btn_draw.setChecked(False)
            self.btn_draw_3d.setChecked(False)
        else:
            self.btn_edit_3d.setText("Edit ROI in 3D")

    def _toggle_draw_3d(self, checked):
        if self.viewer_3d is not None:
            self.viewer_3d.set_roi_draw_mode(checked)
        if checked:
            self.btn_draw_3d.setText("DRAWING ROI - Click & drag in 3D")
            self.btn_draw.setChecked(False)
            self.btn_edit_3d.setChecked(False)
        else:
            self.btn_draw_3d.setText("Draw ROI in 3D")

    def _on_drawing_finished(self):
        self.btn_draw.blockSignals(True)
        self.btn_draw.setChecked(False)
        self.btn_draw.blockSignals(False)
        self._toggle_draw(False)

    def _on_drawing_finished_3d(self):
        self.btn_draw_3d.blockSignals(True)
        self.btn_draw_3d.setChecked(False)
        self.btn_draw_3d.blockSignals(False)
        if self.viewer_3d is None or self.tool is None:
            return

        self._ensure_rois()
        if self._active_index >= len(self.tool.params.get("rois", [])):
            add_roi(self.tool.params)

        # Try point-picking first (selects actual cloud points under the drag
        # start/end). Fallback to screen-rect back-projection if picking fails.
        roi_2d = self._roi_from_3d_point_pick()
        if roi_2d is None:
            screen_rect = self.viewer_3d.get_draw_screen_rect()
            if screen_rect is not None:
                roi_2d = self._roi_from_3d_screen_rect(screen_rect)

        if roi_2d is None:
            QToolTip.showText(
                self.btn_draw_3d.mapToGlobal(self.btn_draw_3d.rect().bottomLeft()),
                "No point cloud detected under the drag.\n"
                "Draw over the cloud, not on an empty area.",
                self.btn_draw_3d,
            )
            self._update_viewers()
            return

        w_roi = float(roi_2d[2])
        h_roi = float(roi_2d[3])
        if w_roi < 2.0 or h_roi < 2.0:
            QToolTip.showText(
                self.btn_draw_3d.mapToGlobal(self.btn_draw_3d.rect().bottomLeft()),
                "Drag too small — draw a larger area.",
                self.btn_draw_3d,
            )
            self._update_viewers()
            return

        update_roi_2d_in_params(self.tool.params, self._active_index, roi_2d)

        # Auto-fit Z from the depth data for the newly created 2D ROI.
        img_obj = self.tool.inputs.get("input image") if hasattr(self.tool, "inputs") else None
        if img_obj is not None:
            depth = img_obj.data if hasattr(img_obj, "data") else img_obj
            calib = calib_from_vimg(img_obj)
            if depth is not None and hasattr(depth, "ndim") and depth.ndim == 2:
                fix_to_img = img_obj.fixture_to_image if hasattr(img_obj, "fixture_to_image") else None
                auto_fit_z(self.tool.params, self._active_index, depth, calib[2], calib[3],
                           fixture_to_image_fn=fix_to_img)

        z = z_range_from_params(self.tool.params, self._active_index)

        self._update_spins()
        self._update_viewers()
        self._update_list_label()
        self.roi_list_changed.emit()

    def _roi_from_3d_point_pick(self):
        """Create a 2D ROI from the cloud points under the drag start/end.

        This is much more accurate than projecting to a plane when the camera
        is far from the cloud: it selects actual displayed points and converts
        their world coordinates back to image pixels.

        Returns a fixture-coordinate ROI [x, y, w, h, a] or None on failure.
        """
        img_obj = self.tool.inputs.get("input image") if hasattr(self.tool, "inputs") else None
        if img_obj is None or self.viewer_3d is None:
            return None

        start_pos = self.viewer_3d.get_draw_start_screen()
        end_pos = self.viewer_3d.get_draw_current_screen()
        if start_pos is None or end_pos is None:
            return None

        start_world = self.viewer_3d.screen_pos_to_nearest_world_point(start_pos)
        end_world = self.viewer_3d.screen_pos_to_nearest_world_point(end_pos)
        if start_world is None or end_world is None:
            return None

        calib = calib_from_vimg(img_obj)
        scale_x, scale_y, scale_z, offset_z = calib
        depth = img_obj.data if hasattr(img_obj, "data") else img_obj
        if depth is None or not hasattr(depth, "ndim") or depth.ndim != 2:
            return None
        img_h, img_w = depth.shape[:2]

        def world_to_image(pt):
            col = (img_w - 1) - float(pt[0]) / scale_x
            row = float(pt[1]) / scale_y
            return col, row

        c0, r0 = world_to_image(start_world)
        c1, r1 = world_to_image(end_world)

        x0 = float(np.clip(min(c0, c1), 0, img_w - 1))
        y0 = float(np.clip(min(r0, r1), 0, img_h - 1))
        x1 = float(np.clip(max(c0, c1), 0, img_w - 1))
        y1 = float(np.clip(max(r0, r1), 0, img_h - 1))

        if abs(x1 - x0) < 1.0:
            x1 = x0 + 1.0
        if abs(y1 - y0) < 1.0:
            y1 = y0 + 1.0

        roi_image = [x0, y0, x1 - x0, y1 - y0, 0.0]

        if hasattr(img_obj, "image_to_fixture"):
            roi_fixture = img_obj.image_to_fixture(roi_image)
        else:
            roi_fixture = roi_image

        return roi_fixture

    def _roi_from_3d_screen_rect(self, screen_rect):
        """Back-project a screen rectangle to an image ROI (fallback).

        Returns a fixture-coordinate ROI [x, y, w, h, a] or None on failure.
        """
        img_obj = self.tool.inputs.get("input image") if hasattr(self.tool, "inputs") else None
        if img_obj is None:
            return None

        depth = img_obj.data if hasattr(img_obj, "data") else img_obj
        if depth is None or not hasattr(depth, "ndim") or depth.ndim != 2:
            return None

        calib = calib_from_vimg(img_obj)
        scale_x, scale_y, scale_z, offset_z = calib
        img_h, img_w = depth.shape[:2]

        plane_z = self.viewer_3d.get_drawing_plane_z()
        world_pts = self.viewer_3d.screen_rect_to_world_points(screen_rect, plane_z)
        if world_pts is None:
            return None

        # Undo the flipped-X convention used by generate_pointcloud:
        #   world_x = (img_w - 1 - col) * scale_x
        #   world_y = row * scale_y
        cols = (img_w - 1) - world_pts[:, 0] / scale_x
        rows = world_pts[:, 1] / scale_y

        x0 = float(np.clip(np.min(cols), 0, img_w - 1))
        y0 = float(np.clip(np.min(rows), 0, img_h - 1))
        x1 = float(np.clip(np.max(cols), 0, img_w - 1))
        y1 = float(np.clip(np.max(rows), 0, img_h - 1))

        # Avoid zero-area ROIs.
        if abs(x1 - x0) < 1.0:
            x1 = x0 + 1.0
        if abs(y1 - y0) < 1.0:
            y1 = y0 + 1.0

        x = min(x0, x1)
        y = min(y0, y1)
        w = abs(x1 - x0)
        h = abs(y1 - y0)

        roi_image = [x, y, w, h, 0.0]

        if hasattr(img_obj, "image_to_fixture"):
            roi_fixture = img_obj.image_to_fixture(roi_image)
        else:
            roi_fixture = roi_image

        return roi_fixture

    @safe_refresh
    def _on_viewer_2d_roi(self, roi_data):
        if self._updating or self.tool is None:
            return
        # Convert to fixture coordinates
        from falcocv.core.vision_image import VisionImage
        from PyQt6.QtCore import QRectF
        if isinstance(roi_data, QRectF):
            roi_abs = [roi_data.x(), roi_data.y(), roi_data.width(), roi_data.height(), 0.0]
        elif isinstance(roi_data, (list, tuple)):
            roi_abs = [float(v) for v in roi_data]
        else:
            return

        img_input = self.tool.inputs.get("input image") if hasattr(self.tool, "inputs") else None
        if img_input is not None:
            v_img = VisionImage.from_array(img_input)
            roi_fixture = v_img.image_to_fixture(roi_abs)
        else:
            roi_fixture = roi_abs

        self._ensure_rois()
        if self._active_index >= len(self.tool.params.get("rois", [])):
            add_roi(self.tool.params)

        update_roi_2d_in_params(self.tool.params, self._active_index, roi_fixture)

        # Auto-fit Z when drawing a new ROI
        img_obj = self.tool.inputs.get("input image")
        if img_obj is not None:
            depth = img_obj.data if hasattr(img_obj, "data") else img_obj
            if depth is not None and hasattr(depth, "ndim") and depth.ndim == 2:
                calib = calib_from_vimg(img_obj)
                fix_to_img = img_obj.fixture_to_image if hasattr(img_obj, "fixture_to_image") else None
                auto_fit_z(self.tool.params, self._active_index, depth, calib[2], calib[3],
                           fixture_to_image_fn=fix_to_img)

        z = z_range_from_params(self.tool.params, self._active_index)

        self._update_spins()
        self._update_viewer_3d_roi()
        self._update_list_label()
        self.roi_list_changed.emit()

    @safe_refresh
    def _on_viewer_3d_roi(self, roi_dict):
        if self._updating or self.tool is None or self.viewer_3d is None:
            return

        # During a 3D draw gesture we only update the visual box; the real
        # ROI is computed from the screen rectangle when the gesture finishes.
        if self.viewer_3d.roi_draw_mode_active():
            return

        center_3d = roi_dict.get("center", [0, 0, 0])
        size_3d = roi_dict.get("size", [1, 1, 1])

        # Get calibration to convert 3D XY back to 2D pixels
        img_obj = self.tool.inputs.get("input image") if hasattr(self.tool, "inputs") else None
        calib = calib_from_vimg(img_obj) if img_obj is not None else (1.0, 1.0, 1.0, 0.0)
        scale_x, scale_y, scale_z, offset_z = calib

        image_width = None
        if img_obj is not None:
            depth = img_obj.data if hasattr(img_obj, "data") else (img_obj if hasattr(img_obj, "ndim") else None)
            if depth is not None and hasattr(depth, "shape") and depth.ndim >= 2:
                image_width = int(depth.shape[1])

        roi_2d_image = roi_3d_to_2d(center_3d, size_3d, scale_x, scale_y, image_width=image_width)
        z_min = float(center_3d[2]) - float(size_3d[2]) * 0.5
        z_max = float(center_3d[2]) + float(size_3d[2]) * 0.5

        # Convert image-pixel ROI to fixture coordinates to match how 2D-drawn
        # ROIs are stored in params.
        if img_obj is not None and hasattr(img_obj, "image_to_fixture"):
            roi_2d = img_obj.image_to_fixture(roi_2d_image)
        else:
            roi_2d = roi_2d_image

        self._ensure_rois()
        if self._active_index >= len(self.tool.params.get("rois", [])):
            add_roi(self.tool.params)

        update_roi_2d_in_params(self.tool.params, self._active_index, roi_2d)
        update_z_range_in_params(self.tool.params, self._active_index, z_min, z_max)

        self._update_spins()
        self._update_viewer_2d_roi()
        self._update_list_label()
        self.roi_list_changed.emit()

    def _update_list_label(self):
        rois = self.tool.params.get("rois", [])
        if self._active_index < len(rois):
            entry = rois[self._active_index]
            roi = entry.get("roi", [0, 0, 0, 0, 0])
            label = f"ROI {self._active_index + 1}: ({roi[0]:.0f}, {roi[1]:.0f}) {roi[2]:.0f}x{roi[3]:.0f}"
            item = self.roi_list.item(self._active_index)
            if item:
                item.setText(label)

    def _add_roi(self):
        if self.tool is None:
            return
        idx = add_roi(self.tool.params)
        self._active_index = idx
        self._refresh()

    def _remove_roi(self):
        if self.tool is None:
            return
        rois = self.tool.params.get("rois", [])
        if not rois or self._active_index >= len(rois):
            return
        remove_roi(self.tool.params, self._active_index)
        if self._active_index >= len(self.tool.params.get("rois", [])):
            self._active_index = max(0, len(self.tool.params.get("rois", [])) - 1)
        self._refresh()
