"""Configuration widget for Height Blobs 3D."""

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QLabel,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QSizePolicy,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from falcocv.core.pointcloud_utils import ensure_pointcloud_colors
from falcocv.core.roi_3d import get_3d_data
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
    add_record_header,
    make_record_selector,
    wrap_params_scroll,
)
from falcocv.ui.widgets.roi_editor_3d import RoiEditor3DWidget
from falcocv.ui.widgets.viewer import ImageViewer

from .logic import HeightBlobs3D, SORT_MODES


class HeightBlobs3DConfig(QWidget):
    def __init__(self, tool: HeightBlobs3D):
        super().__init__()
        self.tool = tool
        self._updating = False
        self._last_cloud_id = None

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self.content_splitter = QSplitter(Qt.Orientation.Horizontal)

        left = QWidget()
        layout = QVBoxLayout(left)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()

        params_page = QWidget()
        params_layout = QVBoxLayout(params_page)
        params_layout.setContentsMargins(0, 0, 0, 0)
        self.roi_editor = RoiEditor3DWidget()
        params_layout.addWidget(self.roi_editor)
        self._build_params(params_layout)
        params_layout.addStretch()
        self.params_page = params_page
        self.tabs.addTab(wrap_params_scroll(params_page), "Parameters")

        results_page = QWidget()
        results_layout = QVBoxLayout(results_page)
        results_layout.setContentsMargins(0, 0, 0, 0)
        self._build_results(results_layout)
        results_layout.addStretch()
        self.tabs.addTab(results_page, "Results")

        layout.addWidget(self.tabs)
        self.left_scroll = left
        self.content_splitter.addWidget(self.left_scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.record_selector = make_record_selector()
        self.record_selector.addItem("input image", "input image")
        self.record_selector.addItem("output image", "output image")
        self.record_selector.addItem("binary mask", "binary mask")
        self.record_selector.currentIndexChanged.connect(self.update_viewer)
        self.chk_show3d = QCheckBox("Show 3D")
        self.chk_show3d.setVisible(False)
        self.chk_show3d.toggled.connect(self._on_show3d_toggled)

        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(
            bool(self.tool.params.get("show_overlays", True))
        )
        self.chk_overlays.toggled.connect(self._on_overlays)
        add_record_header(
            right_layout, self.record_selector,
            (self.chk_show3d, self.chk_overlays),
            tool=self.tool,
        )

        self.viewer_stack = QStackedWidget()
        self.viewer = ImageViewer()
        self.viewer_stack.addWidget(self.viewer)
        try:
            from falcocv.ui.widgets.viewer3d import Viewer3D

            self.viewer3d = Viewer3D()
            self.viewer_stack.addWidget(self.viewer3d)
        except ImportError:
            self.viewer3d = None
        right_layout.addWidget(self.viewer_stack)

        self.content_splitter.addWidget(right)

        self.roi_editor.set_tool(self.tool, self.viewer, self.viewer3d)

        configure_params_splitter(
            self.content_splitter,
            self.left_scroll,
            right,
            "HeightBlobs3D",
            measure=self.params_page,
        )
        self.main_layout.addWidget(self.content_splitter, 1)

        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)
        self.refresh_ui()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------
    @staticmethod
    def _field(label, widget):
        """Compact field: caption above the control (BlobDetector style)."""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(1)
        layout.addWidget(QLabel(label))
        layout.addWidget(widget)
        return container

    def _double_spin(self, key, low, high, decimals, step, suffix=""):
        spin = QDoubleSpinBox()
        spin.setRange(low, high)
        spin.setDecimals(decimals)
        spin.setSingleStep(step)
        spin.setMaximumWidth(160)
        if suffix:
            spin.setSuffix(suffix)
        spin.setValue(float(self.tool.params.get(key, 0.0)))
        spin.valueChanged.connect(
            lambda value, name=key: self._set_param(name, float(value))
        )
        return spin

    def _int_spin(self, key, low, high, suffix="", width=160):
        spin = QSpinBox()
        spin.setRange(low, high)
        spin.setMaximumWidth(width)
        if width <= 120:
            # Two-column grids must be able to shrink below the text width.
            spin.setMinimumWidth(56)
            spin.setSizePolicy(
                QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed
            )
        if suffix:
            spin.setSuffix(suffix)
        spin.setValue(int(self.tool.params.get(key, low)))
        spin.valueChanged.connect(
            lambda value, name=key: self._set_param(name, int(value))
        )
        return spin

    def _combo_field(self, key, options):
        combo = QComboBox()
        for value, label in options:
            combo.addItem(label, value)
        current = str(self.tool.params.get(key, options[0][0]))
        index = combo.findData(current)
        combo.setCurrentIndex(index if index >= 0 else 0)
        combo.setMaximumWidth(160)
        combo.currentIndexChanged.connect(
            lambda _index, control=combo, name=key: self._set_param(
                name, control.currentData()
            )
        )
        return combo

    def _build_params(self, parent_layout):
        plane = QGroupBox("Plane")
        plane_layout = QVBoxLayout(plane)
        self.spin_offset = self._double_spin(
            "plane_offset", -1000.0, 1000.0, 3, 0.1, " mm"
        )
        plane_layout.addWidget(self._field("Plane offset", self.spin_offset))
        self.combo_direction = self._combo_field(
            "direction",
            [("above", "Above the plane"), ("below", "Below the plane")],
        )
        plane_layout.addWidget(self._field("Measure", self.combo_direction))
        parent_layout.addWidget(plane)

        height = QGroupBox("Height mask")
        height_layout = QVBoxLayout(height)
        self.spin_min = self._double_spin(
            "min_height", -1000.0, 1000.0, 3, 0.05, " mm"
        )
        height_layout.addWidget(self._field("Min height", self.spin_min))
        self.spin_seed = self._double_spin(
            "seed_height", -1000.0, 1000.0, 3, 0.05, " mm"
        )
        height_layout.addWidget(self._field("Seed height", self.spin_seed))
        self.spin_tip_band = self._double_spin(
            "tip_band", 0.0, 1000.0, 3, 0.01, " mm"
        )
        height_layout.addWidget(self._field("Tip band", self.spin_tip_band))

        self.spin_median = self._int_spin("median_px", 0, 9, width=110)
        self.spin_close = self._int_spin("close_px", 0, 20, width=110)
        self.spin_open = self._int_spin("open_px", 0, 20, width=110)
        self.spin_min_points = self._int_spin(
            "min_points", 1, 1_000_000, width=110
        )
        self.spin_min_points.setToolTip("Minimum pixels per blob")
        self.spin_max_blobs = self._int_spin(
            "max_blobs", 1, 10_000, width=110
        )
        self.spin_max_blobs.setToolTip("Maximum number of blobs")
        grid = QGridLayout()
        grid.setHorizontalSpacing(6)
        grid.setVerticalSpacing(6)
        for index, (label, control) in enumerate((
            ("Median", self.spin_median),
            ("Close", self.spin_close),
            ("Open", self.spin_open),
            ("Min pts", self.spin_min_points),
            ("Max", self.spin_max_blobs),
        )):
            grid.addWidget(self._field(label, control), index // 2, index % 2)
        height_layout.addLayout(grid)
        parent_layout.addWidget(height)

        split = QGroupBox("Separation")
        split_layout = QVBoxLayout(split)
        self.chk_watershed = QCheckBox("Watershed split")
        self.chk_watershed.setChecked(bool(self.tool.params.get("watershed")))
        self.chk_watershed.toggled.connect(
            lambda checked: self._set_param("watershed", bool(checked))
        )
        split_layout.addWidget(self.chk_watershed)
        self.spin_erode = self._int_spin(
            "watershed_erode_px", 1, 20, " px"
        )
        split_layout.addWidget(self._field("Seed erode", self.spin_erode))
        parent_layout.addWidget(split)

        order = QGroupBox("Order")
        order_layout = QVBoxLayout(order)
        self.combo_sort = self._combo_field(
            "sort_mode", [(mode, mode) for mode in SORT_MODES]
        )
        order_layout.addWidget(self._field("Sort by", self.combo_sort))
        self.chk_desc = QCheckBox("Descending")
        self.chk_desc.setChecked(bool(self.tool.params.get("sort_descending")))
        self.chk_desc.toggled.connect(
            lambda checked: self._set_param("sort_descending", bool(checked))
        )
        order_layout.addWidget(self.chk_desc)
        parent_layout.addWidget(order)

        visibility = QGroupBox("Visibility")
        visibility_layout = QVBoxLayout(visibility)
        self.chk_draw_overlays = QCheckBox("Show Overlays")
        self.chk_draw_overlays.setChecked(
            bool(self.tool.params.get("show_overlays", True))
        )
        self.chk_draw_overlays.toggled.connect(self._on_show_overlays)
        visibility_layout.addWidget(self.chk_draw_overlays)
        self.chk_search_roi = QCheckBox("Show Search ROI")
        self.chk_search_roi.setToolTip(
            "Draw the 2D footprint of each search ROI on the output image"
        )
        self.chk_search_roi.setChecked(
            bool(self.tool.params.get("show_search_roi", True))
        )
        self.chk_search_roi.toggled.connect(
            lambda checked: self._set_param("show_search_roi", bool(checked))
        )
        visibility_layout.addWidget(self.chk_search_roi)
        self.chk_contours = QCheckBox("Show Boundaries")
        self.chk_contours.setToolTip("Draw the contour of each blob")
        self.chk_contours.setChecked(
            bool(self.tool.params.get("show_contours", True))
        )
        self.chk_contours.toggled.connect(
            lambda checked: self._set_param("show_contours", bool(checked))
        )
        visibility_layout.addWidget(self.chk_contours)
        self.chk_centroids = QCheckBox("Show Centroids")
        self.chk_centroids.setToolTip(
            "Draw the blob centroid in 2D and the measured tip in 3D"
        )
        self.chk_centroids.setChecked(
            bool(self.tool.params.get("show_centroids", True))
        )
        self.chk_centroids.toggled.connect(
            lambda checked: self._set_param("show_centroids", bool(checked))
        )
        visibility_layout.addWidget(self.chk_centroids)
        self.chk_labels = QCheckBox("Show Labels")
        self.chk_labels.setToolTip("Draw the blob index and height")
        self.chk_labels.setChecked(
            bool(self.tool.params.get("show_labels", True))
        )
        self.chk_labels.toggled.connect(
            lambda checked: self._set_param("show_labels", bool(checked))
        )
        visibility_layout.addWidget(self.chk_labels)
        self.chk_copy_overlays = QCheckBox("Copy overlays")
        self.chk_copy_overlays.setToolTip(
            "Copy the blob overlays to the input image"
        )
        self.chk_copy_overlays.setChecked(
            bool(self.tool.params.get("copy_overlays_to_input", True))
        )
        self.chk_copy_overlays.toggled.connect(self._on_copy_overlays)
        visibility_layout.addWidget(self.chk_copy_overlays)
        parent_layout.addWidget(visibility)

    def _build_results(self, parent_layout):
        group = QGroupBox("Results")
        layout = QVBoxLayout(group)
        self.label_count = QLabel("Count: 0")
        layout.addWidget(self.label_count)
        self.table = QTableWidget(0, 6)
        self.table.setHorizontalHeaderLabels(
            ["#", "u", "v", "h", "depth", "area"]
        )
        self.table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        layout.addWidget(self.table)
        parent_layout.addWidget(group)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------
    def _set_param(self, key, value):
        if self._updating:
            return
        self.tool.params[key] = value
        self.tool.mark_dirty()

    def _on_show_overlays(self, checked):
        self.tool.params["show_overlays"] = bool(checked)
        self.chk_overlays.blockSignals(True)
        self.chk_overlays.setChecked(bool(checked))
        self.chk_overlays.blockSignals(False)
        self.tool.mark_dirty()

    def _on_copy_overlays(self, checked):
        self.tool.params["copy_overlays_to_input"] = bool(checked)
        self.tool.mark_dirty()

    def _on_overlays(self, checked):
        self.chk_draw_overlays.blockSignals(True)
        self.chk_draw_overlays.setChecked(bool(checked))
        self.chk_draw_overlays.blockSignals(False)
        self.tool.params["show_overlays"] = bool(checked)
        self.tool.mark_dirty()

    def _on_show3d_toggled(self, checked):
        self.viewer_stack.setCurrentIndex(1 if checked else 0)
        self.update_viewer()

    @safe_refresh
    def refresh_ui(self):
        self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")

        count = int(self.tool.outputs.get("count") or 0)
        self.label_count.setText(f"Count: {count}")
        results = self.tool.outputs.get("results") or []
        self.table.setRowCount(len(results))
        for row, result in enumerate(results):
            values = (
                result.get("index"),
                f"{result.get('u', 0.0):.3f}",
                f"{result.get('v', 0.0):.3f}",
                f"{result.get('h', 0.0):.3f}",
                f"{result.get('depth', 0.0):.3f}",
                f"{result.get('area_mm2', 0.0):.3f}",
            )
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(str(value)))

        warnings = self.tool.outputs.get("warnings") or []
        if warnings and self.tool.last_error is None:
            self.label_time.setToolTip("\n".join(str(w) for w in warnings))
        self.update_viewer()

    def update_viewer(self):
        key = self.record_selector.currentData()
        if key == "input image":
            image = self.tool.inputs.get("input image")
        else:
            image = self.tool.outputs.get(key)
        self.viewer.set_image(
            image,
            auto_fit=True,
            render_overlays_flag=(
                key != "binary mask" and self.chk_overlays.isChecked()
            ),
        )
        if key == "binary mask":
            # The ROI belongs to the image, not to the mask.
            self.viewer.set_roi(None)
            self.roi_editor.btn_draw.setChecked(False)
            self.roi_editor.btn_edit_3d.setChecked(False)
        else:
            self.roi_editor._update_viewer_2d_roi()
        self._update_3d_viewer()

    def _update_3d_viewer(self):
        if self.viewer3d is None:
            return
        source = self.tool.inputs.get("input image")
        cloud = get_3d_data(source) if source is not None else None
        has_3d = (
            cloud is not None
            and cloud.get("points") is not None
            and len(cloud["points"]) > 0
        )
        self.chk_show3d.setVisible(has_3d)
        if not has_3d:
            self.viewer3d.set_overlays_from_image(None)
            self.viewer3d.set_data(np.zeros((0, 3), dtype=np.float32))
            if self.chk_show3d.isChecked():
                self.chk_show3d.setChecked(False)
            return
        if not self.chk_show3d.isChecked():
            return

        overlays_image = self.tool.outputs.get("output image") or source
        if self.chk_overlays.isChecked():
            self.viewer3d.set_overlays_from_image(overlays_image)
        else:
            self.viewer3d.set_overlays_from_image(None)

        points = cloud.get("points")
        colors = cloud.get("colors")
        if colors is None or len(colors) != len(points):
            ensure_pointcloud_colors(source)
            colors = cloud.get("colors")
        if colors is None or len(colors) != len(points):
            colors = np.full((len(points), 3), 180, dtype=np.uint8)
        self.viewer3d.set_data(
            points,
            colors,
            bbox_center=cloud.get("bbox_center"),
            bbox_size=cloud.get("bbox_size"),
        )

    def on_config_closed(self):
        save_params_splitter(self.content_splitter, "HeightBlobs3D")
