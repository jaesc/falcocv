import numpy as np
from falcocv.core.vision_image import VisionImage
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QSlider, QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QStackedWidget,
)
from PyQt6.QtCore import Qt, QTimer
from falcocv.ui.style import TEXT_MUTED, disabled_css
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)
from falcocv.ui.widgets.viewer import ImageViewer
from falcocv.ui.widgets.roi_editor import RoiEditorWidget

from .logic import (
    FILTER_DEFS, FILTER_TYPE_ORDER,
    _filter_summary, _make_default_filter, FiltersTool,
)


class FiltersConfig(QWidget):
    def __init__(self, tool: FiltersTool):
        super().__init__()
        self.tool = tool
        self._suppress = False

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self.content_layout = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT PANEL ---
        self.left_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.left_panel)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)

        # ROI (above filter table)
        self.roi_editor = RoiEditorWidget()
        self.controls_layout.addWidget(self.roi_editor)

        # Filters group
        filters_group = QGroupBox("Filters")
        filters_lyt = QVBoxLayout(filters_group)

        # Toolbar
        toolbar = QHBoxLayout()
        self.btn_add = QPushButton("+ Add")
        self.btn_add.clicked.connect(self._add_filter)
        self.btn_remove = QPushButton("x Remove")
        self.btn_remove.clicked.connect(self._remove_filter)
        self.btn_up = QPushButton("▲")
        self.btn_up.setFixedWidth(30)
        self.btn_up.clicked.connect(self._move_up)
        self.btn_down = QPushButton("▼")
        self.btn_down.setFixedWidth(30)
        self.btn_down.clicked.connect(self._move_down)
        toolbar.addWidget(self.btn_add)
        toolbar.addWidget(self.btn_remove)
        toolbar.addWidget(self.btn_up)
        toolbar.addWidget(self.btn_down)
        toolbar.addStretch()
        filters_lyt.addLayout(toolbar)

        # Table
        self.filter_table = QTableWidget(0, 3)
        self.filter_table.setHorizontalHeaderLabels(["", "Type", "Parameters"])
        self.filter_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.filter_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.filter_table.verticalHeader().setVisible(False)
        self.filter_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.filter_table.setColumnWidth(0, 30)
        self.filter_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.filter_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.filter_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        filters_lyt.addWidget(self.filter_table)

        # Param stack (one page per filter type)
        self.param_stack = QStackedWidget()
        self._param_pages = {}
        for ftype in FILTER_TYPE_ORDER:
            page = self._build_param_page(ftype)
            self.param_stack.addWidget(page)
            self._param_pages[ftype] = page
        self.param_stack.setVisible(False)
        filters_lyt.addWidget(self.param_stack)

        self.controls_layout.addWidget(filters_group)

        # Output mode
        mode_group = QGroupBox("Output Mode")
        mode_lyt = QVBoxLayout(mode_group)
        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Full Image", "ROI Only"])
        self.combo_mode.setCurrentText(tool.params["output_mode"])
        self.combo_mode.currentTextChanged.connect(self._on_output_mode_changed)
        mode_lyt.addWidget(self.combo_mode)
        self.controls_layout.addWidget(mode_group)

        # Visibility
        vis_group = QGroupBox("Visibility")
        vis_lyt = QVBoxLayout(vis_group)
        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(tool.params.get("show_overlays", True))
        self.chk_overlays.toggled.connect(self._on_overlays_toggled)
        vis_lyt.addWidget(self.chk_overlays)
        self.controls_layout.addWidget(vis_group)

        self.controls_layout.addStretch()
        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)

        # --- RIGHT PANEL: VIEWER ---
        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)

        header_lyt = QHBoxLayout()
        self.record_selector = QComboBox()
        self.record_selector.addItem("Input Image", "input image")
        self.record_selector.addItem("Output Image", "output image")
        self.record_selector.setCurrentIndex(0)
        self.record_selector.currentIndexChanged.connect(self.on_record_changed)
        header_lyt.addWidget(self.record_selector)
        self.viewer_layout.addLayout(header_lyt)

        self.viewer = ImageViewer()
        self.roi_editor.set_tool(self.tool, self.viewer)
        self.viewer_layout.addWidget(self.viewer)

        self.content_layout.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "FiltersTool"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        # Signals
        self.tool.data_changed.connect(self.refresh_ui)
        self._rebuild_table()
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))
        QTimer.singleShot(50, self.on_record_changed)

    # ------------------------------------------------------------------
    # Param page builder
    # ------------------------------------------------------------------

    def _build_param_page(self, ftype):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        defs = FILTER_DEFS.get(ftype)
        if not defs:
            layout.addWidget(QLabel("No parameters"))
            widget._controls = {}
            return widget

        controls = {}
        for pdef in defs["params"]:
            hlyt = QHBoxLayout()
            label = pdef["label"]
            if pdef["type"] in ("int", "float"):
                parts = []
                if "min" in pdef and "max" in pdef:
                    if pdef["type"] == "int":
                        parts.append(f"{pdef['min']}–{pdef['max']}")
                    else:
                        parts.append(f"{pdef['min']:.1f}–{pdef['max']:.1f}")
                elif "min" in pdef:
                    parts.append(f"≥{pdef['min']}" if pdef["type"] == "int" else f"≥{pdef['min']:.1f}")
                elif "max" in pdef:
                    parts.append(f"≤{pdef['max']}" if pdef["type"] == "int" else f"≤{pdef['max']:.1f}")
                if parts:
                    label += f" [{', '.join(parts)}]"
            hlyt.addWidget(QLabel(label))
            if pdef["type"] == "int":
                spin = QSpinBox()
                spin.setRange(-999999, 999999)
                if "min" in pdef: spin.setMinimum(pdef["min"])
                if "max" in pdef: spin.setMaximum(pdef["max"])
                spin.setSingleStep(pdef.get("step", 1))
                spin.setValue(pdef["default"])
                spin.valueChanged.connect(self._on_param_changed)
                controls[pdef["key"]] = spin
                if "min" in pdef and "max" in pdef:
                    slider = QSlider(Qt.Orientation.Horizontal)
                    slider.setRange(pdef["min"], pdef["max"])
                    slider.setSingleStep(pdef.get("step", 1))
                    slider.setValue(pdef["default"])
                    slider.valueChanged.connect(spin.setValue)
                    spin.valueChanged.connect(slider.setValue)
                    hlyt.addWidget(slider)
                hlyt.addWidget(spin)
            elif pdef["type"] == "float":
                spin = QDoubleSpinBox()
                spin.setRange(-999999.0, 999999.0)
                if "min" in pdef: spin.setMinimum(pdef["min"])
                if "max" in pdef: spin.setMaximum(pdef["max"])
                spin.setSingleStep(pdef.get("step", 0.1))
                spin.setDecimals(1)
                spin.setValue(pdef["default"])
                spin.valueChanged.connect(self._on_param_changed)
                controls[pdef["key"]] = spin
                hlyt.addWidget(spin)
            elif pdef["type"] == "bool":
                chk = QCheckBox()
                chk.setChecked(pdef["default"])
                chk.toggled.connect(self._on_param_changed)
                controls[pdef["key"]] = chk
                hlyt.addWidget(chk)
            elif pdef["type"] == "choice":
                combo = QComboBox()
                for val, label in pdef.get("choices", []):
                    combo.addItem(label, val)
                combo.setCurrentIndex(0)
                combo.currentIndexChanged.connect(self._on_param_changed)
                controls[pdef["key"]] = combo
                hlyt.addWidget(combo)
            layout.addLayout(hlyt)
        layout.addStretch()
        widget._controls = controls
        return widget

    def _populate_page(self, page, f):
        if not hasattr(page, "_controls"):
            return
        for key, widget in page._controls.items():
            val = f.get(key)
            if val is None:
                continue
            widget.blockSignals(True)
            if isinstance(widget, QSpinBox):
                widget.setValue(int(val))
            elif isinstance(widget, QDoubleSpinBox):
                widget.setValue(float(val))
            elif isinstance(widget, QCheckBox):
                widget.setChecked(bool(val))
            elif isinstance(widget, QComboBox):
                idx = widget.findData(val) if hasattr(widget, "findData") else -1
                if idx >= 0:
                    widget.setCurrentIndex(idx)
            widget.blockSignals(False)

    def _read_page_values(self, page):
        if not hasattr(page, "_controls"):
            return {}
        values = {}
        for key, widget in page._controls.items():
            if isinstance(widget, QSpinBox):
                values[key] = widget.value()
            elif isinstance(widget, QDoubleSpinBox):
                values[key] = widget.value()
            elif isinstance(widget, QCheckBox):
                values[key] = widget.isChecked()
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentData()
        return values

    # ------------------------------------------------------------------
    # Table management
    # ------------------------------------------------------------------

    def _rebuild_table(self):
        self._suppress = True
        filters = self.tool.params.get("filters", [])
        self.filter_table.setRowCount(len(filters))
        for i, f in enumerate(filters):
            chk = QCheckBox()
            chk.setChecked(f.get("enabled", True))
            chk.toggled.connect(lambda checked, r=i: self._on_enabled_toggled(r, checked))
            self.filter_table.setCellWidget(i, 0, chk)

            combo = QComboBox()
            for ftype in FILTER_TYPE_ORDER:
                defs = FILTER_DEFS.get(ftype)
                combo.addItem(defs["label"] if defs else ftype, ftype)
            combo.setCurrentText(FILTER_DEFS.get(f["type"], {}).get("label", f["type"]))
            combo.currentIndexChanged.connect(lambda idx, r=i: self._on_type_changed(r, idx))
            self.filter_table.setCellWidget(i, 1, combo)

            item = QTableWidgetItem(_filter_summary(f))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.filter_table.setItem(i, 2, item)

        self._suppress = False
        if self.filter_table.rowCount() == 0:
            self.param_stack.setVisible(False)

    def _on_table_selection_changed(self):
        if self._suppress:
            return
        self._show_param_for_row(self.filter_table.currentRow())

    def _show_param_for_row(self, row):
        if row < 0:
            self.param_stack.setVisible(False)
            return
        filters = self.tool.params.get("filters", [])
        if row >= len(filters):
            self.param_stack.setVisible(False)
            return
        f = filters[row]
        ftype = f["type"]
        page = self._param_pages.get(ftype)
        if page:
            self.param_stack.setCurrentWidget(page)
            self.param_stack.setVisible(True)
            self._populate_page(page, f)

    # ------------------------------------------------------------------
    # Filter CRUD
    # ------------------------------------------------------------------

    def _add_filter(self):
        filters = list(self.tool.params.get("filters", []))
        filters.append(_make_default_filter("gaussian_blur"))
        self.tool.params["filters"] = filters
        self._rebuild_table()
        self.filter_table.selectRow(len(filters) - 1)
        self.tool.mark_dirty()

    def _remove_filter(self):
        row = self.filter_table.currentRow()
        if row < 0:
            return
        filters = list(self.tool.params.get("filters", []))
        filters.pop(row)
        self.tool.params["filters"] = filters
        self._rebuild_table()
        if filters:
            target = min(row, len(filters) - 1)
            self.filter_table.selectRow(target)
        self.tool.mark_dirty()

    def _move_up(self):
        row = self.filter_table.currentRow()
        if row <= 0:
            return
        filters = list(self.tool.params.get("filters", []))
        filters[row], filters[row - 1] = filters[row - 1], filters[row]
        self.tool.params["filters"] = filters
        self._rebuild_table()
        self.filter_table.selectRow(row - 1)
        self.tool.mark_dirty()

    def _move_down(self):
        row = self.filter_table.currentRow()
        filters = self.tool.params.get("filters", [])
        if row < 0 or row >= len(filters) - 1:
            return
        filters = list(filters)
        filters[row], filters[row + 1] = filters[row + 1], filters[row]
        self.tool.params["filters"] = filters
        self._rebuild_table()
        self.filter_table.selectRow(row + 1)
        self.tool.mark_dirty()

    def _on_enabled_toggled(self, row, checked):
        if self._suppress:
            return
        filters = list(self.tool.params.get("filters", []))
        if row < len(filters):
            filters[row] = dict(filters[row])
            filters[row]["enabled"] = checked
            self.tool.params["filters"] = filters
            self.tool.mark_dirty()

    def _on_type_changed(self, row, idx):
        if self._suppress:
            return
        combo = self.filter_table.cellWidget(row, 1)
        new_type = combo.currentData()
        filters = list(self.tool.params.get("filters", []))
        f = _make_default_filter(new_type)
        f["enabled"] = filters[row].get("enabled", True)
        filters[row] = f
        self.tool.params["filters"] = filters
        self.filter_table.item(row, 2).setText(_filter_summary(f))
        self._show_param_for_row(row)
        self.tool.mark_dirty()

    def _on_param_changed(self):
        if self._suppress:
            return
        row = self.filter_table.currentRow()
        if row < 0:
            return
        page = self.param_stack.currentWidget()
        if not hasattr(page, "_controls"):
            return
        values = self._read_page_values(page)
        filters = list(self.tool.params.get("filters", []))
        if row >= len(filters):
            return
        f = dict(filters[row])
        f.update(values)
        filters[row] = f
        self.tool.params["filters"] = filters
        self.filter_table.item(row, 2).setText(_filter_summary(f))
        self.tool.mark_dirty()

    # ------------------------------------------------------------------
    # UI callbacks
    # ------------------------------------------------------------------

    def _on_output_mode_changed(self, text):
        self.tool.params["output_mode"] = text
        self.tool.mark_dirty()

    def _on_overlays_toggled(self, checked):
        self.tool.params["show_overlays"] = checked
        self.refresh_ui()

    def on_record_changed(self):
        key = self.record_selector.currentData()
        is_input = (key == "input image")
        if hasattr(self, "roi_editor"):
            if not is_input:
                self.roi_editor.btn_edit.setChecked(False)
            self.roi_editor.setEnabled(is_input)
            disabled = disabled_css()
            self.roi_editor.setStyleSheet("" if is_input else disabled)
            if not is_input:
                self.viewer.set_roi_editing(False)
        self.refresh_ui(auto_fit=True)

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if not key:
            return
        is_input = (key == "input image")
        img = self.tool.inputs.get("input image") if is_input else self.tool.outputs.get("output image")
        if isinstance(img, (np.ndarray, VisionImage)):
            self.viewer.set_image(img, auto_fit=auto_fit,
                                  render_overlays_flag=self.chk_overlays.isChecked() if hasattr(self, "chk_overlays") else True)
            if is_input:
                roi_data = self.tool.params["roi"]
                if roi_data[2] > 0 and roi_data[3] > 0:
                    raw = self.tool.inputs.get("input image")
                    abs_roi = raw.fixture_to_image(roi_data) if isinstance(raw, VisionImage) else roi_data
                    self.viewer.set_roi(abs_roi)
                else:
                    self.viewer.set_roi(None)
            else:
                self.viewer.set_roi(None)
        else:
            self.viewer.set_image(None)
        if hasattr(self, "label_time"):
            self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")
        if hasattr(self, "roi_editor"):
            self.roi_editor.update_from_params()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "FiltersTool")
