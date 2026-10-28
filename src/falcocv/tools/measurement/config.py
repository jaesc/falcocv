import cv2
import numpy as np
import math
from falcocv.core.tool_base import BaseTool, ToolResult, ToolResults
from falcocv.core.vision_image import VisionImage
from falcocv.core.overlay_layer import OverlayType
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QComboBox, QGroupBox, QCheckBox, QDoubleSpinBox,
                             QPushButton, QSpinBox, QScrollArea, QFrame,
                             QSplitter)
from PyQt6.QtCore import Qt, QTimer
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)
from falcocv.ui.style import BORDER, SUCCESS
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.widgets.viewer import ImageViewer


from .logic import (
    MeasurementTool, MeasurementResult,
    MEASURABLE_TYPES, STRATEGIES, COLOR_PRESETS, COLOR_NAMES,
    MODE_NAMES, SINGLE_ELEM_MODES, MODE_VALID_TYPES, MODE_UNIT,
    DEFAULT_ROW,
    _closest_on_segment, _color_name_from_bgr, _enumerate_elements,
    _find_element, _get_elem_key, _make_label, _reference_point,
)


class MeasurementConfig(QWidget):
    def __init__(self, tool: MeasurementTool):
        super().__init__()
        self.tool = tool
        self._suppress = False
        self._row_widgets = []

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        self.content_layout = QSplitter(Qt.Orientation.Horizontal)
        self.controls_container = QWidget()
        self.controls_layout = QVBoxLayout(self.controls_container)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)

        # Overlay Measurements
        self.overlay_group = QGroupBox("Overlay Measurements")
        self.overlay_outer_lyt = QVBoxLayout(self.overlay_group)

        btn_row = QHBoxLayout()
        self.btn_add = QPushButton("+ Add Row")
        self.btn_add.clicked.connect(self._add_row)
        self.btn_remove = QPushButton("x Remove Last")
        self.btn_remove.clicked.connect(self._remove_last)
        btn_row.addWidget(self.btn_add)
        btn_row.addWidget(self.btn_remove)
        btn_row.addStretch()
        self.overlay_outer_lyt.addLayout(btn_row)

        self.rows_scroll = QScrollArea()
        self.rows_scroll.setWidgetResizable(True)
        self.scroll_inner = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_inner)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.rows_scroll.setWidget(self.scroll_inner)
        self.overlay_outer_lyt.addWidget(self.rows_scroll)
        self.controls_layout.addWidget(self.overlay_group)

        # Calibration
        cal_group = QGroupBox("Calibration")
        cal_lyt = QVBoxLayout(cal_group)
        self.spin_pxmm = QDoubleSpinBox()
        self.spin_pxmm.setRange(0.001, 100000); self.spin_pxmm.setDecimals(4)
        self.spin_pxmm.setValue(self.tool.params["pixels_per_mm"])
        self.spin_pxmm.valueChanged.connect(lambda v: self._set_param("pixels_per_mm", v))
        cal_lyt.addWidget(QLabel("Pixels per mm:")); cal_lyt.addWidget(self.spin_pxmm)
        self.controls_layout.addWidget(cal_group)

        # Visibility
        vis_group = QGroupBox("Visibility")
        vis_lyt = QVBoxLayout(vis_group)
        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(True)
        self.chk_overlays.toggled.connect(self._on_overlays_toggled)
        vis_lyt.addWidget(self.chk_overlays)
        self.chk_lines = QCheckBox("Show Measurement Lines")
        self.chk_lines.setChecked(self.tool.params.get("show_measurement_lines", True))
        self.chk_lines.toggled.connect(lambda v: self._set_param("show_measurement_lines", v))
        vis_lyt.addWidget(self.chk_lines)
        self.chk_labels = QCheckBox("Show Labels")
        self.chk_labels.setChecked(self.tool.params.get("show_labels", True))
        self.chk_labels.toggled.connect(lambda v: self._set_param("show_labels", v))
        vis_lyt.addWidget(self.chk_labels)
        self.chk_copy_in = QCheckBox("Copy overlays to input image")
        self.chk_copy_in.setChecked(self.tool.params.get("copy_overlays_to_input", True))
        self.chk_copy_in.toggled.connect(lambda v: self._set_param("copy_overlays_to_input", v))
        vis_lyt.addWidget(self.chk_copy_in)
        self.controls_layout.addWidget(vis_group)

        self.controls_layout.addStretch()
        self.left_scroll = wrap_params_scroll(self.controls_container)
        self.content_layout.addWidget(self.left_scroll)

        # Viewer
        self.viewer_container = QWidget()
        self.viewer_layout = QVBoxLayout(self.viewer_container)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.record_selector = QComboBox()
        self.record_selector.addItem("Input Image", "input image")
        self.record_selector.addItem("Output Image", "output image")
        self.record_selector.setCurrentIndex(1)
        self.record_selector.currentIndexChanged.connect(self.on_record_changed)
        self.viewer_layout.addWidget(self.record_selector)
        self.viewer = ImageViewer()
        self.viewer_layout.addWidget(self.viewer)
        self.content_layout.addWidget(self.viewer_container)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.viewer_container, "MeasurementTool"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)
        self._rebuild_rows()
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))

    # ── Overlay row management ──────────────────────────────────────────

    def _clear_row_widgets(self):
        self._row_widgets.clear()
        self.rows_scroll.setWidget(None)
        if self.scroll_inner:
            self.scroll_inner.deleteLater()
            self.scroll_inner = None
        self.scroll_inner = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_inner)
        self.scroll_layout.setContentsMargins(0, 0, 0, 0)
        self.scroll_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.rows_scroll.setWidget(self.scroll_inner)

    def _rebuild_rows(self):
        self._suppress = True
        self._clear_row_widgets()
        measurements = self.tool.params.get("overlay_measurements", [])
        all_items = self._get_current_items()
        for idx, row in enumerate(measurements):
            mode = row.get("mode", "distance")
            valid_types = MODE_VALID_TYPES.get(mode, {"POINT", "AXES", "LINE", "CIRCLE", "ROI", "CONTOUR", "POLYGON"})
            filtered = [it for it in all_items if it["type"] in valid_types]
            w = self._create_row_widget(idx, row, filtered)
            self.scroll_layout.addWidget(w)
            self._row_widgets.append(w)
            self._update_second_elem_visibility(idx)
        self._suppress = False

    def _on_mode_changed(self, row_idx, mode):
        self._update_row_field(row_idx, "mode", mode)
        self._update_second_elem_visibility(row_idx)
        # Rebuild this row's combos with mode-filtered items
        self._rebuild_row_widget(row_idx)

    def _update_second_elem_visibility(self, row_idx):
        if 0 <= row_idx < len(self._row_widgets):
            w = self._row_widgets[row_idx]
            mode_combo = w.findChild(QComboBox, "mode_combo")
            is_single = mode_combo and mode_combo.currentText() in SINGLE_ELEM_MODES
            for name in ("elem2", "strategy2"):
                cb = w.findChild(QComboBox, name)
                if cb:
                    cb.setVisible(not is_single)

    def _rebuild_row_widget(self, row_idx):
        """Replace a single row's widget with fresh combos filtered by mode."""
        if not (0 <= row_idx < len(self._row_widgets)):
            return
        old_w = self._row_widgets[row_idx]
        measurements = self.tool.params.get("overlay_measurements", [])
        if row_idx >= len(measurements):
            return
        row = measurements[row_idx]
        items = self._get_current_items()
        mode = row.get("mode", "distance")
        valid_types = MODE_VALID_TYPES.get(mode, set(MODE_NAMES))
        filtered = [it for it in items if it["type"] in valid_types]
        w = self._create_row_widget(row_idx, row, filtered)
        self.scroll_layout.insertWidget(row_idx, w)
        self.scroll_layout.removeWidget(old_w)
        old_w.setParent(None)
        old_w.deleteLater()
        self._row_widgets[row_idx] = w
        self._update_second_elem_visibility(row_idx)

    def _create_row_widget(self, idx, row, items):
        frame = QFrame()
        frame.setFrameShape(QFrame.Shape.StyledPanel)
        frame.setStyleSheet("QFrame { border: 1px solid " + BORDER + "; margin: 1px; }")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(3, 2, 3, 2)

        chk = QCheckBox()
        chk.setObjectName("row_enabled")
        chk.setChecked(row.get("enabled", True))
        chk.toggled.connect(lambda v, i=idx: self._update_row_field(i, "enabled", v))
        chk.setToolTip("Enable/disable this measurement")
        layout.addWidget(chk)

        mode_combo = QComboBox()
        mode_combo.setObjectName("mode_combo")
        mode_combo.setMinimumWidth(70)
        mode_combo.addItems(MODE_NAMES)
        mode_combo.setCurrentText(row.get("mode", "distance"))
        mode_combo.currentTextChanged.connect(lambda v, i=idx: self._on_mode_changed(i, v))
        layout.addWidget(mode_combo)

        c1 = QComboBox()
        c1.setObjectName("elem1")
        c1.setMinimumWidth(160)
        c1.addItem("-- select --", None)
        for it in items:
            c1.addItem(it["label"], it["key"])
        self._select_combo_by_key(c1, (row.get("source1",""), row.get("ref_id1",""),
                                       row.get("type1","POINT"), idx))
        c1.currentIndexChanged.connect(lambda _, i=idx: self._on_elem_changed(i, 1))
        layout.addWidget(c1)

        s1 = QComboBox()
        s1.setObjectName("strategy1")
        s1.setMinimumWidth(80)
        s1.setToolTip("Strategy for element 1")
        t1 = row.get("type1", "POINT")
        strats = STRATEGIES.get(t1, ["center"])
        s1.addItems(strats)
        s1.setCurrentText(row.get("strategy1", "center"))
        s1.currentTextChanged.connect(lambda v, i=idx: self._update_row_field(i, "strategy1", v))
        layout.addWidget(s1)

        c2 = QComboBox()
        c2.setObjectName("elem2")
        c2.setMinimumWidth(160)
        c2.addItem("-- select --", None)
        for it in items:
            c2.addItem(it["label"], it["key"])
        self._select_combo_by_key(c2, (row.get("source2",""), row.get("ref_id2",""),
                                       row.get("type2","POINT"), idx))
        c2.currentIndexChanged.connect(lambda _, i=idx: self._on_elem_changed(i, 2))
        layout.addWidget(c2)

        s2 = QComboBox()
        s2.setObjectName("strategy2")
        s2.setMinimumWidth(80)
        s2.setToolTip("Strategy for element 2")
        t2 = row.get("type2", "POINT")
        strats2 = STRATEGIES.get(t2, ["center"])
        s2.addItems(strats2)
        s2.setCurrentText(row.get("strategy2", "center"))
        s2.currentTextChanged.connect(lambda v, i=idx: self._update_row_field(i, "strategy2", v))
        layout.addWidget(s2)

        color_combo = QComboBox()
        color_combo.setObjectName("row_color")
        color_combo.setMinimumWidth(70)
        color_combo.addItems(COLOR_NAMES)
        stored_color = row.get("color", [0, 255, 255])
        color_combo.setCurrentText(_color_name_from_bgr(stored_color))
        color_combo.currentTextChanged.connect(lambda v, i=idx: self._update_row_field(i, "color", list(COLOR_PRESETS[v])))
        layout.addWidget(color_combo)

        thick_spin = QSpinBox()
        thick_spin.setObjectName("row_thickness")
        thick_spin.setRange(0, 5)
        thick_spin.setValue(row.get("thickness", 2))
        thick_spin.valueChanged.connect(lambda v, i=idx: self._update_row_field(i, "thickness", v))
        layout.addWidget(thick_spin)

        result_label = QLabel("--")
        result_label.setObjectName("result_label")
        result_label.setMinimumWidth(80)
        result_label.setStyleSheet("color: " + SUCCESS + ";")
        layout.addWidget(result_label)

        return frame

    def _add_row(self):
        measurements = list(self.tool.params.get("overlay_measurements", []))
        measurements.append(dict(DEFAULT_ROW))
        self.tool.params["overlay_measurements"] = measurements
        self._rebuild_rows()
        self.tool.mark_dirty()

    def _remove_last(self):
        measurements = list(self.tool.params.get("overlay_measurements", []))
        if measurements:
            measurements.pop()
            self.tool.params["overlay_measurements"] = measurements
            self._rebuild_rows()
            self.tool.mark_dirty()

    def _on_elem_changed(self, row_idx, elem_idx):
        # First update strategy combo items to match the new element type
        if 0 <= row_idx < len(self._row_widgets):
            w = self._row_widgets[row_idx]
            elem_combo = w.findChild(QComboBox, f"elem{elem_idx}")
            strat_combo = w.findChild(QComboBox, f"strategy{elem_idx}")
            if elem_combo and strat_combo:
                key = elem_combo.currentData()
                if key:
                    tname = key[2]
                    strats = STRATEGIES.get(tname, ["center"])
                    current = strat_combo.currentText()
                    strat_combo.blockSignals(True)
                    strat_combo.clear()
                    strat_combo.addItems(strats)
                    if current in strats:
                        strat_combo.setCurrentText(current)
                    strat_combo.blockSignals(False)
        # Then save from UI (picks up the updated strategy)
        self._save_row_from_ui()

    def _save_row_from_ui(self):
        measurements = []
        for w in self._row_widgets:
            row = dict(DEFAULT_ROW)
            chk = w.findChild(QCheckBox, "row_enabled")
            mode_combo = w.findChild(QComboBox, "mode_combo")
            c1 = w.findChild(QComboBox, "elem1")
            s1 = w.findChild(QComboBox, "strategy1")
            c2 = w.findChild(QComboBox, "elem2")
            s2 = w.findChild(QComboBox, "strategy2")
            color_combo = w.findChild(QComboBox, "row_color")
            thick_spin = w.findChild(QSpinBox, "row_thickness")
            if chk: row["enabled"] = chk.isChecked()
            if mode_combo: row["mode"] = mode_combo.currentText()
            key1 = c1.currentData() if c1 else None
            key2 = c2.currentData() if c2 else None
            if key1:
                row["source1"], row["ref_id1"], row["type1"], _ = key1
            if key2:
                row["source2"], row["ref_id2"], row["type2"], _ = key2
            if s1: row["strategy1"] = s1.currentText()
            if s2: row["strategy2"] = s2.currentText()
            if color_combo: row["color"] = list(COLOR_PRESETS[color_combo.currentText()])
            if thick_spin: row["thickness"] = thick_spin.value()
            measurements.append(row)
        self.tool.params["overlay_measurements"] = measurements
        self.tool.mark_dirty()

    def _update_row_field(self, idx, field, value):
        measurements = list(self.tool.params.get("overlay_measurements", []))
        if 0 <= idx < len(measurements):
            measurements[idx] = dict(measurements[idx])
            measurements[idx][field] = value
            self.tool.params["overlay_measurements"] = measurements
            if not self._suppress:
                self.tool.mark_dirty()

    def _get_current_items(self):
        items = []
        img_input = self.tool.inputs.get("input image")
        if img_input is not None:
            v_img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)
            items = _enumerate_elements(v_img)
        img_output = self.tool.outputs.get("output image")
        if isinstance(img_output, VisionImage):
            seen = {(it["source_tool"], it["ref_id"], it["type"]) for it in items}
            for it in _enumerate_elements(img_output):
                key = (it["source_tool"], it["ref_id"], it["type"])
                if key not in seen and it["source_tool"] == self.tool.name and it["ref_id"].startswith("line_"):
                    items.append(it)
                    seen.add(key)
        return items

    def _select_combo_by_key(self, combo, target_key):
        src, rid, tname, fallback_idx = target_key
        if rid:
            for i in range(combo.count()):
                data = combo.itemData(i)
                if data and data[0] == src and data[1] == rid and data[2] == tname:
                    combo.setCurrentIndex(i)
                    return
            return
        idx = 0
        for i in range(combo.count()):
            data = combo.itemData(i)
            if data and data[0] == src and data[2] == tname:
                if idx == fallback_idx:
                    combo.setCurrentIndex(i)
                    return
                idx += 1

    # ── UI handlers ─────────────────────────────────────────────────────

    def _set_param(self, key, value):
        self.tool.params[key] = value
        self.tool.mark_dirty()

    def _on_overlays_toggled(self, checked):
        self.tool.params["show_overlays"] = checked
        self.refresh_ui()

    def on_record_changed(self):
        self.refresh_ui()

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if not key:
            return

        is_input = (key == "input image")
        img = self.tool.inputs.get("input image") if is_input else self.tool.outputs.get("output image")

        try:
            if isinstance(img, (np.ndarray, VisionImage)):
                self.viewer.set_image(img, auto_fit=auto_fit,
                                      render_overlays_flag=self.chk_overlays.isChecked() if hasattr(self, "chk_overlays") else True)
            else:
                self.viewer.set_image(None)
        except Exception:
            self.viewer.set_image(None)

        if hasattr(self, "label_time"):
            self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")

        results = self.tool.outputs.get("results", [])
        ridx = 0
        for w in self._row_widgets:
            rlbl = w.findChild(QLabel, "result_label")
            if rlbl:
                if ridx < len(results):
                    r = results[ridx]
                    utype = r.type if r.type in MODE_UNIT else "distance"
                    unit = MODE_UNIT[utype]
                    if utype == "angle":
                        rlbl.setText(f"{r.value:.2f}deg")
                        rlbl.setToolTip(f"Angle: {r.value:.4f}deg")
                    elif utype in SINGLE_ELEM_MODES:
                        rlbl.setText(f"{r.value:.2f}{unit}")
                        rlbl.setToolTip(f"{utype.capitalize()}: {r.value:.4f}{unit}")
                    else:
                        rlbl.setText(f"{r.value:.2f}{unit} / {r.angle:.1f}deg")
                        rlbl.setToolTip(f"P1: ({r.x1:.2f},{r.y1:.2f})\nP2: ({r.x2:.2f},{r.y2:.2f})\nDist: {r.value:.4f}{unit}  Angle: {r.angle:.4f}deg")
                    ridx += 1
                else:
                    rlbl.setText("--")
                    rlbl.setToolTip("")

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "MeasurementTool")
