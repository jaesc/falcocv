import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi, create_roi_only_output, composite_roi_to_full
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QSpinBox, QDoubleSpinBox, QCheckBox,
    QSplitter, QTableWidget, QTableWidgetItem, QHeaderView, QStackedWidget,
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

OPERATION_DEFS = {
    "add": {
        "label": "Add (A + B)",
        "inputs": 2,
        "params": [],
    },
    "subtract": {
        "label": "Subtract (A − B)",
        "inputs": 2,
        "params": [
            {"key": "reverse", "label": "Reverse (B - A)", "type": "bool", "default": False},
        ],
    },
    "abs_diff": {
        "label": "Abs Diff (A, B)",
        "inputs": 2,
        "params": [],
    },
    "multiply": {
        "label": "Multiply (A * B)",
        "inputs": 2,
        "params": [],
    },
    "divide": {
        "label": "Divide (A / B)",
        "inputs": 2,
        "params": [
            {"key": "reverse", "label": "Reverse (B / A)", "type": "bool", "default": False},
            {"key": "scale", "label": "Scale", "type": "float", "default": 1.0, "min": 0.001},
        ],
    },
    "max": {
        "label": "Max (A, B)",
        "inputs": 2,
        "params": [],
    },
    "min": {
        "label": "Min (A, B)",
        "inputs": 2,
        "params": [],
    },
    "weighted_add": {
        "label": "Weighted Add (A, B)",
        "inputs": 2,
        "params": [
            {"key": "alpha", "label": "Alpha (A weight)", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            {"key": "beta", "label": "Beta (B weight)", "type": "float", "default": 0.5, "min": 0.0, "max": 1.0},
            {"key": "gamma", "label": "Gamma (offset)", "type": "float", "default": 0.0},
        ],
    },
    "bitwise_and": {
        "label": "AND (A, B)",
        "inputs": 2,
        "params": [],
    },
    "bitwise_or": {
        "label": "OR (A, B)",
        "inputs": 2,
        "params": [],
    },
    "bitwise_xor": {
        "label": "XOR (A, B)",
        "inputs": 2,
        "params": [],
    },
    "bitwise_not": {
        "label": "NOT (A)",
        "inputs": 1,
        "params": [],
    },
    "add_constant": {
        "label": "Add Const (A)",
        "inputs": 1,
        "params": [
            {"key": "value", "label": "Value", "type": "int", "default": 0, "min": 0, "max": 255},
        ],
    },
    "multiply_constant": {
        "label": "Mul Const (A)",
        "inputs": 1,
        "params": [
            {"key": "value", "label": "Value", "type": "float", "default": 1.0, "min": 0.0},
        ],
    },
}

OPERATION_ORDER = [
    "add", "subtract", "abs_diff", "multiply", "divide",
    "max", "min", "weighted_add",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not",
    "add_constant", "multiply_constant",
]


from .logic import _make_default_op, _op_summary, ImageCalculatorTool
class ImageCalculatorConfig(QWidget):
    def __init__(self, tool: ImageCalculatorTool):
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

        # ROI
        self.roi_editor = RoiEditorWidget()
        self.controls_layout.addWidget(self.roi_editor)

        # Operations table
        op_group = QGroupBox("Operations")
        op_lyt = QVBoxLayout(op_group)

        btn_lyt = QHBoxLayout()
        self.btn_add = QPushButton("+ Add")
        self.btn_add.clicked.connect(self._add_op)
        btn_lyt.addWidget(self.btn_add)
        self.btn_remove = QPushButton("− Remove")
        self.btn_remove.clicked.connect(self._remove_op)
        btn_lyt.addWidget(self.btn_remove)
        op_lyt.addLayout(btn_lyt)

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Type", "Params"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(0, 30)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.itemSelectionChanged.connect(self._on_selection_changed)
        op_lyt.addWidget(self.table)

        self.controls_layout.addWidget(op_group)

        # Param stack
        self.param_stack = QStackedWidget()
        self._param_pages = {}
        for op_type in OPERATION_ORDER:
            page = self._build_param_page(op_type)
            self.param_stack.addWidget(page)
            self._param_pages[op_type] = page
        self.param_stack.setVisible(False)
        self.controls_layout.addWidget(self.param_stack)

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

        # --- RIGHT PANEL ---
        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)

        header_lyt = QHBoxLayout()
        self.record_selector = QComboBox()
        self.record_selector.addItem("Output Image", "output image")
        self.record_selector.addItem("Image A", "image_A")
        self.record_selector.addItem("Image B", "image_B")
        self.record_selector.currentIndexChanged.connect(self._on_record_changed)
        header_lyt.addWidget(self.record_selector)
        self.viewer_layout.addLayout(header_lyt)

        self.viewer = ImageViewer()
        self.roi_editor.set_tool(self.tool, self.viewer)
        self.roi_editor.set_input_key("image_A")
        self.viewer_layout.addWidget(self.viewer)

        self.content_layout.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "ImageCalculatorTool"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))
        QTimer.singleShot(50, self._rebuild_table)
        QTimer.singleShot(500, self._on_record_changed)

    # --- param page building ---

    def _build_param_page(self, op_type):
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(5, 5, 5, 5)
        defs = OPERATION_DEFS.get(op_type)
        if defs:
            for pdef in defs["params"]:
                key = pdef["key"]
                label = pdef["label"]
                typ = pdef["type"]
                default = pdef.get("default")
                pmin = pdef.get("min")
                pmax = pdef.get("max")
                step = pdef.get("step", 1 if typ == "int" else 0.1)
                h = QHBoxLayout()
                label_w = QLabel(label + ":")
                h.addWidget(label_w)
                if typ == "int":
                    w = QSpinBox()
                    w.setRange(pmin if pmin is not None else 0, pmax if pmax is not None else 99999)
                    w.setSingleStep(step)
                    w.setValue(default)
                    w.setFixedWidth(100)
                elif typ == "float":
                    w = QDoubleSpinBox()
                    w.setRange(pmin if pmin is not None else -999999, pmax if pmax is not None else 999999)
                    w.setSingleStep(step)
                    w.setValue(default)
                    w.setFixedWidth(110)
                elif typ == "bool":
                    w = QCheckBox(label)
                    w.setChecked(default)
                    h = QHBoxLayout()
                    h.addWidget(w)
                    layout.addLayout(h)
                    page._controls = getattr(page, "_controls", {})
                    page._controls[key] = w
                    continue
                elif typ == "choice":
                    w = QComboBox()
                    for val, lbl in pdef.get("choices", []):
                        w.addItem(lbl, val)
                    idx = w.findData(default)
                    if idx >= 0:
                        w.setCurrentIndex(idx)
                else:
                    continue
                w.valueChanged.connect(lambda: self._on_param_changed())
                h.addWidget(w)
                layout.addLayout(h)
                page._controls = getattr(page, "_controls", {})
                page._controls[key] = w
        layout.addStretch()
        return page

    def _on_param_changed(self):
        if self._suppress:
            return
        row = self.table.currentRow()
        if row < 0:
            return
        ops = list(self.tool.params.get("ops", []))
        if row >= len(ops):
            return
        page = self._param_pages.get(ops[row].get("type"))
        if page:
            ops[row].update(self._read_page_values(page))
            self.tool.params["ops"] = ops
            self.tool.mark_dirty()

    def _read_page_values(self, page):
        vals = {}
        controls = getattr(page, "_controls", {})
        for key, w in controls.items():
            if isinstance(w, QSpinBox) or isinstance(w, QDoubleSpinBox):
                vals[key] = w.value()
            elif isinstance(w, QCheckBox):
                vals[key] = w.isChecked()
            elif isinstance(w, QComboBox):
                vals[key] = w.currentData()
        return vals

    def _on_selection_changed(self):
        if self._suppress:
            return
        row = self.table.currentRow()
        self._show_param_for_row(row)

    def _show_param_for_row(self, row):
        for i in range(self.param_stack.count()):
            self.param_stack.setVisible(False)
        if row < 0:
            return
        ops = self.tool.params.get("ops", [])
        if row >= len(ops):
            return
        op_type = ops[row].get("type")
        page = self._param_pages.get(op_type)
        if page:
            self.param_stack.setCurrentWidget(page)
            self.param_stack.setVisible(True)
            self._populate_page(page, ops[row])

    def _populate_page(self, page, op_dict):
        controls = getattr(page, "_controls", {})
        self._suppress = True
        for key, w in controls.items():
            val = op_dict.get(key)
            if val is None:
                continue
            if isinstance(w, QSpinBox) or isinstance(w, QDoubleSpinBox):
                w.setValue(val)
            elif isinstance(w, QCheckBox):
                w.setChecked(val)
            elif isinstance(w, QComboBox):
                idx = w.findData(val)
                if idx >= 0:
                    w.setCurrentIndex(idx)
        self._suppress = False

    # --- table management ---

    def _add_op(self):
        ops = list(self.tool.params.get("ops", []))
        ftype = OPERATION_ORDER[0]
        ops.append(_make_default_op(ftype))
        self.tool.params["ops"] = ops
        self._rebuild_table()
        self.table.setCurrentCell(len(ops) - 1, 0)
        self.tool.mark_dirty()

    def _remove_op(self):
        row = self.table.currentRow()
        if row < 0:
            return
        ops = list(self.tool.params.get("ops", []))
        if row >= len(ops):
            return
        ops.pop(row)
        self.tool.params["ops"] = ops
        self._rebuild_table()
        if ops:
            new_row = min(row, len(ops) - 1)
            self.table.setCurrentCell(new_row, 0)
        else:
            self.param_stack.setVisible(False)
        self.tool.mark_dirty()

    def _rebuild_table(self):
        self._suppress = True
        self.table.setRowCount(0)
        ops = self.tool.params.get("ops", [])
        for i, op in enumerate(ops):
            self.table.insertRow(i)
            chk = QCheckBox()
            chk.setChecked(op.get("enabled", True))
            chk.toggled.connect(lambda checked, r=i: self._toggle_op(r, checked))
            self.table.setCellWidget(i, 0, chk)

            combo = QComboBox()
            for otype in OPERATION_ORDER:
                defs = OPERATION_DEFS.get(otype)
                combo.addItem(defs["label"] if defs else otype, otype)
            combo.setCurrentText(OPERATION_DEFS.get(op["type"], {}).get("label", op["type"]))
            combo.currentIndexChanged.connect(lambda idx, r=i: self._on_type_changed(r, idx))
            self.table.setCellWidget(i, 1, combo)

            summary_item = QTableWidgetItem(_op_summary(op))
            summary_item.setFlags(summary_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.table.setItem(i, 2, summary_item)

        self._suppress = False

    def _on_type_changed(self, row, idx):
        if self._suppress:
            return
        combo = self.table.cellWidget(row, 1)
        new_type = combo.currentData()
        ops = list(self.tool.params.get("ops", []))
        op = _make_default_op(new_type)
        op["enabled"] = ops[row].get("enabled", True)
        ops[row] = op
        self.tool.params["ops"] = ops
        self.table.item(row, 2).setText(_op_summary(op))
        self._show_param_for_row(row)
        self.tool.mark_dirty()

    def _toggle_op(self, row, checked):
        if self._suppress:
            return
        ops = list(self.tool.params.get("ops", []))
        if row < len(ops):
            ops[row]["enabled"] = checked
            self.tool.params["ops"] = ops
            self.tool.mark_dirty()

    # --- viewer ---

    # --- handlers ---

    def _on_output_mode_changed(self, text):
        self.tool.params["output_mode"] = text
        self.tool.mark_dirty()

    def _on_overlays_toggled(self, checked):
        self.tool.params["show_overlays"] = checked
        self.refresh_ui()

    def _on_record_changed(self):
        key = self.record_selector.currentData()
        is_a = (key == "image_A")
        if hasattr(self, "roi_editor"):
            if not is_a:
                self.roi_editor.btn_edit.setChecked(False)
            self.roi_editor.setEnabled(is_a)
            disabled = disabled_css()
            self.roi_editor.setStyleSheet("" if is_a else disabled)
            if not is_a:
                self.viewer.set_roi_editing(False)
        self.refresh_ui(auto_fit=True)

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if not key:
            return
        is_a = (key == "image_A")
        if key == "image_B":
            img = self.tool.inputs.get("image_B")
        elif is_a:
            img = self.tool.inputs.get("image_A")
        else:
            img = self.tool.outputs.get("output image")
        if isinstance(img, (np.ndarray, VisionImage)):
            self.viewer.set_image(img, auto_fit=auto_fit,
                                  render_overlays_flag=self.chk_overlays.isChecked() if hasattr(self, "chk_overlays") else True)
            if is_a:
                roi_data = self.tool.params["roi"]
                if roi_data[2] > 0 and roi_data[3] > 0:
                    raw = self.tool.inputs.get("image_A")
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
        save_params_splitter(self.content_layout, "ImageCalculatorTool")
