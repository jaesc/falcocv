import math
import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QComboBox,
    QPushButton, QSplitter, QGroupBox
)
from PyQt6.QtCore import Qt, QTimer
from falcocv.core.tool_base import BaseTool
from falcocv.core.vision_image import VisionImage
from falcocv.core.overlay_layer import OverlayType
from falcocv.ui.widgets.viewer import ImageViewer
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)


def _make_label(elem, idx):
    src = elem.source_tool or "?"
    rid = elem.ref_id
    tname = elem.element_type.name
    if rid:
        return f"{src} / {rid} ({tname})"
    return f"{src} #{idx} ({tname})"


def _get_elem_key(elem, idx):
    return (elem.source_tool, elem.ref_id, elem.element_type.name, idx)


def _enumerate_elements(v_img):
    items = []
    if not isinstance(v_img, VisionImage):
        return items
    for layer_name, layer in v_img.overlay_layers.items():
        for idx, elem in enumerate(layer.elements):
            items.append({
                "key": _get_elem_key(elem, idx),
                "label": _make_label(elem, idx),
                "source_tool": elem.source_tool,
                "ref_id": elem.ref_id,
                "type": elem.element_type.name,
                "elem_idx": idx,
                "layer": layer_name,
                "elem": elem,
            })
    return items


def _compute_fixture(elem):
    t = elem.element_type
    d = elem.data
    if t == OverlayType.LINE:
        sx, sy = d["start"]
        ex, ey = d["end"]
        cx = (sx + ex) / 2.0
        cy = (sy + ey) / 2.0
        angle = math.degrees(math.atan2(ey - sy, ex - sx))
    elif t == OverlayType.POINT:
        cx, cy = d["center"]
        angle = 0.0
    elif t == OverlayType.CIRCLE:
        cx, cy = d["center"]
        angle = 0.0
    elif t == OverlayType.ROI:
        roi = d["roi"]
        cx = roi[0] + roi[2] / 2.0
        cy = roi[1] + roi[3] / 2.0
        angle = float(roi[4]) if len(roi) > 4 else 0.0
    elif t in (OverlayType.CONTOUR, OverlayType.POLYGON):
        raw = d.get("contour") or d.get("points", [])
        if raw:
            arr = np.array(raw, dtype=np.float64).reshape(-1, 2)
            cx = float(arr[:, 0].mean())
            cy = float(arr[:, 1].mean())
        else:
            cx, cy = 0.0, 0.0
        angle = 0.0
    elif t == OverlayType.TEXT:
        cx, cy = d["position"]
        angle = 0.0
    elif t == OverlayType.AXES:
        cx, cy = d["center"]
        angle = float(d.get("angle", 0.0))
    else:
        cx, cy = 0.0, 0.0
        angle = 0.0
    return (cx, cy, angle)


from .logic import _compute_fixture, _enumerate_elements, _get_elem_key, _make_label, FixtureOverlay
class FixtureOverlayConfig(QWidget):
    def __init__(self, tool: FixtureOverlay):
        super().__init__()
        self.tool = tool

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)

        self.content_layout = QSplitter(Qt.Orientation.Horizontal)

        # --- LEFT PANEL ---
        self.left_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.left_panel)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)

        group = QGroupBox("Fixture Source")
        group_lyt = QVBoxLayout(group)

        group_lyt.addWidget(QLabel("Select overlay element:"))

        self.combo = QComboBox()
        self.combo.setMinimumWidth(200)
        self.combo.currentIndexChanged.connect(self._on_selection_changed)
        group_lyt.addWidget(self.combo)

        btn_refresh = QPushButton("Refresh")
        btn_refresh.clicked.connect(self._populate_combo)
        group_lyt.addWidget(btn_refresh)

        self.controls_layout.addWidget(group)
        self.controls_layout.addStretch()

        # --- RIGHT PANEL: VIEWER ---
        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)

        header_lyt = QHBoxLayout()
        self.record_selector = QComboBox()
        self.record_selector.addItem("Input Image", "input image")
        self.record_selector.addItem("Output Image", "output image")
        self.record_selector.setCurrentIndex(0)
        self.record_selector.currentIndexChanged.connect(self._on_record_changed)
        header_lyt.addWidget(self.record_selector)
        self.viewer_layout.addLayout(header_lyt)

        self.viewer = ImageViewer()
        self.viewer_layout.addWidget(self.viewer)

        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)
        self.content_layout.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "FixtureOverlay"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        # Connections
        self.tool.data_changed.connect(self._refresh_ui)
        self._populate_combo()
        self._refresh_ui()

    def _populate_combo(self):
        self.combo.blockSignals(True)
        self.combo.clear()
        self.combo.addItem("-- none --", None)

        img = self.tool.inputs.get("input image")
        v_img = img if isinstance(img, VisionImage) else None
        if v_img is not None:
            items = _enumerate_elements(v_img)
            for it in items:
                self.combo.addItem(it["label"], it["key"])

            # Restore selection by stable fields (source_tool, ref_id, type)
            st = self.tool.params.get("selected_source_tool")
            rid = self.tool.params.get("selected_ref_id")
            tp = self.tool.params.get("selected_type")
            if st is not None and rid is not None:
                for i in range(self.combo.count()):
                    d = self.combo.itemData(i)
                    if d is not None and d[0] == st and d[1] == rid and d[2] == tp:
                        self.combo.setCurrentIndex(i)
                        break
        self.combo.blockSignals(False)

    def _on_selection_changed(self):
        key = self.combo.currentData()
        if key is not None:
            self.tool.params["selected_source_tool"] = key[0]
            self.tool.params["selected_ref_id"] = key[1]
            self.tool.params["selected_type"] = key[2]
        else:
            self.tool.params["selected_source_tool"] = None
            self.tool.params["selected_ref_id"] = None
            self.tool.params["selected_type"] = None
        self.tool.mark_dirty()

    def _on_record_changed(self):
        self._refresh_ui(auto_fit=True)

    @safe_refresh
    def _refresh_ui(self, auto_fit=False):
        self._populate_combo()
        key = self.record_selector.currentData()
        if not key:
            return
        is_input = (key == "input image")
        img = self.tool.inputs.get("input image") if is_input else self.tool.outputs.get("output image")
        if isinstance(img, (np.ndarray, VisionImage)):
            self.viewer.set_image(img, auto_fit=auto_fit, render_overlays_flag=True)
        else:
            self.viewer.set_image(None)

        # Update footer
        self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "FixtureOverlay")
