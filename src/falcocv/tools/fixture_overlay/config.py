import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QComboBox,
    QPushButton, QSplitter, QGroupBox
)
from PyQt6.QtCore import Qt
from falcocv.core.vision_image import VisionImage
from falcocv.ui.widgets.viewer import ImageViewer
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    add_fixture_axis_row,
    add_record_header,
    make_record_selector,
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)
from .logic import _enumerate_elements, FixtureOverlay


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

        # The tool always acts as a fixture, so the axis selection is always
        # active (unlike the match tools, which gate it behind a toggle).
        self.fixture_axes = add_fixture_axis_row(group_lyt, self.tool)

        self.controls_layout.addWidget(group)
        self.controls_layout.addStretch()

        # --- RIGHT PANEL: VIEWER ---
        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)

        self.record_selector = make_record_selector()
        self.record_selector.addItem("Input Image", "input image")
        self.record_selector.addItem("Output Image", "output image")
        self.record_selector.setCurrentIndex(0)
        self.record_selector.currentIndexChanged.connect(self._on_record_changed)
        add_record_header(self.viewer_layout, self.record_selector, tool=self.tool)

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
