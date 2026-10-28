import numpy as np
from falcocv.core.config import ConfigManager
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QComboBox, QGroupBox, QDoubleSpinBox, QSizePolicy,
    QLineEdit, QFileDialog, QSplitter, QScrollArea, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, QTimer
from falcocv.ui.style import ALT_BG, BORDER, CAUTION, DANGER, INFO, SUCCESS, TEXT_MUTED, WHITE
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)
from falcocv.ui.widgets.viewer import ImageViewer

from .logic import GigeCameraLoader
class GigeCameraConfig(QWidget):
    def __init__(self, tool: GigeCameraLoader):
        super().__init__()
        self.tool = tool
        self.config_manager = ConfigManager()
        self._fps_counter = 0
        self._fps_timer = QTimer()
        self._fps_timer.timeout.connect(self._update_fps)

        # Main Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        
        # Content Layout (using Splitter)
        self.content_layout = QSplitter(Qt.Orientation.Horizontal)
        self.main_layout.addWidget(self.content_layout, 1)

        # --- LEFT PANEL: CONTROLS ---
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        controls_widget = QWidget()
        self.controls_layout = QVBoxLayout(controls_widget)
        self.controls_layout.setContentsMargins(8, 8, 8, 8)
        self.controls_layout.setSpacing(6)

        # --- Connection ---
        conn_group = QGroupBox("Camera Connection")
        conn_lyt = QVBoxLayout(conn_group)

        type_row = QHBoxLayout()
        type_row.addWidget(QLabel("Type:"))
        self.combo_type = QComboBox()
        self.combo_type.addItems(["Simulation", "Hardware"])
        self.combo_type.setCurrentText(self.tool.params["camera_type"])
        self.combo_type.currentTextChanged.connect(self._on_type_changed)
        type_row.addWidget(self.combo_type)
        conn_lyt.addLayout(type_row)
        
        cti_row = QHBoxLayout()
        cti_row.addWidget(QLabel("CTI File:"))
        self.edit_cti = QLineEdit(self.tool.params.get("cti_path", ""))
        self.edit_cti.textChanged.connect(self._on_cti_changed)
        btn_browse = QPushButton("...")
        btn_browse.setAutoDefault(False)
        btn_browse.setDefault(False)
        btn_browse.setFixedWidth(30)
        btn_browse.clicked.connect(self._on_browse_cti)
        cti_row.addWidget(self.edit_cti)
        cti_row.addWidget(btn_browse)
        conn_lyt.addLayout(cti_row)
        
        sn_row = QHBoxLayout()
        sn_row.addWidget(QLabel("Serial Number:"))
        self.edit_serial = QLineEdit(self.tool.params.get("camera_serial", ""))
        self.edit_serial.textChanged.connect(self._on_serial_changed)
        sn_row.addWidget(self.edit_serial)
        conn_lyt.addLayout(sn_row)

        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera:"))
        self.combo_cam = QComboBox()
        self.combo_cam.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.combo_cam.currentIndexChanged.connect(self._on_cam_index_changed)
        cam_row.addWidget(self.combo_cam)
        btn_discover = QPushButton("⟳ Discover")
        btn_discover.setFixedWidth(90)
        btn_discover.clicked.connect(self._on_discover)
        cam_row.addWidget(btn_discover)
        conn_lyt.addLayout(cam_row)

        btn_row = QHBoxLayout()
        self.btn_connect = QPushButton("● Connect")
        self.btn_connect.setStyleSheet(f"QPushButton {{ color: {SUCCESS}; font-weight: bold; }}")
        self.btn_connect.clicked.connect(self._on_connect)
        self.btn_disconnect = QPushButton("○ Disconnect")
        self.btn_disconnect.setStyleSheet(f"QPushButton {{ color: {DANGER}; font-weight: bold; }}")
        self.btn_disconnect.clicked.connect(self._on_disconnect)
        btn_row.addWidget(self.btn_connect)
        btn_row.addWidget(self.btn_disconnect)
        conn_lyt.addLayout(btn_row)

        self.label_status = QLabel("○  Disconnected")
        self.label_status.setStyleSheet(f"color: {TEXT_MUTED}; font-style: italic;")
        conn_lyt.addWidget(self.label_status)
        self.controls_layout.addWidget(conn_group)

        # --- Acquisition Settings ---
        acq_group = QGroupBox("Acquisition Settings")
        acq_lyt = QVBoxLayout(acq_group)

        trig_row = QHBoxLayout()
        trig_row.addWidget(QLabel("Trigger Mode:"))
        self.combo_trigger = QComboBox()
        self.combo_trigger.addItems(["SoftwareTrigger", "HardwareTrigger", "FreeRun"])
        self.combo_trigger.setCurrentText(self.tool.params["trigger_mode"])
        self.combo_trigger.currentTextChanged.connect(self._on_trigger_changed)
        trig_row.addWidget(self.combo_trigger)
        acq_lyt.addLayout(trig_row)

        fmt_row = QHBoxLayout()
        fmt_row.addWidget(QLabel("Output Format:"))
        self.combo_fmt = QComboBox()
        self.combo_fmt.addItems(["BGR", "Gray"])
        self.combo_fmt.setCurrentText(self.tool.params["output_format"])
        self.combo_fmt.currentTextChanged.connect(self._on_fmt_changed)
        fmt_row.addWidget(self.combo_fmt)
        acq_lyt.addLayout(fmt_row)

        exp_row = QHBoxLayout()
        exp_row.addWidget(QLabel("Exposure (µs):"))
        self.spin_exposure = QDoubleSpinBox()
        self.spin_exposure.setRange(10.0, 10_000_000.0)
        self.spin_exposure.setSingleStep(100.0)
        self.spin_exposure.setDecimals(0)
        self.spin_exposure.setValue(self.tool.params["exposure_us"])
        self.spin_exposure.valueChanged.connect(self._on_exposure_changed)
        # returnPressed is not standard on QDoubleSpinBox, but let's keep it if user added it or just ignore if it fails
        # Actually it's better to keep it if it was there. 
        if hasattr(self.spin_exposure, "returnPressed"):
            self.spin_exposure.returnPressed.connect(self._on_snap)
        exp_row.addWidget(self.spin_exposure)
        acq_lyt.addLayout(exp_row)
        self.controls_layout.addWidget(acq_group)

        # --- Custom Features Table ---
        custom_group = QGroupBox("Custom GenICam Features")
        custom_lyt = QVBoxLayout(custom_group)
        
        add_feat_row = QHBoxLayout()
        self.combo_features = QComboBox()
        self.combo_features.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        btn_add_feat = QPushButton("+ Add")
        btn_add_feat.setFixedWidth(60)
        btn_add_feat.clicked.connect(self._on_add_custom_feature)
        add_feat_row.addWidget(self.combo_features)
        add_feat_row.addWidget(btn_add_feat)
        custom_lyt.addLayout(add_feat_row)
        
        self.table_features = QTableWidget(0, 3)
        self.table_features.setHorizontalHeaderLabels(["Feature", "Value", ""])
        self.table_features.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table_features.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table_features.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_features.verticalHeader().setVisible(False)
        self.table_features.cellChanged.connect(self._on_table_cell_changed)
        custom_lyt.addWidget(self.table_features)
        
        btn_refresh_nodes = QPushButton("⟳ Refresh Node List")
        btn_refresh_nodes.clicked.connect(self._refresh_node_list)
        custom_lyt.addWidget(btn_refresh_nodes)
        
        self.controls_layout.addWidget(custom_group)
        self.controls_layout.addStretch()

        # Preview Controls at the bottom of left panel
        prev_group = QGroupBox("UI Preview Controls")
        prev_lyt = QHBoxLayout(prev_group)
        self.btn_snap = QPushButton("📷 Snap Frame")
        self.btn_snap.clicked.connect(self._on_snap)
        self.btn_live = QPushButton("▶ Live Video")
        self.btn_live.setCheckable(True)
        self.btn_live.clicked.connect(self._on_live_toggled)
        prev_lyt.addWidget(self.btn_snap)
        prev_lyt.addWidget(self.btn_live)
        self.controls_layout.addWidget(prev_group)

        scroll_area.setWidget(controls_widget)
        self.left_scroll = scroll_area
        self.content_layout.addWidget(scroll_area)

        # --- RIGHT PANEL: VIEWER ---
        viewer_widget = QWidget()
        self.viewer_layout = QVBoxLayout(viewer_widget)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)

        header_lyt = QHBoxLayout()
        header_lyt.addStretch()
        self.label_fps = QLabel("FPS: —")
        self.label_fps.setStyleSheet(f"color: {TEXT_MUTED}; font-weight: bold; margin-right: 10px;")
        header_lyt.addWidget(self.label_fps)
        self.viewer_layout.addLayout(header_lyt)

        self.viewer = ImageViewer()
        self.viewer_layout.addWidget(self.viewer)
        self.content_layout.addWidget(viewer_widget)

        configure_params_splitter(
            self.content_layout, self.left_scroll, viewer_widget, "GigeCamera"
        )

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        # Connect signals
        self.tool.data_changed.connect(self._on_data_changed)
        self.tool.connection_changed.connect(self._on_connection_changed)

        # Init UI states
        if self.tool.params["camera_type"] == "Simulation":
            self.combo_cam.addItem("Simulated Camera")
            self.combo_features.setEnabled(False)
            
        self._populate_custom_features_table()
        self._update_ui_state()
        self._update_status()

        # Disable autoDefault on all buttons so Enter key doesn't trigger them
        for btn in self.findChildren(QPushButton):
            btn.setAutoDefault(False)
            btn.setDefault(False)

    def _on_connection_changed(self, connected: bool):
        self._update_status()

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------
    def _update_ui_state(self):
        is_hw = (self.combo_type.currentText() == "Hardware")
        self.edit_cti.setEnabled(is_hw)
        self.edit_serial.setEnabled(is_hw)
        self.combo_features.setEnabled(self.tool.is_connected() and is_hw)

    def _on_type_changed(self, text: str):
        self.tool.params["camera_type"] = text
        self.combo_cam.clear()
        if text == "Simulation":
            self.combo_cam.addItem("Simulated Camera")
        self._update_ui_state()
            
    def _on_cti_changed(self, text):
        self.tool.params["cti_path"] = text
        # Persist the last valid CTI path in global config
        self.config_manager.set_gige_cti_path(text)

    def _on_serial_changed(self, text: str):
        self.tool.params["camera_serial"] = text
        
    def _on_browse_cti(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Select CTI File", self.tool.params.get("cti_path", ""), "CTI Files (*.cti *.dll)"
        )
        if path:
            self.edit_cti.setText(path)

    def _on_discover(self):
        self.label_status.setText("⟳  Discovering...")
        self.label_status.setStyleSheet(f"color: {CAUTION};")
        from PyQt6.QtWidgets import QApplication
        QApplication.processEvents()
        try:
            devices = self.tool.discover()
            self.combo_cam.clear()
            
            saved_serial = self.tool.params.get("camera_serial", "")
            match_idx = 0
            
            if not devices:
                self.combo_cam.addItem("No cameras found")
            else:
                for i, d in enumerate(devices):
                    label = f"{d['vendor']} {d['model']} [SN: {d['serial']}]"
                    self.combo_cam.addItem(label)
                    # Automatically select the one matching saved serial
                    if saved_serial and d['serial'] == saved_serial:
                        match_idx = i
                        
                self.combo_cam.setCurrentIndex(match_idx)
        except Exception as e:
            self.label_status.setText(f"⚠  {type(e).__name__}: {e}")
            self.label_status.setStyleSheet(f"color: {DANGER};")
            return
        self._update_status()

    def _on_cam_index_changed(self, idx: int):
        if idx >= 0 and getattr(self.tool, '_discovered', None) and idx < len(self.tool._discovered):
            serial = self.tool._discovered[idx]['serial']
            if self.edit_serial.text() != serial:
                self.edit_serial.setText(serial)

    def _on_connect(self):
        try:
            self.tool.connect()
            self._update_status()
            self._refresh_node_list()
        except Exception as e:
            self.label_status.setText(f"⚠  {type(e).__name__}: {e}")
            self.label_status.setStyleSheet(f"color: {DANGER};")

    def _on_disconnect(self):
        self._stop_freerun()
        self.btn_live.setChecked(False)
        self.tool.disconnect()
        self._update_status()
        self.combo_features.clear()
        self._update_ui_state()

    def _on_trigger_changed(self, text: str):
        self.tool.params["trigger_mode"] = text
        self.tool._apply_camera_params()

    def _on_fmt_changed(self, text: str):
        self.tool.params["output_format"] = text

    def _on_exposure_changed(self, value: float):
        self.tool.params["exposure_us"] = value
        self.tool._apply_camera_params()

    # --- Custom Features ---
    def _refresh_node_list(self):
        self.combo_features.clear()
        if not self.tool.is_connected() or self.tool._ia is None:
            return
        try:
            nm = self.tool._ia.remote_device.node_map
            nodes = [n for n in dir(nm) if not n.startswith('_')]
            self.combo_features.addItems(sorted(nodes))
        except Exception as e:
            print(f"Failed to fetch node list: {e}")
            
    def _populate_custom_features_table(self):
        self.table_features.blockSignals(True)
        self.table_features.setRowCount(0)
        customs = self.tool.params.get("custom_features", {})
        for feat, val in customs.items():
            self._add_feature_row(feat, val)
        self.table_features.blockSignals(False)

    def _add_feature_row(self, feat: str, val: str):
        row = self.table_features.rowCount()
        self.table_features.insertRow(row)
        
        item_feat = QTableWidgetItem(feat)
        item_feat.setFlags(item_feat.flags() & ~Qt.ItemFlag.ItemIsEditable) # Read-only
        self.table_features.setItem(row, 0, item_feat)
        
        item_val = QTableWidgetItem(str(val))
        self.table_features.setItem(row, 1, item_val)
        
        btn_del = QPushButton("X")
        btn_del.setFixedWidth(24)
        btn_del.setStyleSheet(f"color: {DANGER}; font-weight: bold;")
        btn_del.clicked.connect(lambda _, r=row: self._on_delete_feature(r))
        self.table_features.setCellWidget(row, 2, btn_del)
        
    def _on_add_custom_feature(self):
        feat = self.combo_features.currentText()
        if not feat: return
        
        # Don't add duplicates
        customs = self.tool.params.setdefault("custom_features", {})
        if feat in customs:
            return
            
        # Try to read current value
        val = ""
        if self.tool.is_connected() and self.tool._ia is not None:
            try:
                node = getattr(self.tool._ia.remote_device.node_map, feat)
                val = str(node.value)
            except Exception:
                pass
                
        customs[feat] = val
        self.table_features.blockSignals(True)
        self._add_feature_row(feat, val)
        self.table_features.blockSignals(False)

    def _on_table_cell_changed(self, row, col):
        if col != 1: return
        feat = self.table_features.item(row, 0).text()
        val = self.table_features.item(row, 1).text()
        
        self.tool.params.setdefault("custom_features", {})[feat] = val
        # Immediately apply to camera
        self.tool._apply_camera_params()
        
    def _on_delete_feature(self, row):
        feat = self.table_features.item(row, 0).text()
        if "custom_features" in self.tool.params and feat in self.tool.params["custom_features"]:
            del self.tool.params["custom_features"][feat]
        self._populate_custom_features_table()

    # --- Preview Controls ---
    def on_config_closed(self):
        if self.btn_live.isChecked():
            self.btn_live.setChecked(False)
            self._on_live_toggled(False)
        save_params_splitter(self.content_layout, "GigeCamera")

    def _on_snap(self):
        # Suspend Live Video if it's on
        if self.btn_live.isChecked():
            self.btn_live.setChecked(False)
            self._on_live_toggled(False)

        try:
            result = self.tool.snap()
            if result is None:
                self.label_status.setText("⚠ Timeout grabbing snap frame")
        except Exception as e:
            self.label_status.setText(f"⚠ Snap err: {type(e).__name__}: {e}")

    def _on_live_toggled(self, checked: bool):
        if checked:
            try:
                self.tool.start_preview()
            except Exception as e:
                self.label_status.setText(f"⚠ Live err: {type(e).__name__}: {e}")
                self.btn_live.setChecked(False)
                return
            self._start_freerun()
            self.btn_live.setText("⏹ Stop Live")
            self.btn_live.setStyleSheet(f"QPushButton {{ color: {DANGER}; font-weight: bold; }}")
        else:
            self._stop_freerun()
            self.tool.stop_preview()
            self.btn_live.setText("▶ Live Video")
            self.btn_live.setStyleSheet("")

    @safe_refresh
    def _on_data_changed(self):
        img = self.tool.outputs.get("output image")
        if img is not None:
            self.viewer.set_image(img)
            self._fps_counter += 1
        self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")

    # ------------------------------------------------------------------
    # FreeRun timer
    # ------------------------------------------------------------------
    def _start_freerun(self):
        if not hasattr(self, "_freerun_timer"):
            self._freerun_timer = QTimer()
            # In live UI mode, we just grab and emit manually to avoid triggering the whole Workflow via execute()
            self._freerun_timer.timeout.connect(self._do_live_grab)
        
        # Fast interval for UI rendering (e.g. 30fps = 33ms)
        self._freerun_timer.start(33)
        self._fps_timer.start(1000)

    def _do_live_grab(self):
        if not self.tool.is_connected():
            return
        frame = self.tool.grab_preview_frame()
        if frame is None:
            return # Ignore timeouts in continuous live mode
        self.tool._last_frame = frame
        self.tool.outputs["output image"] = frame
        # Only update the viewer, DO NOT emit data_changed, to avoid executing the whole pipeline on every live frame!
        self.viewer.set_image(frame)
        self._fps_counter += 1

    def _stop_freerun(self):
        if hasattr(self, "_freerun_timer"):
            self._freerun_timer.stop()
        self._fps_timer.stop()
        self.label_fps.setText("FPS: —")

    def _update_fps(self):
        self.label_fps.setText(f"FPS: {self._fps_counter}")
        self._fps_counter = 0

    # ------------------------------------------------------------------
    # Status
    # ------------------------------------------------------------------
    def _update_status(self):
        self._update_ui_state()
        if self.tool.is_connected():
            if self.tool.params["camera_type"] == "Simulation":
                model = "Fake Camera"
            else:
                try:
                    nm = self.tool._ia.remote_device.node_map
                    vendor = nm.DeviceVendorName.value if hasattr(nm, "DeviceVendorName") else "Camera"
                    model = nm.DeviceModelName.value if hasattr(nm, "DeviceModelName") else ""
                    model = f"{vendor} {model}"
                except Exception:
                    model = "Camera"
            self.label_status.setText(f"●  LIVE — {model}")
            self.label_status.setStyleSheet(f"color: {SUCCESS}; font-weight: bold;")
        else:
            self.label_status.setText("○  Disconnected")
            self.label_status.setStyleSheet(f"color: {TEXT_MUTED}; font-style: italic;")
