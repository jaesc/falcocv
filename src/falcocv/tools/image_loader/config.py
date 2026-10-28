import os
import cv2
from falcocv.core.tool_base import BaseTool
from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QPushButton, QFileDialog, QSplitter,
                             QLabel, QHBoxLayout, QComboBox, QGroupBox, QRadioButton, QSpinBox)
from PyQt6.QtCore import Qt
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)
from falcocv.ui.widgets.viewer import ImageViewer

from .logic import ImageLoader
class ImageLoaderConfig(QWidget):
    def __init__(self, tool: ImageLoader):
        super().__init__()
        self.tool = tool

        # Main Layout
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(5, 5, 5, 5)
        
        # Content Layout
        self.content_layout = QSplitter(Qt.Orientation.Horizontal)
        
        # --- LEFT SIDE: CONTROLS ---
        self.left_panel = QWidget()
        self.controls_layout = QVBoxLayout(self.left_panel)
        self.controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # Mode Selection
        mode_group = QGroupBox("Mode")
        mode_lyt = QVBoxLayout(mode_group)
        self.radio_file = QRadioButton("Single File")
        self.radio_folder = QRadioButton("Folder Stack")
        
        if tool.params["mode"] == "File":
            self.radio_file.setChecked(True)
        else:
            self.radio_folder.setChecked(True)
            
        self.radio_file.toggled.connect(self._on_mode_changed)
        mode_lyt.addWidget(self.radio_file)
        mode_lyt.addWidget(self.radio_folder)
        self.controls_layout.addWidget(mode_group)
        
        # Path and Selection
        self.label_path = QLabel()
        self.label_path.setWordWrap(True)
        self.controls_layout.addWidget(self.label_path)
        
        self.btn_select = QPushButton()
        self.btn_select.clicked.connect(self._on_select_path)
        self.controls_layout.addWidget(self.btn_select)
        
        # Folder specific controls
        self.folder_controls = QWidget()
        fc_lyt = QVBoxLayout(self.folder_controls)
        fc_lyt.setContentsMargins(0, 0, 0, 0)
        
        fc_lyt.addWidget(QLabel("Sort By:"))
        self.combo_sort = QComboBox()
        self.combo_sort.addItems(["Name", "Date"])
        self.combo_sort.setCurrentText(tool.params["sort_by"])
        self.combo_sort.currentTextChanged.connect(self._on_sort_changed)
        fc_lyt.addWidget(self.combo_sort)
        
        fc_lyt.addWidget(QLabel("Loop Delay (ms):"))
        self.spin_delay = QSpinBox()
        self.spin_delay.setRange(0, 60000)
        self.spin_delay.setValue(tool.params["delay"])
        self.spin_delay.valueChanged.connect(self._on_delay_changed)
        fc_lyt.addWidget(self.spin_delay)
        
        self.label_count = QLabel()
        fc_lyt.addWidget(self.label_count)
        
        self.controls_layout.addWidget(self.folder_controls)
        self.controls_layout.addStretch()
        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)

        # --- RIGHT SIDE: VIEWER ---
        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        header_lyt = QHBoxLayout()

        self.record_selector = QComboBox()
        self.record_selector.addItem(f"{tool.name}.output_image")
        header_lyt.addWidget(self.record_selector)

        self.viewer_layout.addLayout(header_lyt)

        self.viewer = ImageViewer()
        self.viewer_layout.addWidget(self.viewer)
        self.content_layout.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "ImageLoader"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)
        
        self.tool.data_changed.connect(self.refresh_ui)
        self.update_ui_state()

    def _on_mode_changed(self):
        mode = "File" if self.radio_file.isChecked() else "Folder"
        self.tool.params["mode"] = mode
        self.update_ui_state()
        self.tool.mark_dirty()

    def _on_sort_changed(self, text):
        self.tool.params["sort_by"] = text
        if self.tool.params["mode"] == "Folder":
            self.tool._refresh_folder()
            self.update_ui_state()
        self.tool.mark_dirty()

    def _on_delay_changed(self, value):
        self.tool.params["delay"] = value

    def update_ui_state(self):
        mode = self.tool.params["mode"]
        if mode == "File":
            self.folder_controls.hide()
            self.btn_select.setText("Select Image File")
            self.label_path.setText(f"File: {self.tool.params['file_path'] or 'None'}")
        else:
            self.folder_controls.show()
            self.btn_select.setText("Select Directory")
            self.label_path.setText(f"Folder: {self.tool.params['folder_path'] or 'None'}")
            total = len(self.tool._image_files)
            current = self.tool.params["current_index"] + 1 if total > 0 else 0
            self.label_count.setText(f"Images: {current} / {total}")

    def _on_select_path(self):
        if self.radio_file.isChecked():
            path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.jpg *.jpeg *.bmp)")
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

    @safe_refresh
    def refresh_ui(self):
        img = self.tool.get_output("output image")
        if img is not None:
            self.viewer.set_image(img, auto_fit=True)
        self.label_time.setText(f"{self.tool.last_execution_time_ms:.1f} ms")
            
    def showEvent(self, event):
        super().showEvent(event)
        self.update_ui_state()
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(100, self.refresh_ui)

    def on_config_closed(self):
        save_params_splitter(self.content_layout, "ImageLoader")
