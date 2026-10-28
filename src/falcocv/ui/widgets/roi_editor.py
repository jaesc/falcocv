from PyQt6.QtWidgets import (QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                             QPushButton, QSpinBox, QDoubleSpinBox, QGroupBox,
                             QSizePolicy, QGridLayout)
from PyQt6.QtCore import pyqtSignal, QRectF, Qt
from falcocv.ui.style import CAUTION, WHITE


class RoiEditorWidget(QGroupBox):
    """
    Common widget for ROI management across all vision tools.
    Handles numerical input, interactive drawing, and resetting.
    """
    roi_changed = pyqtSignal(list)  # Emits [x, y, w, h, a]

    def __init__(self, title="Region of Interest", parent=None, param_key="roi"):
        super().__init__(title, parent)
        self.tool = None
        self.viewer = None
        self.param_key = param_key
        self.input_key = "input image"
        self._image_getter = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)

        # 1. Edit Button
        self.btn_edit = QPushButton("Select new ROI")
        self.btn_edit.setCheckable(True)
        self.btn_edit.toggled.connect(self._toggle_edit)
        self.btn_edit.setStyleSheet(f"""
            QPushButton:checked {{
                background-color: {CAUTION};
                color: {WHITE};
                font-weight: bold;
            }}
        """)
        layout.addWidget(self.btn_edit)

        # 2. Numerical Inputs (compact 2-row grid)
        grid = QGridLayout()
        grid.setSpacing(2)
        grid.setContentsMargins(0, 0, 0, 0)
        self.spins = {}

        def _add_spin(label, col, row):
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
                spin.setRange(-10000, 10000)
                spin.setDecimals(1)
            spin.setButtonSymbols(QDoubleSpinBox.ButtonSymbols.NoButtons)
            spin.setMinimumWidth(50)
            spin.setMaximumWidth(65)
            spin.setAlignment(Qt.AlignmentFlag.AlignRight)
            spin.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Fixed)
            spin.setStyleSheet("font-size: 9px;")
            spin.valueChanged.connect(self._on_spin_changed)
            grid.addWidget(spin, row * 2 + 1, col)
            self.spins[["X", "Y", "W", "H", "A"].index(label)] = spin

        _add_spin("X", 0, 0)
        _add_spin("Y", 1, 0)
        _add_spin("A", 2, 0)
        _add_spin("W", 0, 1)
        _add_spin("H", 1, 1)

        for c in range(3):
            grid.setColumnMinimumWidth(c, 50)

        layout.addLayout(grid)

        # 3. Reset Button
        self.btn_reset = QPushButton("Reset to Full Image")
        self.btn_reset.clicked.connect(self.reset_roi)
        layout.addWidget(self.btn_reset)

    def _get_image_data_source(self):
        if self._image_getter is not None:
            return self._image_getter()
        if self.tool:
            return self.tool.inputs.get(self.input_key)
        return None

    def _get_image_data(self):
        image = self._get_image_data_source()
        return image.data if hasattr(image, "data") else image

    def _get_image_size(self):
        """Return (width, height) of the current image, or (0,0) if unavailable."""
        if not self.tool:
            return 0, 0
        data = self._get_image_data()
        if data is None:
            return 0, 0
        if hasattr(data, 'shape'):
            h, w = data.shape[0], data.shape[1]
            return w, h
        return 0, 0

    def _update_ranges(self):
        """Adjust spinbox ranges based on image size and ROI type."""
        width, height = self._get_image_size()
        if width <= 0 or height <= 0:
            return

        x_min, x_max = -width, width
        y_min, y_max = -height, height
        w_min, w_max = 0, width * 2
        h_min, h_max = 0, height * 2

        if 0 in self.spins:
            self.spins[0].setRange(x_min, x_max)
        if 1 in self.spins:
            self.spins[1].setRange(y_min, y_max)
        if 2 in self.spins:
            self.spins[2].setRange(w_min, w_max)
        if 3 in self.spins:
            self.spins[3].setRange(h_min, h_max)

    def set_tool(self, tool, viewer):
        """Connect the widget to a tool and its viewer."""
        self.tool = tool
        self.viewer = viewer
        if viewer is not None:
            viewer.roi_changed.connect(self._on_viewer_roi_changed)
            viewer.roi_drawing_finished.connect(self._on_drawing_finished)
        self.update_from_params()

    def set_input_key(self, key: str):
        """Set the input key used to find the image for fixture conversion."""
        self.input_key = key

    def set_image_getter(self, getter):
        """Use a callable returning the image (VisionImage or ndarray) for this ROI."""
        self._image_getter = getter

    def update_from_params(self):
        """Update spinboxes from tool params."""
        if not self.tool:
            return

        # Block all signals during update
        for spin in self.spins.values():
            spin.blockSignals(True)

        self._update_ranges()

        roi_data = self.tool.params.get(self.param_key)
        if not isinstance(roi_data, list):
            roi_vals = [0.0, 0.0, 0.0, 0.0, 0.0]
        elif len(roi_data) == 4:
            roi_vals = [float(v) for v in roi_data] + [0.0]
        elif len(roi_data) == 5:
            roi_vals = [float(v) for v in roi_data]
        else:
            roi_vals = [0.0, 0.0, 0.0, 0.0, 0.0]

        for i, val in enumerate(roi_vals):
            if i in self.spins:
                self.spins[i].setValue(val)

        for spin in self.spins.values():
            spin.blockSignals(False)

    def _toggle_edit(self, checked):
        if checked:
            self.btn_edit.setText("DRAWING ROI - Drag in viewer")
        else:
            self.btn_edit.setText("Select new ROI")

        if self.viewer:
            self.viewer.set_drawing_new_roi(checked)

    def _on_spin_changed(self):
        if not self.tool:
            return
        roi_vals = [self.spins[i].value() for i in range(len(self.spins))]
        self.tool.params[self.param_key] = roi_vals
        self.roi_changed.emit(roi_vals)
        self.tool.data_changed.emit()

    def _on_viewer_roi_changed(self, roi_data):
        if not self.tool:
            return

        # Only react if this ROI is the one currently displayed in the viewer
        active_key = getattr(self.tool, '_active_roi_key', None)
        if active_key is not None and active_key != self.param_key:
            return

        # roi_data can be QRectF (legacy) or list [x, y, w, h, a]
        if isinstance(roi_data, QRectF):
            roi_abs = [roi_data.x(), roi_data.y(), roi_data.width(), roi_data.height(), 0.0]
        else:
            roi_abs = [float(x) for x in roi_data]

        # Convert absolute viewer ROI to fixture-relative coordinates
        from falcocv.core.vision_image import VisionImage
        img_input = self._get_image_data_source() if hasattr(self.tool, 'inputs') else None
        if isinstance(img_input, VisionImage):
            v_img = img_input
        else:
            v_img = VisionImage.from_array(img_input) if img_input is not None else None
        if v_img is not None:
            roi_fixture = v_img.image_to_fixture(roi_abs)
        else:
            roi_fixture = roi_abs

        self.tool.params[self.param_key] = roi_fixture

        # Update spinboxes without triggering signals
        for spin in self.spins.values():
            spin.blockSignals(True)
        for i, val in enumerate(roi_fixture):
            if i in self.spins:
                self.spins[i].setValue(float(val))
        for spin in self.spins.values():
            spin.blockSignals(False)

        self.roi_changed.emit(roi_fixture)
        self.tool.data_changed.emit()

    def _on_drawing_finished(self):
        self.btn_edit.blockSignals(True)
        self.btn_edit.setChecked(False)
        self.btn_edit.blockSignals(False)
        self._toggle_edit(False)

    def reset_roi(self):
        if not self.tool:
            return
        self.tool.params[self.param_key] = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.update_from_params()
        self.roi_changed.emit([0.0, 0.0, 0.0, 0.0, 0.0])
        self.tool.data_changed.emit()
