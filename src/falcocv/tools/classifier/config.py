"""Classifier — multi-class classification via embedding similarity + k-NN.

Record mode captures ROI images assigned to the currently active class tab.
Training: backbone ONNX → triplet loss projection head → k-NN memory bank.
"""

__all__ = ["Classifier"]

import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton,
    QLabel, QSlider, QComboBox, QMessageBox,
    QGroupBox, QCheckBox, QTabWidget, QSpinBox, QProgressBar,
    QListWidget, QListWidgetItem, QDialog, QListView,
    QGraphicsView, QGraphicsScene, QLineEdit, QMenu,
)
from PyQt6.QtCore import Qt, QTimer, QSize, QThread
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter

from falcocv.core.tool_base import BaseTool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi, create_roi_only_output
from falcocv.tools.classifier.train_worker import TrainWorker
from falcocv.ui.widgets.viewer import ImageViewer
from falcocv.ui.style import ALT_BG, TEXT, SUCCESS, CAUTION, WHITE, BORDER
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)

_DEFAULT_THRESHOLD = 0.65

from .logic import _b64_to_img, _img_to_b64, Classifier
class ClassifierConfig(QWidget):
    def __init__(self, tool: Classifier):
        super().__init__()
        self.tool = tool
        self._suppress_execute = True

        self.setWindowTitle("Classifier Config")
        self.setMinimumSize(600, 400)

        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)

        self.content_layout = QSplitter(Qt.Orientation.Horizontal)
        self.content_layout.setChildrenCollapsible(False)

        self.viewer = ImageViewer()
        self.viewer.roi_changed.connect(self._on_roi_changed)

        self.left_panel = QWidget()
        left_outer = QVBoxLayout(self.left_panel)
        left_outer.setContentsMargins(8, 8, 8, 8)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_main_tab(), "Main")
        self.tabs.addTab(self._build_classes_tab(), "Classes")
        left_outer.addWidget(self.tabs)
        self.left_scroll = wrap_params_scroll(self.left_panel)
        self.content_layout.addWidget(self.left_scroll)

        self.right_panel = QWidget()
        self.viewer_layout = QVBoxLayout(self.right_panel)
        self.viewer_layout.setContentsMargins(0, 0, 0, 0)
        self.record_selector = QComboBox()
        self.record_selector.addItem("input image", "input image")
        self.record_selector.addItem("output image", "output image")
        self.record_selector.currentIndexChanged.connect(self.on_record_changed)
        self.viewer_layout.addWidget(self.record_selector)
        self.viewer_layout.addWidget(self.viewer)
        self.content_layout.addWidget(self.right_panel)

        configure_params_splitter(
            self.content_layout, self.left_scroll, self.right_panel, "Classifier"
        )
        self.main_layout.addWidget(self.content_layout, 1)

        from falcocv.ui.style import disabled_css
        self._disabled_css = disabled_css()

        # Footer
        self.footer, self.label_time = create_execution_footer(self)
        self.main_layout.addWidget(self.footer)

        self.tool.data_changed.connect(self.refresh_ui)
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))
        self._suppress_execute = False
        self._update_classes_ui()

    # ── Main tab ─────────────────────────────────────────────────────────

    def _build_main_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        from falcocv.ui.widgets.roi_editor import RoiEditorWidget
        self.roi_editor = RoiEditorWidget(param_key="roi")
        layout.addWidget(self.roi_editor)
        self.roi_editor.set_tool(self.tool, self.viewer)

        detect_group = QGroupBox("Classifier")
        detect_lyt = QVBoxLayout(detect_group)

        self.lbl_threshold = QLabel(f"Threshold: {self.tool.params['threshold']:.2f}")
        detect_lyt.addWidget(self.lbl_threshold)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(50, 95)
        self.threshold_slider.setValue(int(self.tool.params["threshold"] * 100))
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        detect_lyt.addWidget(self.threshold_slider)

        exp_lyt = QHBoxLayout()
        exp_lyt.addWidget(QLabel("Expected class:"))
        self.cmb_expected = QComboBox()
        self.cmb_expected.addItem("(any)", "")
        self.cmb_expected.currentIndexChanged.connect(self._on_expected_changed)
        exp_lyt.addWidget(self.cmb_expected)
        detect_lyt.addLayout(exp_lyt)

        k_lyt = QHBoxLayout()
        k_lyt.addWidget(QLabel("k-NN neighbors:"))
        self.spin_k = QSpinBox()
        self.spin_k.setRange(1, 11)
        self.spin_k.setValue(self.tool.params.get("k_nn", 3))
        self.spin_k.valueChanged.connect(self._on_k_changed)
        k_lyt.addWidget(self.spin_k)
        detect_lyt.addLayout(k_lyt)

        self.cmb_output = QComboBox()
        self.cmb_output.addItems(["Full Image", "ROI Only"])
        self.cmb_output.setCurrentText(
            "ROI Only" if self.tool.params["output_mode"] == "ROI Only" else "Full Image"
        )
        self.cmb_output.currentTextChanged.connect(self._on_output_mode)
        detect_lyt.addWidget(QLabel("Output Mode:"))
        detect_lyt.addWidget(self.cmb_output)

        layout.addWidget(detect_group)

        vis_group = QGroupBox("Visibility")
        vis_lyt = QVBoxLayout(vis_group)
        self.chk_overlays = QCheckBox("Show Overlays")
        self.chk_overlays.setChecked(True)
        self.chk_overlays.toggled.connect(self.refresh_ui)
        vis_lyt.addWidget(self.chk_overlays)
        self.chk_labels = QCheckBox("Show Labels")
        self.chk_labels.setChecked(self.tool.params.get("show_labels", True))
        self.chk_labels.toggled.connect(self._on_labels_toggled)
        vis_lyt.addWidget(self.chk_labels)
        self.chk_roi_overlay = QCheckBox("Show Region")
        self.chk_roi_overlay.setChecked(self.tool.params.get("show_roi_overlay", True))
        self.chk_roi_overlay.toggled.connect(self._on_roi_overlay_toggled)
        vis_lyt.addWidget(self.chk_roi_overlay)
        self.chk_copy_in = QCheckBox("Copy overlays to input image")
        self.chk_copy_in.setChecked(self.tool.params.get("copy_overlays_to_input", True))
        self.chk_copy_in.toggled.connect(lambda v: self._set_param("copy_overlays_to_input", v))
        vis_lyt.addWidget(self.chk_copy_in)
        layout.addWidget(vis_group)

        layout.addStretch()
        return tab

    # ── Classes tab (replaces Training tab) ──────────────────────────────

    def _build_classes_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        # Add new class
        add_lyt = QHBoxLayout()
        add_lyt.addWidget(QLabel("New class:"))
        self.txt_class_name = QLineEdit()
        self.txt_class_name.setPlaceholderText("Enter class name...")
        add_lyt.addWidget(self.txt_class_name)
        self.btn_add_class = QPushButton("Add Class")
        self.btn_add_class.clicked.connect(self._on_add_class)
        add_lyt.addWidget(self.btn_add_class)
        layout.addLayout(add_lyt)

        # Class tabs
        self.class_tabs = QTabWidget()
        self.class_tabs.setDocumentMode(True)
        self.class_tabs.currentChanged.connect(self._on_class_tab_changed)
        layout.addWidget(self.class_tabs, 1)

        # Train / Untrain
        trn_lyt = QHBoxLayout()
        self.btn_train = QPushButton("Train")
        self.btn_train.clicked.connect(self._on_train)
        trn_lyt.addWidget(self.btn_train)
        self.btn_untrain = QPushButton("Untrain")
        self.btn_untrain.clicked.connect(self._on_untrain)
        trn_lyt.addWidget(self.btn_untrain)
        layout.addLayout(trn_lyt)

        # Status
        self.lbl_train_status = QLabel("Not trained")
        self.lbl_train_status.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(self.lbl_train_status)

        return tab

    def _build_class_tab(self, class_id: str, label: str):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        # Header
        hdr_lyt = QHBoxLayout()
        lbl_name = QLabel(f"Class: {label}")
        lbl_name.setStyleSheet(f"color: {WHITE}; font-weight: bold; font-size: 14px;")
        hdr_lyt.addWidget(lbl_name)
        hdr_lyt.addStretch()
        btn_remove = QPushButton("Remove Class")
        btn_remove.setStyleSheet(f"color: {CAUTION};")
        btn_remove.clicked.connect(lambda: self._on_remove_class(class_id))
        hdr_lyt.addWidget(btn_remove)
        btn_remove_pics = QPushButton("Remove Pictures")
        btn_remove_pics.setStyleSheet(f"color: {CAUTION};")
        btn_remove_pics.clicked.connect(lambda: self._on_remove_pictures(class_id))
        hdr_lyt.addWidget(btn_remove_pics)
        layout.addLayout(hdr_lyt)

        # Recording controls
        rec_lyt = QHBoxLayout()
        btn_capture = QPushButton("Capture")
        btn_capture.clicked.connect(lambda: self._on_capture_image(class_id))
        rec_lyt.addWidget(btn_capture)
        rec_lyt.addSpacing(20)
        btn_start = QPushButton("Start Recording")
        btn_start.clicked.connect(lambda: self._on_start_recording(class_id, btn_start, btn_stop))
        rec_lyt.addWidget(btn_start)
        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(lambda: self._on_stop_recording(btn_start, btn_stop))
        btn_stop.setEnabled(False)
        rec_lyt.addWidget(btn_stop)
        layout.addLayout(rec_lyt)

        # Max captures (per-class)
        max_lyt = QHBoxLayout()
        max_lyt.addWidget(QLabel("Max captures:"))
        spin_max = QSpinBox()
        spin_max.setRange(5, 999)
        spin_max.setValue(self.tool.params.get("max_samples_per_class", 50))
        spin_max.valueChanged.connect(lambda v: self._on_max_changed(v))
        max_lyt.addWidget(spin_max)
        layout.addLayout(max_lyt)

        # Progress
        progress = QProgressBar()
        progress.setRange(0, spin_max.value())
        progress.setValue(0)
        layout.addWidget(progress)

        lbl_count = QLabel(f"Captured: 0 / {spin_max.value()}")
        layout.addWidget(lbl_count)

        # Thumbnail grid
        thumb_list = QListWidget()
        thumb_list.setViewMode(QListView.ViewMode.IconMode)
        thumb_list.setIconSize(QSize(64, 64))
        thumb_list.setGridSize(QSize(72, 72))
        thumb_list.setMovement(QListView.Movement.Static)
        thumb_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        thumb_list.itemDoubleClicked.connect(lambda item: self._on_thumb_clicked(item, class_id))
        thumb_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        thumb_list.customContextMenuRequested.connect(
            lambda pos: self._on_thumb_context_menu(pos, class_id))
        layout.addWidget(thumb_list, 1)

        # Store references for UI updates
        tab.btn_capture = btn_capture
        tab.btn_start = btn_start
        tab.btn_stop = btn_stop
        tab.spin_max = spin_max
        tab.progress = progress
        tab.lbl_count = lbl_count
        tab.thumb_list = thumb_list

        # Initial refresh
        self._refresh_class_tab(tab, class_id)
        return tab

    def _refresh_class_tab(self, tab, class_id):
        imgs = self.tool.get_class_images(class_id)
        max_s = tab.spin_max.value() if hasattr(tab, "spin_max") else self.tool.params.get("max_samples_per_class", 50)
        tab.progress.setMaximum(max(max_s, 1))
        is_recording = self.tool.recording and self.tool._active_class == class_id
        tab.progress.setValue(self.tool._recording_count if is_recording else 0)
        tab.lbl_count.setText(f"Captured: {len(imgs)}")
        tab.thumb_list.clear()
        for idx, img in enumerate(imgs):
            h, w = img.shape[:2]
            scale = min(64 / max(h, w, 1), 1.0)
            small = cv2.resize(img, (int(w * scale), int(h * scale)))
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
            qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                          rgb.strides[0], QImage.Format.Format_RGB888)
            item = QListWidgetItem(QIcon(QPixmap.fromImage(qimg)), "")
            item.setData(Qt.ItemDataRole.UserRole, idx)
            tab.thumb_list.addItem(item)
        # Recording button state
        if self.tool.recording and self.tool._active_class == class_id:
            tab.btn_start.setStyleSheet(f"""
                QPushButton {{ background-color: {CAUTION}; color: {WHITE}; font-weight: bold; }}
            """)
            tab.btn_start.setText("Recording...")
            tab.btn_start.setEnabled(False)
            tab.btn_stop.setEnabled(True)
        else:
            tab.btn_start.setStyleSheet(f"background-color: {ALT_BG}; color: {TEXT};")
            tab.btn_start.setText("Start Recording")
            tab.btn_start.setEnabled(True)
            tab.btn_stop.setEnabled(False)

    # ── Class tab callbacks ──────────────────────────────────────────────

    def _on_add_class(self):
        name = self.txt_class_name.text().strip()
        if not name:
            QMessageBox.warning(self, "Add Class", "Enter a class name.")
            return
        cid = self.tool.add_class(name)
        tab = self._build_class_tab(cid, name)
        tab.setProperty("class_id", cid)
        idx = self.class_tabs.addTab(tab, name)
        self.class_tabs.setCurrentIndex(idx)
        self.txt_class_name.clear()
        self._update_classes_ui()
        self.tool.data_changed.emit()

    def _on_remove_class(self, class_id):
        label = self.tool._class_labels.get(class_id, class_id)
        reply = QMessageBox.question(
            self, "Remove Class",
            f"Remove class '{label}' and all its captured images?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        # Find and remove the tab
        for i in range(self.class_tabs.count()):
            if self.class_tabs.widget(i).property("class_id") == class_id:
                self.class_tabs.removeTab(i)
                break
        self.tool.remove_class(class_id)
        self._update_classes_ui()
        self.tool.data_changed.emit()

    def _on_remove_pictures(self, class_id):
        label = self.tool._class_labels.get(class_id, class_id)
        reply = QMessageBox.question(
            self, "Remove Pictures",
            f"Remove all captured images for class '{label}'?\nThe class itself will be kept.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        if class_id in self.tool._class_data:
            self.tool._class_data[class_id].clear()
        for i in range(self.class_tabs.count()):
            tab = self.class_tabs.widget(i)
            if tab.property("class_id") == class_id:
                self._refresh_class_tab(tab, class_id)
                break
        self.tool.data_changed.emit()

    def _on_class_tab_changed(self, idx):
        pass  # No action needed; recording targets the tab the user started

    def _on_start_recording(self, class_id, btn_start, btn_stop):
        self.tool.start_recording(class_id)
        btn_start.setStyleSheet(f"""
            QPushButton {{ background-color: {CAUTION}; color: {WHITE}; font-weight: bold; }}
        """)
        btn_start.setText("Recording...")
        btn_start.setEnabled(False)
        btn_stop.setEnabled(True)

    def _on_stop_recording(self, btn_start, btn_stop):
        self.tool.stop_recording()
        btn_start.setStyleSheet(f"background-color: {ALT_BG}; color: {TEXT};")
        btn_start.setText("Start Recording")
        btn_start.setEnabled(True)
        btn_stop.setEnabled(False)

    def _on_capture_image(self, class_id: str):
        img_input = self.tool.inputs.get("input image")
        if img_input is None:
            QMessageBox.warning(self, "Capture", "No input image available.")
            return

        img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)
        roi = extract_roi(img, self.tool.params["roi"])
        roi_src = roi["roi_src"]

        if class_id not in self.tool._class_data:
            self.tool._class_data[class_id] = []
        self.tool._class_data[class_id].append(roi_src.copy())

        for i in range(self.class_tabs.count()):
            tab = self.class_tabs.widget(i)
            if tab.property("class_id") == class_id:
                self._refresh_class_tab(tab, class_id)
                break

        self.tool.data_changed.emit()

    def _on_max_changed(self, value):
        self.tool.params["max_samples_per_class"] = value
        for i in range(self.class_tabs.count()):
            tab = self.class_tabs.widget(i)
            if hasattr(tab, "spin_max"):
                tab.spin_max.blockSignals(True)
                tab.spin_max.setValue(value)
                tab.spin_max.blockSignals(False)
            if hasattr(tab, "progress"):
                tab.progress.setMaximum(max(value, 1))
            if hasattr(tab, "lbl_count"):
                imgs = self.tool._class_data.get(tab.property("class_id"), [])
                tab.lbl_count.setText(f"Captured: {len(imgs)}")
        self.tool.mark_dirty()

    def _on_train(self):
        if self.tool.num_classes() < 2:
            QMessageBox.warning(self, "Training", "Need at least 2 classes with >=5 images each.")
            return
        has_valid = any(len(imgs) >= 5 for imgs in self.tool._class_data.values())
        if not has_valid:
            QMessageBox.warning(self, "Training", "Need >=5 images per class.")
            return

        self.btn_train.setEnabled(False)
        self.btn_train.setText("Training...")
        self.btn_train.setStyleSheet(f"""
            QPushButton {{ background-color: {CAUTION}; color: {WHITE}; font-weight: bold; }}
        """)
        self.btn_train.repaint()
        self.lbl_train_status.setText("Starting training...")

        self._train_thread = QThread(self)
        self._train_worker = TrainWorker(
            class_data={cid: list(imgs) for cid, imgs in self.tool._class_data.items()},
            class_labels=dict(self.tool._class_labels),
            params=dict(self.tool.params),
            k_nn=self.tool.params.get("k_nn", 3),
        )
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.finished.connect(self._on_train_complete)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.finished.connect(self._train_worker.deleteLater)
        self._train_thread.start()

    def _on_train_progress(self, pct: int, msg: str):
        self.lbl_train_status.setText(msg)

    def _on_train_complete(self, msg: str):
        engine = self._train_worker.result_engine
        if engine is None:
            self._on_train_error("Training produced no engine")
            return

        self.tool._inference = engine
        self.tool._trained = True
        self.tool.params["is_trained"] = True
        self.tool.params["projection_b64"] = engine.projection.to_b64()
        self.tool.params["memory_bank_data"] = engine.memory_bank.to_dict()
        self.tool.params["thresholds"] = engine.thresholds

        self.btn_train.setEnabled(True)
        self.btn_train.setText("Train")
        self.btn_train.setStyleSheet("")

        self._update_classes_ui()
        self.lbl_train_status.setText(msg)
        self.lbl_train_status.setStyleSheet(f"color: {SUCCESS};")
        self.tool.data_changed.emit()
        self._cleanup_train_thread()

    def _on_train_error(self, msg: str):
        self.btn_train.setEnabled(True)
        self.btn_train.setText("Train")
        self.btn_train.setStyleSheet("")
        self.lbl_train_status.setText(f"Training failed: {msg}")
        self.lbl_train_status.setStyleSheet(f"color: {CAUTION};")
        QMessageBox.critical(self, "Training Error", msg)
        self._cleanup_train_thread()

    def _cleanup_train_thread(self):
        if hasattr(self, '_train_thread') and self._train_thread is not None:
            if self._train_thread.isRunning():
                self._train_thread.quit()
                self._train_thread.wait(5000)
            self._train_thread = None
            self._train_worker = None

    def _on_untrain(self):
        self.tool.clear_training()
        self._update_classes_ui()
        self.lbl_train_status.setText("Not trained")
        self.lbl_train_status.setStyleSheet(f"color: {TEXT};")
        self.tool.data_changed.emit()

    def _update_classes_ui(self):
        # Rebuild class tabs from tool data
        self.class_tabs.blockSignals(True)
        self.class_tabs.clear()
        for cid, label in self.tool._class_labels.items():
            tab = self._build_class_tab(cid, label)
            tab.setProperty("class_id", cid)
            self.class_tabs.addTab(tab, label)
        self.class_tabs.blockSignals(False)

        # Update expected class combo
        current_exp = self.tool.params.get("expected_class", "")
        self.cmb_expected.blockSignals(True)
        self.cmb_expected.clear()
        self.cmb_expected.addItem("(any)", "")
        for cid, label in self.tool._class_labels.items():
            self.cmb_expected.addItem(label, label)
        idx = self.cmb_expected.findData(current_exp)
        if idx >= 0:
            self.cmb_expected.setCurrentIndex(idx)
        self.cmb_expected.blockSignals(False)

        # Update status
        total = sum(len(imgs) for imgs in self.tool._class_data.values())
        n_cls = len(self.tool._class_data)
        if self.tool.is_trained:
            self.lbl_train_status.setText(f"Trained: {n_cls} classes, {total} images")
            self.lbl_train_status.setStyleSheet(f"color: {SUCCESS};")
        elif n_cls > 0:
            self.lbl_train_status.setText(f"{n_cls} classes, {total} images — not trained")
            self.lbl_train_status.setStyleSheet(f"color: {TEXT};")

    def _on_thumb_context_menu(self, pos, class_id):
        tab = self.class_tabs.currentWidget()
        if not hasattr(tab, "thumb_list"):
            return
        item = tab.thumb_list.itemAt(pos)
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        menu = QMenu(self)
        menu.addAction("Delete Picture")
        action = menu.exec(tab.thumb_list.mapToGlobal(pos))
        if action and action.text() == "Delete Picture":
            imgs = self.tool._class_data.get(class_id, [])
            if 0 <= idx < len(imgs):
                del imgs[idx]
            for i in range(self.class_tabs.count()):
                t = self.class_tabs.widget(i)
                if t.property("class_id") == class_id:
                    self._refresh_class_tab(t, class_id)
                    break
            self.tool.data_changed.emit()

    def _on_thumb_clicked(self, item, class_id):
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        imgs = self.tool._class_data.get(class_id, [])
        if idx >= len(imgs):
            return
        img = imgs[idx]
        dlg = QDialog(self, Qt.WindowType.Window)
        dlg.setWindowTitle(f"Image #{idx} — {self.tool._class_labels.get(class_id, class_id)}")
        dlg.resize(600, 450)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        qimg = QImage(rgb.data, w, h, rgb.strides[0], QImage.Format.Format_RGB888)
        pix = QPixmap.fromImage(qimg)
        scene = QGraphicsScene()
        scene.addPixmap(pix)
        view = QGraphicsView(scene)
        view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        view.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        def _zoom(e):
            f = 1.15 if e.angleDelta().y() > 0 else 0.85
            view.scale(f, f)
        view.wheelEvent = _zoom
        vlyt = QVBoxLayout(dlg)
        vlyt.addWidget(view)
        dlg.exec()

    # ── Main tab callbacks ───────────────────────────────────────────────

    @safe_refresh
    def refresh_ui(self, auto_fit=False):
        key = self.record_selector.currentData()
        if not key:
            return
        is_input = key == "input image"
        img = self.tool.inputs.get("input image") if is_input else self.tool.outputs.get(key)

        if isinstance(img, np.ndarray):
            self.viewer.set_image(img, auto_fit=auto_fit)
            self.viewer.set_roi(None)
        elif isinstance(img, VisionImage):
            self.viewer.set_image(img, auto_fit=auto_fit,
                                  render_overlays_flag=self.chk_overlays.isChecked())
            if is_input:
                roi_data = self.tool.params["roi"]
                if roi_data[2] > 0 and roi_data[3] > 0:
                    img_input = self.tool.inputs.get("input image")
                    if isinstance(img_input, VisionImage):
                        abs_roi = img_input.fixture_to_image(roi_data)
                    else:
                        abs_roi = roi_data
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
            self.roi_editor.setEnabled(is_input)
            self.roi_editor.setStyleSheet(self._disabled_css if not is_input else "")
            if not is_input:
                self.roi_editor.btn_edit.setChecked(False)
                self.viewer.set_roi_editing(False)

        # Update class tab UIs
        for i in range(self.class_tabs.count()):
            tab = self.class_tabs.widget(i)
            cid = tab.property("class_id")
            if cid and hasattr(tab, "thumb_list"):
                self._refresh_class_tab(tab, cid)

    def on_record_changed(self):
        if self.record_selector.currentData() != "input image":
            if hasattr(self, "roi_editor"):
                self.roi_editor.btn_edit.setChecked(False)
            self.viewer.set_roi_editing(False)
        self.refresh_ui()

    def _on_roi_changed(self, roi):
        self.tool.params["roi"] = roi

    def _on_threshold_changed(self, value):
        thr = value / 100.0
        self.tool.params["threshold"] = thr
        self.lbl_threshold.setText(f"Threshold: {thr:.2f}")
        self.tool.mark_dirty()

    def _on_expected_changed(self, idx):
        label = self.cmb_expected.currentData()
        self.tool.params["expected_class"] = label
        self.tool.mark_dirty()

    def _on_k_changed(self, value):
        self.tool.params["k_nn"] = value
        if self.tool._inference is not None:
            self.tool._inference.k_nn = value

    def _on_output_mode(self, mode):
        self.tool.params["output_mode"] = "ROI Only" if mode == "ROI Only" else "Full Image"
        self.tool.mark_dirty()

    def _on_labels_toggled(self, checked):
        self.tool.params["show_labels"] = checked
        self.tool.mark_dirty()

    def _on_roi_overlay_toggled(self, checked):
        self.tool.params["show_roi_overlay"] = checked
        self.tool.mark_dirty()

    def _set_param(self, key, value):
        self.tool.params[key] = value
        self.tool.mark_dirty()

    def showEvent(self, event):
        super().showEvent(event)
        QTimer.singleShot(100, lambda: self.refresh_ui(auto_fit=True))

    def on_config_closed(self):
        if hasattr(self, '_train_thread') and self._train_thread is not None:
            if self._train_thread.isRunning():
                self._train_thread.quit()
                self._train_thread.wait(5000)
            self._train_thread = None
            self._train_worker = None
        save_params_splitter(self.content_layout, "Classifier")
