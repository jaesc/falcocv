"""AnomalyDetector — PatchCore-based unsupervised defect detection.

Recording mode captures ROI images from the pipeline for training.
Uses ResNet18-Layer3 ONNX backbone + greedy coreset + per-patch k-NN.
"""

__all__ = ["AnomalyDetector"]

import base64
import io
import numpy as np
import cv2

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QPushButton,
    QLabel, QSlider, QComboBox, QMessageBox,
    QGroupBox, QCheckBox, QTabWidget, QSpinBox, QProgressBar,
    QListWidget, QListWidgetItem, QDialog, QListView,
    QGraphicsView, QGraphicsScene,
)
from PyQt6.QtCore import Qt, QTimer, QSize, QObject, pyqtSignal, QThread
from PyQt6.QtGui import QPixmap, QImage, QIcon, QPainter

from falcocv.core.tool_base import BaseTool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi, create_roi_only_output
from falcocv.core.anomaly.detector import PatchCoreDetector, AnomalyResult
from falcocv.ui.widgets.viewer import ImageViewer
from falcocv.ui.style import ALT_BG, TEXT, SUCCESS, CAUTION, WHITE, BORDER
from falcocv.ui.safe_refresh import safe_refresh
from falcocv.ui.tool_config_helpers import (
    wrap_params_scroll,
    configure_params_splitter,
    save_params_splitter,
    create_execution_footer,
)


def _img_to_b64(img: np.ndarray) -> str:
    ret, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("ascii")


def _b64_to_img(data: str) -> np.ndarray:
    buf = base64.b64decode(data)
    arr = np.frombuffer(buf, dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


from .logic import _b64_to_img, _img_to_b64, AnomalyDetector
class TrainWorker(QObject):
    """Runs PatchCore training in a background thread."""

    progress = pyqtSignal(int, str)
    finished = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, images: list[np.ndarray], alignment_enabled: bool, backbone_type: str = "balanced"):
        super().__init__()
        self.images = images
        self.alignment_enabled = alignment_enabled
        self.backbone_type = backbone_type
        self.result_detector: PatchCoreDetector | None = None

    def run(self):
        try:
            if len(self.images) < 5:
                self.error.emit("Need at least 5 captured images.")
                return

            detector = PatchCoreDetector(backbone_type=self.backbone_type)
            detector.alignment_enabled = self.alignment_enabled
            self.progress.emit(10, "Extracting features...")

            metrics = detector.fit(self.images)

            self.progress.emit(80, "Building memory bank...")
            self.progress.emit(100, "Training complete.")
            self.result_detector = detector
            self.finished.emit(
                f"Trained: {metrics['n_images']} images, "
                f"threshold={metrics['threshold']:.4f}"
            )

        except Exception as e:
            import traceback
            traceback.print_exc()
            self.error.emit(str(e))


class AnomalyDetectorConfig(QWidget):
    def __init__(self, tool: AnomalyDetector):
        super().__init__()
        self.tool = tool
        self._suppress_execute = True

        self.setWindowTitle("Anomaly Detector Config")
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
        self.tabs.addTab(self._build_training_tab(), "Training")
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
            self.content_layout, self.left_scroll, self.right_panel, "AnomalyDetector"
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

    # ── Main tab ─────────────────────────────────────────────────────────

    def _build_main_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)

        from falcocv.ui.widgets.roi_editor import RoiEditorWidget
        self.roi_editor = RoiEditorWidget(param_key="roi")
        layout.addWidget(self.roi_editor)
        self.roi_editor.set_tool(self.tool, self.viewer)

        detect_group = QGroupBox("Detection")
        detect_lyt = QVBoxLayout(detect_group)

        self.lbl_threshold = QLabel(f"Threshold: {self.tool.params['threshold']:.2f}")
        detect_lyt.addWidget(self.lbl_threshold)
        self.threshold_slider = QSlider(Qt.Orientation.Horizontal)
        self.threshold_slider.setRange(0, 100)
        self.threshold_slider.setValue(int(self.tool.params["threshold"] * 100 + 0.5))
        self.threshold_slider.valueChanged.connect(self._on_threshold_changed)
        detect_lyt.addWidget(self.threshold_slider)

        # Backbone description (updates on hover / selection)
        bb_val = self.tool.params.get("backbone_type", "balanced")
        current_bb = {"effnet": "balanced", "patch": "quality"}.get(bb_val, bb_val)
        self.lbl_bb_info = QLabel(self._bb_info(current_bb))
        self.lbl_bb_info.setStyleSheet("color: #999; font-size: 11px;")
        self.lbl_bb_info.setWordWrap(True)
        detect_lyt.addWidget(self.lbl_bb_info)

        # Backbone selector
        bb_lyt = QHBoxLayout()
        bb_lyt.addWidget(QLabel("Backbone:"))
        self.cmb_backbone = QComboBox()
        self.cmb_backbone.addItems(["fast", "balanced", "quality", "detail", "hd"])
        self.cmb_backbone.setCurrentText(current_bb)
        self.cmb_backbone.currentTextChanged.connect(self._on_backbone_changed)
        self.cmb_backbone.highlighted.connect(self._on_bb_hover)
        bb_lyt.addWidget(self.cmb_backbone)
        detect_lyt.addLayout(bb_lyt)

        self.cmb_output = QComboBox()
        self.cmb_output.addItems(["Full Image", "ROI Only"])
        self.cmb_output.setCurrentText(
            "ROI Only" if self.tool.params["output_mode"] == "ROI Only" else "Full Image"
        )
        self.cmb_output.currentTextChanged.connect(self._on_output_mode)
        detect_lyt.addWidget(QLabel("Output Mode:"))
        detect_lyt.addWidget(self.cmb_output)

        # Alignment checkbox
        align_lyt = QHBoxLayout()
        self.chk_alignment = QCheckBox("Enable alignment")
        self.chk_alignment.setChecked(self.tool.params.get("alignment_enabled", True))
        self.chk_alignment.toggled.connect(self._on_alignment_toggled)
        align_lyt.addWidget(self.chk_alignment)
        detect_lyt.addLayout(align_lyt)

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
        self.chk_heatmap = QCheckBox("Show Heatmap")
        self.chk_heatmap.setChecked(self.tool.params.get("show_heatmap", True))
        self.chk_heatmap.toggled.connect(self._on_heatmap_toggled)
        vis_lyt.addWidget(self.chk_heatmap)
        self.chk_copy_in = QCheckBox("Copy overlays to input image")
        self.chk_copy_in.setChecked(self.tool.params.get("copy_overlays_to_input", True))
        self.chk_copy_in.toggled.connect(lambda v: self._set_param("copy_overlays_to_input", v))
        vis_lyt.addWidget(self.chk_copy_in)
        layout.addWidget(vis_group)

        layout.addStretch()
        return tab

    # ── Training tab ─────────────────────────────────────────────────────

    def _build_training_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 4)

        # Capture + Recording
        rec_lyt = QHBoxLayout()
        btn_capture = QPushButton("Capture")
        btn_capture.clicked.connect(self._on_capture_image)
        rec_lyt.addWidget(btn_capture)
        rec_lyt.addSpacing(20)
        btn_start = QPushButton("Start Recording")
        btn_start.clicked.connect(lambda: self._on_start_recording(btn_start, btn_stop))
        rec_lyt.addWidget(btn_start)
        btn_stop = QPushButton("Stop")
        btn_stop.clicked.connect(lambda: self._on_stop_recording(btn_start, btn_stop))
        btn_stop.setEnabled(False)
        rec_lyt.addWidget(btn_stop)
        layout.addLayout(rec_lyt)

        # Max captures
        max_lyt = QHBoxLayout()
        max_lyt.addWidget(QLabel("Max captures:"))
        spin_max = QSpinBox()
        spin_max.setRange(5, 999)
        spin_max.setValue(self.tool.params.get("max_train_samples", 20))
        spin_max.valueChanged.connect(self._on_max_changed)
        max_lyt.addWidget(spin_max)
        layout.addLayout(max_lyt)

        # Progress
        progress = QProgressBar()
        progress.setRange(0, spin_max.value())
        progress.setValue(0)
        layout.addWidget(progress)

        lbl_count = QLabel("Captured: 0")
        layout.addWidget(lbl_count)

        # Remove Pictures / Train / Untrain
        rm_lyt = QHBoxLayout()
        btn_remove_pics = QPushButton("Remove Pictures")
        btn_remove_pics.clicked.connect(self._on_remove_pictures)
        rm_lyt.addWidget(btn_remove_pics)
        rm_lyt.addStretch()
        btn_train = QPushButton("Train")
        btn_train.clicked.connect(self._on_train)
        rm_lyt.addWidget(btn_train)
        btn_untrain = QPushButton("Untrain")
        btn_untrain.clicked.connect(self._on_untrain)
        rm_lyt.addWidget(btn_untrain)
        layout.addLayout(rm_lyt)

        # Status
        lbl_status = QLabel("Not trained")
        lbl_status.setStyleSheet(f"color: {TEXT};")
        layout.addWidget(lbl_status)

        # Thumbnail grid
        thumb_list = QListWidget()
        thumb_list.setViewMode(QListView.ViewMode.IconMode)
        thumb_list.setIconSize(QSize(64, 64))
        thumb_list.setGridSize(QSize(72, 72))
        thumb_list.setMovement(QListView.Movement.Static)
        thumb_list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        thumb_list.itemDoubleClicked.connect(self._on_thumb_clicked)
        thumb_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        thumb_list.customContextMenuRequested.connect(self._on_thumb_context_menu)
        layout.addWidget(thumb_list, 1)

        tab.btn_capture = btn_capture
        tab.btn_start = btn_start
        tab.btn_stop = btn_stop
        tab.spin_max = spin_max
        tab.progress = progress
        tab.lbl_count = lbl_count
        tab.btn_train = btn_train
        tab.btn_untrain = btn_untrain
        tab.lbl_status = lbl_status
        tab.thumb_list = thumb_list

        self._refresh_training_ui()
        return tab

    # ── Training tab callbacks ───────────────────────────────────────────

    def _on_max_changed(self, value):
        self.tool.params["max_train_samples"] = value
        for i in range(self.tabs.count()):
            t = self.tabs.widget(i)
            if hasattr(t, "spin_max"):
                t.spin_max.blockSignals(True)
                t.spin_max.setValue(value)
                t.spin_max.blockSignals(False)
            if hasattr(t, "progress"):
                t.progress.setMaximum(max(value, 1))
            if hasattr(t, "lbl_count"):
                t.lbl_count.setText(f"Captured: {len(self.tool._train_images)}")
        self.tool.data_changed.emit()

    def _on_capture_image(self):
        img_input = self.tool.inputs.get("input image")
        if img_input is None:
            QMessageBox.warning(self, "Capture", "No input image available.")
            return
        img = img_input if isinstance(img_input, VisionImage) else VisionImage.from_array(img_input)
        roi = extract_roi(img, self.tool.params["roi"])
        roi_src = roi["roi_src"]
        self.tool._train_images.append(roi_src.copy())
        self._refresh_training_ui()
        self.tool.data_changed.emit()

    def _on_start_recording(self, btn_start, btn_stop):
        self.tool.start_recording()
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
        self._refresh_training_ui()

    def _on_train(self):
        if self.tool.train_count < 5:
            QMessageBox.warning(self, "Training", "Capture at least 5 images first.")
            return

        tab = self.tabs.widget(1)
        tab.btn_train.setEnabled(False)
        tab.btn_train.setText("Training...")
        tab.btn_train.setStyleSheet(f"""
            QPushButton {{ background-color: {CAUTION}; color: {WHITE}; font-weight: bold; }}
        """)
        tab.btn_train.repaint()
        tab.lbl_status.setText("Starting training...")

        self._train_thread = QThread(self)
        self._train_worker = TrainWorker(
            images=[img.copy() for img in self.tool._train_images],
            alignment_enabled=self.tool.params.get("alignment_enabled", True),
            backbone_type=self.tool.params.get("backbone_type", "balanced"),
        )
        self._train_worker.moveToThread(self._train_thread)
        self._train_thread.started.connect(self._train_worker.run)
        self._train_worker.finished.connect(self._on_train_complete)
        self._train_worker.error.connect(self._on_train_error)
        self._train_worker.progress.connect(self._on_train_progress)
        self._train_worker.finished.connect(self._train_worker.deleteLater)
        self._train_thread.start()

    def _on_train_progress(self, pct: int, msg: str):
        tab = self.tabs.widget(1)
        tab.lbl_status.setText(msg)

    def _on_train_complete(self, msg: str):
        detector = self._train_worker.result_detector
        if detector is None:
            self._on_train_error("Training produced no detector")
            return

        self.tool._detector = detector
        self.tool._trained = True
        self.tool.params["is_trained"] = True
        self.tool.params["threshold"] = detector.threshold

        # Update threshold slider to calibrated value
        slider_val = int(detector.threshold * 100 + 0.5)
        slider_val = max(0, min(100, slider_val))
        self.threshold_slider.blockSignals(True)
        self.threshold_slider.setValue(slider_val)
        self.threshold_slider.blockSignals(False)
        self.lbl_threshold.setText(f"Threshold: {detector.threshold:.2f}")

        tab = self.tabs.widget(1)
        tab.btn_train.setEnabled(True)
        tab.btn_train.setText("Train")
        tab.btn_train.setStyleSheet("")

        self._refresh_training_ui()
        tab.lbl_status.setText(msg)
        tab.lbl_status.setStyleSheet(f"color: {SUCCESS};")
        self.tool.data_changed.emit()
        self._cleanup_train_thread()

    def _on_train_error(self, msg: str):
        tab = self.tabs.widget(1)
        tab.btn_train.setEnabled(True)
        tab.btn_train.setText("Train")
        tab.btn_train.setStyleSheet("")
        tab.lbl_status.setText(f"Training failed: {msg}")
        tab.lbl_status.setStyleSheet(f"color: {CAUTION};")
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
        self._refresh_training_ui()
        tab = self.tabs.widget(1)
        tab.lbl_status.setText("Not trained")
        tab.lbl_status.setStyleSheet(f"color: {TEXT};")
        self.tool.data_changed.emit()

    def _on_remove_pictures(self):
        reply = QMessageBox.question(
            self, "Remove Pictures",
            "Remove all captured images?\nThe training state will be preserved.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return
        self.tool._train_images.clear()
        self._refresh_training_ui()
        self.tool.data_changed.emit()

    def _on_thumb_context_menu(self, pos):
        tab = self.tabs.widget(1)
        if not hasattr(tab, "thumb_list"):
            return
        item = tab.thumb_list.itemAt(pos)
        if item is None:
            return
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        from PyQt6.QtWidgets import QMenu
        menu = QMenu(self)
        menu.addAction("Delete Picture")
        action = menu.exec(tab.thumb_list.mapToGlobal(pos))
        if action and action.text() == "Delete Picture":
            imgs = self.tool._train_images
            if 0 <= idx < len(imgs):
                del imgs[idx]
            self._refresh_training_ui()
            self.tool.data_changed.emit()

    def _on_thumb_clicked(self, item):
        idx = item.data(Qt.ItemDataRole.UserRole)
        if idx is None:
            return
        imgs = self.tool._train_images
        if idx >= len(imgs):
            return
        img = imgs[idx]
        dlg = QDialog(self, Qt.WindowType.Window)
        dlg.setWindowTitle(f"Captured Image #{idx}")
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

    def _refresh_training_ui(self):
        imgs = self.tool._train_images
        tab = self.tabs.widget(1)
        if tab is None:
            return
        max_s = tab.spin_max.value() if hasattr(tab, "spin_max") else self.tool.params.get("max_train_samples", 20)
        tab.progress.setMaximum(max(max_s, 1))
        is_recording = self.tool.recording
        tab.progress.setValue(self.tool._recording_count if is_recording else 0)
        tab.lbl_count.setText(f"Captured: {len(imgs)}")

        if is_recording:
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

        if self.tool.is_trained and self.tool._detector is not None:
            thr = self.tool._detector.threshold
            tab.lbl_status.setText(f"Trained: {len(imgs)} images, threshold={thr:.4f}")
            tab.lbl_status.setStyleSheet(f"color: {SUCCESS};")
        elif len(imgs) > 0:
            tab.lbl_status.setText(f"{len(imgs)} images — not trained")
            tab.lbl_status.setStyleSheet(f"color: {TEXT};")
        else:
            tab.lbl_status.setText("Not trained")
            tab.lbl_status.setStyleSheet(f"color: {TEXT};")

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

        self._refresh_training_ui()

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

    _BB_INFO = {
        "fast":     "~0.5\u00d7 speed, 784 patches. For high-throughput inspection of large defects.",
        "balanced": "1\u00d7 speed (default), 784 patches. Best speed/quality trade-off for general use.",
        "quality":  "~3\u00d7 speed, 784 patches. Deeper semantic features for subtle defect discrimination.",
        "detail":   "~4\u00d7 speed, 3136 patches. 4\u00d7 finer grid for detecting small defects (8\u201316 px).",
        "hd":       "~12\u00d7 speed, 3136 patches. Maximum resolution and detection quality.",
    }

    @staticmethod
    def _bb_info(key: str) -> str:
        return AnomalyDetectorConfig._BB_INFO.get(key, "")

    def _on_bb_hover(self, index: int):
        backbone = self.cmb_backbone.itemText(index)
        self.lbl_bb_info.setText(self._bb_info(backbone))

    def _on_backbone_changed(self, backbone):
        if backbone == self.tool.params.get("backbone_type", "balanced"):
            return
        if self.tool._trained:
            reply = QMessageBox.question(
                self, "Backbone",
                "Changing backbone requires retraining. Confirm?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Ok:
                self._on_untrain()
            else:
                self.cmb_backbone.blockSignals(True)
                bb_val = self.tool.params.get("backbone_type", "balanced")
                self.cmb_backbone.setCurrentText({"effnet": "balanced", "patch": "quality"}.get(bb_val, bb_val))
                self.cmb_backbone.blockSignals(False)
                return
        self.tool.params["backbone_type"] = backbone
        self.lbl_bb_info.setText(self._bb_info(backbone))

    def _on_alignment_toggled(self, checked):
        if checked and self.tool._detector is not None and self.tool._detector.template is None:
            reply = QMessageBox.question(
                self, "Alignment",
                "Enabling alignment requires retraining. Confirm?",
                QMessageBox.StandardButton.Ok | QMessageBox.StandardButton.Cancel,
            )
            if reply == QMessageBox.StandardButton.Ok:
                self._on_untrain()
            else:
                self.chk_alignment.blockSignals(True)
                self.chk_alignment.setChecked(False)
                self.chk_alignment.blockSignals(False)
                return
        self.tool.params["alignment_enabled"] = checked

    def _on_output_mode(self, mode):
        self.tool.params["output_mode"] = "ROI Only" if mode == "ROI Only" else "Full Image"
        self.tool.mark_dirty()

    def _on_labels_toggled(self, checked):
        self.tool.params["show_labels"] = checked
        self.tool.mark_dirty()

    def _on_roi_overlay_toggled(self, checked):
        self.tool.params["show_roi_overlay"] = checked
        self.tool.mark_dirty()

    def _on_heatmap_toggled(self, checked):
        self.tool.params["show_heatmap"] = checked
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
        save_params_splitter(self.content_layout, "AnomalyDetector")
