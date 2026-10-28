"""Modal dialog for drawing polygon contours on captured training images."""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QGraphicsView, QGraphicsScene, QGraphicsPixmapItem,
    QGraphicsPathItem, QGraphicsEllipseItem, QMessageBox,
    QMenu,
)
from PyQt6.QtCore import Qt, QPointF, pyqtSignal, QTimer
from PyQt6.QtGui import QPixmap, QImage, QPainterPath, QPen, QColor, QBrush


class _AnnotationView(QGraphicsView):
    point_added = pyqtSignal(int, int)
    polygon_closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setRenderHint(
            self.renderHints() |
            self.renderHints().SmoothPixmapTransform |
            self.renderHints().Antialiasing
        )
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu)
        self._drawing = False
        self._closed = False
        self._pixmap_item = None
        self._parent_dialog = parent

    def set_pixmap_item(self, item):
        self._pixmap_item = item

    def wheelEvent(self, event):
        f = 1.15 if event.angleDelta().y() > 0 else 0.85
        self.scale(f, f)

    def mousePressEvent(self, event):
        if not self._drawing or self._closed:
            super().mousePressEvent(event)
            return
        scene_pos = self.mapToScene(event.position().toPoint())
        x, y = int(scene_pos.x()), int(scene_pos.y())
        if event.button() == Qt.MouseButton.RightButton:
            self.polygon_closed.emit()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self.point_added.emit(x, y)

    def mouseDoubleClickEvent(self, event):
        if not self._drawing or self._closed:
            super().mouseDoubleClickEvent(event)
            return
        self.polygon_closed.emit()

    def _on_context_menu(self, pos):
        menu = QMenu(self)
        menu.addAction("Fit Image").triggered.connect(self._fit_image)
        menu.exec(self.mapToGlobal(pos))

    def _fit_image(self):
        if self._pixmap_item is not None:
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)


class ContourEditorDialog(QDialog):
    def __init__(self, image_bgr: np.ndarray, parent=None):
        super().__init__(parent, Qt.WindowType.Window)
        self.setWindowTitle("Annotate Contour")
        self.setMinimumSize(700, 600)
        self.resize(800, 700)

        self._image_bgr = image_bgr
        self._rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        self._h, self._w = image_bgr.shape[:2]

        self._points: list[tuple[float, float]] = []
        self._closed = False
        self._drawing = False

        self._build_ui()
        self._setup_scene()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        btn_lyt = QHBoxLayout()
        btn_lyt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.btn_draw = QPushButton("Draw")
        self.btn_draw.setCheckable(True)
        self.btn_draw.toggled.connect(self._on_draw_toggled)
        btn_lyt.addWidget(self.btn_draw)

        self.btn_undo = QPushButton("Undo Point")
        self.btn_undo.clicked.connect(self._on_undo)
        self.btn_undo.setEnabled(False)
        btn_lyt.addWidget(self.btn_undo)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._on_clear)
        self.btn_clear.setEnabled(False)
        btn_lyt.addWidget(self.btn_clear)

        btn_lyt.addSpacing(60)

        self.btn_accept = QPushButton("Accept")
        self.btn_accept.clicked.connect(self._on_accept)
        self.btn_accept.setEnabled(False)
        btn_lyt.addWidget(self.btn_accept)

        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_lyt.addWidget(self.btn_cancel)

        layout.addLayout(btn_lyt)

        self._view = _AnnotationView(self)
        self._view.point_added.connect(self._add_point)
        self._view.polygon_closed.connect(self._close_polygon)
        self._scene = QGraphicsScene(self)
        self._view.setScene(self._scene)
        layout.addWidget(self._view, 1)

        info = QLabel(
            "Left-click to add points  ·  Right-click or double-click to close polygon"
        )
        info.setStyleSheet("color: #888; font-size: 11px; padding: 2px 0;")
        layout.addWidget(info)

    def _setup_scene(self):
        h, w = self._h, self._w
        qimg = QImage(
            self._rgb.data, w, h, self._rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        self._pixmap_item = QGraphicsPixmapItem(QPixmap.fromImage(qimg))
        self._scene.addItem(self._pixmap_item)
        self._view.set_pixmap_item(self._pixmap_item)
        self._scene.setSceneRect(0, 0, w, h)
        QTimer.singleShot(0, self._fit_image)

        self._path_item = QGraphicsPathItem()
        self._path_item.setPen(QPen(QColor(0, 255, 0), 0))
        self._scene.addItem(self._path_item)

        self._dot_items: list[QGraphicsEllipseItem] = []

    def _on_draw_toggled(self, checked):
        self._drawing = checked
        self._view._drawing = checked
        if checked:
            self._view._closed = False
            self._view.setDragMode(QGraphicsView.DragMode.NoDrag)
            self._view.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self._view.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self._view.setCursor(Qt.CursorShape.ArrowCursor)

    def _add_point(self, x, y):
        x = max(0, min(x, self._w))
        y = max(0, min(y, self._h))
        self._points.append((float(x), float(y)))
        self._update_path()

        dot = QGraphicsEllipseItem(-3, -3, 6, 6)
        dot.setPen(QPen(QColor(0, 255, 0), 0))
        dot.setBrush(QBrush(QColor(0, 255, 0)))
        dot.setPos(float(x), float(y))
        dot.setZValue(10)
        self._scene.addItem(dot)
        self._dot_items.append(dot)

        self.btn_undo.setEnabled(True)
        self.btn_clear.setEnabled(True)
        self.btn_accept.setEnabled(len(self._points) >= 3)

    def _close_polygon(self):
        if len(self._points) < 3:
            return
        self._closed = True
        self._view._closed = True
        self._update_path()
        self._drawing = False
        self._view._drawing = False
        self.btn_draw.setChecked(False)
        self.btn_accept.setEnabled(True)

    def _update_path(self):
        if len(self._points) < 2:
            self._path_item.setPath(QPainterPath())
            return
        path = QPainterPath()
        path.moveTo(*self._points[0])
        for pt in self._points[1:]:
            path.lineTo(*pt)
        if self._closed and len(self._points) >= 3:
            path.closeSubpath()
        # Path in scene coords, ItemIgnoresTransformations means it renders
        # at screen size regardless of zoom; pen width=0 is cosmetic.
        self._path_item.setPath(path)

    def _on_undo(self):
        if not self._points:
            return
        self._points.pop()
        if self._dot_items:
            item = self._dot_items.pop()
            self._scene.removeItem(item)
        self._update_path()
        self.btn_undo.setEnabled(len(self._points) > 0)
        self.btn_clear.setEnabled(len(self._points) > 0)
        self.btn_accept.setEnabled(len(self._points) >= 3)
        self._closed = False
        self._view._closed = False

    def _on_clear(self):
        self._points.clear()
        self._closed = False
        self._view._closed = False
        for item in self._dot_items:
            self._scene.removeItem(item)
        self._dot_items.clear()
        self._path_item.setPath(QPainterPath())
        self.btn_undo.setEnabled(False)
        self.btn_clear.setEnabled(False)
        self.btn_accept.setEnabled(False)

    def _on_accept(self):
        if len(self._points) < 3:
            QMessageBox.warning(self, "Annotation", "Need at least 3 points.")
            return
        self.accept()

    def _fit_image(self):
        v = self._view
        if v.width() < 10 or v.height() < 10:
            QTimer.singleShot(100, self._fit_image)
            return
        v.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def get_mask(self) -> np.ndarray:
        mask = np.zeros((self._h, self._w), dtype=np.uint8)
        if len(self._points) >= 3:
            pts = np.array(self._points, dtype=np.int32).reshape((-1, 1, 2))
            cv2.fillPoly(mask, [pts], 255)
        return mask

    def get_points(self) -> list[tuple[float, float]]:
        return self._points.copy()
