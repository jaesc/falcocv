"""Interactive training-mask editor for ShapeMatch."""

from __future__ import annotations

import cv2
import numpy as np
from PyQt6.QtCore import QRect, QRectF, Qt
from PyQt6.QtGui import QBrush, QColor, QImage, QPainter, QPen
from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class _MaskCanvas(QWidget):
    def __init__(self, image: np.ndarray, mask: np.ndarray, parent=None):
        super().__init__(parent)
        self.image = image
        self.mask = mask.copy()
        self.brush_radius = 12
        self.paint_value = 0
        self._drawing = False
        self._last = None
        self._hover_pos = None
        self.setMinimumSize(480, 360)
        self.setMouseTracking(True)

    def _image_rect(self):
        height, width = self.image.shape[:2]
        available_w = max(1, self.width() - 8)
        available_h = max(1, self.height() - 8)
        scale = min(available_w / float(width), available_h / float(height))
        draw_w = max(1, int(round(width * scale)))
        draw_h = max(1, int(round(height * scale)))
        return (
            (self.width() - draw_w) // 2,
            (self.height() - draw_h) // 2,
            draw_w,
            draw_h,
            scale,
        )

    def _to_image(self, pos):
        x, y, draw_w, draw_h, scale = self._image_rect()
        image_x = int((pos.x() - x) / scale)
        image_y = int((pos.y() - y) / scale)
        height, width = self.image.shape[:2]
        if 0 <= image_x < width and 0 <= image_y < height:
            return image_x, image_y
        return None

    def _paint_at(self, pos) -> bool:
        point = self._to_image(pos)
        if point is None:
            return False
        _x, _y, _draw_w, _draw_h, scale = self._image_rect()
        radius = max(1, int(round(float(self.brush_radius) / scale)))
        if self._last is not None:
            cv2.line(
                self.mask,
                self._last,
                point,
                int(self.paint_value),
                thickness=max(1, radius * 2),
            )
        else:
            cv2.circle(
                self.mask,
                point,
                radius,
                int(self.paint_value),
                -1,
            )
        self._last = point
        return True

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(24, 24, 24))
        x, y, draw_w, draw_h, _scale = self._image_rect()

        rgb = self.image
        if rgb.ndim == 2:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_GRAY2RGB)
        elif rgb.shape[2] == 3:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGRA2RGB)
        rgb = np.ascontiguousarray(rgb)
        image = QImage(
            rgb.data,
            rgb.shape[1],
            rgb.shape[0],
            rgb.strides[0],
            QImage.Format.Format_RGB888,
        )
        painter.drawImage(QRect(x, y, draw_w, draw_h), image)

        overlay = np.zeros((*self.mask.shape[:2], 4), dtype=np.uint8)
        overlay[self.mask == 0] = (255, 0, 0, 120)
        overlay = np.ascontiguousarray(overlay)
        overlay_image = QImage(
            overlay.data,
            overlay.shape[1],
            overlay.shape[0],
            overlay.strides[0],
            QImage.Format.Format_RGBA8888,
        )
        painter.drawImage(QRect(x, y, draw_w, draw_h), overlay_image)

        painter.setPen(QPen(QColor(180, 180, 180), 1))
        painter.drawRect(QRect(x, y, draw_w, draw_h))

        if self._hover_pos is not None:
            radius = max(1.0, float(self.brush_radius))
            center_x = float(self._hover_pos.x())
            center_y = float(self._hover_pos.y())
            cursor_rect = QRectF(
                center_x - radius,
                center_y - radius,
                radius * 2.0,
                radius * 2.0,
            )
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setBrush(QBrush())
            painter.setPen(QPen(QColor(16, 16, 16), 4))
            painter.drawEllipse(cursor_rect)
            painter.setPen(QPen(QColor(255, 255, 255), 2))
            painter.drawEllipse(cursor_rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = True
            self._last = None
            point = self._to_image(event.pos())
            self._hover_pos = event.pos() if point is not None else None
            if self._paint_at(event.pos()):
                self.update()

    def mouseMoveEvent(self, event):
        point = self._to_image(event.pos())
        self._hover_pos = event.pos() if point is not None else None
        if self._drawing and event.buttons() & Qt.MouseButton.LeftButton:
            if self._paint_at(event.pos()):
                self.update()
        else:
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            self._last = None

    def leaveEvent(self, event):
        self._hover_pos = None
        self.update()
        super().leaveEvent(event)


class ShapeMatchMaskDialog(QDialog):
    """Paint areas of the training crop that must not produce features."""

    def __init__(self, image: np.ndarray, mask: np.ndarray | None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ShapeMatch Training Mask")
        self.resize(760, 620)

        if mask is None or mask.shape[:2] != image.shape[:2]:
            current = np.full(image.shape[:2], 255, dtype=np.uint8)
            mismatch = mask is not None
        else:
            current = np.where(mask > 0, 255, 0).astype(np.uint8)
            mismatch = False

        layout = QVBoxLayout(self)
        info = QLabel(
            "Paint red areas to exclude them from feature extraction. "
            "Everything else keeps its normal shape features."
        )
        info.setWordWrap(True)
        layout.addWidget(info)
        legend = QLabel(
            '<span style="color:#FF3D00; font-weight:bold;">Red</span>'
            " = excluded &nbsp;·&nbsp; "
            "no red = included &nbsp;·&nbsp; "
            "white ring = brush"
        )
        legend.setWordWrap(True)
        legend.setStyleSheet("font-size: 11px;")
        layout.addWidget(legend)
        if mismatch:
            warning = QLabel(
                "The previous mask did not match the current training ROI, "
                "so the editor started with an empty mask."
            )
            warning.setWordWrap(True)
            layout.addWidget(warning)

        self.canvas = _MaskCanvas(image, current)
        layout.addWidget(self.canvas, 1)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Brush:"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 100)
        self.spin_brush.setValue(self.canvas.brush_radius)
        self.spin_brush.setToolTip(
            "Brush radius in screen pixels; the white ring shows it."
        )
        self.spin_brush.valueChanged.connect(self._on_brush_changed)
        controls.addWidget(self.spin_brush)

        self.btn_mask = QPushButton("Mask")
        self.btn_mask.setCheckable(True)
        self.btn_mask.setChecked(True)
        self.btn_mask.setStyleSheet(
            "QPushButton:checked {"
            " background-color: #5C1A12; color: white;"
            " border: 1px solid #FF3D00;"
            "}"
        )
        self.btn_erase = QPushButton("Erase")
        self.btn_erase.setCheckable(True)
        self.btn_erase.setStyleSheet(
            "QPushButton:checked {"
            " background-color: #1B5E20; color: white;"
            " border: 1px solid #00E676;"
            "}"
        )
        self.btn_mask.clicked.connect(lambda: self._set_mode(0))
        self.btn_erase.clicked.connect(lambda: self._set_mode(255))
        controls.addWidget(self.btn_mask)
        controls.addWidget(self.btn_erase)

        self.btn_clear = QPushButton("Clear Mask")
        self.btn_clear.clicked.connect(self._clear_mask)
        controls.addWidget(self.btn_clear)
        controls.addStretch()
        layout.addLayout(controls)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_brush_changed(self, value):
        self.canvas.brush_radius = int(value)
        self.canvas.update()

    def _set_mode(self, value: int):
        self.canvas.paint_value = int(value)
        self.btn_mask.setChecked(value == 0)
        self.btn_erase.setChecked(value == 255)

    def _clear_mask(self):
        self.canvas.mask[:] = 255
        self.canvas.update()

    @property
    def mask(self) -> np.ndarray:
        return self.canvas.mask.copy()
