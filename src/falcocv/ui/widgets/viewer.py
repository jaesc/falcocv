from PyQt6.QtWidgets import (QGraphicsView, QGraphicsScene, QGraphicsPixmapItem, 
                             QGraphicsRectItem, QRubberBand, QMenu)
from PyQt6.QtGui import QPixmap, QImage, QPen, QColor, QBrush, QPainter, QAction, QFont, QPolygonF
from PyQt6.QtCore import Qt, QRectF, pyqtSignal, QPoint, QSize, QRect, QPointF, QTimer, QEvent
import numpy as np
import cv2
from falcocv.ui.style import HIGHLIGHT, ALT_BG, PANEL_BG, TEXT, CAUTION, HANDLE_STROKE, HIGHLIGHT_ALPHA, HANDLE_FILL, OVERLAY_FALLBACK
from falcocv.core.depth_viz import apply_depth_colormap, depth_to_gray_u8

class ResizableRectItem(QGraphicsRectItem):
    def __init__(self, rect, callback=None):
        super().__init__(rect)
        self.setFlags(QGraphicsRectItem.GraphicsItemFlag.ItemSendsGeometryChanges)
        self.setAcceptHoverEvents(True)
        self.setEnabled(False)
        self.callback = callback
        self.handle_size = 10
        self.handles = {} 
        self._drag_start_scene_pos = None
        self._initial_scene_mouse = None
        self._initial_rect = None
        self._initial_rotation = 0
        self._dragging_handle = None
        self.update_handles()
        
    def update_handles(self):
        r = self.rect()
        s = self.handle_size
        hs = max(2.0, min(4.0, min(r.width(), r.height()) * 0.15))
        self.handles = {
            'topleft': QRectF(r.left(), r.top(), hs, hs),
            'topright': QRectF(r.right() - hs, r.top(), hs, hs),
            'bottomleft': QRectF(r.left(), r.bottom() - hs, hs, hs),
            'bottomright': QRectF(r.right() - hs, r.bottom() - hs, hs, hs),
            'rotate': QRectF(r.center().x() - s / 2, r.top() + 3, s, s)
        }

    def shape(self):
        # Even simpler shape that definitely covers the handles
        path = super().shape()
        for h_rect in self.handles.values():
            path.addRect(h_rect)
        return path

    def paint(self, painter, option, widget):
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        view = self.scene().views()[0] if self.scene().views() else None
        if view:
            zoom = view.viewportTransform().m11()
            s = max(4.0, 10.0 / zoom)
        else:
            s = 10.0

        self.handle_size = s
        self.update_handles()

        main_pen = QPen(QColor(HIGHLIGHT), 2)
        main_pen.setCosmetic(True)
        handle_pen = QPen(HANDLE_STROKE, 1)
        handle_pen.setCosmetic(True)
        rotate_pen = QPen(QColor(CAUTION), 1)
        rotate_pen.setCosmetic(True)

        painter.setPen(main_pen)
        painter.setBrush(QBrush(HIGHLIGHT_ALPHA))
        painter.drawRect(self.rect())

        if view and view._roi_show_direction:
            tri_size = s * 0.4
            cx = self.rect().center().x()
            cy = self.rect().bottom()
            tri = QPolygonF([
                QPointF(cx - tri_size, cy),
                QPointF(cx + tri_size, cy),
                QPointF(cx, cy + tri_size * 1.5),
            ])
            tri_pen = QPen(QColor(CAUTION), 1)
            tri_pen.setCosmetic(True)
            painter.setPen(tri_pen)
            painter.setBrush(QBrush(QColor(CAUTION)))
            painter.drawPolygon(tri)

        for name, h_rect in self.handles.items():
            if name == 'rotate':
                painter.setPen(rotate_pen)
                painter.setBrush(QBrush(QColor(CAUTION)))
                painter.drawEllipse(h_rect)
            else:
                painter.setPen(handle_pen)
                painter.setBrush(QBrush(HANDLE_FILL))
                painter.drawRect(h_rect)

    def hoverMoveEvent(self, event):
        handle = self.get_handle_at(event.pos())
        if handle:
            if handle in ['topleft', 'topright', 'bottomleft', 'bottomright']:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
            elif handle == 'rotate':
                self.setCursor(Qt.CursorShape.PointingHandCursor)
            else:
                self.setCursor(Qt.CursorShape.CrossCursor)
        else:
            self.setCursor(Qt.CursorShape.CrossCursor)
        super().hoverMoveEvent(event)

    def get_handle_at(self, pos):
        for name, rect in self.handles.items():
            if rect.contains(pos):
                return name
        return None

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._dragging_handle = self.get_handle_at(event.pos())
            self._drag_start_scene_pos = event.scenePos()
            self._initial_rect = self.rect()
            self._initial_rotation = self.rotation()
            self._initial_pos = self.pos()
            self._initial_scene_mouse = event.scenePos()
            self.setTransformOriginPoint(self.rect().center())
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag_start_scene_pos is not None:
            if self._dragging_handle == 'rotate':
                center = self.mapToScene(self.rect().center())
                current_pos = event.scenePos()
                diff = current_pos - center
                import math
                angle_rad = math.atan2(diff.y(), diff.x())
                angle_deg = math.degrees(angle_rad) + 90
                self.prepareGeometryChange()
                self.setRotation(angle_deg)
            elif self._dragging_handle:
                import math
                scene_delta = event.scenePos() - self._initial_scene_mouse
                angle_rad = math.radians(self._initial_rotation)
                cos_a = math.cos(angle_rad)
                sin_a = math.sin(angle_rad)

                # Rotate scene delta to local coordinates
                local_delta = QPointF(
                    cos_a * scene_delta.x() + sin_a * scene_delta.y(),
                    -sin_a * scene_delta.x() + cos_a * scene_delta.y()
                )

                r = self._initial_rect
                handle = self._dragging_handle
                if handle == 'bottomright':
                    rect = QRectF(r.topLeft(), r.bottomRight() + local_delta)
                    fixed_local = r.topLeft()
                elif handle == 'topleft':
                    rect = QRectF(r.topLeft() + local_delta, r.bottomRight())
                    fixed_local = r.bottomRight()
                elif handle == 'bottomleft':
                    rect = QRectF(
                        QPointF(r.left() + local_delta.x(), r.top()),
                        QPointF(r.right(), r.bottom() + local_delta.y())
                    )
                    fixed_local = r.topRight()
                elif handle == 'topright':
                    rect = QRectF(
                        QPointF(r.left(), r.top() + local_delta.y()),
                        QPointF(r.right() + local_delta.x(), r.bottom())
                    )
                    fixed_local = r.bottomLeft()

                if rect.width() > 1.0 and rect.height() > 1.0:
                    self.prepareGeometryChange()
                    self.setRect(rect)

                    new_center = rect.center()
                    self.setTransformOriginPoint(new_center)

                    old_center = r.center()

                    def rot(pt):
                        return QPointF(
                            cos_a * pt.x() - sin_a * pt.y(),
                            sin_a * pt.x() + cos_a * pt.y()
                        )

                    old_scene_fixed = self._initial_pos + rot(fixed_local - old_center) + old_center
                    new_pos = old_scene_fixed - rot(fixed_local - new_center) - new_center
                    self.setPos(new_pos)
            else:
                delta_scene = event.scenePos() - self._drag_start_scene_pos
                self.setPos(self.pos() + delta_scene)
                self._drag_start_scene_pos = event.scenePos()

            self.update_handles()

            if self.callback:
                r = self.rect()
                p = self.pos()
                self.callback([r.x() + p.x(), r.y() + p.y(), r.width(), r.height(), self.rotation()])
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self.setTransformOriginPoint(self.rect().center())
        self._dragging_handle = None
        self._drag_start_scene_pos = None
        self._initial_scene_mouse = None
        self._initial_rect = None
        self._initial_rotation = 0
        self._initial_pos = None
        super().mouseReleaseEvent(event)

class ImageViewer(QGraphicsView):
    roi_changed = pyqtSignal(object) # Can be QRectF or list [x, y, w, h, angle]
    roi_edit_finished = pyqtSignal()
    roi_drawing_finished = pyqtSignal()
    mouse_moved = pyqtSignal(int, int, object) # x, y, value (int or tuple)
    mouse_clicked = pyqtSignal(int, int)       # x, y in scene coordinates

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.viewport().setMouseTracking(True)
        self.viewport().installEventFilter(self) # Install filter to catch all events
        
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        
        self.pixmap_item = QGraphicsPixmapItem()
        self.scene.addItem(self.pixmap_item)
        
        self.roi_rect_item = ResizableRectItem(QRectF(0,0,0,0), callback=self.on_roi_item_changed)
        roi_pen = QPen(QColor(HIGHLIGHT), 2)
        roi_pen.setCosmetic(True)
        self.roi_rect_item.setPen(roi_pen)
        self.roi_rect_item.setBrush(QBrush(HIGHLIGHT_ALPHA))
        self.roi_rect_item.setZValue(10)
        self.roi_rect_item.hide()
        self.scene.addItem(self.roi_rect_item)
        
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setBackgroundBrush(QBrush(QColor(PANEL_BG)))
        
        self._is_editing_roi = False
        self._is_drawing_new_roi = False
        self._edge_config_count = 0
        self._roi_show_direction = False
        self.rubber_band = QRubberBand(QRubberBand.Shape.Rectangle, self)
        self.origin = QPoint()
        self._current_raw_image = None
        self._current_vision_image = None
        self._current_vision_img_id = None
        self._show_overlays = False
        self._overlay_primitives = []
        self._pixmap_ready = False
        # Scale between original image pixels and on-screen (downsampled) pixels.
        self._display_scale_x = 1.0
        self._display_scale_y = 1.0
        self.set_roi_editing(False)

    def eventFilter(self, source, event):
        if source is self.viewport():
            if event.type() == QEvent.Type.MouseMove:
                self._handle_mouse_move(event.pos())
            elif event.type() == QEvent.Type.MouseButtonPress:
                if event.button() == Qt.MouseButton.LeftButton:
                    orig_pos = self._display_to_original_point(self.mapToScene(event.pos()))
                    self.mouse_clicked.emit(int(orig_pos.x()), int(orig_pos.y()))
        return super().eventFilter(source, event)
                    
        return super().eventFilter(source, event)

    def _display_to_original_point(self, point_f):
        return QPointF(point_f.x() * self._display_scale_x,
                       point_f.y() * self._display_scale_y)

    def _handle_mouse_move(self, pos):
        scene_pos = self.mapToScene(pos)
        orig_pos = self._display_to_original_point(scene_pos)
        if self._current_raw_image is not None:
            x = int(orig_pos.x())
            y = int(orig_pos.y())
            h, w = self._current_raw_image.shape[:2]

            if 0 <= x < w and 0 <= y < h:
                try:
                    val = self._current_raw_image[y, x]
                    self.mouse_moved.emit(x, y, val)
                except:
                    pass
            else:
                self.mouse_moved.emit(-1, -1, None)

    def on_roi_item_changed(self, rect_f):
        # The item lives in on-screen (possibly downsampled) coordinates; report
        # ROI changes back in original image pixels.
        if isinstance(rect_f, (list, tuple)):
            angle = float(rect_f[4]) if len(rect_f) > 4 else 0.0
            rect_f = QRectF(rect_f[0], rect_f[1], rect_f[2], rect_f[3])
        else:
            angle = 0.0
        orig_rect = self._display_to_original_rect(rect_f)
        self.roi_changed.emit([
            orig_rect.x(), orig_rect.y(),
            orig_rect.width(), orig_rect.height(),
            angle,
        ])

    def set_image(self, img_obj, auto_fit=False, depth_display=None, render_overlays_flag=True):
        """
        img_obj: VisionImage or numpy array
        depth_display: optional dict for 2D range/depth arrays
        render_overlays_flag: if False, skip overlay rendering even if present
        """
        from falcocv.core.vision_image import VisionImage
        is_vision_img = isinstance(img_obj, VisionImage)
        
        if is_vision_img:
            # If same VisionImage object, only refresh overlays
            if (self._current_vision_image is img_obj
                and self._pixmap_ready):
                self._show_overlays = render_overlays_flag
                self._overlay_primitives = self._build_overlay_cache() if self._current_vision_image is not None else []
                self.viewport().update()
                return

            self._current_vision_image = img_obj
            self._current_vision_img_id = id(img_obj)
            self._show_overlays = render_overlays_flag
            img = img_obj.data
            self._pixmap_ready = True
        else:
            self._current_vision_image = None
            self._current_vision_img_id = None
            self._show_overlays = False
            self._pixmap_ready = False
            img = img_obj
        
        self._current_raw_image = img
        if img is None:
            self.pixmap_item.setPixmap(QPixmap())
            self._current_image_shape = None
            self._overlay_primitives = self._build_overlay_cache()
            return

        # Float range images must not use raw float buffers as Grayscale8.
        if (
            depth_display is None
            and isinstance(img, np.ndarray)
            and img.ndim == 2
            and img.dtype.kind == "f"
        ):
            depth_display = {
                "use_colormap": True,
                "colormap": "TURBO",
                "vmin": None,
                "vmax": None,
            }

        # Downsample very large images BEFORE colormap/processing. The working image
        # stays full-res; only the on-screen pixmap is reduced to keep UI responsive.
        _MAX_DISPLAY_DIM = 4096
        orig_h, orig_w = img.shape[:2]
        if max(orig_h, orig_w) > _MAX_DISPLAY_DIM:
            display_scale = _MAX_DISPLAY_DIM / max(orig_h, orig_w)
            new_w = max(1, int(orig_w * display_scale))
            new_h = max(1, int(orig_h * display_scale))
            interp = cv2.INTER_AREA if display_scale < 0.5 else cv2.INTER_LINEAR
            img = cv2.resize(img, (new_w, new_h), interpolation=interp)
            self._display_scale_x = orig_w / new_w
            self._display_scale_y = orig_h / new_h
        else:
            self._display_scale_x = 1.0
            self._display_scale_y = 1.0

        self._overlay_primitives = self._build_overlay_cache()

        display_img = img
        if (
            depth_display
            and isinstance(img, np.ndarray)
            and img.ndim == 2
            and img.dtype.kind in "uif"
        ):
            try:
                if depth_display.get("use_colormap", True):
                    display_img = apply_depth_colormap(
                        img,
                        str(depth_display.get("colormap", "TURBO")),
                        depth_display.get("vmin"),
                        depth_display.get("vmax"),
                    )
                else:
                    gray = depth_to_gray_u8(
                        img,
                        depth_display.get("vmin"),
                        depth_display.get("vmax"),
                    )
                    display_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            except Exception:
                display_img = apply_depth_colormap(img, "TURBO", None, None)

        img = display_img

        # QImage wraps the raw buffer: layout must match format (PNG BGRA vs BGR, stride).
        if len(img.shape) == 3:
            h, w, ch = img.shape
            if ch == 1:
                h, w = img.shape[0], img.shape[1]
                img = np.ascontiguousarray(img.squeeze(axis=2))
                q_img = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)
            elif ch == 3:
                img = np.ascontiguousarray(img)
                q_img = QImage(
                    img.data, w, h, 3 * w, QImage.Format.Format_BGR888
                )
            elif ch == 4:
                # OpenCV BGRA (e.g. PNG with alpha from IMREAD_UNCHANGED)
                img = np.ascontiguousarray(img)
                q_img = QImage(
                    img.data, w, h, 4 * w, QImage.Format.Format_BGRA8888
                )
            else:
                # Rare multi-channel: show first 3 planes as BGR for debugging
                img = np.ascontiguousarray(img[:, :, :3])
                q_img = QImage(
                    img.data, w, h, 3 * w, QImage.Format.Format_BGR888
                )
        else:
            h, w = img.shape
            img = np.ascontiguousarray(img)
            q_img = QImage(img.data, w, h, w, QImage.Format.Format_Grayscale8)

        pixmap = QPixmap.fromImage(q_img)
        self.pixmap_item.setPixmap(pixmap)
        r = pixmap.rect()
        margin = max(r.width(), r.height()) * 2
        self.scene.setSceneRect(QRectF(r.x() - margin, r.y() - margin,
                                       r.width() + margin * 2, r.height() + margin * 2))
        
        if auto_fit:
            if getattr(self, '_current_image_shape', None) != img.shape:
                QTimer.singleShot(100, self.fit_in_view)
                
        self._current_image_shape = img.shape

    def _display_to_original_rect(self, rect_f):
        """Convert on-screen (possibly downsampled) rect to original image coords."""
        return QRectF(
            rect_f.x() * self._display_scale_x,
            rect_f.y() * self._display_scale_y,
            rect_f.width() * self._display_scale_x,
            rect_f.height() * self._display_scale_y,
        )

    def _original_to_display_rect(self, rect_f):
        """Convert original image rect to on-screen (possibly downsampled) coords."""
        if self._display_scale_x <= 0 or self._display_scale_y <= 0:
            return rect_f
        return QRectF(
            rect_f.x() / self._display_scale_x,
            rect_f.y() / self._display_scale_y,
            rect_f.width() / self._display_scale_x,
            rect_f.height() / self._display_scale_y,
        )

    def drawForeground(self, painter, rect):
        if not self._show_overlays:
            return
        if not self._overlay_primitives:
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for prim in self._overlay_primitives:
            cmd = prim[0]
            if cmd == "line":
                _, s, e, p = prim
                painter.setPen(p)
                painter.drawLine(s, e)
            elif cmd == "circle":
                _, c, r, p, b = prim
                painter.setPen(p)
                painter.setBrush(b)
                painter.drawEllipse(c, r, r)
            elif cmd == "point":
                _, c, r, p = prim
                painter.setPen(p)
                painter.drawLine(QPointF(c.x() - r, c.y()), QPointF(c.x() + r, c.y()))
                painter.drawLine(QPointF(c.x(), c.y() - r), QPointF(c.x(), c.y() + r))
            elif cmd == "contour":
                _, poly, p, b = prim
                painter.setPen(p)
                painter.setBrush(b)
                painter.drawPolygon(poly)
            elif cmd == "roi":
                _, rect_f, angle, p, b = prim
                painter.save()
                painter.translate(rect_f.center())
                painter.rotate(angle)
                painter.translate(-rect_f.center())
                painter.setPen(p)
                painter.setBrush(b)
                painter.drawRect(rect_f)
                painter.restore()
            elif cmd == "roi_image":
                _, rect_f, angle, pixmap, alpha, p = prim
                painter.save()
                painter.translate(rect_f.center())
                painter.rotate(angle)
                painter.translate(-rect_f.center())
                painter.setOpacity(alpha)
                painter.drawPixmap(rect_f.topLeft(), pixmap)
                painter.setOpacity(1.0)
                if p.width() > 0:
                    painter.setPen(p)
                    painter.setBrush(QBrush())
                    painter.drawRect(rect_f)
                painter.restore()
            elif cmd == "axes":
                _, c, ex, ey, p_x, p_y = prim
                painter.setPen(p_x)
                painter.drawLine(c, ex)
                painter.setPen(p_y)
                painter.drawLine(c, ey)
            elif cmd == "text":
                _, text, pos, f, p = prim
                painter.setPen(p)
                painter.setFont(f)
                painter.drawText(pos, text)
            painter.setBrush(QBrush())

    def _build_overlay_cache(self):
        primitives = []
        from falcocv.core.overlay_layer import OverlayType
        if self._current_vision_image is None:
            return primitives
        def _bgr_to_qcolor(c):
            return QColor(c[2], c[1], c[0]) if len(c) >= 3 else QColor(OVERLAY_FALLBACK)
        for layer in self._current_vision_image.overlay_layers.values():
            if not layer.visible:
                continue
            for elem in layer.get_visible_elements():
                color_data = elem.data.get("color", (0, 255, 0))
                color = _bgr_to_qcolor(color_data)
                thickness = elem.data.get("thickness", 2)
                if thickness <= 0:
                    pen = QPen(Qt.GlobalColor.transparent, 0)
                else:
                    pen = QPen(color, thickness)
                    pen.setCosmetic(True)

                # All overlay primitives are defined in original image pixels; scale
                # them to the on-screen (downsampled) coordinate system when needed.
                sx = self._display_scale_x
                sy = self._display_scale_y

                if elem.element_type == OverlayType.LINE:
                    s = QPointF(elem.data["start"][0] / sx, elem.data["start"][1] / sy)
                    e = QPointF(elem.data["end"][0] / sx, elem.data["end"][1] / sy)
                    primitives.append(("line", s, e, pen))

                elif elem.element_type == OverlayType.CIRCLE:
                    c = QPointF(elem.data["center"][0] / sx, elem.data["center"][1] / sy)
                    r = elem.data["radius"] / max(sx, sy)
                    fill = elem.data.get("fill", False)
                    b = QBrush(color) if fill else QBrush()
                    primitives.append(("circle", c, r, pen, b))

                elif elem.element_type == OverlayType.POINT:
                    c = QPointF(elem.data["center"][0] / sx, elem.data["center"][1] / sy)
                    r = elem.data.get("radius", 4) / max(sx, sy) / 2
                    p = QPen(color, 2)
                    p.setCosmetic(True)
                    primitives.append(("point", c, int(r), p))

                elif elem.element_type == OverlayType.CONTOUR:
                    pts = elem.data.get("contour", [])
                    if len(pts) >= 2:
                        first = pts[0]
                        if isinstance(first[0], (list, tuple, np.ndarray)):
                            poly = QPolygonF([QPointF(p[0][0] / sx, p[0][1] / sy) for p in pts])
                        else:
                            poly = QPolygonF([QPointF(p[0] / sx, p[1] / sy) for p in pts])
                        poly.append(poly.first())
                        fill = elem.data.get("fill", False)
                        b = QBrush(color) if fill else QBrush()
                        primitives.append(("contour", poly, pen, b))

                elif elem.element_type == OverlayType.POLYGON:
                    pts = elem.data.get("points", [])
                    if len(pts) >= 2:
                        poly = QPolygonF([QPointF(p[0] / sx, p[1] / sy) for p in pts])
                        fill = elem.data.get("fill", False)
                        if fill:
                            fill_color = QColor(color)
                            fill_color.setAlpha(int(255 * elem.data.get("fill_alpha", 0.5)))
                            b = QBrush(fill_color)
                        else:
                            b = QBrush()
                        primitives.append(("contour", poly, pen, b))

                elif elem.element_type == OverlayType.ROI:
                    roi = elem.data["roi"]
                    x, y, w, h = roi[:4]
                    angle = roi[4] if len(roi) > 4 else 0
                    fill = elem.data.get("fill", False)
                    fill_image = elem.data.get("fill_image")
                    if fill_image is not None and fill:
                        rgb = cv2.cvtColor(fill_image, cv2.COLOR_BGR2RGB)
                        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0],
                                      rgb.strides[0], QImage.Format.Format_RGB888)
                        pixmap = QPixmap.fromImage(qimg)
                        alpha = elem.data.get("fill_alpha", 0.6)
                        p = QPen(color, thickness) if thickness > 0 else QPen(Qt.GlobalColor.transparent, 0)
                        p.setCosmetic(True)
                        primitives.append(("roi_image", QRectF(x / sx, y / sy, w / sx, h / sy), angle, pixmap, alpha, p))
                    else:
                        b = QBrush(color) if fill else QBrush()
                        primitives.append(("roi", QRectF(x / sx, y / sy, w / sx, h / sy), angle, pen, b))

                elif elem.element_type == OverlayType.AXES:
                    cx, cy = elem.data["center"]
                    angle_rad = np.deg2rad(elem.data.get("angle", 0))
                    length = elem.data.get("length", 30)
                    c = QPointF(cx / sx, cy / sy)
                    ex = QPointF(cx / sx + length * np.cos(angle_rad) / sx,
                                 cy / sy + length * np.sin(angle_rad) / sy)
                    perp = angle_rad + np.pi / 2
                    ey = QPointF(cx / sx + length * np.cos(perp) / sx,
                                 cy / sy + length * np.sin(perp) / sy)
                    axt = elem.data.get("thickness", 2)
                    p_x = QPen(_bgr_to_qcolor(elem.data.get("color_x", (0, 0, 255))), axt)
                    p_x.setCosmetic(True)
                    p_y = QPen(_bgr_to_qcolor(elem.data.get("color_y", (0, 255, 0))), axt)
                    p_y.setCosmetic(True)
                    primitives.append(("axes", c, ex, ey, p_x, p_y))

                elif elem.element_type == OverlayType.TEXT:
                    text = str(elem.data["text"])
                    pos = QPointF(elem.data["position"][0] / sx, elem.data["position"][1] / sy)
                    fs = elem.data.get("font_scale", 0.5)
                    font = QFont("Consolas", max(8, int(fs * 20)))
                    primitives.append(("text", text, pos, font, pen))
        return primitives

    def mouseMoveEvent(self, event):
        # We handle this in eventFilter for better reliability with ScrollHandDrag
        super().mouseMoveEvent(event)

    def set_roi(self, data):
        if data is None:
            self.roi_rect_item.hide()
            self.roi_rect_item.setEnabled(False)
            return

        if isinstance(data, QRectF):
            rect = self._original_to_display_rect(data)
            angle = 0
            pos = QPointF(0, 0)
        else:
            # data is [x, y, w, h] or [x, y, w, h, a] in original image pixels
            angle = data[4] if len(data) > 4 else 0
            rect = QRectF(0, 0, data[2] / self._display_scale_x,
                          data[3] / self._display_scale_y)
            pos = QPointF(data[0] / self._display_scale_x,
                          data[1] / self._display_scale_y)

        if rect.width() > 0 and rect.height() > 0:
            self.roi_rect_item.setRect(rect)
            self.roi_rect_item.setPos(pos)
            self.roi_rect_item.setRotation(angle)
            self.roi_rect_item.update_handles()
            self.roi_rect_item.setTransformOriginPoint(rect.center())
            self.roi_rect_item.show()
            self.roi_rect_item.setEnabled(True)
        else:
            self.roi_rect_item.hide()
            self.roi_rect_item.setEnabled(False)
            
    def set_roi_editing(self, enabled):
        self._is_editing_roi = enabled
        self.roi_rect_item.setEnabled(enabled)
        if enabled:
            self.roi_rect_item.show()
        else:
            self.roi_rect_item.hide()
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)
        if not enabled:
            self.rubber_band.hide()
            self.origin = QPoint()
            self.roi_rect_item.setSelected(False)

    def set_drawing_new_roi(self, enabled):
        self._is_drawing_new_roi = enabled
        if enabled:
            self.set_roi(None)
            self.origin = QPoint()

    def set_roi_direction_visible(self, visible):
        if visible:
            self._edge_config_count += 1
        else:
            self._edge_config_count = max(0, self._edge_config_count - 1)
        self._roi_show_direction = self._edge_config_count > 0
        if self.roi_rect_item.isVisible():
            self.roi_rect_item.update()

    def fit_in_view(self):
        if not self.pixmap_item.pixmap().isNull():
            # If the view is too small (e.g., not yet shown), wait a bit and retry
            if self.width() < 10 or self.height() < 10:
                QTimer.singleShot(100, self.fit_in_view)
                return
            self.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def set_zoom_level(self, x, y, factor):
        """Programmatically set zoom factor and center on a point."""
        if factor <= 0: return
        self.resetTransform()
        self.scale(factor, factor)
        self.centerOn(float(x), float(y))

    def reset_view(self):
        """Auto-fit the image to the current view size."""
        self.fit_in_view()

    def wheelEvent(self, event):
        zoom_in_factor = 1.25
        zoom_out_factor = 1 / zoom_in_factor
        if event.angleDelta().y() > 0:
            self.scale(zoom_in_factor, zoom_in_factor)
        else:
            self.scale(zoom_out_factor, zoom_out_factor)

    def mousePressEvent(self, event):
        if self._is_drawing_new_roi and event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.pos()
            self.rubber_band.setGeometry(QRect(self.origin, QSize()))
            self.rubber_band.show()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            item = self.itemAt(event.pos())
            if item == self.roi_rect_item or (item and item.parentItem() == self.roi_rect_item):
                super().mousePressEvent(event)
            else:
                orig_pos = self._display_to_original_point(self.mapToScene(event.pos()))
                self.mouse_clicked.emit(int(orig_pos.x()), int(orig_pos.y()))
                super().mousePressEvent(event)
        else:
            orig_pos = self._display_to_original_point(self.mapToScene(event.pos()))
            self.mouse_clicked.emit(int(orig_pos.x()), int(orig_pos.y()))
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._is_drawing_new_roi and not self.origin.isNull():
            self.rubber_band.setGeometry(QRect(self.origin, event.pos()).normalized())
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._is_drawing_new_roi and not self.origin.isNull():
            self.rubber_band.hide()
            rect = self.rubber_band.geometry()
            p1 = self.mapToScene(rect.topLeft())
            p2 = self.mapToScene(rect.bottomRight())
            scene_rect = QRectF(p1, p2).normalized()
            img_rect = self.pixmap_item.boundingRect()
            scene_rect = scene_rect.intersected(img_rect)

            # Convert from on-screen (downsampled) coordinates to original pixels.
            orig_rect = self._display_to_original_rect(scene_rect)
            roi_data = [orig_rect.x(), orig_rect.y(), orig_rect.width(), orig_rect.height(), 0.0]
            self.set_roi(roi_data)
            self.roi_changed.emit(roi_data)
            self.roi_drawing_finished.emit()
            self.origin = QPoint()
            self._is_drawing_new_roi = False
            return
        super().mouseReleaseEvent(event)
        self.viewport().setCursor(Qt.CursorShape.CrossCursor)

    def contextMenuEvent(self, event):
        menu = QMenu(self)
        fit_action = menu.addAction("Fit Image")
        fit_action.triggered.connect(self.fit_in_view)
        save_action = menu.addAction("Save Image...")
        save_action.triggered.connect(self._save_image_dialog)
        save_action.setEnabled(self._current_raw_image is not None)
        menu.exec(event.globalPos())

    def _save_image_dialog(self):
        if self._current_raw_image is None:
            return
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Image", "",
            "Images (*.jpg *.jpeg *.bmp *.png *.tiff *.webp);;All Files (*)"
        )
        if path:
            cv2.imwrite(path, self._current_raw_image)
