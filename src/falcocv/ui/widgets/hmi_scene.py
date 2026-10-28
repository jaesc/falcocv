from enum import IntEnum

from PyQt6.QtWidgets import QGraphicsScene, QGraphicsRectItem, QGraphicsView, QMenu
from PyQt6.QtCore import Qt, QRectF, QPointF, pyqtSignal
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QCursor,
    QAction, QPainterPath,
)

from falcocv.core.hmi import HMIProject


GRID = 24
HANDLE = 7


class HandleIdx(IntEnum):
    NONE = -1
    TOP_LEFT = 0
    TOP = 1
    TOP_RIGHT = 2
    RIGHT = 3
    BOTTOM_RIGHT = 4
    BOTTOM = 5
    BOTTOM_LEFT = 6
    LEFT = 7


HANDLE_CURSORS = {
    HandleIdx.TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
    HandleIdx.TOP: Qt.CursorShape.SizeVerCursor,
    HandleIdx.TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
    HandleIdx.RIGHT: Qt.CursorShape.SizeHorCursor,
    HandleIdx.BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
    HandleIdx.BOTTOM: Qt.CursorShape.SizeVerCursor,
    HandleIdx.BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
    HandleIdx.LEFT: Qt.CursorShape.SizeHorCursor,
}

TYPE_COLORS = {
    "ImageViewer": QColor(0x4e, 0xc9, 0xb0),
    "StatusIndicator": QColor(0xce, 0x91, 0x78),
    "LedIndicator": QColor(0xff, 0xa7, 0x26),
    "MeasureLabel": QColor(0x56, 0x9c, 0xd6),
    "TextLabel": QColor(0x6a, 0x6a, 0x6a),
    "LogPanel": QColor(0xd7, 0xba, 0x7d),
}


def _snap(v):
    return round(v / GRID) * GRID


def _clamped(v, lo, hi):
    return max(lo, min(hi, v))


class HMIWidgetItem(QGraphicsRectItem):
    def __init__(self, data: dict, project: HMIProject, parent=None):
        super().__init__(parent)
        self._data = data
        self._project = project
        self._handle = HandleIdx.NONE
        self._drag_origin = QPointF()
        self._rebuilding = False

        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self.setFlag(self.GraphicsItemFlag.ItemIsSelectable, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsGeometryChanges, True)
        self.setFlag(self.GraphicsItemFlag.ItemSendsScenePositionChanges, True)
        self.setAcceptHoverEvents(True)

        self._rebuild()

    @property
    def widget_data(self):
        return self._data

    def _notify_modified(self):
        scene = self.scene()
        if isinstance(scene, HMIWidgetScene):
            scene.project_modified.emit()

    def _rebuild(self):
        self._rebuilding = True
        x = self._data["x"] * GRID
        y = self._data["y"] * GRID
        w = self._data["w"] * GRID
        h = self._data["h"] * GRID
        self.setZValue(self._data.get("z", 0))
        self.setPos(x, y)
        self.setRect(0, 0, w, h)
        self._rebuilding = False

    def sync_to_data(self):
        if self._rebuilding:
            return
        rect = self.rect()
        pos = self.pos()
        self._data["x"] = round(pos.x() / GRID)
        self._data["y"] = round(pos.y() / GRID)
        self._data["w"] = max(1, round(rect.width() / GRID))
        self._data["h"] = max(1, round(rect.height() / GRID))
        self._data["z"] = int(self.zValue())

    def itemChange(self, change, value):
        if change == self.GraphicsItemChange.ItemPositionChange:
            value.setX(_snap(value.x()))
            value.setY(_snap(value.y()))
            scene = self.scene()
            if scene:
                sr = scene.sceneRect()
                r = self.rect()
                value.setX(_clamped(value.x(), 0, sr.width() - r.width()))
                value.setY(_clamped(value.y(), 0, sr.height() - r.height()))
            return value
        if change == self.GraphicsItemChange.ItemPositionHasChanged:
            self.sync_to_data()
        return super().itemChange(change, value)

    def paint(self, p: QPainter, option, widget=None):
        rect = self.rect()
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        wt = self._data.get("type", "ImageViewer")
        wname = self._data.get("name", wt)
        label = self._data.get("label", "")
        slot = self._data.get("slot", "")
        output = self._data.get("output", "")
        ac = TYPE_COLORS.get(wt, QColor(0x88, 0x88, 0x88))

        display_name = wname

        # background
        bg = ac.darker(200)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(rect, 4, 4)

        # colored left bar
        bar = QRectF(rect.x(), rect.y(), 4, rect.height())
        p.setBrush(QBrush(ac))
        p.drawRoundedRect(bar, 2, 2)

        # selection highlight
        if self.isSelected():
            p.setPen(QPen(ac.lighter(150), 2))
            p.setBrush(Qt.BrushStyle.NoBrush)
            p.drawRoundedRect(rect.adjusted(1, 1, -1, -1), 4, 4)

        # type label with index
        p.setPen(QColor(0xcc, 0xcc, 0xcc))
        f = QFont("Segoe UI", 8, QFont.Weight.Bold)
        p.setFont(f)
        type_rect = QRectF(rect.x() + 8, rect.y() + 4, rect.width() - 16, 16)
        p.drawText(type_rect, Qt.AlignmentFlag.AlignLeft, display_name)

        # extra info: slot, output, field
        info_lines = []
        if slot:
            info_lines.append(f"Slot: {slot}")
        if wt in ("ImageViewer", "MeasureLabel") and output:
            info_lines.append(f"Out: {output}")
        field = self._data.get("field", "")
        if wt == "MeasureLabel" and field:
            info_lines.append(f"Field: {field}")
        if label:
            info_lines.append(f"Label: {label}")

        if info_lines:
            p.setPen(QColor(0x88, 0x88, 0x88))
            f2 = QFont("Segoe UI", 7)
            p.setFont(f2)
            y_off = 20
            for line in info_lines:
                lr = QRectF(rect.x() + 8, rect.y() + y_off, rect.width() - 16, 12)
                p.drawText(lr, Qt.AlignmentFlag.AlignLeft, line)
                y_off += 12

        if self.isSelected():
            self._paint_handles(p)

    def _paint_handles(self, p: QPainter):
        rect = self.rect()
        r = rect.getRect()  # (x, y, w, h)
        hs = HANDLE

        handles = [
            QRectF(r[0] - hs, r[1] - hs, hs * 2, hs * 2),           # TL
            QRectF(r[0] + r[2] / 2 - hs, r[1] - hs, hs * 2, hs * 2), # T
            QRectF(r[0] + r[2] - hs, r[1] - hs, hs * 2, hs * 2),     # TR
            QRectF(r[0] + r[2] - hs, r[1] + r[3] / 2 - hs, hs * 2, hs * 2), # R
            QRectF(r[0] + r[2] - hs, r[1] + r[3] - hs, hs * 2, hs * 2), # BR
            QRectF(r[0] + r[2] / 2 - hs, r[1] + r[3] - hs, hs * 2, hs * 2), # B
            QRectF(r[0] - hs, r[1] + r[3] - hs, hs * 2, hs * 2),     # BL
            QRectF(r[0] - hs, r[1] + r[3] / 2 - hs, hs * 2, hs * 2), # L
        ]

        p.setBrush(QBrush(QColor(0xff, 0xff, 0xff)))
        p.setPen(QPen(QColor(0x00, 0x7a, 0xcc), 1))
        for h in handles:
            p.drawRect(h)

    def _handle_at(self, pos: QPointF) -> HandleIdx:
        rect = self.rect()
        if rect.isEmpty():
            return HandleIdx.NONE
        hw = self.mapToScene(pos) - self.scenePos()

        r = rect.getRect()
        hs = HANDLE * 1.5
        cx, cy = r[0] + r[2] / 2, r[1] + r[3] / 2

        if abs(hw.x() - r[0]) <= hs:
            if abs(hw.y() - r[1]) <= hs:
                return HandleIdx.TOP_LEFT
            if abs(hw.y() - (r[1] + r[3])) <= hs:
                return HandleIdx.BOTTOM_LEFT
            if abs(hw.y() - cy) <= hs:
                return HandleIdx.LEFT
        if abs(hw.x() - (r[0] + r[2])) <= hs:
            if abs(hw.y() - r[1]) <= hs:
                return HandleIdx.TOP_RIGHT
            if abs(hw.y() - (r[1] + r[3])) <= hs:
                return HandleIdx.BOTTOM_RIGHT
            if abs(hw.y() - cy) <= hs:
                return HandleIdx.RIGHT
        if abs(hw.y() - r[1]) <= hs and abs(hw.x() - cx) <= hs:
            return HandleIdx.TOP
        if abs(hw.y() - (r[1] + r[3])) <= hs and abs(hw.x() - cx) <= hs:
            return HandleIdx.BOTTOM

        return HandleIdx.NONE

    def hoverMoveEvent(self, event):
        if self.isSelected() and not self._handle != HandleIdx.NONE:
            h = self._handle_at(event.pos())
            if h != HandleIdx.NONE:
                self.setCursor(HANDLE_CURSORS.get(h, Qt.CursorShape.ArrowCursor))
            else:
                self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        super().hoverMoveEvent(event)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.isSelected():
            h = self._handle_at(event.pos())
            if h != HandleIdx.NONE:
                self._handle = h
                self._drag_origin = event.scenePos()
                self.setFlag(self.GraphicsItemFlag.ItemIsMovable, False)
                event.accept()
                return
        self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
        self._drag_origin = self.pos()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._handle != HandleIdx.NONE:
            self._resize(event.scenePos())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._handle != HandleIdx.NONE:
            self._handle = HandleIdx.NONE
            self.setFlag(self.GraphicsItemFlag.ItemIsMovable, True)
            self.sync_to_data()
            self._notify_modified()
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if self.pos() != self._drag_origin:
            self._notify_modified()

    def _resize(self, mouse_scene: QPointF):
        h = self._handle
        mx = _snap(mouse_scene.x())
        my = _snap(mouse_scene.y())

        pos = self.pos()
        rect = self.rect()
        l = pos.x()
        t = pos.y()
        r = l + rect.width()
        b = t + rect.height()

        if h == HandleIdx.TOP_LEFT:
            l = min(r - GRID, mx)
            t = min(b - GRID, my)
        elif h == HandleIdx.TOP:
            t = min(b - GRID, my)
        elif h == HandleIdx.TOP_RIGHT:
            t = min(b - GRID, my)
            r = max(l + GRID, mx)
        elif h == HandleIdx.RIGHT:
            r = max(l + GRID, mx)
        elif h == HandleIdx.BOTTOM_RIGHT:
            r = max(l + GRID, mx)
            b = max(t + GRID, my)
        elif h == HandleIdx.BOTTOM:
            b = max(t + GRID, my)
        elif h == HandleIdx.BOTTOM_LEFT:
            b = max(t + GRID, my)
            l = min(r - GRID, mx)
        elif h == HandleIdx.LEFT:
            l = min(r - GRID, mx)

        cols = self._project.layout.get("cols", 50)
        rows = self._project.layout.get("rows", 50)
        l = _clamped(l, 0, cols * GRID - (r - l))
        t = _clamped(t, 0, rows * GRID - (b - t))
        r = _clamped(r, l + GRID, cols * GRID)
        b = _clamped(b, t + GRID, rows * GRID)

        self.setPos(QPointF(l, t))
        self.setRect(QRectF(0, 0, r - l, b - t))

    def contextMenuEvent(self, event):
        menu = QMenu()

        act_top = QAction("Bring to Front", menu)
        act_top.triggered.connect(self._bring_to_front)
        menu.addAction(act_top)

        act_fwd = QAction("Bring forward", menu)
        act_fwd.triggered.connect(self._bring_forward)
        menu.addAction(act_fwd)

        act_bwd = QAction("Send backward", menu)
        act_bwd.triggered.connect(self._send_backward)
        menu.addAction(act_bwd)

        act_bottom = QAction("Send to Back", menu)
        act_bottom.triggered.connect(self._send_to_back)
        menu.addAction(act_bottom)

        menu.addSeparator()

        act_edit = QAction("Edit properties", menu)
        act_edit.triggered.connect(lambda: self.scene().on_edit_widget(self._data["id"]))
        menu.addAction(act_edit)

        menu.exec(event.screenPos())

    def _bring_to_front(self):
        scene = self.scene()
        if scene is None:
            return
        all_z = [it.zValue() for it in scene.items() if isinstance(it, HMIWidgetItem)]
        new_z = max(all_z, default=-1) + 1
        self.setZValue(new_z)
        self.sync_to_data()
        self._notify_modified()
        scene.update()

    def _bring_forward(self):
        scene = self.scene()
        if scene is None:
            return
        items = [it for it in scene.items() if isinstance(it, HMIWidgetItem) and it is not self]
        my_z = self.zValue()
        above = [(it.zValue(), it) for it in items if it.zValue() > my_z]
        if not above:
            self.setZValue(my_z + 1)
        else:
            above.sort(key=lambda x: x[0])
            target_z = above[0][0]
            self.setZValue(target_z)
            above[0][1].setZValue(my_z)
            above[0][1].sync_to_data()
        self.sync_to_data()
        self._notify_modified()
        scene.update()

    def _send_backward(self):
        scene = self.scene()
        if scene is None:
            return
        items = [it for it in scene.items() if isinstance(it, HMIWidgetItem) and it is not self]
        my_z = self.zValue()
        below = [(it.zValue(), it) for it in items if it.zValue() < my_z]
        if not below:
            self.setZValue(my_z - 1)
        else:
            below.sort(key=lambda x: x[0], reverse=True)
            target_z = below[0][0]
            self.setZValue(target_z)
            below[0][1].setZValue(my_z)
            below[0][1].sync_to_data()
        self.sync_to_data()
        self._notify_modified()
        scene.update()

    def _send_to_back(self):
        scene = self.scene()
        if scene is None:
            return
        all_z = [it.zValue() for it in scene.items() if isinstance(it, HMIWidgetItem)]
        new_z = min(all_z, default=0) - 1
        self.setZValue(new_z)
        self.sync_to_data()
        self._notify_modified()
        scene.update()


class HMIWidgetScene(QGraphicsScene):
    widget_double_clicked = pyqtSignal(str)
    project_modified = pyqtSignal()

    def __init__(self, project: HMIProject, parent=None):
        super().__init__(parent)
        self.project = project
        self._items: dict[str, HMIWidgetItem] = {}
        self._show_grid = True
        rows = project.layout.get("rows", 50)
        cols = project.layout.get("cols", 50)
        self.setSceneRect(0, 0, cols * GRID, rows * GRID)

    def on_edit_widget(self, widget_id: str):
        self.widget_double_clicked.emit(widget_id)

    def add_widget(self, data: dict) -> HMIWidgetItem:
        item = HMIWidgetItem(data, self.project)
        self.addItem(item)
        self._items[data["id"]] = item
        return item

    def remove_widget(self, widget_id: str):
        item = self._items.pop(widget_id, None)
        if item:
            self.removeItem(item)

    def rebuild_from_project(self):
        for item in list(self._items.values()):
            self.removeItem(item)
        self._items.clear()
        for w in self.project.layout.get("widgets", []):
            item = HMIWidgetItem(w, self.project)
            self.addItem(item)
            self._items[w["id"]] = item

    def selected_widget_id(self):
        sel = self.selectedItems()
        if sel:
            item = sel[0]
            if isinstance(item, HMIWidgetItem):
                return item.widget_data["id"]
        return None

    def sync_all(self):
        for item in self._items.values():
            item.sync_to_data()

    def drawBackground(self, p: QPainter, rect: QRectF):
        p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
        p.fillRect(rect, QColor(0x1a, 0x1a, 0x1a))

    def drawForeground(self, p: QPainter, rect: QRectF):
        if self._show_grid:
            p.setRenderHint(QPainter.RenderHint.Antialiasing, False)
            p.setPen(QPen(QColor(0x60, 0x60, 0x60), 1.0))
            s = self.sceneRect()
            left = max(s.left(), rect.left() - GRID)
            top = max(s.top(), rect.top() - GRID)
            right = min(rect.right(), s.right())
            bottom = min(rect.bottom(), s.bottom())
            x = _snap(left)
            while x <= right:
                p.drawLine(QPointF(x, top), QPointF(x, bottom))
                x += GRID
            y = _snap(top)
            while y <= bottom:
                p.drawLine(QPointF(left, y), QPointF(right, y))
                y += GRID
        s = self.sceneRect()
        p.setPen(QPen(QColor(0x56, 0x9c, 0xd6), 2))
        p.setBrush(Qt.BrushStyle.NoBrush)
        p.drawRect(s)

        p.setPen(QPen(QColor(0x80, 0x80, 0x80), 1, Qt.PenStyle.DashLine))
        cx = s.width() / 2
        cy = s.height() / 2
        p.drawLine(QPointF(cx, s.top()), QPointF(cx, s.bottom()))
        p.drawLine(QPointF(s.left(), cy), QPointF(s.right(), cy))

    def mouseDoubleClickEvent(self, event):
        items = self.items(event.scenePos())
        for item in items:
            if isinstance(item, HMIWidgetItem):
                self.widget_double_clicked.emit(item.widget_data["id"])
                event.accept()
                return
        super().mouseDoubleClickEvent(event)


class HMIWidgetView(QGraphicsView):
    def __init__(self, scene: HMIWidgetScene, parent=None):
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setBackgroundBrush(QBrush(QColor(0x1a, 0x1a, 0x1a)))
        self.setFrameShape(self.Shape.NoFrame)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        s = self.scene()
        if s:
            self.fitInView(s.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
