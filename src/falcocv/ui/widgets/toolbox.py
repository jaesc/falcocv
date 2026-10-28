from PyQt6.QtWidgets import QWidget, QVBoxLayout, QPushButton, QGroupBox, QLabel
from PyQt6.QtCore import Qt, QMimeData, QPoint
from PyQt6.QtGui import QDrag, QPainter, QColor
from falcocv.ui.style import ALT_BG
from falcocv.core.tool_registry import get_all_tools



class ToolButton(QPushButton):
    def __init__(self, text, tool_type, parent=None):
        super().__init__(text, parent)
        self.tool_type = tool_type
        self.mouse_pos = QPoint()
        self._dragging = False

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.mouse_pos = event.pos()
            self._dragging = False
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            if (event.pos() - self.mouse_pos).manhattanLength() < 10:
                return

            self._dragging = True
            drag = QDrag(self)
            mime = QMimeData()
            mime.setText(self.tool_type)
            drag.setMimeData(mime)
            drag.exec(Qt.DropAction.CopyAction)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self._dragging:
            self._dragging = False
            event.accept()
            return
        super().mouseReleaseEvent(event)


class Toolbox(QWidget):
    _active_editor = None

    def __init__(self, parent=None, embedded=False):
        super().__init__(parent)
        if not embedded:
            self.setWindowTitle("Toolbox")
            self.setWindowFlags(Qt.WindowType.Window | Qt.WindowType.WindowStaysOnTopHint)
            self.resize(250, 300)

        layout = QVBoxLayout(self)

        if not embedded:
            layout.addWidget(QLabel("Click to Add or Drag to Workspace:"))

        groups: dict[str, QVBoxLayout] = {}
        for group_name in ["Acquisition", "Preprocess", "Vision Tools", "Machine Learning", "3D", "Logic"]:
            gb = QGroupBox(group_name)
            groups[group_name] = QVBoxLayout(gb)
            layout.addWidget(gb)

        for reg in get_all_tools():
            gl = groups.get(reg.group)
            if gl is None:
                continue
            label = reg.default_name.replace("_", " ")
            btn = ToolButton(label, reg.tool_type)
            btn.clicked.connect(lambda checked, tt=reg.tool_type: self.request_tool(tt))
            gl.addWidget(btn)

        layout.addStretch()

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(ALT_BG))
        p.end()

    def request_tool(self, tool_type):
        if Toolbox._active_editor:
            Toolbox._active_editor.tree.tool_dropped.emit(tool_type, -1)

    @staticmethod
    def set_active_editor(editor):
        Toolbox._active_editor = editor
