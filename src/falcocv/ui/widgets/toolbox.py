from PyQt6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QPushButton,
    QGroupBox,
    QLabel,
    QScrollArea,
    QFrame,
    QSizePolicy,
)
from PyQt6.QtCore import Qt, QMimeData, QPoint
from PyQt6.QtGui import QDrag, QPainter, QColor
from falcocv.ui.style import ALT_BG
from falcocv.core.tool_registry import get_all_tools


#: Preferred toolbox order.  A group registered by a tool but missing here is
#: appended automatically, so a new group never hides its tools.
GROUP_ORDER = [
    "Acquisition",
    "Processing",
    "Inspection",
    "3D",
    "Machine Learning",
    "Logic",
]

#: Display order inside each group.  Unlisted tools follow alphabetically, so a
#: new tool never disappears from its section.
TOOL_ORDER = {
    "Acquisition": ["GigE Camera", "Image Loader", "Depth Range Loader"],
    "Processing": [
        "Filters",
        "Image Calculator",
        "Histogram",
        "Edge Detection",
        "Fixture Overlay",
    ],
    "Inspection": [
        "Blob Detector",
        "Template Match",
        "Shape Match",
        "Barcode Decoder",
        "OCR",
        "Measurement",
        "Line Fit",
        "Circle Fit",
    ],
    "3D": ["Plane Fitter", "Height Blobs 3D"],
    "Machine Learning": ["Classifier", "Anomaly Detector"],
    "Logic": ["Python Script", "Block"],
}



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

        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        self.scroll_area = QScrollArea(self)
        self.scroll_area.setObjectName("ToolboxScroll")
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self.scroll_area.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding,
        )
        self.scroll_area.setMinimumSize(0, 0)

        content = QWidget()
        content.setObjectName("ToolboxContent")
        layout = QVBoxLayout(content)
        layout.setContentsMargins(0, 0, 0, 0)

        if not embedded:
            layout.addWidget(QLabel("Click to Add or Drag to Workspace:"))

        tools = get_all_tools()
        group_names = list(GROUP_ORDER)
        for group_name in sorted({reg.group for reg in tools if reg.group}):
            if group_name not in group_names:
                group_names.append(group_name)

        group_rank = {name: index for index, name in enumerate(group_names)}

        def tool_rank(reg):
            order = TOOL_ORDER.get(reg.group, [])
            try:
                return order.index(reg.default_name)
            except ValueError:
                return len(order)

        tools = sorted(
            tools,
            key=lambda reg: (
                group_rank.get(reg.group, len(group_rank)),
                tool_rank(reg),
                reg.default_name,
            ),
        )

        groups: dict[str, QVBoxLayout] = {}
        for group_name in group_names:
            gb = QGroupBox(group_name)
            groups[group_name] = QVBoxLayout(gb)
            layout.addWidget(gb)

        for reg in tools:
            gl = groups.get(reg.group)
            if gl is None:
                continue
            label = reg.default_name.replace("_", " ")
            btn = ToolButton(label, reg.tool_type)
            btn.clicked.connect(lambda checked, tt=reg.tool_type: self.request_tool(tt))
            gl.addWidget(btn)

        layout.addStretch()
        self.scroll_area.setWidget(content)
        outer_layout.addWidget(self.scroll_area)

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
