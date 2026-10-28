from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont

from falcocv.core.hmi import HMIProject


class HMIPreviewWidget(QWidget):
    def __init__(self, project: HMIProject, parent=None):
        super().__init__(parent)
        self.project = project
        self.setMinimumSize(400, 300)

    def set_project(self, project: HMIProject):
        self.project = project
        self.update()

    def paintEvent(self, event):
        if not self.project:
            return
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing)

        layout = self.project.layout
        rows = layout.get('rows', 20)
        cols = layout.get('cols', 30)
        widgets = layout.get('widgets', [])

        if rows == 0 or cols == 0:
            p.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, 'No layout')
            p.end()
            return

        cell_w = self.width() / cols
        cell_h = self.height() / rows

        p.fillRect(self.rect(), QColor('#1e1e1e'))

        sorted_widgets = sorted(widgets, key=lambda w: w.get('z', 0))
        for widget in sorted_widgets:
            x = widget.get('x', 0)
            y = widget.get('y', 0)
            w = widget.get('w', 4)
            h = widget.get('h', 3)
            rect = QRectF(x * cell_w, y * cell_h, w * cell_w, h * cell_h)
            wt = widget.get('type', 'None')
            self._draw_cell(p, rect, wt, widget)

        p.end()

    def _draw_cell(self, p: QPainter, rect: QRectF, widget_type: str, cell: dict | None):
        bg = QColor('#2d2d2d')
        border = QColor('#444')
        p.setPen(QPen(border, 1))
        p.setBrush(QBrush(bg))
        p.drawRoundedRect(rect, 4, 4)

        if widget_type == 'None':
            p.setPen(QColor('#555'))
            p.drawText(rect, Qt.AlignmentFlag.AlignCenter, '\u2014')
            return

        label = cell.get('label', '') if cell else ''
        slot = cell.get('slot', '') if cell else ''
        output = cell.get('output', '') if cell else ''
        field = cell.get('field', '') if cell else ''

        inner = rect.adjusted(8, 8, -8, -8)

        if widget_type == 'ImageViewer':
            self._draw_image_viewer(p, inner, slot, output)
        elif widget_type == 'MeasureLabel':
            self._draw_measure_label(p, inner, label, cell)
        elif widget_type == 'StatusIndicator':
            self._draw_status_indicator(p, inner, label, cell and cell.get('passed', True), cell)
        elif widget_type == 'LedIndicator':
            self._draw_led_indicator(p, inner, cell)
        elif widget_type == 'TextLabel':
            self._draw_text_label(p, inner, label, cell)
        elif widget_type == 'LogPanel':
            self._draw_log_panel(p, inner, cell)

        p.setPen(QColor('#888'))
        small = QFont('Consolas')
        small.setPixelSize(7)
        p.setFont(small)
        tw = p.fontMetrics().horizontalAdvance(widget_type)
        p.drawText(
            QRectF(rect.right() - tw - 4, rect.bottom() - 12, tw + 4, 10),
            Qt.AlignmentFlag.AlignLeft, widget_type,
        )

    def _draw_image_viewer(self, p: QPainter, rect: QRectF, slot: str, output: str):
        inner = rect.adjusted(4, 4, -4, -4)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor('#1a1a1a')))
        p.drawRoundedRect(inner, 2, 2)
        p.setPen(QPen(QColor('#2a2a2a'), 1))
        for i in range(0, int(inner.width()), 20):
            p.drawLine(QPointF(inner.x() + i, inner.y()), QPointF(inner.x() + i, inner.bottom()))
        for i in range(0, int(inner.height()), 20):
            p.drawLine(QPointF(inner.x(), inner.y() + i), QPointF(inner.right(), inner.y() + i))
        p.setPen(QColor('#888'))
        f = QFont('Consolas')
        f.setPixelSize(8)
        p.setFont(f)
        txt = f'[{slot}] {output}' if slot else 'No image'
        p.drawText(inner, Qt.AlignmentFlag.AlignCenter, txt)

    def _draw_measure_label(self, p: QPainter, rect: QRectF, label: str, cell: dict | None = None):
        fs = cell.get("font_size", 14) if cell else 14
        ha = cell.get("halign", "center") if cell else "center"
        ll = cell.get("label_layout", "stacked") if cell else "stacked"
        align_flag = Qt.AlignmentFlag.AlignLeft
        if ha == "center":
            align_flag = Qt.AlignmentFlag.AlignHCenter
        elif ha == "right":
            align_flag = Qt.AlignmentFlag.AlignRight

        if ll == "inline" and label:
            p.setPen(QColor('#ccc'))
            f = QFont('Consolas')
            f.setPixelSize(max(8, fs - 4))
            p.setFont(f)
            label_w = p.fontMetrics().horizontalAdvance(label + ": ") + 4
            lbl_rect = QRectF(rect.x(), rect.y(), label_w, rect.height())
            p.drawText(lbl_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, label + ":")
            val_rect = QRectF(rect.x() + label_w, rect.y(), rect.width() - label_w, rect.height())
            p.setPen(QColor('#4ec9b0'))
            big = QFont('Consolas')
            big.setWeight(QFont.Weight.Bold)
            big.setPixelSize(fs)
            p.setFont(big)
            p.drawText(val_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft, '123.45')
            return

        lbl_h = max(8, fs - 4) + 4 if label else 0
        val_h = fs + 4
        total_h = lbl_h + val_h
        start_y = rect.y() + (rect.height() - total_h) / 2

        if label:
            p.setPen(QColor('#ccc'))
            f = QFont('Consolas')
            f.setPixelSize(max(8, fs - 4))
            p.setFont(f)
            lbl_rect = QRectF(rect.x(), start_y, rect.width(), max(8, fs - 4))
            p.drawText(lbl_rect, align_flag, label)

        val_rect = QRectF(rect.x(), start_y + lbl_h, rect.width(), fs)
        value_color = '#c62828' if cell and cell.get('status') == 'fail' else '#4ec9b0'
        p.setPen(QColor(value_color))
        big = QFont('Consolas')
        big.setWeight(QFont.Weight.Bold)
        big.setPixelSize(fs)
        p.setFont(big)
        p.drawText(val_rect, align_flag, '123.45')

    def _draw_led_indicator(self, p: QPainter, rect: QRectF, cell: dict | None = None):
        label = cell.get("label", "") if cell else ""
        status = cell.get("status", "idle") if cell else "idle"
        colors = {
            "pass": cell.get("color_pass", "#4ec9b0") if cell else "#4ec9b0",
            "fail": cell.get("color_fail", "#c62828") if cell else "#c62828",
            "pending": cell.get("color_pending", "#ffa726") if cell else "#ffa726",
        }
        color = QColor(colors.get(status, "#888888"))
        circle_rect = rect.adjusted(0, 0, 0, -16 if label else 0)
        d = min(circle_rect.width(), circle_rect.height())
        cx = circle_rect.x() + (circle_rect.width() - d) / 2
        cy = circle_rect.y() + (circle_rect.height() - d) / 2
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        p.drawEllipse(QPointF(cx + d / 2, cy + d / 2), d / 2, d / 2)
        if label:
            p.setPen(QColor('#888'))
            f = QFont('Consolas')
            f.setPixelSize(8)
            p.setFont(f)
            lbl_rect = QRectF(rect.x(), rect.bottom() - 14, rect.width(), 12)
            p.drawText(lbl_rect, Qt.AlignmentFlag.AlignCenter, label)

    def _draw_status_indicator(self, p: QPainter, rect: QRectF, label: str, passed: bool, cell: dict | None = None):
        status = "pass" if passed else "fail"
        default_colors = {"pass": "#2e7d32", "fail": "#c62828", "pending": "#c1c100", "idle": "#888888"}
        hex_color = default_colors.get(status, "#888888")
        if cell:
            hex_color = cell.get(f"color_{status}", hex_color)
        color = QColor(hex_color)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(color))
        p.drawRoundedRect(rect, 6, 6)
        p.setPen(QColor('#fff'))
        big = QFont('Consolas')
        big.setWeight(QFont.Weight.Bold)
        big.setPixelSize(14)
        p.setFont(big)
        txt = 'PASS' if passed else 'FAIL'
        p.drawText(rect, Qt.AlignmentFlag.AlignCenter, txt)
        if label:
            small = QFont('Consolas')
            small.setPixelSize(8)
            p.setFont(small)
            p.drawText(rect.adjusted(0, 0, 0, -20), Qt.AlignmentFlag.AlignCenter, label)

    def _draw_text_label(self, p: QPainter, rect: QRectF, text: str, cell: dict | None = None):
        bg = cell.get("bg_color", [45, 45, 45]) if cell else [45, 45, 45]
        tc = cell.get("text_color", [204, 204, 204]) if cell else [204, 204, 204]
        fs = cell.get("font_size", 14) if cell else 14
        ha = cell.get("halign", "center") if cell else "center"
        va = cell.get("valign", "center") if cell else "center"
        p.fillRect(rect, QColor(*bg))
        p.setPen(QColor(*tc))
        f = QFont('Segoe UI')
        f.setPixelSize(fs)
        p.setFont(f)
        align = Qt.AlignmentFlag.AlignLeft
        if ha == "center":
            align = Qt.AlignmentFlag.AlignHCenter
        elif ha == "right":
            align = Qt.AlignmentFlag.AlignRight
        if va == "top":
            align |= Qt.AlignmentFlag.AlignTop
        elif va == "bottom":
            align |= Qt.AlignmentFlag.AlignBottom
        else:
            align |= Qt.AlignmentFlag.AlignVCenter
        p.drawText(rect, align, text or 'Label')

    def _draw_log_panel(self, p: QPainter, rect: QRectF, cell: dict | None = None):
        inner = rect.adjusted(2, 2, -2, -2)
        p.setPen(Qt.PenStyle.NoPen)
        p.setBrush(QBrush(QColor('#1e1e1e')))
        p.drawRoundedRect(inner, 2, 2)
        logs = [
            '[12:34:56] Starting job...',
            '[12:34:57] Result: PASS',
            '[12:34:58] Measurement: 12.34',
        ]
        fs = cell.get("font_size", 14) if cell else 14
        f = QFont('Consolas')
        f.setPixelSize(fs)
        p.setFont(f)
        p.setPen(QColor('#888'))
        y = inner.y() + fs + 2
        for line in logs:
            p.drawText(QPointF(inner.x() + 4, y), line)
            y += fs + 4
