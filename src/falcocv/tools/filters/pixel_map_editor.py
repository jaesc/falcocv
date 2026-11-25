"""Interactive editor for the Filters pixel-map transfer curve."""

from __future__ import annotations

import math

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt, pyqtSignal
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from falcocv.ui.style import BORDER, HIGHLIGHT, PANEL_BG, SUCCESS, TEXT, TEXT_MUTED

from .pixel_map import (
    apply_range,
    fit_curve_to_range,
    identity_points,
    input_domain,
    normalize_curve,
    output_range,
    validate_curve,
)

_HISTOGRAM_BINS = 256
_HISTOGRAM_MAX_SAMPLES = 2_000_000


def _fmt(value: float, integer: bool) -> str:
    if integer:
        return f"{int(round(value))}"
    return f"{value:.3f}".rstrip("0").rstrip(".")


class PixelMapPlot(QWidget):
    """Draws the transfer curve and lets the user drag its control points."""

    curve_changed = pyqtSignal()
    selection_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(380, 260)
        self.setMouseTracking(True)
        self._domain = (0.0, 255.0)
        self._out_range = (0.0, 255.0)
        self._points: list[list[float]] = identity_points(self._domain)
        self._histogram = None
        self._integer = True
        self._selected = -1
        self._drag = None
        self._plot_rect = QRectF()

    # -- data ---------------------------------------------------------
    def set_curve(self, points, domain, out_range, integer: bool):
        self._points = [list(point) for point in points]
        self._domain = (float(domain[0]), float(domain[1]))
        self._out_range = (float(out_range[0]), float(out_range[1]))
        self._integer = bool(integer)
        self._selected = -1
        self.update()

    def set_histogram(self, counts):
        self._histogram = None if counts is None else np.asarray(counts)
        self.update()

    def points(self):
        return [list(point) for point in self._points]

    def selected(self) -> int:
        return self._selected

    def select(self, index: int):
        self._selected = index if 0 <= index < len(self._points) else -1
        self.update()
        self.selection_changed.emit(self._selected)

    # -- geometry -----------------------------------------------------
    def _rect(self) -> QRectF:
        return QRectF(
            52.0,
            14.0,
            max(20.0, self.width() - 66.0),
            max(20.0, self.height() - 54.0),
        )

    def _to_px(self, x: float, y: float) -> QPointF:
        rect = self._rect()
        d0, d1 = self._domain
        o0, o1 = self._out_range
        fx = (float(x) - d0) / max(d1 - d0, 1e-12)
        fy = (float(y) - o0) / max(o1 - o0, 1e-12)
        return QPointF(
            rect.left() + fx * rect.width(),
            rect.bottom() - fy * rect.height(),
        )

    def _to_data(self, pos) -> tuple[float, float]:
        rect = self._rect()
        d0, d1 = self._domain
        o0, o1 = self._out_range
        fx = (pos.x() - rect.left()) / max(rect.width(), 1.0)
        fy = (rect.bottom() - pos.y()) / max(rect.height(), 1.0)
        return d0 + fx * (d1 - d0), o0 + fy * (o1 - o0)

    def _hit_test(self, pos) -> int:
        best = -1
        best_distance = 11.0
        for index, point in enumerate(self._points):
            pixel = self._to_px(point[0], point[1])
            distance = math.hypot(
                pixel.x() - pos.x(), pixel.y() - pos.y()
            )
            if distance < best_distance:
                best_distance = distance
                best = index
        return best

    def _segment_index(self, x: float) -> int | None:
        for index, (left, right) in enumerate(
            zip(self._points, self._points[1:])
        ):
            if left[0] < x < right[0]:
                return index
        return None

    # -- painting -----------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(PANEL_BG))
        rect = self._rect()
        self._plot_rect = rect

        painter.setPen(QPen(QColor(BORDER), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect)

        d0, d1 = self._domain
        o0, o1 = self._out_range

        # grid
        painter.setPen(QPen(QColor(BORDER), 1))
        for step in range(1, 5):
            fx = rect.left() + rect.width() * step / 5.0
            fy = rect.top() + rect.height() * step / 5.0
            painter.drawLine(QPointF(fx, rect.top()), QPointF(fx, rect.bottom()))
            painter.drawLine(QPointF(rect.left(), fy), QPointF(rect.right(), fy))

        # input histogram
        if self._histogram is not None and self._histogram.size:
            counts = np.asarray(self._histogram, dtype=np.float64)
            peak = float(counts.max())
            if peak > 0.0:
                path = QPainterPath()
                path.moveTo(rect.left(), rect.bottom())
                count = counts.size
                for index, value in enumerate(counts):
                    fx = index / max(count - 1, 1)
                    height = (value / peak) * rect.height() * 0.85
                    path.lineTo(
                        rect.left() + fx * rect.width(),
                        rect.bottom() - height,
                    )
                path.lineTo(rect.right(), rect.bottom())
                path.closeSubpath()
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(61, 109, 173, 38))
                painter.drawPath(path)

        # identity reference
        identity_pen = QPen(QColor(TEXT_MUTED), 1, Qt.PenStyle.DashLine)
        painter.setPen(identity_pen)
        painter.drawLine(self._to_px(d0, d0), self._to_px(d1, d1))

        # curve
        painter.setPen(QPen(QColor(SUCCESS), 2))
        path = QPainterPath()
        first = self._points[0]
        path.moveTo(self._to_px(first[0], first[1]))
        for point in self._points[1:]:
            path.lineTo(self._to_px(point[0], point[1]))
        painter.drawPath(path)

        # handles
        for index, point in enumerate(self._points):
            center = self._to_px(point[0], point[1])
            selected = index == self._selected
            radius = 7.0 if selected else 5.0
            painter.setBrush(
                QColor(HIGHLIGHT) if selected else QColor(PANEL_BG)
            )
            painter.setPen(
                QPen(QColor(HIGHLIGHT if selected else TEXT), 2)
            )
            painter.drawEllipse(center, radius, radius)

        # labels
        painter.setPen(QColor(TEXT_MUTED))
        painter.drawText(
            QPointF(6.0, rect.top() + 4.0), _fmt(o1, self._integer)
        )
        painter.drawText(
            QPointF(6.0, rect.bottom()), _fmt(o0, self._integer)
        )
        painter.drawText(
            QPointF(rect.left(), rect.bottom() + 18.0), _fmt(d0, self._integer)
        )
        painter.drawText(
            QPointF(rect.right() - 30.0, rect.bottom() + 18.0),
            _fmt(d1, self._integer),
        )

        if 0 <= self._selected < len(self._points):
            point = self._points[self._selected]
            center = self._to_px(point[0], point[1])
            painter.setPen(QColor(TEXT))
            painter.drawText(
                QPointF(center.x() + 10.0, center.y() - 10.0),
                f"{_fmt(point[0], self._integer)} → "
                f"{_fmt(point[1], self._integer)}",
            )

    # -- mouse --------------------------------------------------------
    def mousePressEvent(self, event):
        index = self._hit_test(event.position())
        if event.button() == Qt.MouseButton.LeftButton:
            if index >= 0:
                self.select(index)
                self._drag = {
                    "index": index,
                    "pos": event.position(),
                    "points": [list(point) for point in self._points],
                }
                event.accept()
                return
        elif event.button() == Qt.MouseButton.RightButton:
            if 0 < index < len(self._points) - 1:
                del self._points[index]
                self._points = normalize_curve(self._points, self._domain)
                self._selected = -1
                self.selection_changed.emit(-1)
                self.curve_changed.emit()
                self.update()
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._drag is None:
            super().mouseMoveEvent(event)
            return
        index = self._drag["index"]
        origin = self._drag["points"][index]
        start = self._drag["pos"]
        d0, d1 = self._domain
        o0, o1 = self._out_range
        rect = self._rect()
        dx = (event.position().x() - start.x()) / max(rect.width(), 1.0)
        dy = (event.position().y() - start.y()) / max(rect.height(), 1.0)
        new_x = origin[0] + dx * (d1 - d0)
        new_y = origin[1] - dy * (o1 - o0)
        if self._integer:
            new_x = float(round(new_x))
            new_y = float(round(new_y))
        new_y = min(o1, max(o0, new_y))

        group = [
            i for i, point in enumerate(self._points)
            if abs(point[0] - self._points[index][0]) < 1e-12
        ]
        if len(group) > 1:
            first = group[0]
            last = group[-1]
            low = self._points[first - 1][0] if first > 0 else d0
            high = self._points[last + 1][0] if last < len(self._points) - 1 else d1
            clamped = min(high, max(low, new_x))
            for i in group:
                self._points[i][0] = clamped
            self._points[index][1] = new_y
        else:
            previous = self._points[index - 1][0] if index > 0 else d0
            following = (
                self._points[index + 1][0]
                if index < len(self._points) - 1
                else d1
            )
            if index == 0:
                new_x = d0
            elif index == len(self._points) - 1:
                new_x = d1
            else:
                gap = 1.0 if self._integer else 1e-6
                new_x = min(following - gap, max(previous + gap, new_x))
            self._points[index] = [new_x, new_y]
        self.update()

    def mouseReleaseEvent(self, event):
        if self._drag is not None:
            self._points = normalize_curve(self._points, self._domain)
            self._drag = None
            self._selected = -1
            self.selection_changed.emit(-1)
            self.curve_changed.emit()
            self.update()
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton:
            super().mouseDoubleClickEvent(event)
            return
        x, _y = self._to_data(event.position())
        if self._integer:
            x = float(round(x))
        index = self._segment_index(x)
        if index is None:
            return
        left = self._points[index]
        right = self._points[index + 1]
        t = (x - left[0]) / (right[0] - left[0])
        y = left[1] + t * (right[1] - left[1])
        if self._integer:
            y = float(round(y))
        self._points.insert(index + 1, [x, y])
        self._points = normalize_curve(self._points, self._domain)
        self._selected = index + 1
        self.selection_changed.emit(self._selected)
        self.curve_changed.emit()
        self.update()


class PixelMapDialog(QDialog):
    """Graph + range fields + presets for one pixel-map filter."""

    def __init__(self, parent, filter_dict, data, scale_z=1.0, offset_z=0.0):
        # A normal top-level window, not a transient WM dialog: some window
        # managers group transient dialogs with their parent and dragging the
        # dialog moves the parent instead.  Being parentless guarantees the
        # two windows move independently; exec() is already application-modal.
        flags = (
            Qt.WindowType.Window
            | Qt.WindowType.WindowTitleHint
            | Qt.WindowType.WindowSystemMenuHint
            | Qt.WindowType.WindowCloseButtonHint
            | Qt.WindowType.WindowMaximizeButtonHint
            | Qt.WindowType.WindowMinimizeButtonHint
        )
        super().__init__(None, flags)
        self._host_window = parent.window() if parent is not None else None
        self.setWindowTitle("Pixel Map")
        self.resize(720, 640)
        if self._host_window is not None:
            center = self._host_window.frameGeometry().center()
            self.move(center - self.rect().center())
        self._filter = dict(filter_dict)
        self._data = data
        self._scale_z = float(scale_z)
        self._offset_z = float(offset_z)
        self._integer = bool(
            data is not None and data.dtype.kind in "ui"
        )
        self._dtype = data.dtype if data is not None else None

        domain = self._initial_domain()
        stored = self._filter.get("points")
        points = stored if stored else identity_points(domain)
        points = fit_curve_to_range(points, domain, *self._bounds())
        out_range = self._plot_range(domain, points)

        layout = QVBoxLayout(self)
        self.label_input = QLabel(self._input_text(domain))
        self.label_input.setWordWrap(True)
        layout.addWidget(self.label_input)

        self.plot = PixelMapPlot()
        self.plot.set_curve(points, domain, out_range, self._integer)
        self.plot.set_histogram(self._histogram(domain))
        self.plot.selection_changed.connect(self._on_selection_changed)
        layout.addWidget(self.plot, 1)

        layout.addLayout(self._build_range_row())
        layout.addLayout(self._build_preset_row())
        layout.addLayout(self._build_point_row())

        self.chk_preserve = QCheckBox("Preserve 0 / NaN (depth)")
        self.chk_preserve.setToolTip(
            "Invalid depth values (0 and NaN) are never remapped."
        )
        self.chk_preserve.setChecked(
            bool(self._filter.get("preserve_invalid", True))
        )
        self.chk_preserve.setVisible(not self._integer)
        layout.addWidget(self.chk_preserve)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        self.btn_fit = buttons.addButton(
            "Fit domain to data", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.btn_reset = buttons.addButton(
            "Reset", QDialogButtonBox.ButtonRole.ResetRole
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        self.btn_fit.clicked.connect(self._fit_domain)
        self.btn_reset.clicked.connect(self._reset_curve)
        layout.addWidget(buttons)

        self._sync_fields()

    # -- data helpers -------------------------------------------------
    def _initial_domain(self):
        natural = input_domain(self._data)
        if self._dtype is not None and self._dtype.kind in "ui":
            return natural
        stored = self._filter.get("domain")
        if (
            isinstance(stored, (list, tuple))
            and len(stored) >= 2
            and float(stored[1]) > float(stored[0])
        ):
            return float(stored[0]), float(stored[1])
        return natural or (0.0, 255.0)

    def _bounds(self):
        if self._integer:
            return 0.0, 255.0 if self._dtype is None else float(
                np.iinfo(self._dtype).max
            )
        return -float("inf"), float("inf")

    def _plot_range(self, domain, points):
        if self._integer:
            return output_range(domain, self._dtype)
        values = [point[1] for point in points]
        low = min(float(domain[0]), min(values))
        high = max(float(domain[1]), max(values))
        if high <= low:
            high = low + 1.0
        return low, high

    def _input_text(self, domain):
        if self._data is None:
            return "No input image."
        dtype = self._data.dtype
        d0, d1 = domain
        if dtype.kind in "ui":
            bits = dtype.itemsize * 8
            return f"Input: {dtype} · {bits}-bit · {int(d0)}–{int(d1)}"
        finite = self._data[np.isfinite(self._data)]
        if finite.size == 0:
            return f"Input: {dtype} · no finite values"
        data_min = float(finite.min())
        data_max = float(finite.max())
        text = (
            f"Input: {dtype} · raw counts · data "
            f"{data_min:.3f}–{data_max:.3f}"
        )
        if abs(self._scale_z - 1.0) > 1e-12 or abs(self._offset_z) > 1e-12:
            text += (
                f" · Z {data_min * self._scale_z + self._offset_z:.3f}–"
                f"{data_max * self._scale_z + self._offset_z:.3f} mm"
            )
        return text

    def _histogram(self, domain):
        if self._data is None or self._data.size == 0:
            return None
        flat = self._data.reshape(-1)
        step = max(1, int(math.ceil(flat.size / _HISTOGRAM_MAX_SAMPLES)))
        sample = flat[::step]
        if self._data.dtype.kind == "f":
            sample = sample[np.isfinite(sample)]
            sample = sample[sample != 0.0]
        if sample.size == 0:
            return None
        counts, _edges = np.histogram(
            sample, bins=_HISTOGRAM_BINS, range=domain
        )
        return counts

    # -- widgets ------------------------------------------------------
    def _spin(self, value, integer):
        spin = QDoubleSpinBox()
        spin.setDecimals(0 if integer else 3)
        if integer:
            spin.setRange(-1e9, 1e9)
        else:
            spin.setRange(-1e12, 1e12)
        spin.setValue(float(value))
        spin.setKeyboardTracking(False)
        return spin

    def _build_range_row(self):
        row = QHBoxLayout()
        integer = self._integer
        d0, d1 = self.plot._domain
        o0, o1 = self.plot._out_range
        self.spin_in0 = self._spin(d0, integer)
        self.spin_in1 = self._spin(d1, integer)
        self.spin_out0 = self._spin(o0, integer)
        self.spin_out1 = self._spin(o1, integer)
        row.addWidget(QLabel("In"))
        row.addWidget(self.spin_in0)
        row.addWidget(QLabel("→"))
        row.addWidget(self.spin_in1)
        row.addSpacing(12)
        row.addWidget(QLabel("Out"))
        row.addWidget(self.spin_out0)
        row.addWidget(QLabel("→"))
        row.addWidget(self.spin_out1)
        btn = QPushButton("Apply range")
        btn.clicked.connect(self._on_apply_range)
        row.addWidget(btn)
        return row

    def _build_preset_row(self):
        row = QHBoxLayout()
        self.combo_preset = QComboBox()
        self.combo_preset.addItem("Identity", "identity")
        self.combo_preset.addItem("Invert", "invert")
        self.combo_preset.addItem("Threshold", "threshold")
        self.combo_preset.addItem("Normalize", "normalize")
        self.combo_preset.currentIndexChanged.connect(
            lambda _index: self.spin_threshold.setEnabled(
                self.combo_preset.currentData() == "threshold"
            )
        )
        row.addWidget(QLabel("Preset"))
        row.addWidget(self.combo_preset)
        self.spin_threshold = self._spin(
            (self.plot._domain[0] + self.plot._domain[1]) / 2.0,
            self._integer,
        )
        self.spin_threshold.setEnabled(False)
        row.addWidget(self.spin_threshold)
        btn = QPushButton("Apply preset")
        btn.clicked.connect(self._on_apply_preset)
        row.addWidget(btn)
        return row

    def _build_point_row(self):
        row = QHBoxLayout()
        self.spin_px = self._spin(0.0, self._integer)
        self.spin_py = self._spin(0.0, self._integer)
        self.spin_px.setEnabled(False)
        self.spin_py.setEnabled(False)
        self.btn_delete = QPushButton("Delete point")
        self.btn_delete.setEnabled(False)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.spin_px.valueChanged.connect(self._on_point_spin_changed)
        self.spin_py.valueChanged.connect(self._on_point_spin_changed)
        row.addWidget(QLabel("Selected"))
        row.addWidget(QLabel("X"))
        row.addWidget(self.spin_px)
        row.addWidget(QLabel("Y"))
        row.addWidget(self.spin_py)
        row.addWidget(self.btn_delete)
        return row

    # -- interactions -------------------------------------------------
    def _set_points(self, points):
        points = normalize_curve(points, self.plot._domain)
        plot_range = self._plot_range(self.plot._domain, points)
        self.plot.set_curve(
            points, self.plot._domain, plot_range, self._integer
        )
        self._sync_fields()
        self._on_selection_changed(-1)

    def _sync_fields(self):
        d0, d1 = self.plot._domain
        o0, o1 = self.plot._out_range
        for spin, value in (
            (self.spin_in0, d0),
            (self.spin_in1, d1),
            (self.spin_out0, o0),
            (self.spin_out1, o1),
        ):
            spin.blockSignals(True)
            spin.setValue(value)
            spin.blockSignals(False)
        self.spin_in0.setMaximum(d1)
        self.spin_in1.setMinimum(d0)
        if self._integer:
            self.spin_out0.setRange(0, o1)
            self.spin_out1.setRange(0, o1)

    def _on_apply_range(self):
        domain = self.plot._domain
        try:
            points = apply_range(
                self.plot.points(),
                domain,
                self.spin_in0.value(),
                self.spin_in1.value(),
                self.spin_out0.value(),
                self.spin_out1.value(),
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Pixel Map", str(exc))
            return
        self._set_points(points)

    def _on_apply_preset(self):
        domain = self.plot._domain
        d0, d1 = domain
        o0, o1 = self.plot._out_range
        kind = self.combo_preset.currentData()
        if kind == "identity":
            points = identity_points(domain)
        elif kind == "invert":
            points = [[d0, o1], [d1, o0]]
        elif kind == "threshold":
            threshold = self.spin_threshold.value()
            if not (d0 < threshold < d1):
                QMessageBox.warning(
                    self,
                    "Pixel Map",
                    "The threshold must be inside the input domain.",
                )
                return
            points = [
                [d0, o0],
                [threshold, o0],
                [threshold, o1],
                [d1, o1],
            ]
        else:
            data = self._data
            if data is None or data.size == 0:
                QMessageBox.warning(self, "Pixel Map", "No input image.")
                return
            values = data.reshape(-1)
            if data.dtype.kind == "f":
                values = values[np.isfinite(values)]
                values = values[values != 0.0]
            if values.size == 0:
                QMessageBox.warning(
                    self, "Pixel Map", "No finite input values."
                )
                return
            low = max(d0, float(values.min()))
            high = min(d1, float(values.max()))
            if high <= low:
                QMessageBox.warning(
                    self, "Pixel Map", "The input range is empty."
                )
                return
            points = [
                [d0, o0],
                [low, o0],
                [high, o1],
                [d1, o1],
            ]
        self._set_points(points)

    def _on_selection_changed(self, index):
        enabled = 0 <= index < len(self.plot.points())
        self.spin_px.setEnabled(enabled)
        self.spin_py.setEnabled(enabled)
        self.btn_delete.setEnabled(
            enabled and 0 < index < len(self.plot.points()) - 1
        )
        if not enabled:
            return
        point = self.plot.points()[index]
        self.spin_px.blockSignals(True)
        self.spin_py.blockSignals(True)
        self.spin_px.setValue(point[0])
        self.spin_py.setValue(point[1])
        self.spin_px.blockSignals(False)
        self.spin_py.blockSignals(False)

    def _on_point_spin_changed(self):
        index = self.plot.selected()
        points = self.plot.points()
        if not (0 <= index < len(points)):
            return
        domain = self.plot._domain
        o0, o1 = self.plot._out_range
        x = self.spin_px.value()
        y = min(o1, max(o0, self.spin_py.value()))
        if 0 < index < len(points) - 1:
            gap = 1.0 if self._integer else 1e-6
            x = min(
                points[index + 1][0] - gap,
                max(points[index - 1][0] + gap, x),
            )
        else:
            x = domain[0] if index == 0 else domain[1]
        points[index] = [x, y]
        self._set_points(points)
        new_points = self.plot.points()
        if new_points:
            target = min(index, len(new_points) - 1)
            self.plot.select(target)
            self._on_selection_changed(target)

    def _delete_selected(self):
        index = self.plot.selected()
        points = self.plot.points()
        if 0 < index < len(points) - 1:
            del points[index]
            self._set_points(points)

    def _fit_domain(self):
        natural = input_domain(self._data)
        if not natural:
            return
        points = fit_curve_to_range(
            self.plot.points(), natural, *self._bounds()
        )
        self.plot.set_curve(
            points,
            natural,
            self._plot_range(natural, points),
            self._integer,
        )
        self.plot.set_histogram(self._histogram(natural))
        self.label_input.setText(self._input_text(natural))
        self.spin_threshold.setValue((natural[0] + natural[1]) / 2.0)
        self._sync_fields()

    def _reset_curve(self):
        domain = self.plot._domain
        self._set_points(identity_points(domain))

    def _on_accept(self):
        domain = self.plot._domain
        bounds = self._bounds()
        error = validate_curve(self.plot.points(), domain, *bounds)
        if error:
            QMessageBox.warning(self, "Pixel Map", error)
            return
        self.accept()

    def result_filter(self):
        return {
            **self._filter,
            "domain": [float(self.plot._domain[0]), float(self.plot._domain[1])],
            "points": self.plot.points(),
            "preserve_invalid": bool(self.chk_preserve.isChecked()),
        }
