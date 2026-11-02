"""Manual feature editor for ShapeMatch models."""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QColor, QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QGraphicsView,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from falcocv.core.vision_image import VisionImage
from falcocv.ui.widgets.viewer import ImageViewer

_SOURCE = "ShapeMatchFeatureEditor"

# Overlay colors are BGR because VisionImage stores OpenCV-style images.
_CANDIDATE_BGR = (255, 212, 0)   # #00D4FF cyan
_AUTO_BGR = (118, 230, 0)        # #00E676 green
_MANUAL_BGR = (0, 214, 255)      # #FFD600 yellow
_WHITE_BGR = (255, 255, 255)

_CANDIDATE_HEX = "#00D4FF"
_AUTO_HEX = "#00E676"
_MANUAL_HEX = "#FFD600"

_BUTTON_ADD_STYLE = (
    "QPushButton:checked {"
    " background-color: #1B5E20; color: white;"
    " border: 1px solid #00E676;"
    "}"
)
_BUTTON_REMOVE_STYLE = (
    "QPushButton:checked {"
    " background-color: #5C1A12; color: white;"
    " border: 1px solid #FF3D00;"
    "}"
)


class _CandidateGrid:
    """Small spatial index for candidate pixels inside a brush stamp."""

    def __init__(self, xs, ys, cell_size: int = 16):
        self.xs = np.asarray(xs, dtype=np.int32)
        self.ys = np.asarray(ys, dtype=np.int32)
        self.cell_size = max(1, int(cell_size))
        self._cells: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index, (x, y) in enumerate(zip(self.xs, self.ys)):
            key = (
                int(x) // self.cell_size,
                int(y) // self.cell_size,
            )
            self._cells[key].append(index)

    def query(self, x: int, y: int, radius: int) -> list[int]:
        """Return candidate indices whose centres are inside the circle."""
        radius = max(1, int(radius))
        radius_sq = radius * radius
        min_x = int(x) - radius
        max_x = int(x) + radius
        min_y = int(y) - radius
        max_y = int(y) + radius
        cell = self.cell_size
        hits: list[int] = []
        for cell_x in range(min_x // cell, max_x // cell + 1):
            for cell_y in range(min_y // cell, max_y // cell + 1):
                for index in self._cells.get((cell_x, cell_y), ()):
                    dx = int(self.xs[index]) - int(x)
                    dy = int(self.ys[index]) - int(y)
                    if dx * dx + dy * dy <= radius_sq:
                        hits.append(index)
        return hits


class ShapeMatchFeatureEditorDialog(QDialog):
    """Add/remove Canny candidates from a trained ShapeMatch model."""

    def __init__(self, tool, parent=None):
        super().__init__(parent)
        self.tool = tool
        self.setWindowTitle("ShapeMatch Feature Editor")
        self.resize(1100, 800)

        self._candidates = tool.training_candidates()
        self._xs = self._candidates["x"]
        self._ys = self._candidates["y"]
        self._values = self._candidates["mag"]
        self._grid = _CandidateGrid(self._xs, self._ys)
        self._selected = set(tool.selected_candidate_indices(self._candidates))
        self._added = set(tool.manual_candidate_indices(self._candidates))
        self._manual_edit = bool(tool.is_manual_edit)
        self._undo_stack = []
        self._mode = "add"
        self._brush_radius = 6
        self._hover_point = None
        self._stroke_before = None
        self._stroke_changed = False
        self._last_stroke_point = None
        self._last_omitted = 0

        self._build_ui()
        self._build_display()
        self._update_overlays(auto_fit=True)
        self._update_status()
        self._set_mode("add")

        self._undo_shortcut = QShortcut(
            QKeySequence.StandardKey.Undo, self
        )
        self._undo_shortcut.activated.connect(self._undo)
        self._brush_down_shortcut = QShortcut(QKeySequence("["), self)
        self._brush_down_shortcut.activated.connect(
            lambda: self._adjust_brush(-1)
        )
        self._brush_up_shortcut = QShortcut(QKeySequence("]"), self)
        self._brush_up_shortcut.activated.connect(
            lambda: self._adjust_brush(1)
        )

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QHBoxLayout(self)
        self.viewer = ImageViewer()
        self.viewer.set_roi_editing(False)
        self.viewer.set_drawing_new_roi(False)
        self.viewer.mouse_moved.connect(self._on_mouse_moved)
        self.viewer.mouse_left.connect(self._on_mouse_left)
        self.viewer.stroke_started.connect(self._begin_stroke)
        self.viewer.stroke_moved.connect(self._extend_stroke)
        self.viewer.stroke_finished.connect(self._end_stroke)
        layout.addWidget(self.viewer, 1)

        side = QWidget()
        side.setFixedWidth(300)
        side_layout = QVBoxLayout(side)

        title = QLabel("Feature selection")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        side_layout.addWidget(title)

        instructions = QLabel(
            "Drag to paint. Add selects candidates inside the brush; "
            "Remove deletes selected features. Use Pan to move the image, "
            "the wheel to zoom, and Ctrl+Z to undo a whole stroke."
        )
        instructions.setWordWrap(True)
        side_layout.addWidget(instructions)

        self.btn_add = QPushButton("Add")
        self.btn_add.setCheckable(True)
        self.btn_add.setChecked(True)
        self.btn_add.setStyleSheet(_BUTTON_ADD_STYLE)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setCheckable(True)
        self.btn_remove.setStyleSheet(_BUTTON_REMOVE_STYLE)
        self.btn_pan = QPushButton("Pan")
        self.btn_pan.setCheckable(True)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.btn_add)
        mode_group.addButton(self.btn_remove)
        mode_group.addButton(self.btn_pan)
        self.btn_add.clicked.connect(lambda: self._set_mode("add"))
        self.btn_remove.clicked.connect(lambda: self._set_mode("remove"))
        self.btn_pan.clicked.connect(lambda: self._set_mode("pan"))
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.btn_add)
        mode_row.addWidget(self.btn_remove)
        mode_row.addWidget(self.btn_pan)
        side_layout.addLayout(mode_row)

        brush_row = QHBoxLayout()
        brush_row.addWidget(QLabel("Brush radius:"))
        self.spin_brush = QSpinBox()
        self.spin_brush.setRange(1, 200)
        self.spin_brush.setValue(self._brush_radius)
        self.spin_brush.setSuffix(" px")
        self.spin_brush.setToolTip("Brush radius in template pixels.")
        self.spin_brush.valueChanged.connect(
            self._on_brush_radius_changed
        )
        brush_row.addWidget(self.spin_brush, 1)
        side_layout.addLayout(brush_row)

        undo_row = QHBoxLayout()
        self.btn_undo = QPushButton("Undo")
        self.btn_undo.clicked.connect(self._undo)
        self.btn_reset = QPushButton("Reset to auto")
        self.btn_reset.clicked.connect(self._reset)
        undo_row.addWidget(self.btn_undo)
        undo_row.addWidget(self.btn_reset)
        side_layout.addLayout(undo_row)

        self.label_status = QLabel("")
        self.label_status.setWordWrap(True)
        self.label_status.setStyleSheet("font-size: 11px;")
        side_layout.addWidget(self.label_status)

        legend = QGroupBox("Legend")
        legend_layout = QGridLayout(legend)
        legend_layout.setContentsMargins(6, 6, 6, 6)
        legend_layout.setHorizontalSpacing(6)
        legend_layout.setVerticalSpacing(4)
        legend_layout.setColumnStretch(1, 1)
        self._add_legend_row(
            legend_layout, 0, _CANDIDATE_HEX, _CANDIDATE_HEX, "Candidate"
        )
        self._add_legend_row(
            legend_layout, 1, _AUTO_HEX, _AUTO_HEX, "Auto feature"
        )
        self._add_legend_row(
            legend_layout, 2, _MANUAL_HEX, "#FFFFFF", "Manual feature"
        )
        self._add_legend_row(
            legend_layout,
            3,
            "rgba(0, 230, 118, 120)",
            "#FFFFFF",
            "Brush add (white/green)",
        )
        self._add_legend_row(
            legend_layout,
            4,
            "rgba(255, 61, 0, 120)",
            "#FFFFFF",
            "Brush remove (white/red)",
        )
        side_layout.addWidget(legend)
        side_layout.addStretch()

        button_row = QHBoxLayout()
        self.btn_apply = QPushButton("Apply")
        self.btn_apply.clicked.connect(self._apply)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self.reject)
        button_row.addWidget(self.btn_apply)
        button_row.addWidget(self.btn_cancel)
        side_layout.addLayout(button_row)

        layout.addWidget(side)

    @staticmethod
    def _add_legend_row(layout, row, fill, border, text):
        swatch = QLabel()
        swatch.setFixedSize(12, 12)
        swatch.setStyleSheet(
            f"background-color: {fill};"
            f" border: 2px solid {border};"
            " border-radius: 6px;"
        )
        layout.addWidget(swatch, row, 0)
        label = QLabel(text)
        label.setWordWrap(True)
        label.setStyleSheet("font-size: 11px;")
        layout.addWidget(label, row, 1)

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def _build_display(self):
        source = self.tool._template_color
        if source is None:
            raise ValueError("ShapeMatch has no template image.")
        display = (source.astype(np.float32) * 0.55).astype(np.uint8)
        # Paint candidates once into the base image. They are deliberately
        # small; the selected features are drawn on top as vector circles.
        display[self._ys, self._xs] = _CANDIDATE_BGR
        self._display = VisionImage.from_array(display)
        self._layer = self._display.get_overlay_layer(
            "default", source_tool=_SOURCE
        )
        self._layer.source_tool = _SOURCE

    def _update_overlays(self, auto_fit: bool = False):
        layer = self._layer
        layer.elements = [
            element
            for element in layer.elements
            if element.source_tool != _SOURCE
        ]
        for index in sorted(self._selected):
            center = (int(self._xs[index]), int(self._ys[index]))
            if index in self._added:
                # White rim makes manual features distinguishable even for
                # colour-blind users.
                layer.add_circle(
                    center,
                    radius=4.5,
                    color=_WHITE_BGR,
                    thickness=1,
                    fill=True,
                    priority=10,
                )
                layer.add_circle(
                    center,
                    radius=3.0,
                    color=_MANUAL_BGR,
                    thickness=1,
                    fill=True,
                    priority=11,
                )
            else:
                layer.add_circle(
                    center,
                    radius=3.0,
                    color=_AUTO_BGR,
                    thickness=1,
                    fill=True,
                    priority=10,
                )
        self.viewer.set_image(
            self._display,
            auto_fit=auto_fit,
            render_overlays_flag=True,
        )

    def _on_mouse_moved(self, x, y, _value):
        if x < 0 or y < 0:
            self._hover_point = None
            self._hide_brush_cursor()
            return
        self._hover_point = (int(x), int(y))
        self._update_brush_cursor()

    def _on_mouse_left(self):
        self._hover_point = None
        self._hide_brush_cursor()

    def _update_brush_cursor(self):
        if self._mode == "pan" or self._hover_point is None:
            self._hide_brush_cursor()
            return
        x, y = self._hover_point
        if x < 0 or y < 0:
            self._hide_brush_cursor()
            return
        scale_x, scale_y = self.viewer.display_scale
        scale = max(scale_x, scale_y)
        if scale <= 0.0:
            self._hide_brush_cursor()
            return
        radius = max(1.0, float(self._brush_radius) / scale)
        center = self.viewer.original_to_display_point(x, y)
        fill = (
            QColor(0, 230, 118, 35)
            if self._mode == "add"
            else QColor(255, 61, 0, 35)
        )
        self.viewer.set_brush_cursor(center, radius, fill)

    def _hide_brush_cursor(self):
        self.viewer.set_brush_cursor(None)

    def _update_status(self):
        count = len(self._selected)
        budget = int(self.tool.params.get("max_features", 800))
        lines = [
            f"Selected: {count} / {budget}",
            f"Candidates: {len(self._xs)}",
            f"Brush radius: {self._brush_radius} px",
        ]
        if self._manual_edit:
            lines.append("Manual selection")
        if self._last_omitted:
            lines.append(
                f"Budget full: {self._last_omitted} candidate(s) omitted."
            )
        if count >= budget:
            lines.append("Budget full: remove a feature before adding.")
        if count < 20:
            lines.append("At least 20 features are required.")
        self.label_status.setText("\n".join(lines))
        self.btn_apply.setEnabled(count >= 20)
        self.btn_undo.setEnabled(bool(self._undo_stack))

    # ------------------------------------------------------------------
    # Editing
    # ------------------------------------------------------------------
    def _set_mode(self, mode: str):
        if mode not in ("add", "remove", "pan"):
            return
        self._mode = mode
        self.btn_add.setChecked(mode == "add")
        self.btn_remove.setChecked(mode == "remove")
        self.btn_pan.setChecked(mode == "pan")
        if mode == "pan":
            self.viewer.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
            self.viewer.viewport().setCursor(
                Qt.CursorShape.OpenHandCursor
            )
            self._hide_brush_cursor()
        else:
            self.viewer.setDragMode(QGraphicsView.DragMode.NoDrag)
            self.viewer.viewport().setCursor(Qt.CursorShape.CrossCursor)
            self._update_brush_cursor()

    def _on_brush_radius_changed(self, value):
        self._brush_radius = int(value)
        self._update_brush_cursor()
        self._update_status()

    def _adjust_brush(self, delta: int):
        self.spin_brush.setValue(
            max(1, min(200, self._brush_radius + int(delta)))
        )

    def _push_undo(self):
        self._undo_stack.append(
            (set(self._selected), set(self._added), self._manual_edit)
        )
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def _remember_stroke(self):
        if self._stroke_changed or self._stroke_before is None:
            return
        self._undo_stack.append(self._stroke_before)
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)
        self._stroke_changed = True

    def _undo(self):
        if not self._undo_stack:
            return
        selected, added, manual = self._undo_stack.pop()
        self._selected = selected
        self._added = added
        self._manual_edit = manual
        self._last_omitted = 0
        self._update_overlays()
        self._update_status()

    def _reset(self):
        self._push_undo()
        limit = int(self.tool.params.get("max_features", 800))
        order = np.argsort(self._values)[::-1][:limit]
        self._selected = {int(index) for index in order}
        self._added = set()
        self._manual_edit = False
        self._last_omitted = 0
        self._update_overlays()
        self._update_status()

    def _stamp_at(self, x: int, y: int) -> bool:
        if self._mode not in ("add", "remove"):
            return False
        radius = max(1, int(self._brush_radius))
        hits = self._grid.query(x, y, radius)
        if not hits:
            return False

        if self._mode == "add":
            candidates = [
                index for index in hits if index not in self._selected
            ]
            if not candidates:
                return False
            budget = int(self.tool.params.get("max_features", 800))
            remaining = budget - len(self._selected)
            if remaining <= 0:
                self._last_omitted = len(candidates)
                return False
            if len(candidates) > remaining:
                candidates.sort(
                    key=lambda index: float(self._values[index]),
                    reverse=True,
                )
                self._last_omitted = len(candidates) - remaining
                candidates = candidates[:remaining]
            else:
                self._last_omitted = 0
            self._remember_stroke()
            for index in candidates:
                self._selected.add(index)
                self._added.add(index)
            self._manual_edit = True
            return bool(candidates)

        candidates = [
            index for index in hits if index in self._selected
        ]
        if not candidates:
            return False
        self._remember_stroke()
        for index in candidates:
            self._selected.discard(index)
            self._added.discard(index)
        self._manual_edit = True
        self._last_omitted = 0
        return True

    def _interpolated_points(self, x0, y0, x1, y1):
        dx = int(x1) - int(x0)
        dy = int(y1) - int(y0)
        distance = math.hypot(dx, dy)
        spacing = max(1.0, float(self._brush_radius) * 0.5)
        steps = max(1, int(math.ceil(distance / spacing)))
        for step in range(1, steps + 1):
            yield (
                int(round(int(x0) + dx * step / steps)),
                int(round(int(y0) + dy * step / steps)),
            )

    def _begin_stroke(self, x: int, y: int):
        if self._mode == "pan":
            return
        self._stroke_before = (
            set(self._selected),
            set(self._added),
            self._manual_edit,
        )
        self._stroke_changed = False
        self._last_omitted = 0
        if x < 0 or y < 0:
            self._last_stroke_point = None
            self._update_status()
            return
        self._last_stroke_point = (int(x), int(y))
        changed = self._stamp_at(int(x), int(y))
        if changed:
            self._update_overlays()
        self._update_status()

    def _extend_stroke(self, x: int, y: int):
        if self._mode == "pan" or self._stroke_before is None:
            return
        x = int(x)
        y = int(y)
        if x < 0 or y < 0:
            self._last_stroke_point = None
            return
        if self._last_stroke_point is None:
            self._last_stroke_point = (x, y)
            changed = self._stamp_at(x, y)
        else:
            last_x, last_y = self._last_stroke_point
            if (x, y) == (last_x, last_y):
                return
            changed = False
            for px, py in self._interpolated_points(
                last_x, last_y, x, y
            ):
                if self._stamp_at(px, py):
                    changed = True
            self._last_stroke_point = (x, y)
        if changed:
            self._update_overlays()
        self._update_status()

    def _end_stroke(self, x: int, y: int):
        if self._stroke_before is None:
            return
        if x >= 0 and y >= 0:
            self._extend_stroke(x, y)
        self._stroke_before = None
        self._stroke_changed = False
        self._last_stroke_point = None
        self._update_status()

    def _on_click(self, x: int, y: int):
        """Compatibility helper used by tests and simple programmatic edits."""
        self._begin_stroke(x, y)
        self._end_stroke(x, y)

    def _apply(self):
        if len(self._selected) < 20:
            self._update_status()
            return
        try:
            self.tool.apply_manual_selection(
                self._candidates,
                sorted(self._selected),
                manual=self._added,
                edited=self._manual_edit,
            )
        except Exception as exc:
            self.label_status.setText(f"Apply failed: {exc}")
            return
        self.accept()
