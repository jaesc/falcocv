"""Manual feature editor for ShapeMatch models."""

from __future__ import annotations

import time

import cv2
import numpy as np
from PyQt6.QtGui import QKeySequence, QShortcut
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from falcocv.core.vision_image import VisionImage
from falcocv.ui.widgets.viewer import ImageViewer

_SOURCE = "ShapeMatchFeatureEditor"


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
        self._selected = set(tool.selected_candidate_indices(self._candidates))
        self._added = set(tool.manual_candidate_indices(self._candidates))
        self._manual_edit = bool(tool.is_manual_edit)
        self._undo_stack = []
        self._mode = "add"
        self._last_click = None
        self._last_click_time = 0.0

        self._build_ui()
        self._build_display()
        self._update_overlays(auto_fit=True)
        self._update_status()

        self._undo_shortcut = QShortcut(
            QKeySequence.StandardKey.Undo, self
        )
        self._undo_shortcut.activated.connect(self._undo)

    # ------------------------------------------------------------------
    # UI
    # ------------------------------------------------------------------
    def _build_ui(self):
        layout = QHBoxLayout(self)
        self.viewer = ImageViewer()
        self.viewer.set_roi_editing(False)
        self.viewer.set_drawing_new_roi(False)
        self.viewer.mouse_clicked.connect(self._on_click)
        layout.addWidget(self.viewer, 1)

        side = QWidget()
        side.setFixedWidth(285)
        side_layout = QVBoxLayout(side)

        title = QLabel("Feature selection")
        title.setStyleSheet("font-weight: bold; font-size: 13px;")
        side_layout.addWidget(title)

        instructions = QLabel(
            "Click a candidate edge to add it. Switch to Remove to delete "
            "a selected feature. The total is limited by Max features."
        )
        instructions.setWordWrap(True)
        side_layout.addWidget(instructions)

        self.btn_add = QPushButton("Add")
        self.btn_add.setCheckable(True)
        self.btn_add.setChecked(True)
        self.btn_remove = QPushButton("Remove")
        self.btn_remove.setCheckable(True)
        mode_group = QButtonGroup(self)
        mode_group.setExclusive(True)
        mode_group.addButton(self.btn_add)
        mode_group.addButton(self.btn_remove)
        self.btn_add.clicked.connect(lambda: self._set_mode("add"))
        self.btn_remove.clicked.connect(lambda: self._set_mode("remove"))
        mode_row = QHBoxLayout()
        mode_row.addWidget(self.btn_add)
        mode_row.addWidget(self.btn_remove)
        side_layout.addLayout(mode_row)

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

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------
    def _build_display(self):
        source = self.tool._template_color
        if source is None:
            raise ValueError("ShapeMatch has no template image.")
        display = (source.astype(np.float32) * 0.55).astype(np.uint8)
        display[self._ys, self._xs] = (80, 80, 80)
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
            layer.add_point(
                (int(self._xs[index]), int(self._ys[index])),
                color=(0, 255, 255),
                radius=4,
                ref_id=f"feature_{index}",
            )
        self.viewer.set_image(
            self._display,
            auto_fit=auto_fit,
            render_overlays_flag=True,
        )

    def _update_status(self):
        count = len(self._selected)
        budget = int(self.tool.params.get("max_features", 800))
        lines = [
            f"Selected: {count} / {budget}",
            f"Candidates: {len(self._xs)}",
        ]
        if self._manual_edit:
            lines.append("Manual selection")
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
        self._mode = mode
        self.btn_add.setChecked(mode == "add")
        self.btn_remove.setChecked(mode == "remove")

    def _push_undo(self):
        self._undo_stack.append(
            (set(self._selected), set(self._added), self._manual_edit)
        )
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)

    def _undo(self):
        if not self._undo_stack:
            return
        selected, added, manual = self._undo_stack.pop()
        self._selected = selected
        self._added = added
        self._manual_edit = manual
        self._update_overlays()
        self._update_status()

    def _reset(self):
        self._push_undo()
        limit = int(self.tool.params.get("max_features", 800))
        order = np.argsort(self._values)[::-1][:limit]
        self._selected = {int(index) for index in order}
        self._added = set()
        self._manual_edit = False
        self._update_overlays()
        self._update_status()

    def _nearest(self, x: int, y: int, include: bool):
        if include:
            pool = np.fromiter(self._selected, dtype=np.int64)
        else:
            mask = np.ones(len(self._xs), dtype=bool)
            if self._selected:
                mask[list(self._selected)] = False
            pool = np.flatnonzero(mask)
        if pool.size == 0:
            return None
        dx = self._xs[pool].astype(np.float64) - float(x)
        dy = self._ys[pool].astype(np.float64) - float(y)
        distance = dx * dx + dy * dy
        position = int(np.argmin(distance))
        min_size = min(self.tool._template_size)
        tolerance = max(4.0, float(min_size) * 0.01)
        if distance[position] > tolerance * tolerance:
            return None
        return int(pool[position])

    def _on_click(self, x: int, y: int):
        now = time.perf_counter()
        if (
            self._last_click == (x, y)
            and now - self._last_click_time < 0.08
        ):
            return
        self._last_click = (x, y)
        self._last_click_time = now
        if self._mode == "add":
            budget = int(self.tool.params.get("max_features", 800))
            if len(self._selected) >= budget:
                self._update_status()
                return
            index = self._nearest(x, y, include=False)
            if index is None:
                return
            self._push_undo()
            self._selected.add(index)
            self._added.add(index)
            self._manual_edit = True
        else:
            index = self._nearest(x, y, include=True)
            if index is None:
                return
            self._push_undo()
            self._selected.discard(index)
            self._added.discard(index)
            self._manual_edit = True
        self._update_overlays()
        self._update_status()

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
