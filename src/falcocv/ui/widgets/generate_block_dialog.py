"""Dialog used by WorkflowEditor to generate a Block from selected tools."""

from __future__ import annotations

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLineEdit,
    QRadioButton,
    QVBoxLayout,
)


class GenerateBlockDialog(QDialog):
    def __init__(self, tool_names, default_name="Block", parent=None):
        super().__init__(parent)
        self.setWindowTitle("Generate Block")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        layout.addWidget(QLabel("Block name:"))
        self.name_edit = QLineEdit(str(default_name or "Block"))
        layout.addWidget(self.name_edit)

        summary = QLabel(
            "Selected tools:\n  " + " \u2192 ".join(str(n) for n in tool_names)
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self.copy_radio = QRadioButton(
            "Copy tools into the Block (keep originals)"
        )
        self.copy_radio.setChecked(True)
        layout.addWidget(self.copy_radio)

        self.copy_note = QLabel(
            "A new Block is added after the selection. Its external ports are "
            "created but left unconnected."
        )
        self.copy_note.setWordWrap(True)
        self.copy_note.setIndent(24)
        layout.addWidget(self.copy_note)

        self.move_radio = QRadioButton(
            "Move tools into the Block (remove originals)"
        )
        layout.addWidget(self.move_radio)

        self.move_note = QLabel(
            "The Block replaces the selection in the workflow. External "
            "connections are rewired to the Block ports."
        )
        self.move_note.setWordWrap(True)
        self.move_note.setIndent(24)
        layout.addWidget(self.move_note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.copy_radio.toggled.connect(self._update_notes)
        self._update_notes()
        self.name_edit.selectAll()

    def _update_notes(self):
        self.copy_note.setEnabled(self.copy_radio.isChecked())
        self.move_note.setEnabled(self.move_radio.isChecked())

    @property
    def keep_originals(self) -> bool:
        return self.copy_radio.isChecked()

    def block_name(self) -> str:
        return self.name_edit.text().strip() or "Block"
