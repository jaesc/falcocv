from PyQt6.QtWidgets import (QDialog, QVBoxLayout, QHBoxLayout, QLabel, 
                             QCheckBox, QPushButton, QGroupBox, QWidget)
from PyQt6.QtCore import Qt
from falcocv.ui.style import BORDER, TEXT


class JobConfigDialog(QDialog):
    """Dialog for per-job configuration settings (serialized with the job)."""

    def __init__(self, workflow, parent=None):
        super().__init__(parent)
        self.workflow = workflow
        self.setWindowTitle("Job Configuration")
        self.setMinimumWidth(400)
        self.setWindowFlags(
            self.windowFlags() |
            Qt.WindowType.WindowMaximizeButtonHint |
            Qt.WindowType.WindowMinimizeButtonHint
        )

        main_layout = QVBoxLayout(self)
        main_layout.setSpacing(15)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # ── Execution Settings ──────────────────────────────────────
        group_exec = QGroupBox("Execution")
        group_exec.setStyleSheet(
            f"QGroupBox {{ font-weight: bold; color: {TEXT}; border: 1px solid {BORDER}; "
            "border-radius: 4px; margin-top: 8px; padding-top: 8px; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; }"
        )
        exec_layout = QVBoxLayout(group_exec)

        self.chk_stop_on_error = QCheckBox("Stop pipeline on first error")
        self.chk_stop_on_error.setToolTip(
            "If enabled, the pipeline will stop executing subsequent tools "
            "as soon as any tool raises an error."
        )
        self.chk_stop_on_error.setChecked(workflow.stop_on_error)
        self.chk_stop_on_error.toggled.connect(self._on_stop_on_error_changed)
        exec_layout.addWidget(self.chk_stop_on_error)

        main_layout.addWidget(group_exec)

        # ── Footer ──────────────────────────────────────────────────
        main_layout.addStretch()
        footer = QHBoxLayout()
        footer.addStretch()
        close_btn = QPushButton("Close")
        close_btn.setFixedWidth(100)
        close_btn.clicked.connect(self.accept)
        footer.addWidget(close_btn)
        main_layout.addLayout(footer)

        for btn in self.findChildren(QPushButton):
            btn.setAutoDefault(False)
            btn.setDefault(False)

    def _on_stop_on_error_changed(self, checked: bool):
        self.workflow.stop_on_error = checked
