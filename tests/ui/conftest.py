"""Shared fixtures for UI tests (offscreen Qt)."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

import falcocv.main as main_module
from falcocv.core.job_manager import JobManager
from falcocv.core.workflow import Workflow
from falcocv.core.workflow_controller import ControllerState
from falcocv.ui.widgets.job_config_dialog import JobConfigDialog
from falcocv.ui.widgets.toolbox import Toolbox
from falcocv.ui.widgets.workflow_editor import WorkflowEditor

# MainWindow resolves these names from module globals (they are normally set
# inside main()).  Tests construct the window directly, so provide them.
main_module.Workflow = Workflow
main_module.ControllerState = ControllerState
main_module.JobManager = JobManager
main_module.WorkflowEditor = WorkflowEditor
main_module.Toolbox = Toolbox
main_module.JobConfigDialog = JobConfigDialog


@pytest.fixture
def main_window(qtbot, monkeypatch):
    """A real MainWindow with the web server and modal save prompt disabled."""
    monkeypatch.setattr(main_module.MainWindow, "_web_start", lambda self: None)
    monkeypatch.setattr(main_module.MainWindow, "_web_stop", lambda self: None)
    monkeypatch.setattr(main_module.MainWindow, "_web_check", lambda self: None)
    monkeypatch.setattr(main_module.MainWindow, "_web_blink_tick", lambda self: None)
    monkeypatch.setattr(
        main_module.MainWindow, "_confirm_save_if_modified", lambda self: True
    )

    window = main_module.MainWindow()
    qtbot.addWidget(window)
    window.show()
    qtbot.wait(50)
    yield window

    for tool in window.workflow.tools:
        cfg = getattr(tool, "_config_window", None)
        if cfg is not None:
            cfg.close()
            tool._config_window = None
