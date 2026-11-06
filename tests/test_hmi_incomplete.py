"""HMI Designer behavior for incomplete communications drafts."""

from __future__ import annotations


def _app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _incomplete_project():
    from falcocv.core.hmi import HMIProject

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "plc",
                "driver": "siemens",
                "enabled": True,
                "config": {},
                "tags": [
                    {
                        "name": "trigger",
                        "address": "",
                        "type": "bool",
                        "access": "rw",
                    }
                ],
            }
        ],
    }
    return project


def test_hmi_designer_marks_and_saves_incomplete_comms(monkeypatch, tmp_path):
    app = _app()
    assert app is not None
    from falcocv.ui.widgets.hmi_designer_dialog import HMIDesignerDialog

    monkeypatch.setattr(
        "falcocv.core.config.ConfigManager.add_recent_hmi_file",
        lambda *_args, **_kwargs: None,
    )
    path = tmp_path / "incomplete.hmi"
    dialog = HMIDesignerDialog(_incomplete_project())
    try:
        index = dialog._tabs.indexOf(dialog._comms_tab)
        assert "⚠" in dialog._tabs.tabText(index)
        assert "config.host is required" in dialog._tabs.tabToolTip(index)
        assert dialog._deploy_status.text() == "Comms incomplete"

        dialog._current_path = str(path)
        assert dialog._save_project() is True
        assert path.is_file()
        assert dialog._deploy_status.text() == "Saved (comms incomplete)"
        assert dialog._comms_tab.validate()[0] is False
    finally:
        dialog._modified = False
        dialog.close()


def test_hmi_project_save_rejects_invalid_comms(tmp_path):
    import pytest

    from falcocv.comms import CommsConfigError

    project = _incomplete_project()
    tag = project.comms_config["devices"][0]["tags"][0]
    project.comms_config["devices"][0]["tags"] = [tag, dict(tag)]
    path = tmp_path / "invalid.hmi"

    with pytest.raises(CommsConfigError, match="duplicate tag"):
        project.save(str(path))
    assert not path.exists()


def test_hmi_runtime_refuses_incomplete_comms():
    import pytest

    from falcocv.comms import CommsConfigError

    project = _incomplete_project()
    with pytest.raises(CommsConfigError, match="config.host is required"):
        project.start_runtime()
    assert project.comms is None


def test_hmi_designer_blocks_structurally_invalid_comms_save(
    monkeypatch, tmp_path
):
    app = _app()
    assert app is not None
    from falcocv.ui.widgets.hmi_designer_dialog import HMIDesignerDialog

    monkeypatch.setattr(
        "falcocv.core.config.ConfigManager.add_recent_hmi_file",
        lambda *_args, **_kwargs: None,
    )
    warnings = []
    project = _incomplete_project()
    tag = project.comms_config["devices"][0]["tags"][0]
    project.comms_config["devices"][0]["tags"] = [tag, dict(tag)]
    dialog = HMIDesignerDialog(project)
    try:
        monkeypatch.setattr(
            dialog._comms_tab,
            "_show_error",
            lambda message: warnings.append(message),
        )
        dialog._current_path = str(tmp_path / "invalid.hmi")
        assert dialog._save_project() is False
        assert warnings
        assert "duplicate tag" in warnings[0]
        assert not (tmp_path / "invalid.hmi").exists()
    finally:
        dialog._modified = False
        dialog.close()
