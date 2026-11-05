"""Open Recent support for the HMI Designer."""

from __future__ import annotations


def _app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_config_manager_tracks_recent_hmi_files(tmp_path):
    from falcocv.core.config import ConfigManager

    config = ConfigManager()
    original = list(config.get_recent_hmi_files())
    try:
        config.data["recent_hmi_files"] = []
        config.add_recent_hmi_file(str(tmp_path / "one.hmi"))
        config.add_recent_hmi_file(str(tmp_path / "two.hmi"))
        config.add_recent_hmi_file(str(tmp_path / "one.hmi"))

        recents = config.get_recent_hmi_files()
        assert recents[0] == str((tmp_path / "one.hmi").resolve())
        assert recents[1] == str((tmp_path / "two.hmi").resolve())
        assert len(recents) == 2
    finally:
        config.data["recent_hmi_files"] = original
        config.save()


def test_hmi_designer_open_recent_loads_project(tmp_path):
    app = _app()
    assert app is not None
    from falcocv.core.config import ConfigManager
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_designer_dialog import HMIDesignerDialog

    path = tmp_path / "recent.hmi"
    HMIProject().save(str(path))

    config = ConfigManager()
    original = list(config.get_recent_hmi_files())
    dialog = None
    try:
        config.data["recent_hmi_files"] = []
        dialog = HMIDesignerDialog()
        assert dialog._open_project_from_path(str(path)) is True
        assert dialog._current_path == str(path.resolve())
        assert str(path.resolve()) in config.get_recent_hmi_files()
    finally:
        if dialog is not None:
            dialog.close()
        config.data["recent_hmi_files"] = original
        config.save()
