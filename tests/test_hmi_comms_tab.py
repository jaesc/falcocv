"""Offscreen tests for the HMI Designer Communications tab."""

from __future__ import annotations


def _app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_comms_tab_adds_device_and_tag():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    tab = HMICommsWidget(project)

    tab._add_device()
    tab._add_tag()
    assert tab.flush_to_project()

    devices = project.comms_config["devices"]
    assert len(devices) == 1
    assert devices[0]["id"] == "device"
    assert devices[0]["driver"] == "dummy"
    assert devices[0]["tags"][0]["name"] == "tag1"
    assert tab.validate() == (True, "")


def test_comms_tab_keeps_changes_in_draft_until_flush():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    tab = HMICommsWidget(project)
    tab._add_device()
    tab._add_tag()

    assert project.comms_config["devices"] == []
    assert len(tab._draft_config["devices"]) == 1
    assert tab.flush_to_project()
    assert project.comms_config["devices"][0]["id"] == "device"
    assert project.comms_config["devices"][0]["tags"][0]["name"] == "tag1"


def test_comms_tab_uses_driver_fields_and_preserves_advanced_config():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "dummy",
                "driver": "dummy",
                "config": {
                    "read_delay_ms": 5,
                    "simulate": {
                        "counter": {"kind": "counter", "period_ms": 10}
                    },
                },
                "tags": [],
            }
        ],
    }
    tab = HMICommsWidget(project)

    assert "read_delay_ms" in tab._config_widgets
    assert tab._config_widgets["read_delay_ms"].value() == 5
    assert tab.advanced_check.isChecked()
    assert not tab.advanced_edit.isHidden()
    assert "simulate" in tab.advanced_edit.toPlainText()

    assert tab.flush_to_project()
    config = project.comms_config["devices"][0]["config"]
    assert config["read_delay_ms"] == 5
    assert "simulate" in config


def test_comms_tab_hides_advanced_config_when_there_are_no_extra_keys():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "dummy",
                "driver": "dummy",
                "config": {"read_delay_ms": 5},
                "tags": [],
            }
        ],
    }
    tab = HMICommsWidget(project)
    assert not tab.advanced_check.isChecked()
    assert tab.advanced_edit.isHidden()


def test_comms_tab_rejects_invalid_tag_default():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    tab = HMICommsWidget(project)
    tab._add_device()
    tab._add_tag()

    # tag1 is bool; this default cannot be parsed.
    tab.tags_table.item(0, 5).setText("not-a-bool")
    assert tab.flush_to_project() is False
    assert "Row 1" in tab.status.text()
