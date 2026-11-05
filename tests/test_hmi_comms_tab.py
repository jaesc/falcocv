"""Offscreen tests for the HMI Designer Communications tab."""

from __future__ import annotations

import copy


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


def test_comms_tab_change_driver_clears_config_and_tags(monkeypatch):
    app = _app()
    assert app is not None
    from PyQt6.QtWidgets import QMessageBox

    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "plc",
                "driver": "dummy",
                "enabled": True,
                "config": {
                    "read_delay_ms": 5,
                    "simulate": {
                        "counter": {"kind": "counter", "period_ms": 10}
                    },
                },
                "tags": [
                    {
                        "name": "trigger",
                        "address": "trigger",
                        "type": "bool",
                        "access": "rw",
                    }
                ],
            }
        ],
    }
    tab = HMICommsWidget(project)
    prompts = []

    def confirm(_parent, _title, text, *_args):
        prompts.append(text)
        return QMessageBox.StandardButton.Yes

    monkeypatch.setattr(QMessageBox, "warning", confirm)
    tab.driver_combo.setCurrentText("omron")

    assert prompts and "clear" in prompts[0].lower()
    assert tab._driver_schema.name == "omron"
    assert tab.tags_table.rowCount() == 0
    assert tab.advanced_edit.toPlainText() == ""
    assert not tab.advanced_check.isChecked()

    device = tab._draft_config["devices"][0]
    assert device["driver"] == "omron"
    assert device["config"] == {}
    assert device["tags"] == []
    assert device["id"] == "plc"
    assert device["enabled"] is True
    assert "[omron]" in tab.device_list.item(0).text()

    tab._config_widgets["host"].setText("192.168.1.20")
    assert tab.flush_to_project()
    saved = project.comms_config["devices"][0]
    assert saved["driver"] == "omron"
    assert saved["config"]["host"] == "192.168.1.20"
    assert saved["tags"] == []


def test_comms_tab_change_driver_can_be_cancelled(monkeypatch):
    app = _app()
    assert app is not None
    from PyQt6.QtWidgets import QMessageBox

    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "plc",
                "driver": "dummy",
                "config": {"read_delay_ms": 5},
                "tags": [
                    {
                        "name": "trigger",
                        "address": "trigger",
                        "type": "bool",
                        "access": "rw",
                    }
                ],
            }
        ],
    }
    tab = HMICommsWidget(project)
    original = copy.deepcopy(tab._draft_config)

    monkeypatch.setattr(
        QMessageBox,
        "warning",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.No,
    )
    tab.driver_combo.setCurrentText("omron")

    assert tab.driver_combo.currentText() == "dummy"
    assert tab._driver_schema.name == "dummy"
    assert tab.tags_table.rowCount() == 1
    assert tab._draft_config == original
    assert tab._config_widgets["read_delay_ms"].value() == 5


def test_comms_tab_change_driver_without_data_does_not_prompt(monkeypatch):
    app = _app()
    assert app is not None
    from PyQt6.QtWidgets import QMessageBox

    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    tab = HMICommsWidget(project)
    tab._add_device()

    def unexpected_prompt(*_args, **_kwargs):
        raise AssertionError("empty driver change should not prompt")

    monkeypatch.setattr(QMessageBox, "warning", unexpected_prompt)
    tab.driver_combo.setCurrentText("omron")

    assert tab._driver_schema.name == "omron"
    assert tab.tags_table.rowCount() == 0
    assert tab._draft_config["devices"][0]["driver"] == "omron"


def test_comms_tab_default_follows_tag_type():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    tab = HMICommsWidget(HMIProject())
    tab._add_device()
    tab._add_tag()
    type_combo = tab.tags_table.cellWidget(0, 2)
    default_item = tab.tags_table.item(0, 5)

    assert default_item.text() == "false"
    type_combo.setCurrentText("str")
    assert default_item.text() == ""
    type_combo.setCurrentText("int")
    assert default_item.text() == "0"
    type_combo.setCurrentText("float")
    assert default_item.text() == "0.0"
    type_combo.setCurrentText("bool")
    assert default_item.text() == "false"

    default_item.setText("123")
    type_combo.setCurrentText("int")
    assert default_item.text() == "123"


def test_comms_tab_columns_are_resizable():
    app = _app()
    assert app is not None
    from PyQt6.QtWidgets import QHeaderView

    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    tab = HMICommsWidget(HMIProject())
    header = tab.tags_table.horizontalHeader()
    assert (
        header.sectionResizeMode(0)
        == QHeaderView.ResizeMode.Interactive
    )
    tab.tags_table.setColumnWidth(0, 222)
    assert tab.tags_table.columnWidth(0) == 222


def test_comms_tab_preserves_tag_string_length():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "plc",
                "driver": "dummy",
                "config": {},
                "tags": [
                    {
                        "name": "message",
                        "address": "message",
                        "type": "str",
                        "access": "rw",
                        "length": 4,
                    }
                ],
            }
        ],
    }
    tab = HMICommsWidget(project)
    assert tab.tags_table.item(0, 7).text() == "4"
    assert tab.flush_to_project()
    assert project.comms_config["devices"][0]["tags"][0]["length"] == 4


def test_comms_tab_device_id_updates_device_list_live():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    tab = HMICommsWidget(project)
    tab._add_device()
    tab.device_id.setText("plc")

    assert tab.device_list.item(0).text() == "plc  [dummy]"


def test_comms_tab_type_and_access_are_enum_controls():
    app = _app()
    assert app is not None
    from PyQt6.QtWidgets import QComboBox

    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    tab = HMICommsWidget(project)
    tab._add_device()
    tab._add_tag()

    type_combo = tab.tags_table.cellWidget(0, 2)
    access_combo = tab.tags_table.cellWidget(0, 3)
    assert isinstance(type_combo, QComboBox)
    assert isinstance(access_combo, QComboBox)
    assert [type_combo.itemText(i) for i in range(type_combo.count())] == [
        "bool", "int", "float", "str"
    ]
    assert [access_combo.itemText(i) for i in range(access_combo.count())] == [
        "r", "rw"
    ]
    assert type_combo.currentText() == "bool"
    assert access_combo.currentText() == "rw"
    assert "on demand" in tab.tag_help.text()
    assert "Driver-specific" in tab.tags_table.horizontalHeaderItem(1).toolTip()

    type_combo.setCurrentText("float")
    access_combo.setCurrentText("r")
    tab.tags_table.item(0, 5).setText("0")
    assert tab.flush_to_project()
    tag = project.comms_config["devices"][0]["tags"][0]
    assert tag["type"] == "float"
    assert tag["access"] == "r"


def test_comms_tab_device_id_migrates_tag_widget_references():
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
                "config": {},
                "tags": [
                    {
                        "name": "count",
                        "address": "count",
                        "type": "int",
                        "access": "rw",
                    }
                ],
            }
        ],
    }
    project.layout["widgets"] = [
        {
            "id": "measure",
            "type": "MeasureLabel",
            "source": "tag",
            "tag": "dummy.count",
        }
    ]
    project.script = "value = comms.read('dummy.count')"
    tab = HMICommsWidget(project)
    tab.device_id.setText("plc")

    assert tab.flush_to_project()
    assert project.comms_config["devices"][0]["id"] == "plc"
    assert project.layout["widgets"][0]["tag"] == "plc.count"
    assert "Updated 1 tag-bound widget" in tab.status.text()
    assert "dummy" in tab.status.text()
    assert "script" in tab.status.text().lower()


def test_comms_tab_device_config_is_in_top_scroll_area():
    app = _app()
    assert app is not None
    from PyQt6.QtWidgets import QScrollArea

    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    tab = HMICommsWidget(HMIProject())
    assert isinstance(tab.device_scroll, QScrollArea)
    assert tab.device_scroll.widgetResizable()
    assert tab.device_scroll.widget() is tab.device_group
    assert tab.vertical_splitter.widget(0) is tab.device_scroll
    assert tab.vertical_splitter.widget(1) is tab.tags_group

    tab.resize(900, 500)
    tab.show()
    app.processEvents()
    tab.vertical_splitter.setSizes([1, 999])
    app.processEvents()
    assert tab.vertical_splitter.sizes()[0] < 150
    assert tab.device_scroll.verticalScrollBar().maximum() > 0
    tab.close()
