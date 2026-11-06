"""Offscreen tests for the HMI Designer Servers tab."""

from __future__ import annotations


def _app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _project():
    from falcocv.core.hmi import HMIProject

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
                        "name": "trigger",
                        "address": "trigger",
                        "type": "bool",
                        "access": "rw",
                    },
                    {
                        "name": "count",
                        "address": "count",
                        "type": "int",
                        "access": "rw",
                    },
                ],
            }
        ],
    }
    return project


def test_servers_tab_adds_modbus_mapping_and_flushes():
    app = _app()
    assert app is not None
    from falcocv.ui.widgets.hmi_servers_tab import HMIServersWidget

    project = _project()
    tab = HMIServersWidget(project)
    assert tab.horizontal_splitter.objectName() == "hmi_servers.horizontal"
    assert tab.vertical_splitter.objectName() == "hmi_servers.vertical"
    assert (
        tab.mappings_table.horizontalHeader().objectName()
        == "hmi_servers.mappings"
    )
    tab._add_server()
    tab.server_id.setText("mb")
    tab._add_mapping()
    tab.mappings_table.item(0, 0).setText("coil:0")
    tab.mappings_table.item(0, 1).setText("plc.trigger")

    assert tab.flush_to_project()
    server = project.servers_config["servers"][0]
    assert server["id"] == "mb"
    assert server["type"] == "modbus_tcp_server"
    assert server["mappings"] == [
        {"address": "coil:0", "tag": "plc.trigger", "access": "r"}
    ]


def test_servers_tab_supports_opcua_and_validation(monkeypatch):
    app = _app()
    assert app is not None
    from PyQt6.QtWidgets import QMessageBox

    from falcocv.ui.widgets.hmi_servers_tab import HMIServersWidget

    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *_args, **_kwargs: QMessageBox.StandardButton.Yes,
    )
    project = _project()
    tab = HMIServersWidget(project)
    tab._add_server()
    tab.server_id.setText("ua")
    tab.server_type.setCurrentIndex(1)
    tab._add_mapping()
    tab.mappings_table.item(0, 0).setText("ns=2;s=Count")
    tab.mappings_table.item(0, 1).setText("plc.count")
    assert tab.flush_to_project()
    server = project.servers_config["servers"][0]
    assert server["type"] == "opcua_server"
    assert server["mappings"][0]["address"] == "ns=2;s=Count"
    assert tab.validate() == (True, "")

    tab.server_id.setText("")
    assert tab.flush_to_project() is False
    assert "id" in tab.status.text()
