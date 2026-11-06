"""HMI project integration for configured protocol servers."""

from __future__ import annotations

import socket

import pytest

pytest.importorskip("pymodbus")


def _free_port() -> int:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return port


def test_hmi_project_starts_and_stops_modbus_server(tmp_path):
    from pymodbus.client import ModbusTcpClient

    from falcocv.core.hmi import HMIProject

    port = _free_port()
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
                        "default": 4,
                    }
                ],
            }
        ],
    }
    project.servers_config = {
        "version": 1,
        "servers": [
            {
                "id": "mb",
                "type": "modbus_tcp_server",
                "enabled": True,
                "host": "127.0.0.1",
                "port": port,
                "unit_id": 1,
                "mappings": [
                    {
                        "address": "holding:0",
                        "tag": "dummy.count",
                        "access": "rw",
                    }
                ],
            }
        ],
    }

    project.start_runtime()
    client = ModbusTcpClient("127.0.0.1", port=port, timeout=2)
    try:
        assert project.comms is not None
        assert project.comms.read("dummy.count") == 4
        assert client.connect()
        assert client.read_holding_registers(
            0, count=1, device_id=1
        ).registers == [4]
        assert not client.write_register(0, 17, device_id=1).isError()
        assert project.comms.read("dummy.count") == 17
    finally:
        client.close()
        project.stop_runtime()
    assert project.comms is None

    path = tmp_path / "servers.hmi"
    project.save(path)
    loaded = HMIProject.load(path)
    assert loaded.servers_config["servers"][0]["id"] == "mb"


def test_hmi_designer_servers_tab_is_present():
    from PyQt6.QtWidgets import QApplication

    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_designer_dialog import HMIDesignerDialog

    app = QApplication.instance() or QApplication([])
    dialog = HMIDesignerDialog(HMIProject())
    try:
        labels = [
            dialog._tabs.tabText(index)
            for index in range(dialog._tabs.count())
        ]
        assert "Servers" in labels
    finally:
        dialog._modified = False
        dialog.close()
    assert app is not None
