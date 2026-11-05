"""Offscreen smoke test for the HMI Designer signal monitor."""

from __future__ import annotations

import time

from falcocv.comms import TagValue, parse_comms_config


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
                "id": "dummy",
                "driver": "dummy",
                "tags": [
                    {
                        "name": "count",
                        "type": "int",
                        "access": "rw",
                        "default": 0,
                    },
                    {
                        "name": "counter",
                        "type": "int",
                        "access": "r",
                        "default": 0,
                    },
                ],
                "config": {
                    "simulate": {
                        "counter": {
                            "kind": "counter",
                            "period_ms": 10,
                            "step": 1,
                        }
                    }
                },
            }
        ],
    }
    return project


def test_monitor_normalises_localhost_and_stale_callbacks():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_monitor import HMICommsMonitorWidget

    tab = HMICommsMonitorWidget(HMIProject())
    assert tab.live_url_edit.text().startswith("http://127.0.0.1")
    assert tab._normalise_live_url("http://localhost:8000") == (
        "http://127.0.0.1:8000"
    )
    assert tab._normalise_live_url("http://[::1]:8000") == (
        "http://127.0.0.1:8000"
    )

    old_token = object()
    tab._connection_token = old_token
    tab._set_connection_state("connected")
    tab.disconnect_monitor()
    tab._on_live_refreshed(
        {"tags": [], "values": {}, "statuses": []},
        old_token,
    )
    assert tab._connection_state == "disconnected"


def test_monitor_connection_state_updates_buttons():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_monitor import HMICommsMonitorWidget

    tab = HMICommsMonitorWidget(HMIProject())
    assert tab.connect_btn.isEnabled()
    assert not tab.disconnect_btn.isEnabled()

    tab._set_connection_state("connecting")
    assert not tab.connect_btn.isEnabled()
    assert tab.disconnect_btn.isEnabled()

    tab._set_connection_state("connected")
    assert tab.refresh_btn.isEnabled()
    assert not tab.connect_btn.isEnabled()

    tab._set_connection_state("error", "not deployed")
    assert tab.connect_btn.isEnabled()
    assert not tab.disconnect_btn.isEnabled()
    assert not tab.refresh_btn.isEnabled()


def test_monitor_enter_writes_selected_value():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_monitor import HMICommsMonitorWidget

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "dummy",
                "driver": "dummy",
                "tags": [
                    {
                        "name": "count",
                        "type": "int",
                        "access": "rw",
                        "default": 0,
                    }
                ],
            }
        ],
    }
    tab = HMICommsMonitorWidget(project)
    tag = parse_comms_config(project.comms_config).devices[0].tags[0]
    tab._tags = {tag.id: tag}
    tab._selected_tag_id = tag.id
    tab._connection_state = "connected"
    tab._worker = object()
    written = []
    tab.write_requested.connect(lambda name, value: written.append((name, value)))
    tab.value_edit.setText("7")
    tab.value_edit.returnPressed.emit()
    assert written == [("dummy.count", 7)]


def test_monitor_enter_is_consumed_without_default_button():
    app = _app()
    assert app is not None
    from PyQt6.QtCore import Qt
    from PyQt6.QtTest import QTest
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_monitor import HMICommsMonitorWidget

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "dummy",
                "driver": "dummy",
                "tags": [
                    {"name": "count", "type": "int", "access": "rw", "default": 0}
                ],
            }
        ],
    }
    tab = HMICommsMonitorWidget(project)
    tag = parse_comms_config(project.comms_config).devices[0].tags[0]
    tab._tags = {tag.id: tag}
    tab._snapshot = {tag.id: TagValue(0, "good", 1.0)}
    tab._connection_state = "connected"
    tab._worker = object()
    tab._set_connection_state("connected")
    tab._visible_tag_ids = []
    tab._populate_table()
    tab.table.selectRow(0)
    tab._on_selection_changed()
    written = []
    tab.write_requested.connect(lambda name, value: written.append((name, value)))

    tab.value_edit.setText("7")
    tab.value_edit.setFocus()
    QTest.keyClick(tab.value_edit, Qt.Key.Key_Return)
    app.processEvents()

    assert written == [("dummy.count", 7)]
    assert not tab.connect_btn.autoDefault()
    assert not tab.disconnect_btn.autoDefault()
    assert not tab.write_btn.autoDefault()


def test_monitor_bool_buttons_send_explicit_values():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_monitor import HMICommsMonitorWidget

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "dummy",
                "driver": "dummy",
                "tags": [
                    {
                        "name": "manual_flag",
                        "type": "bool",
                        "access": "rw",
                        "default": False,
                    }
                ],
            }
        ],
    }
    tab = HMICommsMonitorWidget(project)
    tag = parse_comms_config(project.comms_config).devices[0].tags[0]
    tab._tags = {tag.id: tag}
    tab._snapshot = {tag.id: TagValue(False, "good", 1.0)}
    tab._connection_state = "connected"
    tab._worker = object()
    tab._set_connection_state("connected")
    tab._visible_tag_ids = []
    tab._populate_table()
    tab.table.selectRow(0)
    tab._on_selection_changed()
    written = []
    tab.write_requested.connect(lambda name, value: written.append((name, value)))

    tab.bool_true_btn.click()
    assert written == [("dummy.manual_flag", True)]

    tab._on_write_done(tag.id, True, "")
    tab.bool_false_btn.click()
    assert written[-1] == ("dummy.manual_flag", False)


def test_monitor_updates_table_rows_in_place():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.comms import TagValue
    from falcocv.ui.widgets.hmi_comms_monitor import HMICommsMonitorWidget

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "dummy",
                "driver": "dummy",
                "tags": [
                    {"name": "count", "type": "int", "access": "rw", "default": 0}
                ],
            }
        ],
    }
    tab = HMICommsMonitorWidget(project)
    tag = parse_comms_config(project.comms_config).devices[0].tags[0]
    tab._tags = {tag.id: tag}
    tab._snapshot = {tag.id: TagValue(1, "good", 1.0)}
    tab._populate_table()
    item = tab.table.item(0, 1)
    tab._snapshot = {tag.id: TagValue(2, "good", 2.0)}
    tab._populate_table()
    assert tab.table.item(0, 1) is item
    assert item.text() == "2"


def test_monitor_live_payload_populates_table_and_values():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_monitor import HMICommsMonitorWidget

    tab = HMICommsMonitorWidget(HMIProject())
    tab._on_live_refreshed(
        {
            "tags": [
                {
                    "id": "dummy.count",
                    "device": "dummy",
                    "name": "count",
                    "address": "count",
                    "type": "int",
                    "access": "rw",
                    "poll_ms": 0,
                    "default": 0,
                }
            ],
            "values": {
                "dummy.count": {
                    "value": 9,
                    "quality": "good",
                    "timestamp": 1.0,
                    "error": "",
                }
            },
            "statuses": [{"connected": True}],
            "warning": "",
        }
    )
    assert tab.table.rowCount() == 1
    assert tab.table.item(0, 1).text() == "9"
    assert tab.status_label.text().startswith("Live:")


def test_monitor_does_not_overwrite_dirty_editor_on_refresh():
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_monitor import HMICommsMonitorWidget

    project = HMIProject()
    project.comms_config = {
        "version": 1,
        "devices": [
            {
                "id": "dummy",
                "driver": "dummy",
                "tags": [
                    {
                        "name": "message",
                        "type": "str",
                        "access": "rw",
                        "default": "",
                    }
                ],
            }
        ],
    }
    tab = HMICommsMonitorWidget(project)
    tab._tags = {
        "dummy.message": parse_comms_config(
            project.comms_config
        ).devices[0].tags[0]
    }
    tab._selected_tag_id = "dummy.message"
    tab._snapshot = {
        "dummy.message": TagValue("from-device", "good", 1.0)
    }
    tab.value_edit.setText("user edit")
    tab._editor_dirty = True

    tab._on_refreshed({"dummy.message": TagValue("changed", "good", 2.0)})

    assert tab.value_edit.text() == "user edit"


def test_monitor_widget_connects_lists_tags_and_stops():
    app = _app()
    assert app is not None
    from falcocv.ui.widgets.hmi_comms_monitor import HMICommsMonitorWidget

    tab = HMICommsMonitorWidget(_project())
    try:
        tab.connect_monitor()
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and not tab._snapshot:
            app.processEvents()
            time.sleep(0.01)
        assert tab.table.rowCount() == 2
        assert tab._snapshot
        assert "dummy.count" in tab._snapshot
    finally:
        tab.disconnect_monitor()
        tab.close()
