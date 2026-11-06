"""Offscreen tests for window/panel view-state persistence."""

from __future__ import annotations

import pytest

from falcocv.ui import view_state


class _FakeConfig:
    def __init__(self):
        self.data = {
            "views": {
                "windows": {},
                "splitters": {},
                "headers": {},
            }
        }
        self.saved = 0

    def save(self):
        self.saved += 1


def _app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


@pytest.fixture
def fake_config(monkeypatch):
    config = _FakeConfig()
    monkeypatch.setattr(view_state, "ConfigManager", lambda: config)
    return config


def test_window_geometry_and_splitter_round_trip(fake_config):
    app = _app()
    assert app is not None
    from PyQt6.QtWidgets import QSplitter, QWidget

    window = QWidget()
    window.resize(640, 480)
    view_state.save_window(window, "test")
    window.resize(200, 200)
    assert view_state.restore_window(window, "test", (300, 200))
    assert window.size().width() == 640
    assert window.size().height() == 480

    splitter = QSplitter()
    splitter.addWidget(QWidget())
    splitter.addWidget(QWidget())
    splitter.resize(800, 200)
    splitter.show()
    app.processEvents()
    splitter.setSizes([300, 500])
    view_state.save_splitter(splitter, "test.splitter")
    splitter.setSizes([100, 100])
    assert view_state.restore_splitter(
        splitter, "test.splitter", [200, 600]
    )
    sizes = splitter.sizes()
    assert sizes[0] > 200
    assert sizes[1] > 400
    splitter.close()


def test_header_round_trip_and_reset(fake_config):
    app = _app()
    assert app is not None
    from PyQt6.QtWidgets import QTableWidget

    table = QTableWidget(0, 2)
    header = table.horizontalHeader()
    table.setColumnWidth(0, 222)
    table.setColumnWidth(1, 111)
    view_state.save_header(header, "test.header")
    table.setColumnWidth(0, 10)
    table.setColumnWidth(1, 10)
    assert view_state.restore_header(header, "test.header")
    assert table.columnWidth(0) == 222
    assert table.columnWidth(1) == 111

    view_state.reset_keys(("test",))
    assert fake_config.data["views"]["headers"] == {}
    assert fake_config.data["views"]["splitters"] == {}


def test_hmi_communications_panels_reset_to_defaults(fake_config):
    app = _app()
    assert app is not None
    from falcocv.core.hmi import HMIProject
    from falcocv.ui.widgets.hmi_comms_monitor import HMICommsMonitorWidget
    from falcocv.ui.widgets.hmi_comms_tab import HMICommsWidget

    project = HMIProject()
    comms = HMICommsWidget(project)
    comms.resize(1000, 700)
    comms.show()
    app.processEvents()
    comms.horizontal_splitter.setSizes([10, 10])
    comms.vertical_splitter.setSizes([10, 10])
    comms.reset_view_state()
    assert comms.horizontal_splitter.sizes()[0] > 100
    assert comms.horizontal_splitter.sizes()[1] > 100
    assert comms.vertical_splitter.sizes()[0] > 100
    assert comms.vertical_splitter.sizes()[1] > 100

    monitor = HMICommsMonitorWidget(project)
    monitor.table.setColumnWidth(0, 10)
    monitor.reset_view_state()
    assert monitor.table.columnWidth(0) == 220
    assert monitor._format_value(None) == "—"
