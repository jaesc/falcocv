"""Communications core: configuration, tag store, dummy driver and manager."""

from __future__ import annotations

import time

import pytest

from falcocv.comms import (
    CommsConfigError,
    CommsError,
    CommsManager,
    CommsMonitorSession,
    DeviceDriver,
    parse_comms_config,
    validate_comms_config,
)


def _config(*tag_overrides, device_overrides=None) -> dict:
    device = {
        "id": "dummy",
        "driver": "dummy",
        "tags": [
            {
                "name": "trigger",
                "type": "bool",
                "access": "rw",
                "default": False,
            },
            {
                "name": "count",
                "type": "int",
                "access": "rw",
                "default": 0,
            },
            {
                "name": "message",
                "type": "str",
                "access": "rw",
                "default": "",
            },
        ],
    }
    device.update(device_overrides or {})
    device["tags"] = list(device["tags"]) + list(tag_overrides)
    return {"version": 1, "devices": [device]}


def test_dummy_driver_is_registered_but_not_offered():
    from falcocv.comms.drivers import available_drivers, registered_drivers

    assert "dummy" in registered_drivers()
    assert "dummy" not in available_drivers()


def test_parse_comms_config_builds_tag_identifiers():
    config = parse_comms_config(_config())

    assert config.version == 1
    assert len(config.devices) == 1
    dummy = config.devices[0]
    assert dummy.id == "dummy"
    assert [tag.id for tag in dummy.tags] == [
        "dummy.trigger",
        "dummy.count",
        "dummy.message",
    ]


@pytest.mark.parametrize(
    "bad_config",
    [
        {"version": 2, "devices": []},
        {"version": 1, "devices": {}},
        {
            "version": 1,
            "devices": [
                {"id": "bad.id", "driver": "dummy", "tags": []}
            ],
        },
        {
            "version": 1,
            "devices": [
                {"id": "dummy", "driver": "missing", "tags": []}
            ],
        },
        {
            "version": 1,
            "devices": [
                {
                    "id": "dummy",
                    "driver": "dummy",
                    "tags": [
                        {"name": "x", "type": "int", "default": "wrong"}
                    ],
                }
            ],
        },
    ],
)
def test_invalid_comms_config_is_rejected(bad_config):
    with pytest.raises(CommsConfigError):
        parse_comms_config(bad_config)


def test_validate_comms_config_reports_incomplete_fields():
    config = {
        "version": 1,
        "devices": [
            {
                "id": "plc",
                "driver": "siemens",
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

    result = validate_comms_config(config)

    assert result.errors == ()
    assert result.config is not None
    assert result.is_complete is False
    assert any("config.host is required" in item for item in result.warnings)
    assert any("address is required" in item for item in result.warnings)
    assert result.config.devices[0].tags[0].address == ""
    with pytest.raises(CommsConfigError, match="config.host is required"):
        parse_comms_config(config)


def test_validate_comms_config_keeps_structural_errors():
    result = validate_comms_config(
        {
            "version": 1,
            "devices": [
                {
                    "id": "plc",
                    "driver": "siemens",
                    "config": {},
                    "tags": [
                        {
                            "name": "trigger",
                            "type": "int",
                            "default": "not-an-int",
                        }
                    ],
                }
            ],
        }
    )

    assert result.config is None
    assert result.errors
    assert result.warnings == ()


def test_duplicate_tag_is_rejected():
    config = _config(
        {
            "name": "count",
            "type": "int",
            "default": 1,
        }
    )
    with pytest.raises(CommsConfigError, match="duplicate tag"):
        parse_comms_config(config)


def test_driver_schema_validates_known_config_fields():
    config = _config(device_overrides={"config": {"read_delay_ms": "fast"}})
    with pytest.raises(CommsConfigError, match="read_delay_ms"):
        parse_comms_config(config)


def test_dummy_manager_reads_writes_and_tracks_quality():
    manager = CommsManager(parse_comms_config(_config()))
    manager.start()
    try:
        assert manager.read("dummy.trigger") is False
        manager.write("dummy.trigger", True)
        assert manager.read("dummy.trigger") is True
        assert manager.tag("dummy.trigger").quality == "good"
        assert manager.read_many(["dummy.count", "dummy.message"]) == {
            "dummy.count": 0,
            "dummy.message": "",
        }
        manager.write_many({"dummy.count": 3, "dummy.message": "OK"})
        assert manager.read("dummy.count") == 3
        assert manager.read("dummy.message") == "OK"
    finally:
        manager.stop()


def test_dummy_read_failure_marks_tag_bad_then_recovers():
    config = _config(
        device_overrides={"config": {"fail_reads": 1}},
    )
    manager = CommsManager(parse_comms_config(config))
    manager.start()
    try:
        with pytest.raises(CommsError):
            manager.read("dummy.count")
        assert manager.tag("dummy.count").quality == "bad"
        assert manager.read("dummy.count") == 0
        assert manager.tag("dummy.count").quality == "good"
    finally:
        manager.stop()


def test_read_only_access_blocks_writes():
    config = _config(
        {"name": "readonly", "type": "int", "access": "r", "default": 1},
    )
    manager = CommsManager(parse_comms_config(config))
    manager.start()
    try:
        with pytest.raises(CommsError, match="read-only"):
            manager.write("dummy.readonly", 5)
    finally:
        manager.stop()


def test_write_only_access_is_rejected():
    config = _config(
        {"name": "writeonly", "type": "int", "access": "w", "default": 2},
    )
    with pytest.raises(CommsConfigError, match="access must be one of"):
        parse_comms_config(config)


def test_wait_for_value_and_timeout():
    manager = CommsManager(parse_comms_config(_config()))
    manager.start()
    try:
        manager.write("dummy.trigger", True)
        value = manager.wait_for("dummy.trigger", True, timeout=1.0)
        assert value.value is True

        with pytest.raises(TimeoutError):
            manager.wait_for("dummy.trigger", False, timeout=0.05)
    finally:
        manager.stop()


def test_poll_updates_tag_store():
    config = _config(
        {"name": "polled", "type": "int", "access": "rw", "default": 11, "poll_ms": 10},
    )
    manager = CommsManager(parse_comms_config(config))
    manager.start()
    try:
        # Change the driver value behind the manager's back and wait for the
        # polling thread to publish it to the tag store.
        manager._devices["dummy"].driver._values["polled"] = 12
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            value = manager.tag("dummy.polled")
            if value is not None and value.value == 12:
                break
            time.sleep(0.01)
        assert manager.tag("dummy.polled").value == 12
        assert manager.tag("dummy.polled").quality == "good"
    finally:
        manager.stop()


class _BatchDriver(DeviceDriver):
    driver_name = "batch"

    def __init__(self, config):
        super().__init__(config)
        self._connected = False
        self.values = {tag.name: tag.default for tag in config.tags}
        self.read_batches = []
        self.write_batches = []

    def start(self):
        self._connected = True

    def stop(self):
        self._connected = False

    def is_connected(self):
        return self._connected

    def read(self, tag):
        self.read_batches.append([tag.id])
        return self.values[tag.name]

    def write(self, tag, value):
        self.write_batches.append([(tag.id, value)])
        self.values[tag.name] = value

    def read_many(self, tags):
        self.read_batches.append([tag.id for tag in tags])
        return {tag.id: self.values[tag.name] for tag in tags}

    def write_many(self, items):
        self.write_batches.append([(tag.id, value) for tag, value in items])
        for tag, value in items:
            self.values[tag.name] = value


def test_dummy_simulated_signals_change_without_writes():
    config = _config(
        {"name": "toggle", "type": "bool", "access": "r", "default": False},
        {"name": "sine", "type": "float", "access": "r", "default": 0.0},
        device_overrides={
            "config": {
                "simulate": {
                    "count": {"kind": "counter", "period_ms": 5, "step": 2},
                    "toggle": {"kind": "toggle", "period_ms": 10},
                    "sine": {
                        "kind": "sine",
                        "period_ms": 20,
                        "min": 10.0,
                        "max": 20.0,
                    },
                }
            }
        },
    )
    manager = CommsManager(parse_comms_config(config))
    manager.start()
    try:
        time.sleep(0.03)
        counter = manager.read("dummy.count")
        toggle = manager.read("dummy.toggle")
        sine = manager.read("dummy.sine")
        assert isinstance(counter, int)
        assert counter > 0
        assert isinstance(toggle, bool)
        assert 10.0 <= sine <= 20.0
    finally:
        manager.stop()


def test_monitor_session_refreshes_and_writes():
    session = CommsMonitorSession(parse_comms_config(_config()))
    session.start()
    try:
        values = session.refresh()
        assert values["dummy.count"].value == 0
        session.write("dummy.count", 8)
        values = session.refresh()
        assert values["dummy.count"].value == 8
        assert values["dummy.count"].quality == "good"
    finally:
        session.stop()


def test_manager_uses_driver_batch_hooks(monkeypatch):
    manager = CommsManager(parse_comms_config(_config()))
    driver = _BatchDriver(manager.config.devices[0])
    monkeypatch.setattr(
        "falcocv.comms.manager.create_driver",
        lambda _config: driver,
    )
    manager.start()
    try:
        values = manager.read_many(["dummy.trigger", "dummy.count"])
        assert values == {"dummy.trigger": False, "dummy.count": 0}
        assert driver.read_batches == [["dummy.trigger", "dummy.count"]]

        manager.write_many({"dummy.trigger": True, "dummy.count": 5})
        assert driver.write_batches == [
            [("dummy.trigger", True), ("dummy.count", 5)]
        ]
        assert manager.read_many(["dummy.trigger", "dummy.count"]) == {
            "dummy.trigger": True,
            "dummy.count": 5,
        }
    finally:
        manager.stop()


def test_polling_uses_driver_batch_hook(monkeypatch):
    config = _config()
    for tag in config["devices"][0]["tags"]:
        tag["poll_ms"] = 20
    manager = CommsManager(parse_comms_config(config))
    driver = _BatchDriver(manager.config.devices[0])
    monkeypatch.setattr(
        "falcocv.comms.manager.create_driver",
        lambda _config: driver,
    )
    manager.start()
    try:
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if any(len(batch) > 1 for batch in driver.read_batches):
                break
            time.sleep(0.01)
        assert any(len(batch) > 1 for batch in driver.read_batches)
    finally:
        manager.stop()


def test_manager_stop_is_idempotent_and_clears_runtime():
    manager = CommsManager(parse_comms_config(_config()))
    manager.start()
    manager.stop()
    manager.stop()
    assert manager.tag("dummy.count") is None
    assert manager.snapshot() == {}
