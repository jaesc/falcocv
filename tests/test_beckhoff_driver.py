"""Beckhoff ADS driver tests using a fake pyads client."""

from __future__ import annotations

import pytest

from falcocv.comms import CommsError, parse_comms_config
from falcocv.comms.drivers import available_drivers, get_driver_schema
from falcocv.comms.drivers import beckhoff as beckhoff_driver


class _FakeClient:
    def __init__(
        self,
        values,
        reject_writes=0,
        write_status=None,
        probe_error=False,
    ):
        self.values = dict(values)
        self.reject_writes = reject_writes
        self.write_status = write_status
        self.probe_error = probe_error
        self.is_open = False
        self.read_calls = 0
        self.write_calls = 0
        self.last_read = []
        self.last_write = {}

    def open(self):
        self.is_open = True

    def close(self):
        self.is_open = False

    def read_state(self):
        if self.probe_error:
            raise OSError("target unreachable")
        if not self.is_open:
            raise OSError("not open")
        return (0, 0)

    def read_list_by_name(self, names):
        self.read_calls += 1
        self.last_read = list(names)
        return {name: self.values[name] for name in names}

    def write_list_by_name(self, values):
        self.write_calls += 1
        self.last_write = dict(values)
        if self.reject_writes:
            self.reject_writes -= 1
            return {name: "no error" for name in values}
        for name, value in values.items():
            self.values[name] = value
        status = self.write_status or "no error"
        return {name: status for name in values}


def _config(verify=False, retries=0):
    return {
        "version": 1,
        "devices": [
            {
                "id": "ads",
                "driver": "beckhoff",
                "config": {
                    "ams_net_id": "192.168.1.10.1.1",
                    "ams_port": 851,
                    "verify_writes": verify,
                    "write_retries": retries,
                },
                "tags": [
                    {
                        "name": "trigger",
                        "address": "MAIN.bTrigger",
                        "type": "bool",
                        "access": "rw",
                    },
                    {
                        "name": "count",
                        "address": "MAIN.nCount",
                        "type": "int",
                        "access": "rw",
                    },
                ],
            }
        ],
    }


def _device(verify=False, retries=0):
    return parse_comms_config(_config(verify, retries)).devices[0]


def test_beckhoff_schema_and_optional_ip():
    schema = get_driver_schema("beckhoff")
    fields = {field.key: field for field in schema.fields}
    assert "beckhoff" in available_drivers()
    assert fields["verify_writes"].default is False
    assert "ip_address" in fields
    assert fields["write_retries"].default == 0
    assert fields["write_retries"].visible_when == ("verify_writes", True)

    config = _config()
    config["devices"][0]["config"]["ip_address"] = "10.0.0.50"
    assert parse_comms_config(config).devices[0].config["ip_address"] == "10.0.0.50"


def test_beckhoff_read_batches_symbols(monkeypatch):
    client = _FakeClient({"MAIN.bTrigger": False, "MAIN.nCount": 7})
    monkeypatch.setattr(beckhoff_driver, "_create_client", lambda _config: client)
    device = _device()
    driver = beckhoff_driver.BeckhoffAdsDriver(device)

    driver.start()
    try:
        values = driver.read_many([device.tags[0], device.tags[1]])
        assert values == {"ads.trigger": False, "ads.count": 7}
        assert client.read_calls == 1
        assert client.last_read == ["MAIN.bTrigger", "MAIN.nCount"]
    finally:
        driver.stop()


def test_beckhoff_write_batches_and_does_not_verify_by_default(monkeypatch):
    client = _FakeClient({"MAIN.bTrigger": False, "MAIN.nCount": 7})
    monkeypatch.setattr(beckhoff_driver, "_create_client", lambda _config: client)
    device = _device()
    driver = beckhoff_driver.BeckhoffAdsDriver(device)

    driver.start()
    try:
        driver.write_many([(device.tags[0], True), (device.tags[1], 9)])
        assert client.write_calls == 1
        assert client.last_write == {"MAIN.bTrigger": True, "MAIN.nCount": 9}
        assert client.read_calls == 0
    finally:
        driver.stop()


def test_beckhoff_verify_write_reads_once(monkeypatch):
    client = _FakeClient({"MAIN.nCount": 7})
    monkeypatch.setattr(beckhoff_driver, "_create_client", lambda _config: client)
    device = _device(verify=True)
    driver = beckhoff_driver.BeckhoffAdsDriver(device)

    driver.start()
    try:
        driver.write(device.tags[1], 9)
        assert client.write_calls == 1
        assert client.read_calls == 1
    finally:
        driver.stop()


def test_beckhoff_verify_failure_does_not_retry(monkeypatch):
    client = _FakeClient({"MAIN.nCount": 7}, reject_writes=1)
    monkeypatch.setattr(beckhoff_driver, "_create_client", lambda _config: client)
    device = _device(verify=True)
    driver = beckhoff_driver.BeckhoffAdsDriver(device)

    driver.start()
    try:
        with pytest.raises(CommsError, match="verification"):
            driver.write(device.tags[1], 9)
        assert client.write_calls == 1
        assert client.read_calls == 1
    finally:
        driver.stop()


def test_beckhoff_verification_retries_until_match(monkeypatch):
    client = _FakeClient({"MAIN.nCount": 7}, reject_writes=1)
    monkeypatch.setattr(beckhoff_driver, "_create_client", lambda _config: client)
    device = _device(verify=True, retries=1)
    driver = beckhoff_driver.BeckhoffAdsDriver(device)

    driver.start()
    try:
        driver.write(device.tags[1], 9)
        assert client.write_calls == 2
        assert client.read_calls == 2
        assert client.values["MAIN.nCount"] == 9
    finally:
        driver.stop()


def test_beckhoff_verify_false_ignores_retries(monkeypatch):
    client = _FakeClient({"MAIN.nCount": 7})
    monkeypatch.setattr(beckhoff_driver, "_create_client", lambda _config: client)
    device = _device(verify=False, retries=3)
    driver = beckhoff_driver.BeckhoffAdsDriver(device)

    driver.start()
    try:
        driver.write(device.tags[1], 9)
        assert client.write_calls == 1
        assert client.read_calls == 0
    finally:
        driver.stop()


def test_beckhoff_write_status_error_is_reported(monkeypatch):
    client = _FakeClient({"MAIN.nCount": 7}, write_status="symbol not found")
    monkeypatch.setattr(beckhoff_driver, "_create_client", lambda _config: client)
    device = _device(verify=True, retries=3)
    driver = beckhoff_driver.BeckhoffAdsDriver(device)

    driver.start()
    try:
        with pytest.raises(CommsError, match="symbol not found"):
            driver.write(device.tags[1], 9)
        assert client.write_calls == 1
    finally:
        driver.stop()


def test_beckhoff_unreachable_target_fails_start(monkeypatch):
    client = _FakeClient({}, probe_error=True)
    monkeypatch.setattr(beckhoff_driver, "_create_client", lambda _config: client)
    driver = beckhoff_driver.BeckhoffAdsDriver(_device())

    with pytest.raises(CommsError, match="reach"):
        driver.start()
    assert client.is_open is False


def test_beckhoff_dependency_status_reports_route(monkeypatch, tmp_path):
    from falcocv.core.native_deps import NativeLibrary

    dll = tmp_path / "TcAdsDll.dll"
    dll.write_bytes(b"fake")
    monkeypatch.setattr(
        beckhoff_driver,
        "resolve_tcads_dll",
        lambda: NativeLibrary(dll, "test"),
    )
    monkeypatch.setattr(
        beckhoff_driver.importlib.util,
        "find_spec",
        lambda name: object() if name == "pyads" else None,
    )

    statuses = beckhoff_driver.BeckhoffAdsDriver.check_dependencies({})
    states = {status.key: status.state for status in statuses}
    assert states["tcads"] == "ok"
    assert states["ads_route"] == "unknown"


def test_beckhoff_probe_opens_and_closes_connection(monkeypatch):
    class _ProbeClient:
        def __init__(self):
            self.opened = False
            self.closed = False

        def open(self):
            self.opened = True

        def read_state(self):
            assert self.opened
            return (0, 0)

        def close(self):
            self.closed = True

    client = _ProbeClient()
    monkeypatch.setattr(
        beckhoff_driver, "_create_client", lambda _config: client
    )

    assert beckhoff_driver.BeckhoffAdsDriver.probe(
        {"ams_net_id": "192.168.1.10.1.1"}
    ) == "ADS connection OK"
    assert client.opened
    assert client.closed


def test_beckhoff_missing_pyads_is_reported(monkeypatch):
    def fail(_config):
        raise CommsError("driver 'beckhoff' requires the pyads package")

    monkeypatch.setattr(beckhoff_driver, "_create_client", fail)
    driver = beckhoff_driver.BeckhoffAdsDriver(_device())
    with pytest.raises(CommsError, match="pyads"):
        driver.start()
