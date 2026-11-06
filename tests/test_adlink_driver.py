"""ADLINK DASK driver tests using a fake native API."""

from __future__ import annotations

import sys

import pytest

from falcocv.comms import CommsError, parse_comms_config
from falcocv.comms.drivers import available_drivers, get_driver_schema
from falcocv.comms.drivers import adlink as adlink_driver


class _FakeApi:
    directory_handle = None

    def __init__(self):
        self.di_lines = {0: {1: True, 2: False}}
        self.do_lines = {0: {1: False, 2: True}}
        self.ai_values = {0: 1.25, 1: -2.5}
        self.register_calls = []
        self.release_calls = []
        self.port_reads = 0
        self.port_read_error = False
        self.ao_writes = []
        self.verify_writes = 0
        self.reject_do_verify = 0

    def register(self):
        self.register_calls.append(True)
        return 0

    def release(self):
        self.release_calls.append(True)

    def read_di_line(self, port, line):
        return self.di_lines[port][line]

    def read_di_port(self, port):
        self.port_reads += 1
        if self.port_read_error:
            raise CommsError("port not supported")
        value = 0
        for line, state in self.di_lines[port].items():
            if state:
                value |= 1 << line
        return value

    def read_do_line(self, port, line):
        return self.do_lines[port][line]

    def read_do_port(self, port):
        self.port_reads += 1
        if self.port_read_error:
            raise CommsError("port not supported")
        value = 0
        for line, state in self.do_lines[port].items():
            if state:
                value |= 1 << line
        return value

    def write_do_line(self, port, line, value):
        if self.reject_do_verify:
            self.reject_do_verify -= 1
            return
        self.do_lines[port][line] = bool(value)

    def read_ai(self, channel, ai_range):
        return self.ai_values[channel]

    def write_ao(self, channel, value):
        self.ao_writes.append((channel, value))


def _config(tags=None, **overrides):
    config = {
        "card_type": "USB_1902",
        "card_number": 2,
        "ai_range": 1,
        "verify_writes": False,
        "write_retries": 0,
    }
    config.update(overrides)
    return {
        "version": 1,
        "devices": [
            {
                "id": "io",
                "driver": "adlink",
                "config": config,
                "tags": tags
                or [
                    {
                        "name": "input",
                        "address": "DI0.1",
                        "type": "bool",
                        "access": "r",
                    },
                    {
                        "name": "output",
                        "address": "DO0.1",
                        "type": "bool",
                        "access": "rw",
                    },
                    {
                        "name": "analog_in",
                        "address": "AI0",
                        "type": "float",
                        "access": "r",
                    },
                    {
                        "name": "analog_out",
                        "address": "AO0",
                        "type": "float",
                        "access": "rw",
                        "default": 0.5,
                    },
                ],
            }
        ],
    }


def _driver(tags=None, **overrides):
    return adlink_driver.AdlinkDaskDriver(
        parse_comms_config(_config(tags, **overrides)).devices[0]
    )


def test_adlink_schema_exposes_all_known_models_and_analog_range():
    schema = get_driver_schema("adlink")
    fields = {field.key: field for field in schema.fields}

    assert "adlink" in available_drivers()
    assert len(fields["card_type"].choices) == len(adlink_driver.ADLINK_MODELS)
    assert ("USB-7250", "USB_7250") in fields["card_type"].choices
    assert fields["card_type"].default == "USB_7250"
    assert fields["ai_range"].default == 1
    assert fields["verify_writes"].default is False
    assert fields["write_retries"].visible_when == ("verify_writes", True)
    assert fields["verify_delay_ms"].default == 20
    assert fields["verify_delay_ms"].visible_when == ("verify_writes", True)


def test_adlink_reads_digital_and_analog_and_batches_ports(monkeypatch):
    api = _FakeApi()
    monkeypatch.setattr(adlink_driver, "_load_dask_api", lambda *_args: api)
    driver = _driver(
        tags=[
            {
                "name": "in_a",
                "address": "DI0.1",
                "type": "bool",
                "access": "r",
            },
            {
                "name": "in_b",
                "address": "DI0.2",
                "type": "bool",
                "access": "r",
            },
            {
                "name": "analog_in",
                "address": "AI0",
                "type": "float",
                "access": "r",
            },
            {
                "name": "analog_out",
                "address": "AO0",
                "type": "float",
                "access": "rw",
                "default": 0.5,
            },
        ]
    )
    driver.start()
    try:
        assert api.register_calls
        values = driver.read_many(driver.config.tags)
        assert values == {
            "io.in_a": True,
            "io.in_b": False,
            "io.analog_in": 1.25,
            "io.analog_out": 0.5,
        }
        assert api.port_reads == 1
        assert driver.read(driver.config.tags[0]) is True
    finally:
        driver.stop()
    assert api.release_calls


def test_adlink_writes_digital_and_analog(monkeypatch):
    api = _FakeApi()
    monkeypatch.setattr(adlink_driver, "_load_dask_api", lambda *_args: api)
    driver = _driver()
    driver.start()
    try:
        tags = {tag.name: tag for tag in driver.config.tags}
        driver.write(tags["output"], True)
        driver.write(tags["analog_out"], 4.5)
        assert api.do_lines[0][1] is True
        assert api.ao_writes == [(0, 4.5)]
        assert driver.read(tags["analog_out"]) == 4.5
    finally:
        driver.stop()


def test_adlink_verify_writes_retries_only_after_mismatch(monkeypatch):
    api = _FakeApi()
    api.reject_do_verify = 1
    monkeypatch.setattr(adlink_driver, "_load_dask_api", lambda *_args: api)
    driver = _driver(verify_writes=True, write_retries=1)
    driver.start()
    try:
        tag = next(tag for tag in driver.config.tags if tag.name == "output")
        driver.write(tag, True)
        assert api.do_lines[0][1] is True
    finally:
        driver.stop()


def test_adlink_verify_writes_fails_after_retries(monkeypatch):
    api = _FakeApi()
    api.reject_do_verify = 2
    monkeypatch.setattr(adlink_driver, "_load_dask_api", lambda *_args: api)
    driver = _driver(verify_writes=True, write_retries=1)
    driver.start()
    try:
        tag = next(tag for tag in driver.config.tags if tag.name == "output")
        with pytest.raises(CommsError, match="verification"):
            driver.write(tag, True)
    finally:
        driver.stop()


@pytest.mark.parametrize(
    "tag, message",
    [
        (
            {
                "name": "bad",
                "address": "DI0.1",
                "type": "bool",
                "access": "rw",
            },
            "access 'r'",
        ),
        (
            {
                "name": "bad",
                "address": "AI0",
                "type": "bool",
                "access": "r",
            },
            "type 'float'",
        ),
        (
            {
                "name": "bad",
                "address": "0.1",
                "type": "bool",
                "access": "r",
            },
            "must be DI",
        ),
    ],
)
def test_adlink_rejects_invalid_tag_contracts(tag, message):
    with pytest.raises(CommsError, match=message):
        _driver(tags=[tag])


def test_adlink_read_many_falls_back_to_lines_when_ports_are_unsupported(
    monkeypatch,
):
    api = _FakeApi()
    api.port_read_error = True
    monkeypatch.setattr(adlink_driver, "_load_dask_api", lambda *_args: api)
    driver = _driver(
        tags=[
            {
                "name": "in_a",
                "address": "DI0.1",
                "type": "bool",
                "access": "r",
            },
            {
                "name": "in_b",
                "address": "DI0.2",
                "type": "bool",
                "access": "r",
            },
        ]
    )
    driver.start()
    try:
        assert driver.read_many(driver.config.tags) == {
            "io.in_a": True,
            "io.in_b": False,
        }
    finally:
        driver.stop()


def test_adlink_register_uses_selected_card_number(monkeypatch):
    calls = []

    class _RegisterApi(_FakeApi):
        def register(self):
            calls.append(True)
            return 0

    api = _RegisterApi()
    monkeypatch.setattr(
        adlink_driver,
        "_load_dask_api",
        lambda _model, _path, card_number: (
            calls.append(card_number) or api
        ),
    )
    driver = _driver(card_number=3)
    driver.start()
    driver.stop()
    assert calls == [3, True]


def test_adlink_card_number_comes_from_config():
    driver = _driver()
    assert driver._card_number == 2


def test_adlink_ao_read_is_cached_value():
    api = _FakeApi()
    driver = _driver()
    driver._api = api
    driver._connected = True
    tag = next(tag for tag in driver.config.tags if tag.name == "analog_out")
    driver.write(tag, 2.25)
    assert driver.read(tag) == 2.25


@pytest.mark.skipif(
    sys.platform != "win32", reason="ADLINK DASK is Windows-only"
)
def test_adlink_dependency_status_and_probe(monkeypatch, tmp_path):
    from pathlib import Path

    from falcocv.core.native_deps import NativeLibrary

    native_dll = tmp_path / "USB-Dask64.dll"
    native_dll.write_bytes(b"fake")
    monkeypatch.setattr(
        adlink_driver,
        "resolve_library",
        lambda *_args, **_kwargs: NativeLibrary(
            Path(native_dll), "test"
        ),
    )
    statuses = adlink_driver.AdlinkDaskDriver.check_dependencies(
        {"card_type": "USB_7250"}
    )
    assert statuses[0].state == "ok"
    assert statuses[0].test_kind == "adlink_card"

    api = _FakeApi()
    monkeypatch.setattr(adlink_driver, "_load_dask_api", lambda *_args: api)
    assert adlink_driver.AdlinkDaskDriver.probe(
        {"card_type": "USB_7250", "card_number": 2}
    ) == "USB-7250 detected (card 2)"
    assert api.register_calls
    assert api.release_calls


@pytest.mark.skipif(
    sys.platform != "win32", reason="ADLINK DASK is Windows-only"
)
def test_adlink_dependency_status_is_missing_without_runtime(monkeypatch):
    from falcocv.core.native_deps import NativeLibrary

    monkeypatch.setattr(
        adlink_driver,
        "resolve_library",
        lambda *_args, **_kwargs: NativeLibrary(None, "not found"),
    )
    status = adlink_driver.AdlinkDaskDriver.check_dependencies(
        {"card_type": "USB_7250"}
    )[0]
    assert status.state == "missing"
    assert "USB-DASK 64-bit" in status.detail


def test_usb7250_capabilities_reject_unsupported_tags():
    with pytest.raises(CommsError, match="analog inputs"):
        _driver(
            card_type="USB_7250",
            tags=[
                {
                    "name": "ai",
                    "address": "AI0",
                    "type": "float",
                    "access": "r",
                }
            ],
        )
    with pytest.raises(CommsError, match="no digital output port 1"):
        _driver(
            card_type="USB_7250",
            tags=[
                {
                    "name": "do",
                    "address": "DO1.0",
                    "type": "bool",
                    "access": "rw",
                }
            ],
        )
    with pytest.raises(CommsError, match="no digital line 8"):
        _driver(
            card_type="USB_7250",
            tags=[
                {
                    "name": "di",
                    "address": "DI0.8",
                    "type": "bool",
                    "access": "r",
                }
            ],
        )


def test_adlink_error_codes_are_translated():
    assert "-302" in adlink_driver._dask_error_message(
        "Register_Card", -302
    )
    assert "USB-DASK" in adlink_driver._dask_error_message(
        "Register_Card", -302
    )
    assert "digital port" in adlink_driver._dask_error_message(
        "DI_ReadPort", -31
    )


def test_adlink_card_type_can_select_pci_model():
    driver = _driver(card_type="PCI_7250")
    assert driver._model.label == "PCI-7250"
    assert driver._model.card_type == 11
    assert adlink_driver.adlink_dll_names("PCI_7250") == (
        adlink_driver.adlink_dll_names("USB_7250")[0].replace(
            "USB-Dask", "PCI-Dask"
        ),
    )
