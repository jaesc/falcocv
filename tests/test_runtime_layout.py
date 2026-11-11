"""Layout helpers for development and frozen (standalone) builds."""

import sys
from pathlib import Path

import falcocv.core.runtime as runtime


def test_is_frozen_is_false_in_development():
    assert runtime.is_frozen() is False


def test_app_dir_is_the_checkout_root_in_development():
    root = Path(__file__).resolve().parents[1]
    assert runtime.app_dir() == root


def test_bundle_dir_is_the_checkout_root_in_development():
    root = Path(__file__).resolve().parents[1]
    assert runtime.bundle_dir() == root


def test_server_executable_is_next_to_the_app(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "executable", str(tmp_path / "FalcoCV.exe"))

    server = runtime.server_executable()

    assert server.parent == tmp_path
    assert server.name.startswith("FalcoCVServer")
