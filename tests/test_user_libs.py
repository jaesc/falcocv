"""User library directories and the ``uv`` package manager."""

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from falcocv.core import user_libs


def _directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def test_user_dirs_orders_env_then_bundle_then_user_packages(monkeypatch, tmp_path):
    env_dir = _directory(tmp_path / "env")
    site = _directory(tmp_path / "site-packages")
    packages = _directory(tmp_path / "user-packages")
    monkeypatch.setenv(user_libs.USER_PACKAGES_ENV, str(env_dir))
    monkeypatch.setattr(user_libs, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(user_libs, "user_packages_dir", lambda: packages)

    directories = user_libs.user_dirs()

    assert directories == [env_dir, site, packages]


def test_user_dirs_skips_missing_and_deduplicates(monkeypatch, tmp_path):
    shared = _directory(tmp_path / "shared")
    monkeypatch.setenv(
        user_libs.USER_PACKAGES_ENV, f"{shared}{os.pathsep}{shared}"
    )
    monkeypatch.setattr(user_libs, "app_dir", lambda: tmp_path)
    monkeypatch.setattr(user_libs, "user_packages_dir", lambda: shared)

    assert user_libs.user_dirs() == [shared]


def test_install_user_libs_appends_and_never_shadows(monkeypatch, tmp_path):
    library = _directory(tmp_path / "libs")
    (library / "numpy.py").write_text(
        "raise RuntimeError('the user copy must not win')\n", encoding="utf-8"
    )
    monkeypatch.setattr(user_libs, "user_dirs", lambda: [library])
    original = list(sys.path)
    try:
        added = user_libs.install_user_libs()

        assert added == [str(library)]
        assert sys.path[-1] == str(library)
        assert user_libs.install_user_libs() == []

        spec = importlib.util.find_spec("numpy")
        assert spec is not None
        assert spec.origin != str(library / "numpy.py")
    finally:
        sys.path[:] = original


def test_bundled_modules_reports_packages_already_in_the_bundle():
    assert "numpy" in user_libs.bundled_modules(["numpy"])
    assert user_libs.bundled_modules(["definitely-not-a-package-xyz"]) == []


def test_list_and_remove_packages(monkeypatch, tmp_path):
    target = _directory(tmp_path / "user-packages")
    dist = target / "demo_pkg-1.2.3.dist-info"
    dist.mkdir()
    module = target / "demo_pkg.py"
    module.write_text("VALUE = 1\n", encoding="utf-8")
    (dist / "RECORD").write_text(
        "demo_pkg.py,,\ndemo_pkg-1.2.3.dist-info/RECORD,,\n", encoding="utf-8"
    )
    monkeypatch.setattr(user_libs, "user_packages_dir", lambda: target)

    assert user_libs.list_packages() == [
        {"name": "demo_pkg", "version": "1.2.3", "path": str(dist)}
    ]

    ok, message = user_libs.remove_package("demo-pkg")

    assert ok and "removed" in message
    assert not module.exists()
    assert not dist.exists()


def test_install_packages_builds_the_uv_command(monkeypatch, tmp_path):
    uv = tmp_path / ("uv.exe" if sys.platform == "win32" else "uv")
    uv.write_text("", encoding="utf-8")
    target = tmp_path / "user-packages"
    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        return subprocess.CompletedProcess(command, 0, "installed", "")

    monkeypatch.setattr(user_libs, "find_uv", lambda: uv)
    monkeypatch.setattr(user_libs, "user_packages_dir", lambda: target)
    monkeypatch.setattr(user_libs.subprocess, "run", fake_run)

    ok, output = user_libs.install_packages(
        ["pandas"], wheelhouse=str(tmp_path / "wheels")
    )

    assert ok and "installed" in output
    command = captured["command"]
    assert command[:4] == [str(uv), "pip", "install", "--target"]
    assert str(target) in command
    assert "--only-binary=:all:" in command
    assert "--find-links" in command and "--no-index" in command
    assert "pandas" in command
    assert target.is_dir()


def test_install_packages_without_uv_reports_a_clear_error(monkeypatch):
    monkeypatch.setattr(user_libs, "find_uv", lambda: None)

    ok, message = user_libs.install_packages(["pandas"])

    assert not ok
    assert "uv" in message
