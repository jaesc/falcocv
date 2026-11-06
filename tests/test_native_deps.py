"""Native library resolution used by optional device drivers."""

from __future__ import annotations


def test_resolve_library_finds_native_driver_dir(monkeypatch, tmp_path):
    from falcocv.core import native_deps

    native = tmp_path / "adlink"
    native.mkdir()
    dll = native / "PCI-Dask64.dll"
    dll.write_bytes(b"fake")
    monkeypatch.setenv("FALCOCV_NATIVE_DIR", str(tmp_path))

    result = native_deps.resolve_library(
        ("PCI-Dask64.dll",), driver="adlink"
    )

    assert result.found
    assert result.path == dll.resolve()
    assert result.source == "native directory"


def test_resolve_library_accepts_configured_file_and_directory(
    monkeypatch, tmp_path
):
    from falcocv.core import native_deps

    monkeypatch.setenv("FALCOCV_NATIVE_DIR", str(tmp_path / "native"))
    direct = tmp_path / "custom" / "USB-Dask64.dll"
    direct.parent.mkdir()
    direct.write_bytes(b"fake")

    file_result = native_deps.resolve_library(
        ("USB-Dask64.dll",),
        driver="adlink",
        configured=str(direct),
    )
    directory_result = native_deps.resolve_library(
        ("USB-Dask64.dll",),
        driver="adlink",
        configured=str(direct.parent),
    )

    assert file_result.path == direct.resolve()
    assert file_result.source == "configured file"
    assert directory_result.path == direct.resolve()
    assert directory_result.source == "configured directory"


def test_resolve_library_reports_missing_candidates(monkeypatch, tmp_path):
    from falcocv.core import native_deps

    monkeypatch.setenv("FALCOCV_NATIVE_DIR", str(tmp_path))

    result = native_deps.resolve_library(
        ("not-a-real-dask-library.dll",), driver="adlink"
    )

    assert result.found is False
    assert result.path is None
    assert any(
        path.name == "not-a-real-dask-library.dll"
        for path in result.searched
    )
    assert "FalcoCV" in native_deps.native_library_hint(
        "adlink", ("PCI-Dask64.dll",)
    ) or "native" in native_deps.native_library_hint(
        "adlink", ("PCI-Dask64.dll",)
    )


def test_resolve_tcads_prefers_twincat_directory(monkeypatch, tmp_path):
    from falcocv.core import native_deps

    monkeypatch.setenv("FALCOCV_NATIVE_DIR", str(tmp_path / "native"))
    twin = tmp_path / "TwinCAT" / "3.1"
    twin.mkdir(parents=True)
    common = tmp_path / "TwinCAT" / "Common64"
    common.mkdir()
    dll = common / "TcAdsDll.dll"
    dll.write_bytes(b"fake")
    monkeypatch.setenv("TWINCAT3DIR", str(twin))

    result = native_deps.resolve_tcads_dll()

    assert result.found
    assert result.path == dll.resolve()
    assert result.source == "TWINCAT3DIR"


def test_native_deps_cli_reports_adlink_and_siemens(capsys, monkeypatch, tmp_path):
    from falcocv.cli import main

    monkeypatch.setenv("FALCOCV_NATIVE_DIR", str(tmp_path))

    assert main(["native-deps"]) == 0
    output = capsys.readouterr().out
    assert "FalcoCV native dependencies" in output
    assert "adlink:" in output
    assert "siemens:" in output
    assert "pure Python" in output


def test_adlink_native_names_follow_architecture():
    from falcocv.comms.drivers import adlink

    usb = adlink.adlink_dll_names("USB_7250")
    pci = adlink.adlink_dll_names("PCI_7250")
    all_names = adlink.adlink_dll_names()

    assert len(usb) == 1
    assert "USB-Dask" in usb[0]
    assert "PCI-Dask" in pci[0]
    assert usb[0] in all_names
    assert pci[0] in all_names
