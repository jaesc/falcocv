"""Console dispatcher for ``falcocv`` and ``python -m falcocv``."""
from __future__ import annotations

import importlib.util
import sys


def _native_deps_report() -> int:
    import os

    from .comms.drivers.adlink import adlink_dll_names
    from .core.native_deps import (
        native_driver_dir,
        native_root,
        resolve_library,
        resolve_tcads_dll,
    )

    adlink_names = adlink_dll_names()
    print("FalcoCV native dependencies")
    print(f"Native root: {native_root()}")
    for name in adlink_names:
        adlink = resolve_library((name,), driver="adlink")
        if adlink.found:
            print(f"adlink:   {name} found at {adlink.path}")
        else:
            print(
                f"adlink:   {name} missing "
                f"(expected in {native_driver_dir('adlink')})"
            )

    pyads_installed = importlib.util.find_spec("pyads") is not None
    tcads = resolve_tcads_dll()
    if not pyads_installed:
        print("beckhoff: pyads is not installed")
    elif tcads.found:
        print(f"beckhoff: pyads installed; TcAdsDll.dll found at {tcads.path}")
        if os.name == "nt" and "common32" in str(tcads.path).lower():
            print(
                "beckhoff: warning: 32-bit TcAdsDll.dll detected; "
                "install TwinCAT x64 for 64-bit Python"
            )
        print(
            "beckhoff: ADS route must be configured in TwinCAT; "
            "use Test connection in the HMI Designer"
        )
    elif os.environ.get("TWINCAT3DIR"):
        print(
            "beckhoff: pyads installed; TcAdsDll.dll should be provided by "
            "TWINCAT3DIR"
        )
    else:
        print(
            "beckhoff: pyads installed; TcAdsDll.dll not found "
            f"(expected in {native_driver_dir('beckhoff')} or TwinCAT)"
        )

    snap7_installed = importlib.util.find_spec("snap7") is not None
    print(
        "siemens:  "
        + (
            "python-snap7 installed (pure Python; no native DLL required)"
            if snap7_installed
            else "python-snap7 is not installed"
        )
    )
    pymodbus_installed = importlib.util.find_spec("pymodbus") is not None
    asyncua_installed = importlib.util.find_spec("asyncua") is not None
    print(
        "protocols: pymodbus "
        + ("installed" if pymodbus_installed else "missing")
        + "; asyncua "
        + ("installed" if asyncua_installed else "missing")
    )
    return 0


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"download-models", "download_models"}:
        from .core.models import download_models_cli
        return download_models_cli()
    if args and args[0] in {"native-deps", "native_deps"}:
        return _native_deps_report()
    if args and args[0] in {"self-test", "self_test"}:
        from .self_test import main as self_test_main
        return self_test_main(args[1:])
    if args and args[0] in {"run", "run-job", "run_job"}:
        from .run_job import main as run_job_main
        return run_job_main(args[1:])
    from .main import main as gui_main
    return gui_main()


if __name__ == "__main__":
    raise SystemExit(main())
