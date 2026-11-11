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
            "python-snap7 installed"
            if snap7_installed
            else "python-snap7 is not installed"
        )
        + " (pure Python; no native DLL required)"
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


def _packages_cli(argv) -> int:
    import argparse

    from .core import user_libs

    parser = argparse.ArgumentParser(prog="falcocv packages")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="List installed user packages")
    sub.add_parser("dir", help="Print the user package directory")

    install = sub.add_parser("install", help="Install wheels for user scripts")
    install.add_argument("names", nargs="+")
    install.add_argument("--find-links", default=None, help="Local wheelhouse")

    remove = sub.add_parser("remove", help="Remove an installed package")
    remove.add_argument("names", nargs="+")

    options = parser.parse_args(argv)

    if options.command == "dir":
        print(user_libs.user_packages_dir())
        return 0

    if options.command == "list":
        packages = user_libs.list_packages()
        if not packages:
            print(f"No user packages in {user_libs.user_packages_dir()}")
            return 0
        for package in packages:
            print(f"{package['name']} {package['version']}")
        return 0

    if options.command == "install":
        conflicts = user_libs.bundled_modules(options.names)
        if conflicts:
            print(
                "warning: already bundled; the FalcoCV version keeps priority: "
                + ", ".join(conflicts)
            )
        ok, output = user_libs.install_packages(
            options.names, wheelhouse=options.find_links
        )
        print(output)
        if ok:
            print(f"Installed into {user_libs.user_packages_dir()}")
        return 0 if ok else 1

    ok = True
    for name in options.names:
        success, message = user_libs.remove_package(name)
        print(message)
        ok = ok and success
    return 0 if ok else 1


def _python_cli(argv) -> int:
    import argparse
    import os

    parser = argparse.ArgumentParser(prog="falcocv python")
    parser.add_argument("-c", "--code", default=None, help="Program passed as a string")
    parser.add_argument("file", nargs="?", default=None, help="Script file")
    options = parser.parse_args(argv)
    if not options.code and not options.file:
        parser.error("provide -c CODE or a script file")

    namespace = {"__name__": "__main__"}
    if options.file:
        path = os.path.abspath(options.file)
        directory = os.path.dirname(path)
        if directory not in sys.path:
            sys.path.insert(0, directory)
        namespace["__file__"] = path
        with open(path, "r", encoding="utf-8") as handle:
            code = compile(handle.read(), path, "exec")
    else:
        namespace["__file__"] = "<falcocv-python>"
        code = compile(options.code, "<falcocv-python>", "exec")

    exec(code, namespace)
    return 0


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    from .core.user_libs import install_user_libs

    install_user_libs()

    if args and args[0] in {"download-models", "download_models"}:
        from .core.models import download_models_cli
        return download_models_cli()
    if args and args[0] in {"native-deps", "native_deps"}:
        return _native_deps_report()
    if args and args[0] in {"packages", "libs"}:
        return _packages_cli(args[1:])
    if args and args[0] in {"python", "exec"}:
        return _python_cli(args[1:])
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
