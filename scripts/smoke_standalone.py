"""Smoke-test a built FalcoCV standalone bundle.

Checks, without opening the GUI:

1. the expected executables exist,
2. ``FalcoCVCLI self-test`` passes (tools, help, pipeline, job round-trip),
3. ``FalcoCVCLI native-deps`` runs,
4. ``FalcoCVServer`` answers ``/api/health`` with ``standalone: true``,
5. the native kernel and (optionally) the models are in the bundle.

Usage:
    python scripts/smoke_standalone.py
    python scripts/smoke_standalone.py --bundle dist/standalone/FalcoCV --no-models
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _timeout_default() -> int:
    return 120


def _exe(bundle: Path, name: str) -> Path:
    suffix = ".exe" if sys.platform == "win32" else ""
    return bundle / f"{name}{suffix}"


def _run(command: list[str], timeout: int) -> subprocess.CompletedProcess:
    print(f"[smoke] $ {' '.join(command)}")
    return subprocess.run(
        command, capture_output=True, text=True, timeout=timeout
    )


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_health(port: int, timeout: float) -> dict:
    url = f"http://127.0.0.1:{port}/api/health"
    deadline = time.time() + timeout
    last_error: Exception | None = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:  # not up yet
            last_error = exc
            time.sleep(0.5)
    raise RuntimeError(f"server did not answer {url}: {last_error}")


def check_server(bundle: Path, log_path: Path, timeout: float) -> dict:
    port = _free_port()
    command = [
        str(_exe(bundle, "FalcoCVServer")),
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
    print(f"[smoke] $ {' '.join(command)}")
    with open(log_path, "wb") as log:
        process = subprocess.Popen(
            command,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=creationflags,
        )
        try:
            return _wait_for_health(port, timeout)
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Smoke-test the standalone bundle")
    parser.add_argument(
        "--bundle", default=str(ROOT / "dist" / "standalone" / "FalcoCV")
    )
    parser.add_argument("--no-models", action="store_true")
    parser.add_argument(
        "--timeout", type=int, default=_timeout_default(), help="Server start timeout (s)"
    )
    args = parser.parse_args(argv)

    bundle = Path(args.bundle).resolve()
    failures: list[str] = []

    if not bundle.is_dir():
        raise SystemExit(f"bundle not found: {bundle}")

    for name in ("FalcoCV", "FalcoCVHMI", "FalcoCVServer", "FalcoCVCLI"):
        if not _exe(bundle, name).is_file():
            failures.append(f"missing executable: {_exe(bundle, name).name}")

    kernel = list((bundle / "_internal" / "falcocv").glob("_falco_kernel*"))
    if not kernel:
        failures.append("missing native kernel in _internal/falcocv")

    if not args.no_models and not (bundle / "models").is_dir():
        failures.append("missing models directory")

    cli = _exe(bundle, "FalcoCVCLI")
    if cli.is_file():
        result = _run([str(cli), "self-test"], timeout=args.timeout)
        if result.returncode != 0 or "SELF-TEST: PASS" not in result.stdout:
            failures.append(f"self-test failed (rc={result.returncode})")
            print(result.stdout[-4000:])
            print(result.stderr[-4000:])

        result = _run([str(cli), "native-deps"], timeout=60)
        if result.returncode != 0:
            failures.append(f"native-deps failed (rc={result.returncode})")
        else:
            print(result.stdout.strip())

    server = _exe(bundle, "FalcoCVServer")
    if server.is_file():
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "server.log"
            try:
                health = check_server(bundle, log_path, args.timeout)
            except Exception as exc:
                failures.append(str(exc))
                health = {}
                print(log_path.read_text(errors="replace")[-4000:])
        if health:
            print(f"[smoke] health: {health}")
            if not health.get("ok"):
                failures.append("health did not report ok")
            if not health.get("standalone"):
                failures.append("health did not report standalone=true")

    if failures:
        print("\n[smoke] FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("\n[smoke] PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
