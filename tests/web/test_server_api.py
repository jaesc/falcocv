"""Web server / HMI API integration tests."""

import socket
import subprocess
import sys
import time
import urllib.request

import pytest

pytestmark = pytest.mark.web

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def test_hmi_status_and_logs_without_project():
    from fastapi.testclient import TestClient
    from falcocv.web.server import app

    with TestClient(app) as client:
        status = client.get("/api/hmi?project_id=missing")
        assert status.status_code == 200
        assert status.json()["slots"] == []
        assert status.json()["layout"]["widgets"] == []

        logs = client.get("/api/hmi/logs?project_id=missing")
        assert logs.status_code == 200
        assert logs.json()["logs"] == []


def test_websocket_ping():
    from fastapi.testclient import TestClient
    from falcocv.web.server import app

    with TestClient(app) as client:
        with client.websocket_connect("/api/ws?project_id=test") as ws:
            hello = ws.receive_json()
            assert hello["type"] == "connected"
            assert hello["project_id"] == "test"
            ws.send_json({"cmd": "ping"})
            assert ws.receive_json()["type"] == "pong"


def test_server_subprocess_starts_and_serves_projects(tmp_path):
    port = _free_port()
    env = {
        **__import__("os").environ,
        "FALCOCV_CONFIG_DIR": str(tmp_path / "config"),
    }
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "falcocv.web",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ],
        cwd=ROOT,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    try:
        url = f"http://127.0.0.1:{port}/api/server/projects"
        deadline = time.time() + 20
        last_error = None
        while time.time() < deadline:
            if proc.poll() is not None:
                output = proc.stdout.read() if proc.stdout else ""
                pytest.fail(f"server exited early ({proc.returncode}):\n{output}")
            try:
                with urllib.request.urlopen(url, timeout=1) as response:
                    assert response.status == 200
                    assert response.read().strip().startswith(b"[")
                    return
            except Exception as exc:  # noqa: BLE001 - polling
                last_error = exc
                time.sleep(0.25)
        pytest.fail(f"server did not become ready: {last_error}")
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5)
