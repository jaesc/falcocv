import argparse
import json
import os
import sys
import uuid as _uuid
from urllib.parse import urljoin
from urllib.request import Request, urlopen

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _get_web_port():
    from falcocv.core.config import ConfigManager
    return ConfigManager().get_web_port()


def _read_hmi(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _read_source_files(source_dir: str) -> dict[str, str]:
    files = {}
    for root, dirs, fnames in os.walk(source_dir):
        for fn in fnames:
            if fn.endswith(".py"):
                full = os.path.join(root, fn)
                rel = os.path.relpath(full, source_dir).replace("\\", "/")
                with open(full, "r", encoding="utf-8") as f:
                    files[rel] = f.read()
    return files


def _resolve_project_id(hmi_data: dict, cli_value: str | None) -> str:
    if cli_value:
        return cli_value
    return hmi_data.get("project_id") or str(_uuid.uuid4())[:8]


def main():
    parser = argparse.ArgumentParser(
        description="Deploy or stop a .hmi project on a running FalcoCV server"
    )
    parser.add_argument("project", nargs="?", help="Path to .hmi file")
    parser.add_argument("--project-id", help="Project ID (auto if not given)")
    parser.add_argument(
        "--url", help="Full server URL (e.g. http://10.0.0.5:8000)"
    )
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument(
        "--stop", action="store_true", help="Stop the running HMI project"
    )
    args = parser.parse_args()

    port = args.port if args.port is not None else _get_web_port()
    base = args.url or f"http://{args.host}:{port}"

    if args.stop:
        pid = args.project_id or ""
        body = json.dumps({"project_id": pid}).encode("utf-8")
        url = urljoin(base.rstrip("/") + "/", "api/hmi/stop")
        req = Request(url, data=body, method="POST")
        req.add_header("Content-Type", "application/json")
        try:
            with urlopen(req) as resp:
                result = json.loads(resp.read())
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        if result.get("ok"):
            print("HMI stopped")
        else:
            print(f"Server error: {result.get('error', 'unknown')}")
            sys.exit(1)
        return

    if not args.project:
        parser.error("project path is required (or use --stop)")

    if not os.path.isfile(args.project):
        print(f"Error: file not found — {args.project}")
        sys.exit(1)

    hmi_data = _read_hmi(args.project)

    url = urljoin(base.rstrip("/") + "/", "api/hmi/load")
    payload: dict[str, object] = {
        "script": hmi_data.get("script", ""),
        "layout": hmi_data.get("layout", {}),
        "slots": hmi_data.get("slots", []),
        "project_id": _resolve_project_id(hmi_data, args.project_id),
    }
    sd = hmi_data.get("source_dir")
    if sd:
        payload["files"] = _read_source_files(sd)
        payload["entry_point"] = hmi_data.get("entry_point", "main.py")
    body = json.dumps(payload).encode("utf-8")

    req = Request(url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")

    try:
        with urlopen(req) as resp:
            result = json.loads(resp.read())
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

    if result.get("ok"):
        slots = result.get("slots", "?")
        print(f"Deployed: pid={result.get('project_id', '?')}  {slots} slot(s) loaded")
    else:
        print(f"Server error: {result.get('error', 'unknown')}")
        sys.exit(1)


if __name__ == "__main__":
    main()
