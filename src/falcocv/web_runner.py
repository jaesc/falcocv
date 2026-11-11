"""Entry point for the FalcoCV web/HMI server (development and frozen builds).

Usage:
    python -m falcocv.web_runner --port 8000
    python -m falcocv.web_runner --worker      # HMI worker subprocess
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")
warnings.filterwarnings("ignore", message="remove second argument of ws_handler")


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)

    if "--worker" in args:
        from falcocv.web.hmi_worker import main as worker_main

        worker_main()
        return 0

    if getattr(sys, "frozen", False):
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")
        if sys.stdout is None:
            sys.stdout = open(os.devnull, "w")

        from falcocv.core.environment import (
            bundled_python_available,
            ensure_venv,
            get_venv_site_packages,
        )

        if bundled_python_available():
            ensure_venv()
            site_packages = get_venv_site_packages()
            if site_packages and site_packages not in sys.path:
                sys.path.insert(0, site_packages)

    from falcocv.core.config import ConfigManager

    parser = argparse.ArgumentParser(description="FalcoCV Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=ConfigManager().get_web_port(), help="Port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    options = parser.parse_args(args)

    import uvicorn

    if options.reload and not getattr(sys, "frozen", False):
        uvicorn.run(
            "falcocv.web.server:app",
            host=options.host,
            port=options.port,
            reload=True,
        )
    else:
        # Import the app object so a frozen build does not depend on
        # uvicorn's string import machinery.
        from falcocv.web.server import app

        uvicorn.run(app, host=options.host, port=options.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
