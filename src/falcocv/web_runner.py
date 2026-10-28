"""Entry point for standalone web server (development and frozen builds).

Usage:
    python web_runner.py --port 8000
"""
import os
import sys
import warnings
warnings.filterwarnings("ignore", message="websockets.legacy is deprecated")
warnings.filterwarnings("ignore", message="websockets.server.WebSocketServerProtocol is deprecated")
warnings.filterwarnings("ignore", message="remove second argument of ws_handler")

if "--worker" in sys.argv:
    from falcocv.web.hmi_worker import main
    main()
    sys.exit(0)

import uvicorn
from falcocv.core.config import ConfigManager

if __name__ == "__main__":
    if getattr(sys, 'frozen', False):
        if sys.stderr is None:
            sys.stderr = open(os.devnull, 'w')
        if sys.stdout is None:
            sys.stdout = open(os.devnull, 'w')
        from falcocv.core.environment import ensure_venv, get_venv_site_packages
        ensure_venv()
        sp = get_venv_site_packages()
        if sp and sp not in sys.path:
            sys.path.insert(0, sp)

    import argparse
    parser = argparse.ArgumentParser(description="FalcoCV Web Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind address")
    parser.add_argument("--port", type=int, default=ConfigManager().get_web_port(), help="Port")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on changes")
    args = parser.parse_args()

    uvicorn.run(
        "falcocv.web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
