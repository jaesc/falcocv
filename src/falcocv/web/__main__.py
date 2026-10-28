import argparse
import uvicorn
from falcocv.core.config import ConfigManager


def main():
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


if __name__ == "__main__":
    main()
