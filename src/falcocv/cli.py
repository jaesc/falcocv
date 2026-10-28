"""Console dispatcher for ``falcocv`` and ``python -m falcocv``."""
from __future__ import annotations

import sys


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in {"download-models", "download_models"}:
        from .core.models import download_models_cli
        return download_models_cli()
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
