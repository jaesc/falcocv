"""Headless job runner: ``falcocv run job.cvj [--output results.json]``.

The runner executes ``Workflow.run()`` in the exact order stored in the job.
It does not reorder tools, validate the pipeline, or open Qt.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

from falcocv.core.job_manager import JobManager
from falcocv.core.tool_base import ToolResult, ToolResults
from falcocv.core.workflow import Workflow
from falcocv.core.workflow_controller import WorkflowController

SCHEMA_VERSION = 1
_MAX_DEPTH = 8
_MAX_ITEMS = 10_000
_MAX_STRING = 4_000


def _json_safe(value: Any, depth: int = 0) -> Any:
    """Convert tool outputs to JSON-compatible values without huge payloads."""
    if depth > _MAX_DEPTH:
        return "<max-depth>"

    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        return value[:_MAX_STRING]
    if isinstance(value, Path):
        return str(value)

    try:
        import numpy as np

        if isinstance(value, np.generic):
            return _json_safe(value.item(), depth + 1)
        if isinstance(value, np.ndarray):
            summary: dict[str, Any] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
            }
            if value.size and value.dtype.kind in "uifb":
                try:
                    summary["min"] = _json_safe(float(value.min()), depth + 1)
                    summary["max"] = _json_safe(float(value.max()), depth + 1)
                except (TypeError, ValueError):
                    pass
            return summary
    except ImportError:
        pass

    if hasattr(value, "__dataclass_fields__"):
        from dataclasses import asdict

        return {key: _json_safe(item, depth + 1) for key, item in asdict(value).items()}

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item, depth + 1)
            for key, item in list(value.items())[:_MAX_ITEMS]
        }

    if isinstance(value, (list, tuple, set)):
        items = list(value)
        return [_json_safe(item, depth + 1) for item in items[:_MAX_ITEMS]]

    if isinstance(value, ToolResult):
        return {
            str(key): _json_safe(item, depth + 1)
            for key, item in value.items()
        }

    if hasattr(value, "items") and not isinstance(value, (str, bytes, bytearray)):
        try:
            return {
                str(key): _json_safe(item, depth + 1)
                for key, item in list(value.items())[:_MAX_ITEMS]
            }
        except (TypeError, ValueError):
            pass

    data = getattr(value, "data", None)
    if data is not None and hasattr(data, "shape"):
        return {
            "type": type(value).__name__,
            "shape": list(data.shape),
            "dtype": str(getattr(data, "dtype", "")),
        }

    return repr(value)[:_MAX_STRING]


def _tool_report(tool) -> dict:
    return {
        "name": tool.name,
        "type": type(tool).__name__,
        "enabled": bool(getattr(tool, "enabled", True)),
        "time_ms": round(float(getattr(tool, "last_execution_time_ms", 0.0)), 3),
        "error": tool.last_error,
        "outputs": {
            name: _json_safe(value)
            for name, value in tool.outputs.items()
        },
    }


def _collect_results(workflow: Workflow) -> list[dict]:
    results: list[dict] = []
    for tool in workflow.tools:
        for output_name, value in tool.outputs.items():
            if isinstance(value, ToolResults):
                for index, result in enumerate(value):
                    item = _json_safe(result)
                    if isinstance(item, dict):
                        item = {"tool": tool.name, "output": output_name, "index": index, **item}
                    else:
                        item = {"tool": tool.name, "output": output_name, "index": index, "value": item}
                    results.append(item)
            elif isinstance(value, ToolResult):
                item = _json_safe(value)
                results.append({"tool": tool.name, "output": output_name, **item})
            if len(results) >= _MAX_ITEMS:
                return results
    return results


def run_job(job_path: str | Path, trace: bool = False) -> tuple[dict, int]:
    """Load and execute a job. Returns ``(report, exit_code)``."""
    report: dict = {
        "schema_version": SCHEMA_VERSION,
        "job": str(job_path),
        "ok": False,
        "total_time_ms": 0.0,
        "error": None,
        "tools": [],
        "results": [],
    }

    try:
        workflow = Workflow()
        JobManager().load_job(workflow, str(job_path))
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
        return report, 1

    controller = WorkflowController(workflow)
    controller.set_scheduler(lambda callback: callback())
    started = time.perf_counter()
    controller.run(advance=False)
    elapsed_ms = (time.perf_counter() - started) * 1000.0

    report["ok"] = workflow.last_error is None
    report["total_time_ms"] = round(elapsed_ms, 3)
    report["error"] = workflow.last_error
    report["tools"] = [_tool_report(tool) for tool in workflow.tools]
    report["results"] = _collect_results(workflow)

    if trace:
        for tool in workflow.tools:
            status = "ERROR" if tool.last_error else "ok"
            print(
                f"[run] {tool.name}: {tool.last_execution_time_ms:.1f} ms ({status})",
                file=sys.stderr,
                flush=True,
            )
            if tool.last_error:
                print(f"[run]   {tool.last_error}", file=sys.stderr, flush=True)

    return report, 0 if report["ok"] else 1


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="falcocv run",
        description="Run a FalcoCV job headlessly and emit JSON results.",
    )
    parser.add_argument("job", help="Path to a .cvj job")
    parser.add_argument("-o", "--output", help="Write JSON results to this file")
    parser.add_argument(
        "--trace",
        action="store_true",
        help="Print per-tool timing/errors to stderr",
    )
    args = parser.parse_args(argv)

    report, code = run_job(args.job, trace=args.trace)
    text = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)

    if args.output:
        try:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            print(f"Could not write {args.output}: {exc}", file=sys.stderr)
            return 1
    else:
        print(text)

    return code


if __name__ == "__main__":
    raise SystemExit(main())
