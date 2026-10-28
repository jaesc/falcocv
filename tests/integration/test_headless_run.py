"""F0.5: headless job execution and JSON results."""

import json

import pytest

from falcocv.core.job_manager import JobManager
from falcocv.core.results import InspectionResult
from falcocv.core.workflow import Workflow
from falcocv.run_job import _json_safe, main as run_main, run_job
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.filters.logic import FiltersTool
from falcocv.tools.image_loader.logic import ImageLoader

from .helpers import make_blob_image

pytestmark = pytest.mark.integration


def _blob_params():
    return {
        "threshold_mode": "Fixed",
        "threshold_value": 128,
        "polarity": "Bright Blobs",
        "area_min": 50,
        "area_max": 100000,
        "show_contours": False,
        "show_centroids": False,
        "show_labels": False,
        "show_search_roi": False,
    }


def _save_pipeline(tmp_path, image_path, reverse_order=False):
    workflow = Workflow()
    loader = ImageLoader("Loader")
    filters = FiltersTool("Filters")
    blob = BlobDetector("Blob")
    loader.params.update({"mode": "File", "file_path": str(image_path)})
    blob.params.update(_blob_params())

    tools = [filters, loader, blob] if reverse_order else [loader, filters, blob]
    for tool in tools:
        workflow.add_tool(tool)
    workflow.connect(loader, "output image", filters, "input image")
    workflow.connect(filters, "output image", blob, "input image")

    path = tmp_path / "headless.cvj"
    JobManager().save_job(workflow, str(path))
    return path


def test_headless_pipeline_and_json(tmp_path):
    image_path = make_blob_image(tmp_path / "blobs.png")
    job_path = _save_pipeline(tmp_path, image_path)

    report, code = run_job(job_path)
    assert code == 0
    assert report["ok"] is True
    assert [tool["name"] for tool in report["tools"]] == ["Loader", "Filters", "Blob"]

    blob = report["tools"][2]
    assert blob["outputs"]["count"] == 3
    assert any(item["tool"] == "Blob" for item in report["results"])

    output_path = tmp_path / "results.json"
    assert run_main([str(job_path), "--output", str(output_path)]) == 0
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert data["schema_version"] == 1
    assert data["total_time_ms"] >= 0


def test_headless_preserves_manual_tool_order(tmp_path):
    image_path = make_blob_image(tmp_path / "blobs.png")
    job_path = _save_pipeline(tmp_path, image_path, reverse_order=True)

    report, code = run_job(job_path)
    assert code == 0

    # The list order is the contract: Filters executes before its source and
    # therefore sees None on the first run.  The runner must not reorder it.
    assert [tool["name"] for tool in report["tools"]] == ["Filters", "Loader", "Blob"]
    filters = report["tools"][0]
    assert filters["outputs"]["output image"] is None


def test_headless_error_exit_code(tmp_path):
    workflow = Workflow()
    loader = ImageLoader("Loader")
    loader.params.update({"mode": "File", "file_path": str(tmp_path / "missing.png")})
    workflow.add_tool(loader)
    path = tmp_path / "missing.cvj"
    JobManager().save_job(workflow, str(path))

    report, code = run_job(path)
    assert code == 1
    assert report["ok"] is False
    assert "FileNotFound" in report["error"]

    output_path = tmp_path / "error.json"
    assert run_main([str(path), "--output", str(output_path)]) == 1
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["ok"] is False


def test_cli_runs_from_a_clean_process(tmp_path):
    import subprocess
    import sys
    from pathlib import Path

    image_path = make_blob_image(tmp_path / "blobs.png")
    job_path = _save_pipeline(tmp_path, image_path)
    output_path = tmp_path / "cli.json"
    root = Path(__file__).resolve().parents[2]
    env = {
        **__import__("os").environ,
        "PYTHONPATH": str(root / "src"),
        "FALCOCV_CONFIG_DIR": str(tmp_path / "config"),
    }

    result = subprocess.run(
        [sys.executable, "-m", "falcocv", "run", str(job_path), "--output", str(output_path)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["ok"] is True
    assert data["tools"][0]["name"] == "Loader"


def test_headless_example_block():
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    report, code = run_job(root / "examples" / "demo_job1.cvj")
    assert code == 0
    assert report["ok"] is True
    assert report["tools"][0]["type"] == "Block"


def test_headless_trace_to_stderr(tmp_path, capsys):
    image_path = make_blob_image(tmp_path / "blobs.png")
    job_path = _save_pipeline(tmp_path, image_path)
    output_path = tmp_path / "trace.json"

    assert run_main([str(job_path), "--trace", "--output", str(output_path)]) == 0
    captured = capsys.readouterr()
    assert "[run] Loader:" in captured.err
    assert "[run] Blob:" in captured.err
    assert captured.out == ""


def test_inspection_result_contract():
    data = _json_safe(
        InspectionResult(
            value=12.5,
            unit="mm",
            tolerance=0.1,
            verdict="OK",
            score=0.98,
            uncertainty=0.02,
            message="measured",
        )
    )
    assert data == {
        "value": 12.5,
        "unit": "mm",
        "tolerance": 0.1,
        "verdict": "OK",
        "score": 0.98,
        "uncertainty": 0.02,
        "message": "measured",
    }
