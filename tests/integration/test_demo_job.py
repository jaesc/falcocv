"""The shipped demo job must run end to end on the generated dataset."""

from pathlib import Path

import pytest

from falcocv.core.job_manager import JobManager
from falcocv.core.workflow import Workflow
from falcocv.core.workflow_controller import WorkflowController
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.image_loader.logic import ImageLoader
from falcocv.tools.measurement.logic import MeasurementTool

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "examples" / "demo_inspection.cvj"


def _run(image: str):
    workflow = Workflow()
    JobManager().load_job(workflow, str(JOB))
    loader = next(t for t in workflow.tools if isinstance(t, ImageLoader))
    loader.params.update(
        {"mode": "File", "file_path": str(ROOT / "examples" / "images" / image)}
    )
    controller = WorkflowController(workflow)
    controller.set_scheduler(lambda callback: callback())
    controller.run(advance=False)
    blobs = next(t for t in workflow.tools if isinstance(t, BlobDetector))
    measurement = next(t for t in workflow.tools if isinstance(t, MeasurementTool))
    return workflow, blobs.outputs["count"], measurement


def test_demo_job_ok_image():
    workflow, holes, measurement = _run("02_ok.jpg")
    assert workflow.last_error is None
    assert holes == 2
    assert measurement.outputs["count"] == 1
    assert 12.5 <= measurement.outputs["results"][0].value <= 15.5


def test_demo_job_missing_hole_is_ng():
    workflow, holes, measurement = _run("11_ng_missing_hole.jpg")
    assert workflow.last_error is None
    assert holes == 1
    assert measurement.outputs["count"] == 0
