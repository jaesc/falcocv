"""Job (.cvj) persistence round-trips."""

import pytest

from falcocv.core.job_manager import JobManager
from falcocv.core.workflow import Workflow
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.filters.logic import FiltersTool
from falcocv.tools.image_loader.logic import ImageLoader

from .helpers import make_blob_image, run_sync, tool_by_type

pytestmark = pytest.mark.integration

ROOT = __import__("pathlib").Path(__file__).resolve().parents[2]


def _build_pipeline(image_path):
    workflow = Workflow()
    loader = ImageLoader("Loader")
    filters = FiltersTool("Filters")
    blob = BlobDetector("Blob")
    loader.params.update({"mode": "File", "file_path": str(image_path)})
    blob.params.update(
        {
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
    )
    for tool in (loader, filters, blob):
        workflow.add_tool(tool)
    workflow.connect(loader, "output image", filters, "input image")
    workflow.connect(filters, "output image", blob, "input image")
    return workflow


def test_pipeline_survives_save_and_load(tmp_path):
    image_path = make_blob_image(tmp_path / "blobs.png")
    workflow = _build_pipeline(image_path)
    path = tmp_path / "pipeline.cvj"

    JobManager().save_job(workflow, str(path))
    assert path.exists()

    loaded = Workflow()
    JobManager().load_job(loaded, str(path))
    assert [type(t).__name__ for t in loaded.tools] == [
        "ImageLoader",
        "FiltersTool",
        "BlobDetector",
    ]
    assert len(loaded.connections) == 2

    run_sync(loaded)
    assert loaded.last_error is None
    assert tool_by_type(loaded, "BlobDetector").outputs["count"] == 3


@pytest.mark.parametrize("name", ["demo_job1.cvj", "demo_job2.cvj"])
def test_example_jobs_load(name):
    path = ROOT / "examples" / name
    workflow = Workflow()
    JobManager().load_job(workflow, str(path))
    assert workflow.tools
    assert all(type(t).__name__ == "Block" for t in workflow.tools)
