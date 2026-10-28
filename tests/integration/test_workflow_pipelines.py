"""End-to-end headless pipelines (no QApplication required)."""

import numpy as np
import pytest

from falcocv.core.vision_image import VisionImage
from falcocv.core.workflow import Workflow
from falcocv.tools.barcode_decoder.logic import BarcodeDecoder
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.depth_range_loader.logic import DepthRangeLoader
from falcocv.tools.filters.logic import FiltersTool
from falcocv.tools.image_loader.logic import ImageLoader
from falcocv.tools.measurement.logic import MeasurementTool

from .helpers import make_blob_image, make_depth_plane, run_sync

pytestmark = pytest.mark.integration


def _blob_pipeline(image_path):
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
    return workflow, loader, filters, blob


def test_loader_filters_blob_pipeline(tmp_path):
    image_path = make_blob_image(tmp_path / "blobs.png")
    workflow, loader, filters, blob = _blob_pipeline(image_path)

    run_sync(workflow)

    assert workflow.last_error is None
    assert loader.outputs["output image"] is not None
    assert filters.outputs["output image"] is not None
    assert blob.outputs["count"] == 3
    areas = sorted(r.area for r in blob.outputs["results"])
    assert all(3400 <= area <= 3600 for area in areas)


def test_measurement_area_from_overlay():
    image = VisionImage(np.zeros((100, 100, 3), dtype=np.uint8))
    layer = image.get_overlay_layer("test", source_tool="Source")
    layer.add_roi([10, 20, 30, 40])

    measurement = MeasurementTool()
    measurement.params["overlay_measurements"] = [
        {
            "enabled": True,
            "mode": "area",
            "source1": "Source",
            "type1": "ROI",
            "ref_id1": "",
        }
    ]
    measurement.inputs["input image"] = image
    measurement.run()

    assert measurement.outputs["count"] == 1
    result = measurement.outputs["results"][0]
    assert result.type == "area"
    assert result.value == pytest.approx(1200.0)
    assert result.unit == "mm2"


def test_barcode_1d_pipeline(tmp_path):
    from barcode import Code128
    from barcode.writer import ImageWriter

    barcode_path = tmp_path / "code128"
    Code128("FALCO-123", writer=ImageWriter()).save(str(barcode_path))
    png = tmp_path / "code128.png"
    assert png.exists()

    workflow = Workflow()
    loader = ImageLoader("Loader")
    decoder = BarcodeDecoder("Barcode")
    loader.params.update({"mode": "File", "file_path": str(png)})
    decoder.params["code_type"] = "Barcode 1D"
    workflow.add_tool(loader)
    workflow.add_tool(decoder)
    workflow.connect(loader, "output image", decoder, "input image")

    run_sync(workflow)

    assert workflow.last_error is None
    assert decoder.outputs["count"] >= 1
    texts = [r.text for r in decoder.outputs["results"]]
    assert any("FALCO-123" in text for text in texts)


def test_depth_loader_generates_cloud(tmp_path):
    depth_path = make_depth_plane(tmp_path / "plane.tif")

    workflow = Workflow()
    loader = DepthRangeLoader("Depth")
    loader.params.update(
        {
            "mode": "File",
            "file_path": str(depth_path),
            "generate_3d": True,
            "target_points": 10000,
            "zero_as_invalid": True,
            "x_res": 0.1,
            "y_res": 0.1,
            "z_res": 0.01,
            "z_offset": 0.0,
        }
    )
    workflow.add_tool(loader)

    run_sync(workflow)

    assert workflow.last_error is None
    output = loader.outputs["output image"]
    assert output is not None
    assert output._3d_data is not None
    assert output._3d_data["points"].shape[0] > 0


def test_missing_loader_file_reports_error():
    workflow = Workflow()
    loader = ImageLoader("Loader")
    filters = FiltersTool("Filters")
    loader.params.update({"mode": "File", "file_path": "does-not-exist.png"})
    workflow.add_tool(loader)
    workflow.add_tool(filters)
    workflow.connect(loader, "output image", filters, "input image")

    run_sync(workflow)

    assert workflow.last_error is not None
    assert "FileNotFound" in workflow.last_error
    assert filters.outputs["output image"] is None
