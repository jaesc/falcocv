"""Build ``examples/demo_inspection.cvj`` from the synthetic demo images.

    Image Loader -> Shape Match (fixture) -> Blob Detector (holes) -> Measurement

Shape Match locates the part and becomes the fixture, so the hole-detection ROI
follows the part's rotation and position.  The measurement is the distance
between the two hole centroids (10 px/mm).

Usage:
    python examples/build_demo_job.py
"""

from __future__ import annotations

from pathlib import Path

import cv2

from falcocv.core.job_manager import JobManager
from falcocv.core.workflow import Workflow
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.image_loader.logic import ImageLoader
from falcocv.tools.measurement.logic import MeasurementTool
from falcocv.tools.shape_match.logic import ShapeMatch

ROOT = Path(__file__).resolve().parent
JOB = ROOT / "demo_inspection.cvj"
REFERENCE = ROOT / "train" / "part_reference.png"

# Reference geometry (see generate_demo_images.py)
PART_W, PART_H = 260, 170
CX, CY = 640, 480
MARGIN = 6


def build() -> Workflow:
    reference = cv2.imread(str(REFERENCE))
    if reference is None:
        raise SystemExit(f"missing {REFERENCE}; run examples/generate_demo_images.py")

    loader = ImageLoader("Image Loader")
    loader.params.update(
        {
            "mode": "Folder",
            "folder_path": "examples/images",
            "sort_by": "Name",
        }
    )

    matcher = ShapeMatch("Shape Match")
    matcher._static_train_image = reference.copy()
    matcher._static_train_fixture = {"x": 0.0, "y": 0.0, "angle": 0.0}
    matcher.params.update(
        {
            "train_roi": [CX - PART_W / 2 - MARGIN, CY - PART_H / 2 - MARGIN,
                          PART_W + 2 * MARGIN, PART_H + 2 * MARGIN, 0.0],
            "search_roi": [0, 0, 0, 0, 0],
            "angle_start": -25.0,
            "angle_end": 25.0,
            "scale_start": 0.92,
            "scale_end": 1.08,
            "min_score": 0.60,
            "max_matches": 1,
            "max_features": 800,
            "act_as_fixture": True,
            "show_search_roi": False,
            "show_found_region": True,
            "show_found_origin": True,
            "show_labels": True,
            "copy_overlays_to_input": True,
        }
    )
    matcher.train_pattern()

    blobs = BlobDetector("Blob Detector")
    blobs.params.update(
        {
            "threshold_mode": "Fixed",
            "threshold_value": 100,
            "polarity": "Dark Blobs",
            "area_min": 1150,
            "area_max": 4000,
            "roi": [-110.0, -62.0, 220.0, 124.0, 0.0],
            "show_search_roi": True,
            "show_contours": True,
            "show_centroids": True,
            "show_labels": True,
        }
    )

    measurement = MeasurementTool("Measurement")
    measurement.params.update(
        {
            "pixels_per_mm": 10.0,
            "show_measurement_lines": True,
            "show_labels": True,
            "overlay_measurements": [
                {
                    "enabled": True,
                    "mode": "distance",
                    "source1": "Blob Detector",
                    "ref_id1": "blob_0",
                    "type1": "POINT",
                    "strategy1": "center",
                    "source2": "Blob Detector",
                    "ref_id2": "blob_1",
                    "type2": "POINT",
                    "strategy2": "center",
                    "color": [0, 255, 255],
                    "thickness": 2,
                }
            ],
        }
    )

    workflow = Workflow()
    for tool in (loader, matcher, blobs, measurement):
        workflow.add_tool(tool)
    workflow.connect(loader, "output image", matcher, "input image")
    workflow.connect(matcher, "output image", blobs, "input image")
    workflow.connect(blobs, "output image", measurement, "input image")
    return workflow


def main() -> int:
    workflow = build()
    JobManager().save_job(workflow, str(JOB))
    print(f"saved {JOB}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
