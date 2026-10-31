"""Blob Detector contour modes: outer-only vs nested regions."""

import cv2
import numpy as np

from falcocv.core.vision_image import VisionImage
from falcocv.tools.blob_detector.logic import BlobDetector


def _detector(**params):
    detector = BlobDetector("Blob Detector")
    detector.params.update(
        {
            "threshold_mode": "Fixed",
            "threshold_value": 110,
            "polarity": "Dark Blobs",
            "area_min": 900,
            "area_max": 3000,
            "roi": [0, 0, 0, 0, 0],
            "sort_mode": "Area Descending",
            "show_contours": False,
            "show_centroids": False,
            "show_labels": False,
            "show_search_roi": False,
            "copy_overlays_to_input": False,
        }
    )
    detector.params.update(params)
    return detector


def _run(detector, image):
    detector.inputs["input image"] = VisionImage(image)
    detector.run()
    return [(round(float(r.x), 1), round(float(r.y), 1)) for r in detector.outputs["results"]]


def test_outer_mode_keeps_the_outermost_contours_only():
    image = np.full((240, 360, 3), 20, np.uint8)
    cv2.rectangle(image, (60, 40), (300, 200), (220, 220, 220), -1)
    cv2.circle(image, (140, 120), 22, (20, 20, 20), -1)
    cv2.circle(image, (220, 120), 22, (20, 20, 20), -1)

    detector = _detector(blob_mode="Outer")
    assert _run(detector, image) == []


def test_nested_mode_includes_dark_regions_inside_a_bright_hole():
    image = np.full((240, 360, 3), 20, np.uint8)
    cv2.rectangle(image, (60, 40), (300, 200), (220, 220, 220), -1)
    cv2.circle(image, (140, 120), 22, (20, 20, 20), -1)
    cv2.circle(image, (220, 120), 22, (20, 20, 20), -1)

    detector = _detector(blob_mode="Nested")
    found = _run(detector, image)

    assert len(found) == 2
    assert found[0] == (140.0, 120.0) or found[0] == (220.0, 120.0)
    assert {found[0], found[1]} == {(140.0, 120.0), (220.0, 120.0)}


def test_nested_mode_does_not_report_the_hole_boundary_itself():
    image = np.full((240, 360, 3), 220, np.uint8)
    cv2.circle(image, (180, 120), 50, (20, 20, 20), -1)
    cv2.circle(image, (180, 120), 25, (220, 220, 220), -1)

    detector = _detector(blob_mode="Nested", area_min=1000, area_max=8000)
    found = _run(detector, image)

    assert len(found) == 1
    assert found[0] == (180.0, 120.0)


def test_blob_mode_survives_serialization():
    detector = _detector(blob_mode="Nested")
    data = detector.serialize()

    restored = BlobDetector("restored")
    restored.deserialize(data)

    assert restored.params["blob_mode"] == "Nested"


def test_rotated_roi_crossing_an_image_edge_does_not_crash():
    image = np.zeros((960, 1280, 3), np.uint8)
    vision = VisionImage(image, x=500.0, y=500.0, angle=18.5)
    detector = _detector(
        roi=[-300, -300, 1148, 865, 0],
        blob_mode="Outer",
        polarity="Dark Blobs",
        area_min=10,
        area_max=1000000,
    )
    detector.inputs["input image"] = vision

    detector.run()

    assert detector.outputs["binary mask"].data.shape == image.shape[:2]


def test_unrotated_roi_clipped_by_an_image_edge_does_not_crash():
    image = np.zeros((960, 1280, 3), np.uint8)
    vision = VisionImage(image, x=500.0, y=500.0, angle=0.0)
    detector = _detector(
        roi=[-200, -200, 1148, 865, 0],
        blob_mode="Outer",
        polarity="Dark Blobs",
        area_min=10,
        area_max=1000000,
    )
    detector.inputs["input image"] = vision

    detector.run()

    assert detector.outputs["binary mask"].data.shape == image.shape[:2]


def test_binary_mask_is_aligned_for_a_rotated_roi():
    image = np.full((240, 360, 3), 220, np.uint8)
    cv2.circle(image, (180, 120), 25, (20, 20, 20), -1)
    vision = VisionImage(image, x=180.0, y=120.0, angle=30.0)
    detector = _detector(
        roi=[-100, -100, 200, 200, 0],
        blob_mode="Outer",
        polarity="Dark Blobs",
        area_min=500,
        area_max=5000,
    )
    detector.inputs["input image"] = vision

    detector.run()

    assert detector.outputs["count"] == 1
    assert detector.outputs["binary mask"].data[120, 180] == 255
    assert detector.outputs["binary mask"].data[10, 10] == 0


def test_old_jobs_default_to_outer_mode():
    detector = _detector()
    data = detector.serialize()
    data["params"].pop("blob_mode", None)

    restored = BlobDetector("old")
    restored.deserialize(data)

    assert restored.params.get("blob_mode", "Outer") == "Outer"
