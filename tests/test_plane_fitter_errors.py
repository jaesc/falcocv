"""PlaneFitter fails when it cannot fit a plane from the input."""

import numpy as np

from falcocv.core.pointcloud_utils import ensure_pointcloud_data
from falcocv.core.vision_image import VisionImage
from falcocv.tools.plane_fitter import logic as plane_logic
from falcocv.tools.plane_fitter.logic import PlaneFitter


def _depth_image(h=120, w=160, value=10.0):
    depth = np.full((h, w), value, dtype=np.float32)
    return VisionImage(
        depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0
    )


def _tool_with_cloud(points=None):
    img = _depth_image()
    if points is None:
        ensure_pointcloud_data(img, target_points=10_000)
    else:
        img._3d_data = {
            "points": np.asarray(points, dtype=np.float32),
            "intensity": np.zeros(len(points), dtype=np.float32),
            "colors": None,
        }
    tool = PlaneFitter()
    tool.set_input("input image", img)
    return tool


def test_missing_input_is_an_error():
    tool = PlaneFitter()
    tool.run()
    assert tool.last_error == "Input image is missing"
    assert tool.outputs["output image"] is None
    assert len(tool.outputs["results"]) == 0


def test_image_without_depth_data_is_an_error():
    tool = PlaneFitter()
    tool.set_input("input image", VisionImage(np.zeros((120, 160, 3), np.uint8)))
    tool.run()
    assert tool.last_error == "Input image has no depth data"
    assert tool.outputs["output image"] is None


def test_missing_pointcloud_is_an_error():
    tool = PlaneFitter()
    tool.set_input("input image", _depth_image())
    tool.run()
    assert (
        tool.last_error
        == "Input image has no 3D point cloud; connect a 3D source"
    )
    assert tool.outputs["output image"] is None


def test_empty_pointcloud_is_an_error():
    tool = _tool_with_cloud(np.zeros((0, 3), dtype=np.float32))
    tool.run()
    assert tool.last_error == "3D point cloud is empty"
    assert tool.outputs["output image"] is None


def test_kernel_error_is_reported():
    # Two points are not enough for a plane fit.
    tool = _tool_with_cloud([[0.0, 0.0, 10.0], [1.0, 1.0, 10.0]])
    tool.run()
    assert tool.last_error == "at least 3 points required for plane fitting"
    assert tool.outputs["output image"] is None
    assert len(tool.outputs["results"]) == 0


def test_fit_without_inliers_is_an_error(monkeypatch):
    tool = _tool_with_cloud()

    def fake_plane_fit(_input_dict):
        return {
            "normal": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            "center": np.array([0.0, 0.0, 10.0], dtype=np.float32),
            "distance": 10.0,
            "inlier_count": 0,
            "inlier_ratio": 0.0,
            "rms_error": 0.0,
            "total_points": 100,
        }

    monkeypatch.setattr(plane_logic, "_plane_fit", fake_plane_fit)
    tool.run()
    assert tool.last_error == "Plane fit failed: no inlier points"
    assert tool.outputs["output image"] is None
    assert len(tool.outputs["results"]) == 0


def test_fit_with_non_finite_parameters_is_an_error(monkeypatch):
    tool = _tool_with_cloud()

    def fake_plane_fit(_input_dict):
        return {
            "normal": np.array([np.nan, 0.0, 1.0], dtype=np.float32),
            "center": np.array([0.0, 0.0, 10.0], dtype=np.float32),
            "distance": 10.0,
            "inlier_count": 100,
            "inlier_ratio": 1.0,
            "rms_error": 0.0,
            "total_points": 100,
        }

    monkeypatch.setattr(plane_logic, "_plane_fit", fake_plane_fit)
    tool.run()
    assert tool.last_error == "Plane fit failed: invalid plane parameters"
    assert tool.outputs["output image"] is None


def test_valid_fit_has_no_error():
    tool = _tool_with_cloud()
    tool.params["method"] = "lsq"
    tool.params["rois"] = []
    tool.run()
    assert tool.last_error is None
    assert len(tool.outputs["results"]) == 1
    assert tool.outputs["results"][0].inlier_count > 0


def test_workflow_reports_the_plane_fitter_error():
    from falcocv.core.workflow import Workflow

    workflow = Workflow()
    workflow.add_tool(PlaneFitter())
    workflow.stop_on_error = True
    workflow.run()
    assert workflow.last_error == "Plane Fitter: Input image is missing"

