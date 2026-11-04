"""Height Blobs 3D: segmentation, offsets, ordering and watershed."""

import cv2
import numpy as np
import pytest

from falcocv.core.plane_fit import PlaneModel
from falcocv.core.vision_image import VisionImage
from falcocv.tools.height_blobs.logic import HeightBlobs3D

WIDTH = 160
HEIGHT = 120
BASE_Z = 20.0


def _plane(z=BASE_Z):
    return PlaneModel(
        normal=(0.0, 0.0, 1.0),
        distance=z,
        center=(0.0, 0.0, z),
    )


def _disc(depth, cx, cy, radius, z):
    mask = np.zeros(depth.shape, dtype=np.uint8)
    cv2.circle(mask, (int(cx), int(cy)), int(radius), 1, -1)
    depth[mask > 0] = z


def _tool(depth, plane=None, **params):
    image = VisionImage(
        depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0
    )
    tool = HeightBlobs3D()
    tool.set_input("input image", image)
    tool.set_input("reference plane", plane or _plane())
    tool.params["rois"] = [
        {"roi": [0.0, 0.0, float(WIDTH), float(HEIGHT), 0.0],
         "z_min": 0.0, "z_max": 1.0}
    ]
    for key, value in params.items():
        tool.params[key] = value
    tool.run()
    return tool


def test_eight_pins_are_detected():
    depth = np.full((HEIGHT, WIDTH), BASE_Z, dtype=np.float32)
    for index in range(8):
        _disc(depth, 30 + index * 12, 60, 2, BASE_Z + 2.0)

    tool = _tool(depth)
    assert tool.last_error is None
    features = tool.outputs["results"]
    assert len(features) == 8
    assert tool.outputs["count"] == 8
    for feature in features:
        assert feature.h == pytest.approx(2.0, abs=0.15)
        assert feature.area_px > 0
        assert feature.touches_border is False


def test_features_are_sorted_along_u():
    depth = np.full((HEIGHT, WIDTH), BASE_Z, dtype=np.float32)
    for index in (5, 1, 6, 2):
        _disc(depth, 20 + index * 15, 60, 3, BASE_Z + 1.5)

    tool = _tool(depth, sort_mode="u")
    u_values = [feature.u for feature in tool.outputs["results"]]
    assert u_values == sorted(u_values)
    assert [feature.index for feature in tool.outputs["results"]] == list(
        range(len(u_values))
    )


def test_negative_plane_offset_measures_pins_below_the_plane():
    # Only the pins return depth (the body does not), so the measurement is
    # dominated by the pins and not by the surrounding surface.
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    for index in range(4):
        _disc(depth, 40 + index * 15, 60, 3, BASE_Z - 2.0)

    tool = _tool(depth)
    assert tool.outputs["results"] is not None
    assert len(tool.outputs["results"]) == 0
    assert tool.outputs["count"] == 0
    assert tool.outputs["warnings"]
    assert "plane_offset" in tool.outputs["warnings"][0]

    tool = _tool(depth, plane_offset=-2.6)
    features = tool.outputs["results"]
    assert len(features) == 4
    for feature in features:
        assert feature.h == pytest.approx(0.6, abs=0.05)


def test_plane_offset_moves_the_measured_height():
    depth = np.full((HEIGHT, WIDTH), BASE_Z, dtype=np.float32)
    _disc(depth, 80, 60, 4, BASE_Z + 3.0)

    plain = _tool(depth)
    shifted = _tool(depth, plane_offset=0.5)
    assert plain.outputs["results"][0].h == pytest.approx(3.0, abs=0.05)
    assert shifted.outputs["results"][0].h == pytest.approx(2.5, abs=0.05)


def test_tiny_and_large_blobs_are_both_detected():
    depth = np.full((HEIGHT, WIDTH), BASE_Z, dtype=np.float32)
    _disc(depth, 40, 60, 1, BASE_Z + 2.0)     # ~5 px
    _disc(depth, 100, 60, 10, BASE_Z + 2.0)   # ~300 px

    tool = _tool(depth, min_points=1, close_px=1, open_px=0)
    areas = sorted(feature.area_px for feature in tool.outputs["results"])
    assert len(areas) == 2
    assert areas[0] < 20
    assert areas[1] > 200


def test_watershed_splits_touching_pins():
    depth = np.full((HEIGHT, WIDTH), BASE_Z, dtype=np.float32)
    _disc(depth, 70, 60, 5, BASE_Z + 2.0)
    _disc(depth, 80, 60, 5, BASE_Z + 2.0)

    merged = _tool(depth, min_points=1, close_px=0)
    assert len(merged.outputs["results"]) == 1

    split = _tool(
        depth, min_points=1, close_px=0, watershed=True,
        watershed_erode_px=2,
    )
    assert len(split.outputs["results"]) == 2


def test_direction_below_measures_pins_under_the_plane():
    depth = np.full((HEIGHT, WIDTH), BASE_Z, dtype=np.float32)
    for index in range(4):
        _disc(depth, 40 + index * 15, 60, 3, BASE_Z - 2.0)

    tool = _tool(depth, direction="below")
    features = tool.outputs["results"]
    assert len(features) == 4
    for feature in features:
        assert feature.h == pytest.approx(-2.0, abs=0.05)
        assert feature.depth == pytest.approx(2.0, abs=0.05)


def test_below_tip_is_the_deepest_point():
    depth = np.full((HEIGHT, WIDTH), BASE_Z, dtype=np.float32)
    # A pin whose left half is 2 mm deep and right half 4 mm deep.
    depth[50:70, 60:70] = BASE_Z - 2.0
    depth[50:70, 70:80] = BASE_Z - 4.0

    tool = _tool(
        depth, direction="below", close_px=1, open_px=0, sort_mode="none"
    )
    features = tool.outputs["results"]
    assert len(features) == 1
    feature = features[0]
    assert feature.h == pytest.approx(-4.0, abs=0.1)
    assert feature.depth == pytest.approx(4.0, abs=0.1)


def test_warning_mentions_the_direction():
    depth = np.full((HEIGHT, WIDTH), BASE_Z, dtype=np.float32)
    _disc(depth, 80, 60, 3, BASE_Z + 2.0)   # above the plane

    tool = _tool(depth, direction="below")
    warnings = tool.outputs.get("warnings") or []
    assert warnings
    assert "below" in warnings[0]
    assert "above" in warnings[0]


def test_features_are_published_as_world_points():
    from falcocv.core.overlay_layer import OverlayType

    depth = np.full((HEIGHT, WIDTH), BASE_Z, dtype=np.float32)
    for index in range(3):
        _disc(depth, 50 + index * 20, 60, 3, BASE_Z + 2.0)

    tool = _tool(depth)
    output = tool.outputs["output image"]
    layer = output.get_overlay_layer("default")
    world = [
        element for element in layer.elements
        if element.element_type == OverlayType.WORLD_POINTS
    ]
    assert len(world) == 1
    points = world[0].data["world_points"]
    assert len(points) == 3


def test_missing_plane_uses_capture_zero_plane():
    depth = np.zeros((HEIGHT, WIDTH), dtype=np.float32)
    _disc(depth, 80, 60, 4, 2.0)

    image = VisionImage(
        depth, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=20.0
    )
    tool = HeightBlobs3D()
    tool.set_input("input image", image)
    tool.set_input("reference plane", None)
    tool.params["rois"] = [
        {"roi": [0.0, 0.0, float(WIDTH), float(HEIGHT), 0.0],
         "z_min": 0.0, "z_max": 1.0}
    ]
    tool.run()

    assert tool.last_error is None
    assert tool.outputs["count"] == 1
    assert tool.outputs["results"][0].h == pytest.approx(2.0, abs=0.1)
    assert any("capture zero" in warning for warning in tool.outputs["warnings"])


def test_height_blobs_prefix_is_3dbl():
    from falcocv.core.tool_registry import get_registration_by_class

    assert get_registration_by_class(HeightBlobs3D).prefix == "[3DBL]"


def test_binary_mask_matches_the_detected_pins():
    depth = np.full((HEIGHT, WIDTH), BASE_Z, dtype=np.float32)
    _disc(depth, 40, 60, 3, BASE_Z + 2.0)

    tool = _tool(depth)
    mask = tool.outputs["binary mask"]
    assert mask is not None
    assert int(np.count_nonzero(mask.data)) > 0
