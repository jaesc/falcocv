"""Pixel-map filter: transfer curve, dtypes and the empty-chain fast path."""

import numpy as np
import pytest

from falcocv.core.pointcloud_utils import ensure_pointcloud_data
from falcocv.core.vision_image import VisionImage
from falcocv.tools.filters.logic import FiltersTool, apply_filter
from falcocv.tools.filters.pixel_map import (
    apply_pixel_map,
    apply_range,
    identity_points,
    validate_curve,
)

DOMAIN = (0.0, 255.0)
CURVE = [
    [0.0, 0.0],
    [100.0, 0.0],
    [100.0, 180.0],
    [150.0, 255.0],
    [150.0, 150.0],
    [255.0, 255.0],
]


def test_identity_curve():
    assert identity_points(DOMAIN) == [[0.0, 0.0], [255.0, 255.0]]


def test_apply_range_identity_is_a_no_op():
    points = apply_range(identity_points(DOMAIN), DOMAIN, 0, 255, 0, 255)
    assert points == [[0.0, 0.0], [255.0, 255.0]]


def test_apply_range_builds_flat_ranges_and_a_ramp():
    points = identity_points(DOMAIN)
    points = apply_range(points, DOMAIN, 0, 100, 0, 0)
    points = apply_range(points, DOMAIN, 100, 150, 180, 255)
    assert points == [
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 180.0],
        [150.0, 255.0],
        [150.0, 150.0],
        [255.0, 255.0],
    ]


def test_apply_range_overwrites_a_previous_range():
    points = apply_range(identity_points(DOMAIN), DOMAIN, 0, 200, 0, 0)
    points = apply_range(points, DOMAIN, 100, 150, 180, 255)
    image = np.array([50, 100, 150, 200], dtype=np.uint8)
    mapped = apply_pixel_map(image, {"domain": DOMAIN, "points": points})
    assert mapped.tolist() == [0, 180, 0, 200]


def test_validate_curve_rejects_bad_shapes():
    assert validate_curve([[0, 0]], DOMAIN, 0, 255) is not None
    assert validate_curve([[0, 0], [300, 300]], DOMAIN, 0, 255) is not None
    assert validate_curve([[0, 0], [255, 300]], DOMAIN, 0, 255) is not None
    assert validate_curve([[255, 255], [0, 0]], DOMAIN, 0, 255) is not None


@pytest.mark.parametrize("value, expected", [
    (0, 0), (50, 0), (99, 0), (100, 180), (125, 218), (149, 254),
    (150, 150), (200, 200), (240, 240), (255, 255),
])
def test_uint8_mapping_values(value, expected):
    image = np.full((4, 4), value, dtype=np.uint8)
    mapped = apply_pixel_map(image, {"domain": DOMAIN, "points": CURVE})
    assert mapped.dtype == np.uint8
    assert int(mapped[0, 0]) == expected


def test_uint8_bgr_is_mapped_per_channel():
    image = np.zeros((2, 2, 3), dtype=np.uint8)
    image[..., 0] = 100
    image[..., 1] = 125
    image[..., 2] = 255
    mapped = apply_pixel_map(image, {"domain": DOMAIN, "points": CURVE})
    assert mapped.shape == image.shape
    assert mapped[0, 0].tolist() == [180, 218, 255]


def test_uint16_mapping_uses_the_full_dtype_range():
    domain = (0.0, 65535.0)
    points = [[0.0, 0.0], [65535.0, 0.0]]
    image = np.array([[0, 1, 65535]], dtype=np.uint16)
    mapped = apply_pixel_map(image, {"domain": domain, "points": points})
    assert mapped.dtype == np.uint16
    assert mapped.tolist() == [[0, 0, 0]]


def test_float32_keeps_values_outside_the_curve():
    domain = (0.0, 255.0)
    image = np.array([[50.0, 125.0, 300.0]], dtype=np.float32)
    mapped = apply_pixel_map(image, {"domain": domain, "points": CURVE})
    assert mapped.dtype == np.float32
    assert mapped[0, 0] == pytest.approx(0.0)
    assert mapped[0, 1] == pytest.approx(217.5)
    assert mapped[0, 2] == pytest.approx(300.0)  # outside the domain


def test_preserve_invalid_keeps_zero_and_nan():
    domain = (0.0, 255.0)
    white = [[0.0, 255.0], [255.0, 255.0]]
    image = np.array([[0.0, float("nan"), 10.0]], dtype=np.float32)
    kept = apply_pixel_map(
        image, {"domain": domain, "points": white, "preserve_invalid": True}
    )
    assert kept[0, 0] == 0.0
    assert np.isnan(kept[0, 1])
    assert kept[0, 2] == pytest.approx(255.0)

    mapped = apply_pixel_map(
        image, {"domain": domain, "points": white, "preserve_invalid": False}
    )
    assert mapped[0, 0] == pytest.approx(255.0)
    assert np.isnan(mapped[0, 1])
    assert mapped[0, 2] == pytest.approx(255.0)


def test_apply_filter_pixel_map_dispatches():
    image = np.array([[0, 100, 255]], dtype=np.uint8)
    mapped = apply_filter(
        image, "pixel_map", {"domain": DOMAIN, "points": CURVE}
    )
    assert mapped.tolist() == [[0, 180, 255]]


# ── FiltersTool integration ────────────────────────────────────────────


def _depth_image():
    rng = np.random.default_rng(0)
    depth = rng.uniform(1000, 3000, (64, 80)).astype(np.float32)
    depth[::7, ::7] = 0.0
    return VisionImage(
        depth, scale_x=0.05, scale_y=0.05, scale_z=0.01, offset_z=-20.0
    )


def _tool(image):
    tool = FiltersTool()
    tool.inputs["input image"] = image
    return tool


def test_empty_chain_publishes_the_input_by_reference():
    image = _depth_image()
    ensure_pointcloud_data(image, target_points=100_000)
    tool = _tool(image)
    tool.run()

    out = tool.outputs["output image"]
    assert out is image
    assert out._3d_data is image._3d_data
    assert tool.last_execution_time_ms >= 0.0


def test_disabled_filters_also_short_circuit():
    image = _depth_image()
    tool = _tool(image)
    tool.params["filters"] = [
        {"enabled": False, "type": "gaussian_blur", "kernel_size": 5, "sigma": 1.0},
    ]
    tool.run()
    assert tool.outputs["output image"] is image


def test_empty_chain_with_roi_overlay_shares_pixels_and_cloud():
    image = _depth_image()
    ensure_pointcloud_data(image, target_points=100_000)
    tool = _tool(image)
    tool.params["roi"] = [10, 10, 40, 30, 0]
    tool.run()

    out = tool.outputs["output image"]
    assert out is not image
    assert out.data is image.data
    assert out._3d_data is image._3d_data
    assert out.overlay_layers


def test_gray_on_range_defaults_to_normalised_float32():
    image = _depth_image()
    tool = _tool(image)
    tool.params["filters"] = [
        {"enabled": True, "type": "color_conversion", "mode": "gray"}
    ]
    tool.run()

    out = tool.outputs["output image"]
    assert out.data.dtype == np.float32
    assert out.data.min() >= 0.0
    assert out.data.max() <= 255.0


def test_gray_on_range_can_publish_uint8():
    image = _depth_image()
    tool = _tool(image)
    tool.params["filters"] = [
        {"enabled": True, "type": "color_conversion", "mode": "gray"}
    ]
    tool.params["output_dtype"] = "uint8"
    tool.run()

    out = tool.outputs["output image"]
    assert out.data.dtype == np.uint8
    assert out.data.max() <= 255
    assert out._3d_data is None


def test_pixel_map_regenerates_the_depth_cloud():
    image = _depth_image()
    ensure_pointcloud_data(image, target_points=100_000)
    tool = _tool(image)
    tool.params["filters"] = [{
        "enabled": True,
        "type": "pixel_map",
        "domain": [0.0, 3200.0],
        "points": [[0.0, 0.0], [3200.0, 0.0]],
        "preserve_invalid": True,
    }]
    tool.run()

    out = tool.outputs["output image"]
    assert out is not image
    assert out.data.dtype == np.float32
    assert out._3d_data is not image._3d_data
    assert float(out.data.max()) == pytest.approx(0.0, abs=1e-6)
