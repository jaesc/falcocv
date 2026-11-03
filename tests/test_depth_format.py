"""FalcoCV depth container and depth-loader format tests."""

import numpy as np
import tifffile

from falcocv.core.depth_format import load_depth_fcd, save_depth_fcd
from falcocv.core.pointcloud_utils import (
    _intensity_colors,
    ensure_pointcloud_colors,
)
from falcocv.tools.depth_range_loader.logic import DepthRangeLoader


def test_fcd_roundtrip_keeps_shape_dtype_and_values(tmp_path):
    depth = np.arange(48 * 32, dtype=np.uint16).reshape(48, 32)
    path = tmp_path / "frame.fcd"

    save_depth_fcd(path, depth)
    restored = load_depth_fcd(path)

    assert restored.dtype == np.uint16
    assert restored.shape == depth.shape
    assert np.array_equal(restored, depth)


def test_depth_loader_reads_fcd(tmp_path):
    depth = np.arange(48 * 32, dtype=np.uint16).reshape(48, 32)
    path = tmp_path / "frame.fcd"
    save_depth_fcd(path, depth)

    loader = DepthRangeLoader()
    loader.params.update(
        {
            "mode": "File",
            "file_path": str(path),
            "scale": 1.0,
            "generate_3d": False,
        }
    )
    loader.run()

    assert loader.last_error is None
    output = loader.outputs["output image"]
    assert output is not None
    assert output.data.dtype == np.float32
    assert np.array_equal(output.data, depth.astype(np.float32))


def test_depth_loader_scales_fcd_through_float32(tmp_path):
    depth = np.arange(48 * 32, dtype=np.uint16).reshape(48, 32)
    path = tmp_path / "frame.fcd"
    save_depth_fcd(path, depth)

    loader = DepthRangeLoader()
    loader.params.update(
        {
            "mode": "File",
            "file_path": str(path),
            "scale": 0.5,
            "generate_3d": False,
        }
    )
    loader.run()

    assert loader.last_error is None
    assert np.array_equal(
        loader.outputs["output image"].data,
        depth.astype(np.float32) * np.float32(0.5),
    )


def test_depth_loader_defers_pointcloud_colors(tmp_path):
    depth = np.arange(48 * 32, dtype=np.uint16).reshape(48, 32)
    path = tmp_path / "frame.fcd"
    save_depth_fcd(path, depth)

    loader = DepthRangeLoader()
    loader.params.update(
        {
            "mode": "File",
            "file_path": str(path),
            "scale": 1.0,
            "generate_3d": True,
            "target_points": 1_000_000,
        }
    )
    loader.run()

    output = loader.outputs["output image"]
    assert output._3d_data is not None
    assert output._3d_data["colors"] is None
    assert ensure_pointcloud_colors(output) is True
    expected = _intensity_colors(output._3d_data["intensity"], "TURBO")
    assert np.array_equal(output._3d_data["colors"], expected)
    assert ensure_pointcloud_colors(output) is True


def test_depth_loader_reads_lzw_tiff(tmp_path):
    depth = np.arange(48 * 32, dtype=np.uint16).reshape(48, 32)
    path = tmp_path / "frame.tiff"
    tifffile.imwrite(path, depth, compression="lzw", predictor=2)

    loader = DepthRangeLoader()
    loader.params.update(
        {
            "mode": "File",
            "file_path": str(path),
            "scale": 1.0,
            "generate_3d": False,
        }
    )
    loader.run()

    assert loader.last_error is None
    output = loader.outputs["output image"]
    assert output is not None
    assert np.array_equal(output.data, depth.astype(np.float32))
