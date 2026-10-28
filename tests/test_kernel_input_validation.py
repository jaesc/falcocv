"""Input-validation tests for the minimal native kernel.

The kernel only contains point-cloud generation and plane fitting. These tests
verify that invalid inputs fail safely (exceptions or structured ``error``
dicts) and that the normal paths return the expected shapes.
"""
import unittest

import numpy as np

import falcocv.core  # noqa: F401  # sets up sys.path for _falco_kernel
from _falco_kernel import generate_pointcloud, plane_fit


class TestPointCloudKernel(unittest.TestCase):
    def test_1d_depth_raises(self):
        with self.assertRaises(Exception):
            generate_pointcloud(np.zeros(10, dtype=np.float32))

    def test_empty_depth_returns_empty_cloud(self):
        cloud = generate_pointcloud(
            np.zeros((0, 0), dtype=np.float32), 1.0, 1.0, 1.0, 0.0, 1000, True
        )
        self.assertEqual(cloud["points"].shape, (0, 3))
        self.assertEqual(cloud["intensity"].shape, (0,))

    def test_zero_as_invalid_filters_zeros(self):
        depth = np.ones((20, 20), dtype=np.float32)
        depth[:10] = 0.0
        cloud = generate_pointcloud(
            depth, 1.0, 1.0, 1.0, 0.0, 100000, True
        )
        self.assertEqual(len(cloud["points"]), 200)

    def test_zero_as_valid_keeps_zeros(self):
        depth = np.ones((20, 20), dtype=np.float32)
        depth[:10] = 0.0
        cloud = generate_pointcloud(
            depth, 1.0, 1.0, 1.0, 0.0, 100000, False
        )
        self.assertEqual(len(cloud["points"]), 400)

    def test_non_contiguous_input_is_supported(self):
        depth = np.asfortranarray(np.ones((30, 40), dtype=np.float32))
        cloud = generate_pointcloud(
            depth, 0.5, 0.5, 1.0, 2.0, 100000, True
        )
        self.assertEqual(len(cloud["points"]), 30 * 40)
        self.assertAlmostEqual(float(cloud["points"][0, 2]), 3.0, places=5)

    def test_bbox_and_subampling(self):
        depth = np.ones((100, 100), dtype=np.float32)
        cloud = generate_pointcloud(
            depth, 1.0, 1.0, 1.0, 0.0, 1000, True, "", True
        )
        self.assertLessEqual(len(cloud["points"]), 1200)
        self.assertEqual(cloud["bbox_center"].shape, (3,))
        self.assertEqual(cloud["bbox_size"].shape, (3,))
        self.assertTrue(np.all(cloud["bbox_size"] > 0))


class TestPlaneFitKernel(unittest.TestCase):
    def _cloud(self, n=5000):
        rng = np.random.default_rng(0)
        x = rng.uniform(-5, 5, n).astype(np.float32)
        y = rng.uniform(-5, 5, n).astype(np.float32)
        z = (0.25 * x - 0.1 * y + 3.0 + rng.normal(0, 0.005, n)).astype(np.float32)
        return np.column_stack([x, y, z]).astype(np.float32)

    def _input(self, points, method="lsq", rois=None):
        return {
            "image": np.ones((100, 100), dtype=np.float32),
            "points": points,
            "rois": rois or [],
            "calib": [1.0, 1.0, 1.0, 0.0],
            "params": {
                "method": method,
                "distance_threshold": 0.05,
                "max_iterations": 100,
                "min_points": 10,
                "max_fit_points": 50000,
            },
        }

    def test_missing_points_returns_error(self):
        self.assertIn("error", plane_fit({}))

    def test_wrong_shape_returns_error(self):
        out = plane_fit(self._input(np.zeros((10, 2), dtype=np.float32)))
        self.assertIn("error", out)

    def test_too_few_points_returns_error(self):
        out = plane_fit(self._input(np.zeros((2, 3), dtype=np.float32)))
        self.assertIn("error", out)

    def test_no_points_in_roi_returns_error(self):
        points = self._cloud()
        out = plane_fit(self._input(points, rois=[{
            "roi": [0, 0, 10, 10, 0], "z_min": 1000.0, "z_max": 1001.0,
        }]))
        self.assertIn("error", out)

    def test_lsq_fit(self):
        points = self._cloud()
        out = plane_fit(self._input(points, method="lsq"))
        self.assertNotIn("error", out)
        normal = np.asarray(out["normal"], dtype=np.float64)
        expected = np.array([-0.25, 0.1, 1.0]) / np.linalg.norm([-0.25, 0.1, 1.0])
        self.assertGreater(abs(float(np.dot(normal, expected))), 0.999)
        self.assertEqual(len(out["selected_indices"]), len(points))
        self.assertEqual(len(out["inlier_mask"]), len(points))
        self.assertGreaterEqual(len(out["polygon_2d"]), 3)

    def test_ransac_fit_with_outliers(self):
        points = self._cloud()
        rng = np.random.default_rng(1)
        idx = rng.choice(len(points), len(points) // 10, replace=False)
        points[idx, 2] += 4.0
        out = plane_fit(self._input(points, method="ransac"))
        self.assertNotIn("error", out)
        self.assertGreater(out["inlier_count"], 0)
        self.assertLess(out["rms_error"], 0.1)


if __name__ == "__main__":
    unittest.main()
