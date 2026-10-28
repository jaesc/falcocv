"""
Tests for the VisionImage world coordinate mapping system (C++ backend).

Covers:
  - Default state (no calibration)
  - world_matrix shape and dtype
  - Round-trip identity when no calibration is set
  - clear_world behavior
  - Stubs that are not yet fully implemented are marked as expectedFailure/skip.
"""

import numpy as np
import cv2
import unittest
from falcocv.core.vision_image import VisionImage


def _make_test_image(height=480, width=640):
    """Create a simple test image."""
    return np.zeros((height, width, 3), dtype=np.uint8)


class TestWorldDefaultState(unittest.TestCase):
    """World system must start clean and valid."""

    def test_identity_matrix(self):
        img = VisionImage(_make_test_image())
        M = img.world_matrix
        self.assertEqual(M.shape, (4, 4))
        self.assertEqual(M.dtype, np.float64)
        np.testing.assert_array_equal(M, np.eye(4))

    def test_image_to_world_default_returns_identity(self):
        img = VisionImage(_make_test_image())
        wx, wy, wz = img.image_to_world(100, 200)
        self.assertAlmostEqual(wx, 100.0)
        self.assertAlmostEqual(wy, 200.0)
        self.assertAlmostEqual(wz, 0.0)

    def test_world_to_image_default_returns_identity(self):
        img = VisionImage(_make_test_image())
        ix, iy = img.world_to_image(100, 200)
        self.assertAlmostEqual(ix, 100.0)
        self.assertAlmostEqual(iy, 200.0)


class TestAffineCalibration(unittest.TestCase):
    """3-point affine transform: translation, rotation, scale.

    NOTE: set_world_from_points is currently a stub in the C++ backend.
    These tests document the expected behavior once calibration is implemented.
    """

    def setUp(self):
        self.img = VisionImage(_make_test_image())

    def test_translation_only(self):
        """3 points shifted by (50, 30) in world."""
        img_pts = [(100, 100), (200, 100), (100, 200)]
        world_pts = [(150, 130), (250, 130), (150, 230)]
        self.img.set_world_from_points(img_pts, world_pts, method='affine')
        # If calibration is implemented, world_matrix must differ from identity.
        self.assertFalse(np.allclose(self.img.world_matrix, np.eye(4)))
        wx, wy, _ = self.img.image_to_world(100, 100)
        self.assertAlmostEqual(wx, 150.0, places=4)
        self.assertAlmostEqual(wy, 130.0, places=4)

    def test_round_trip(self):
        """image → world → image must return original point."""
        img_pts = [(100, 100), (200, 100), (100, 200)]
        world_pts = [(150, 130), (250, 130), (150, 230)]
        self.img.set_world_from_points(img_pts, world_pts, method='affine')
        self.assertFalse(np.allclose(self.img.world_matrix, np.eye(4)))
        for px, py in [(100, 100), (175, 150), (200, 200)]:
            wx, wy, _ = self.img.image_to_world(px, py)
            ix, iy = self.img.world_to_image(wx, wy)
            self.assertAlmostEqual(px, ix, places=4)
            self.assertAlmostEqual(py, iy, places=4)

    def test_set_world_from_points_does_not_crash(self):
        """The stub must accept valid input without crashing."""
        self.img.set_world_from_points(
            [(0, 0), (100, 0), (0, 100)],
            [(10, 10), (110, 10), (10, 110)],
            method='affine',
        )


class TestHomographyCalibration(unittest.TestCase):
    """4-point homography: perspective transform (currently a stub)."""

    def setUp(self):
        self.img = VisionImage(_make_test_image())

    def test_round_trip_with_perspective(self):
        """Simulate a perspective distortion and verify round-trip."""
        img_pts = [(100, 100), (300, 120), (80, 300), (320, 280)]
        world_pts = [(0, 0), (100, 0), (0, 100), (100, 100)]
        self.img.set_world_from_points(img_pts, world_pts)
        self.assertFalse(np.allclose(self.img.world_matrix, np.eye(4)))
        for px, py in img_pts:
            wx, wy, wz = self.img.image_to_world(px, py)
            ix, iy = self.img.world_to_image(wx, wy)
            self.assertAlmostEqual(px, ix, places=2)
            self.assertAlmostEqual(py, iy, places=2)
            self.assertAlmostEqual(wz, 0.0)

    def test_set_world_from_points_homography_does_not_crash(self):
        self.img.set_world_from_points(
            [(100, 100), (300, 120), (80, 300), (320, 280)],
            [(0, 0), (100, 0), (0, 100), (100, 100)],
        )


class TestCopyPreservesWorld(unittest.TestCase):
    """copy() must preserve world_matrix."""

    def test_copy_preserves_identity_matrix(self):
        img = VisionImage(_make_test_image())
        copied = img.copy()
        np.testing.assert_array_equal(copied.world_matrix, img.world_matrix)
        self.assertEqual(copied.world_matrix.dtype, np.float64)


class TestClearWorld(unittest.TestCase):
    """clear_world() must reset to identity."""

    def test_clear_resets_state(self):
        img = VisionImage(_make_test_image())
        img.set_world_from_points(
            [(0, 0), (100, 0), (0, 100), (100, 100)],
            [(10, 10), (110, 10), (10, 110), (110, 110)]
        )
        img.clear_world()
        np.testing.assert_array_equal(img.world_matrix, np.eye(4))


@unittest.skip("find_calibration_grid is not implemented in the C++ backend yet")
class TestFindCalibrationGrid(unittest.TestCase):
    """Static method for automatic pattern detection."""

    def test_checkerboard_none_on_noise(self):
        noise = np.random.randint(0, 256, (200, 200), dtype=np.uint8)
        result = VisionImage.find_calibration_grid(
            noise, pattern='checkerboard', grid_size=(4, 4)
        )
        self.assertIsNone(result)

    def test_checkerboard_detects_synthetic(self):
        grid = np.zeros((240, 320), dtype=np.uint8)
        cell_h, cell_w = 30, 40
        for row in range(8):
            for col in range(8):
                if (row + col) % 2 == 1:
                    y0, x0 = row * cell_h, col * cell_w
                    grid[y0:y0 + cell_h, x0:x0 + cell_w] = 255
        grid = cv2.GaussianBlur(grid, (3, 3), 0.5)
        result = VisionImage.find_calibration_grid(
            grid, pattern='checkerboard', grid_size=(7, 7), square_size=10.0
        )
        self.assertIsNotNone(result)
        img_pts, world_pts = result
        self.assertEqual(len(img_pts), 49)
        self.assertEqual(len(world_pts), 49)
        np.testing.assert_almost_equal(world_pts[1, 0] - world_pts[0, 0], 10.0)
        np.testing.assert_almost_equal(world_pts[7, 1] - world_pts[0, 1], 10.0)


@unittest.skip("draw_world_grid is not implemented in the C++ backend yet")
class TestDrawWorldGrid(unittest.TestCase):
    """draw_world_grid must create overlay lines."""

    def test_grid_creates_overlay(self):
        img = VisionImage(_make_test_image())
        img.draw_world_grid(50, (0, 200), (0, 200))
        layer = img.get_overlay_layer('world_grid', create=False)
        self.assertIsNotNone(layer)
        self.assertTrue(len(layer.elements) > 0)


class TestInvalidInputs(unittest.TestCase):
    """Edge cases and error handling."""

    def test_set_world_from_points_accepts_insufficient_points(self):
        """The current stub does not validate point count; it must not crash."""
        img = VisionImage(_make_test_image())
        img.set_world_from_points([(0, 0)], [(10, 10)])


if __name__ == '__main__':
    unittest.main()
