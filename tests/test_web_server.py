"""Smoke tests for the web HMI server."""

import unittest
import sys
import subprocess
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.web


class TestImageUtils(unittest.TestCase):
    """Regression tests for web.image_utils duck-typing."""

    def test_image_utils_does_not_import_core_vision_image(self):
        """web.image_utils must work without depending on core.vision_image."""
        root = Path(__file__).resolve().parents[1]
        script = ("import sys; sys.path.insert(0, 'src'); "
                  "import falcocv.web.image_utils; "
                  "print('falcocv.core.vision_image' in sys.modules)")
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=root,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        self.assertEqual(result.stdout.strip(), "False")

    def test_encode_vision_image_accepts_duck_typed_object(self):
        """Any object with .data and .overlay_layers should be encodable."""
        from falcocv.web.image_utils import encode_vision_image

        class FakeLayer:
            visible = True
            elements = []

        class FakeVisionImage:
            data = np.zeros((10, 20, 3), dtype=np.uint8)
            overlay_layers = {"layer0": FakeLayer()}

        result = encode_vision_image(FakeVisionImage())
        self.assertIn("data_b64", result)
        self.assertEqual(result["width"], 20)
        self.assertEqual(result["height"], 10)
        self.assertEqual(result["overlays"], [])


class TestWebServer(unittest.TestCase):

    def setUp(self):
        try:
            from fastapi.testclient import TestClient
            from falcocv.web.server import app
        except ImportError as e:
            self.skipTest(f"FastAPI / server dependencies not available: {e}")
        self.client = TestClient(app)

    def test_server_projects_returns_list(self):
        response = self.client.get("/api/server/projects")
        self.assertEqual(response.status_code, 200)
        self.assertIsInstance(response.json(), list)

    def test_api_not_found(self):
        response = self.client.get("/api/does-not-exist")
        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
