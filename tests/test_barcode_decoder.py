#!/usr/bin/env python3
"""
Unit tests for Barcode Decoder Tool (C++ backend via _cv_core).
"""
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import unittest
import cv2
import numpy as np
from falcocv.core.vision_image import VisionImage
from falcocv.tools.barcode_decoder import BarcodeDecoder, CodeResult


def _make_qr(text, img_size=200, scale=6):
    """Generate a QR code image scaled up for reliable detection."""
    params = cv2.QRCodeEncoder.Params()
    params.correction_level = 1  # M
    encoder = cv2.QRCodeEncoder.create(params)
    qr = encoder.encode(text)

    qr_scaled = cv2.resize(qr, (qr.shape[0] * scale, qr.shape[1] * scale),
                           interpolation=cv2.INTER_NEAREST)

    border = 4 * scale
    white = np.ones((qr_scaled.shape[0] + 2 * border,
                     qr_scaled.shape[1] + 2 * border), dtype=np.uint8) * 255
    white[border:border + qr_scaled.shape[0],
          border:border + qr_scaled.shape[1]] = qr_scaled

    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    h, w = white.shape
    ox = max(0, (img_size - w) // 2)
    oy = max(0, (img_size - h) // 2)
    qr_bgr = cv2.cvtColor(white, cv2.COLOR_GRAY2BGR)
    img[oy:oy + h, ox:ox + w] = qr_bgr
    return img


def _make_datamatrix(text, img_size=200):
    """Generate a DataMatrix code image using pylibdmtx."""
    import types as _t
    import sys as _sys
    _dv = _t.ModuleType('distutils.version')
    try:
        from setuptools._distutils.version import LooseVersion
        _dv.LooseVersion = LooseVersion
    except ImportError:
        class _LV:
            def __init__(self, v): self.v = v
            def __ge__(self, o): return True
        _dv.LooseVersion = _LV
    _d = _t.ModuleType('distutils')
    _d.version = _dv
    _sys.modules['distutils'] = _d
    _sys.modules['distutils.version'] = _dv

    from pylibdmtx import pylibdmtx
    encoded = pylibdmtx.encode(text.encode() if isinstance(text, str) else text)
    arr = np.frombuffer(encoded.pixels, dtype=np.uint8).reshape(
        (encoded.height, encoded.width, 3))

    h, w = arr.shape[:2]
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    ox = max(0, (img_size - w) // 2)
    oy = max(0, (img_size - h) // 2)
    img[oy:oy + h, ox:ox + w] = arr
    return img


def _make_aruco(marker_id=42, dict_name="DICT_6X6_250", img_size=300, marker_size=80):
    """Generate an ArUco marker image centered in the canvas."""
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dict_name))
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    marker_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    pad = (img_size - marker_size) // 2
    img[pad:pad + marker_size, pad:pad + marker_size] = marker_bgr
    return img


def _run_decoder(img, code_type="Auto", **param_overrides):
    """Helper: create decoder, set params, run and return tool."""
    vision_img = VisionImage(img)
    decoder = BarcodeDecoder("Barcode Decoder")
    decoder.params["code_type"] = code_type
    decoder.params["roi"] = [0.0, 0.0, 0.0, 0.0, 0.0]
    for key, value in param_overrides.items():
        if isinstance(value, dict) and key in decoder.params:
            decoder.params[key].update(value)
        else:
            decoder.params[key] = value
    decoder.inputs["input image"] = vision_img
    decoder.run()
    return decoder


class TestQRDetection(unittest.TestCase):
    def test_qr_detection(self):
        decoder = _run_decoder(_make_qr("Hello FalcoCV"), code_type="QR Code")
        self.assertGreater(decoder.outputs["count"], 0)
        results = decoder.outputs["results"]
        self.assertGreater(results.Count, 0)
        res = results[0]
        self.assertIsInstance(res, CodeResult)
        self.assertEqual(res.text, "Hello FalcoCV")
        self.assertEqual(res.format, "QR")
        self.assertIsNotNone(res.angle)

    def test_qr_with_params(self):
        decoder = _run_decoder(
            _make_qr("EpsTest"),
            code_type="QR Code",
            qr_params={"eps_x": 0.5, "eps_y": 0.5},
        )
        self.assertGreater(decoder.outputs["count"], 0)
        res = decoder.outputs["results"][0]
        self.assertEqual(res.text, "EpsTest")


class TestArUcoDetection(unittest.TestCase):
    def test_aruco_detection(self):
        decoder = _run_decoder(
            _make_aruco(marker_id=42, dict_name="DICT_6X6_250"),
            code_type="ArUco",
            aruco_params={"dict_name": "DICT_6X6_250"},
        )
        self.assertGreater(decoder.outputs["count"], 0)
        res = decoder.outputs["results"][0]
        self.assertEqual(res.text, "42")
        self.assertIn("ArUco", res.format)

    def test_aruco_with_params(self):
        decoder = _run_decoder(
            _make_aruco(marker_id=99, dict_name="DICT_6X6_250"),
            code_type="ArUco",
            aruco_params={
                "dict_name": "DICT_6X6_250",
                "adaptiveThreshConstant": 5,
                "cornerRefinementMethod": 1,
            },
        )
        self.assertGreater(decoder.outputs["count"], 0)
        res = decoder.outputs["results"][0]
        self.assertEqual(res.text, "99")


class TestDataMatrixDetection(unittest.TestCase):
    def test_datamatrix_detection(self):
        decoder = _run_decoder(_make_datamatrix("Hello DM"), code_type="DataMatrix")
        self.assertGreater(decoder.outputs["count"], 0)
        res = decoder.outputs["results"][0]
        self.assertTrue("Hello" in res.text or "DM" in res.text, f"Unexpected text: {res.text}")
        self.assertEqual(res.format, "DataMatrix")

    def test_datamatrix_with_params(self):
        decoder_default = _run_decoder(_make_datamatrix("ParamsTest"), code_type="DataMatrix")
        self.assertGreater(decoder_default.outputs["count"], 0)

        decoder_custom = _run_decoder(
            _make_datamatrix("ParamsTest"),
            code_type="DataMatrix",
            dmtx_params={"timeout_ms": 10000, "shrink": 1},
        )
        self.assertGreater(decoder_custom.outputs["count"], 0)

    def test_datamatrix_sharpen(self):
        decoder = _run_decoder(
            _make_datamatrix("SharpTest"),
            code_type="DataMatrix",
            dmtx_params={"enable_sharpening": True},
        )
        self.assertGreater(decoder.outputs["count"], 0)


class TestDeduplication(unittest.TestCase):
    def test_auto_mode_returns_single_result_for_one_code(self):
        """Auto mode should return a single result for a single QR code."""
        decoder = _run_decoder(_make_qr("DedupTest"), code_type="Auto")
        self.assertEqual(decoder.outputs["count"], 1)


class TestBarcodeDecoderPipeline(unittest.TestCase):
    def test_pipeline_with_fixture(self):
        decoder = _run_decoder(_make_qr("FixtureTest"), code_type="QR Code")
        self.assertGreater(decoder.outputs["count"], 0)
        results = decoder.outputs["results"]
        self.assertGreater(results.Count, 0)
        self.assertIsInstance(results[0], CodeResult)

    def test_output_shares_data(self):
        img = _make_qr("NoCopyTest")
        vision_img = VisionImage(img)
        decoder = BarcodeDecoder("Barcode Decoder")
        decoder.params["code_type"] = "QR Code"
        decoder.params["roi"] = [0.0, 0.0, 0.0, 0.0, 0.0]
        decoder.inputs["input image"] = vision_img
        decoder.run()

        out = decoder.outputs["output image"]
        self.assertIsNot(out, vision_img)
        self.assertIs(out.data, vision_img.data)

    def test_rotated_fixture_coordinates(self):
        img = _make_qr("RotatedFix", img_size=300, scale=4)
        h, w = img.shape[:2]

        vision_img = VisionImage(img, x=w/2.0, y=h/2.0, angle=30.0)
        decoder = BarcodeDecoder("Barcode Decoder")
        decoder.params["code_type"] = "QR Code"
        decoder.params["roi"] = [-120.0, -120.0, 240.0, 240.0, 0.0]
        decoder.inputs["input image"] = vision_img
        decoder.run()

        self.assertGreater(decoder.outputs["count"], 0)
        results = decoder.outputs["results"]
        out_img = decoder.outputs["output image"]

        layer = out_img.get_overlay_layer("default", create=False)
        self.assertIsNotNone(layer)

        for res in results:
            self.assertTrue(np.isfinite(res.x))
            self.assertTrue(np.isfinite(res.y))
            self.assertGreaterEqual(res.x, 0)
            self.assertLess(res.x, w)
            self.assertGreaterEqual(res.y, 0)
            self.assertLess(res.y, h)


def _make_1d_barcode(text, code_type="code128", img_size=300, height=80):
    """Generate a 1D barcode image using python-barcode."""
    from barcode import Code128, EAN13
    from barcode.writer import ImageWriter
    from io import BytesIO

    if code_type == "code128":
        barcode_obj = Code128(text, writer=ImageWriter())
    elif code_type == "ean13":
        # EAN-13 requires 12 or 13 digits; pad with a leading zero if needed.
        digits = "".join(c for c in text if c.isdigit())
        if len(digits) < 12:
            digits = digits.zfill(12)
        elif len(digits) > 13:
            digits = digits[:13]
        barcode_obj = EAN13(digits, writer=ImageWriter())
    else:
        raise ValueError(f"Unsupported barcode type: {code_type}")

    buffer = BytesIO()
    barcode_obj.write(buffer, options={
        "module_height": height,
        "module_width": 2,
        "quiet_zone": 2,
        "font_size": 0,
        "text_distance": 0,
    })
    buffer.seek(0)
    pil_img = cv2.imdecode(np.frombuffer(buffer.read(), dtype=np.uint8), cv2.IMREAD_COLOR)
    if pil_img is None:
        raise RuntimeError("Failed to generate barcode image")

    # Resize to fit the requested canvas while preserving aspect ratio.
    h, w = pil_img.shape[:2]
    scale = min((img_size - 40) / w, (img_size - 40) / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(pil_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.ones((img_size, img_size, 3), dtype=np.uint8) * 255
    ox = (img_size - new_w) // 2
    oy = (img_size - new_h) // 2
    canvas[oy:oy + new_h, ox:ox + new_w] = resized
    return canvas


class Test1DBarcodeDetection(unittest.TestCase):
    def test_code128_detection(self):
        decoder = _run_decoder(_make_1d_barcode("TEST1234"), code_type="Barcode 1D")
        self.assertGreater(decoder.outputs["count"], 0)
        res = decoder.outputs["results"][0]
        self.assertIn("TEST1234", res.text)
        self.assertIn("CODE128", res.format.upper())

    def test_ean13_detection(self):
        decoder = _run_decoder(_make_1d_barcode("123456789012", code_type="ean13"), code_type="Barcode 1D")
        self.assertGreater(decoder.outputs["count"], 0)
        res = decoder.outputs["results"][0]
        self.assertIn("EAN13", res.format.upper())

    def test_bgr_image_does_not_crash_on_otsu(self):
        """Ensure a color image with low contrast does not crash the Otsu fallback."""
        img = _make_1d_barcode("BGRTEST")
        # Pass through run() with a color image.
        decoder = _run_decoder(img, code_type="Barcode 1D", preprocessing={"strategy": "always_no_clahe"})
        self.assertIsInstance(decoder.outputs["count"], int)

    def test_1d_with_clahe(self):
        img = _make_1d_barcode("CLAHE123")
        decoder = _run_decoder(img, code_type="Barcode 1D", preprocessing={"strategy": "always_clahe"})
        self.assertGreater(decoder.outputs["count"], 0)

    def test_multiple_barcodes_same_image(self):
        """Two 1D codes in the same image should both be detected."""
        img1 = _make_1d_barcode("CODE1111", img_size=200, height=60)
        img2 = _make_1d_barcode("CODE2222", img_size=200, height=60)
        canvas = np.ones((300, 500, 3), dtype=np.uint8) * 255
        canvas[50:250, 20:220] = cv2.resize(img1, (200, 200))
        canvas[50:250, 260:460] = cv2.resize(img2, (200, 200))
        decoder = _run_decoder(canvas, code_type="Barcode 1D", max_count=0)
        self.assertGreaterEqual(decoder.outputs["count"], 2)


class TestReal1DDataset(unittest.TestCase):
    """Validate against the real barcode_1D dataset in test_pictures/.

    NOTE: some images are intentionally hard (motion blur, low contrast,
    rotation).  Tests for those are marked as expectedFailure until the
    decoder can handle them.
    """

    @classmethod
    def setUpClass(cls):
        cls.dataset_dir = Path(__file__).resolve().parents[1] / "test_pictures" / "barcode_1D"
        cls.has_dataset = cls.dataset_dir.is_dir()

    def _decode_file(self, filename, code_type="Barcode 1D", **overrides):
        if not self.has_dataset:
            self.skipTest("Real barcode dataset not available")
        path = self.dataset_dir / filename
        self.assertTrue(path.exists(), f"Missing dataset file: {path}")
        img = cv2.imread(str(path))
        self.assertIsNotNone(img)
        return _run_decoder(img, code_type=code_type, max_count=0, **overrides)

    def test_image_1_code128(self):
        decoder = self._decode_file("1.jpg")
        self.assertGreaterEqual(decoder.outputs["count"], 1)
        self.assertTrue(any("128" in r.format.upper() for r in decoder.outputs["results"]))

    @unittest.expectedFailure
    def test_image_2_rotated_code128(self):
        decoder = self._decode_file("2.jpg")
        self.assertGreaterEqual(decoder.outputs["count"], 1)

    @unittest.expectedFailure
    def test_image_3_rotated_code128(self):
        decoder = self._decode_file("3.jpg")
        self.assertGreaterEqual(decoder.outputs["count"], 1)

    @unittest.expectedFailure
    def test_image_4_two_code128(self):
        decoder = self._decode_file("4.jpg")
        self.assertGreaterEqual(decoder.outputs["count"], 2)

    def test_image_5_at_least_one_code128(self):
        decoder = self._decode_file("5.jpg")
        self.assertGreaterEqual(decoder.outputs["count"], 1)
        self.assertTrue(any("128" in r.format.upper() for r in decoder.outputs["results"]))

    def test_image_6_ean13(self):
        decoder = self._decode_file("6.jpg")
        self.assertGreaterEqual(decoder.outputs["count"], 1)
        self.assertTrue(any("EAN" in r.format.upper() for r in decoder.outputs["results"]))

    def test_images_7_and_8_ean13(self):
        for filename in ("7.jpg", "8.jpg"):
            decoder = self._decode_file(filename)
            self.assertGreaterEqual(decoder.outputs["count"], 1, f"{filename}")
            self.assertTrue(any("EAN" in r.format.upper() for r in decoder.outputs["results"]), f"{filename}")

    def test_image_9_multiple_codes(self):
        decoder = self._decode_file("9.jpg")
        self.assertGreaterEqual(decoder.outputs["count"], 1)


class TestQRAdvanced(unittest.TestCase):
    def test_qr_rotated(self):
        base = _make_qr("RotatedQR")
        for angle in (15, 45, 90):
            h, w = base.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(base, M, (w, h), borderValue=(255, 255, 255))
            decoder = _run_decoder(rotated, code_type="QR Code")
            self.assertGreater(decoder.outputs["count"], 0, f"angle={angle}")
            self.assertEqual(decoder.outputs["results"][0].text, "RotatedQR")

    def test_qr_perspective(self):
        base = _make_qr("PerspectiveQR", img_size=300, scale=8)
        h, w = base.shape[:2]
        src_pts = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        dst_pts = np.array([[30, 20], [w - 40, 10], [w - 20, h - 30], [10, h - 20]], dtype=np.float32)
        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        warped = cv2.warpPerspective(base, M, (w, h), borderValue=(255, 255, 255))
        decoder = _run_decoder(warped, code_type="QR Code")
        self.assertGreater(decoder.outputs["count"], 0)
        self.assertEqual(decoder.outputs["results"][0].text, "PerspectiveQR")


class TestArUcoAdvanced(unittest.TestCase):
    def test_aruco_rotated(self):
        for angle in (15, 45, 90):
            base = _make_aruco(marker_id=77, dict_name="DICT_6X6_250", img_size=400, marker_size=100)
            h, w = base.shape[:2]
            M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
            rotated = cv2.warpAffine(base, M, (w, h), borderValue=(200, 200, 200))
            decoder = _run_decoder(rotated, code_type="ArUco", aruco_params={"dict_name": "DICT_6X6_250"})
            self.assertGreater(decoder.outputs["count"], 0, f"angle={angle}")
            self.assertEqual(decoder.outputs["results"][0].text, "77")

    def test_aruco_blur(self):
        base = _make_aruco(marker_id=88, dict_name="DICT_6X6_250")
        blurred = cv2.GaussianBlur(base, (3, 3), 0.8)
        decoder = _run_decoder(blurred, code_type="ArUco", aruco_params={"dict_name": "DICT_6X6_250"})
        self.assertGreater(decoder.outputs["count"], 0)
        self.assertEqual(decoder.outputs["results"][0].text, "88")

    def test_aruco_multiple_markers(self):
        canvas = np.ones((400, 600, 3), dtype=np.uint8) * 200
        for idx, (x, y) in enumerate([(50, 50), (300, 50), (150, 220)]):
            marker = cv2.aruco.generateImageMarker(
                cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250),
                idx + 10, 80
            )
            marker_bgr = cv2.cvtColor(marker, cv2.COLOR_GRAY2BGR)
            canvas[y:y + 80, x:x + 80] = marker_bgr
        decoder = _run_decoder(canvas, code_type="ArUco", aruco_params={"dict_name": "DICT_6X6_250"}, max_count=0)
        self.assertGreaterEqual(decoder.outputs["count"], 3)


class TestPreprocessing(unittest.TestCase):
    def test_preprocessing_strategies(self):
        img = _make_qr("PreprocTest")
        for strategy in ("always_no_clahe", "always_clahe"):
            decoder = _run_decoder(
                img,
                code_type="QR Code",
                preprocessing={"strategy": strategy},
            )
            self.assertGreater(decoder.outputs["count"], 0, f"strategy={strategy}")


class TestCanonicalAngle(unittest.TestCase):
    @staticmethod
    def _rect_points(w, h, angle):
        rad = math.radians(angle)
        c, s = math.cos(rad), math.sin(rad)
        corners = [(-w / 2, -h / 2), (w / 2, -h / 2), (w / 2, h / 2), (-w / 2, h / 2)]
        return [(c * x - s * y, s * x + c * y) for x, y in corners]

    def test_long_edge_is_clockwise(self):
        pts = self._rect_points(100.0, 30.0, 20.0)
        angle = BarcodeDecoder._canonical_angle(pts)
        self.assertAlmostEqual(angle, 20.0, places=6)

    def test_square_is_modulo_90(self):
        pts = self._rect_points(60.0, 60.0, 20.0)
        angle = BarcodeDecoder._canonical_angle(pts)
        diff = (angle - 20.0 + 45.0) % 90.0 - 45.0
        self.assertAlmostEqual(diff, 0.0, places=6)

    def test_qr_rotated_angle_and_fixture_offset(self):
        base = _make_qr("RotatedAngle")
        h, w = base.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), -15.0, 1.0)
        rotated = cv2.warpAffine(base, matrix, (w, h), borderValue=(255, 255, 255))
        decoder = _run_decoder(
            rotated,
            code_type="QR Code",
            act_as_fixture=True,
            fixture_angle_offset=90.0,
        )
        self.assertGreater(decoder.outputs["count"], 0)
        result = decoder.outputs["results"][0]
        diff = (result.angle - 15.0 + 45.0) % 90.0 - 45.0
        self.assertLessEqual(abs(diff), 4.0)
        expected = (result.angle + 90.0 + 180.0) % 360.0 - 180.0
        self.assertAlmostEqual(decoder.outputs["output image"].angle, expected, places=6)


if __name__ == "__main__":
    unittest.main()
