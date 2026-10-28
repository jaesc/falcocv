import cv2
import numpy as np
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_base import ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.core.vision_image import VisionImage
from falcocv.core.roi_utils import extract_roi
from falcocv.core.pointcloud_utils import propagate_3d


class CodeResult(ToolResult):
    """Result for a single detected/decoded code."""
    def __init__(self, id, x, y, w, h, angle, text, format, quality):
        super().__init__(id=id, x=x, y=y, w=w, h=h, angle=angle,
                         text=text, format=format, quality=quality)


@register_tool(prefix="[BDEC]", default_name="Barcode Decoder", group="Vision Tools")
class BarcodeDecoder(BaseTool):
    """Detect and decode QR codes, 1D barcodes, DataMatrix, and ArUco markers.

    Backends:
      - 1D barcodes              -> Python pipeline (pyzbar + zxing-cpp fallback).
      - QR codes                 -> Python (OpenCV QRCodeDetector + zxing-cpp fallback).
      - DataMatrix               -> Python (pylibdmtx with unsharp preprocessing).
      - ArUco                    -> Python (OpenCV aruco).
    """

    def __init__(self, name="Barcode Decoder"):
        super().__init__(name)
        self.params = {
            "code_type": "Auto",
            "roi": [0, 0, 0, 0, 0],
            "show_overlays": True,
            "show_search_roi": True,
            "show_found_region": True,
            "show_labels": True,
            "show_found_origin": True,
            "act_as_fixture": False,
            "max_count": 1,
            "qr_params": {"eps_x": 0.1, "eps_y": 0.1},
            # Kept for backwards compatibility with saved jobs.  The UI no
            # longer exposes them because the 1D pipeline uses pyzbar+zxing-cpp.
            "barcode_params": {
                "downsampling_threshold": 512,
                "gradient_threshold": 64,
            },
            "dmtx_params": {
                "shrink": 2,
                "timeout_ms": 500,
                "enable_sharpening": True,
            },
            "copy_overlays_to_input": True,
            "aruco_params": {
                "dict_name": "DICT_6X6_250",
                "adaptiveThreshWinSizeMin": 3,
                "adaptiveThreshWinSizeMax": 23,
                "adaptiveThreshWinSizeStep": 10,
                "adaptiveThreshConstant": 7,
                "minMarkerPerimeterRate": 0.03,
                "maxMarkerPerimeterRate": 4.0,
                "polygonalApproxAccuracyRate": 0.03,
                "cornerRefinementMethod": 0,
                "cornerRefinementWinSize": 5,
                "cornerRefinementMaxIterations": 30,
                "cornerRefinementMinAccuracy": 0.1,
                "detectInvertedMarker": False,
            },
            "preprocessing": {
                "strategy": "no_clahe_then_fallback",
                "clahe_clip_limit": 2.0,
                "clahe_tile_pct": 2.0,
            },
        }
        self.inputs = {"input image": None}
        self.outputs = {"output image": None, "count": 0, "results": ToolResults()}

    def run(self, advance=False):
        img = self.inputs.get("input image")
        if img is None:
            self.outputs["output image"] = None
            self.outputs["count"] = 0
            self.outputs["results"] = ToolResults()
            return

        roi = extract_roi(img, self.params["roi"])
        roi_src = roi["roi_src"]
        ax, ay, aw, ah = roi["ax"], roi["ay"], roi["aw"], roi["ah"]

        # Ensure grayscale for 1D and for CLAHE preprocessing.
        gray = roi_src if roi_src.ndim == 2 else cv2.cvtColor(roi_src, cv2.COLOR_BGR2GRAY)

        strategy = self.params.get("preprocessing", {}).get("strategy", "no_clahe_then_fallback")

        raw_results = []
        code_type = self.params["code_type"]

        # Build the image list once according to the preprocessing strategy.
        if strategy == "always_no_clahe":
            images = [gray]
        elif strategy == "always_clahe":
            images = [self._apply_clahe(gray)]
        else:  # no_clahe_then_fallback
            images = [gray, self._apply_clahe(gray)]

        if code_type == "Barcode 1D":
            for img_try in images:
                raw_results = self._decode_1d(img_try)
                if raw_results:
                    break

        elif code_type == "QR Code":
            for img_try in images:
                raw_results = self._decode_qr(img_try)
                if raw_results:
                    break
                raw_results = self._decode_qr_with_zxing(img_try)
                if raw_results:
                    break

        elif code_type == "DataMatrix":
            for img_try in images:
                raw_results = self._decode_dm(img_try)
                if raw_results:
                    break

        elif code_type == "ArUco":
            for img_try in images:
                raw_results = self._decode_aruco(img_try)
                if raw_results:
                    break

        else:  # Auto: fast 1D first, then 2D decoders
            for img_try in images:
                raw_results = self._decode_1d(img_try)
                if raw_results:
                    break
                raw_results = self._decode_qr(img_try)
                if raw_results:
                    break
                raw_results = self._decode_qr_with_zxing(img_try)
                if raw_results:
                    break
                raw_results = self._decode_dm(img_try)
                if raw_results:
                    break
                raw_results = self._decode_aruco(img_try)
                if raw_results:
                    break

        # Deduplicate across all detection paths (C++ and Python).
        raw_results = self._deduplicate_results(raw_results)

        # max_count general (applies to all code types)
        max_count = self.params.get("max_count", 0)
        if max_count > 0 and len(raw_results) > max_count:
            raw_results = raw_results[:max_count]

        # Coordinate mapping: ROI-space → image-space
        abs_angle = roi["abs_angle"]
        M_inv = roi.get("M_inv")

        def _to_image(x, y):
            if M_inv is not None:
                pt = M_inv @ np.array([x, y, 1.0])
                return float(pt[0]), float(pt[1])
            return x + ax, y + ay

        def _pts_to_image(pts):
            if M_inv is not None:
                pts_h = np.column_stack([pts, np.ones(len(pts))])
                return pts_h @ M_inv.T
            return pts + [ax, ay]

        # Build results
        results = ToolResults()
        for i, r in enumerate(raw_results):
            cx_abs, cy_abs = _to_image(r["x"], r["y"])
            res = CodeResult(
                id=i, x=cx_abs, y=cy_abs,
                w=r["w"], h=r["h"], angle=r["angle"],
                text=r["text"], format=r["format"], quality=r["quality"],
            )
            results.append(res)
        self.outputs["count"] = results.Count
        self.outputs["results"] = results

        # Output VisionImage
        out_img = VisionImage(img.data, x=img.x, y=img.y, angle=img.angle)
        out_img.inherit_overlays_from(img)
        propagate_3d(img, out_img, regenerate=False)
        layer = out_img.get_overlay_layer("default")
        layer.source_tool = self.name
        layer.elements = [e for e in layer.elements if e.source_tool != self.name]

        # Search ROI overlay
        if self.params.get("show_search_roi", True) and not roi["is_full_img"]:
            layer.add_roi([float(ax), float(ay), float(aw), float(ah), abs_angle],
                          color=(255, 128, 0), thickness=2)

        for i, r in enumerate(raw_results):
            cx_abs, cy_abs = _to_image(r["x"], r["y"])

            if self.params["show_found_region"] and "pts" in r:
                pts_img = _pts_to_image(np.array(r["pts"], dtype=np.float32))
                layer.add_contour(pts_img.reshape(-1, 2).tolist(),
                                  color=(0, 255, 0), thickness=2)

            if self.params["show_found_origin"]:
                layer.add_point((cx_abs, cy_abs), color=(0, 0, 255), radius=5)

            if self.params["show_labels"]:
                label = f"#{i} {r['format']}: {r['text']}"
                layer.add_text(label, position=(cx_abs + 8, cy_abs - 8),
                               color=(255, 255, 0), font_scale=0.5, thickness=1)

        if self.params.get("act_as_fixture") and results.Count > 0:
            best = results[0]
            out_img.x = best.x
            out_img.y = best.y
            out_img.angle = best.angle

        self.outputs["output image"] = out_img
        self._copy_overlays_to_input(out_img, img)
        self.data_changed.emit()

    def _apply_clahe(self, gray):
        """Apply CLAHE using params from preprocessing config."""
        pre = self.params.get("preprocessing", {})
        clip_limit = pre.get("clahe_clip_limit", 2.0)
        tile_pct = pre.get("clahe_tile_pct", 2.0)
        h, w = gray.shape[:2]
        tile_size = max(4, min(w, h) // max(1, int(100.0 / max(tile_pct, 0.1))))
        try:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            return clahe.apply(gray)
        except Exception:
            return gray

    @staticmethod
    def _result_from_corners(corners, text, fmt, quality=1.0):
        """Build a raw result dict from 4 corner points (same shape as C++)."""
        pts = np.asarray(corners, dtype=np.float32).reshape(-1, 2)
        if len(pts) < 4 or not text:
            return None
        pts = pts[:4]
        cx, cy = pts.mean(axis=0)
        rect = cv2.minAreaRect(pts)
        w, h = rect[1]
        return {
            "x": float(cx), "y": float(cy),
            "w": float(w), "h": float(h),
            "angle": float(rect[2]),
            "text": str(text), "format": fmt, "quality": float(quality),
            "pts": pts.tolist(),
        }

    def _decode_qr(self, gray):
        """Decode QR codes with OpenCV (Aruco detector first, then classic)."""
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        results = []

        # OpenCV 5: ArUco-based QR detector is faster/more robust.
        try:
            detector_cls = getattr(cv2, "QRCodeDetectorAruco", None)
            if detector_cls is not None:
                data, points, _ = detector_cls().detectAndDecode(gray)
                result = self._result_from_corners(points, data, "QR")
                if result is not None:
                    results.append(result)
                    return results
        except Exception:
            pass

        # Classic contour-based detector.
        try:
            detector = cv2.QRCodeDetector()
            qp = self.params.get("qr_params", {})
            if hasattr(detector, "setEpsX"):
                detector.setEpsX(float(qp.get("eps_x", 0.1)))
            if hasattr(detector, "setEpsY"):
                detector.setEpsY(float(qp.get("eps_y", 0.1)))
            data, points, _ = detector.detectAndDecode(gray)
            result = self._result_from_corners(points, data, "QR")
            if result is not None:
                results.append(result)
        except Exception:
            pass

        return results

    def _decode_dm(self, gray):
        """Decode DataMatrix codes with pylibdmtx (same preprocessing as C++)."""
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        try:
            import pylibdmtx.pylibdmtx as dmtx
        except Exception:
            return []

        p = self.params.get("dmtx_params", {})
        work = gray
        if p.get("enable_sharpening", True):
            blurred = cv2.GaussianBlur(work, (0, 0), 3.0)
            work = cv2.addWeighted(work, 1.5, blurred, -0.5, 0)

        max_count = int(self.params.get("max_count", 1) or 1)
        try:
            decoded = dmtx.decode(
                work,
                shrink=int(p.get("shrink", 2)),
                timeout=int(p.get("timeout_ms", 500)),
                max_count=max_count,
            )
        except Exception:
            return []

        img_h = gray.shape[0]
        results = []
        for obj in decoded:
            try:
                rx, ry, rw, rh = obj.rect
                text = obj.data.decode("utf-8", errors="replace")
            except Exception:
                continue
            if not text:
                continue
            # libdmtx uses a bottom-left origin: flip Y for the contour.
            pts = [
                [rx, img_h - ry],
                [rx + rw, img_h - ry],
                [rx + rw, img_h - (ry + rh)],
                [rx, img_h - (ry + rh)],
            ]
            results.append({
                "x": float(rx + rw / 2.0), "y": float(ry + rh / 2.0),
                "w": float(rw), "h": float(rh), "angle": 0.0,
                "text": text, "format": "DataMatrix", "quality": 1.0,
                "pts": pts,
            })
        return results

    def _decode_aruco(self, gray):
        """Decode ArUco markers with OpenCV's Python aruco module."""
        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        p = self.params.get("aruco_params", {})
        dict_name = p.get("dict_name", "DICT_6X6_250")
        try:
            dict_id = getattr(cv2.aruco, dict_name)
            dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        except Exception:
            return []

        det_params = cv2.aruco.DetectorParameters()
        for key, value in p.items():
            if key == "dict_name":
                continue
            if hasattr(det_params, key):
                try:
                    setattr(det_params, key, value)
                except Exception:
                    pass

        try:
            detector = cv2.aruco.ArucoDetector(dictionary, det_params)
            corners, ids, _ = detector.detectMarkers(gray)
        except Exception:
            return []
        if ids is None:
            return []

        results = []
        for marker_id, marker_corners in zip(np.ravel(ids), corners):
            result = self._result_from_corners(
                marker_corners, str(int(marker_id)), f"ArUco:{dict_name}"
            )
            if result is not None:
                results.append(result)
        return results

    def _decode_1d(self, gray):
        """Decode 1D barcodes using pyzbar + zxing-cpp.

        pyzbar handles common codes fast: EAN/UPC/CODE128/CODE39/ITF/
        CODABAR/CODE93/DATABAR. zxing-cpp covers the same set plus DataBar
        variants, rotated codes and difficult lighting.

        Returns a list of result dicts (same shape as C++ compute output).
        """
        import os
        import contextlib
        import pyzbar.pyzbar as zbar
        from pyzbar.pyzbar import ZBarSymbol
        import zxingcpp

        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape[:2]
        all_syms = [v for k, v in vars(ZBarSymbol).items() if not k.startswith("_")]

        # zxing-cpp formats for 1D codes (all common industrial formats).
        zx_1d_fmts = [
            zxingcpp.BarcodeFormat.EAN13,
            zxingcpp.BarcodeFormat.EAN8,
            zxingcpp.BarcodeFormat.UPCA,
            zxingcpp.BarcodeFormat.UPCE,
            zxingcpp.BarcodeFormat.Code128,
            zxingcpp.BarcodeFormat.Code39,
            zxingcpp.BarcodeFormat.Code93,
            zxingcpp.BarcodeFormat.ITF,
            zxingcpp.BarcodeFormat.Codabar,
            zxingcpp.BarcodeFormat.DataBar,
            zxingcpp.BarcodeFormat.DataBarExp,
            zxingcpp.BarcodeFormat.DataBarExpStk,
            zxingcpp.BarcodeFormat.DataBarLimited,
            zxingcpp.BarcodeFormat.DataBarOmni,
            zxingcpp.BarcodeFormat.DataBarStk,
            zxingcpp.BarcodeFormat.DataBarStkOmni,
        ]

        results = []
        seen_texts = set()

        def _add(result):
            key = (result["text"], result["format"])
            if key in seen_texts:
                return
            seen_texts.add(key)
            results.append(result)

        def _pyzbar_try(image, s=1.0):
            try:
                for obj in zbar.decode(image, symbols=all_syms):
                    _add(self._pyzbar_to_result(obj, s))
            except Exception:
                pass

        def _zxing_try(image, s=1.0):
            try:
                for b in zxingcpp.read_barcodes(
                    image,
                    formats=zx_1d_fmts,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                ):
                    _add(self._zxing_to_result(b, s))
            except Exception:
                pass

        target = int(self.params.get("max_count", 1) or 0)
        limit = target if target > 0 else 10**9

        def _enough():
            return len(results) >= limit

        with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
            # 1. Fast path: pyzbar on a downscaled image.
            scale = min(1.0, 1200.0 / max(w, h))
            if scale < 1.0:
                work = cv2.resize(gray, None, fx=scale, fy=scale)
                _pyzbar_try(work, scale)
                if _enough():
                    return results

            # 2. pyzbar on the original image.
            _pyzbar_try(gray)
            if _enough():
                return results

            # 3. pyzbar + Otsu for low-contrast codes.
            try:
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                _pyzbar_try(otsu)
                if _enough():
                    return results
            except Exception:
                pass

            # 4. pyzbar + CLAHE.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            _pyzbar_try(clahe.apply(gray))
            if _enough():
                return results

            # 5. zxing-cpp fallback (DataBar family, rotated/difficult codes).
            #    Try downsample first, then original and CLAHE.
            if scale < 1.0:
                try:
                    _zxing_try(cv2.resize(gray, None, fx=scale, fy=scale), scale)
                except Exception:
                    pass
                if _enough():
                    return results
            _zxing_try(gray)
            if _enough():
                return results
            _zxing_try(clahe.apply(gray))

        return results

    def _decode_qr_with_zxing(self, gray):
        """Fallback QR decoder using zxing-cpp (covers QR, MicroQR, rMQR)."""
        import os
        import contextlib
        import zxingcpp

        if gray.ndim == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)

        qr_fmts = [zxingcpp.BarcodeFormat.QRCode]
        for name in ("MicroQRCode", "RectangularMicroQRCode"):
            fmt = getattr(zxingcpp.BarcodeFormat, name, None)
            if fmt is not None:
                qr_fmts.append(fmt)

        with open(os.devnull, "w") as null, contextlib.redirect_stderr(null):
            try:
                decoded = zxingcpp.read_barcodes(
                    gray,
                    formats=qr_fmts,
                    try_rotate=True,
                    try_downscale=True,
                    try_invert=True,
                )
            except Exception:
                decoded = []

        results = []
        for b in decoded:
            results.append(self._zxing_to_result(b, 1.0))
        return results

    def _deduplicate_results(self, results, iou_threshold=0.5):
        """Deduplicate results by IoU, keeping highest quality."""
        if len(results) <= 1:
            return results

        def _iou(a, b):
            ax1, ay1, ax2, ay2 = a["x"] - a["w"]/2, a["y"] - a["h"]/2, a["x"] + a["w"]/2, a["y"] + a["h"]/2
            bx1, by1, bx2, by2 = b["x"] - b["w"]/2, b["y"] - b["h"]/2, b["x"] + b["w"]/2, b["y"] + b["h"]/2
            inter_x1 = max(ax1, bx1)
            inter_y1 = max(ay1, by1)
            inter_x2 = min(ax2, bx2)
            inter_y2 = min(ay2, by2)
            if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
                return 0.0
            inter = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            area_a = a["w"] * a["h"]
            area_b = b["w"] * b["h"]
            return inter / (area_a + area_b - inter + 1e-6)

        # Sort by quality descending.
        sorted_results = sorted(results, key=lambda r: r.get("quality", 1.0), reverse=True)
        kept = []
        for r in sorted_results:
            if not any(_iou(r, k) > iou_threshold for k in kept):
                kept.append(r)
        return kept

    def _pyzbar_to_result(self, obj, scale):
        r = obj.rect
        cx = (r.left + r.width / 2.0) / scale
        cy = (r.top + r.height / 2.0) / scale
        w_b = r.width / scale
        h_b = r.height / scale

        pts = []
        try:
            if len(obj.polygon) >= 4:
                pts = [[p.x / scale, p.y / scale] for p in obj.polygon]
        except AttributeError:
            pass
        if not pts:
            rx = r.left / scale
            ry = r.top / scale
            pts = [
                [rx, ry], [rx + w_b, ry],
                [rx + w_b, ry + h_b], [rx, ry + h_b],
            ]

        return {
            "x": cx, "y": cy,
            "w": w_b, "h": h_b,
            "angle": 0.0,
            "text": obj.data.decode(errors="replace"),
            "format": obj.type,
            "quality": 1.0,
            "pts": pts,
        }

    def _zxing_to_result(self, b, scale):
        p = b.position
        pts = [
            [p.top_left.x / scale, p.top_left.y / scale],
            [p.top_right.x / scale, p.top_right.y / scale],
            [p.bottom_right.x / scale, p.bottom_right.y / scale],
            [p.bottom_left.x / scale, p.bottom_left.y / scale],
        ]
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        return {
            "x": sum(xs) / 4.0,
            "y": sum(ys) / 4.0,
            "w": max(xs) - min(xs),
            "h": max(ys) - min(ys),
            "angle": float(b.orientation),
            "text": b.text,
            "format": b.format.name,
            "quality": 1.0 if b.valid else 0.0,
            "pts": pts,
        }

    def serialize(self, job_manager=None):
        data = super().serialize(job_manager)
        data["type"] = "BarcodeDecoder"
        return data

    def get_config_widget(self):
        from .config import BarcodeDecoderConfig
        return BarcodeDecoderConfig(self)
