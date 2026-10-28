# FalcoCV — TODO

## New tools
- [ ] Primitive drawing (lines, circles, rectangles, text over overlay)
- [ ] Color Threshold / Range Extraction (HSV range based)
- [ ] Image File Writer (save results to disk)
- [ ] Calibration Tools (camera, lens correction)
- [x] OCR (PP-OCRv5 ONNX) — implemented in `tools/ocr/`, `core/ocr/`
- [ ] Image Transform (rotate, scale, affine, perspective warp)
- [ ] Pixel Inspector / Line Profile

## HMI Designer
- [ ] Widget: Stats/History Table — last N results (timestamp + value + pass/fail)
- [ ] Widget: Static Image — logo/image with fit options (fit, stretch, center)

## Optimizations / Bugs
- [x] Fix 1D Barcode (pyzbar) — implemented with pyzbar + zxing-cpp fallback
- [x] Validate QR and ArUco under real conditions (lighting, rotation, perspective)
- [x] Unified OpenCV 5 (C++ 5.0.0 world + `opencv-python>=5.0.0`)

## Open source readiness
- [x] Public repo with fresh history (single "Initial public release" commit)
- [x] Models distributed as public GitHub Release assets (`models-v1`);
  `cpp/bin/` stays out of git
- [x] Third-party model provenance and licenses documented in `CREDITS.md`
- [x] Customer images and internal audits/reports removed from the public tree
- [ ] Manual GUI smoke checklist (`docs/SMOKE_TEST.md`)
- [ ] PyPI publication (`1.0.0rc1` on TestPyPI, then `1.0.0`)
