# FalcoCV — TODO

Status: 1.1.0 released; `dev` carries the post-1.1.0 work. Phases and closed
decisions live in [`plans/falcocv-plan-v2.md`](plans/falcocv-plan-v2.md); this
file tracks concrete, actionable tasks only.

## Before publishing the next release

- [ ] Rebuild the wheel: the `falcocv-1.1.0` wheel predates the geometry-only
  ShapeMatch, the communications work and the async execution service.
- [ ] Run the manual checklist in [`docs/SMOKE_TEST.md`](docs/SMOKE_TEST.md).
- [ ] Publish following [`docs/RELEASING.md`](docs/RELEASING.md). Bump only
  `src/falcocv/version.py`; `pyproject.toml` reads the version dynamically.

## New tools

- [ ] Primitive drawing (lines, circles, rectangles, text over overlay)
- [ ] Color Threshold / Range Extraction (HSV range based)
- [ ] Image File Writer (save results to disk)
- [ ] Calibration Tools (camera, lens correction)
- [x] OCR (PP-OCRv5 ONNX) — implemented in `tools/ocr/`, `core/ocr/`
- [ ] Image Transform (rotate, scale, affine, perspective warp)
- [ ] Pixel Inspector / Line Profile

## Metrology (M1 / M2)

- [x] `LineFit` / `CircleFit` with a ROI caliper array as the default source
  (parallel line calipers / inward radial circle calipers) and robust fitting
- [x] Points mode: overlay points and contour vertices (no `input points`
  port)
- [ ] Calibration and millimetre units for measurement results

## HMI Designer

- [ ] Widget: Stats/History Table — last N results (timestamp + value + pass/fail)
- [ ] Widget: Static Image — logo/image with fit options (fit, stretch, center)

## Communications

- [x] Omron driver validated against real hardware
- [x] ADLINK DASK USB-7250 validated, including `verify_writes`
- [ ] Validate Beckhoff ADS against real hardware
- [ ] Validate Siemens S7 against real hardware
- [x] Modbus TCP / OPC UA clients and servers (loopback tests)

## ShapeMatch

- [ ] Review the two >5 px translation frames (`0867`, `0957`) in the 1005-frame gate
- [ ] Resolve the TemplateMatch scale reference reporting exactly 1.0

## Optimizations / bugs

- [x] Fix 1D Barcode (pyzbar) — pyzbar + zxing-cpp fallback
- [x] Validate QR and ArUco under real conditions (lighting, rotation, perspective)
- [x] OpenCV 5 only (`opencv-python>=5.0.0`); no separate OpenCV C++ build

## Open source readiness

- [x] Public repo with fresh history
- [x] Models distributed as public GitHub Release assets (`models-v1`);
  `cpp/bin/` stays out of git
- [x] Third-party model provenance and licenses documented in `CREDITS.md`
- [x] Customer images and internal audits/reports removed from the public tree
- [x] Manual GUI smoke checklist (`docs/SMOKE_TEST.md`); run it before each release
