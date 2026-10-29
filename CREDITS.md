# Credits & third-party licenses

FalcoCV is free software under the **GNU General Public License v3.0 or later**
(see [LICENSE](LICENSE)).

## Brand

The FalcoCV falcon mark and wordmark are a custom design provided by the project
author. Sources and derivatives live in [`ui/assets/`](ui/assets/):

- `brand/FalcoCV_logo_source.jpg` — original design (raster)
- `logo.svg`, `logo_light.svg` — vectorized mark
- `logo_lockup_dark.png`, `logo_lockup_light.png` — horizontal lockups
- `icon.ico`, `icon_light.ico`, `favicon.png` — application icons (16–256 px)

During exploration, the following open-source icon sets were evaluated but are
**not** part of the final mark: [Lucide](https://lucide.dev) (ISC),
[Tabler Icons](https://tabler.io/icons) (MIT),
[Bootstrap Icons](https://icons.getbootstrap.com) (MIT).

## Runtime dependencies

All are compatible with GPLv3.

| Component | License |
|---|---|
| OpenCV / `opencv-python` | Apache-2.0 (wheel bundles FFmpeg, LGPL-2.1) |
| ONNX Runtime (`onnxruntime`) | MIT |
| PyQt6, PyQt6-QScintilla | GPL-3.0 or commercial (Riverbank) |
| NumPy | BSD-3-Clause |
| FastAPI | MIT |
| Uvicorn | BSD-3-Clause |
| WebSockets | BSD-3-Clause |
| python-multipart | Apache-2.0 |
| psutil | BSD-3-Clause |
| pybind11 | BSD-3-Clause |
| PyOpenGL | BSD-3-Clause |
| tifffile | BSD-3-Clause |
| imagecodecs | BSD-3-Clause |
| pyzbar | MIT |
| pylibdmtx | MIT |
| zxing-cpp | Apache-2.0 |
| python-Levenshtein | MIT |
| python-barcode | MIT |
| setuptools | MIT |

`harvesters` and `genicam` are installed by default for GigE/USB3 Vision
cameras. `harvesters` is Apache-2.0; the GenICam reference implementation is
distributed under the EMVA GenICam 1.6 license.

## Models (`models/`)

| Model | Origin | License |
|---|---|---|
| `ch_PP-OCRv5_rec_server.onnx`, `ch_PP-OCRv5_rec_mobile.onnx`, `ppocrv5_dict.txt` | PaddleOCR / RapidOCR export | Apache-2.0 |
| `ResNet18-Layer2.onnx`, `ResNet18-Layer3.onnx` | torchvision `ResNet18_Weights.DEFAULT` (IMAGENET1K_V1) | BSD-3-Clause |
| `EfficientNet-B0.onnx`, `EfficientNet-B0-MultiLayer.onnx` | timm `efficientnet_b0` default weights (`efficientnet_b0.ra_in1k`) | Apache-2.0 |

The weights are exported to ONNX by `scripts/export_*.py`; the published model
bundle ships this attribution in its release notes.

## Demo images (`examples/`)

`examples/images/`, `examples/train/part_reference.png` and
`examples/result_*.jpg` are synthetic scenes rendered from scratch by
`examples/generate_demo_images.py` with NumPy/OpenCV. No third-party photos,
textures or models are used. Released under **CC0-1.0**.

## Development-only toolchain

`torch`, `torchvision`, `timm`, `onnx` and `albumentations` are used only by the
export scripts (`scripts/export_*.py`) and optional training flows. They are not
required to run FalcoCV.

---

If you believe an attribution is missing or incorrect, please open an issue.
