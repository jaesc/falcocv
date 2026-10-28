"""Download PP-OCRv5 ONNX models for the Industrial OCR tool.

Downloads detection, recognition, and direction classifier models
from ModelScope (RapidAI) in ONNX format. No PaddlePaddle needed.

Usage:
    python scripts/download_ocr_models.py
"""

import os, sys, urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

_BASE = "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/onnx/PP-OCRv5"

FILES = {
    # Detection
    "ch_PP-OCRv5_det_mobile.onnx":  f"{_BASE}/det/ch_PP-OCRv5_det_mobile.onnx",
    "ch_PP-OCRv5_det_server.onnx":  f"{_BASE}/det/ch_PP-OCRv5_det_server.onnx",
    # Recognition (Chinese + English + Japanese mixed)
    "ch_PP-OCRv5_rec_mobile.onnx":  f"{_BASE}/rec/ch_PP-OCRv5_rec_mobile.onnx",
    "ch_PP-OCRv5_rec_server.onnx":  f"{_BASE}/rec/ch_PP-OCRv5_rec_server.onnx",
    # Direction classifier
    "ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx": f"{_BASE}/cls/ch_PP-LCNet_x0_25_textline_ori_cls_mobile.onnx",
    "ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx":  f"{_BASE}/cls/ch_PP-LCNet_x1_0_textline_ori_cls_server.onnx",
    # Dictionary
    "ppocrv5_dict.txt": "https://www.modelscope.cn/models/RapidAI/RapidOCR/resolve/v3.8.0/paddle/PP-OCRv5/rec/ch_PP-OCRv5_rec_mobile/ppocrv5_dict.txt",
}

def step(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")

def download():
    step("Downloading PP-OCRv5 ONNX models")
    for name, url in FILES.items():
        path = MODELS_DIR / name
        if path.exists():
            print(f"  [OK] Already exists: {name}")
            continue
        print(f"  Downloading {name}...")
        try:
            urllib.request.urlretrieve(url, path)
            size_mb = path.stat().st_size / (1024*1024)
            print(f"  [OK] {name} ({size_mb:.1f} MB)")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")

if __name__ == "__main__":
    download()
    step("Done!")
    print(f"Models saved to: {MODELS_DIR}")
