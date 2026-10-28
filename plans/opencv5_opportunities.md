# FalcoCV — Opportunities with OpenCV 5

> Based on: official OpenCV 5 announcement (June 2026, CVPR Denver), FalcoCV code analysis, comparative benchmark OpenCV 5 DNN vs ONNX Runtime.
>
> **Plan date:** June 2026. Living document.

---

## 1. Summary

OpenCV 5 (released June 4, 2026, pip June 8, 2026) is the most important OpenCV release in years. The new features that directly impact FalcoCV:

| OpenCV 5 new feature | Impact on FalcoCV |
|------------------|--------------------|
| **New DNN engine** (graph, 80%+ ONNX coverage, FlashAttention fusions) | Remove ONNX Runtime (~40 MB). Faster on YOLO, DINOv2, OWLv2. |
| **LightGlue + ALIKED/DISK** (attention-based matching with confidence) | Much more robust Pattern Matcher (rotation, scale, deformation). New confidence scores. |
| **Native LLM/VLM** (Qwen 2.5, Gemma 3, PaliGemma, GPT-2/4) | OCR post-processing with contextual autocorrection. Vision-language queries. |
| **LaMa inpainting** via native DNN | Remove fixtures/artifacts before inspecting. Defect filling. |
| **Native FP16/BF16** in cv::Mat | Classifier/Anomaly memory banks at half size. |
| **New 3d module** (TSDF, HashTSDF, ColorTSDF) + **multi-camera calib** | Prepare for 3D cameras in FalcoCV. |
| **HAL + Universal Intrinsics 2.0** + Arm KleidiCV | When porting tools to C++, automatic acceleration (3-4x on ARM). |
| **Python: named arguments, NumPy 2.x** | More readable stubs and scripts. |

---

## 2. Current situation: ONNX Runtime in FalcoCV

FalcoCV uses **4 modules** with **6 ONNX models**, all loaded through `onnxruntime.InferenceSession`:

| Module | File | Model | Purpose |
|--------|---------|--------|-----------|
| Classifier backbone | `core/classifier/inference.py` | `EfficientNet-B0.onnx` | 1280-d feature vectors |
| Anomaly (quality) | `core/anomaly/patch_backbone.py` | `ResNet18-Layer3.onnx` | Spatial features 256×28×28 |
| Anomaly (balanced) | `core/anomaly/backbone_effnet.py` | `EfficientNet-B0-MultiLayer.onnx` | Spatial features |
| Anomaly (fast) | `core/anomaly/backbone_layer2.py` | `ResNet18-Layer2.onnx` | Spatial features |
| OCR (server) | `core/ocr/industrial_ocr_engine.py` | `ch_PP-OCRv5_rec_server.onnx` | Text recognition |
| OCR (mobile) | `core/ocr/industrial_ocr_engine.py` | `ch_PP-OCRv5_rec_mobile.onnx` | Text recognition (fast) |

**Zero use of `cv2.dnn`** anywhere in the code. There is no `readNet`, `blobFromImage`, `setInput`, `forward` anywhere.

---

## 3. Level 1 — Immediate, high impact

### 3.1 Remove ONNX Runtime → native `cv2.dnn`

**What**: Replace all calls to `onnxruntime.InferenceSession.run()` with `cv.dnn.readNetFromONNX()` + `net.forward()`, using the `ENGINE_NEW` engine.

**Benefit**: -40 MB in the installer, -1 DLL, -1 dependency. Benchmarks: OpenCV 5 DNN is faster than ORT on every model tested (XFeat +31%, YOLOv8n +11%, DINOv2 +24%, OWLv2 +36%).

**How**:
```python
# Before (current):
sess = onnxruntime.InferenceSession("model.onnx")
blob = ...  # manual preprocessing
out = sess.run(None, {sess.get_inputs()[0].name: blob})[0]

# After:
net = cv.dnn.readNetFromONNX("model.onnx", engine=cv.dnn.ENGINE_NEW)
blob = cv.dnn.blobFromImage(img, ...)  # native preprocessing
net.setInput(blob)
out = net.forward()
```

**Files to touch**: `core/classifier/inference.py`, `core/anomaly/*.py`, `core/ocr/industrial_ocr_engine.py`

**Risk**: Low. The default `ENGINE_AUTO` tries `ENGINE_NEW` first and falls back to `ENGINE_CLASSIC` if it fails. If any model has ops the new engine does not support (~20%), the fallback is automatic and error-free.

**Estimate**: 2-3 days (one module per day).

---

### 3.2 New tool: Object Detector

**What**: New tool that loads a YOLOv8/RF-DETR ONNX model and detects objects in a ROI. Output: bounding boxes with class + confidence + centroid, native overlays.

**Why it is missing**: FalcoCV **has no object detection**. It only has pattern matcher (template matching), blob detector (contours), classifier (full image → class). There is no tool that answers "where is each part in the image".

**Industrial use cases**:
- Count parts in a field of view
- Verify complete assembly (screw A present, nut B present)
- Detect multiple defects simultaneously
- Locate regions for sub-analysis (pass each detection to Classifier)

**Tool API**:
```python
class ObjectDetector(BaseTool):
    inputs: {"input image": VisionImage}
    params: {
        "model": "yolov8n.onnx" | "rf-detr.onnx",
        "confidence_threshold": 0.5,
        "nms_threshold": 0.45,
        "classes_of_interest": [...]  # filter by class
    }
    outputs: {
        "output image": VisionImage,  # with detection overlays
        "detections": [{
            "class_id": int, "label": str, "confidence": float,
            "bbox": [x, y, w, h],
            "center": [cx, cy],
            "area": float,
            "pass": bool
        }],
        "count": int,
        "count_by_class": {str: int},
        "pass": bool  # global PASS/FAIL
    }
```

**Relevant benchmarks (OpenCV 5 vs ORT)**:
| Model | OpenCV 5 DNN (ms) | ONNX Runtime (ms) | Difference |
|--------|:-----------------:|:------------------:|:----------:|
| YOLOv8n | 10.9 | 12.15 | **+11.5%** |
| YOLOX-S | 23.46 | 25.16 | **+7.2%** |
| RF-DETR | 102.01 | 106.49 | **+4.4%** |

**Estimate**: 5-7 days (tool wrapper + preproc + NMS + overlays + tests).

---

### 3.3 LightGlue in Pattern Matcher

**What**: Add an attention-based matching phase using `cv::LightGlueMatcher` + `cv::ALIKED` or `cv::DISK` as a step before the current NCC. The matcher returns matches with confidence scores → automatic PASS/FAIL before the pyramid search.

**Why it improves things**: The current NCC is sensitive to:
- Rotation (requires an angle sweep, expensive)
- Scale (requires a pyramid, imprecise between levels)
- Deformations / slight perspective (NCC fails)
- Lighting changes (NCC helps with normalized, but not enough)

LightGlue solves all of this in a single forward: the attention mechanism finds correspondences even with pose, lighting, or deformation changes.

**Proposed architecture**:
```
Input: template image + search image
  │
  ├─ Phase 1: ALIKED keypoints + descriptors (both images)
  ├─ Phase 2: LightGlue matcher → matches with confidence
  ├─ Phase 3: Homography estimation (RANSAC) → transformation
  ├─ Phase 4: Refine with existing NCC (subpixel)
  └─ Output: score, transform, PASS/FAIL
```

**Confidence score**: LightGlue gives per-match confidence. A global score can be derived (inlier ratio, mean confidence, maximum). This allows:
- **PASS** if score > predefined threshold
- **FAIL** if there are not enough quality matches
- **Boundary** if the score is near the threshold → use NCC as fallback

**Estimate**: 5-7 days (integrate LightGlue + 4-phase pipeline + equivalence tests).

---

## 4. Level 2 — Improving existing tools

### 4.1 OCR + LLM Post-Processing

**What**: Add an optional post-processing step with Qwen 2.5 or Gemma 3 (small ONNX model) after OCR. The LLM autocorrects typical errors: "S3R1AL" → "SERIAL", contextual "0" vs "O".

**OpenCV 5 enables**: native tokenizer + KV-cache for autoregressive decoding. Without installing transformers/ctransformers. A single `cv.dnn.readNetFromONNX`.

**How**:
```
OCR ROI → PP-OCRv5 → raw text → Qwen/Gemma → corrected text
                                               ↑
                                  prompt: "Correct OCR errors in industrial text: '{text}'"
```

**Tool config**:
```python
params: {
    "llm_model": "qwen2.5-0.5b.onnx" | "gemma-2b.onnx" | None,
    "llm_prompt": str,  # default: "Correct OCR errors in industrial text: '{text}'"
    "llm_confidence_threshold": 0.7,  # only autocorrect if confidence < threshold
}
```

**Estimate**: 2-3 days (LLM loader + pipeline + configuration).

---

### 4.2 FP16 Memory Banks (Classifier + Anomaly)

**What**: Quantize the k-NN memory banks (Classifier) and coresets (Anomaly) from float32 to float16 using `cv::hfloat` / CV_16F.

**OpenCV 5 enables**: CV_16F is a native cv::Mat type, with direct conversion and supported operations.

**Impact**: Classifier memory banks can hold thousands of 1280-d vectors. FP16 halves the memory (~5 MB → ~2.5 MB per 1000 samples). Anomaly coresets similar.

**Risk**: Practically none — fp16 has a range of ±65,504, EfficientNet embeddings are typically in [-5, 5]. Precision loss in cosine distance is <0.1%.

**Estimate**: 1-2 days (conversion in the memory banks + equivalence tests).

---

### 4.3 Anomaly Alignment with LightGlue

**What**: Replace `cv2.matchTemplate` in `core/anomaly/detector.py:367` (motion compensation) with the LightGlue matcher.

**Why**: The current alignment is translation-only (central patch NCC → shift). With LightGlue, rotation and scale are also compensated, which makes the anomaly detector more robust to small pose variations between train and test.

**Estimate**: 1-2 days.

---

## 5. Level 3 — New capabilities

### 5.1 LaMa Inpainting Tool

**What**: New tool that loads `lama.onnx` (LaMa inpainting model) and removes objects/artifacts from an image given a mask.

**Code** (5 lines):
```python
net = cv.dnn.readNetFromONNX("lama.onnx", engine=cv.dnn.ENGINE_NEW)
blob = cv.dnn.blobFromImages([img, mask], scalefactor=1/255.)
net.setInput(blob)
result = net.forward()  # image without the object
```

**Industrial use cases**:
- Remove known fixtures/conveyors from the image before inspecting
- Erase assembly marks, screws, clips
- Fill small defects to simulate an "ideal part"
- Preprocessing for Anomaly Detector (train with images without fixtures)

**Tool config**:
```python
inputs: {"input image": VisionImage, "mask": VisionImage}
params: {
    "model": "lama.onnx",
    "blend_original": 0.0,  # 0 = inpainted only, 1 = original only
}
outputs: {
    "output image": VisionImage,
}
```

**Estimate**: 2-3 days.

---

### 5.2 Depth from Stereo (Future preparation)

**What**: New tool for multi-camera calibration and depth map generation from stereo pairs.

**OpenCV 5 provides**: `3d`, `calib`, `stereo` modules with TSDF/HashTSDF/ColorTSDF volume fusion, `calibrateMultiview`, `loadMesh`/`saveMesh` (PLY, OBJ), `loadPointCloud`.

**External dependency**: Calibrated stereo camera (RealSense, OAK-D, or stereo rig) — not in FalcoCV's immediate roadmap.

**Status**: Document as a future opportunity. No implementation for now.

---

### 5.3 OWLv2 Open-Vocabulary Detection

**What**: Free-text detection. "find the scratched area", "locate the connector", "count all bolts". Load OWLv2 ONNX with `cv.dnn` → detect objects without training specific models.

**Benchmark**: OWLv2 is 36% faster with `ENGINE_NEW` than with ORT.

**Status**: High GPU dependency (large model, ~600 MB). Evaluate viability on industrial CPU. Possible as an optional tool for systems with a dedicated GPU.

---

## 6. Infrastructure (free)

### 6.1 Universal Intrinsics 2.0 + KleidiCV

When porting tools from Python to C++ (Phase 2 of the C++ migration plan), OpenCV 5 automatically provides:
- **Intel IPP** on x64 (GaussianBlur, resize, warpAffine, color conversion)
- **Arm KleidiCV** on AArch64 with NEON/SVE/SME (AWS Graviton, Cortex-A, Raspberry Pi 5)
- **Qualcomm FastCV** on Snapdragon (Hexagon DSP/NPU)
- **RISC-V Vector** (RVV)
- 3-4x speedup reported on resize, warpAffine, blur on ARM

### 6.2 Non-CPU HAL (Future 5.x)

OpenCV 5's roadmap includes a non-CPU HAL for GPUs/NPUs. When it arrives, imgproc functions (resize, cvtColor, warpAffine, blur, etc.) will be able to run on the same accelerator as DNN inference, eliminating CPU↔GPU copies.

### 6.3 Native cv::Mat FP16/BF16

CV_16F and CV_16BF are first-class citizens in OpenCV 5. FalcoCV currently uses float32 in depth maps, embeddings, and some intermediate data. Migrating to fp16 where it makes sense reduces memory and bandwidth.

### 6.4 Named arguments in Python

OpenCV 5 adds keyword arguments in Python for C++ functions:
```python
# Before:
cv2.GaussianBlur(img, (5,5), 1.5)

# After:
cv.GaussianBlur(img, ksize=(5,5), sigmaX=1.5)
```
Improves readability in user scripts and internal tools. No code changes required — it is additive.

---

## 7. Implementation plan

```
Phase A (1 week):      Remove ONNX Runtime
  └─ Migrate 4 ORT modules → cv.dnn.ENGINE_NEW
  └─ Equivalence tests on each module

Phase B (2 weeks):     Object Detector tool
  └─ YOLOv8n + RF-DETR
  └─ NMS, overlays, count_by_class, PASS/FAIL
  └─ Equivalence test vs ORT baseline

Phase C (2 weeks):     LightGlue in Pattern Matcher
  └─ Phase 1-4 pipeline (ALIKED → LightGlue → RANSAC → NCC refine)
  └─ Confidence score → PASS/FAIL
  └─ Tests with rotation+scale dataset

Phase D (1 week):      OCR + LLM Post-Processing
  └─ Qwen 2.5 0.5B ONNX
  └─ Tool config: prompt + confidence threshold
  └─ Tests with synthetic OCR errors

Phase E (1 week):      FP16 Memory Banks
  └─ Classifier + Anomaly
  └─ Cosine precision equivalence test

Phase F (1 week):      LaMa Inpainting tool
  └─ 5-line tool wrapper + config
  └─ Tests with synthetic masks

Phase G (future):      Depth from Stereo, OWLv2, non-CPU HAL
  └─ Depend on hardware or ecosystem maturity
```

**Estimated total**: 8 weeks for Phases A-F.

---

## 8. Risks and mitigations

| Risk | Impact | Prob. | Mitigation |
|--------|:-------:|:-----:|------------|
| `ENGINE_NEW` does not support some existing ONNX model | Blocking for Phase A | Low | `ENGINE_AUTO` automatically falls back to `ENGINE_CLASSIC`. If it does not work even with classic, keep ORT as a fallback with a `use_ort=True` flag. |
| LightGlue not available in the OpenCV 5 pip wheel | Postpones Phase C | Medium | Check whether `cv2.dnn.LightGlueMatcher` is in the 5.0 pip release or only in a build from source. Alternative: use standalone ALIKED + classic nearest-neighbor matching. |
| LaMa inpainting requires an additional ONNX to download | Extra load in the installer | Low | ~50 MB per lama.onnx. Optional: on-demand download like the existing ONNX models. |
| OWLv2 is too heavy for industrial CPU | Only viable on GPU | High | Keep as an optional, non-core tool. Detect GPU on load, show a warning if absent. |
| vcpkg has no OpenCV 5 port | Postpones the C++ build with 5.x | Medium | **Resolved (July 2026):** C++ with a manual build from source (`cpp/build_opencv5.bat`) and Python with the official `opencv-python>=5.0.0` wheel. |
| PP-OCRv5 model does not load with `readNetFromONNX` | Blocking for Phase A | Low | PP-OCRv5 uses dynamic ops (shape infer). `ENGINE_NEW` has dynamic shape support. If it fails, use `ENGINE_ORT`, which does work. |
| API change in OpenCV 5 (4.x→5.x breaking changes) | Blocking for the migration | Low | FalcoCV's current code uses the classic API (imread, cvtColor, resize, matchTemplate, etc.) which is compatible. OpenCV 5 deprecated the legacy C API, which we do not use. |

---

## Appendix A: Files to touch

### Phase A — Remove ONNX Runtime
| File | Change |
|---------|--------|
| `core/classifier/inference.py` | `onnxruntime.InferenceSession` → `cv.dnn.readNetFromONNX(..., engine=ENGINE_NEW)`. Refactor preproc to `blobFromImage`. |
| `core/anomaly/patch_backbone.py` | Same. Multi-batch support (spatial features). |
| `core/anomaly/backbone_effnet.py` | Same. Multi-layer output. |
| `core/anomaly/backbone_layer2.py` | Same. Layer2 features. |
| `core/ocr/industrial_ocr_engine.py` | Same. CHW→BGR conversion. |
| `requirements.txt` | Remove `onnxruntime`. |
| Packaging (removed) | No longer applies: the repo runs from source. |

### Phase B — Object Detector
| File | Change |
|---------|--------|
| `tools/object_detector/logic.py` | New. Tool wrapper. |
| `tools/object_detector/config.py` | New. PyQt6 QWidget. |
| `tools/__init__.py` | Register tool. |
| `models/` | Add YOLOv8n.onnx + RF-DETR.onnx. |

### Phase C — LightGlue
| File | Change |
|---------|--------|
| `tools/pattern_matcher/logic.py` | Add Phase 1-3 pipeline. Phase 4 (NCC refine) is kept. |
| `tools/pattern_matcher/config.py` | Add "Use LightGlue" checkbox, confidence threshold. |
| `core/pattern_matcher/` | Possible refactor into a separate module. |

### Phase D — OCR + LLM
| File | Change |
|---------|--------|
| `core/ocr/industrial_ocr_engine.py` | Add `LLMPostProcessor` class. |
| `tools/ocr/config.py` | Add "LLM model" combo, prompt template. |

### Phase E — FP16
| File | Change |
|---------|--------|
| `core/classifier/trainer.py` | fp16 memory bank. |
| `core/anomaly/detector.py` | fp16 coreset. |

### Phase F — LaMa
| File | Change |
|---------|--------|
| `tools/inpainting/logic.py` | New. 5 lines + config. |
| `tools/inpainting/config.py` | New. |
| `models/` | Add lama.onnx. |

---

## Appendix B: Official OpenCV 5 DNN vs ONNX Runtime benchmarks

Hardware: Intel Core i9-14900KS, Ubuntu 24.04 LTS.

| Model | OpenCV 5 DNN (ms) | ONNX Runtime (ms) | Difference |
|--------|:-----------------:|:------------------:|:----------:|
| XFeat (feature matching) | 6.56 | 8.61 | **+31.2%** |
| YOLOv8n (detection) | 10.9 | 12.15 | **+11.5%** |
| YOLOX-S (detection) | 23.46 | 25.16 | **+7.2%** |
| DINOv2 small (vision transformer) | 23.78 | 29.58 | **+24.4%** |
| RF-DETR (transformer detection) | 102.01 | 106.49 | **+4.4%** |
| OWLv2 (open-vocabulary detection) | 1,090 | 1,489 | **+36.6%** |
| BiRefNet (segmentation) | 7,178 | 9,503 | **+32.4%** |

Source: https://github.com/opencv/opencv/wiki/OpenCV-5-DNN-Benchmarks (June 2026).
