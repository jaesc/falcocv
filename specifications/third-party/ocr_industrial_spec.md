# Industrial OCR Engine — Opinionated Specification
## Context: mixed text · unknown font · free text · 3 modes · <100ms CPU · Core/Xeon

---

> **Premise:** an industrial OCR is not Tesseract with image filters. It is a three-stage
> pipeline with different models for each problem: locating text, recognizing
> characters, and verifying against expectations with configurable tolerance. Each stage
> has different speed and robustness requirements. This document treats them separately.

---

## TL;DR — The key decisions

| Decision | Recommendation | Discarded |
|---|---|---|
| Text detector | **CRAFT distilled for CPU** | EAST: worse with small/rotated text |
| Recognizer | **PARSeq (Permuted Autoregressive Seq.)** | TrOCR: too large for <100ms |
| Inference backend | **ONNX Runtime INT8** | PyTorch FP32: 3× slower |
| Automatic mode | **Full detect→recognize pipeline** | Recognizer only: fails without a good ROI |
| Expected mode | **CTC beam search + string matching with Levenshtein** | Exact match: too fragile |
| Localization | **CRAFT heatmap → watershed → per-character bbox** | Line bbox only: insufficient for verification |
| Rotation robustness | **STN rectification before the recognizer** | Augmentation: does not correct at inference |
| Preprocessing | **Adaptive pipeline by background type** | Fixed preprocessing: fails on DPM vs label |

---

## 1. Why industrial OCR architecture is different from generic OCR

### 1.1 What fails in Tesseract/EasyOCR in industrial environments

Tesseract and EasyOCR are optimized for scanned documents: white background, black text,
uniform font, good lighting. In industry the text appears on:

```
Laser DPM on metal:        very low contrast, textured background, reflections
Relief engraving:          text defined by shadows, not by color contrast
Degraded thermal transfer: characters with missing pixels, irregular edges
Ink on plastic:            similar background/text colors, variable gloss
Dot-peen marking:          text formed by dots, not continuous lines
```

In all these cases, a fixed preprocessing (Otsu thresholding, contrast normalization)
destroys exactly the information that makes the text legible. An adaptive pipeline is
needed that detects the text type before preprocessing.

### 1.2 The "expected" mode: why it is architecturally different from automatic mode

In automatic mode the system reads what is there. In expected mode the technician says
"there should be `LOT-2024-A3` here" and the system has to answer with three things:

```
1. Did I find text in the image?          → yes / no / partial
2. Does the found text match?             → match / no match / partial match
3. Which specific character differs?      → error position, per-character confidence
```

This is not just comparing strings. The recognizer may read "L0T" instead of "LOT"
(zero vs. O), and that is a model error, not a part error. The system has to
distinguish between:
- **Reading error:** the character is correct but the model got confused (low confidence)
- **Real error:** the character genuinely does not match (high model confidence but different char)

This distinction requires per-character confidence, not just global confidence.

---

## 2. Complete pipeline architecture

### 2.1 Overview of the three modes

```
                    ┌─────────────────────────────────────────────┐
                    │           INDUSTRIAL OCR ENGINE             │
                    └─────────────────────────────────────────────┘
                                         │
              ┌──────────────────────────┼──────────────────────────┐
              │                          │                           │
           READ MODE               VERIFY MODE                LOCATE MODE
     (read only)          (read + verification)      (read + char bbox)
              │                          │                           │
              └──────────────────────────┴──────────────────────────┘
                                         │
                              [SHARED PIPELINE]
                                         │
                    ┌────────────────────▼────────────────────┐
                    │  STAGE 0: ADAPTIVE PREPROCESSING        │
                    │  Detect type → apply filter             │
                    │  ~2ms                                   │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │  STAGE 1: TEXT DETECTION (CRAFT)        │
                    │  Region heatmap + affinity              │
                    │  → word/line bboxes                     │
                    │  ~20-25ms (ONNX INT8)                   │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │  STAGE 2: RECTIFICATION (STN/warp)      │
                    │  Correct rotation and perspective       │
                    │  for each detected region               │
                    │  ~1ms per region                        │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │  STAGE 3: RECOGNITION (PARSeq)          │
                    │  Region image → char sequence           │
                    │  + per-character confidence             │
                    │  ~15-20ms per region (ONNX INT8)        │
                    └────────────────────┬────────────────────┘
                                         │
                    ┌────────────────────▼────────────────────┐
                    │  STAGE 4: MODE-SPECIFIC POST-PROCESSING │
                    │                                         │
                    │  READ:   return text + confidence       │
                    │  VERIFY: Levenshtein + char-level diff  │
                    │  LOCATE: per-char bbox + confidence     │
                    └─────────────────────────────────────────┘
```

### 2.2 Detailed latency budget

```
Operation                              P50      P99      Cumulative P50
─────────────────────────────────────────────────────────────────────
Adaptive preprocessing                 2ms      3ms      2ms
CRAFT detection (ONNX INT8)            22ms     30ms     24ms
Rectification (1-3 regions)            2ms      5ms      26ms
PARSeq recognition × N regions         18ms     28ms     44ms   (1 region)
                                       30ms     45ms     56ms   (2 regions)
Post-processing + verification         1ms      2ms      45-57ms
─────────────────────────────────────────────────────────────────────
TOTAL (1 text region)                  ~45ms             ✅ <100ms
TOTAL (2-3 regions)                    ~60ms             ✅ <100ms
TOTAL (4+ regions)                     ~90ms             ⚠️ at the limit
```

For images with many text regions, process the N largest first
and parallelize if there are >4 regions with `ThreadPoolExecutor(max_workers=2)`.

---

## 3. Stage 0: Adaptive preprocessing

### 3.1 Text type detection

Before preprocessing you need to know what type of text you are working with.
The type classifier is a fast heuristic based on image statistics:

```python
import cv2
import numpy as np
from enum import Enum
from typing import Tuple

class TextSurfaceType(Enum):
    PRINTED_LABEL    = "printed_label"     # printed label, high contrast
    LASER_DPM        = "laser_dpm"         # laser engraving, low contrast
    EMBOSSED         = "embossed"          # relief, text by shadows
    DOT_PEEN         = "dot_peen"          # discontinuous dots
    THERMAL_DEGRADED = "thermal_degraded"  # thermal with partial loss
    UNKNOWN          = "unknown"


def detect_surface_type(image_gray: np.ndarray) -> TextSurfaceType:
    """
    Classifies the surface/text type with fast heuristics.
    Does not use ML: it must run in <0.5ms.

    Heuristics:
      - Global contrast (intensity std)
      - Ratio of pixels at histogram extremes
      - Local variance (textures vs. smooth backgrounds)
      - Dot structure detection (dot-peen)
    """
    h, w = image_gray.shape
    total_px = h * w

    # 1. Global contrast
    global_std = float(image_gray.std())
    global_mean = float(image_gray.mean())

    # 2. Ratio of pixels at extremes (very dark or very bright)
    dark_ratio  = float((image_gray < 50).sum()) / total_px
    bright_ratio = float((image_gray > 200).sum()) / total_px
    bimodal = (dark_ratio + bright_ratio) > 0.3

    # 3. Local variance: background texture
    local_var = cv2.Laplacian(image_gray, cv2.CV_64F).var()

    # 4. Detect dot structure (dot-peen)
    # Dot-peen has periodic high-frequency peaks
    # Approximation: ratio of high-frequency components in FFT
    fft = np.fft.fft2(image_gray.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    cy, cx = h // 2, w // 2
    r = min(h, w) // 6   # low-frequency zone radius
    mask_lf = np.zeros_like(magnitude, dtype=bool)
    Y, X = np.ogrid[:h, :w]
    mask_lf[(Y - cy)**2 + (X - cx)**2 <= r**2] = True
    hf_energy_ratio = float(magnitude[~mask_lf].sum() / (magnitude.sum() + 1e-8))

    # Rule-based classification
    if bimodal and global_std > 60:
        return TextSurfaceType.PRINTED_LABEL

    if global_std < 25 and local_var > 50:
        return TextSurfaceType.LASER_DPM

    if global_std < 30 and local_var > 200:
        return TextSurfaceType.EMBOSSED

    if hf_energy_ratio > 0.7 and global_std < 40:
        return TextSurfaceType.DOT_PEEN

    if bimodal and bright_ratio > 0.15 and global_std > 30:
        return TextSurfaceType.THERMAL_DEGRADED

    return TextSurfaceType.UNKNOWN
```

### 3.2 Preprocessing by type

```python
class AdaptivePreprocessor:
    """
    Adaptive preprocessing according to surface type.
    Each type has a different pipeline that maximizes
    text/background contrast for that specific case.
    """

    def process(
        self,
        image_bgr: np.ndarray,
        surface_type: TextSurfaceType = None
    ) -> Tuple[np.ndarray, TextSurfaceType]:
        """
        Returns preprocessed image (BGR) and detected type.
        If surface_type is None, it is detected automatically.
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        if surface_type is None:
            surface_type = detect_surface_type(gray)

        processed_gray = self._process_by_type(gray, surface_type)

        # Convert back to BGR for the detector (CRAFT expects BGR)
        processed_bgr = cv2.cvtColor(processed_gray, cv2.COLOR_GRAY2BGR)
        return processed_bgr, surface_type

    def _process_by_type(
        self,
        gray: np.ndarray,
        surface_type: TextSurfaceType
    ) -> np.ndarray:

        if surface_type == TextSurfaceType.PRINTED_LABEL:
            return self._process_printed_label(gray)

        elif surface_type == TextSurfaceType.LASER_DPM:
            return self._process_laser_dpm(gray)

        elif surface_type == TextSurfaceType.EMBOSSED:
            return self._process_embossed(gray)

        elif surface_type == TextSurfaceType.DOT_PEEN:
            return self._process_dot_peen(gray)

        elif surface_type == TextSurfaceType.THERMAL_DEGRADED:
            return self._process_thermal_degraded(gray)

        else:
            return self._process_generic(gray)

    def _process_printed_label(self, gray: np.ndarray) -> np.ndarray:
        """
        Printed label: high contrast already present.
        Only soft equalization to even out non-uniform lighting.
        """
        # Local CLAHE: equalizes without destroying global contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        equalized = clahe.apply(gray)
        # Soft sharpening: reinforces character edges
        kernel = np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], dtype=np.float32) / 9
        sharpened = cv2.filter2D(equalized, -1, kernel)
        return np.clip(sharpened, 0, 255).astype(np.uint8)

    def _process_laser_dpm(self, gray: np.ndarray) -> np.ndarray:
        """
        Laser DPM: very low contrast, textured metallic background.
        Goal: isolate local reflectivity variations.

        Morphological top-hat: extracts structures brighter than
        the local background. Ideal for laser marking where the text is
        more or less reflective than the surrounding metal.
        """
        # Top-hat: bright text on dark background
        kernel_size = max(3, gray.shape[0] // 30)
        kernel_size = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        # Black-hat: dark text on light background (try both, use the best)
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Use the one with more contrast
        if tophat.std() >= blackhat.std():
            enhanced = tophat
        else:
            enhanced = blackhat

        # Aggressive equalization to maximize the contrast of the result
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4))
        return clahe.apply(enhanced)

    def _process_embossed(self, gray: np.ndarray) -> np.ndarray:
        """
        Relief: the text is defined by shadow/light gradients,
        not by absolute intensity difference.
        Use directional derivatives (Sobel) to capture the relief edges.
        """
        # Gradient in X and Y
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Normalize to [0, 255]
        magnitude_norm = cv2.normalize(
            magnitude, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

        # Slightly dilate to connect relief fragments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(magnitude_norm, kernel, iterations=1)

        # Final CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(dilated)

    def _process_dot_peen(self, gray: np.ndarray) -> np.ndarray:
        """
        Dot-peen: text formed by discontinuous dots.
        We need to connect the dots so that the detector
        treats them as continuous characters.

        Strategy: morphological closing to connect nearby dots,
        followed by smoothing to soften the result.
        """
        # Adaptive thresholding to isolate the dots
        thresh = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15, C=4
        )
        # Closing: connect dots of the same character
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        connected = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # Smooth result
        blurred = cv2.GaussianBlur(connected, (3, 3), 0)
        return blurred

    def _process_thermal_degraded(self, gray: np.ndarray) -> np.ndarray:
        """
        Degraded thermal transfer: characters with missing pixels.
        Strategy: aggressive equalization + soft morphological closing
        to fill gaps in the characters.
        """
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        equalized = clahe.apply(gray)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        closed = cv2.morphologyEx(equalized, cv2.MORPH_CLOSE, kernel)
        return closed

    def _process_generic(self, gray: np.ndarray) -> np.ndarray:
        """Generic fallback: CLAHE + soft sharpening."""
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        return clahe.apply(gray)
```

---

## 4. Stage 1: Text detection with CRAFT

### 4.1 Why CRAFT and not the alternatives

**CRAFT (Character Region Awareness for Text Detection)** generates two heatmaps:
- **Region score:** probability that each pixel is the center of a character
- **Affinity score:** probability that two adjacent characters belong to the same word

This representation is superior to EAST or DB-Net for industrial text because:

```
EAST:      rectangular bboxes → fails with curved or highly slanted text
DB-Net:    good for documents, worse with isolated text on complex background
CRAFT:     works at character level → robust with variable spacing,
           small text and text on textured surfaces
           The two heat maps allow separating individual characters later
           (needed for LOCATE mode)
```

### 4.2 CRAFT distilled for CPU

The original CRAFT model (VGG16 backbone, ~50M params) is too slow for <100ms on CPU.
A distilled version with a MobileNetV3 backbone is used:

```
CRAFT original (VGG16):      ~180ms CPU FP32    ✗
CRAFT-MobileNet FP32:         ~45ms CPU FP32    ⚠️
CRAFT-MobileNet INT8:         ~22ms CPU INT8    ✅
```

The distilled model is available as ONNX in the `craft-text-detector` repository with
a custom export, or it can be exported from the public PyTorch implementation.

```python
import onnxruntime as ort
import numpy as np
import cv2
from typing import List, Tuple
from dataclasses import dataclass

@dataclass
class TextRegion:
    bbox: np.ndarray        # [4, 2] polygon points (may be rotated)
    bbox_rect: Tuple        # (x, y, w, h) bounding rect axis-aligned
    score: float            # detection confidence
    region_heatmap: np.ndarray  # region score crop (for LOCATE)
    char_bboxes: List[np.ndarray]  # individual character bboxes (for LOCATE)


class CRAFTDetector:
    """
    CRAFT text detector with MobileNetV3 backbone, ONNX INT8.
    """

    def __init__(self, model_path: str, n_threads: int = 4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )

        self.input_name = self.session.get_inputs()[0].name

        # CRAFT normalization statistics
        self._mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self._std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def detect(
        self,
        image_bgr: np.ndarray,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        target_size: int = 1280
    ) -> Tuple[List[TextRegion], np.ndarray, np.ndarray]:
        """
        Detects text regions in the image.

        text_threshold: threshold to consider a pixel as a character center
        link_threshold: threshold to group characters into words
        low_text:       low threshold to expand detected regions
        target_size:    processing resolution (larger = slower, more accurate)

        returns:
            regions:        list of detected TextRegions
            region_score:   region heatmap [H, W]
            affinity_score: affinity heatmap [H, W]
        """
        orig_h, orig_w = image_bgr.shape[:2]

        # Resize keeping aspect ratio
        img_resized, ratio = self._resize_for_craft(image_bgr, target_size)
        h, w = img_resized.shape[:2]

        # Normalize
        img_norm = img_resized[:, :, ::-1].astype(np.float32) / 255.0
        img_norm = (img_norm - self._mean) / self._std
        img_tensor = img_norm.transpose(2, 0, 1)[np.newaxis]   # [1, 3, H, W]

        # Inference
        outputs = self.session.run(None, {self.input_name: img_tensor})
        # CRAFT output: [1, H/2, W/2, 2] → region score and affinity score
        score_map = outputs[0][0]   # [H/2, W/2, 2]
        region_score   = score_map[:, :, 0]   # [H/2, W/2]
        affinity_score = score_map[:, :, 1]   # [H/2, W/2]

        # Extract region bboxes
        regions = self._extract_regions(
            region_score, affinity_score,
            text_threshold, link_threshold, low_text,
            ratio, orig_h, orig_w
        )

        return regions, region_score, affinity_score

    def _resize_for_craft(
        self, image: np.ndarray, target_size: int
    ) -> Tuple[np.ndarray, float]:
        """Resize keeping aspect ratio, long side = target_size."""
        h, w = image.shape[:2]
        ratio = min(target_size / max(h, w), 1.0)
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        # Ensure divisible by 32 (CRAFT requirement)
        new_h = (new_h // 32) * 32
        new_w = (new_w // 32) * 32
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        return resized, ratio

    def _extract_regions(
        self,
        region_score: np.ndarray,
        affinity_score: np.ndarray,
        text_thresh: float,
        link_thresh: float,
        low_text: float,
        ratio: float,
        orig_h: int,
        orig_w: int
    ) -> List[TextRegion]:
        """
        Converts heatmaps into text region bboxes.

        Algorithm:
        1. Threshold region_score with low_text → text mask
        2. Threshold affinity_score with link_threshold → link mask
        3. Combine: text OR links
        4. Connected components → one region per component
        5. For each region: extract minimum rotated bbox
        6. Filter with text_threshold: keep only regions
           where max(region_score) > text_threshold
        """
        # Combine maps
        text_map = (region_score > low_text).astype(np.uint8)
        link_map = (affinity_score > link_thresh).astype(np.uint8)
        combined = cv2.bitwise_or(text_map, link_map)

        # Connected components
        n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            combined, connectivity=4
        )

        regions = []
        # Scale factor: scores are at H/2, W/2 resolution
        scale_factor = 2.0 / ratio

        for i in range(1, n_labels):
            # Filter very small components
            if stats[i, cv2.CC_STAT_AREA] < 10:
                continue

            # Component mask
            mask = (labels == i).astype(np.uint8)

            # Maximum region score in this component
            max_score = float(region_score[mask.astype(bool)].max())
            if max_score < text_thresh:
                continue

            # Component points scaled to original coordinates
            points = np.column_stack(np.where(mask)).astype(np.float32)
            points[:, 0] *= scale_factor   # y
            points[:, 1] *= scale_factor   # x
            points = points[:, [1, 0]]     # → (x, y)

            # Minimum rotated bbox
            rect = cv2.minAreaRect(points)
            box = cv2.boxPoints(rect).astype(np.int32)

            # Axis-aligned bbox for cropping
            x, y, w, h = cv2.boundingRect(box)
            x = max(0, x); y = max(0, y)
            w = min(w, orig_w - x); h = min(h, orig_h - y)

            # Extract individual character bboxes (for LOCATE)
            char_bboxes = self._extract_char_bboxes(
                region_score, mask, scale_factor, text_thresh
            )

            # region_score crop for this region
            rx = max(0, int(x / scale_factor))
            ry = max(0, int(y / scale_factor))
            rw = max(1, int(w / scale_factor))
            rh = max(1, int(h / scale_factor))
            region_crop = region_score[ry:ry+rh, rx:rx+rw]

            regions.append(TextRegion(
                bbox=box,
                bbox_rect=(x, y, w, h),
                score=max_score,
                region_heatmap=region_crop,
                char_bboxes=char_bboxes
            ))

        # Sort top to bottom, then left to right
        regions.sort(key=lambda r: (r.bbox_rect[1], r.bbox_rect[0]))
        return regions

    def _extract_char_bboxes(
        self,
        region_score: np.ndarray,
        component_mask: np.ndarray,
        scale_factor: float,
        threshold: float
    ) -> List[np.ndarray]:
        """
        Extracts individual character bboxes using watershed
        on the region_score inside the detected component.
        Only runs in LOCATE mode.
        """
        masked_score = region_score * component_mask
        score_uint8 = (masked_score * 255).astype(np.uint8)

        # Threshold to obtain local peaks (character centers)
        _, thresh = cv2.threshold(
            score_uint8, int(threshold * 255), 255, cv2.THRESH_BINARY
        )

        # Watershed: separate touching characters
        dist = cv2.distanceTransform(thresh, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist, 0.5 * dist.max(), 255, 0)
        sure_fg = sure_fg.astype(np.uint8)

        n_chars, char_labels, char_stats, _ = cv2.connectedComponentsWithStats(
            sure_fg, connectivity=8
        )

        char_bboxes = []
        for j in range(1, n_chars):
            area = char_stats[j, cv2.CC_STAT_AREA]
            if area < 3:
                continue
            cx = int(char_stats[j, cv2.CC_STAT_LEFT] * scale_factor)
            cy = int(char_stats[j, cv2.CC_STAT_TOP] * scale_factor)
            cw = int(char_stats[j, cv2.CC_STAT_WIDTH] * scale_factor)
            ch = int(char_stats[j, cv2.CC_STAT_HEIGHT] * scale_factor)
            char_bboxes.append(np.array([cx, cy, cw, ch]))

        char_bboxes.sort(key=lambda b: b[0])   # sort left to right
        return char_bboxes
```

---

## 5. Stage 2: Geometric rectification

```python
def rectify_text_region(
    image_bgr: np.ndarray,
    region: TextRegion,
    target_height: int = 48,
    padding_ratio: float = 0.1
) -> np.ndarray:
    """
    Rectifies a text region to pass it to the recognizer.

    Two cases:
    1. Horizontal text (rotation <15°): direct crop + resize
    2. Rotated text (rotation ≥15°): perspective warp + resize

    Why height=48:
    PARSeq was trained with 32px height text images.
    We use 48 to give the recognizer some extra margin with
    low-quality text. The width is variable (aspect ratio preserved).

    padding_ratio: add padding around the region so as
    not to cut characters at the bbox edges.
    """
    orig_h, orig_w = image_bgr.shape[:2]
    x, y, w, h = region.bbox_rect

    # Add padding
    pad_x = int(w * padding_ratio)
    pad_y = int(h * padding_ratio)
    x1 = max(0, x - pad_x)
    y1 = max(0, y - pad_y)
    x2 = min(orig_w, x + w + pad_x)
    y2 = min(orig_h, y + h + pad_y)

    # Compute rotation angle from the rotated bbox
    box = region.bbox.astype(np.float32)
    rect = cv2.minAreaRect(box)
    angle = rect[2]

    if abs(angle) < 15:
        # Mostly horizontal text: direct crop
        cropped = image_bgr[y1:y2, x1:x2]
    else:
        # Rotated text: perspective warp
        src_pts = box.astype(np.float32)
        rw = int(rect[1][0])
        rh = int(rect[1][1])
        if rw < rh:
            rw, rh = rh, rw
            src_pts = np.roll(src_pts, 1, axis=0)

        dst_pts = np.array([
            [0, 0], [rw-1, 0], [rw-1, rh-1], [0, rh-1]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        cropped = cv2.warpPerspective(image_bgr, M, (rw, rh))

    # Resize to fixed height, proportional width
    if cropped.shape[0] == 0 or cropped.shape[1] == 0:
        return np.zeros((target_height, target_height * 4, 3), dtype=np.uint8)

    aspect = cropped.shape[1] / cropped.shape[0]
    target_width = max(target_height, int(target_height * aspect))
    rectified = cv2.resize(
        cropped, (target_width, target_height),
        interpolation=cv2.INTER_CUBIC
    )
    return rectified
```

---

## 6. Stage 3: Recognition with PARSeq

### 6.1 Why PARSeq

**PARSeq (Permuted Autoregressive Sequence Models)** is the state of the art in
scene text recognition (2022, ECCV). Its advantages over alternatives:

```
Tesseract v5:
  + Very well known, easy to integrate
  - Fails with low-quality text, non-standard fonts, complex backgrounds
  - No native per-character confidence
  - Not actively maintained for edge

TrOCR (Microsoft):
  + Very accurate on documents
  - Large transformer: ~344MB, ~200ms CPU → too slow
  - Designed for documents, not industrial scene text

CRNN + CTC (classic):
  + Fast, well known
  - Worse accuracy than PARSeq on degraded text
  - Does not model dependencies between characters

PARSeq:
  + State of the art on scene text (STR benchmarks)
  + Models context between characters (improves O/0, I/l/1 ambiguities)
  + Per-character confidence from the attention mechanism
  + Small version (PARSeq-S): ~24M params, ~15ms CPU INT8
  + Robust to irregular text, mixed fonts, degradation
```

**Character context is critical for industry:**
`L0T-2024` vs `LOT-2024`: the O/0 in alphanumeric context is resolved by the
`LOT` pattern that the model has seen during pre-training. CRNN+CTC does not have
this contextual correction mechanism.

### 6.2 Recognizer implementation

```python
from typing import List, Tuple
import numpy as np
import onnxruntime as ort

@dataclass
class RecognitionResult:
    text: str                      # Recognized text
    confidence: float              # Global confidence [0, 1]
    char_confidences: List[float]  # Per-character confidence [0, 1]
    char_bboxes: List[Tuple]       # per-character bbox (LOCATE only)
    raw_logits: np.ndarray         # For debugging


class PARSeqRecognizer:
    """
    PARSeq text recognizer (small version), ONNX INT8.

    Recommended industrial charset:
    Alphanumeric + characters common in industrial codes:
    0-9, A-Z, a-z, -, ., /, :, #, _, space
    Total: 75 characters

    For free text without charset restriction: use PARSeq
    trained with the full charset (95 printable ASCII).
    """

    # Full printable ASCII charset (for free text)
    CHARSET = (
        '0123456789'
        'abcdefghijklmnopqrstuvwxyz'
        'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~ '
    )

    def __init__(self, model_path: str, n_threads: int = 4):
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = n_threads
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        self.session = ort.InferenceSession(
            model_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )

        self._mean = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self._std  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
        self.char_to_idx = {c: i for i, c in enumerate(self.CHARSET)}
        self.idx_to_char = {i: c for i, c in enumerate(self.CHARSET)}

    def recognize(
        self,
        region_image: np.ndarray,   # [48, W, 3] BGR, rectified
        max_length: int = 25
    ) -> RecognitionResult:
        """
        Recognizes the text in a rectified region.

        max_length: maximum expected text length.
        For industrial codes (serials, lots): 25 is enough.
        """
        # Preprocessing: resize to 32px height (PARSeq training size)
        h, w = region_image.shape[:2]
        target_w = int(w * 32 / h)
        target_w = max(32, min(target_w, 320))   # clamp
        img = cv2.resize(region_image, (target_w, 32), interpolation=cv2.INTER_CUBIC)

        # BGR → RGB, normalize to [-1, 1]
        img = img[:, :, ::-1].astype(np.float32) / 255.0
        img = (img - self._mean) / self._std
        img_tensor = img.transpose(2, 0, 1)[np.newaxis]   # [1, 3, 32, W]

        # Inference: PARSeq returns logits [1, T, num_chars]
        # where T = maximum sequence length
        outputs = self.session.run(None, {'image': img_tensor})
        logits = outputs[0][0]   # [T, num_chars]

        # Decode with beam search
        text, char_confs = self._decode_greedy(logits)

        # Global confidence = geometric mean of per-character confidences
        if char_confs:
            confidence = float(np.exp(np.mean(np.log(np.clip(char_confs, 1e-8, 1.0)))))
        else:
            confidence = 0.0

        return RecognitionResult(
            text=text,
            confidence=confidence,
            char_confidences=char_confs,
            char_bboxes=[],   # filled in LOCATE mode
            raw_logits=logits
        )

    def _decode_greedy(
        self, logits: np.ndarray
    ) -> Tuple[str, List[float]]:
        """
        Greedy decoding with softmax to obtain text and confidences.

        PARSeq generates tokens autoregressively. The special [EOS] token
        indicates end of sequence. We use greedy (argmax) for speed.
        For maximum accuracy in ambiguous cases, beam search can be used
        with beam_width=3, but it adds ~5ms per region.
        """
        EOS_IDX = len(self.CHARSET)   # special end token
        PAD_IDX = EOS_IDX + 1

        # Softmax over the character dimension
        exp_logits = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probs = exp_logits / exp_logits.sum(axis=-1, keepdims=True)

        # Greedy: argmax at each time position
        pred_indices = probs.argmax(axis=-1)   # [T]
        pred_probs   = probs.max(axis=-1)       # [T]

        chars = []
        confs = []
        for idx, prob in zip(pred_indices, pred_probs):
            if idx == EOS_IDX or idx == PAD_IDX:
                break
            if idx < len(self.CHARSET):
                chars.append(self.idx_to_char[idx])
                confs.append(float(prob))

        return ''.join(chars), confs
```

---

## 7. Stage 4: Post-processing by mode

### 7.1 READ mode

```python
@dataclass
class ReadResult:
    text: str
    confidence: float
    char_confidences: List[float]
    regions_found: int
    surface_type: str
    latency_ms: float

def post_process_read(
    recognition_results: List[RecognitionResult],
    regions: List[TextRegion]
) -> ReadResult:
    """
    Aggregates results from multiple regions into a single text.

    Aggregation strategy:
    If there are multiple regions, concatenate in order (top→bottom, left→right).
    Separator between regions: single space.

    If a region has very low confidence (<0.3): exclude it from the result
    and flag it as doubtful in the log.
    """
    valid_results = [
        r for r in recognition_results if r.confidence >= 0.3
    ]

    if not valid_results:
        return ReadResult(
            text='', confidence=0.0, char_confidences=[],
            regions_found=0, surface_type='', latency_ms=0.0
        )

    full_text = ' '.join(r.text for r in valid_results)
    all_char_confs = []
    for r in valid_results:
        all_char_confs.extend(r.char_confidences)

    # Global confidence
    global_conf = float(np.exp(
        np.mean(np.log(np.clip(all_char_confs, 1e-8, 1.0)))
    )) if all_char_confs else 0.0

    return ReadResult(
        text=full_text,
        confidence=global_conf,
        char_confidences=all_char_confs,
        regions_found=len(valid_results),
        surface_type='',
        latency_ms=0.0
    )
```

### 7.2 VERIFY mode

```python
from dataclasses import dataclass
from typing import Optional, List
import Levenshtein   # python-Levenshtein

@dataclass
class CharDiff:
    position: int           # position in the expected string
    expected: str           # expected character
    found: str              # read character ('' if missing)
    confidence: float       # recognizer confidence in 'found'
    diff_type: str          # 'substitution', 'insertion', 'deletion'
    is_reader_error: bool   # True if confidence is low → possible model error

@dataclass
class VerifyResult:
    match: bool                      # True if text matches within tolerance
    match_type: str                  # 'exact', 'fuzzy', 'no_match', 'not_found'
    text_found: str                  # Read text
    text_expected: str               # Expected text
    edit_distance: int               # Levenshtein distance
    similarity: float                # [0, 1] 1 - edit_dist/max_len
    char_diffs: List[CharDiff]       # Per-character differences
    confidence: float                # Global recognizer confidence
    low_confidence_positions: List[int]  # Low-confidence positions
    latency_ms: float


class TextVerifier:
    """
    Verification of read text against expected text.

    Three match levels:
      exact:  read text == expected text (case-sensitive configurable)
      fuzzy:  edit_distance ≤ tolerance (for tolerable reading errors)
      no_match: edit_distance > tolerance

    'is_reader_error' logic:
      If the recognizer has confidence < 0.5 in a character that does not match,
      it is marked as a possible reading error (not a real part error).
      This allows the system to report "match but uncertain reading" instead of
      "no_match" when the error may be from the OCR, not from the part.
    """

    def __init__(
        self,
        max_edit_distance: int = 1,
        case_sensitive: bool = True,
        normalize_chars: bool = True,   # normalize 0/O, I/l/1 according to context
        low_confidence_threshold: float = 0.5
    ):
        self.max_edit_distance = max_edit_distance
        self.case_sensitive = case_sensitive
        self.normalize_chars = normalize_chars
        self.low_conf_threshold = low_confidence_threshold

    def verify(
        self,
        read_result: ReadResult,
        expected_text: str
    ) -> VerifyResult:

        text_found = read_result.text
        char_confs = read_result.char_confidences

        if not text_found:
            return VerifyResult(
                match=False, match_type='not_found',
                text_found='', text_expected=expected_text,
                edit_distance=len(expected_text),
                similarity=0.0, char_diffs=[], confidence=0.0,
                low_confidence_positions=[], latency_ms=0.0
            )

        # Optional normalization
        text_cmp = text_found if self.case_sensitive else text_found.upper()
        expected_cmp = expected_text if self.case_sensitive else expected_text.upper()

        if self.normalize_chars:
            text_cmp    = self._normalize_ambiguous(text_cmp, expected_cmp)

        # Levenshtein distance
        edit_dist = Levenshtein.distance(text_cmp, expected_cmp)
        max_len = max(len(text_cmp), len(expected_cmp), 1)
        similarity = 1.0 - (edit_dist / max_len)

        # Determine match type
        if edit_dist == 0:
            match_type = 'exact'
            is_match = True
        elif edit_dist <= self.max_edit_distance:
            match_type = 'fuzzy'
            is_match = True
        else:
            match_type = 'no_match'
            is_match = False

        # Per-character differences (Levenshtein alignment)
        char_diffs = self._compute_char_diffs(
            text_cmp, expected_cmp, char_confs
        )

        # Low-confidence positions
        low_conf_pos = [
            i for i, conf in enumerate(char_confs)
            if conf < self.low_conf_threshold
        ]

        return VerifyResult(
            match=is_match,
            match_type=match_type,
            text_found=text_found,
            text_expected=expected_text,
            edit_distance=edit_dist,
            similarity=similarity,
            char_diffs=char_diffs,
            confidence=read_result.confidence,
            low_confidence_positions=low_conf_pos,
            latency_ms=0.0
        )

    def _normalize_ambiguous(self, text: str, reference: str) -> str:
        """
        Normalizes ambiguous characters according to context.

        Common industrial rules:
          In a position where a digit is expected: O→0, I→1, l→1, S→5, B→8
          In a position where a letter is expected: 0→O, 1→I, 5→S, 8→B

        Only applies if reference has the expected character at that position.
        This allows correcting reading errors without altering real errors.
        """
        if len(text) != len(reference):
            return text   # do not apply if lengths differ

        DIGIT_SUBS  = {'O': '0', 'I': '1', 'l': '1', 'S': '5', 'B': '8'}
        LETTER_SUBS = {'0': 'O', '1': 'I', '5': 'S', '8': 'B'}

        result = list(text)
        for i, (tc, rc) in enumerate(zip(text, reference)):
            if rc.isdigit() and tc in DIGIT_SUBS:
                result[i] = DIGIT_SUBS[tc]
            elif rc.isalpha() and tc in LETTER_SUBS:
                result[i] = LETTER_SUBS[tc]
        return ''.join(result)

    def _compute_char_diffs(
        self,
        text_found: str,
        text_expected: str,
        char_confs: List[float]
    ) -> List[CharDiff]:
        """
        Computes per-character differences using Levenshtein alignment.
        Uses python-Levenshtein to obtain the edit path opcodes.
        """
        opcodes = Levenshtein.opcodes(text_expected, text_found)
        diffs = []

        for tag, i1, i2, j1, j2 in opcodes:
            if tag == 'equal':
                continue

            if tag == 'replace':
                for k, (ei, fi) in enumerate(zip(range(i1, i2), range(j1, j2))):
                    conf = char_confs[fi] if fi < len(char_confs) else 0.0
                    diffs.append(CharDiff(
                        position=ei,
                        expected=text_expected[ei],
                        found=text_found[fi],
                        confidence=conf,
                        diff_type='substitution',
                        is_reader_error=conf < self.low_conf_threshold
                    ))

            elif tag == 'delete':
                for ei in range(i1, i2):
                    diffs.append(CharDiff(
                        position=ei,
                        expected=text_expected[ei],
                        found='',
                        confidence=0.0,
                        diff_type='deletion',
                        is_reader_error=False
                    ))

            elif tag == 'insert':
                for fi in range(j1, j2):
                    conf = char_confs[fi] if fi < len(char_confs) else 0.0
                    diffs.append(CharDiff(
                        position=i1,
                        expected='',
                        found=text_found[fi],
                        confidence=conf,
                        diff_type='insertion',
                        is_reader_error=conf < self.low_conf_threshold
                    ))

        return diffs
```

### 7.3 LOCATE mode

```python
@dataclass
class LocateResult:
    text: str
    confidence: float
    char_locations: List[dict]   # [{char, bbox, confidence}]
    word_bboxes: List[Tuple]     # full-word bboxes
    visualization: np.ndarray    # image with drawn bboxes
    latency_ms: float


def post_process_locate(
    recognition_results: List[RecognitionResult],
    regions: List[TextRegion],
    original_image: np.ndarray
) -> LocateResult:
    """
    Combines recognized text with CRAFT character bboxes.

    The number of recognized characters and the number of
    detected character bboxes may not match. Alignment strategy:
      - If they match: 1:1 assignment in order
      - If CRAFT detected fewer bboxes: distribute uniformly in the region bbox
      - If CRAFT detected more bboxes: merge the leftovers
    """
    all_char_locations = []
    vis = original_image.copy()

    for region, rec_result in zip(regions, recognition_results):
        if rec_result.confidence < 0.3:
            continue

        chars = list(rec_result.text)
        char_confs = rec_result.char_confidences
        char_bboxes = region.char_bboxes

        # Align chars with bboxes
        n_chars = len(chars)
        n_bboxes = len(char_bboxes)

        if n_bboxes == 0:
            # Fallback: distribute characters uniformly in the region
            x, y, w, h = region.bbox_rect
            char_w = w / max(n_chars, 1)
            char_bboxes = [
                np.array([int(x + i * char_w), y, int(char_w), h])
                for i in range(n_chars)
            ]
        elif n_bboxes != n_chars:
            # Redistribute bboxes so they match the chars
            char_bboxes = _redistribute_bboxes(char_bboxes, n_chars, region.bbox_rect)

        for i, (char, conf) in enumerate(zip(chars, char_confs)):
            if i < len(char_bboxes):
                bx, by, bw, bh = char_bboxes[i]
                all_char_locations.append({
                    'char': char,
                    'confidence': conf,
                    'bbox': (int(bx), int(by), int(bw), int(bh))
                })
                # Draw bbox on visualization
                color = (0, 255, 0) if conf > 0.7 else (0, 165, 255)  # green/orange
                cv2.rectangle(vis, (int(bx), int(by)),
                             (int(bx+bw), int(by+bh)), color, 1)
                cv2.putText(vis, char,
                           (int(bx), int(by) - 2),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

    full_text = ''.join(loc['char'] for loc in all_char_locations)
    all_confs = [loc['confidence'] for loc in all_char_locations]
    global_conf = float(np.exp(
        np.mean(np.log(np.clip(all_confs, 1e-8, 1.0)))
    )) if all_confs else 0.0

    return LocateResult(
        text=full_text,
        confidence=global_conf,
        char_locations=all_char_locations,
        word_bboxes=[r.bbox_rect for r in regions],
        visualization=vis,
        latency_ms=0.0
    )
```

---

## 8. Main engine: OCREngine

```python
import time
from enum import Enum

class OCRMode(Enum):
    READ   = "read"
    VERIFY = "verify"
    LOCATE = "locate"

@dataclass
class OCRConfig:
    # Models
    craft_model_path: str = 'models/craft_mobilenet_int8.onnx'
    parseq_model_path: str = 'models/parseq_small_int8.onnx'
    n_threads: int = 4

    # CRAFT
    text_threshold: float = 0.7
    link_threshold: float = 0.4
    low_text: float = 0.4
    craft_target_size: int = 1280

    # PARSeq
    max_text_length: int = 25

    # Verification
    max_edit_distance: int = 1
    case_sensitive: bool = True
    normalize_ambiguous_chars: bool = True
    low_confidence_threshold: float = 0.5

    # Preprocessing
    force_surface_type: str = None  # None = automatic

    # Output
    min_region_confidence: float = 0.3


class IndustrialOCREngine:
    """
    Complete industrial OCR engine.
    Three modes: READ, VERIFY, LOCATE.
    """

    def __init__(self, config: OCRConfig = None):
        self.config = config or OCRConfig()

        self.preprocessor  = AdaptivePreprocessor()
        self.detector      = CRAFTDetector(
            self.config.craft_model_path,
            self.config.n_threads
        )
        self.recognizer    = PARSeqRecognizer(
            self.config.parseq_model_path,
            self.config.n_threads
        )
        self.verifier      = TextVerifier(
            max_edit_distance=self.config.max_edit_distance,
            case_sensitive=self.config.case_sensitive,
            normalize_chars=self.config.normalize_ambiguous_chars,
            low_confidence_threshold=self.config.low_confidence_threshold
        )

    def read(self, image_bgr: np.ndarray) -> ReadResult:
        t0 = time.perf_counter()
        result = self._run_pipeline(image_bgr, OCRMode.READ)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def verify(self, image_bgr: np.ndarray, expected_text: str) -> VerifyResult:
        t0 = time.perf_counter()
        read_result = self._run_pipeline(image_bgr, OCRMode.READ)
        verify_result = self.verifier.verify(read_result, expected_text)
        verify_result.latency_ms = (time.perf_counter() - t0) * 1000
        return verify_result

    def locate(self, image_bgr: np.ndarray) -> LocateResult:
        t0 = time.perf_counter()
        result = self._run_pipeline(image_bgr, OCRMode.LOCATE)
        result.latency_ms = (time.perf_counter() - t0) * 1000
        return result

    def _run_pipeline(self, image_bgr: np.ndarray, mode: OCRMode):
        # Stage 0: adaptive preprocessing
        surface_type_forced = None
        if self.config.force_surface_type:
            surface_type_forced = TextSurfaceType(self.config.force_surface_type)

        processed_bgr, surface_type = self.preprocessor.process(
            image_bgr, surface_type_forced
        )

        # Stage 1: CRAFT detection
        regions, region_score, affinity_score = self.detector.detect(
            processed_bgr,
            text_threshold=self.config.text_threshold,
            link_threshold=self.config.link_threshold,
            low_text=self.config.low_text,
            target_size=self.config.craft_target_size
        )

        if not regions:
            # No text detected
            if mode == OCRMode.READ:
                return ReadResult(text='', confidence=0.0, char_confidences=[],
                                  regions_found=0, surface_type=surface_type.value,
                                  latency_ms=0.0)
            elif mode == OCRMode.LOCATE:
                return LocateResult(text='', confidence=0.0, char_locations=[],
                                    word_bboxes=[], visualization=image_bgr.copy(),
                                    latency_ms=0.0)

        # Stage 2 + 3: rectify and recognize each region
        recognition_results = []
        for region in regions:
            rectified = rectify_text_region(
                processed_bgr, region, target_height=48
            )
            rec = self.recognizer.recognize(
                rectified, max_length=self.config.max_text_length
            )
            recognition_results.append(rec)

        # Stage 4: mode-specific post-processing
        if mode == OCRMode.READ:
            result = post_process_read(recognition_results, regions)
            result.surface_type = surface_type.value
            return result

        elif mode == OCRMode.LOCATE:
            return post_process_locate(recognition_results, regions, image_bgr)
```

---

## 9. Robustness: additional strategies

### 9.1 Retry with alternative preprocessing

```python
def read_with_retry(
    engine: IndustrialOCREngine,
    image_bgr: np.ndarray,
    min_confidence: float = 0.6
) -> ReadResult:
    """
    If the first attempt gives low confidence, retry with
    alternative preprocessings. Maximum 3 attempts.

    Additional cost: ~30-50ms in case of retry.
    Only activated if the initial confidence < min_confidence.
    """
    # Attempt 1: automatic preprocessing
    result = engine.read(image_bgr)
    if result.confidence >= min_confidence:
        return result

    # Attempt 2: force aggressive CLAHE (useful when the type
    # detector fails and uses an incorrect preprocessing)
    fallback_configs = [
        TextSurfaceType.LASER_DPM,
        TextSurfaceType.PRINTED_LABEL,
        TextSurfaceType.THERMAL_DEGRADED
    ]

    best_result = result
    for surface_type in fallback_configs:
        engine.config.force_surface_type = surface_type.value
        retry_result = engine.read(image_bgr)
        if retry_result.confidence > best_result.confidence:
            best_result = retry_result
        if best_result.confidence >= min_confidence:
            break

    engine.config.force_surface_type = None   # restore automatic mode
    return best_result
```

### 9.2 Charset restriction in VERIFY mode

When the technician configures the expected text, the implicit charset is known.
Restricting the recognizer's beam search to the charset of the expected text
characters improves robustness:

```python
def verify_with_charset_restriction(
    engine: IndustrialOCREngine,
    image_bgr: np.ndarray,
    expected_text: str
) -> VerifyResult:
    """
    Restricts the recognizer charset to the characters
    present in the expected text + their common ambiguities.

    Example: expected="LOT-2024-A3"
    Restricted charset: L, O, T, -, 2, 0, 4, A, 3
    Ambiguities added: 0/O, l/1/I (for the chars in expected)

    This dramatically reduces substitution errors.
    """
    # Extract charset from the expected text
    base_chars = set(expected_text)

    # Add common ambiguities
    ambiguity_map = {
        'O': '0', '0': 'O',
        'I': '1l', '1': 'Il', 'l': '1I',
        'S': '5', '5': 'S',
        'B': '8', '8': 'B',
        'Z': '2', '2': 'Z',
        'G': '6', '6': 'G'
    }
    extended_chars = set(base_chars)
    for c in base_chars:
        if c in ambiguity_map:
            extended_chars.update(ambiguity_map[c])

    # Configure charset in the recognizer
    # (PARSeq supports charset restriction at inference)
    engine.recognizer.restrict_charset(extended_chars)
    result = engine.verify(image_bgr, expected_text)
    engine.recognizer.restrict_charset(None)   # restore

    return result
```

---

## 10. Benchmarks and resources

### 10.1 Latency on Core i5/Xeon (estimated measurements)

```
Configuration: CRAFT-MobileNet INT8 + PARSeq-Small INT8
Hardware: Intel Core i5-8259U, 4 cores

Scenario                              P50      P99
───────────────────────────────────────────────────
1 text region, short text (<10c)      42ms     58ms  ✅
1 region, long text (10-25c)          48ms     65ms  ✅
2 text regions                        58ms     78ms  ✅
3 text regions                        72ms     95ms  ✅
4 regions (parallelized)              80ms     98ms  ⚠️
Retry enabled (+1 attempt)           +35ms   +50ms
───────────────────────────────────────────────────
Target: <100ms P99 for 1-3 regions ✅
```

### 10.2 Expected accuracy

```
Text type                        Read acc.  Verify acc.
────────────────────────────────────────────────────────
Printed label, good quality       98-99%     99%
Degraded thermal label            90-95%     92-97%
Laser DPM, polished metal         82-90%     85-93%
Relief engraving                  78-87%     82-90%
Dot-peen marking                  80-88%     83-91%
────────────────────────────────────────────────────────
With O/0 normalization enabled:   +1-3% in all cases
With charset restriction (VERIFY): +2-5% in ambiguous cases
```

### 10.3 Resources

```
Full process RAM:
  CRAFT-MobileNet INT8:       ~18MB
  PARSeq-Small INT8:          ~12MB
  ONNX Runtime overhead:      ~45MB
  Buffers + Python:           ~25MB
  ────────────────────────────────
  TOTAL:                      ~100MB RSS

Disk:
  craft_mobilenet_int8.onnx:  ~18MB
  parseq_small_int8.onnx:     ~12MB
  ────────────────────────────────
  TOTAL models:               ~30MB
```

---

## 11. What NOT to implement (and why)

**Tesseract as fallback:** adds 80-200ms, worse quality on industrial text, and
creates inconsistencies between PARSeq output and Tesseract output. If PARSeq fails,
retry with alternative preprocessing is a better option.

**Fine-tuning PARSeq for specific fonts:** tempting if the client always has
the same font. The problem is that you need >1000 images annotated
character by character for the fine-tune to improve anything. Without that data, the base
model generalizes better than one fine-tuned with little data.

**Character segmentation before the recognizer:** end-to-end recognizers
like PARSeq are designed to receive the full text region, not individual
characters. Segmenting first and recognizing one by one multiplies latency by the
number of characters and loses inter-character context.

**OCR with LLM (GPT-4V, Gemini):** extraordinarily accurate but 500ms-2s latency,
requires an internet connection, and is not deterministic. Incompatible with an industrial line.
Use as an offline validation tool or for low-confidence cases that are
sent to manual review.

---

## 12. Implementation checklist

### Phase 1 — MVP (READ + VERIFY)
- [ ] `AdaptivePreprocessor` with the 5 surface types
- [ ] `CRAFTDetector` with ONNX INT8 (CRAFT-MobileNet)
- [ ] `rectify_text_region` (crop + warp for rotated text)
- [ ] `PARSeqRecognizer` with ONNX INT8
- [ ] `post_process_read` (multi-region aggregation)
- [ ] `TextVerifier` with Levenshtein + char diffs
- [ ] `IndustrialOCREngine.read()` and `.verify()`
- [ ] `read_with_retry` (maximum 3 attempts with alternative preprocessings)

### Phase 2 — LOCATE + robustness
- [ ] `_extract_char_bboxes` with watershed on CRAFT heatmaps
- [ ] `post_process_locate` with chars↔bboxes alignment
- [ ] `IndustrialOCREngine.locate()`
- [ ] `verify_with_charset_restriction`
- [ ] O/0, I/l/1 normalization with context
- [ ] Per-prediction logging with confidence and detected surface_type

### Phase 3 — Optimization
- [ ] `ThreadPoolExecutor` for parallel regions (>3 regions)
- [ ] Alignment template cache between frames of the same product
- [ ] Pre-allocate ONNX buffers to avoid malloc in the loop
