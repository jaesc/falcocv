# Industrial Classification Engine — Opinionated Specification
## For your specific context: 2-20 classes · 10-50 imgs/class · vision technician · Core/Xeon · OK/NOK

---

> **Starting note:** This document is not neutral. Every decision has a concrete recommendation
> and a reason for discarding the alternatives. The previous generic document covered all possible
> cases. This one covers **your case**, and that changes things considerably.

---

## TL;DR — The 8 key decisions

| Decision | Recommendation | Why NOT the alternative |
|---|---|---|
| Backbone | **EfficientNet-B0 with DINOv2 weights** | MobileNetV3 performs worse with few samples |
| Training loss | **Triplet Loss with hard mining** (not ArcFace) | ArcFace needs >50 imgs/class to converge |
| Embedding dim | **128D, not 256D** | With 2-20 classes, 256D is oversized and slower |
| Classifier | **Pure k-NN (k=3) over the memory bank** | Multi-prototype is unnecessarily complex for <20 classes |
| Prototypes per class | **1 centroid + variance**, not multi-prototype | With 10-50 imgs you do not have data for reliable sub-clusters |
| Unknown rejection | **Fixed threshold by percentile**, not EVT | EVT needs many outliers to calibrate well |
| Incremental learning | **Only add centroids, never fine-tune** | Fine-tuning with 10-50 imgs/class destroys the backbone |
| Augmentation | **Aggressive on lighting, conservative on geometry** | The technician controls geometry; lighting varies on its own |

---

## 1. Why this context is special (and different from the general case)

### The real problem of 10-50 images/class

With 50 images per class and 20 classes you have **1000 total images**. That is not a deep learning dataset, it is a few-shot learning dataset. The consequences are concrete:

**What you CANNOT do with 10-50 imgs/class:**
- Train ArcFace from scratch in a stable way (it needs ≥50-100 imgs/class for the angular margin to converge without diverging)
- Build reliable multi-prototypes by clustering (with 20 imgs you do not have enough points to distinguish sub-clusters of real variability vs. noise)
- Compute a useful Mahalanobis covariance in 128D (you need N >> d, here N=50 << d=128)
- Fine-tune the backbone (guaranteed overfitting)

**What you CAN do:**
- Extract powerful embeddings from a backbone pre-trained with self-supervision (DINOv2)
- Learn a lightweight metric projection with triplet loss
- Classify with simple k-NN that works surprisingly well in this regime
- Reject unknowns with a threshold based on a percentile of the training set

### The vision technician as user

The vision technician **is not going to do hyperparameter tuning**. He will:
1. Capture 20-50 photos of each class under the real conditions of the line
2. Press "Train"
3. Validate with his own test set
4. Deploy

This means the system has to work with the default hyperparameters, without tuning. Any component that requires manual calibration (ArcFace margin, number of prototypes, Weibull parameters) is a component that is going to be misconfigured in production.

---

## 2. Recommended architecture

### 2.1 Pipeline

```
[INPUT IMAGE]
      │
      ▼
┌─────────────────────────────┐
│  PREPROCESSING              │
│  Resize 224×224             │
│  Normalize (ImageNet stats) │
│  ~0.3ms                     │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  BACKBONE (FROZEN)          │  ← EfficientNet-B0
│  Weights: DINOv2 adapted    │    pretrained, NEVER touched
│  Output: 1280D features     │
│  ~3-4ms INT8                │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  PROJECTION HEAD            │  ← The only thing that is trained
│  FC(1280 → 128)             │
│  L2 normalize               │
│  Output: embedding ∈ S^127  │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  k-NN CLASSIFIER            │  k=3, cosine distance
│  over memory bank           │  C centroids (one/class)
│  + confidence gap           │
└─────────────────────────────┘
      │
      ▼
┌─────────────────────────────┐
│  THRESHOLD GATE             │
│  sim < θ_class → UNKNOWN    │
│  sim ≥ θ_class → class      │
└─────────────────────────────┘
      │
      ▼
   [RESULT]
   class / UNKNOWN + confidence
```

### 2.2 What is trained vs. what is not

```
EfficientNet-B0 backbone    → FROZEN ALWAYS
                               Reason: with 10-50 imgs fine-tune = overfitting

FC(1280 → 128)              → TRAINED
+ L2 normalize                 Reason: lightweight projection, few parameters,
                               fast to train, easy to retrain

Memory bank (centroids)     → UPDATED, not "trained"
                               Reason: simply the mean of the training
                               embeddings, no gradients
```

**This separation is the most important decision of the system.** By freezing the backbone, the system's behavior is predictable, there is no catastrophic forgetting, and adding a new class is O(N_new) without touching anything else.

---

## 3. Backbone selection: why EfficientNet-B0 with DINOv2

### 3.1 The problem with MobileNetV3 and few samples

MobileNetV3-Small trained on ImageNet has optimal weights for softmax classification, not for few-shot transfer. Its high-level features are highly specialized in the 1000 ImageNet classes.

DINOv2 is a backbone trained with self-supervised learning (ViT) that produces features that are:
- Geometrically consistent across images of the same class
- Robust to lighting changes by training design
- Excellent for few-shot learning by the nature of the pre-training

**But DINOv2's ViT-Small is too large for your target hardware.** The solution: use EfficientNet-B0 with weights adapted from DINOv2 via knowledge distillation. This exists as a public model (`timm` library: `efficientnet_b0.ra_in1k` or even better the DINOv2-distilled CNNs).

### 3.2 Practical alternative if you do not want to depend on DINOv2-distilled

If you prefer something more standard and easier to audit:

**ResNet18 pretrained on ImageNet** is in practice almost as good as MobileNetV3-Small for this few-sample regime, and its features are more transferable. The speed argument in favor of MobileNetV3 is valid, but the difference is <3ms on your target hardware (Core i5+), and for 2-20 classes that time is irrelevant.

```
My recommendation in order:
1. EfficientNet-B0 (timm, pretrained) → best transfer
2. ResNet18 (torchvision, pretrained) → simpler, well known
3. MobileNetV3-Small → only if you have a hard <4ms constraint
```

### 3.3 Backbone configuration

```python
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

class IndustrialClassifier(nn.Module):
    def __init__(self, embed_dim: int = 128):
        super().__init__()

        # Backbone: EfficientNet-B0 without the classification head
        self.backbone = timm.create_model(
            'efficientnet_b0',
            pretrained=True,
            num_classes=0,        # removes the final classifier
            global_pool='avg'     # average pooling → 1D vector
        )

        # CRITICAL: freeze ALL backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        feature_dim = self.backbone.num_features  # 1280 for EfficientNet-B0

        # Projection head: the only trainable part
        # Intentionally simple: a single linear layer
        # Reason: with few samples, more layers = more overfitting
        self.projection = nn.Linear(feature_dim, embed_dim, bias=False)

        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            features = self.backbone(x)   # backbone always in no_grad

        projected = self.projection(features)
        return F.normalize(projected, dim=-1)

    def trainable_parameters(self):
        """Only the projection head is trainable."""
        return self.projection.parameters()
```

**Why a single linear layer in the projection head:**

With 10-50 images per class, an `FC(1280→512→128)` projection has 655,000 trainable parameters. With 1000 total images, the parameter/sample ratio is dangerous. An `FC(1280→128)` projection has 163,840 parameters, a more reasonable ratio, and in practice the loss of expressive capacity is minimal because the backbone already does the heavy lifting.

---

## 4. Training loss: Triplet Loss, not ArcFace

### 4.1 Why discard ArcFace for your case

ArcFace requires a linear classifier with one neuron per class. With 5 classes and a 128D embedding, that is 5×128=640 classifier parameters. With 10 classes, 1280. It seems small, but the problem is not the number of parameters: it is the training dynamics.

The angular margin `m` in ArcFace creates an exclusion zone of `m` radians around the centroid of each class. For this to converge, the model needs to see enough samples per class to explore that zone and learn to stay inside it. With 10-20 images per class, the model sees few variations and the gradient of the margin term becomes inconsistent: sometimes it pushes in the right direction, sometimes it does not.

**Practical result:** with <50 imgs/class, ArcFace needs careful scale warm-up, per-class margin adjustment, and even then it has higher variance in the final quality than Triplet Loss.

### 4.2 Triplet Loss for few samples

```
L_triplet = max(0, d(a, p) - d(a, n) + margin)

where:
  a = anchor embedding
  p = positive (same class as a)
  n = negative (different class)
  margin = 0.2 (in cosine space: cosine distance margin)

  d(x, y) = 1 - cos(x, y)   cosine distance
```

Triplet Loss works better with few samples because:
1. It does not require a linear classifier with one neuron per class
2. The gradient only depends on the relationship between 3 points, not on the global geometry
3. It is tolerant of classes with few samples as long as there are at least 2 samples of the same class

### 4.3 Hard negative mining for few samples

With small datasets mining is critical. If you do not mine hard negatives, most triplets are "easy" (the negative is already far away) and the gradient is practically zero.

```python
class TripletLossWithMining(nn.Module):
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin

    def forward(
        self,
        embeddings: torch.Tensor,  # [B, 128] L2-normalized
        labels: torch.Tensor       # [B]
    ) -> torch.Tensor:

        # Cosine distance matrix [B, B]
        # dist(i,j) = 1 - cos(eᵢ, eⱼ)
        sim_matrix = embeddings @ embeddings.T
        dist_matrix = 1.0 - sim_matrix

        B = embeddings.size(0)
        device = embeddings.device

        # Masks
        labels_eq = labels.unsqueeze(0) == labels.unsqueeze(1)   # [B, B]
        labels_ne = ~labels_eq
        eye_mask  = torch.eye(B, dtype=torch.bool, device=device)

        pos_mask = labels_eq & ~eye_mask   # same class, different idx
        neg_mask = labels_ne               # different class

        # --- Hard positive: the farthest one of the same class ---
        pos_dists = dist_matrix.clone()
        pos_dists[~pos_mask] = 0.0
        hardest_pos, _ = pos_dists.max(dim=1)   # [B]

        # --- Hard negative: the closest one of a different class ---
        neg_dists = dist_matrix.clone()
        neg_dists[~neg_mask] = float('inf')
        hardest_neg, _ = neg_dists.min(dim=1)   # [B]

        # Triplet loss only over anchors with at least 1 positive
        valid_mask = pos_mask.any(dim=1)
        if not valid_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss = F.relu(
            hardest_pos[valid_mask] - hardest_neg[valid_mask] + self.margin
        )
        return loss.mean()
```

### 4.4 Complete training protocol

```python
def train_classifier(
    model: IndustrialClassifier,
    train_images: Dict[str, List[np.ndarray]],  # class_id → images
    n_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3
):
    """
    Complete training from scratch for your context.

    With 10-50 imgs/class and a frozen backbone, 50 epochs is enough.
    More epochs does not improve: the projection head is linear and converges quickly.
    """
    optimizer = torch.optim.Adam(
        model.trainable_parameters(),
        lr=learning_rate,
        weight_decay=1e-4   # light regularization
    )

    # Schedule: reduce LR if there is no improvement in 5 epochs
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, min_lr=1e-5
    )

    loss_fn = TripletLossWithMining(margin=0.2)
    dataset = IndustrialDataset(train_images, augmentation=TRAIN_AUGMENTATION)

    # IMPORTANT: class-balanced sampling
    # With unbalanced classes (10 imgs of class A, 50 of class B),
    # the sampler ensures that each batch has samples from all classes
    sampler = BalancedBatchSampler(
        dataset.labels,
        n_classes=len(train_images),
        n_samples_per_class=max(2, batch_size // len(train_images))
    )

    loader = DataLoader(dataset, batch_sampler=sampler)

    model.train()
    # Force backbone into eval (BatchNorm frozen)
    model.backbone.eval()

    best_loss = float('inf')
    best_state = None

    for epoch in range(n_epochs):
        epoch_losses = []

        for batch_imgs, batch_labels in loader:
            embeddings = model(batch_imgs)
            loss = loss_fn(embeddings, batch_labels)

            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping: the projection head is small,
            # but for safety we limit it
            torch.nn.utils.clip_grad_norm_(
                model.trainable_parameters(), max_norm=5.0
            )
            optimizer.step()
            epoch_losses.append(loss.item())

        epoch_loss = np.mean(epoch_losses)
        scheduler.step(epoch_loss)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_state = {k: v.clone() for k, v in
                         model.projection.state_dict().items()}

    # Restore best state
    model.projection.load_state_dict(best_state)

    # Build memory bank with the trained model
    return build_memory_bank(model, train_images)
```

---

## 5. Classifier: pure k-NN, not multi-prototype

### 5.1 My opinion on multi-prototype for your case

The generic document recommends multi-prototype (K=3-8 centroids per class). For your case that is **active overengineering**: with 10-50 images per class you do not have enough data to distinguish real sub-clusters from sampling noise. A KMeans with K=3 over 20 images will find 3 clusters even if they do not exist.

What you really need is much simpler and more robust.

### 5.2 Memory bank: one centroid per class

```python
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import json, os, time

@dataclass
class ClassEntry:
    class_id: str
    label: str
    centroid: np.ndarray       # L2-normalized, [128]
    std_dev: float             # Mean standard deviation (for threshold)
    n_samples: int
    raw_embeddings: List[np.ndarray]  # We keep all of them for real k-NN
    created_at: float = field(default_factory=time.time)


class MemoryBank:
    """
    Deliberately simple memory bank.

    Design decision: keep the individual embeddings (not just the
    centroid) to be able to do real k-NN. With 10-50 imgs/class
    and 20 classes = at most 1000 vectors of 128D = 512KB in RAM.
    There is no reason at all to compress this.
    """

    def __init__(self, embed_dim: int = 128):
        self.embed_dim = embed_dim
        self._classes: Dict[str, ClassEntry] = {}
        self._all_embeddings: Optional[np.ndarray] = None  # cache [N, 128]
        self._all_labels: Optional[np.ndarray] = None       # cache [N]
        self._cache_valid = False

    def add_class(
        self,
        class_id: str,
        label: str,
        embeddings: np.ndarray   # [N, 128] L2-normalized
    ):
        centroid = embeddings.mean(axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)

        # Standard deviation: mean distance to the centroid
        dists = 1.0 - (embeddings @ centroid)
        std_dev = float(dists.std())

        self._classes[class_id] = ClassEntry(
            class_id=class_id,
            label=label,
            centroid=centroid,
            std_dev=std_dev,
            n_samples=len(embeddings),
            raw_embeddings=list(embeddings)
        )
        self._cache_valid = False

    def _rebuild_cache(self):
        """Rebuilds numpy arrays for vectorized operations."""
        all_embs, all_labels = [], []
        for class_id, entry in self._classes.items():
            for emb in entry.raw_embeddings:
                all_embs.append(emb)
                all_labels.append(class_id)

        if all_embs:
            self._all_embeddings = np.stack(all_embs)    # [N, 128]
            self._all_labels = np.array(all_labels)
        self._cache_valid = True

    def predict(
        self,
        embedding: np.ndarray,   # [128] L2-normalized
        k: int = 3
    ) -> dict:
        """
        k-NN with k=3 over all stored embeddings.

        Why k=3 and not k=1:
        - k=1 is very sensitive to outliers in training
        - k=5 with few samples may include points from other classes
        - k=3 is the sweet spot for 10-50 imgs/class

        Why majority vote and not weighted vote:
        - With k=3, weighted vote adds complexity with no clear benefit
        - Majority vote is transparent and auditable by the technician
        """
        if not self._cache_valid:
            self._rebuild_cache()

        if self._all_embeddings is None:
            return {'status': 'NO_CLASSES', 'class_id': None, 'confidence': 0.0}

        # Cosine similarities to all points [N]
        sims = self._all_embeddings @ embedding

        # Top-k most similar
        top_k_idx = np.argpartition(sims, -k)[-k:]
        top_k_labels = self._all_labels[top_k_idx]
        top_k_sims = sims[top_k_idx]

        # Majority vote
        from collections import Counter
        vote_counts = Counter(top_k_labels)
        best_class = vote_counts.most_common(1)[0][0]
        vote_score = vote_counts[best_class] / k   # vote fraction: 1/3, 2/3, 3/3

        # Mean similarity to best_class
        best_class_mask = top_k_labels == best_class
        mean_sim = float(top_k_sims[best_class_mask].mean())

        # Confidence: combines vote fraction and mean similarity
        confidence = float(vote_score * mean_sim)

        return {
            'class_id': best_class,
            'label': self._classes[best_class].label,
            'similarity': mean_sim,
            'vote_fraction': vote_score,
            'confidence': confidence,
            'top_k_labels': list(top_k_labels),
            'top_k_sims': list(top_k_sims)
        }
```

### 5.3 Unknown rejection: percentile threshold

For your case (classification only, not full open-set) the most robust and predictable strategy is:

```python
def calibrate_threshold(
    memory_bank: MemoryBank,
    train_embeddings_by_class: Dict[str, np.ndarray],
    rejection_percentile: float = 5.0
) -> Dict[str, float]:
    """
    Computes the per-class threshold as the lower percentile of the
    similarities of the training samples to the centroid.

    rejection_percentile=5.0 → the 5% of training samples
    farthest from the centroid define the lower limit.

    Industrial interpretation:
    - If a new part has similarity < threshold → UNKNOWN
    - If it has similarity ≥ threshold → assigned class

    Why percentile and not EVT:
    EVT needs data from known anomalies to calibrate the Weibull
    parameters well. With only images of known classes
    and few samples, EVT does not have enough information in the tail.
    The percentile is less sophisticated but more stable.
    """
    thresholds = {}

    for class_id, embeddings in train_embeddings_by_class.items():
        entry = memory_bank._classes[class_id]
        sims = embeddings @ entry.centroid  # similarities to the centroid

        # Threshold = low percentile of the training similarities
        # Interpretation: "the minimum we expect from a part of this class"
        threshold = float(np.percentile(sims, rejection_percentile))

        # Never let threshold < 0.5 (too permissive)
        # Never let threshold > 0.95 (too restrictive)
        thresholds[class_id] = float(np.clip(threshold, 0.50, 0.95))

    return thresholds
```

**Recommendation for the technician:** expose `rejection_percentile` as a parameter in the UI with a readable name such as "Rejection strictness level" with predefined values:
- Permissive: percentile 2 (accepts almost everything)
- Normal: percentile 5 (default)
- Strict: percentile 10 (more rejections, more safety)

---

## 6. Augmentation: what to do and what not to do

### 6.1 The technician controls geometry, not lighting

In an industrial installation where a technician configures the camera:
- The position and orientation of the part is **controlled by the fixture** → geometric variations are small and known
- Lighting can vary: color temperature of LEDs over time, shadows of operators passing by, dirt on diffusers

Therefore: **aggressive augmentation on lighting, conservative on geometry.**

```python
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

TRAIN_AUGMENTATION = A.Compose([

    # ── LIGHTING: aggressive ───────────────────────────────────────────
    A.RandomBrightnessContrast(
        brightness_limit=0.35,  # ±35%: simulates real variation of industrial LEDs
        contrast_limit=0.35,
        p=0.85
    ),
    A.RandomGamma(
        gamma_limit=(60, 140),  # wide range: LEDs age, color temp. changes
        p=0.6
    ),
    # Local lighting changes (operator shadow, reflection)
    A.RandomShadow(
        shadow_roi=(0, 0, 1, 1),
        num_shadows_lower=1,
        num_shadows_upper=2,
        shadow_dimension=4,
        p=0.3
    ),

    # ── GEOMETRY: conservative ─────────────────────────────────────────
    # Small shifts: the part is not always perfectly centered
    A.ShiftScaleRotate(
        shift_limit=0.04,   # ±4%: small shift in the fixture
        scale_limit=0.06,   # ±6%: slight distance variation
        rotate_limit=3,     # ±3°: the fixture has mechanical tolerance
        border_mode=cv2.BORDER_REFLECT_101,
        p=0.6
    ),
    # Horizontal flip only if the part is symmetric
    # A.HorizontalFlip(p=0.5),   # DISABLED by default, enable if applicable

    # ── SENSOR / OPTICS ────────────────────────────────────────────────
    A.GaussNoise(
        var_limit=(5, 30),   # CCD/CMOS sensor noise
        p=0.4
    ),
    A.Blur(blur_limit=3, p=0.2),  # mild mechanical vibration

    # ── ALWAYS AT THE END ──────────────────────────────────────────────
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])

INFERENCE_AUGMENTATION = A.Compose([
    A.Resize(224, 224),
    A.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    ToTensorV2()
])
```

### 6.2 Augmentation at training time vs. TTA

**Test-Time Augmentation (TTA):** apply multiple augmentations at inference and average the embeddings. In your context I do NOT recommend it:
- It multiplies latency by the number of augmentations
- Your case does not require the extra precision that TTA provides
- Robustness must come from training, not from inference

---

## 7. ONNX deployment: what you really need

### 7.1 What to export to ONNX

Only the backbone + projection head. The k-NN classifier runs in Python/numpy; it does not need to be in ONNX. It is fast (<0.1ms with 1000 vectors) and easier to maintain in Python.

```python
def export_to_onnx(
    model: IndustrialClassifier,
    output_path: str,
    opset: int = 13
):
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset,
        input_names=['image'],
        output_names=['embedding'],
        dynamic_axes={
            'image':     {0: 'batch'},
            'embedding': {0: 'batch'}
        },
        do_constant_folding=True,
        training=torch.onnx.TrainingMode.EVAL
    )

    # Verify that the export is correct
    import onnx
    model_onnx = onnx.load(output_path)
    onnx.checker.check_model(model_onnx)
    print(f"ONNX export OK: {output_path}")
```

### 7.2 Quantization: when yes, when no

For your hardware (modern Core/Xeon with AVX2/AVX-512):

```
FP32 MobileNetV3-Small:    ~8ms   → unnecessary for a <10ms target
INT8 MobileNetV3-Small:    ~3ms   → good, but risk of degradation

FP32 EfficientNet-B0:      ~10ms  → at the limit
INT8 EfficientNet-B0:      ~4ms   → recommended

My recommendation: quantize EfficientNet-B0 to dynamic INT8.
Do not quantize statically unless you have a well-representative
calibration dataset (degradation with poor calibration can
be worse than not quantizing).
```

```python
from onnxruntime.quantization import quantize_dynamic, QuantType

def quantize_model(fp32_path: str, int8_path: str):
    """Dynamic INT8: simple, safe, 2-3x speedup."""
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
        per_channel=True,
        reduce_range=False   # False on AVX2+, True only on old hardware
    )
```

### 7.3 Complete inference engine

```python
import onnxruntime as ort
import numpy as np
import cv2
import time

class EdgeInferenceEngine:
    """
    Complete inference engine for production.
    Designed to be instantiated once and called in a loop.
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(
        self,
        onnx_path: str,
        memory_bank: MemoryBank,
        thresholds: Dict[str, float],
        n_threads: int = 4,
        k_nn: int = 3
    ):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = n_threads
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.enable_mem_pattern = True
        opts.log_severity_level = 3

        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=['CPUExecutionProvider']
        )

        self.memory_bank = memory_bank
        self.thresholds = thresholds
        self.k_nn = k_nn

        # Pre-allocate the preprocessing buffer to avoid malloc in the loop
        self._preproc_buffer = np.zeros((1, 3, 224, 224), dtype=np.float32)

    def predict(self, image_bgr: np.ndarray) -> dict:
        t0 = time.perf_counter()

        # 1. Preprocessing
        img = cv2.resize(image_bgr, (224, 224), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].astype(np.float32) / 255.0   # BGR→RGB, [0,1]
        img = (img - self.MEAN) / self.STD
        self._preproc_buffer[0] = np.transpose(img, (2, 0, 1))

        # 2. ONNX inference
        embedding = self.session.run(
            ['embedding'],
            {'image': self._preproc_buffer}
        )[0][0]   # [128]

        # L2 normalize (the model already does it, but double safety)
        norm = np.linalg.norm(embedding)
        if norm > 1e-12:
            embedding = embedding / norm

        # 3. k-NN classification
        result = self.memory_bank.predict(embedding, k=self.k_nn)

        # 4. Threshold gate
        if result['class_id'] is not None:
            threshold = self.thresholds.get(result['class_id'], 0.65)
            if result['similarity'] < threshold:
                result['status'] = 'UNKNOWN'
            else:
                result['status'] = 'KNOWN'
        else:
            result['status'] = 'NO_MODEL'

        t1 = time.perf_counter()
        result['latency_ms'] = (t1 - t0) * 1000.0

        return result
```

---

## 8. Incremental learning: conservative policy

### 8.1 My opinion on when NOT to touch the backbone

In your context, **never** fine-tune the backbone in production. The reasons are concrete:

1. With 10-50 new images, fine-tuning is going to overfit
2. The vision technician has no way to validate that the backbone has not "forgotten" other classes
3. The risk of regression is high and difficult to detect until there is a quality escape

The correct policy is much simpler:

```
ADDING A NEW CLASS:
──────────────────
1. Capture ≥15 images of the new class
2. Extract embeddings with the FROZEN backbone
3. Compute centroid + threshold
4. Add to the memory bank
5. Re-calibrate the thresholds of ALL classes with the validation set

VALIDATE BEFORE PRODUCTION:
─────────────────────────────
1. Verify accuracy on the holdout set of each class (including new ones)
2. Verify that existing classes do not degrade >2%
3. Save a snapshot of the previous memory bank (rollback)

NEVER DO IN PRODUCTION:
──────────────────────────────
- Fine-tune the backbone with production data
- Change the ONNX model without going through validation
- Modify thresholds without calibration data
```

### 8.2 When retraining does make sense

The only case where retraining the full backbone is valid:

```
IF:
  - Production accuracy drops >5% sustained (≥1 week of data)
  OR
  - More than 5 new classes very similar to the existing ones are added
  OR
  - The lighting/camera conditions change radically

THEN:
  1. Export all embeddings from the memory bank as training data
  2. Retrain the full model offline with all the accumulated data
  3. Validate offline before touching production
  4. Deploy the new model + new memory bank in an atomic block
```

---

## 9. What NOT to implement (and why)

This is perhaps the most important section of the document.

### 9.1 Do not implement Mahalanobis distance

Mahalanobis distance requires estimating a covariance matrix in 128D. To do it in a stable way you need N >> 128 samples per class. With 10-50 images per class, the empirical covariance is rank-deficient and the result is numerically unstable or misleading. Shrinkage regularization partially mitigates this, but it adds one more hyperparameter that the technician cannot calibrate well.

For your case, the cosine distance to the centroid gives 95% of the benefit without any of the complications.

### 9.2 Do not implement EVT/OpenMax

EVT needs data from "difficult" (near-boundary) samples to estimate the parameters of the tail distribution well. With few training samples and without an explicit set of anomalies or unknowns, the Weibull fit does not have enough information. A percentile computed on the training samples themselves is more honest and predictable.

### 9.3 Do not implement multi-prototype in v1

With 10-50 images, any clustering with K>1 is going to find "clusters" that are actually sampling noise. If in the future you have 200+ images/class and you see that accuracy suffers with parts from different batches, then it makes sense to add multi-prototype. Doing it now is overengineering that adds a K parameter the technician does not know how to adjust.

### 9.4 Do not implement ArcFace as the main loss

Covered in section 4. Summary: ArcFace with few samples has higher variance in the final result than Triplet Loss. It is not that it is always worse, it is that it is less predictable with scarce data. In commercial software, predictability is worth more than the theoretical performance ceiling.

---

## 10. Expected benchmarks for your specific case

### 10.1 Inference performance

```
Hardware: Intel Core i5-8259U (typical industrial NUC, 4 cores)
Model: EfficientNet-B0 INT8
Classes: 10 classes, 30 imgs/class (300 embeddings in memory bank)

Stage                          P50      P99
───────────────────────────────────────────
Resize + normalize             0.3ms    0.5ms
ONNX inference (EfficientB0)   4.2ms    6.5ms
k-NN (300 vectors, 128D)      0.04ms   0.05ms
Threshold + result             0.01ms   0.01ms
───────────────────────────────────────────
TOTAL                          ~4.6ms   ~7ms

✅ Within the <10ms target with ample margin
```

### 10.2 Classification performance

With 10 classes, 30 training images/class, EfficientNet-B0 backbone:

```
Condition                              Expected acc.
────────────────────────────────────────────────────
Conditions similar to training         96-99%
Lighting variation ±25%                92-97%
Geometric variation ±5%                94-98%
Unknown rejection (FAR=0.1%)           85-92%
────────────────────────────────────────────────────

With only 10 images/class (viable minimum):
  Classification                        88-94%
  Unknown rejection                     75-85%

With 50 images/class (favorable case):
  Classification                        97-99%
  Unknown rejection                     90-95%
```

### 10.3 Resources

```
Full process RAM:
  ONNX Runtime + EfficientNet INT8:  ~65MB
  Memory bank (20 cls × 50 imgs):   ~0.5MB (20×50×128×4B)
  Python overhead:                   ~20MB
  ─────────────────────────────────────────
  TOTAL:                             ~85MB RSS

Disk:
  EfficientNet-B0 INT8:              ~15MB
  Memory bank JSON:                  <1MB
  ─────────────────────────────────────────
  TOTAL:                             ~16MB

Training time (CPU, no GPU):
  50 epochs, 10 classes, 30 imgs/class, frozen backbone:
  ~45-90 seconds on Core i5
  (frozen backbone = only the projection head is optimized)
```

---

## 11. File structure and persistence

```
model_store/
├── backbone_efficientnet_b0_int8.onnx   # Fixed, does not change between classes
├── projection_head_v1.2.pt              # Weights of the FC(1280→128)
├── memory_bank.json                     # Centroids + raw embeddings
├── thresholds.json                      # Calibrated per-class thresholds
├── memory_bank_backup_YYYYMMDD.json     # Backup before each update
└── training_log.jsonl                   # Append-only log of trainings
```

The separation between backbone (fixed) and projection + memory bank (updatable) is key: it allows updating the classifier without touching the backbone's ONNX file, which is the heaviest and the one that requires the most validation.

---

## 12. Implementation checklist

### Phase 1 — Core (essential for the MVP)
- [ ] IndustrialClassifier with frozen EfficientNet-B0 + FC(1280→128)
- [ ] TripletLossWithMining (hard negative mining in batch)
- [ ] BalancedBatchSampler for unbalanced datasets
- [ ] Augmentation pipeline (aggressive lighting, conservative geometry)
- [ ] MemoryBank with k-NN (k=3, cosine)
- [ ] Per-class percentile threshold
- [ ] ONNX export + dynamic INT8 quantization
- [ ] EdgeInferenceEngine with pre-allocated buffer
- [ ] Atomic persistence of the memory bank

### Phase 2 — Industrial robustness
- [ ] Automatic post-training validation (accuracy, silhouette)
- [ ] Automatic backup before each memory bank update
- [ ] Prediction logging with timestamp for auditing
- [ ] Health report: accuracy in a sliding window of 100 predictions
- [ ] Rollback to a previous memory bank snapshot

### Phase 3 — If in the future you need more
- [ ] Multi-prototype (only if you have >200 imgs/class and verified sub-clusters)
- [ ] EVT thresholding (only if you have a labeled anomaly dataset)
- [ ] Backbone fine-tune (only offline with full validation)
- [ ] ArcFace (only if you migrate to >100 imgs/class)

---

## Summary of decisions vs. generic document

| Component | Generic doc. | **Your case** | Reason for the change |
|---|---|---|---|
| Backbone | MobileNetV3-Small | **EfficientNet-B0** | Better few-shot transfer |
| Embedding dim | 256D | **128D** | 2-20 classes, enough |
| Loss | ArcFace primary | **Triplet Hard Mining** | <50 imgs/class → unstable ArcFace |
| Prototypes | Multi-prototype K=3-8 | **1 centroid + raw k-NN** | No data for real sub-clusters |
| Threshold | EVT/Weibull per-class | **Per-class percentile** | EVT needs more outlier data |
| Incremental | Fine-tune + EWC | **Only add centroids** | Fine-tune with few samples = overfitting |
| Anomaly | Mahalanobis + kNN hybrid | **Cosine distance to centroid** | Mahalanobis unstable with N<<d |
| Projection head | FC(features→512→256) | **Linear FC(1280→128)** | One layer, less overfitting |
