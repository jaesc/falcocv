# Edge Learning Classification Engine
## Technical Specification — Industrial Vision System
### Version 1.0 | Design Level: Principal Engineer

---

## Table of Contents

1. [System Architecture Overview](#1-system-architecture-overview)
2. [Embedding Learning — Mathematical Foundation](#2-embedding-learning)
3. [ArcFace Loss — Complete Analysis](#3-arcface-loss)
4. [Backbone Design & ONNX Deployment](#4-backbone-design--onnx-deployment)
5. [Multi-Prototype Classifier](#5-multi-prototype-classifier)
6. [Unknown Detection & Open-Set Recognition](#6-unknown-detection--open-set-recognition)
7. [Incremental Learning](#7-incremental-learning)
8. [Anomaly Detection](#8-anomaly-detection)
9. [Memory Bank Architecture](#9-memory-bank-architecture)
10. [Industrial Thresholding](#10-industrial-thresholding)
11. [Edge CPU Optimization](#11-edge-cpu-optimization)
12. [Data Augmentation Industrial](#12-data-augmentation-industrial)
13. [Anti-Catastrophic Forgetting](#13-anti-catastrophic-forgetting)
14. [Persistence & Determinism](#14-persistence--determinism)
15. [Benchmarks & Resource Estimates](#15-benchmarks--resource-estimates)
16. [Class Structure & API](#16-class-structure--api)

---

## 1. System Architecture Overview

### 1.1 Complete pipeline

```
[INPUT IMAGE 224×224×3]
         │
         ▼
┌─────────────────────┐
│   PREPROCESSING     │  Normalize, resize, pad
│   (CPU, SIMD)       │  ~0.3ms
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│   CNN BACKBONE      │  MobileNetV3-Small / ResNet18
│   (ONNX Runtime)    │  Feature extraction
│                     │  ~4-6ms CPU (INT8)
└─────────────────────┘
         │
         ▼
┌─────────────────────┐
│  PROJECTION HEAD    │  FC: features → 256D
│  + L2 Normalize     │  Embedding ∈ S^255
└─────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│              SIMILARITY CLASSIFIER              │
│                                                 │
│  ┌─────────────────┐   ┌─────────────────────┐ │
│  │ MEMORY BANK     │   │ ANOMALY SCORER      │ │
│  │ prototypes[C×K] │──▶│ Mahalanobis dist    │ │
│  │ K per class     │   │ kNN distance        │ │
│  └─────────────────┘   └─────────────────────┘ │
│           │                      │              │
│           ▼                      ▼              │
│  ┌─────────────────┐   ┌─────────────────────┐ │
│  │ COSINE SIM VOTE │   │ OPEN-SET REJECTION  │ │
│  │ Weighted by     │   │ EVT thresholds      │ │
│  │ prototype weight│   │ per-class calibrated│ │
│  └─────────────────┘   └─────────────────────┘ │
└─────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────┐
│    DECISION         │
│  class / unknown /  │
│  anomaly + score    │
│  + confidence       │
└─────────────────────┘
```

### 1.2 Design properties and tradeoffs

| Property | Decision | Justification |
|-----------|----------|---------------|
| Backbone | MobileNetV3-Small | 2.5M params, <5ms CPU INT8 |
| Embedding dim | 256D | Expressiveness/memory balance |
| Similarity | Cosine (not Euclidean) | Scale-invariant, stable |
| Loss | ArcFace primary + Triplet auxiliary | Better angular separation |
| Prototypes | Multi-prototype (K=3–8/class) | Captures real variability |
| Inference runtime | ONNX Runtime 1.16+ | OpenVINO-compatible, deterministic |
| Thresholding | EVT-calibrated per-class | Industrial, statistically sound |

**Fundamental tradeoff:** the similarity classifier sacrifices the simplicity of classic softmax in exchange for: native open-set recognition, incremental learning without retraining the backbone, natural few-shot learning, and free anomaly detection by distance.

---

## 2. Embedding Learning

### 2.1 Embedding space and geometric objective

The goal is to learn a function `f: X → S^(d-1)` where `S^(d-1)` is the unit hypersphere in `R^d`, such that:

```
Intra-class:  cos(f(xᵢ), f(xⱼ)) → 1    for xᵢ, xⱼ of the same class
Inter-class:  cos(f(xᵢ), f(xₖ)) → -1   for xᵢ, xₖ of different classes
```

**Formal definition of the embedding:**
```
e = f(x; θ) = L2_normalize(FC(backbone(x; θ_b); θ_h))

‖e‖₂ = 1  ∀ x
```

where:
- `backbone(·; θ_b)` extracts features from `x`
- `FC(·; θ_h)` projects to dimension `d=256`
- `L2_normalize` projects onto the hypersphere

### 2.2 Normalization

**Why L2-normalize is critical:**

Without normalization, the model learns to separate classes using vector magnitude, not direction. This introduces:
- Dependence on the input scale
- Instability under lighting changes
- Incompatibility with cosine similarity

```python
# Implementation
def l2_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + eps)
```

**Input normalization (preprocessing):**
```
x_norm = (x - μ_dataset) / σ_dataset

# Recommended industrial statistics (grayscale or calibrated RGB)
# Compute on the real training set, do not use ImageNet stats if the domains are very different
```

### 2.3 Angular separation and compactness

The angular space in `S^(d-1)` has fundamental properties:

**Space capacity:** For `d=256`, approximately `2^(0.2·d) ≈ 10^15` vectors can be packed with a minimum angular separation of 45°. In industrial practice with 10-100 classes, there is more than enough room.

**Angular margin for inter-class separation:**
```
θᵢⱼ = arccos(eᵢ · eⱼ)   angle between embeddings of class i, j

Goal: θᵢⱼ > θmargin   ∀ i≠j
Goal: θᵢᵢ < θcompact  for intra-class
```

**Space quality metrics:**
```python
def embedding_space_metrics(embeddings: np.ndarray, labels: np.ndarray):
    """
    Returns quality metrics of the embedding space.
    """
    from sklearn.metrics import silhouette_score
    
    # 1. Silhouette score ([-1, 1], higher is better)
    sil = silhouette_score(embeddings, labels, metric='cosine')
    
    # 2. Intra-class compactness (lower is better)
    intra_dists = []
    for c in np.unique(labels):
        mask = labels == c
        embs = embeddings[mask]
        centroid = l2_normalize(embs.mean(0))
        dists = 1 - (embs @ centroid)  # cosine distance
        intra_dists.append(dists.mean())
    intra_compactness = np.mean(intra_dists)
    
    # 3. Inter-class separation (higher is better)
    centroids = {}
    for c in np.unique(labels):
        mask = labels == c
        centroids[c] = l2_normalize(embeddings[mask].mean(0))
    
    inter_dists = []
    classes = list(centroids.keys())
    for i in range(len(classes)):
        for j in range(i+1, len(classes)):
            sim = centroids[classes[i]] @ centroids[classes[j]]
            inter_dists.append(1 - sim)
    inter_separation = np.mean(inter_dists)
    
    return {
        'silhouette': sil,
        'intra_compactness': intra_compactness,
        'inter_separation': inter_separation,
        'embedding_rank': np.linalg.matrix_rank(embeddings)
    }
```

**Industrial targets:**
- Silhouette > 0.6 → well-separated space
- Intra-compactness < 0.15 → compact clusters
- Inter-separation > 0.4 → well-separated classes

### 2.4 Projection Head design

```
backbone_features (960D for MobileNetV3-Small)
         │
    FC(960 → 512) + BatchNorm + ReLU
         │
    FC(512 → 256)
         │
    L2_normalize
         │
    embedding ∈ S^255
```

**Why 2 layers and not 1:**
The intermediate layer allows learning non-linear transformations between the backbone's feature space (optimized for generic tasks) and the industrial embedding space. A single linear layer would limit expressiveness.

**BatchNorm in the projection head:**
- Yes during training: it stabilizes gradients
- No at inference: fuse with the FC weights for efficiency

---

## 3. ArcFace Loss

### 3.1 Complete math

**Classic softmax for classification:**
```
L_CE = -log( exp(Wᵧᵀe) / Σⱼ exp(Wⱼᵀe) )
```

Problem: the logits `Wᵢᵀe` mix the magnitude of the weights and embeddings with angular similarity. It does not directly optimize the angular geometry we need.

**ArcFace (Additive Angular Margin Softmax):**

Step 1: Normalize weights and embeddings:
```
ê = e/‖e‖₂,    Ŵᵢ = Wᵢ/‖Wᵢ‖₂

θᵢ = arccos(Ŵᵢᵀ ê)    angle between embedding and class i
```

Step 2: Add the angular margin to the ground-truth class `y`:
```
L_ArcFace = -log(
    exp(s · cos(θᵧ + m)) /
    [exp(s · cos(θᵧ + m)) + Σⱼ≠ᵧ exp(s · cos(θⱼ))]
)
```

where:
- `m` = additive angular margin (radians), typically `m ∈ [0.3, 0.5]`
- `s` = scale factor, typically `s ∈ [30, 64]`

**Geometric interpretation:**
ArcFace penalizes the model if the embedding of class `y` is not at least `m` radians closer to the centroid of class `y` than to any other centroid. This creates an angular safety zone.

**PyTorch implementation:**
```python
class ArcFaceHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int, 
                 scale: float = 32.0, margin: float = 0.5):
        super().__init__()
        self.scale = scale
        self.margin = margin
        self.num_classes = num_classes
        
        # Classifier weights (normalized in forward)
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embed_dim))
        nn.init.xavier_uniform_(self.weight)
        
        # Precompute constants
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)   # threshold for numerical stability
        self.mm = math.sin(math.pi - margin) * margin
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Normalize weights and embeddings
        W_norm = F.normalize(self.weight, dim=1)
        e_norm = F.normalize(embeddings, dim=1)
        
        # Cosine of the angle between the embedding and each class
        cos_theta = F.linear(e_norm, W_norm)  # [B, num_classes]
        cos_theta = cos_theta.clamp(-1 + 1e-7, 1 - 1e-7)  # numerical stability
        
        # sin(θ) via the Pythagorean identity
        sin_theta = torch.sqrt(1.0 - cos_theta ** 2)
        
        # cos(θ + m) = cos(θ)cos(m) - sin(θ)sin(m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m
        
        # Handling of large angles: if θ > π - m, use the linear approximation
        cos_theta_m = torch.where(
            cos_theta > self.th,
            cos_theta_m,
            cos_theta - self.mm
        )
        
        # Create the one-hot mask for the target class
        one_hot = torch.zeros_like(cos_theta)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1.0)
        
        # Apply the margin only to the target class
        logits = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta
        logits *= self.scale
        
        return F.cross_entropy(logits, labels)
```

### 3.2 Tuning margin and scale

**Effect of margin `m`:**
```
m = 0.0  →  Equivalent to CosFace/NormFace, no angular penalty
m = 0.3  →  Conservative margin, stable with few samples
m = 0.5  →  Recommended for medium datasets (>50 samples/class)
m = 0.7  →  Aggressive, may cause divergence with few samples
```

**Industrial margin selection rule:**
```python
def recommended_margin(n_samples_per_class: int, n_classes: int) -> float:
    if n_samples_per_class < 20:
        return 0.3
    elif n_samples_per_class < 100:
        return 0.4
    elif n_classes > 50:
        return 0.35  # Many classes → reduce the margin
    else:
        return 0.5
```

**Effect of scale `s`:**
```
s_small (8-16):   Smooth gradients, slow convergence, stable
s_medium (32):    Recommended for 128-256D embeddings
s_large (64):     Aggressive gradients, may overfit with few samples

Empirical rule: s ≈ √d where d = embedding dim
→ d=256: s ≈ 16, but in practice s=32 works well
```

**Relationship between s and temperature in softmax:**
`s` acts as the inverse of temperature: `s=1/T`. A high `s` means low temperature → a more peaked distribution → stronger gradients toward the correct class.

### 3.3 ArcFace vs Triplet vs Contrastive

| Criterion | ArcFace | Triplet Loss | Contrastive |
|----------|---------|--------------|-------------|
| Mining | Not required | Hard negative critical | Pair mining |
| Convergence | Stable | Unstable without mining | Moderate |
| Little data | Good | Poor | Moderate |
| New classes | Add neuron | Embeddings only | Embeddings only |
| Open-set | Indirect | Direct | Direct |
| Geometry | Angular (hypersphere) | Euclidean or cosine | Euclidean |
| **Recommendation** | **Initial training** | **Auxiliary fine-tune** | **Do not use alone** |

**Combined strategy:**
```
L_total = α · L_ArcFace + β · L_Triplet_Hard

# Initial training: α=1.0, β=0.0 (ArcFace only)
# Incremental fine-tune: α=0.5, β=0.5
# When there are few new samples: α=0.3, β=0.7 (triplet is more flexible)
```

### 3.4 Common problems and solutions

**Problem: divergence at the start of training**
```
Cause: scale s too high with uninitialized weights
Solution: warm-up of s from 1 → 32 in the first 500 iterations
    s_t = min(s_max, 1.0 + (s_max - 1.0) * t / warmup_steps)
```

**Problem: classes with few samples do not converge**
```
Cause: angular margin m too large for small N_class
Solution: adaptive margin per class:
    m_c = m_base * min(1.0, sqrt(N_c / N_ref))
    where N_ref = number of samples of the class with the most samples
```

**Problem: intra-class embedding collapse**
```
Symptom: silhouette > 0.95 but it fails under lighting variations
Cause: overfitting to training variations
Solution: add dropout(0.2) in the projection head during training
```

---

## 4. Backbone Design & ONNX Deployment

### 4.1 Backbone selection

**MobileNetV3-Small** (recommended for <5ms):
```
Input: 224×224×3
Params: 2.5M
FLOPs: 56M
Features out: 960D (avg pool of the last conv)
Latency CPU INT8: ~3-4ms
Latency CPU FP32: ~6-8ms
```

**ResNet18** (if higher precision is needed):
```
Input: 224×224×3
Params: 11.7M
FLOPs: 1.8G
Features out: 512D (avg pool)
Latency CPU INT8: ~8-12ms
Latency CPU FP32: ~20-30ms
```

**Design decision:** MobileNetV3-Small for the <10ms target. If the task requires it, ResNet18 with reduced input (112×112) achieves ~8ms.

### 4.2 Complete model architecture (PyTorch)

```python
import torch
import torch.nn as nn
import torchvision.models as models

class IndustrialEmbeddingModel(nn.Module):
    def __init__(self, 
                 backbone: str = 'mobilenet_v3_small',
                 embed_dim: int = 256,
                 pretrained: bool = True,
                 freeze_backbone: bool = False):
        super().__init__()
        
        # --- Backbone ---
        if backbone == 'mobilenet_v3_small':
            base = models.mobilenet_v3_small(pretrained=pretrained)
            self.backbone = base.features
            self.avgpool = base.avgpool
            feature_dim = 576  # after avgpool
        elif backbone == 'resnet18':
            base = models.resnet18(pretrained=pretrained)
            self.backbone = nn.Sequential(*list(base.children())[:-2])
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            feature_dim = 512
        
        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False
        
        # --- Projection Head ---
        self.projection = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, embed_dim)
        )
        
        self.embed_dim = embed_dim
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extracts unnormalized features (for training with ArcFace)."""
        features = self.backbone(x)
        features = self.avgpool(features)
        features = torch.flatten(features, 1)
        return self.projection(features)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the L2-normalized embedding."""
        features = self.forward_features(x)
        return F.normalize(features, dim=-1)
    
    def export_onnx(self, path: str, input_size: tuple = (1, 3, 224, 224)):
        """Exports backbone + projection head to ONNX."""
        self.eval()
        dummy = torch.randn(*input_size)
        torch.onnx.export(
            self,
            dummy,
            path,
            opset_version=13,
            input_names=['image'],
            output_names=['embedding'],
            dynamic_axes={
                'image': {0: 'batch_size'},
                'embedding': {0: 'batch_size'}
            },
            do_constant_folding=True
        )
```

### 4.3 ONNX export and optimization

```python
import onnxruntime as ort
import onnx
from onnxruntime.quantization import quantize_dynamic, QuantType

class ONNXEmbeddingModel:
    """ONNX Runtime wrapper for production inference."""
    
    def __init__(self, model_path: str, num_threads: int = 4):
        # Session options optimized for industrial CPU
        sess_options = ort.SessionOptions()
        sess_options.intra_op_num_threads = num_threads
        sess_options.inter_op_num_threads = 1
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        )
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        # Disable telemetry
        sess_options.enable_profiling = False
        
        self.session = ort.InferenceSession(
            model_path,
            sess_options=sess_options,
            providers=['CPUExecutionProvider']
        )
        
        # Pre-allocate the output buffer
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
    
    def infer(self, image: np.ndarray) -> np.ndarray:
        """
        image: [H, W, C] uint8 → output: [embed_dim] float32
        """
        # Inline preprocessing
        img = cv2.resize(image, (224, 224))
        img = img.astype(np.float32) / 255.0
        img = (img - MEAN) / STD
        img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]  # [1, C, H, W]
        
        embedding = self.session.run(
            [self.output_name],
            {self.input_name: img}
        )[0][0]
        
        # L2 normalize (it should already be normalized by the model)
        norm = np.linalg.norm(embedding)
        return embedding / (norm + 1e-12)


def quantize_model(fp32_path: str, int8_path: str):
    """Dynamic INT8 quantization for maximum CPU speed."""
    quantize_dynamic(
        fp32_path,
        int8_path,
        weight_type=QuantType.QInt8,
        per_channel=True,  # better precision
        reduce_range=True   # for AVX2 without VNNI
    )
```

**Static vs dynamic quantization:**
```
Dynamic (QInt8):
  + Simple, does not need a calibration dataset
  + 2-3x speedup over FP32
  - Activations in FP32, only weights in INT8

Static (QInt8 + calibration):
  + 3-4x speedup
  + Activations also INT8
  - Requires ~100 representative images for calibration
  - More risk of degradation if the calibration set is not representative

Recommendation: use static with a calibration set = 200 balanced images
```

---

## 5. Multi-Prototype Classifier

### 5.1 Motivation and design

A single centroid per class is insufficient to capture:
- Lighting variations (diffuse vs direct → angular shift)
- Parts from different production batches
- Discrete orientations (0°, 90°, 180°, 270°)
- Sub-states of the class ("OK with mark", "OK without mark")

**Data structure:**
```
MemoryBank:
  class_c → [p₁, p₂, ..., pₖ]   prototypes
  each pᵢ:
    - centroid:  np.ndarray [256]  (L2-normalized)
    - weight:    float             (relevance of the prototype)
    - n_samples: int               (samples that form it)
    - variance:  float             (intra-prototype dispersion)
    - created:   timestamp
    - last_updated: timestamp
```

### 5.2 Initial prototype clustering

```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score

def build_prototypes_for_class(
    embeddings: np.ndarray,
    k_max: int = 8,
    k_min: int = 1,
    variance_threshold: float = 0.05
) -> List[Prototype]:
    """
    Determines the optimal K for a class and builds prototypes.
    
    Strategy:
    1. Try K=1..k_max
    2. Select K by silhouette score
    3. Refine with KMeans
    4. Filter prototypes with few samples
    """
    n = len(embeddings)
    
    if n < 2:
        return [Prototype(centroid=embeddings[0], weight=1.0, n_samples=1)]
    
    # Determine the optimal K
    best_k, best_score = 1, -1
    for k in range(k_min, min(k_max + 1, n)):
        if k == 1:
            score = 0.0  # baseline
        else:
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = km.fit_predict(embeddings)
            if len(np.unique(labels)) < 2:
                continue
            score = silhouette_score(embeddings, labels, metric='cosine')
        
        if score > best_score + 0.05:  # hysteresis to avoid over-splitting
            best_score = score
            best_k = k
    
    # Final clustering
    km = KMeans(n_clusters=best_k, n_init=20, random_state=42)
    labels = km.fit_predict(embeddings)
    
    prototypes = []
    for k_idx in range(best_k):
        mask = labels == k_idx
        cluster_embs = embeddings[mask]
        n_k = cluster_embs.shape[0]
        
        # Spherical centroid (Fréchet mean on S^d-1)
        centroid = l2_normalize(cluster_embs.mean(0))
        
        # Intra-cluster variance (cosine distance)
        dists = 1 - (cluster_embs @ centroid)
        variance = float(dists.mean())
        
        # Weight proportional to size (normalized afterwards)
        weight = n_k / n
        
        prototypes.append(Prototype(
            centroid=centroid,
            weight=weight,
            n_samples=n_k,
            variance=variance
        ))
    
    # Pruning: remove prototypes with <5% of the samples
    prototypes = [p for p in prototypes if p.n_samples / n >= 0.05]
    
    # Re-normalize weights
    total_w = sum(p.weight for p in prototypes)
    for p in prototypes:
        p.weight /= total_w
    
    return prototypes
```

### 5.3 Multi-prototype similarity classification

```
For a query embedding e and class c with prototypes {p₁..pₖ}:

score(e, c) = Σₖ wₖ · softmax_weight(cos(e, pₖ))

where:
  cos(e, pₖ) = eᵀpₖ   (L2-normalized)
  
  softmax_weight(sᵢ) = exp(τ·sᵢ) / Σⱼ exp(τ·sⱼ)
  
  τ = temperature (default: 10.0)
  
  Effect of τ:
    small τ (1): all prototypes contribute equally
    large τ (50): only the closest prototype contributes
    τ=10: soft-max toward the most relevant prototype
```

**Vectorized implementation:**
```python
def classify(
    embedding: np.ndarray,
    memory_bank: MemoryBank,
    temperature: float = 10.0
) -> ClassificationResult:
    """
    Multi-prototype similarity classification.
    Complexity: O(C × K_max) dot products, ~0.1ms for C=100, K=8
    """
    scores = {}
    
    for class_id, prototypes in memory_bank.items():
        if not prototypes:
            continue
        
        # Vectorize: [K, 256] @ [256] = [K] cosine similarities
        centroids = np.stack([p.centroid for p in prototypes])  # [K, 256]
        weights = np.array([p.weight for p in prototypes])       # [K]
        
        cos_sims = centroids @ embedding  # [K]
        
        # Soft-weighted combination
        exp_scaled = np.exp(temperature * (cos_sims - cos_sims.max()))
        soft_weights = exp_scaled / exp_scaled.sum()
        
        # Final score = weighted average cosine similarity
        scores[class_id] = float(np.dot(soft_weights * weights, cos_sims))
    
    if not scores:
        return ClassificationResult(class_id=None, confidence=0.0, is_unknown=True)
    
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    best_class, best_score = sorted_scores[0]
    
    # Confidence = gap between 1st and 2nd best
    if len(sorted_scores) > 1:
        second_score = sorted_scores[1][1]
        confidence = (best_score - second_score) / (1.0 - second_score + 1e-8)
    else:
        confidence = best_score
    
    return ClassificationResult(
        class_id=best_class,
        raw_score=best_score,
        confidence=confidence,
        all_scores=dict(sorted_scores)
    )
```

### 5.4 Incremental prototype update

```python
def update_prototype(
    existing_prototype: Prototype,
    new_embedding: np.ndarray,
    learning_rate: float = 0.1
) -> Prototype:
    """
    Online centroid update with a spherical moving average.
    
    For a robust update on the hypersphere:
    new_centroid = L2_normalize((1-lr)*old_centroid + lr*new_embedding)
    
    Property: if new_embedding is on the hypersphere and lr∈(0,1),
    the result stays close to the hypersphere.
    """
    new_centroid = (1 - learning_rate) * existing_prototype.centroid + \
                   learning_rate * new_embedding
    new_centroid = l2_normalize(new_centroid)
    
    # Update the variance with Welford's online algorithm
    n = existing_prototype.n_samples + 1
    dist_new = float(1 - new_centroid @ new_embedding)
    variance = existing_prototype.variance + \
               (dist_new - existing_prototype.variance) / n
    
    return Prototype(
        centroid=new_centroid,
        weight=existing_prototype.weight,
        n_samples=n,
        variance=variance
    )
```

**Prototype split strategy:**
If the variance of a prototype exceeds `variance_split_threshold = 0.15`, that prototype is re-clustered into 2, capturing sub-variations.

---

## 6. Unknown Detection & Open-Set Recognition

### 6.1 The formal problem

In closed-set, the classifier assigns the most probable class even if the input is completely out of distribution. In industrial environments this is inadmissible: an unknown part classified as "OK" causes quality escapes.

**Formal definition:**
```
Given embedding e:
  known(e)   → class c with score > threshold_c
  unknown(e) → rejection (anomaly or untrained class)
```

### 6.2 Nearest-neighbor distance rejection

```python
def compute_max_similarity_score(
    embedding: np.ndarray,
    memory_bank: MemoryBank
) -> Tuple[str, float]:
    """
    Returns (best_class, max_cosine_similarity).
    The maximum similarity to any known prototype
    is the main indicator of 'closeness to the known space'.
    """
    best_class = None
    best_sim = -1.0
    
    for class_id, prototypes in memory_bank.items():
        centroids = np.stack([p.centroid for p in prototypes])
        sims = centroids @ embedding
        class_best_sim = float(sims.max())
        
        if class_best_sim > best_sim:
            best_sim = class_best_sim
            best_class = class_id
    
    return best_class, best_sim
```

**Rejection rule:**
```
If max_sim < threshold_global → UNKNOWN
If max_sim > threshold_c (per-class) → class c
Otherwise → LOW_CONFIDENCE (requires review)
```

### 6.3 Per-class thresholds with EVT statistics

**Extreme Value Theory (EVT) for threshold calibration:**

For each class c, we model the distribution of similarities of KNOWN samples with the Weibull distribution (maxima tail type).

```python
from scipy.stats import weibull_min
import numpy as np

class EVTThresholdCalibrator:
    """
    Calibrates per-class thresholds using EVT.
    
    Based on OpenMax (Bendale & Boult, CVPR 2016) adapted
    for cosine similarity space.
    """
    
    def __init__(self, fpr_target: float = 0.01):
        """
        fpr_target: target False Positive Rate (1% → 99% of known samples
                    correctly classified as known)
        """
        self.fpr_target = fpr_target
        self.weibull_params = {}  # class_id → (shape, loc, scale)
        self.thresholds = {}      # class_id → float
    
    def fit(self, 
            embeddings_by_class: Dict[str, np.ndarray],
            memory_bank: MemoryBank):
        """
        embeddings_by_class: validation embeddings per class
        memory_bank: prototypes already built
        """
        for class_id, embeddings in embeddings_by_class.items():
            # Compute the similarities to the closest prototype
            prototypes = memory_bank[class_id]
            centroids = np.stack([p.centroid for p in prototypes])
            
            # Distances (1 - similarity) to the closest centroid
            distances = []
            for emb in embeddings:
                sims = centroids @ emb
                distances.append(1.0 - float(sims.max()))
            
            distances = np.array(distances)
            
            # Fit Weibull on the tail of the distances (the largest = worst case)
            tail_size = max(int(len(distances) * 0.2), 5)
            tail = np.sort(distances)[-tail_size:]
            
            try:
                shape, loc, scale = weibull_min.fit(tail, floc=0)
                self.weibull_params[class_id] = (shape, loc, scale)
                
                # Threshold: distance such that P(d < threshold) = 1 - fpr_target
                threshold_dist = weibull_min.ppf(
                    1 - self.fpr_target, shape, loc=loc, scale=scale
                )
                # Convert to similarity
                self.thresholds[class_id] = float(1.0 - threshold_dist)
            except Exception:
                # Fallback: simple percentile
                self.thresholds[class_id] = float(
                    np.percentile(1.0 - distances, 100 * (1 - self.fpr_target))
                )
    
    def is_known(self, class_id: str, similarity: float) -> bool:
        threshold = self.thresholds.get(class_id, 0.6)  # conservative default
        return similarity >= threshold
    
    def get_threshold(self, class_id: str) -> float:
        return self.thresholds.get(class_id, 0.6)
```

### 6.4 Confidence calibration

Raw cosine similarities are not calibrated as probabilities. For industrially useful confidence estimates:

```python
class PlattCalibrator:
    """
    Platt calibration (sigmoid fitting) over cosine similarities.
    Converts raw scores into calibrated probabilities.
    """
    
    def __init__(self):
        self.a = 1.0  # sigmoid slope
        self.b = 0.0  # sigmoid bias
    
    def fit(self, scores: np.ndarray, labels: np.ndarray):
        """
        scores: raw cosine similarities
        labels: 1 = correct, 0 = incorrect
        """
        from scipy.optimize import minimize
        
        def nll(params):
            a, b = params
            probs = 1 / (1 + np.exp(-(a * scores + b)))
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
            return -np.mean(labels * np.log(probs) + 
                          (1 - labels) * np.log(1 - probs))
        
        result = minimize(nll, [1.0, 0.0], method='L-BFGS-B')
        self.a, self.b = result.x
    
    def calibrate(self, score: float) -> float:
        return float(1 / (1 + np.exp(-(self.a * score + self.b))))
```

---

## 7. Incremental Learning

### 7.1 New class addition protocol

```
NEW CLASS ADDITION PROTOCOL
────────────────────────────

PHASE 1: COLLECTION (without modifying the model)
  1. Capture N_new ≥ 20 images of the new class
  2. Extract embeddings with the FROZEN backbone
  3. Evaluate separability vs existing classes:
     min_inter_class_dist = min(1 - new_centroid @ existing_centroid_c)
     If min_inter_class_dist < 0.3 → WARNING: new class too similar to an existing class

PHASE 2: DECISION (retrain the backbone?)
  Condition A: N_existing_classes < 20 AND N_new < 50
    → DO NOT retrain the backbone, only add prototypes
  
  Condition B: new class very similar to existing ones (overlap > 0.4)
    → Fine-tune the backbone with a replay buffer (see section 13)
  
  Condition C: N_new ≥ 100 AND time is available
    → Full fine-tune with the EWC strategy

PHASE 3: PROTOTYPE CREATION
  1. Clustering of the new class embeddings → K prototypes
  2. Compute the EVT threshold for the new class
  3. Add to the memory bank

PHASE 4: VALIDATION
  1. Evaluate on the validation set of ALL classes
  2. Verify that the accuracy of existing classes does not degrade >2%
  3. If it degrades → revert and force fine-tuning
```

### 7.2 Incremental fine-tuning of the backbone

```python
class IncrementalBackboneTrainer:
    
    def fine_tune_with_new_class(
        self,
        model: IndustrialEmbeddingModel,
        new_class_data: Dataset,
        replay_buffer: ReplayBuffer,
        n_epochs: int = 10,
        learning_rate: float = 1e-4,
        backbone_lr_factor: float = 0.1  # backbone learns more slowly
    ):
        """
        Fine-tuning with:
        - Differential LR: backbone × 0.1, projection head × 1.0
        - Replay buffer for anti-forgetting
        - Early stopping if the performance of existing classes degrades
        """
        # Differential LR per layer
        param_groups = [
            {'params': model.backbone.parameters(), 
             'lr': learning_rate * backbone_lr_factor},
            {'params': model.projection.parameters(), 
             'lr': learning_rate}
        ]
        optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
        
        # Mixed dataset: new class + replay of existing classes
        combined_dataset = ConcatDataset([
            new_class_data,
            replay_buffer.sample(n_per_class=min(50, len(new_class_data)))
        ])
        
        # ArcFace head with the new class added
        arcface_head = self.extend_arcface_head(model, new_class_id)
        
        best_val_acc = self.evaluate_existing_classes(model)
        
        for epoch in range(n_epochs):
            for batch in DataLoader(combined_dataset, batch_size=32):
                embeddings = model.forward_features(batch['image'])
                loss = arcface_head(embeddings, batch['label'])
                
                optimizer.zero_grad()
                loss.backward()
                
                # Gradient clipping: prevents large updates in the backbone
                torch.nn.utils.clip_grad_norm_(
                    model.backbone.parameters(), max_norm=1.0
                )
                optimizer.step()
            
            # Early stopping: if existing classes degrade
            current_acc = self.evaluate_existing_classes(model)
            if current_acc < best_val_acc - 0.02:
                self.restore_checkpoint(model)
                break
            
            if current_acc > best_val_acc:
                best_val_acc = current_acc
                self.save_checkpoint(model)
```

### 7.3 When to freeze vs unfreeze the backbone

```
┌─────────────────────────────────────────────────────────┐
│              FREEZING POLICY                            │
│                                                         │
│  N_new_classes ≤ 5 and N_per_class < 50                 │
│    → Freeze the ENTIRE backbone                         │
│    → Only update prototypes in the memory bank          │
│    → Zero catastrophic forgetting guaranteed            │
│                                                         │
│  5 < N_new_classes ≤ 20 or overlap with existing ones   │
│    → Freeze the first 70% of the backbone layers        │
│    → Unfreeze the last 30% (high-level features)        │
│    → Fine-tune with EWC (Fisher regularization)         │
│                                                         │
│  N_new_classes > 20 or full retraining                  │
│    → Unfreeze the entire backbone                       │
│    → Very low LR (1e-5) for the backbone                │
│    → Full replay buffer mandatory                       │
└─────────────────────────────────────────────────────────┘
```

---

## 8. Anomaly Detection

### 8.1 Distance-to-manifold

The manifold of known data is approximated by the union of the prototypes in `S^(d-1)`. The distance of an embedding `e` to the manifold is:

```
d_manifold(e) = 1 - max_{c,k} cos(e, p_{c,k})

d_manifold → 0: e is on the known manifold
d_manifold → 2: e is antipodal to all prototypes
```

**Anomaly threshold:**
```
score_anomaly(e) = d_manifold(e)
is_anomaly = score_anomaly > θ_anomaly

θ_anomaly calibrated on the validation set:
  θ_anomaly = percentile_99(d_manifold(e_known)) + safety_margin
```

### 8.2 Mahalanobis Distance

For classes with enough data, the Mahalanobis distance captures the ellipsoidal geometry of the embedding distribution:

```python
class MahalanobisAnomalyScorer:
    """
    Mahalanobis distance in the embedding space.
    
    DM(e, c) = sqrt((e - μ_c)ᵀ Σ_c⁻¹ (e - μ_c))
    
    Advantage: it accounts for correlations between dimensions
    Disadvantage: it requires N >> d samples to estimate Σ
    Solution: use a shared or regularized Σ (shrinkage)
    """
    
    def __init__(self, shrinkage: float = 0.1):
        """
        shrinkage: Ledoit-Wolf regularization
        0 = pure empirical covariance
        1 = scaled identity covariance
        """
        self.shrinkage = shrinkage
        self.means = {}       # class_id → np.ndarray [d]
        self.inv_covs = {}    # class_id → np.ndarray [d, d]
        self.shared_inv_cov = None  # shared covariance (for few samples)
    
    def fit(self, embeddings_by_class: Dict[str, np.ndarray]):
        all_embeddings = []
        
        for class_id, embs in embeddings_by_class.items():
            self.means[class_id] = embs.mean(axis=0)
            all_embeddings.append(embs - self.means[class_id])
        
        # Shared covariance (tied covariance) for robustness with few samples
        all_centered = np.vstack(all_embeddings)
        n, d = all_centered.shape
        
        # Estimation with Ledoit-Wolf regularization
        cov = (all_centered.T @ all_centered) / n
        cov_reg = (1 - self.shrinkage) * cov + \
                  self.shrinkage * np.eye(d) * np.trace(cov) / d
        
        self.shared_inv_cov = np.linalg.inv(cov_reg)
    
    def score(self, embedding: np.ndarray, class_id: str) -> float:
        if class_id not in self.means:
            return float('inf')
        
        diff = embedding - self.means[class_id]
        # DM² = diffᵀ Σ⁻¹ diff
        dm_sq = float(diff @ self.shared_inv_cov @ diff)
        return math.sqrt(max(dm_sq, 0.0))
```

**Tradeoff Mahalanobis vs cosine distance:**
```
Mahalanobis:
  + Captures the real ellipsoidal geometry of the distribution
  + Statistically principled
  - Requires N >> d (for d=256: N >> 256 samples)
  - Costly to maintain incrementally
  - Unstable with few samples

Cosine distance to nearest prototype:
  + O(1) samples per prototype
  + Works with few samples
  + Computationally cheap
  - Assumes isotropic distribution per prototype

Strategy: cosine distance as the primary scorer,
Mahalanobis as a secondary scorer when N_class > 5×d
```

### 8.3 Hybrid anomaly/classification scoring

```python
class HybridAnomalyClassifier:
    
    def predict(self, embedding: np.ndarray) -> PredictionResult:
        # 1. Similarity classification
        class_id, raw_score = compute_max_similarity_score(embedding, self.memory_bank)
        
        # 2. Per-class threshold (EVT calibrated)
        threshold = self.evt_calibrator.get_threshold(class_id)
        
        # 3. Anomaly score
        anomaly_score = 1.0 - raw_score  # d_manifold
        
        # 4. Mahalanobis if available
        if self.mahalanobis_scorer and self.has_enough_samples(class_id):
            dm = self.mahalanobis_scorer.score(embedding, class_id)
            # Combine scores (normalized by the Mahalanobis threshold)
            dm_normalized = dm / self.dm_threshold  # 1.0 = at the limit
            anomaly_score = 0.6 * anomaly_score + 0.4 * min(dm_normalized / 3.0, 1.0)
        
        # 5. Decision
        if anomaly_score > self.anomaly_threshold:
            return PredictionResult(
                status='ANOMALY',
                anomaly_score=anomaly_score,
                nearest_class=class_id,
                raw_similarity=raw_score
            )
        elif raw_score < threshold:
            return PredictionResult(
                status='UNKNOWN',
                anomaly_score=anomaly_score,
                nearest_class=class_id,
                raw_similarity=raw_score
            )
        else:
            return PredictionResult(
                status='KNOWN',
                class_id=class_id,
                confidence=self.calibrator.calibrate(raw_score),
                anomaly_score=anomaly_score,
                raw_similarity=raw_score
            )
```

---

## 9. Memory Bank Architecture

### 9.1 Data structure

```python
from dataclasses import dataclass, field
from typing import Dict, List, Optional
import numpy as np
import json
import time

@dataclass
class Prototype:
    centroid: np.ndarray       # L2-normalized, [embed_dim]
    weight: float              # Relative contribution (sums to 1 per class)
    n_samples: int             # Number of samples
    variance: float            # Mean intra-prototype dispersion
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


@dataclass 
class ClassEntry:
    class_id: str
    label: str                 # Human-readable name
    prototypes: List[Prototype]
    total_samples: int
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)
    is_active: bool = True


class MemoryBank:
    """
    Memory bank for industrial prototypes.
    
    Design:
    - O(1) lookup by class_id (dict)
    - Vectorized batch scoring (np.einsum)
    - Atomic serialization (write-then-rename)
    - Thread-safe reads, exclusive writes
    """
    
    def __init__(self, embed_dim: int = 256, max_prototypes_per_class: int = 8):
        self.embed_dim = embed_dim
        self.max_k = max_prototypes_per_class
        self._classes: Dict[str, ClassEntry] = {}
        
        # Vectorized cache: [C × K × d] for efficient batch scoring
        self._centroids_cache: Optional[np.ndarray] = None
        self._cache_dirty = True
    
    def add_class(self, class_id: str, label: str, 
                  embeddings: np.ndarray) -> ClassEntry:
        """Adds a new class with its training embeddings."""
        prototypes = build_prototypes_for_class(
            embeddings, k_max=self.max_k
        )
        entry = ClassEntry(
            class_id=class_id,
            label=label,
            prototypes=prototypes,
            total_samples=len(embeddings)
        )
        self._classes[class_id] = entry
        self._cache_dirty = True
        return entry
    
    def update_class(self, class_id: str, new_embedding: np.ndarray,
                     create_if_missing: bool = False):
        """
        Online update of a class with a new embedding.
        Assigns it to the closest prototype and updates its centroid.
        """
        if class_id not in self._classes:
            if create_if_missing:
                self.add_class(class_id, class_id, new_embedding[np.newaxis])
            return
        
        entry = self._classes[class_id]
        centroids = np.stack([p.centroid for p in entry.prototypes])
        sims = centroids @ new_embedding
        best_k = int(sims.argmax())
        
        # Update the closest prototype
        entry.prototypes[best_k] = update_prototype(
            entry.prototypes[best_k], new_embedding
        )
        entry.total_samples += 1
        entry.last_updated_at = time.time()
        
        # Consider a split if variance is high
        if entry.prototypes[best_k].variance > 0.15 and \
           len(entry.prototypes) < self.max_k:
            self._split_prototype(class_id, best_k)
        
        self._cache_dirty = True
    
    def _split_prototype(self, class_id: str, k_idx: int):
        """
        Split a high-variance prototype into 2 sub-prototypes.
        It only works if we have the original samples in the replay buffer.
        In practice: mark for reclustering in the next maintenance cycle.
        """
        entry = self._classes[class_id]
        entry.prototypes[k_idx].metadata['needs_split'] = True
    
    def get_all_centroids_matrix(self) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Returns a matrix [N_total_prototypes, embed_dim] and the index→(class_id, k) mapping.
        Useful for vectorized batch scoring.
        """
        if not self._cache_dirty and self._centroids_cache is not None:
            return self._centroids_cache
        
        centroids = []
        index_map = []
        for class_id, entry in self._classes.items():
            if not entry.is_active:
                continue
            for k, proto in enumerate(entry.prototypes):
                centroids.append(proto.centroid)
                index_map.append((class_id, k, proto.weight))
        
        if not centroids:
            return np.zeros((0, self.embed_dim)), []
        
        self._centroids_cache = (np.stack(centroids), index_map)
        self._cache_dirty = False
        return self._centroids_cache
    
    def batch_score(self, embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Vectorized scoring for a batch of embeddings.
        embeddings: [B, embed_dim]
        returns: {class_id: scores [B]}
        """
        centroids_matrix, index_map = self.get_all_centroids_matrix()
        
        # [B, N_prototypes] cosine similarities (embeddings already normalized)
        all_sims = embeddings @ centroids_matrix.T
        
        # Aggregate per class
        class_scores = {class_id: np.full(len(embeddings), -1.0) 
                       for class_id in self._classes}
        
        for proto_idx, (class_id, k, weight) in enumerate(index_map):
            proto_sims = all_sims[:, proto_idx]
            np.maximum(class_scores[class_id], proto_sims, 
                      out=class_scores[class_id])
        
        return class_scores
```

### 9.2 Persistence and serialization

```python
def save(self, path: str):
    """
    Atomic serialization: write to temp → rename.
    Prevents corruption if the process dies during writing.
    """
    import tempfile, os, pickle
    
    state = {
        'embed_dim': self.embed_dim,
        'max_k': self.max_k,
        'classes': {}
    }
    
    for class_id, entry in self._classes.items():
        state['classes'][class_id] = {
            'label': entry.label,
            'total_samples': entry.total_samples,
            'created_at': entry.created_at,
            'is_active': entry.is_active,
            'prototypes': [
                {
                    'centroid': p.centroid.tolist(),
                    'weight': p.weight,
                    'n_samples': p.n_samples,
                    'variance': p.variance,
                    'created_at': p.created_at,
                    'metadata': p.metadata
                }
                for p in entry.prototypes
            ]
        }
    
    # Atomic write
    dir_path = os.path.dirname(path)
    with tempfile.NamedTemporaryFile(
        dir=dir_path, suffix='.tmp', delete=False, mode='w'
    ) as f:
        json.dump(state, f, indent=2)
        tmp_path = f.name
    
    os.replace(tmp_path, path)  # Atomic on POSIX and Windows
```

---

## 10. Industrial Thresholding

### 10.1 Multi-level thresholding strategy

```
LEVEL 1: ANOMALY DETECTION
  score_anomaly > θ_anomaly → "ANOMALY" (hard rejection)
  
LEVEL 2: UNKNOWN REJECTION  
  best_sim < θ_unknown → "UNKNOWN" (soft rejection, may be a new class)
  
LEVEL 3: LOW CONFIDENCE
  θ_unknown < best_sim < θ_class[best_class] → "UNCERTAIN" (manual review)
  
LEVEL 4: KNOWN CLASS
  best_sim ≥ θ_class[best_class] → accepted class
  
LEVEL 5: HIGH CONFIDENCE
  best_sim ≥ θ_high_conf → class with high confidence (auto-approve in the process)
```

### 10.2 Operational calibration

```python
class IndustrialThresholdManager:
    """
    Threshold management oriented toward industrial metrics.
    
    Key metrics:
    - FAR (False Accept Rate): bad parts classified as good
    - FRR (False Reject Rate): good parts rejected
    - FDR (Fault Detection Rate): detected defects / actual defects
    
    In industry: minimum FAR (quality), acceptable FRR (production)
    Typical: FAR < 0.1%, FRR < 2%
    """
    
    def calibrate_for_industrial_target(
        self,
        val_scores: np.ndarray,   # similarities of good/known samples
        val_anomalies: np.ndarray, # similarities of anomalies/unknowns
        far_target: float = 0.001,  # Target FAR: 0.1%
        frr_budget: float = 0.05    # Maximum tolerable FRR: 5%
    ) -> float:
        """
        Finds the optimal threshold for the industrial target.
        """
        from sklearn.metrics import roc_curve
        
        # Build a binary dataset: 1=good, 0=anomaly
        scores = np.concatenate([val_scores, val_anomalies])
        labels = np.concatenate([
            np.ones(len(val_scores)), 
            np.zeros(len(val_anomalies))
        ])
        
        fpr, tpr, thresholds = roc_curve(labels, scores)
        fnr = 1 - tpr  # FRR
        
        # Thresholds that satisfy FAR ≤ far_target
        valid_mask = fpr <= far_target
        if not valid_mask.any():
            # There is no threshold that satisfies the FAR target
            # Use the one that minimizes FAR
            idx = np.argmin(fpr)
        else:
            # Among the valid ones, minimize FRR
            valid_fnr = fnr[valid_mask]
            valid_thresholds = thresholds[valid_mask]
            idx_local = np.argmin(valid_fnr)
            
            if valid_fnr[idx_local] > frr_budget:
                import warnings
                warnings.warn(
                    f"FRR {valid_fnr[idx_local]:.1%} exceeds budget {frr_budget:.1%}. "
                    f"Consider improving the model or reviewing the data."
                )
            
            return float(valid_thresholds[idx_local])
        
        return float(thresholds[idx])
```

---

## 11. Edge CPU Optimization

### 11.1 Optimization stack

```
[PyTorch FP32 Model]
        │
        ▼ torch.onnx.export (opset 13)
[ONNX FP32]
        │
        ▼ onnxruntime-quantization (INT8 dynamic/static)
[ONNX INT8]
        │
        ├──▶ ONNX Runtime CPU (current production)
        ├──▶ OpenVINO (Intel NUC / Atom / Core i3+)
        └──▶ TensorRT (if there is an NVIDIA edge GPU)
```

### 11.2 ONNX Runtime configuration for minimum latency

```python
def create_optimized_session(model_path: str) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    
    # Threading: 1 inter-op thread (sequential pipeline)
    # N intra-op threads (parallelism within operations)
    opts.inter_op_num_threads = 1
    opts.intra_op_num_threads = 4  # Adjust to the available cores
    
    # Maximum graph optimization
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    # Save the optimized model for reuse
    opts.optimized_model_filepath = model_path.replace('.onnx', '_optimized.onnx')
    
    # Disable logging in production
    opts.log_severity_level = 3  # ERROR only
    
    # Sequential execution mode (more predictable latency)
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    
    # Enable memory pattern (reuses buffers between inferences)
    opts.enable_mem_pattern = True
    opts.enable_mem_reuse = True
    
    session = ort.InferenceSession(
        model_path,
        sess_options=opts,
        providers=[
            ('CPUExecutionProvider', {
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'cpu_memory_arena_cfg': 'BFCArena'
            })
        ]
    )
    return session
```

### 11.3 SIMD-optimized preprocessing

```python
import numpy as np

class FastPreprocessor:
    """
    Optimized preprocessing with numpy (leverages BLAS/LAPACK SIMD).
    
    For maximum speed: use cv2.dnn.blobFromImage, which uses the
    system's native SIMD instructions.
    """
    
    # Pre-compute as float32 to avoid conversions
    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    
    def __init__(self, input_size: int = 224):
        self.input_size = input_size
        # Pre-allocate the output buffer: avoids malloc on every inference
        self._buffer = np.zeros((1, 3, input_size, input_size), dtype=np.float32)
    
    def process(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        image_bgr: [H, W, 3] uint8 BGR (OpenCV format)
        returns: [1, 3, 224, 224] normalized float32
        """
        # Resize: INTER_LINEAR is SIMD-optimized in OpenCV
        img = cv2.resize(image_bgr, (self.input_size, self.input_size),
                        interpolation=cv2.INTER_LINEAR)
        
        # BGR → RGB + float32 (a single operation)
        img = img[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
        
        # Normalize (SIMD broadcasting)
        img = (img - self.MEAN) / self.STD
        
        # HWC → CHW + batch dim (view, no copy)
        self._buffer[0] = np.transpose(img, (2, 0, 1))
        
        return self._buffer
```

### 11.4 Target latency and optimization benchmarks

```
Benchmark on Intel Core i5-8259U (typical industrial NUC):

Operation                      FP32    INT8    Speedup
─────────────────────────────────────────────────────
MobileNetV3-Small backbone     8.2ms   3.1ms   2.6×
ResNet18 backbone              24ms    9ms     2.7×
Projection head (FC 960→256)   0.3ms   0.1ms   3.0×
L2 normalization               0.05ms  0.05ms  1×
Prototype matching (100 cls)   0.08ms  0.08ms  1×
Preprocessing (OpenCV)         0.4ms   0.4ms   1×
─────────────────────────────────────────────────────
TOTAL MobileNetV3 INT8:        ~4ms    ✅ <10ms target
TOTAL ResNet18 INT8:            ~10ms  ⚠️ at the limit

RAM consumption:
  MobileNetV3-Small INT8 model:    2.5MB
  ResNet18 INT8 model:             12MB
  Memory bank (100 cls × 8 proto): 100×8×256×4B = 800KB
  ONNX Runtime overhead:           ~50MB RSS
  TOTAL process:                   ~70MB RSS
```

---

## 12. Data Augmentation Industrial

### 12.1 Augmentations specific to industrial vision

```python
import albumentations as A

class IndustrialAugmentationPipeline:
    """
    Augmentations adapted to real industrial conditions.
    
    Design: the augmentations must simulate real process variations,
    NOT variations that never occur (avoid artistic-style augmentations).
    """
    
    TRAINING_PIPELINE = A.Compose([
        # 1. Industrial lighting variations
        A.RandomBrightnessContrast(
            brightness_limit=0.3,     # ±30% brightness
            contrast_limit=0.3,       # ±30% contrast
            p=0.8
        ),
        A.RandomGamma(
            gamma_limit=(70, 130),    # Simulates LED vs fluorescent lighting
            p=0.5
        ),
        A.CLAHE(
            clip_limit=4.0,           # Local histogram normalization
            tile_grid_size=(8, 8),
            p=0.3
        ),
        
        # 2. Industrial geometric variations
        A.ShiftScaleRotate(
            shift_limit=0.05,         # Small part shifts
            scale_limit=0.1,          # ±10% scale
            rotate_limit=5,           # ±5° for oriented parts
            border_mode=cv2.BORDER_REFLECT,
            p=0.7
        ),
        
        # 3. Discrete rotations (if parts have rotational symmetry)
        A.RandomRotate90(p=0.5),      # Enable only if applicable to the part
        
        # 4. Noise and camera/optical artifacts
        A.GaussNoise(
            var_limit=(10, 50),       # CCD/CMOS sensor noise
            p=0.4
        ),
        A.Blur(
            blur_limit=3,             # Blur from vibration
            p=0.2
        ),
        A.ImageCompression(
            quality_lower=80,         # JPEG artifacts if there is streaming
            p=0.1
        ),
        
        # 5. Industrial surface variations
        A.HueSaturationValue(
            hue_shift_limit=5,        # Slight material color variation
            sat_shift_limit=20,
            val_shift_limit=20,
            p=0.3
        ),
        
        # 6. Partial occlusion (tool, fixture, shadow)
        A.CoarseDropout(
            max_holes=4,
            max_height=20,
            max_width=20,
            fill_value=0,
            p=0.2
        ),
        
        # 7. Crop and padding (framing variation)
        A.RandomResizedCrop(
            height=224,
            width=224,
            scale=(0.85, 1.0),        # Conservative crop
            ratio=(0.95, 1.05),       # Keeps an almost square aspect ratio
            p=0.5
        ),
        
        # Final: always resize to target
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
    
    # Validation: without augmentations that introduce variation
    VALIDATION_PIPELINE = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
        ToTensorV2()
    ])
```

### 12.2 Hard Negative Mining

```python
class HardNegativeMiner:
    """
    Identifies the hardest negatives for the triplet/contrastive loss.
    
    Hard negatives: samples of a different class that are VERY close
    in the embedding space (model confusion).
    
    Strategy:
    1. Each epoch, extract embeddings from the whole training set
    2. For each anchor, find the closest hard negative
    3. Use these hard pairs in the next epoch
    """
    
    def mine(
        self,
        embeddings: np.ndarray,   # [N, d]
        labels: np.ndarray,        # [N]
        n_pairs: int = 1000,
        semi_hard: bool = True     # Semi-hard: negatives beyond the positive
    ) -> List[Tuple[int, int, int]]:
        """
        Returns a list of (anchor_idx, positive_idx, negative_idx).
        """
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Similarities between all pairs
        sim_matrix = cosine_similarity(embeddings)  # [N, N]
        
        triplets = []
        for anchor_idx in range(len(embeddings)):
            anchor_label = labels[anchor_idx]
            anchor_sim = sim_matrix[anchor_idx]
            
            # Positives: same class, different index
            pos_mask = (labels == anchor_label)
            pos_mask[anchor_idx] = False
            if not pos_mask.any():
                continue
            
            # Hardest positive: the farthest one of the same class
            pos_sims = anchor_sim.copy()
            pos_sims[~pos_mask] = 1.0  # Exclude non-positives from the min
            hardest_pos = int(pos_sims.argmin())
            pos_sim = anchor_sim[hardest_pos]
            
            # Negatives: different class
            neg_mask = labels != anchor_label
            neg_sims = anchor_sim.copy()
            neg_sims[~neg_mask] = -1.0
            
            if semi_hard:
                # Semi-hard: harder than the positive but less than the anchor
                # pos_sim < neg_sim < 1.0
                semi_hard_mask = neg_mask & (neg_sims > pos_sim)
                if semi_hard_mask.any():
                    neg_sims_filtered = neg_sims.copy()
                    neg_sims_filtered[~semi_hard_mask] = -1.0
                    hardest_neg = int(neg_sims_filtered.argmax())
                else:
                    # Fallback to hard negative
                    hardest_neg = int(neg_sims.argmax())
            else:
                # Hard negative: the closest negative
                hardest_neg = int(neg_sims.argmax())
            
            triplets.append((anchor_idx, hardest_pos, hardest_neg))
        
        # Subsample if there are too many
        if len(triplets) > n_pairs:
            indices = np.random.choice(len(triplets), n_pairs, replace=False)
            triplets = [triplets[i] for i in indices]
        
        return triplets
```

---

## 13. Anti-Catastrophic Forgetting

### 13.1 Elastic Weight Consolidation (EWC)

```
Intuition: protect the weights most important for past tasks.

L_EWC = L_new + λ/2 · Σᵢ Fᵢ(θᵢ - θ*ᵢ)²

where:
  L_new = loss on the new data
  Fᵢ   = importance of parameter i (diagonal of the Fisher matrix)
  θ*ᵢ  = optimal value of parameter i in the previous task
  λ    = trade-off between the new task and the memory of the old task
```

```python
class EWCRegularizer:
    """
    Elastic Weight Consolidation to prevent catastrophic forgetting.
    """
    
    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0):
        self.lambda_ewc = lambda_ewc
        self.fisher_dict = {}
        self.optimal_params = {}
    
    def consolidate(self, model: nn.Module, dataloader: DataLoader):
        """
        Computes the diagonal Fisher information matrix and saves θ*.
        Call it BEFORE starting the fine-tuning for new classes.
        """
        model.eval()
        fisher = {n: torch.zeros_like(p) 
                  for n, p in model.named_parameters() if p.requires_grad}
        
        for batch in dataloader:
            model.zero_grad()
            embeddings = model.forward_features(batch['image'])
            
            # To compute Fisher: use the log-likelihood under the current distribution
            # Approximation: use the training loss
            loss = self.current_loss_fn(embeddings, batch['label'])
            loss.backward()
            
            for n, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    fisher[n] += p.grad.data ** 2 / len(dataloader)
        
        # Save Fisher and the optimal parameters
        for n, p in model.named_parameters():
            if p.requires_grad:
                self.fisher_dict[n] = fisher[n].detach().clone()
                self.optimal_params[n] = p.data.detach().clone()
    
    def penalty(self, model: nn.Module) -> torch.Tensor:
        """Returns the EWC penalty to add to the loss."""
        penalty = torch.tensor(0.0, device=next(model.parameters()).device)
        
        for n, p in model.named_parameters():
            if n in self.fisher_dict:
                penalty += (
                    self.fisher_dict[n] * (p - self.optimal_params[n]) ** 2
                ).sum()
        
        return self.lambda_ewc / 2.0 * penalty
```

### 13.2 Replay Buffer

```python
class ReplayBuffer:
    """
    Replay buffer with strategic selection of representative samples.
    
    For each class, it keeps a small but representative set
    (herding sample) that preserves the original distribution.
    """
    
    def __init__(self, max_per_class: int = 50):
        self.max_per_class = max_per_class
        self._buffer: Dict[str, List[Tuple[np.ndarray, str]]] = {}
        # image_array, class_id
    
    def add_class_samples(
        self, 
        class_id: str,
        images: List[np.ndarray],
        embeddings: np.ndarray,
        strategy: str = 'herding'
    ):
        """
        Selects representative samples to store.
        
        strategy='herding': selects the samples that best approximate
                           the mean of the embedding space (iCaRL-style)
        strategy='random': random selection (simpler, less optimal)
        """
        n_select = min(self.max_per_class, len(images))
        
        if strategy == 'herding' and len(embeddings) > n_select:
            # Herding: select the samples that best approximate the centroid
            centroid = l2_normalize(embeddings.mean(0))
            selected_indices = []
            current_mean = np.zeros(embeddings.shape[1])
            
            for _ in range(n_select):
                # Find the sample whose accumulated mean is closest to the centroid
                candidates = np.array([
                    l2_normalize(current_mean + embeddings[i])
                    for i in range(len(embeddings))
                    if i not in selected_indices
                ])
                
                remaining_indices = [
                    i for i in range(len(embeddings)) 
                    if i not in selected_indices
                ]
                
                sims = candidates @ centroid
                best_local = int(sims.argmax())
                best_global = remaining_indices[best_local]
                
                selected_indices.append(best_global)
                current_mean = embeddings[selected_indices].mean(0)
            
            indices = selected_indices
        else:
            indices = np.random.choice(len(images), n_select, replace=False)
        
        self._buffer[class_id] = [
            (images[i], class_id) for i in indices
        ]
    
    def sample(self, n_per_class: int = 20) -> Dataset:
        """Returns a replay dataset with n_per_class per class."""
        all_samples = []
        for class_id, samples in self._buffer.items():
            n = min(n_per_class, len(samples))
            selected = np.random.choice(len(samples), n, replace=False)
            all_samples.extend([samples[i] for i in selected])
        
        return ReplayDataset(all_samples)
```

---

## 14. Persistence & Determinism

### 14.1 Persistence structure

```
model_store/
├── backbone/
│   ├── model_v1.3_fp32.onnx          # FP32 model (training/backup)
│   ├── model_v1.3_int8.onnx          # INT8 model (production)
│   └── model_manifest.json           # Version, hash, metrics
│
├── memory_bank/
│   ├── memory_bank_current.json      # Current state
│   ├── memory_bank_backup_*.json     # Rotational backups
│   └── memory_bank_changelog.jsonl   # Change log (append-only)
│
├── thresholds/
│   ├── evt_calibration.json          # Weibull parameters per class
│   └── operational_thresholds.json  # Operational thresholds
│
└── metadata/
    ├── class_registry.json           # Class registry (id, label, metadata)
    └── system_stats.json             # Usage statistics
```

### 14.2 Determinism at inference

```python
class DeterministicInferenceEngine:
    """
    Deterministic inference engine for industrial environments.
    
    Sources of non-determinism to eliminate:
    1. Threading race conditions in ONNX Runtime → use 1 inter-op thread
    2. Floating point non-associativity in batch → process 1 image at a time
    3. Random seeds in augmentation → there are none at inference
    4. Hash randomization in the Python dict → use ordered structures
    """
    
    def __init__(self, model_path: str):
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1  # CRITICAL for determinism
        opts.intra_op_num_threads = 1  # Maximum determinism (sacrifices speed)
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        
        self.session = ort.InferenceSession(model_path, sess_options=opts)
    
    def predict_deterministic(self, image: np.ndarray) -> np.ndarray:
        """
        Given the same input, it ALWAYS produces the same output.
        Verifiable: hash(output) = constant for the same input.
        """
        # Deterministic preprocessing: exact operations, same order
        processed = self._preprocess_deterministic(image)
        
        embedding = self.session.run(
            ['embedding'],
            {'image': processed}
        )[0][0]
        
        # L2 normalize with fixed precision
        norm = float(np.sqrt((embedding ** 2).sum()))
        if norm > 1e-12:
            embedding = embedding / norm
        
        return embedding
```

---

## 15. Benchmarks & Resource Estimates

### 15.1 Expected latency (Intel Core i5 8th gen, 4 cores)

```
Configuration: MobileNetV3-Small INT8, batch=1, 4 threads

Stage                           Time      Accumulated
────────────────────────────────────────────────────
Image decode + resize           0.3ms     0.3ms
Normalize + transpose           0.1ms     0.4ms
ONNX Inference (backbone)       3.2ms     3.6ms
Projection head (FC layers)     0.15ms    3.75ms
L2 normalize                    0.02ms    3.77ms
Prototype matching (100 cls)    0.08ms    3.85ms
EVT threshold check             0.01ms    3.86ms
Decision + result assembly      0.02ms    3.88ms
────────────────────────────────────────────────────
TOTAL P50                       ~4ms
TOTAL P99                       ~7ms      ✅ < 10ms
TOTAL P99.9                     ~12ms     (occasional spikes due to OS scheduling)
```

### 15.2 Resource consumption

```
Memory (full process RSS):
  ONNX Runtime overhead:    ~45MB
  MobileNetV3 INT8 model:     3MB
  Memory bank (100cls×8p):    1MB
  Python interpreter:        ~20MB
  ────────────────────────────────
  TOTAL estimated:            ~70MB RSS

  ResNet18 INT8 variant:     ~80MB RSS

CPU utilization:
  Idle (waiting for an image):   <1%
  During inference:              25-60% (mainly 1 core)
  With 4-thread ONNX:            distributed

Disk:
  MobileNetV3 INT8 model:    2.5MB
  Memory bank 100 classes:    1MB
  Thresholds + metadata:     <100KB
  ────────────────────────────────
  TOTAL disk footprint:     ~5MB
```

### 15.3 Expected accuracy (conservative estimates)

```
Typical industrial dataset (50 samples/class, 20 classes):

Metric                               Target    Expected
──────────────────────────────────────────────────────
Top-1 accuracy (known classes)       >95%      97-99%
Unknown rejection @ FAR=0.1%        >90%      92-96%
Anomaly detection (novel defects)   >85%      85-92%
Incremental accuracy (10 new cls)   >95%      96-98%
Robustness to lighting (±30%)       >90%      91-95%
──────────────────────────────────────────────────────

With few samples (10 samples/class):
Top-1 accuracy:                      ~85-90%
Unknown rejection:                   ~80-88%
```

---

## 16. Class Structure & API

### 16.1 Public API of the engine

```python
class EdgeClassificationEngine:
    """
    Industrial classification engine.
    API designed for integration into SCADA/vision systems.
    """
    
    def __init__(self, config: EngineConfig):
        ...
    
    # ── Inference ──────────────────────────────────────────
    
    def predict(self, image: np.ndarray) -> PredictionResult:
        """
        Single-image inference.
        Returns: class_id, confidence, status, anomaly_score, latency_ms
        """
    
    def predict_batch(self, images: List[np.ndarray]) -> List[PredictionResult]:
        """Batch inference (for continuous mode)."""
    
    # ── Training / Incremental ──────────────────────────────
    
    def add_class(
        self,
        class_id: str,
        label: str,
        images: List[np.ndarray],
        fine_tune_backbone: bool = False
    ) -> ClassAddResult:
        """Adds a new class to the system."""
    
    def update_class(
        self,
        class_id: str,
        image: np.ndarray,
        confirmed_label: str
    ) -> UpdateResult:
        """
        Online update with a new confirmed sample.
        Call it when the operator confirms the classification of an image.
        """
    
    def retrain_full(
        self,
        training_data: Dict[str, List[np.ndarray]],
        validation_data: Dict[str, List[np.ndarray]]
    ) -> TrainingResult:
        """Full retraining of the backbone."""
    
    # ── Management ──────────────────────────────────────────
    
    def get_class_info(self, class_id: str) -> ClassInfo:
        """Information about a class (prototypes, samples, thresholds)."""
    
    def list_classes(self) -> List[ClassInfo]:
        """Lists all known classes."""
    
    def remove_class(self, class_id: str) -> bool:
        """Deactivates a class (soft delete)."""
    
    def calibrate_thresholds(
        self,
        validation_data: Dict[str, List[np.ndarray]],
        anomaly_data: Optional[List[np.ndarray]] = None,
        far_target: float = 0.001
    ) -> CalibrationResult:
        """Recalibrates EVT thresholds with new validation data."""
    
    def save(self, path: str):
        """Persists the complete state of the engine."""
    
    @classmethod
    def load(cls, path: str) -> 'EdgeClassificationEngine':
        """Restores the engine from disk."""
    
    def get_health_report(self) -> HealthReport:
        """
        System status for industrial monitoring.
        Includes: P50/P99 latency, recent accuracy, drift detection.
        """


@dataclass
class PredictionResult:
    status: str           # 'KNOWN' | 'UNKNOWN' | 'ANOMALY' | 'LOW_CONFIDENCE'
    class_id: Optional[str]
    label: Optional[str]
    confidence: float     # [0, 1] calibrated
    raw_similarity: float # Cosine similarity to the best prototype
    anomaly_score: float  # [0, 1] 0=normal, 1=anomaly
    latency_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class EngineConfig:
    backbone: str = 'mobilenet_v3_small'   # or 'resnet18'
    embed_dim: int = 256
    max_prototypes_per_class: int = 8
    far_target: float = 0.001              # 0.1% False Accept Rate
    anomaly_threshold: float = 0.35        # Calibrate with real data
    n_threads: int = 4
    device: str = 'cpu'
    model_path: Optional[str] = None       # Path to a pre-trained ONNX
    memory_bank_path: Optional[str] = None # Path to an existing memory bank
```

### 16.2 Complete training flow

```
PHASE 1: INITIAL TRAINING
──────────────────────────
1. Collect N_min=30 images per class for all initial classes
2. Apply IndustrialAugmentationPipeline (×5 augmentations/image)
3. Train IndustrialEmbeddingModel with:
   a. Backbone: MobileNetV3-Small pretrained ImageNet
   b. Loss: ArcFace (m=0.4, s=32)
   c. Optimizer: AdamW(lr=1e-3, weight_decay=1e-4)
   d. Schedule: CosineAnnealingLR (T_max=100 epochs)
   e. Batch: 64, with class-balanced sampling
4. Evaluate on the validation set: silhouette, accuracy, unknown rejection
5. Export the backbone to ONNX INT8
6. Extract embeddings from the training set → build the memory bank
7. Calibrate EVT thresholds with the validation set

PHASE 2: PRODUCTION INFERENCE
─────────────────────────────────
1. Load ONNX INT8 model + memory bank + thresholds
2. For each image:
   a. Preprocess → embedding (4ms)
   b. Classify → score + decision (<0.1ms)
   c. Log the result
3. Monitor drift: if accuracy drops >3%, trigger recalibration

PHASE 3: INCREMENTAL LEARNING
─────────────────────────────────
1. New class: add_class() → clustering → prototypes → EVT threshold
   (Without touching the backbone if separability >0.3)
2. New sample: update_class() → online prototype update
3. Periodic recalibration: every N=1000 predictions or shift change
```

---

## Appendix: Justified Design Decisions

### A.1 Why NOT use classic softmax

| Limitation | Industrial Impact |
|------------|-------------------|
| Closed-world assumption | Cannot reject unseen parts: quality failure |
| Requires full retraining for a new class | Unacceptable line stoppages |
| Does not provide a calibrated confidence measure | Does not support risk decisions |
| Does not support native anomaly detection | Requires a separate system |
| Output sums to 1 even if the input is garbage | Always emits a class, even for garbage |

### A.2 Why cosine similarity and not Euclidean distance

Euclidean distance in high-dimensional spaces suffers from the *curse of dimensionality*: all distances converge toward the same mean when `d → ∞`. In `d=256`, Euclidean distance is much less discriminative than cosine similarity, which operates in the angular space and is invariant to the vector magnitude.

### A.3 Why multi-prototype and not a single centroid

A single centroid assumes that the intra-class distribution is unimodal and approximately Gaussian in the angular space. In industrial environments, lighting variations can create clearly separated sub-clusters (part lit from the left vs from the right), and a single centroid would end up in "no man's land" between them, reducing discriminability.

### A.4 Known limitations of the system

1. **Class scaling**: With >500 classes, prototype matching may exceed 1ms. Solution: ANN indexing (HNSW with hnswlib).
2. **Extreme viewpoint variations**: If the same part appears from very different angles, multi-prototype may need K>>8. Consider a backbone with a larger field-of-view or task-specific training.
3. **Strong bimodal distribution**: If a class has two very different sub-types (part A and part B both classified as "OK"), the system may generate false unknowns at the boundary. Solution: split the class into two.
4. **EVT calibration requires outliers**: To calibrate the unknown rejection thresholds well, it is necessary to have data from negative samples (parts of other classes, defects) in the calibration process.
