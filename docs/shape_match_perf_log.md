# ShapeMatch performance log

Persistent record of ShapeMatch performance work. Keep this file updated when a
new optimization is attempted. Record **what was tried, what was measured, and
why it was kept or reverted** so future sessions do not repeat dead ends.

## Environment

- Date: 2026-10-30
- Branch: `dev`
- Baseline commit: `dc4f3b4` (`perf(shape-match): speed up voting and refinement`)
- CPU: Intel Core i5-8250U (4 cores / 8 logical threads)
- Python: 3.13.x
- OpenCV: 5.x
- Windows x86_64

The benchmark machine is a laptop and throttles under sustained load. Absolute
milliseconds vary; compare variants in the same session and prefer medians/minima.

## Baselines

### Synthetic benchmark

`benchmarks/bench_shape_match.py` on the baseline commit (approximate medians):

| Case | Time |
|---|---:|
| 480x360 | ~26 ms |
| 1 MP | ~24.5 ms |
| 2.7 MP | ~37.7 ms |
| Occlusion 20% | ~40 ms |
| Occlusion 30% | ~40 ms |
| Combo (occlusion + noise + gamma) | ~37 ms |
| 2 instances | ~46 ms |
| Angle +-90 | ~33 ms |
| Angle +-180 | ~38 ms |
| Fixed scale | ~20 ms |

### Private real job

A private customer job with 142 images (2592x1944), one image loader, three
ShapeMatch tools and one DataMatrix barcode. No images, names or paths are stored
here. Aggregated baseline on the same machine:

| Tool | Median | Detections |
|---|---:|---:|
| Shape Match_0 | ~110-130 ms | 140 / 142 |
| Shape Match_1 | ~100-110 ms | 140 / 142 |
| Shape Match_2 | ~130 ms | 142 / 142 |
| Barcode | ~5 ms | - |
| Full job | ~340-450 ms | - |

Phase profile on a representative image:

- native vote: 17-56 ms (largest cost, especially Shape_0);
- occlusion refinement: 15-42 ms;
- NCC / `matchTemplate`: 8-22 ms;
- ECC: 2-5 ms.

`Shape Match_2` is the canary: it detects 142/142 at baseline.

## What worked

### Native score kernel (2026-10-30)

- Added `shape_match_score` to `_falco_kernel`: bilinear sampling plus the
  weighted orientation/distance score, with no OpenCV dependency.
- Python still prepares `gx`, `gy`, `dt` and the model arrays; the kernel does
  the per-pose/per-feature math.
- Native vs Python score: maximum difference ~5e-8 on synthetic data.
- Microbenchmark: 2.8-4.7x faster than the NumPy path depending on pose count.
- Private 142-image job: same detections (140/140/142), pose deltas < 0.0005 px
  and < 0.0001 degrees, ShapeMatch medians:
  - Shape_0: ~133.5 -> ~60.1 ms (2.22x)
  - Shape_1: ~107.7 -> ~48.6 ms (2.22x)
  - Shape_2: ~136.8 -> ~64.9 ms (2.11x)
- Regression: full suite 275 passed; wheel self-test PASS.
- Verdict: kept. This is the current native baseline.

### Kept in `dc4f3b4`

- Native vote parallelized over angle/scale hypotheses with per-thread response
  buffers.
- Sparse touched-cell response accumulation and peak extraction.
- `cv2.spatialGradient` plus edge-only magnitude/orientation instead of two
  Sobel passes plus a full `cv2.magnitude`.
- `countNonZero` fast path in `_cap_scene_edges`.
- Cheaper NCC translation search.
- ECC translation polish, skip polish when correlation is already very high,
  Euclidean motion for fixed scale.
- Occlusion cleanup: skip redundant ICP/NCC work and coarser scans that still
  pass the synthetic accuracy suite.
- Extended benchmark with 2.7 MP, occlusion, multi-instance, wide-angle and
  fixed-scale cases.

## What did not work (do not repeat)

### Sub-pixel dipoles + uniform probes

- Tried: parabolic sub-pixel edge positions and chain-sampled probes replacing
  the top-gradient feature model.
- Result: synthetic accuracy not better; occlusion angle worse; slower.
- Verdict: reverted. The existing vote/ICP pipeline does not exploit sub-pixel
  model positions well enough.

### Geometric field solver using Canny dipoles

- Tried: template distance/vector field plus point-to-line Gauss-Newton.
- Result: visible parts ~0.3 px worse, occlusion up to ~1.2 px worse; slower.
- Verdict: reverted. Canny correspondences are not reliable enough.

### Caliper-based field solver

- Tried: reuse the Edge tool's 1D derivative-of-Gaussian + parabolic sub-pixel
  peak to generate correspondences along model normals.
- Result: 300-400 ms per Shape, ~0.5-0.7 px translation bias.
- Verdict: reverted. The caliper math is fine for 1D measurements, but this
  integration has a coordinate bias and is far too slow.

### Kernel max-filter / top-K peak extraction

- Tried: replace the 5x5 neighbour scan with a separable sliding-window maximum
  filter and a fixed-size top-K list.
- Result: exactly identical detections and poses on 142 real images, but no
  speedup; sometimes slower.
- Verdict: reverted.

### Parameter reductions on the real job

Tested on all 142 real images (aggregated, no images stored):

| Variant | Result |
|---|---|
| 300 vote features | no clear speedup, pose outliers |
| edge cap 3000 | no clear speedup |
| vote factor 3 | loses 2-3 detections on Shape_2 |
| cap 2000 + 300 features + factor 3 | 20-40% faster, but Shape_2 139/142 and pose outliers |

Verdict: none is acceptable as a default. The current 500 features / 4000 edge
cap / factor 2 is well tuned for this job.

### NCC micro-tuning

- `margin=10/search=8` instead of 12/12: no measurable gain.
- 4-point angle scan: slight scale/pose changes; not worth the risk.
- 4-point scale scan: failed occlusion scale tolerance.
- Verdict: reverted.

### Occlusion scan reductions

- Fewer initial angle/scale poses and half-model coarse scans passed synthetic
  tests, but did not produce a perceptible end-to-end gain.
- Verdict: not kept; the real bottleneck is the native vote and the cost of the
  refinement path, not these few poses.

## Current decision

Native C++ acceleration of ShapeMatch scoring/occlusion (path A) is in progress.
Keep `_falco_kernel` free of OpenCV; Python prepares NumPy arrays and the kernel
performs the hot math.

Status:

1. Native pose scoring (`shape_match_score`) — **done and kept**.
   - Native vs Python max difference ~5e-8.
   - Microbenchmark 2.8-4.7x faster.
   - Private 142-image job: 2.1-2.2x faster, identical detections, pose deltas
     < 0.0005 px.
   - Regression test added: `test_shape_match_native_score_matches_python`.
2. Native occlusion scans / ICP / pose update — next candidate.
3. Native NCC prototype only if it can beat `cv2.matchTemplate` in a fair
   microbenchmark; otherwise keep OpenCV.
4. Re-run the 142-image private job as the gate: no detection or pose changes.

Out of scope for this phase: workflow-level parallel execution, Fast mode,
changing core count heuristics, or changing feature/edge/factor constants.

## Privacy

This document contains only aggregated timings, counts and parameter values. It
must not contain customer images, customer paths, image names, embeddings or any
other client data.
