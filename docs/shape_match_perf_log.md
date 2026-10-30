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

After the inline-rounding fix, the native vote is ~7 ms per Shape and the
remaining cost is distributed across NCC/`matchTemplate`, coarse pose and
occlusion refinement.

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

### Native ICP kernel (2026-10-30)

- Added `shape_match_icp` to `_falco_kernel`: bilinear sampling, robust
  orientation/distance weights and closed-form 2D similarity (no SVD, no
  OpenCV). Python still computes `ddx`/`ddy` with OpenCV and passes arrays.
- Native vs Python: max delta ~1.4e-5 px, ~3e-6 degrees, ~3.5e-8 scale.
- Microbenchmark: ~2.1x faster than the NumPy ICP.
- Private 142-image job, compared with native score only:
  - Shape_0: ~60.1 -> ~54.7 ms
  - Shape_1: ~48.6 -> ~43.0 ms
  - Shape_2: ~64.9 -> ~57.7 ms
  - Detections unchanged (140/140/142); pose deltas < 0.0003 px.
- Combined native score + native ICP vs original baseline:
  - Shape_0: ~133.5 -> ~54.7 ms (~2.4x)
  - Shape_1: ~107.7 -> ~43.0 ms (~2.5x)
  - Shape_2: ~136.8 -> ~57.7 ms (~2.4x)
- Native scan was evaluated but skipped: `_dt_scan_axis` costs ~0.666 ms vs
  ~0.655 ms for the score alone, so Python orchestration overhead is negligible.
- Regression test added: `test_shape_match_native_icp_matches_python`.
- Verdict: kept (pending user validation).

### Native vote inline rounding (2026-10-30)

- The native vote profile showed accumulation dominated, but the cost was not
  the response-buffer memory: the inner loop called `std::lround` twice per
  vote, and MSVC did not inline it.
- Replaced it with an inline round-half-away-from-zero helper in
  `shape_match_vote` (`fast_round`). Coordinates are bounded, so the helper is
  equivalent to `std::lround` for this loop.
- Captured real vote inputs (one image, three Shapes), 20-25 calls each:
  - kernel median: 26.4 / 15.5 / 17.2 ms -> 8.7 / 7.9 / 9.4 ms
    (~3.0x / 2.0x / 1.8x; absolute values vary with laptop throttling);
  - candidate arrays (`x`, `y`, `angle`, `scale`, `votes`) bit-identical.
- Private 142-image job, compared with native score + ICP:
  - Shape_0: ~54.7 -> ~41.3 ms
  - Shape_1: ~43.0 -> ~39.7 ms
  - Shape_2: ~57.7 -> ~53.5 ms
  - Detections unchanged (140/140/142); pose deltas exactly 0.
- Synthetic benchmark medians (current vs the approximate baseline table
  above): 480x360 23.4 vs ~26 ms; 1 MP 21.9 vs ~24.5 ms; 2.7 MP 34.2 vs
  ~37.7 ms; occlusion 20% 25.5 vs ~40 ms; occlusion 30% 27.1 vs ~40 ms;
  combo 26.5 vs ~37 ms; two instances 33.8 vs ~46 ms; angle +-90 21.6 vs
  ~33 ms; angle +-180 23.9 vs ~38 ms; fixed scale 17.3 vs ~20 ms.
- Regression: full suite 276 passed, 3 skipped, 3 xfailed; wheel self-test
  PASS.
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

### Native vote profile (accumulation vs peaks)

- Added a temporary optional `profile` mode to `shape_match_vote` and measured
  one image of the 142-image job per Shape (aggregated numbers only):
  - scene bins: ~0.25 ms
  - accumulation: 31-75 ms CPU (summed across workers)
  - local-maxima/top-K: 1-6 ms CPU
  - merge/sort: <0.02 ms
  - kernel wall time: 11-27 ms
- Conclusion: accumulation dominates by an order of magnitude; peak extraction
  is not the bottleneck. This explains why the max-filter and 3x3 experiments
  could not help.
- Temporary profile code was reverted after the measurement.

### Remove the `seen` buffer during accumulation

- Tried: drop the per-worker `seen` byte buffer and detect the first vote with
  `response[index] == 0.0f`.
- Result: identical detections and poses on all 142 real images, but no
  speedup (measured slower in the same session; thermal noise was significant).
- Verdict: reverted. The dense random-scatter response write, not the `seen`
  lookup, is the cost.

### Scene-outer vote accumulation

- Tried: invert the vote loops to scene point -> model features, so all writes
  for one point target a small response neighbourhood.
- Result: identical detections and poses on 142 real images, but ~1.5x slower.
  The existing model-feature -> scene-point order already writes in scan order
  and is more cache-friendly.
- Verdict: reverted.

### Blocked / tiled vote accumulation

- Tried: block the scene-point loop (sizes 1-256) and bucket votes into
  response tiles (32-512 px) before accumulation.
- Result: exact or near-identical candidates, but no consistent speedup
  (differences were within thermal noise, ~3-5% at best).
- Verdict: reverted. The expensive part was the rounding call, not the
  response-buffer access pattern.

### 3x3 peak neighbourhood

- Tried: reduce the native peak local-maximum neighbourhood from 5x5 to 3x3.
- Result: identical detections and poses on 142 real images, but slower
  (~0.66-0.73x of baseline in the same session).
- Verdict: reverted.

### Coarse angle step 15 degrees

- Tried: increase the coarse vote angle step from 10 to 15 degrees, reducing
  angle hypotheses by ~29%.
- Result: identical detections and poses on 142 real images, but no speedup;
  measured slower in the same session.
- Verdict: reverted. The NCC refinement cost and candidate distribution offset
  the smaller vote.

### Native NCC feasibility

- A direct C++ NCC over the search window is O(template_pixels * offsets) and
  would not beat OpenCV's `matchTemplate` (which uses optimized/DFT paths)
  without a much larger FFT/SIMD effort.
- Verdict: keep OpenCV `matchTemplate`; do not port NCC directly.

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
2. Native ICP — **done, validated and committed** (`8d21a8e`). Native scan was
   measured and skipped because its overhead is ~0.01 ms.
3. Native NCC — **not pursued**: a direct C++ implementation cannot beat
   OpenCV `matchTemplate` without a much larger FFT/SIMD effort. Keep OpenCV.
4. Native vote profile — **done**: accumulation dominates (31-75 ms CPU vs
   1-6 ms CPU for peak extraction). Inline rounding — **done and kept**:
   kernel 26.4/15.5/17.2 -> 8.7/7.9/9.4 ms on captured inputs, bit-identical
   candidates; real job 41.3/39.8/53.4 ms with unchanged detections/poses.
5. Re-run the 142-image private job as the gate: no detection or pose changes.

Out of scope for this phase: workflow-level parallel execution, Fast mode,
changing core count heuristics, or changing feature/edge/factor constants.

## Privacy

This document contains only aggregated timings, counts and parameter values. It
must not contain customer images, customer paths, image names, embeddings or any
other client data.
