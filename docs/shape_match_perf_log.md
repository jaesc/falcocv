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
  vote. Replacing it with an inline helper removed that library-rounding cost.
- Added an inline round-half-away-from-zero helper in
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

### Visible-mask histogram median (2026-10-30)

- `_visible_mask` compares a warped uint8 patch against the uint8 template.
  The absolute difference is therefore integer-valued 0..255, so median and
  MAD can be computed exactly with two small histograms instead of sorting the
  float template.
- Fast path: `cv2.absdiff` + 256-bin `np.bincount`; doubled units keep
  half-integer medians exact. Non-uint8 inputs keep the old NumPy path.
- Synthetic math benchmark: ~2-3x faster for the real template sizes
  (120x253, 145x179, 140x167).
- Private 142-image job: Shape medians ~38.9/37.5/50.9 vs ~41.3/39.7/53.5 ms
  before this change; detections 140/140/142 and pose deltas exactly 0.
- Regression: full suite 276 passed, 3 skipped, 3 xfailed; wheel self-test
  PASS.
- Verdict: kept.

### Native score `pow(x, 2)` fast path (2026-10-30)

- `_ORIENTATION_EXPONENT` is 2.0, but the kernel called `std::pow` for every
  model feature and pose. Added a `positive_dot * positive_dot` fast path when
  the exponent is exactly 2.0; other exponents still use `std::pow`.
- Same-process score benchmark: 0.453 -> 0.328 ms (~28% faster); score arrays
  bit-identical.
- End-to-end gain is below the laptop's thermal noise, but the change is
  arithmetic-only and the 142-image gate keeps 140/140/142 with pose deltas 0.
- Verdict: kept.

### Fused bilinear sampling in score/ICP (2026-10-30)

- `shape_match_score` sampled `gx`, `gy` and `dt` at the same `(x, y)`, and
  `shape_match_icp` sampled `dt`, `gx`, `gy`, `ddx` and `ddy` at the same
  `(xs, ys)`. Each call recomputed `floor`, bounds, `wx` and `wy`.
- Added `BilinearCoords` + `bilinear_at`: compute the coordinates once, then
  apply the unchanged interpolation formula to each array.
- Score microbenchmark: 0.282 -> 0.162 ms (~1.7x faster); score arrays
  bit-identical.
- Private 142-image job: Shape medians ~44.9/44.3/62.9 -> ~39.8/39.6/53.0 ms
  (~11-16% faster); detections 140/140/142 and pose deltas exactly 0.
- Regression: full suite 276 passed, 3 skipped, 3 xfailed; wheel self-test
  PASS.
- Verdict: kept.

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

### ROI-superset monotonicity (2026-10-31)

Symptom: enlarging the search ROI (up to the full image) lost matches that a
smaller ROI found. Measured on the 90-image production job: 84/90 with the job
ROI, 69/90 with the full image, 19 images lost.

Root causes (all depended on the ROI extent, not on the image content):

- `_cap_scene_edges` divided a global target by the number of cells, so a bigger
  ROI lowered the per-cell quota (37 -> 8 edges) and the part kept 1,135 -> 307
  edges.
- `_vote_factor` chose the downscale from the ROI area (2x -> 4x).
- Pre-refine dedupe merged candidates at 0.75 x template (303 px); a false peak
  118 px away discarded the true candidate (rank 2) before refinement.
- Relative peak floor plus a fixed top-8 refine cap.
- The kernel kept only 3 peaks per hypothesis.

Fix:

- Vote factor from the template only (`min_template // 96`, max 4x).
- Fixed 48-edge per-cell quota, independent of the cell count.
- Coarse views cropped from a pyramid anchored to the frame origin: the same
  scene region always uses the same downscaled pixels, and the cost stays
  proportional to the ROI.
- Pose-aware dedupe (position + angle + scale).
- 32 peaks per hypothesis, up to 1024 kernel candidates.
- Two-stage NCC: one single-pose correlation ranks every proposal, then the
  full angle/scale scan runs on the best ones plus the strongest vote peaks.
- Alignment from one cached full-ROI gradient pair instead of a Sobel patch per
  proposal.
- Reduced ECC affine pass plus a rendered NCC translation at full resolution.
- No early stop on `max_matches`; all proposals are measured and the output is
  ranked by final score.

Measured (90-image job, template 404x407, 2588x1940):

| | Before | After |
|---|---:|---:|
| Job ROI detections | 84/90 | 90/90 |
| Full-image detections | 69/90 | 90/90 |
| ROI superset violations | 19 | 0 |
| Job ROI median | ~28 ms | ~44 ms |
| Full-image median | ~150 ms | ~99 ms |

Synthetic bench: 480x360 15.6 -> 55 ms, 1 MP 19.6 -> 104 ms, 2.7 MP 144 ms.
Bake-off: 13/13, 0.064 px / 0.060 deg / 0.080 %. Full suite 277 passed.

Verdict: kept. The extra cost is the number of hypotheses that must be
measured to guarantee the superset property; the full-image case got faster
because the coarse vote is 4x cheaper.

### Linear dedupe and local ECC view (2026-10-31)

Two costs found while profiling the monotonicity fix:

- `_dedupe_candidates` compared every candidate with every kept one (O(n^2)).
  With 32 peaks per hypothesis it reached ~13 ms on a 5 MP frame.
- `_pyramid_views` built the ECC view (factor 2) for the whole frame even
  though the ECC only samples a template-sized patch.

Changes:

- Dedupe candidates on a grid of `min_distance` cells and compare only the
  eight neighbouring cells.  Identical output (proved by a brute-force
  equivalence test) and ~1 ms instead of ~13 ms on 5 MP.
- Build the ECC view lazily from a window around the candidate, anchored to
  the same frame grid (`left = ((ax + wx) // factor) * factor`) and sourced
  from the frame.  The overlapping pixels are byte-identical to the full view
  (verified with `np.array_equal`), so the ECC input does not depend on the
  ROI.  ~0.1 ms instead of a full-frame f2 resize.
- Measured deltas vs the previous pipeline on the 90-image job: max 0.0003 px,
  0.00003 deg, 1e-6 scale (floating-point noise only).

ECC at factor 3 was evaluated and **rejected**: compared with factor 2 on the
same 90 images it moved poses by up to 0.022 px, 0.021 deg and 0.065 % scale,
above the 0.02 px / 0.01 deg / 0.02 % acceptance gate.  Keep factor 2.

Result: full 5 MP ~90 -> ~85 ms, job ROI ~42 -> ~44 ms (noise), 90/90 both,
bake-off unchanged (0.064 px / 0.060 deg / 0.080 %).  278 tests.

### Exact-multiple pyramid crops (2026-10-31)

`cv2.resize(..., INTER_AREA)` drops to a slow generic path when the source
size is not an exact multiple of the factor.  For a 2588x1940 frame the 8x
view (323.5) cost 9.2 ms while the 4x view cost 0.5 ms.

Fix: crop the source to an exact multiple of the factor before resizing (the
dropped pixels are at most `factor-1` at the far frame edge).  Full-frame
pyramid: 9.9 -> 1.8 ms.  5 MP job: ~85 -> ~73 ms.  Bake-off unchanged
(0.064 px / 0.060 deg / 0.080 %); two occlusion-path images moved 0.13 deg
because the grid is now exactly 1/8 instead of 0.1252.

### Render cache and rejected micro-optimizations (2026-10-31)

New harness `benchmarks/bench_shape_match_matrix.py`: every case runs with the
candidate configurations in alternating rounds (thermal drift does not favour
one), on 480x360, 1 MP, 2.7 MP and a 5 MP scene with a 404 px template.

- `cv2.setNumThreads(1)` around the run helps 480x360 (-10 %) and 1 MP (-6 %)
  but hurts the 5 MP / 404 px case (+9 %).  Not generic: **rejected**.
- Building the translation refine from `_render_search` (cropped) instead of
  the diagonal render moved poses by up to 0.47 px.  **Rejected**: precision
  is not negotiable for a few milliseconds.
- Render cache by (template, angle, scale) per run: 103 `_render_search` calls
  become 36 warps on the 5 MP case, ~3 ms saved, bit-identical pixels (it is
  the same render).  **Kept**.
- Reducing the full NCC scans from 6 to 4 and the candidate cap from 10 to 8
  kept 90/90 and the 13 synthetic scenes, but the saving was inside the noise
  and it shrinks the safety margin.  **Not applied** for now.

### Three-sample NCC scans (2026-10-31)

- Angle scan: 5 samples at 4 deg over +-8 -> **3 samples at +-5 deg**.  The
  kernel angle step is capped at 10 deg, so the truth is always inside +-5.
- Scale scan: 5 samples at 0.05 over +-0.10 -> **3 samples at +-half**, with
  `half = max(0.05, kernel_spacing / 2)`.  The half-width follows the kernel's
  scale-hypothesis spacing, so wide user ranges (0.5-1.5) are still covered.
- 6 `matchTemplate` calls per candidate instead of 10.

Validation:

- Bake-off: 13/13, 0.063 px / 0.034 deg / 0.081 % (rotation max improved from
  0.060 to 0.034 deg; scale moved +0.001 %).
- 90-image job: 90/90 in ROI and full image; pose deltas stay inside the
  pyramid-reference band.
- Wide-scale sweep 0.6-1.4 with a 0.5-1.5 range: <= 0.04 % scale, <= 0.017 deg.
- Interleaved matrix: 37.9 / 70.1 / 100.1 / 52.7 ms (480x360, 1 MP, 2.7 MP,
  5 MP with a 404 px template).

Verdict: kept.

### Vote-feature budget 400/300 (2026-10-31)

Tested lowering `_VOTE_MODEL_FEATURES` from 500 to 400 and 300:

- In-process A/B on the 5 MP / 404 px case: 500 = 54.5 ms, 400 = 53.3 ms,
  300 = 55.1 ms.  The ~1 ms difference is inside the run-to-run noise.
- 90-image job: 90/90 in both ROI modes with the same poses as the 500 run.
- Bake-off: bit-identical summary (0.063 px / 0.034 deg / 0.081 %).
- Verdict: **reverted to 500**.  A 20 % reduction of the vote features is a
  robustness margin (the 300 case already broke a bright-border synthetic
  scene in an earlier iteration) and the measured gain does not justify it.

### Scene-edge quota 32 (2026-10-31)

Lowering `_VOTE_EDGE_QUOTA` from 48 to 32 kept 90/90 and the bake-off summary
bit-identical, but the in-process A/B showed no gain:

- 2.7 MP: 45.7 -> 46.4 ms
- 5 MP / 404 px: 54.2 -> 56.4 ms

The weaker vote does not reduce the rest of the pipeline, so the saved kernel
work is lost (and the run-to-run noise is larger than the difference).
**Reverted to 48**: it keeps the stronger vote for the same wall time.

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

### Integer offsets in the native vote

- Tried: replace the per-vote `fast_round(px - rx)` with a precomputed integer
  offset (`px - ceil(rx - 0.5)`), falling back to exact rounding for features
  whose offset is at or near a half-integer.
- Result: candidates stayed exact, but same-process A/B gave 18%, 14% and -6%
  across the three Shapes. No consistent gain.
- Verdict: reverted. `fast_round` is already cheap; the remaining loop cost is
  the scatter itself.

### Precomputed score trigonometry

- Tried: precompute `cos/sin(model_theta)` and use angle-addition identities
  instead of `std::cos(expected)`/`std::sin(expected)` per feature and pose.
- Result: ~30% faster score microbenchmark, but scores changed by ~6e-9 and
  the 142-image poses moved by up to ~2.4e-4 px / ~2.5e-4 degrees.
- Verdict: reverted. Kept only the exact `pow(x, 2)` fast path.

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
   candidates; real job 41.3/39.7/53.5 ms with unchanged detections/poses.
5. `visible_mask` histogram median and native `pow(x, 2)` — **done and kept**:
   2-3x faster median math, 28% faster score microbenchmark, bit-identical
   scores and 140/140/142 detections with pose deltas 0.
6. Fused bilinear sampling in score/ICP — **done and kept**: score
   0.282 -> 0.162 ms, bit-identical; real job 39.8/39.6/53.0 ms with
   140/140/142 detections and pose deltas 0.
7. Re-run the 142-image private job as the gate: no detection or pose changes.

Out of scope for this phase: workflow-level parallel execution, Fast mode,
changing core count heuristics, or changing feature/edge/factor constants.

## Privacy

This document contains only aggregated timings, counts and parameter values. It
must not contain customer images, customer paths, image names, embeddings or any
other client data.
