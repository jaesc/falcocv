# FalcoCV — Context Summary

Industrial machine vision framework. Python + PyQt6 + OpenCV 5, with a small
native kernel for the two measured C++ hot paths.

## Architecture

```
src/falcocv/
  core/        → workflow engine, VisionImage, overlays, ROI, depth/3D helpers
  tools/       → vision tools: logic.py (processing) + config.py (PyQt6 widget)
  ui/          → PyQt6 widgets, viewers, ROI editors, HMI designer
  web/         → FastAPI + HMI worker subprocesses + dashboard
  main.py      → Vision Editor entry point
  cli.py       → `falcocv` / `python -m falcocv`
  version.py
cpp/kernel/    → _falco_kernel (pybind11, NO OpenCV):
                 · generate_pointcloud
                 · plane_fit (LSQ/RANSAC)
tests/         → pytest suite
cpp/bin/       → local kernel build output (gitignored)
```

### Rules
- **FalcoCV is the product.** All tool work, fixes, benchmarks and discussions
  are framed at the tool/framework level: general behavior, parameters and
  regressions. A job, dataset or image is only a way to reproduce a general
  behavior; never tune or design around a specific job unless the user
  explicitly asks for a job-specific change.
- Use **Python/OpenCV/NumPy** for tools unless a measured benchmark shows a
  meaningful (>1.5× end-to-end) win.
- C++ is reserved for **point-cloud generation**, **plane fitting** and
  measured shape-matching hot paths. Any new C++ hot path must ship with a
  benchmark and live in `cpp/kernel`, not in the old `_cv_core` architecture.
- The kernel exposes **dict→dict / ndarray functions**, never `BaseTool`,
  `VisionImage`, overlays or `data_changed` logic.
- `_falco_kernel` must not depend on OpenCV. OpenCV comes from
  `opencv-python` for Python code.
- Models are **not** in the wheel. `core/models.py` resolves/downloads them on
  first run (`%APPDATA%/FalcoCV/models`, `$XDG_DATA_HOME/FalcoCV/models`).

## Key pitfalls (do not repeat)

- **Point-cloud Z convention** → `z = depth * z_res + z_offset` (not minus).
  X is mirrored: `x = (img_width - 1 - col) * x_res` so image-left maps to
  world +X and the default isometric 3D camera looks natural.
- **Point-cloud stride** → `step = ceil(sqrt(total_pixels / target_points))`;
  a Python fallback lives in `core.pointcloud_utils.generate_pointcloud`.
- **PlaneFitter ROIs** → ROIs are stored in fixture coordinates and converted
  to absolute image coordinates before calling the kernel. The kernel converts
  image centers to world coordinates and negates the angle (Qt clockwise
  convention vs world axes).
- **Plane polygon Z range** → use the depth range of the **whole image**, not
  the ROI, so the overlay spans the image.
- **Overlay flow** → `inherit_overlays_from` copies layers (no shared lists).
  Correct flow: inherit → claim the layer with
  `get_overlay_layer("default", source_tool=self.name)` → clean elements whose
  `source_tool` is this tool → add new overlays → `_copy_overlays_to_input`.
  Never accumulate. Claiming is mandatory: `add_roi()/add_contour()/add_text()`
  copy `layer.source_tool` into each element, and `_copy_overlays_to_input`
  filters by `element.source_tool`, so an unclaimed inherited layer tags the
  new overlays with the upstream tool's name and they are silently dropped.
  `composite_roi_to_full(..., source_tool=...)` does the same for ROI borders.
- **Overlay rendering** → render only in visible viewers; never rasterize into
  the image buffer; keep overlays as mathematical primitives.
- **ROI helpers** accept 4- or 5-element ROIs (`[x,y,w,h]` or `[x,y,w,h,a]`).
- **ROI crops with rotated fixtures** → `get_roi_data()` and
  `composite_roi_to_full()` must resolve the ROI origin with
  `fixture_to_image()`. An upright ROI drawn on a rotated fixture is stored as
  `ra = -fixture.angle`, so `total_angle == 0` while the fixture still moves
  the ROI; `x + rx` is only valid when the fixture has no rotation. The
  axis-aligned path slices at `int(fixture_to_image(...)[0:2])` (no resampling)
  so crop and composite round-trip losslessly.
- **Filters on depth** → scrub NaNs, preserve float32 output; Canny/CLAHE/
  equalize normalize float32 to uint8 internally; invert float32 with
  `(max+min)-img`.
- **Histogram on depth** → support float32/NaN with a finite min/max range;
  never call `np.iinfo` on a float dtype.
- **Barcode Auto** → try the fast 1D path first (downscaled pyzbar, then
  zxing), then QR/DataMatrix/ArUco only if needed. zxing MicroQR enum names
  differ between versions; look them up dynamically.
- **DataMatrix** → pylibdmtx uses a bottom-left origin; flip contour Y only.
- **3D ROIs** → stored in fixture coordinates; convert with
  `image_to_fixture()` before storing and `fixture_to_image()` before drawing.
- **3D ROI handles** → large, filled, drawn without depth test; add a center
  handle so they remain usable in dense clouds.
- **3D draw fallback** → if point-picking and plane back-projection both fail,
  show a tooltip and abort; never keep an old ROI silently.
- **`zero_as_invalid` default True** → depth `0` is invalid for point clouds;
  avoid phantom floor planes. Depth images displayed in 2D are normalized to
  0 after cloud generation.
- **3D camera** → default isometric camera `azimuth=45°, elevation=30°`.

- **ShapeMatch solvers** -> the native kernel is required; there is no Python
  coarse fallback. Alignment >=0.90 uses the ECC solver; lower alignment uses the
  explicit occlusion solver. ECC failure in the 0.90-0.98 band reclassifies to
  occlusion; >=0.98 is rejected with a warning. Results carry `solver`,
  `ncc_score`, `visible_fraction`, `ecc_cc` and `warnings`.
- **ShapeMatch model** -> `model_b64` stores features, edges, gray and color
  templates. Training ROI is fixture-relative and supports rotation. Do not add
  TemplateMatch-only parameters (`angle_step`, `scale_step`, `use_pyramid`, ...).
- **ShapeMatch training model v3** -> `model_b64` is v3 only; v2 models are
  rejected and must be retrained. The model stores the training mask and
  `edge_low`/`edge_high`/`min_gradient`/`max_features`. Changing any of those,
  the training ROI, or the training image marks the model **STALE** and blocks
  matching until retrained. The Canny→top-10 % Sobel fallback is allowed but
  shown in the training status.
- **ShapeMatch feature coverage** -> automatic training seeds one feature per
  active 32 px cell, then fills the remaining `max_features` budget by gradient
  magnitude with an adaptive per-cell cap (`ceil(2 * budget / active_cells)`).
  The same selection is used for the 500-feature vote subset and for
  `reset_auto_selection`. It must not silently reduce the feature budget.
- **ShapeMatch manual feature editing** -> the editor works on the stored
  Canny/fallback candidates after mask and `min_gradient`. The fixed
  `max_features` budget is enforced; manual flags are stored in the model as
  provenance only. The final selected set is authoritative: manual features do
  not get global priority in the vote subset, and a manual feature in a covered
  cell competes by the same coverage/magnitude rule as an automatic one.
  Changing any training parameter, ROI, mask or training image invalidates the
  manual selection. The normal preview only shows the final selection.
- **ShapeMatch coarse voting** -> the vote factor depends on the template only
  (`min_template // 96`, capped at 4x), never on the ROI area.  Cap scene edges
  with a fixed **48-edge quota per 32 px cell**, independent of the cell count;
  a global target divided by cells starves the part as the ROI grows.  Keep 32
  peaks per hypothesis (up to 1024 candidates) and use up to 500 model
  features selected by the same spatial coverage + magnitude rule.  Full
  gradients, Canny and the distance map stay lazy.
- **ShapeMatch search stage** -> two-stage NCC on a reduced view (template
  >=48 px, max 8x): one single-pose correlation ranks every proposal, then the
  full angle/scale scan runs on the best ones plus the strongest vote peaks.
  The pre-gate is capped at 20 proposals and the full scan at 4 angle/scale
  proposals; a score-margin cutoff keeps close competitors and at least two
  candidates. Candidates whose NCC is below `_MIN_NCC` are rejected before
  alignment/solver. The alignment score uses one cached full-ROI gradient pair,
  not a Sobel patch per proposal.
- **ShapeMatch geometric pre-gate prototype** -> the per-tool
  `pre_gate_scorer` parameter defaults to `"ncc"` and is exposed in the Search
  tab.  `"alignment"` uses the native gradient-alignment score and passed the
  full suite; `"edge"` failed multi-instance tests and is not exposed.  The
  full scan and ECC still use grayscale, so this is not yet full illumination
  invariance.
- **ShapeMatch geometry-only mode** -> the per-tool `match_mode` parameter
  defaults to `"geometry"`; `"hybrid"` remains selectable as a legacy fallback.
  Geometry skips `matchTemplate` and ECC, using edge distance, gradient
  alignment and ICP.  Existing jobs with an explicit `"hybrid"` value keep it,
  and jobs that predate the field migrate to `"hybrid"` on load (including the
  current `test1`/demo jobs until their tools are changed).  After score-sorting
  the geometry shortlist it matches every hybrid detection on `test1` and adds
  one Shape Match_2 image.  Geometry refinement now uses a local edge/distance window
  when it is much smaller than the ROI, and the native kernel batches
  fixed-angle/scale translation scoring plus angle/scale axis scans.  Full
  geometry gradients use `cv2.spatialGradient` on `uint8` (exactly equivalent
  to the two Sobel calls and faster).  The native ICP computes its distance-map
  derivatives internally (`shape_match_icp_from_dt`), avoiding Python Sobel
  calls while keeping the old entry point.  With the rebuilt kernel, the latest
  `test1` medians were `Shape Match_1` 50.9 vs 70.7 ms, `Shape Match_0`
  68.9 vs 72.8 ms and `Shape Match_2` 100.7 vs 92.6 ms (timings are noisy).
  Pose deltas are still up to ~7 px / ~2 deg / ~1.5 %, so it remains
  experimental.  The production gate ran on 1005 real frames: detection parity
  (0 lost, 0 extra) and -31 % time, but translation is less precise than hybrid
  (median 1.00 vs 0.65 px against the job's TemplateMatch, one 85 px failure).
  Geometry stays the fast mode; hybrid is the precision reference.  Re-run
  `scripts/validate_shape_match_modes.py` before changing that verdict.
  The Python path falls back to the generic score when the loaded kernel
  predates `shape_match_score_translations`/`shape_match_score_axis`, and to
  the old ICP path when it predates `shape_match_icp_from_dt`.
- **ShapeMatch staged translation grid** -> the widest translation search runs a
  coarse grid and then refines around its maximum.  The coarse step must be an
  integer multiple of the fine step; otherwise the refined poses leave the
  lattice of the single full grid and the result changes.  With the multiple the
  staged search is bit-identical (pose delta 0 on the synthetic suite and on the
  job A/B) and drops 221 -> 161 poses per solve.
- **Kernel A/B measurement** -> rebuilding between measurements moved the
  baseline by 13 % on this machine.  Compare native changes inside one binary
  (extra flag or a second exported function), never across builds.
- **DepthRangeLoader TIFF path** -> prefer `imagecodecs.tiff_decode`, then
  `cv2.imread`, then `tifffile`.  On the large `20000x4096` LZW depth frames
  the imagecodecs path is faster than OpenCV (~375-410 ms vs ~420-480 ms in
  memory); do not add a frame cache for production runs.  Integer depth is
  converted with `cv2.addWeighted(..., dtype=CV_32F)`, which combines casting
  and scaling and is faster than `astype` + multiply; `float32` input keeps the
  existing no-copy path when `scale == 1.0`.
- **FCD depth container** -> `.fcd` is a FalcoCV-specific raw-depth container,
  not TIFF.  It stores shape/dtype in a small JSON header and an LZ4-compressed
  uint16 payload.  It is smaller than raw and much faster to decode than LZW
  TIFF (~48 ms vs ~375 ms for the large frames).  `DepthRangeLoader` detects
  `.fcd` by extension; `scripts/convert_depth_to_fcd.py` converts TIFF batches
  with verification.  Preserve the TIFF path for existing files.  On the
  `20000x4096` test frame the loader is ~240-305 ms with `.fcd` vs ~607 ms with
  TIFF when generating 3D; the full job is ~585-700 ms vs ~957 ms.
- **Point-cloud colors are lazy** -> loaders keep `_3d_data["colors"] = None`;
  `ensure_pointcloud_colors(vimg)` generates them only when a 3D viewer uploads
  the cloud.  This avoids ~47 ms per frame in headless/production jobs.
- **PlaneFitter fixed-Z ROIs** -> when every ROI has a fixed Z range, the
  native plane fit skips the full 82 MP image-range scan; it samples the image
  every 32 pixels only for the cover polygon.  The plane and polygon remain
  identical on `depth.cvj`; the tool drops from ~440 ms to ~120 ms.  If a ROI
  uses the automatic `0..1` Z sentinel, the full scan is retained.
- **ShapeMatch/TemplateMatch range input** -> 2D float range images are
  normalized through a grayscale view. `range_vmin`/`range_vmax` are computed
  automatically from all valid pixels of the full training image and persisted;
  inference uses those fixed values. For an axis-aligned ShapeMatch range ROI,
  use the already-normalized ROI directly: never call `cv2.normalize` on the
  complete frame, because that discards the training limits and costs ~120 ms.
  The fast ROI normalization uses `cv2.addWeighted` and an in-place clip; this
  is exact against the previous NumPy expression.
  The output keeps the original range data, calibration and `_3d_data` by
  reference, and sets the 2D fixture when `act_as_fixture` is enabled. No
  reference plane is required for this matching.
- **Workflow port types** -> every tool declares its image and plane ports as
  class attributes (`input_port_types` / `output_port_types` with `IMAGE` /
  `PLANE`); unlisted ports are `ANY`.  Never infer a kind from the port name.
  The connection menu only offers earlier tools (workflow order, which also
  makes cycles impossible) and compatible kinds: an `ANY` output only feeds
  `ANY` inputs, so a typed input cannot be wired to `results`.  Blocks type
  their dynamic ports through `port_metadata` (`Image` / `Plane` / `Any`) and
  answer `get_input_port_type` / `get_output_port_type` from it.  A new image
  port must be declared: `tests/test_port_types.py` fails otherwise.  Both use
  the shared `ui/widgets/port_table.py`, which edits the name and the kind;
  renaming follows `Workflow.rename_port` and removing drops
  `Workflow.remove_port_connections`, so the wiring is never left dangling.
  The viewer's record selector also uses the port kinds (plus a value check for
  untyped ports); it never inspects port names.
- **PlaneModel** -> `PlaneFitter` publishes `outputs["plane"]` as a
  `core.plane_fit.PlaneModel`.  The normal is unit length and its sense is
  fixed at creation (points towards the camera, `normal[2] <= 0`), so
  `plane.height(point) > 0` means the point is above the plane.  Consumers
  connect it to a `reference plane` input and must not flip the sign again.
  `calib` is a guard: warn when it differs from the consumer image.  `edge_on`
  marks a plane seen edge-on, where heights are not meaningful.
- **3D ROI model** -> a 3D ROI is a rotated 2D rectangle plus a Z band, not a
  6-DoF box.  `roi_2d_to_3d` returns `(center, size, z_min, z_max, angle)` with
  the angle flipped to world degrees (`-image angle`), the same convention as
  `plane_fit`.  Point selection rotates into the ROI frame
  (`extract_points_in_rois`); never compare against an axis-aligned AABB, and
  keep the X mirroring (`flip_x`) consistent with the cloud.  The 3D viewer
  draws the rotated footprint: rotate in 2D, move/resize in 3D (centre handle
  moves in XY, Shift + centre moves the Z band, faces resize with the opposite
  face fixed).
- **Match tool defaults** -> `TemplateMatch` and `ShapeMatch` default to
  `angle_start=-3`, `angle_end=3`, `scale_start=1.0`, `scale_end=1.0`.
  Tests that exercise wider rotations/scales must set their own range.
- **ShapeMatch ROI monotonicity** -> a bigger search ROI must never lose a match
  found by a smaller one.  That is why the vote factor, the edge quota, the
  candidate floor and the dedupe distance cannot depend on the ROI extent.
  Coarse views are cropped from a pyramid anchored to the frame origin so the
  same scene region always uses the same pixels.  Pre-refine dedupe is
  pose-aware (position + angle + scale); merging by position alone can discard
  the correct scale before the NCC scan sees it.  Never stop refining early on
  `max_matches`; rank the measured candidates by final score.
- **ShapeMatch ROI-bound preprocessing** -> for an axis-aligned `uint8` ROI,
  the search converts only a frame-aligned gray block covering the ROI plus an
  ECC halo.  The block origin must be a common multiple of the vote, search and
  ECC pyramid factors.  Rotated ROIs and non-`uint8` inputs keep the older
  full-frame fallback.
- **ShapeMatch ECC** -> two levels: affine ECC on a reduced view (factor <=2)
  then a rendered NCC translation at full resolution.  The reduced pass must
  undo its resampling with `template_scale`/`scene_scale`/`scene_offset`;
  without that correction the reported scale drifts by up to 0.65 %.  Small
  templates (factor 1) keep the full-resolution affine + polish path.
- **ShapeMatch padding** -> the rotated template's padding diagonal must be even,
  otherwise its center lands between pixels and translation is biased by ~0.5 px.
  Render NCC padding with the template's median border color, never zero.
  Crop the padding before `matchTemplate` (`_render_search`); a rotated rectangle
  is centrally symmetric so the crop stays centred.
- **Workflow tree rebuilds** -> `WorkflowEditor` renders its workflow in
  `__init__` (the Block config reuses an existing workflow); `WorkflowTree.set_workflow`
  builds the items before clearing and keeps a render-error row; `on_reorder`
  never deletes tools missing from the view; and cross-tree drops accept
  `CopyAction` because the handler moves the tool itself (Qt would otherwise
  remove the source rows a second time).

## Build

### Native kernel (only C++ build)
```bash
build_kernel.bat          # Windows: configure + build + deploy cpp/bin
build_kernel.bat --rebuild
./build_kernel.sh         # Linux/macOS
```
No OpenCV world build. Requirements: CMake 3.20+, C++17 compiler, pybind11,
OpenMP.

### Python package
```bash
pip install -e ".[dev]"   # builds the kernel via scikit-build-core
pytest -q --ignore=tests/test_trainer_equivalence.py
```

## Key facts
- Python 3.13.x (only supported version), branch: `main` (release) / `dev`
  (active development).
- No PyInstaller/Inno distribution build; distribution is the PyPI wheel/sdist.
- `config.json` lives in `%APPDATA%/FalcoCV/config.json` (or XDG config).
- Models live in the user data dir and are downloaded from GitHub Releases.
- `core/` has no PyQt6/UI imports at module level.
- License: **GNU GPL v3 or later** (see `LICENSE`).
- Branding: `src/falcocv/ui/assets/` (`icon.ico`, `icon_light.ico`, `logo.svg`,
  lockups). The splash uses `logo_lockup_dark.png`; the web dashboard serves
  `/hmi/logo_lockup_dark.png`.
- Do not make commits without explicit permission.
- Server: `falcocv-server` or `python -m falcocv.web`; dashboard:
  `http://localhost:8000`; HMI: `http://localhost:8000/?id=<project_id>`.
- Performance harness: `benchmarks/` (`python benchmarks/bench_tools.py`).
- ShapeMatch performance history: `docs/shape_match_perf_log.md`.

## Current handoff (2026-11-02)

- `dev` has the ShapeMatch v3 work and the geometry validation; `main` is the
  1.0.2 release.  No release planned.
- ShapeMatch Phase 1 (`a97c54c`), Phase 2 (`d91ecc7`) and Phase 3 (merged in
  `fe8a135`): model v3, training mask, Canny / `min_gradient` /
  `max_features` controls, manual feature editor with brush, spatial feature
  coverage, ROI-bound search preprocessing and geometry mode.
- ShapeMatch v2 models are rejected on purpose; 1.0.2 jobs must be retrained.
- Spatial coverage is the default automatic selection.  Automatic orientation
  coverage was prototyped and rejected because it changed pose/score on a real
  job; do not reintroduce it without the benchmark gate.
- Manual flags are provenance only.  The selected set is authoritative and
  manual features are not prioritised in the vote subset.
- The normal preview shows only the selected features.  Candidates are shown
  only in the manual editor.
- Speed work (2026-11-02): only the staged translation grid survived
  (`067b8cc`, -8/-14 %, pose delta 0).  Convergence stop, search subset,
  pose-lane batching, Levenberg-Marquardt and the alignment polish were tried
  and reverted; each is recorded with numbers in
  `docs/shape_match_perf_log.md` under "What did not work".
- Geometry mode passed on detections and speed (1005 real frames, 0 lost,
  0 extra, -31 %) but failed on precision (median 1.00 vs 0.65 px against the
  job's TemplateMatch, one 85 px failure).  Geometry stays a fast mode; hybrid
  is the precision reference.  Gate: `scripts/validate_shape_match_modes.py`,
  procedure in `docs/SHAPE_MATCH_VALIDATION.md`.
- Latest verification: `432 passed, 3 skipped, 3 xfailed`; `falcocv self-test`
  PASS.  Synthetic ShapeMatch matrix: 0.001-0.003 px, <=0.010 deg, <=0.035 %.
- Next work: 3D tools (depth, point cloud, plane fitting).  Resume on another
  machine with `git pull origin dev`, `pip install -e ".[dev]"` and
  `pytest -q --ignore=tests/test_trainer_equivalence.py`.
