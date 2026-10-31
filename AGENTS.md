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
  PatternMatcher-only parameters (`angle_step`, `scale_step`, `use_pyramid`, ...).
- **ShapeMatch coarse voting** -> the vote factor depends on the template only
  (`min_template // 96`, capped at 4x), never on the ROI area.  Cap scene edges
  with a fixed **48-edge quota per 32 px cell**, independent of the cell count;
  a global target divided by cells starves the part as the ROI grows.  Keep 32
  peaks per hypothesis (up to 1024 candidates) and use the top 500 model
  features.  Full gradients, Canny and the distance map stay lazy.
- **ShapeMatch search stage** -> two-stage NCC on a reduced view (template
  >=48 px, max 8x): one single-pose correlation ranks every proposal, then the
  full angle/scale scan runs on the best ones plus the strongest vote peaks.
  Candidates whose NCC is below `_MIN_NCC` are rejected before alignment/solver.
  The alignment score uses one cached full-ROI gradient pair, not a Sobel patch
  per proposal.
- **ShapeMatch ROI monotonicity** -> a bigger search ROI must never lose a match
  found by a smaller one.  That is why the vote factor, the edge quota, the
  candidate floor and the dedupe distance cannot depend on the ROI extent.
  Coarse views are cropped from a pyramid anchored to the frame origin so the
  same scene region always uses the same pixels.  Pre-refine dedupe is
  pose-aware (position + angle + scale); merging by position alone can discard
  the correct scale before the NCC scan sees it.  Never stop refining early on
  `max_matches`; rank the measured candidates by final score.
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
- Python 3.13.x (only supported version), branch: `main` / working branch
  `public_refactor`.
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
