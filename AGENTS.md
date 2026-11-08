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
- **ShapeMatch fused vote** -> `shape_match_vote_fused` (kernel) takes the
  int16 `cv2.spatialGradient` gx/gy and the Canny mask, computes magnitude and
  orientation, applies the 32 px / 48-edge cap and runs the vote in one native
  call.  Python no longer builds full-frame mag/theta arrays, calls
  `np.nonzero` or `_cap_scene_edges`.  The legacy `shape_match_vote` entry point
  is kept; both share `vote_from_scene_bins`, and
  `test_shape_match_fused_vote_matches_legacy` requires exact pose/score parity
  (1e-6).  Opt-out with `FALCOCV_SHAPE_MATCH_FUSED_VOTE=0`; the Python path
  falls back to the legacy vote when the loaded kernel lacks the symbol.
  Measured: 1 MP 50.9 -> 35.2 ms, 2.7 MP 104.1 -> 59.8 ms, demo geometry
  51.1 -> 44.2 ms, hybrid 94.3 -> 86.6 ms.  The 39-frame depth A/B and the
  1005-frame shape2 gate passed with exact parity (0 mismatches); an
  in-process 100-frame A/B measured 56.1 ms fused vs 57.8 ms legacy.
- **ShapeMatch training noise filter** -> `edge_filter_mode` (`auto`/`manual`)
  and `edge_filter_min_area` discard small Canny components before feature
  selection.  Auto = `max(8, min(0.10 * largest_component,
  0.0002 * template_area))`; manual uses the explicit area, 0 disables it.
  The applied area and counts are stored in `model_b64`; changing the mode or
  area marks the model STALE until retrained.
- **ShapeMatch edge proposals** -> `edge_proposals` defaults to True in
  `geometry`: a reduced edge-map correlation proposes the top 5 NMS peaks
  before refinement.  The edge score is proposal-only, never the final
  confidence, and no grayscale is used.  The proposal view uses a factor
  derived from the template (`min_template // 32`, never finer than the search
  view, capped at 8x); `FALCOCV_SHAPE_MATCH_PROPOSAL_FACTOR=search` restores the
  old view for A/B.  On 39 real depth frames it raised detections from 16 to
  39 at `min_score=0.15` with zero false positives; the old full search view
  found 38.  The 1005-frame shape2 job already uses search factor 8, so the
  proposal view is unchanged there.  Edge proposals are an independent source
  and now compete in the geometry pre-gate instead of being merged with vote
  peaks before scoring.  When the primary geometry solve scores below 0.50,
  the strongest edge proposal is solved as one extra hypothesis and replaces
  it only if the final score improves by 0.05.  On the 1005-frame shape2 job
  this cut max px 85.68 -> 11.01 with 0 lost/extra; geometry went 59.9 ->
  63.2 ms (+5.5 %).  The gross 80 px failures are gone, but angle/scale still
  miss the strict gate.
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
  tab only while `match_mode` is `"hybrid"`; in `"geometry"` the control is
  hidden because the native path forces gradient alignment internally.
  `"alignment"` uses the native gradient-alignment score and passed the
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
- **FCD v3 depth container** -> `.fcd` is a FalcoCV-specific **calibrated raw
  uint16** container, not TIFF.  The magic is `FCD3`; the JSON header stores
  shape, dtype, `value_type="raw_counts"`, unit, `x_res`/`y_res`/`z_scale`/
  `z_offset`, `invalid_value=0` and optional model/timestamp.  The LZ4 payload
  is the raw uint16 image.  `DepthRangeLoader` exposes float32 raw counts with
  `scale_z`/`offset_z`; TemplateMatch/ShapeMatch keep their raw-count ranges.
  There is no FCD v1/v2 compatibility.  Measured on a `20000x4096` frame:
  ~32.5 MB/file and ~90 ms decode.  TIFF/PNG inputs remain raw-count images
  using the loader parameters.  The TIFF path is kept for existing files.
  `scripts/convert_depth_to_fcd.py` copies the original uint16 TIFF values and
  stores `--x-res/--y-res/--z-res/--z-offset` in the FCD header.
- **Point-cloud colors are lazy** -> loaders keep `_3d_data["colors"] = None`;
  `ensure_pointcloud_colors(vimg)` generates them only when a 3D viewer uploads
  the cloud.  This avoids ~47 ms per frame in headless/production jobs.
- **Tool config record selectors** -> use `make_record_selector()` and
  `add_record_header(parent_layout, combo, extra_widgets, tool)` from
  `ui/tool_config_helpers.py` for the viewer's top row; never add the combo
  directly.  `add_record_header` creates the horizontal row and its trailing
  stretch, so every processing tool has Run/Auto at `x=8`, then the selector
  after them with an 8 px gap, at `y=0`, width `285`.  The Auto button widens
  only enough for its `AUTO ON` checked label.  `add_record_selector` is only for an existing
  `QHBoxLayout`; passing a `QVBoxLayout` turns the left/right spacers into
  vertical spacers and breaks the alignment.  ImageLoader, DepthRangeLoader,
  GigE, Python Script and Block intentionally have no record selector or run
  controls.
- **Tool config Run/Auto** -> `ToolRunControls` lives at the left of the
  selector row.  `Run` calls `tool.execute()`; `Auto` is opt-in and runs the
  same tool 250 ms after the last `parameter_changed` event.  It never reruns
  upstream or downstream tools.  `BaseTool.mark_dirty` emits
  `parameter_changed` separately from `data_changed`, avoiding refresh loops.
  Closing the config dialog stops the timer and disconnects the signal.
  `supports_manual_run = False` marks sources and special tools.
- **HMI runtime context** -> `HMIProject.context` is a runtime dict exposed as
  `context` in the HMI script and `proj.context`.  `run_slot` injects it into
  `workflow.engine_context`; `PythonScriptTool` passes it to
  `CustomTool.__init__(engine_context=...)`, recreating the instance when the
  context identity changes, and `Block` forwards it to nested workflows.  Use
  this for live objects such as an already-connected scanner controller or PLC,
  and for callbacks such as `context['log'] = log`; it is intentionally never
  serialized.  The defaults of both the HMI script and the Python Script
  template document the handoff with the generic `log` callback.
- **Communications runtime** -> `src/falcocv/comms/` owns generic devices, tags
  and quality, independent of Qt.  The optional `comms` section of `.hmi` is
  validated by `parse_comms_config`; `validate_comms_config` separates
  structural errors from incompleteness warnings, so the designer can save a
  draft with missing driver fields or empty tag addresses.  Incomplete drafts
  are marked in the Communications tab and produce Check warnings; Deploy,
  local Signal Monitor Connect and worker startup block until complete.
  Structural errors still block saving.
  `HMIProject.start_runtime()` creates the `CommsManager`, injects
  `context['comms']`, and `stop_runtime()` disconnects it.  The worker starts runtime before the HMI script and stops the script
  before the devices.  The first driver is `dummy`; tag definitions live in
  `.hmi` while runtime values and statuses stay runtime-only.  Future
  TCP/Modbus/PLC drivers implement `DeviceDriver` and register in
  `comms/drivers/__init__.py`; widgets must read `TagStore`, never a driver.
  `DeviceDriver.read_many` / `write_many` are optional batching hooks: the
  manager groups by device and the polling thread uses `read_many`, but protocol
  details such as Omron structs, ADS symbols or S7 addresses stay inside the
  corresponding driver.  Driver config fields are declared by
  `DriverSchema`/`ConfigField` in the driver module and rendered by the HMI
  Designer; unknown keys stay in the collapsed Advanced JSON editor.  The HMI
  Designer has a local **Signal Monitor** tab backed by `CommsMonitorSession`;
  it opens devices explicitly only while connected and is not a live view of a
  deployed worker.  The designer edits an isolated working copy and only Save
  writes the `.hmi`; Deploy with unsaved changes asks whether to save or deploy
  the in-memory copy.  The Signal Monitor has a **Live worker** mode that reads
  and writes the deployed worker's own `CommsManager` through
  `/api/hmi/comms/snapshot` and `/api/hmi/comms/write`; it must not be confused
  with the separate local test session.  Live errors use structured
  `JSONResponse` 4xx/5xx bodies; the monitor shows an explicit connection badge
  and only enables writes after a successful snapshot.  The monitor updates
  table rows in place and chains single-shot refreshes after completion; Live
  uses `/api/hmi/comms/values` after the initial snapshot, and Enter in the value
  field writes the selected tag.  Boolean tags use explicit Set True / Set False
  buttons; focusing the value textbox selects all and marks it dirty so refresh
  cannot prefix a stale value.  Enter is consumed by the monitor and the monitor
  and HMI Designer top buttons are not auto-default, so Enter cannot trigger
  Deploy/Stop.  Live defaults to `127.0.0.1`, not
  `localhost`; `localhost` can add ~2 s per connection on Windows.  Stale
  responses from a disconnected worker are ignored by connection token.
  HMI widgets support `source="tag"` with a namespaced tag id; the designer
  lists tags from `comms_config`, and `/api/hmi/comms/widget-values` returns
  evaluated value/quality/status.  This first phase is read-only.  The `omron`
  driver implements the `omron-fast` strategy: one read per struct, snapshot
  TTL, shadow writes and optional verification.  Its dictionary is session-only
  and never loaded from or written to a persistent file; `aphyt` is imported
  lazily when the driver starts.  The `beckhoff` driver uses symbolic names and
  one `read_list_by_name`/`write_list_by_name` batch; its optional IP is only for
  Linux/routed ADS routes.  The `siemens` driver uses standard absolute S7
  addresses and groups DB/area byte ranges.  The `adlink` driver uses the DASK
  DLL through `ctypes` on Windows x64: its model selector comes from a single
  registry of DASK card types, addresses are `DI0.3`, `DO0.3`, `AI0` and `AO0`,
  digital ports are batched, AI uses the device `ai_range`, and AO reads return
  the last value written by FalcoCV because DASK has no analog-output readback.
  `verify_writes` waits `verify_delay_ms` (default 20 ms) before readback
  because USB-7250 relays need settling time.  The verified USB-7250
  capability entry limits the model to port 0 with 8 DI and 8 DO; common DASK
  errors such as -302/-31/-32 are translated.  The optional `protocols` extra
  installs `pymodbus` and `asyncua`; `modbus_tcp` and `opcua` are client
  drivers, while `comms/servers/` owns `ServerManager`, Modbus TCP server and
  OPC UA server runtimes.  Server mappings are explicit; Modbus/OPC UA server
  writes go through `CommsManager`, and HMI `.hmi` serialization/deploy/worker
  payloads carry a top-level `servers` section.  The HMI Designer has a
  **Servers** tab.  Modbus has no auth; OPC UA server security is
  anonymous/no-security in this first release.  Changing a device driver offers
  Keep tags / Delete tags / Cancel; Keep preserves tag metadata and clears only
  the driver-specific addresses and configuration.  All PLC drivers default
  `verify_writes` to false.  Without verification they trust the library and
  never read back or retry.  With verification they read each written batch
  back; `write_retries` (default 0) controls extra write+verify attempts after
  a value mismatch only, not transport errors.  The optional `plc` extra
  installs `aphyt`, `pyads` and `python-snap7`.  Vendor native libraries are
  resolved by `core/native_deps.py`: explicit `dll_path`, `FALCOCV_NATIVE_DIR`,
  `%APPDATA%/FalcoCV/native/<driver>` and then the system search path.  ADLINK
  DASK DLLs go in `native/adlink`; Beckhoff `TcAdsDll.dll` can be in
  `native/beckhoff`; `python-snap7` 3.x is pure Python and needs no DLL.
  `falcocv native-deps` reports the resolved locations.  Driver classes expose
  `check_dependencies()` and optional `probe()`; the Communications tab shows
  a setup panel only for ADLINK/Beckhoff, hides the DLL path when the runtime
  is detected, offers Recheck/Help/Test connection, and reports missing
  dependencies to Check/Deploy.  Beckhoff always shows the ADS route as
  `PENDING` until `Test connection` succeeds.  Window geometry, dock state,
  splitter sizes and table column widths persist through
  `ConfigManager.data["views"]` using Qt geometry/state blobs; this includes
  both Servers splitters and the Servers mapping columns.  Main and HMI
  Designer `View > Reset all views` clear and reapply defaults.
- **PlaneFitter fixed-Z ROIs** -> when every ROI has a fixed Z range, the
  native plane fit skips the full 82 MP image-range scan; it samples the image
  every 32 pixels only for the cover polygon.  The plane and polygon remain
  identical on `depth.cvj`; the tool drops from ~440 ms to ~120 ms.  If a ROI
  uses the automatic `0..1` Z sentinel, the full scan is retained.
- **PlaneFitter auto-fit Z margin** -> `auto_z_margin` (default `3.0`, `0`
  disables) expands the exact depth range on both sides when Auto-fit Z runs
  or a new ROI is drawn.  It changes the selected Z band only, never the plane
  fit itself.  Existing saved ranges are not rewritten until Auto-fit Z is
  pressed again.  The shared 3D ROI editor reads the parameter from the tool;
  Height Blobs has no such parameter and keeps a zero margin.
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
- **Workflow port types** -> every tool declares its image, plane and structured
  result ports as class attributes (`input_port_types` / `output_port_types`
  with `IMAGE` / `PLANE` / `RESULTS`); unlisted ports are `ANY`.  Never infer a
  kind from the port name.  The connection menu only offers earlier tools
  (workflow order, which also makes cycles impossible) and compatible kinds:
  an `ANY` output only feeds `ANY` inputs, so a typed input cannot be wired to
  an untyped output.  Blocks type their dynamic ports through `port_metadata`
  (`Image` / `Plane` / `Results` / `Any`) and answer `get_input_port_type` /
  `get_output_port_type` from it.  A new image port must be declared:
  `tests/test_port_types.py` fails otherwise.  Both use the shared
  `ui/widgets/port_table.py`, which edits the name and the kind; renaming
  follows `Workflow.rename_port` and removing drops
  `Workflow.remove_port_connections`, so the wiring is never left dangling.
  Every tool that publishes `results` must declare it as `RESULTS`; Block and
  Python Script expose the kind through the shared port table.  The viewer's
  record selector also uses the port kinds (plus a value check for untyped
  ports); it never inspects port names.
- **Height Blobs 3D** -> one feature per connected region above the reference
  plane.  Heights are computed per pixel from the depth grid (mirrored X,
  `z = depth*z_res + offset_z`), not from the strided cloud, so thin pins keep
  full resolution.  `plane_offset` shifts the plane along +normal (positive =
  up/+Z); a negative value is how features below the fitted surface are
  measured.  The tip is the centroid of the `tip_band` under the maximum,
  expressed in `PlaneFrame` (u, v in-plane, h along the normal), which makes
  the measurement tilt-invariant.  Blob statistics run on each component's
  bounding box, never over the whole ROI crop: a 13 MP ROI went from 2.9 s to
  0.26 s with that change.  The public structured output is `results` typed
  `RESULTS`; `count` mirrors the result count.  There is no separate `features`
  output: a duplicate port confused the Results table and port menu.  Without
  a reference-plane connection, the tool uses a horizontal capture-zero plane
  at `z = offset_z` and emits a warning.  Tool prefix: `[3DBL]`.
  `grid` ordering estimates the row/column axis from nearest-neighbour vectors
  and groups rows from the largest projected gap; there is no `row_tolerance`
  parameter.  This keeps 3x2 and 4x4 matrices stable under slight rotations
  and noise.
- **PlaneFitter UI** -> Z min/max spin boxes use `setKeyboardTracking(False)`
  and are not rewritten while focused, so a `mark_dirty` refresh cannot
  interrupt editing.  The left panel has an Overlays group for `show_overlays`
  and `copy_overlays_to_input`, synchronized with the view-bar checkbox.
- **PlaneModel** -> `PlaneFitter` publishes `outputs["plane"]` as a
  `core.plane_fit.PlaneModel`.  The normal is unit length and points towards
  world +Z/up (`normal[2] >= 0`), so `plane.height(point) > 0` means the point
  is above the plane.  Consumers connect it to a `reference plane` input and
  must not flip the sign again.  `calib` is a guard: warn when it differs from
  the consumer image.  `edge_on` marks a plane seen edge-on, where heights are
  not meaningful.
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
- **ShapeMatch compact settings panel** -> the left content minimum is ~228 px
  (was ~670).  The Search/Training tabs use two-column field grids; the preview
  is 180 px, the `Pre-gate scorer` group is hidden in `geometry`, and the old
  saved splitter width is reset once through
  `ConfigManager.data["layout_revisions"]["shape_match_compact_panel"]`.  Keep
  labels short and put explanations in tooltips.
- **Global wheel guard** -> `ui/wheel_guard.py` is installed once from
  `apply_qt_theme`.  The wheel over `QAbstractSpinBox`/`QComboBox` never changes
  the value; it is forwarded to the nearest `QAbstractScrollArea` viewport.
- **ROI editor Qt 6.12 regression** -> `QDoubleSpinBox.ButtonSymbols.NoButtons`
  collapses the internal `QLineEdit` to 1 px with the app stylesheet.  Keep the
  default buttons and hide the arrows with QSS.  The 2D editor uses two columns
  and 80-110 px spins; disabled values use `TEXT_MUTED`.
- **Height Blobs overlay visibility** -> mirrors Blob Detector 2D:
  `show_overlays` (master), `show_search_roi`, `show_contours`,
  `show_centroids`, `show_labels`, `copy_overlays_to_input`.  Centroids draw a
  2D contour centroid plus the 3D tip world points; search ROI skips full-image
  ROIs.  Defaults are True.
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

## Current handoff (2026-11-08)

- `dev` is the active development branch and carries 1.1.0 plus this
  post-1.1.0 work.  `main` is not touched by this push.
- The `omron`, `beckhoff` and `siemens` drivers are implemented.  Omron has
  been validated against real hardware; Beckhoff and Siemens still use fake
  clients pending hardware validation.  The optional `plc` extra installs
  `aphyt`, `pyads` and `python-snap7`.  Write verification is opt-in for all
  three; `write_retries` only applies after a verification mismatch.
- HMI communications drafts can now be saved incomplete (missing required
  driver fields or empty tag addresses) through `validate_comms_config`; the
  Communications tab and HMI Designer tab show the warning.  Check reports
  warnings; Deploy, local Signal Monitor Connect and worker startup block
  incomplete runtime configs.  `HMIProject.save` rejects structural errors but
  preserves incomplete drafts.  Driver changes offer Keep tags / Delete tags /
  Cancel.
- The `adlink` DASK driver is implemented with the full model registry,
  `DI`/`DO`/`AI`/`AO` addresses, digital port batching and analog ranges.
  Native vendor libraries are resolved from
  `%APPDATA%/FalcoCV/native/<driver>` or `FALCOCV_NATIVE_DIR`; `falcocv
  native-deps` reports them.  USB-7250 hardware is validated, including
  `verify_writes` with the relay settling delay.
- Modbus TCP and OPC UA clients plus Modbus TCP/OPC UA servers are implemented.
  The optional `protocols` extra installs `pymodbus`/`asyncua`; servers use
  explicit mappings, start after the clients and are serialized/deployed in
  the top-level `servers` section.  The OPC UA server uses anonymous/no-security
  in this release.
- Window geometry, dock state, splitter sizes and table column widths persist
  through `ConfigManager.data["views"]`; the Servers tab adds
  `hmi_servers.horizontal`, `hmi_servers.vertical` and
  `hmi_servers.mappings`.  **View > Reset all views** restores defaults in the
  main window and HMI Designer.
- ShapeMatch Phase 1 (`a97c54c`), Phase 2 (`d91ecc7`) and Phase 3 (merged in
  `fe8a135`): model v3, training mask, Canny / `min_gradient` /
  `max_features` controls, manual feature editor with brush, spatial feature
  coverage, ROI-bound search preprocessing and geometry mode.
- Post-1.1.0 (2026-11-08): fused native vote (`shape_match_vote_fused`, legacy
  parity, opt-out), training noise filter, geometry edge proposals on a coarser
  template-derived view, edge proposals competing in the geometry pre-gate
  with a low-confidence second chance, compact ShapeMatch settings panel with
  the pre-gate hidden in geometry, global wheel guard, Qt 6.12 ROI spin fix,
  and HeightBlobs visibility options mirroring Blob Detector 2D.
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
- README screenshots are generated by `scripts/capture_readme_screenshots.py`
  into `docs/images/` from the real Vision Editor, Viewer3D and HMI Designer.
  `examples/demo_dashboard.hmi` is the richer HMI demo; its dashboard image is
  the Designer preview with representative data, not a live worker capture.
- Latest verification: `743 passed, 12 skipped, 3 xfailed`; UI
  `147 passed, 5 skipped`; the fused vote matches the legacy path exactly
  (1e-6); the 39-frame depth A/B found 39/39 true detections with zero false
  positives.  The 1005-frame shape2 gate found 0 lost/extra, max px 11.01,
  max angle 2.17 deg, max scale 2.58 %, geometry 63.2 ms vs hybrid 96.9 ms.
  The gross 80 px translation failures are gone, but the strict 5 px /
  2 deg / 1.5 % precision gate still fails.  The `falcocv-1.1.0` wheel was
  built before this post-1.1.0 work and passed `scripts/check_wheel.py` plus
  `scripts/wheel_smoke.py`; rebuild it before publishing a new wheel.
  Synthetic ShapeMatch matrix: 0.001-0.003 px, <=0.010 deg, <=0.035 %.
- Next work: improve the remaining geometry angle/scale precision (max 2.17
  deg / 2.58 %, max px 11.0), validate the edge second chance on more jobs,
  then validate Beckhoff ADS and Siemens S7 against real hardware.  Omron and
  ADLINK USB-7250 hardware validation are complete; their automated tests
  still use fakes where appropriate.  Modbus/OPC UA loopback tests run with the
  optional `protocols` extra.  Resume on another machine with
  `git pull origin dev`, `pip install -e ".[dev,plc,protocols]"` and
  `pytest -q --ignore=tests/test_trainer_equivalence.py`.
