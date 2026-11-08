# ShapeMatch geometry validation

ShapeMatch is geometry-only: the native kernel proposes hypotheses from scene
edges, then edge distance, gradient alignment and ICP refine the pose.  Hybrid
(NCC/ECC) matching was removed; grayscale matching belongs to TemplateMatch.
This procedure is the precision gate: it compares ShapeMatch against a trained
TemplateMatch on the same job and image set.

## What it does

For every frame it runs the whole job once, so ShapeMatch and TemplateMatch see
the same frame, then compares:

- **lost**: a TemplateMatch detection with no ShapeMatch counterpart within
  `--match-px`.  A single lost detection fails the gate.
- **extra**: a ShapeMatch detection with no TemplateMatch counterpart.  A
  single extra detection fails the gate.
- **pose deltas** of the matched detections: translation (px), angle (deg) and
  scale (%).
- **solver**: ShapeMatch runs must report `solver="geometry"`; anything else
  fails the gate.
- **timing** per tool.

The job is loaded in memory and never written back, so the `.cvj` is unchanged.

## Requirements

- The job must have an `Image Loader` connected to each ShapeMatch tool's
  `input image`.  The script overrides the loader path per frame.
- Every ShapeMatch model must be trained with **model v4**.  Model v3 is
  rejected by the tool and must be retrained.
- The job must contain one trained TemplateMatch reference.  If it contains
  several, choose one with `--reference "<tool name>"`.

## Running it

On the machine that holds the production images:

```bash
python scripts/validate_shape_match_reference.py \
    --job path/to/job.cvj \
    --images path/to/frames \
    --json reports/shape_geometry_validation.json
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--limit N` | all | first N frames, for a quick pass |
| `--pattern "*.png"` | image suffixes | glob inside `--images` |
| `--reference NAME` | only trained TemplateMatch | reference tool name |
| `--match-px` | 200 | max distance to pair two detections |
| `--max-px` | 5.0 | gate: max translation delta |
| `--max-angle` | 2.0 | gate: max angle delta (deg) |
| `--max-scale-pct` | 1.5 | gate: max scale delta (%) |
| `--no-gate` | off | report only, always exit 0 |
| `--json PATH` | - | full report for archiving |

The thresholds above are the agreed precision gate for the reference job
(5 px / 2 deg / 1.5 %).  Do not widen them to make a run pass; the JSON `rows`
and `worst` fields identify the frames to inspect.

## Output

```text
job:        shape2_validation.cvj
reference:  Template Match_Ref
frames:     1005
thresholds: dpx<=5.0 dang<=2.0 deg dscale<=1.5 %

tool              lost  extra      dpx     dang   dscale%   shape ms   ref ms
------------------------------------------------------------------------------
Shape Match_0        0      0    5.535    1.162     2.065       60.7    406.5

  worst Shape Match_0: 0867.JPG dpx=5.535 dang=0.499 dscale=0.042%
  worst Shape Match_0: 0957.JPG dpx=5.422 dang=0.477 dscale=0.915%
  worst Shape Match_0: 0051.JPG dpx=4.664 dang=0.228 dscale=1.394%

VERDICT: FAIL
```

`VERDICT: FAIL` exits with code 1 and lists up to 20 failures with their frame
and tool.

The JSON report contains the per-tool summary, the five worst frames per tool,
the thresholds, one row per frame (`px`, `angle`, `scale_pct`, `geo`, `tm`) and
the failure list.  The rows are directly comparable with earlier runs.

## Decision rule

The gate passes when, on the production set:

1. `lost = 0` and `extra = 0` for every ShapeMatch tool.
2. All pose deltas are inside the agreed thresholds.
3. No frame reports an error in either tool.
4. Every detection reports `solver="geometry"`.

If a threshold is exceeded, inspect the frame in `worst`.  A regression
localised to a few frames usually points to occlusion, a tight angular range or
a reference limitation (see below), not to the whole set.

## Reference result (2026-11-08, geometry-only + model v4)

`shape2_validation.cvj`, 1005 frames (`VisA/pcb4`, `Normal`), one ShapeMatch and
one TemplateMatch trained on the same ROI (`764x608`), `max_features=800`,
`min_score=0.3`, angle +-3 deg, scale 0.9-1.1.  Model v4, fused native vote,
edge proposals, geometry solver.

| | value |
|---|---:|
| lost / extra | 0 / 0 |
| median px | 0.901 |
| p90 / p99 | 2.031 / 3.342 |
| max px | 5.535 |
| max angle | 1.162 deg |
| max scale | 2.065 % |
| strict failures (5 px / 2 deg / 1.5 %) | 9 / 1005 |
| frames over 5 px | 2 |
| frames over 2 deg | 0 |
| frames over 1.5 % scale | 7 |
| ShapeMatch median | 60.7 ms |
| TemplateMatch median | 406.5 ms |

The previous geometry-vs-hybrid run on the same job had 16 strict failures,
max 6.996 px, 1.223 deg and 2.057 %.  The solver changes (staged translation
lattice fix, subpixel parabola on translation, final half-pixel scan) reduced
the translation maximum and the median.

Remaining outliers:

- **Translation**: `0867.JPG` (5.54 px) and `0957.JPG` (5.42 px).  On `0867`
  the ShapeMatch pose has a higher edge-alignment score than the TemplateMatch
  pose; on `0957` the TemplateMatch pose scores slightly higher.  Both are
  local-maximum cases, not lost detections.
- **Scale**: `0746`, `0415`, `0313`, `0006`, `0078`, `0709`, `0805` report
  1.52-2.07 % because ShapeMatch estimates 0.979-0.985 while TemplateMatch
  reports exactly 1.0.  Evaluating the edge-alignment score at the same pose
  gives 0.59-0.66 at 0.984 versus 0.44-0.58 at 1.0, so ShapeMatch's scale is
  the better edge fit.  The TemplateMatch reference does not resolve the scale
  between its discrete steps, so these rows are a reference limitation rather
  than a ShapeMatch regression.

The first 20 frames (quick pass) had max 3.20 px / 0.591 deg / 1.592 % before
the final solver tuning.  Use `--limit 20 --no-gate` for a fast smoke test; with
the strict gate the reference's scale row can still fail on this sample.

## Synthetic smoke test

To check the script itself without production data, build a small job with a
trained ShapeMatch and a trained TemplateMatch on the same ROI, render a few
frames, then run the same command.  The comparison path, the pairing logic and
the JSON output are exercised without needing the private image set.

## History

Hybrid mode and the earlier geometry-vs-hybrid gate were removed with model v4.
The historical numbers are kept in `docs/shape_match_perf_log.md` for reference;
they are not a valid gate for the current geometry-only tool.
