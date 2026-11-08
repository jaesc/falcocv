# ShapeMatch geometry validation

`match_mode="geometry"` is the default for new tools, but it is still marked
experimental.  This procedure is the gate it has to pass before the mode can be
considered consolidated: it compares geometry against the hybrid reference on a
production image set and reports every detection and pose difference.

## What it does

For every frame it runs the whole job twice, once forcing
`match_mode="geometry"` and once forcing `"hybrid"`, then compares:

- **lost**: a hybrid detection with no geometry counterpart within `--match-px`.
  A single lost detection fails the gate.
- **extra**: a geometry detection with no hybrid counterpart.  Informational:
  it can be a real improvement, and must be inspected visually.
- **pose deltas** of the matched detections: translation (px), angle (deg) and
  scale (%).
- **solver**: geometry runs must report `solver="geometry"`; anything else means
  the mode did not actually run and fails the gate.
- **timing** per tool and mode.

The job is loaded in memory and never written back, so the `.cvj` is unchanged.

## Requirements

- The job must have an `Image Loader` connected to each ShapeMatch tool's
  `input image`.  The script overrides the loader path per frame.
- ShapeMatch models must be trained (v3).  v2 models are rejected by the tool.

## Running it

On the machine that holds the production images:

```bash
python scripts/validate_shape_match_modes.py \
    --job path/to/job.cvj \
    --images path/to/frames \
    --json reports/shape_geometry_validation.json
```

Useful options:

| Option | Default | Meaning |
|---|---|---|
| `--limit N` | all | first N frames, for a quick pass |
| `--pattern "*.png"` | image suffixes | glob inside `--images` |
| `--match-px` | 200 | max distance to pair two detections |
| `--max-px` | 5.0 | gate: max translation delta |
| `--max-angle` | 2.0 | gate: max angle delta (deg) |
| `--max-scale-pct` | 1.5 | gate: max scale delta (%) |
| `--no-gate` | off | report only, always exit 0 |
| `--json PATH` | - | full report for archiving |

The thresholds above are the documented worst case of the geometry prototype
(about 5 px / 2 deg / 1.5 % on the reference job).  Tighten them once the mode
is trusted; widening them defeats the gate.

## Output

```text
job:        job.cvj
frames:     142
thresholds: dpx<=5.0 dang<=2.0 deg dscale<=1.5 %

tool              lost  extra      dpx     dang   dscale%  hybrid ms  geom ms
-----------------------------------------------------------------------------
Shape Match_0        0      0    2.210    1.900     1.500       72.8     68.9

  worst Shape Match_0: frame_087.png dpx=2.210 dang=1.900 dscale=1.500%

VERDICT: PASS
```

`VERDICT: FAIL` exits with code 1 and lists up to 20 failures with their frame
and tool.

The JSON report contains the per-tool summary, the five worst frames per tool,
the thresholds and the failure list.

## Decision rule

The mode can be consolidated when, on the production set:

1. `lost = 0` for every tool.
2. All pose deltas are inside the agreed thresholds.
3. Every `extra` detection has been reviewed and accepted as real.
4. No frame reports an error in either mode.

If a threshold is exceeded, the frame name in `worst` is the one to inspect.  A
regression localised to a few frames usually points to occlusion or a tight
angular range, not to the whole set.

## Synthetic smoke test

To check the script itself without production data, build a small job with a
trained ShapeMatch and a few rendered frames, then run the same command.  The
comparison path, the pairing logic and the JSON output are exercised without
needing the private image set.

## Reference result (2026-11-02)

First full run of the gate: 1005 frames (`VisA/pcb4`, `Normal`), one ShapeMatch,
`max_features=800`, `min_score=0.3`, angle +-3 deg, scale 0.9-1.1.  The job also
carries a TemplateMatch trained on the same ROI and the same pose range, which
gives an independent third opinion.

Geometry against hybrid:

| | value |
|---|---:|
| lost / extra | 0 / 0 |
| median px | 0.90 |
| p90 / p95 / p99 | 2.80 / 3.74 / 8.06 |
| max px | 85.53 |
| max angle | 2.08 deg |
| max scale | 2.95 % |
| frames within 1 px | 55.2 % |
| frames within 2 px | 82.6 % |
| frames within 5 px | 96.9 % |
| median time | geometry 57.7 ms / hybrid 83.5 ms |

Distance to the TemplateMatch pose:

| | median | p90 | p95 | max |
|---|---:|---:|---:|---:|
| translation, geometry | 0.995 | 2.230 | 2.970 | 84.4 |
| translation, hybrid | **0.654** | **1.294** | **1.751** | 10.8 |
| angle, geometry | **0.173** | **0.476** | **0.593** | 2.46 |
| angle, hybrid | 0.192 | 0.514 | 0.706 | 1.86 |
| scale, geometry | 0.326 | 0.776 | 1.007 | 2.81 |
| scale, hybrid | **0.293** | **0.713** | **0.843** | 2.50 |

Closer to the TemplateMatch: translation hybrid 741 / 1005, angle geometry
555 / 1005, scale hybrid 603 / 1005.

Verdict: **FAIL as a precision gate, PASS on detections**.  Geometry finds the
same parts as hybrid on every frame and is 31 % faster, but its translation is
less precise (median 1.0 px vs 0.65 px) and it has gross failures: frame 0855
lands ~85 px away from both hybrid and the TemplateMatch, which agree within
2.4 px.  The alignment-polish attempt did not fix it.

Consequence: geometry stays a fast mode, not the precision default.  Either the
solver's translation precision is improved, or production jobs that need pose
accuracy keep `match_mode="hybrid"`.

## Re-run 2026-11-08 (fused vote + coarser proposals)

`shape2.cvj`, 1005 frames, geometry with the fused vote and the
template-derived proposal view (the job already uses search factor 8, so the
proposal view is unchanged).  The run used `--no-gate` because the known
`0855.JPG` precision outlier is still present.

| | value |
|---|---:|
| lost / extra | 0 / 0 |
| max px | 85.68 |
| max angle | 2.17 deg |
| max scale | 1.99 % |
| geometry median | 59.9 ms |
| hybrid median | 91.6 ms |

The full geometry-vs-hybrid verdict remains FAIL as a precision gate: the
`0855.JPG` outlier is unchanged, and angle/scale still exceed the 2 deg /
1.5 % thresholds.  The fused vote matched the legacy path exactly on every
frame (0 mismatches, all pose/score deltas 0), and an in-process 100-frame A/B
measured 56.1 ms fused vs 57.8 ms legacy.  Detections remain 0 lost / 0 extra.

## Re-run 2026-11-08 (edge proposals in the geometry pre-gate)

`shape2.cvj`, 1005 frames.  Edge proposals are now scored in the geometry
pre-gate instead of being merged into the vote list first.  A geometry solve
below 0.50 gets one second chance from the strongest edge proposal and keeps
it only when the final score improves by at least 0.05.

| | value |
|---|---:|
| lost / extra | 0 / 0 |
| max px | 11.01 |
| max angle | 2.17 deg |
| max scale | 2.58 % |
| geometry median | 63.2 ms |
| hybrid median | 96.9 ms |

Max translation dropped from 85.68 to 11.01 px and the gross ~80 px failures
are gone; the worst remaining frames are 11.0, 10.3 and 8.1 px.  Geometry is
about 5.5 % slower than the previous run.  The strict 5 px / 2 deg / 1.5 %
precision gate still fails on angle and scale, so geometry remains the fast
mode and hybrid the precision reference.