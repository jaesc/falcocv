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
