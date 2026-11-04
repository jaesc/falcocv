# 3D conventions

## World axes

Point clouds use:

```text
x = (width - 1 - image_col) * x_res
y = image_row * y_res
z = depth * z_res + offset_z
```

The X mirror keeps the image left side on world +X. **World +Z is up**, matching
the default 3D viewer and plane overlays.

## Planes

`PlaneFitter` publishes a `PlaneModel` with:

- a unit normal pointing towards **+Z/up** (`normal[2] >= 0`);
- `height(point) > 0` for points above the plane.

`plane_offset` shifts the plane along its normal:

- positive moves the plane up and reduces measured heights;
- negative lowers it and increases measured heights.

## ROI auto-fit Z margin

`PlaneFitter` has an `auto_z_margin` parameter (default **3.0 mm**). When
**Auto-fit Z** is used, the detected depth range is expanded symmetrically:

```text
z_min = detected_min - auto_z_margin
z_max = detected_max + auto_z_margin
```

This keeps the ROI usable when another part is a few tenths of a millimetre or
a millimetre away. It changes the selected Z band, not the plane-fitting
algorithm. Existing saved ranges are not rewritten automatically; press
**Auto-fit Z** again (or draw a new ROI) to apply the current margin.

## Height Blobs 3D

`reference plane` is optional.

- If connected, the fitted `PlaneModel` is used.
- If not connected, a horizontal **capture-zero plane** is used at
  `z = offset_z` (raw depth zero after calibration). A warning is emitted.
- `direction="above"` measures +Z/up; `direction="below"` measures the other
  side.
- The tool prefix is `[3DBL]`.
- Public outputs are `output image`, `binary mask`, `count` and `results`
  (`RESULTS`). The results contain the measured feature fields; there is no
  separate duplicate `features` port.
