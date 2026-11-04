# Depth formats

FalcoCV can load range/depth images from TIFF, PNG, BMP and JPEG, and from its
native `.fcd` container.

## `.fcd` v3 (FalcoCV Depth)

`.fcd` is **not a TIFF**. It is a small FalcoCV-specific calibrated raw-depth
container:

```text
magic "FCD3"
uint32 little-endian JSON header length
JSON header:
  version=3, shape, dtype="<u2", value_type="raw_counts",
  unit="mm", compression="lz4", payload_size,
  calibration {x_res, y_res, z_scale, z_offset},
  invalid_value=0, optional model/timestamp/source
LZ4 frame payload
```

The payload is a **uint16 raw depth image**. Invalid samples are `0`. The
millimetre Z value is reconstructed with the header calibration:

```text
z_mm = raw * z_scale + z_offset
```

For `.fcd` v3, the header calibration takes precedence over the manual X/Y/Z
fields in `DepthRangeLoader`. The loader exposes the legacy pipeline
representation: `float32` raw counts in `VisionImage.data`, with `scale_z` and
`offset_z` from the header. This preserves the native sensor depth grid and
keeps TemplateMatch/ShapeMatch range models in raw counts.

## Sizes and speed

Measured on a `20000 x 4096` frame:

```text
raw uint16 + LZ4: ~32.5 MB/frame
decode:           ~90 ms
loader:           ~200-300 ms depending on disk cache
```

FCD v1 and FCD v2 are intentionally not supported.

## Converting TIFF batches

Use:

```bash
python scripts/convert_depth_to_fcd.py \
  --input path/to/tiff/folder \
  --output path/to/fcd/folder \
  --x-res 0.025 --y-res 0.025 \
  --z-res 0.00061 --z-offset 20.0
```

The converter:

- reads each TIFF with `imagecodecs`, OpenCV or `tifffile`;
- stores the original uint16 raw values;
- writes FCD v3 with the supplied calibration;
- verifies the uint16 round-trip by default;
- never modifies the source files.

## Loading order

`DepthRangeLoader` chooses the decoder by extension:

- `.fcd`: FCD v3 raw uint16 + LZ4;
- `.tif`/`.tiff`: `imagecodecs.tiff_decode`, then `cv2.imread`, then
  `tifffile`; these remain raw-count images and use the loader calibration;
- other image extensions: OpenCV.

Existing TIFF jobs keep working.

## Point-cloud colors

When `generate_3d` is enabled, the loader builds the point cloud but leaves
`_3d_data["colors"]` as `None`. 3D viewers call
`ensure_pointcloud_colors(vimg)` when they upload the cloud, so headless jobs do
not pay the colormap cost.
