# Depth formats

FalcoCV can load range/depth images from TIFF, PNG, BMP and JPEG, and also
from its native `.fcd` container.

## `.fcd` (FalcoCV Depth)

`.fcd` is **not a TIFF**. It is a small FalcoCV-specific container:

```text
magic "FCD1"
uint32 little-endian JSON header length
JSON header:
  version, shape, dtype, compression="lz4", payload_size
LZ4 frame payload
```

The payload holds the raw 2D depth array without dtype conversion. A typical
16-bit range frame is stored as `uint16`; the loader converts it to `float32`
after loading, applying the configured `scale`.

The format is intended for fast local loading:

- it avoids LZW decoding;
- it keeps the original integer values losslessly;
- it is usually around 1.5x the size of the source LZW TIFF, but loads much
  faster.

`.fcd` is not a frame cache. It is a persistent storage format for new frames.

## Converting TIFF batches

Use:

```bash
python scripts/convert_depth_to_fcd.py \
  --input path/to/tiff/folder \
  --output path/to/fcd/folder
```

The converter:

- reads each TIFF with `imagecodecs`, OpenCV or `tifffile`;
- writes an `.fcd` with the same relative name and `.fcd` suffix;
- optionally verifies that the decoded output matches the source (enabled by
  default);
- never modifies the source files.

## Loading order

`DepthRangeLoader` chooses the decoder by extension:

- `.fcd`: native LZ4 container;
- `.tif`/`.tiff`: `imagecodecs.tiff_decode`, then `cv2.imread`, then
  `tifffile`;
- other image extensions: OpenCV.

Existing TIFF jobs keep working; `.fcd` is additive.

## Point-cloud colors

When `generate_3d` is enabled, the loader builds the point cloud but leaves
`_3d_data["colors"]` as `None`. 3D viewers call
`ensure_pointcloud_colors(vimg)` when they upload the cloud, so headless jobs do
not pay the colormap cost.
