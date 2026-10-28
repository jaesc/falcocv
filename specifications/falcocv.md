# FalcoCV — Architecture Specification

## Runtime

FalcoCV is Python-first: the workflow engine, the tools and the UI are Python
(OpenCV/NumPy). The only native code is the pybind11 kernel in `cpp/kernel`
(`_falco_kernel`), which exposes plain functions — `generate_pointcloud`,
`plane_fit` and `shape_match_vote` — with no OpenCV dependency and no knowledge
of `BaseTool`, `VisionImage` or overlays. Tools never delegate their main work
to C++.

See [`ARCHITECTURE.md`](../ARCHITECTURE.md) for the module layout and
[`AGENTS.md`](../AGENTS.md) for the development rules.

## Overlays

### Definition

Overlays are vector primitives that accompany a `VisionImage`. They live in
named layers; tools draw into the `"default"` layer. Supported types: `ROI`,
`CONTOUR`, `CIRCLE`, `TEXT`, `LINE`, `AXES`, `POINT`, `POLYGON`.

Every element carries a `source_tool` and an optional `ref_id`, which is what
measurement and fixture tools use to select a specific primitive.

### Ownership and propagation

1. **Inheritance**: each tool copies the layers of its input image to its output
   image with `VisionImage.inherit_overlays_from()` (new layer objects, shared
   element instances).
2. **Claiming**: the tool takes the layer with
   `get_overlay_layer("default", source_tool=self.name)`. When a `source_tool`
   is given, an inherited layer is re-tagged, because `add_roi()`, `add_contour()`
   and friends copy `layer.source_tool` into each new element.
3. **Cleanup**: the tool removes the inherited elements it owns:
   ```python
   layer.elements = [e for e in layer.elements if e.source_tool != self.name]
   ```
   This prevents overlays copied to the input on a previous run from being
   duplicated.
4. **Generation**: new primitives are appended to the `"default"` layer of the
   output image, on top of the inherited and cleaned upstream overlays.
5. **Copy to input** (`copy_overlays_to_input`): `_copy_overlays_to_input()`
   removes the tool's previous elements from the input image and appends the new
   ones, so downstream tools see them without accumulation between runs.

### Full flow per run

```python
output = VisionImage(...)
output.inherit_overlays_from(input)                       # 1. inherit
layer = output.get_overlay_layer("default", source_tool=self.name)  # 2. claim
layer.elements = [e for e in layer.elements if e.source_tool != self.name]  # 3. clean
layer.add_roi(...); layer.add_contour(...)                # 4. generate
self._copy_overlays_to_input(output, input)               # 5. publish to input
```

### Rendering

Overlays are only rendered for visualization, both on desktop and on the web.

**Desktop**: `ImageViewer` draws them with QPainter in `drawForeground()`
without modifying the pixels.

**Web HMI**: the frontend receives the overlays as vector data (JSON) and draws
them with the Canvas 2D API over the underlying image.

**Rasterizing overlays into the image buffer is not allowed.**
