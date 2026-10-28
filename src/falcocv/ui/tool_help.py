"""
Tool documentation data for FalcoCV.

Each tool entry is a dict with keys:
  name, category, purpose,
  inputs, outputs, params, result (optional),
  notes (optional), see_also (optional)
"""

TOOL_HELP = {}

# ── Lookup helpers ────────────────────────────────────────────────────────

TYPE_TO_HELP_KEY = {
    "ImageLoader": "Image Loader",
    "GigeCameraLoader": "GigE Camera",
    "DepthRangeLoader": "Depth/Range Loader",
    "FixtureOverlay": "Fixture Overlay",
    "BlobDetector": "Blob Detector",
    "FiltersTool": "Filters",
    "HistogramTool": "Histogram",
    "Histogram": "Histogram",
    "ImageCalculatorTool": "Image Calculator",
    "PatternMatcher": "Pattern Matcher",
    "ShapeMatch": "Shape Match",
    "BarcodeDecoder": "Barcode Decoder",
    "MeasurementTool": "Measurement",
    "PythonScriptTool": "Python Script",
    "EdgeDetection": "Edge Detection",
    "Block": "Block",
    "Classifier": "Classifier",
    "AnomalyDetector": "Anomaly Detector",
    "OCR": "OCR",
}


def get_doc_for_tool(tool) -> dict:
    """Look up the documentation entry for a BaseTool instance."""
    cls_name = tool.__class__.__name__
    key = TYPE_TO_HELP_KEY.get(cls_name, tool.name)
    return TOOL_HELP.get(key, {})


# ── Acquisition ────────────────────────────────────────────────────────────

TOOL_HELP["Image Loader"] = {
    "name": "Image Loader",
    "category": "Acquisition",
    "purpose": (
        "Loads image files from disk. Supports single-file loading or sequential "
        "folder browsing (useful for inspecting datasets frame by frame). "
        "Handles BGRA-to-BGR conversion automatically."
    ),
    "inputs": [],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Loaded image with BGRA→BGR conversion applied if needed."},
    ],
    "params": [
        {"name": "mode", "default": '"File"',
         "description": "'File' loads a single image. 'Folder' enables sequential browsing."},
        {"name": "file_path", "default": '""',
         "description": "Path to an image file (used in File mode)."},
        {"name": "folder_path", "default": '""',
         "description": "Path to a folder (used in Folder mode)."},
        {"name": "sort_by", "default": '"Name"',
         "description": "Sort order when browsing a folder: 'Name' or 'Date'."},
        {"name": "current_index", "default": "0",
         "description": "Index in the sorted file list. The tool advances this automatically on each execution."},
    ],
    "notes": (
        "Supports common formats: PNG, JPEG, BMP, TIFF. In Folder mode the tool increments "
        "current_index each time it runs (or when advance=True), making it easy to step through "
        "a dataset by connecting a trigger or using the Step/Advance buttons."
    ),
    "see_also": ["GigE Camera"],
}

TOOL_HELP["Depth/Range Loader"] = {
    "name": "Depth/Range Loader",
    "category": "Acquisition",
    "purpose": (
        "Loads single-channel depth / range images (TIFF, PNG, BMP, JPEG) as float32 arrays. "
        "Automatically converts multi-channel images to grayscale, applies a configurable scale "
        "factor, and optionally marks zero pixels as invalid (NaN). Can generate a 3D point cloud "
        "from the depth map using simple metric resolution (x_res, y_res, z_res) for interactive "
        "3D visualization in the built-in Viewer3D."
    ),
    "inputs": [],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Float32 single-channel depth map. Contains _3d_data (point cloud) if Generate 3D is enabled — toggle with 'Show 3D' checkbox in the viewer."},
    ],
    "params": [
        {"name": "file_path", "default": '""',
         "description": "Path to a depth/range image file on disk."},
        {"name": "scale", "default": "1.0",
         "description": "Multiplier applied to stored integer depths (e.g. mm-per-count for 16-bit exports)."},
        {"name": "zero_as_invalid", "default": "False",
         "description": "If True, pixels with value 0 become NaN in the float output."},
        {"name": "x_res", "default": "1.0",
         "description": "Physical resolution per column in X direction (e.g. mm/pixel)."},
        {"name": "y_res", "default": "1.0",
         "description": "Physical resolution per row in Y direction (e.g. mm/pixel)."},
        {"name": "z_res", "default": "1.0",
         "description": "Scale factor converting depth values to physical Z (e.g. mm/count)."},
        {"name": "z_offset", "default": "0.0",
         "description": "Subtracted from Z after scaling to center the cloud (e.g. depth_range)."},
        {"name": "generate_3d", "default": "False",
         "description": "If True, converts the depth map to a 3D point cloud and attaches it as _3d_data on the VisionImage."},
    ],
    "notes": (
        "The depth map is always loaded as float32. If the source image is in an integer format "
        "(uint8, uint16, int16), the scale parameter converts it: result = raw * scale. "
        "The 3D point cloud generation uses simple metric scaling (no pinhole model): "
        "x = col * x_res, y = row * y_res, z = depth * z_res - z_offset. "
        "Set generate_3d=True and provide the three resolutions to visualize the depth data "
        "as a 3D point cloud — toggle the 'Show 3D' checkbox in the viewer or use the workflow editor."
    ),
    "see_also": ["Image Loader", "Blob Detector", "Filters"],
}

TOOL_HELP["GigE Camera"] = {
    "name": "GigE Camera",
    "category": "Acquisition",
    "purpose": (
        "Acquires frames from a GigE Vision or USB3 Vision camera using the Harvesters "
        "GenTL producer. Supports hardware triggering, exposure control, and simulation "
        "mode for development without a physical camera."
    ),
    "inputs": [],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Live camera frame in BGR or grayscale format."},
    ],
    "params": [
        {"name": "camera_type", "default": '"Simulation"',
         "description": "'Hardware' connects to a real camera. 'Simulation' uses a virtual camera for testing."},
        {"name": "cti_path", "default": '""',
         "description": "Path to the GenTL producer .cti file (provider-specific transport layer)."},
        {"name": "camera_serial", "default": '""',
         "description": "Serial number of the camera to connect. Leave empty to auto-select."},
        {"name": "trigger_mode", "default": '"FreeRun"',
         "description": "'FreeRun' streams continuously. 'SoftwareTrigger' waits for a software signal. 'HardwareTrigger' waits for an external hardware signal."},
        {"name": "exposure_us", "default": "10000.0",
         "description": "Exposure time in microseconds."},
        {"name": "output_format", "default": '"BGR"',
         "description": "Pixel format: 'BGR' for color, 'Gray' for monochrome."},
    ],
    "notes": (
        "Thread-safe: camera acquisition runs in a separate thread to avoid blocking the UI. "
        "The tool uses a ring buffer of configurable size. In Simulation mode it generates "
        "test patterns so you can develop pipelines without a camera. "
        "Custom GenICam features can be set via the 'custom_features' dict in advanced use."
    ),
    "see_also": ["Image Loader", "Depth/Range Loader"],
}

# ── Vision Tools ───────────────────────────────────────────────────────────

TOOL_HELP["Blob Detector"] = {
    "name": "Blob Detector",
    "category": "Vision Tools",
    "purpose": (
        "Detects and analyzes blobs (connected components) in an image using thresholding "
        "and contour analysis. Supports fixed and adaptive thresholding, area-based filtering, "
        "and configurable result sorting."
    ),
    "inputs": [
        {"name": "input image", "type": "VisionImage / ndarray",
         "description": "Input image for blob detection."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Annotated image with contour, centroid, and label overlays."},
        {"name": "binary mask", "type": "VisionImage",
         "description": "Binary threshold mask used for contour detection. Same size as input."},
        {"name": "count", "type": "int",
         "description": "Number of blobs found after area filtering."},
        {"name": "results", "type": "ToolResults[BlobResult]",
         "description": "Collection of BlobResult items, one per detected blob."},
    ],
    "params": [
        {"name": "threshold_mode", "default": '"Fixed"',
         "description": "'Fixed' uses a global threshold value. 'Adaptive' uses local adaptive thresholding."},
        {"name": "threshold_value", "default": "128",
         "description": "Global threshold (0-255). Used only in Fixed mode."},
        {"name": "adaptive_block_size", "default": "11",
         "description": "Block size for adaptive thresholding (odd number, 3-99)."},
        {"name": "adaptive_const", "default": "2",
         "description": "Constant subtracted from the local mean in adaptive mode."},
        {"name": "polarity", "default": '"Bright Blobs"',
         "description": "'Bright Blobs' detects pixels above threshold. 'Dark Blobs' detects pixels below."},
        {"name": "area_min", "default": "100",
         "description": "Minimum blob area in pixels. Smaller blobs are discarded."},
        {"name": "area_max", "default": "1000000",
         "description": "Maximum blob area in pixels. Larger blobs are discarded."},
        {"name": "sort_mode", "default": '"Area Descending"',
         "description": "Sort order: None, Area Ascending, Area Descending, X Ascending, Y Ascending."},
        {"name": "output_mode", "default": '"Full Image"',
         "description": "'Full Image' outputs the original image with overlays. 'ROI Only' crops to the ROI."},
        {"name": "show_contours", "default": "True",
         "description": "Draw blob boundary contours on the output image."},
        {"name": "show_centroids", "default": "True",
         "description": "Mark the center point of each detected blob."},
        {"name": "show_labels", "default": "True",
         "description": "Show ID and area text next to each blob."},
        {"name": "show_search_roi", "default": "True",
         "description": "Show the search region overlay."},
        {"name": "roi", "default": "[0, 0, 0, 0, 0]",
         "description": "Search region [x, y, width, height, angle]. Zero width/height = full image."},
    ],
    "result": {
        "class": "BlobResult",
        "fields": [
            {"name": "id", "type": "int", "description": "Zero-based index after sorting."},
            {"name": "x", "type": "float", "description": "Centroid X coordinate (image pixels)."},
            {"name": "y", "type": "float", "description": "Centroid Y coordinate (image pixels)."},
            {"name": "area", "type": "float", "description": "Blob area in square pixels."},
        ],
    },
    "notes": (
        "The binary mask output is a full-size image with the ROI area filled and the rest black — "
        "useful for feeding into downstream tools. Coordinates are in image-pixel space, not fixture-relative."
    ),
    "see_also": ["Filters", "Pattern Matcher"],
}

TOOL_HELP["Filters"] = {
    "name": "Filters",
    "category": "Vision Tools",
    "purpose": (
        "Applies a sequential pipeline of preprocessing filters to the input image. "
        "Each filter operates on the result of the previous one, allowing complex "
        "processing chains like blur → threshold → morphology in a single tool. "
        "Supports Gaussian/Median/Bilateral blur, sharpen, morphology, threshold, "
        "CLAHE, and normalization. All filters operate within a global ROI and the "
        "result can be output as the full image (with ROI warped back) or ROI-only."
    ),
    "inputs": [
        {"name": "input image", "type": "VisionImage / ndarray",
         "description": "Input image to process."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Filtered result with optional overlay showing the ROI region."},
    ],
    "params": [
        {"name": "filters", "default": "[]",
         "description": "Ordered list of filter dicts. Each has 'enabled', 'type', and type-specific params."},
        {"name": "roi", "default": "[0, 0, 0, 0, 0]",
         "description": "Global region of interest [x, y, width, height, angle]. Zero size = full image."},
        {"name": "output_mode", "default": '"Full Image"',
         "description": "'Full Image' maps the filtered ROI back onto the original. 'ROI Only' returns just the patch."},
        {"name": "show_overlays", "default": "True",
         "description": "Show the ROI outline on the output image."},
    ],
    "result": None,
    "notes": (
        "Filters execute in order from top to bottom in the table. Unchecked filters are skipped. "
        "Each filter receives the output of the previous filter. The global ROI is applied once "
        "before the pipeline and the result is mapped back (if Full Image mode) or cropped (if ROI Only). "
        "Filter types: Gaussian Blur, Median Blur, Bilateral Filter, Sharpen, Sobel (X/Y/Magnitude/Angle), "
        "Canny Edge, Morphology, Threshold, CLAHE, Normalize, Equalize Histogram, Color Conversion (Gray/R/G/B/H/S/V/Lightness), "
        "Invert, Resize (Factor/Width/Height/Longest Side), Rotate/Flip."
    ),
    "see_also": ["Blob Detector", "Histogram"],
    "filters": {
        "gaussian_blur": {
            "name": "Gaussian Blur",
            "how_it_works": "Convolves the image with a 2D Gaussian kernel. The kernel is separable (applied in X then Y), making it fast even for large sizes.",
            "effect": "Removes Gaussian noise and fine detail. The result looks blurry or out of focus.",
            "params_detail": [
                {"name": "Kernel Size", "detail": "Neighbourhood size (odd). Larger = more blur. Range: ≥3."},
                {"name": "Sigma", "detail": "Standard deviation. Larger = more blur. Set to 0 for auto-sigma."},
            ],
            "when_to_use": "Noise reduction, removing texture before thresholding or edge detection, general smoothing.",
            "limitations": "Does not preserve edges — they blur uniformly. Use Bilateral or Median for edge preservation.",
        },
        "median_blur": {
            "name": "Median Blur",
            "how_it_works": "Replaces each pixel with the median value of its neighbours. A non-linear rank filter.",
            "effect": "Removes salt-and-pepper and impulse noise while preserving edges better than Gaussian blur.",
            "params_detail": [
                {"name": "Kernel Size", "detail": "Neighbourhood diameter (odd). Larger = more smoothing."},
            ],
            "when_to_use": "Removing dead pixels, dust spots, or any sparse impulse noise. Preprocessing before thresholding.",
            "limitations": "Slower than Gaussian for large kernels. Less effective on Gaussian noise.",
        },
        "bilateral": {
            "name": "Bilateral Filter",
            "how_it_works": "Each pixel is a weighted average of its neighbours. Weight = spatial Gaussian × range Gaussian (intensity difference). Pixels across strong edges get near-zero weight.",
            "effect": "Smooths uniform regions while preserving sharp edges. Produces a 'cartoon-like' result at high sigma_space and low sigma_color.",
            "params_detail": [
                {"name": "Diameter", "detail": "Kernel size. Larger = slower. ≤0 = auto. Typical: 5–9."},
                {"name": "Sigma Color", "detail": "Intensity deviation. High = more colours blend (less edge preservation). Typical: 10–100."},
                {"name": "Sigma Space", "detail": "Spatial deviation. High = blurrier. Typical: 10–100."},
            ],
            "when_to_use": "Noise reduction without losing edges, portrait retouching, preparing images for segmentation.",
            "limitations": "Computationally expensive for d > 9. Not real-time at high resolutions with large kernels.",
        },
        "sharpen": {
            "name": "Sharpen",
            "how_it_works": "Applies an unsharp-mask-style 3×3 kernel where the centre weight is boosted relative to its neighbours. Amount=0 gives identity (no change).",
            "effect": "Increases perceived sharpness by enhancing local contrast at edges.",
            "params_detail": [
                {"name": "Amount", "detail": "Sharpening strength. 0 = no change. Typical: 0.5–2.0. Higher values risk artifacts."},
            ],
            "when_to_use": "Compensating for lens softness, improving detail before inspection or pattern matching.",
            "limitations": "Amplifies noise. Can produce halos around strong edges at high Amount.",
        },
        "sobel": {
            "name": "Sobel",
            "how_it_works": "Computes the first spatial derivative using the Sobel operator. X mode convolves with [−1,0,+1], Y mode with its transpose. Magnitude = √(Gx²+Gy²). Angle = arctan2(Gy,Gx).",
            "effect": "Highlights edges: X shows vertical edges, Y shows horizontal edges, Magnitude shows edge strength regardless of direction, Angle shows direction encoded as intensity.",
            "params_detail": [
                {"name": "Mode", "detail": "X: vertical edges. Y: horizontal edges. Magnitude: edge strength. Angle: gradient direction (0–255 = −180°–+180°)."},
                {"name": "Kernel Size", "detail": "Derivative kernel (≥1, odd). 1 is ∂¹/∂x without smoothing. Larger values add Gaussian smoothing before differentiation."},
            ],
            "when_to_use": "Edge detection, texture analysis, feature extraction, gradient computation for further processing.",
            "limitations": "Sensitive to noise. Output is always uint8; true gradient magnitude is lost after normalisation.",
        },
        "canny": {
            "name": "Canny Edge",
            "how_it_works": "Multi-stage: Gaussian smoothing → Sobel gradient → non-maximum suppression (thins edges to 1 px) → double-threshold hysteresis. Weak edges connected to strong edges survive.",
            "effect": "Thin, binary edge map with good localisation. Only significant edges are kept.",
            "params_detail": [
                {"name": "Low Threshold", "detail": "Lower hysteresis bound. Edges below this are discarded. Typical: 20–80."},
                {"name": "High Threshold", "detail": "Upper hysteresis bound. Edges above this are kept as 'strong'. Typical: 60–240. Use ratio 1:2 or 1:3."},
                {"name": "Aperture", "detail": "Sobel kernel for gradient (3, 5, 7). Larger = more smoothing = fewer fine edges."},
            ],
            "when_to_use": "General edge detection for dimensional measurement, alignment, or input to further vision steps.",
            "limitations": "Parameter-dependent. Fixed thresholds fail under varying lighting. Aperture 7 loses fine detail.",
        },
        "morphology": {
            "name": "Morphology",
            "how_it_works": "Applies operations using a structuring element (kernel). Erode: min filter. Dilate: max filter. Open: erode→dilate. Close: dilate→erode. Gradient: dilate minus erode.",
            "effect": "Erode shrinks bright regions, Dilate grows them. Open removes small bright spots. Close fills small dark holes. Gradient extracts contours.",
            "params_detail": [
                {"name": "Operation", "detail": "Erode (shrink), Dilate (grow), Open (remove bright spots), Close (fill dark holes), Gradient (contours)."},
                {"name": "Kernel Shape", "detail": "Rectangle (sharp corners), Ellipse (smooth), Cross (diamond-like)."},
                {"name": "Kernel Size", "detail": "Structuring element size (≥3, odd). Larger = stronger effect."},
                {"name": "Iterations", "detail": "Number of times to apply (≥1). More iterations = stronger effect."},
            ],
            "when_to_use": "Cleaning binary masks, separating touching objects, extracting boundaries, removing noise after thresholding.",
            "limitations": "On grayscale images the effect is less predictable than on binary. Multiple iterations can be slow on large kernels.",
        },
        "threshold": {
            "name": "Threshold",
            "how_it_works": "Simple binarisation: pixel ≥ Value → 255, else → 0. Inverted reverses the logic.",
            "effect": "Produces a binary mask. Good vs bad regions separated by a single intensity cutoff.",
            "params_detail": [
                {"name": "Value", "detail": "Cutoff intensity (0–255). Adjust with slider."},
                {"name": "Inverted", "detail": "When checked, pixels ≥ value become 0 (black). Useful for dark-on-bright inspection."},
            ],
            "when_to_use": "Quick segmentation under controlled lighting, creating masks for blob detection, pass/fail area checks.",
            "limitations": "Requires uniform lighting and high contrast. Single threshold cannot separate more than two classes. Use CLAHE as preconditioner for uneven illumination.",
        },
        "clahe": {
            "name": "CLAHE",
            "how_it_works": "Divides the image into a grid, equalises each tile's histogram locally, clips bins above clip_limit and redistributes. Blends tile boundaries bilinearly.",
            "effect": "Enhances local contrast without over-amplifying noise. Reveals detail in both shadows and highlights.",
            "params_detail": [
                {"name": "Clip Limit", "detail": "Maximum histogram bin height. Higher = more contrast but also more noise. Typical: 1–4."},
                {"name": "Grid Size", "detail": "Number of tiles per dimension. Smaller = more local adaptation. Typical: 4–16."},
            ],
            "when_to_use": "Images with uneven lighting, low local contrast, microscope/X-ray, preprocessing before threshold or edge detection.",
            "limitations": "Single-channel only (converts to grayscale). May produce artefacts with very small grid sizes.",
        },
        "normalize": {
            "name": "Normalize",
            "how_it_works": "Linear stretch: I' = (I − I_min) × (Max − Min) / (I_max − I_min) + Min. Uses NORM_MINMAX scaling.",
            "effect": "Maps the current intensity range to an arbitrary target range. Can maximise or reduce contrast.",
            "params_detail": [
                {"name": "Min", "detail": "Target minimum intensity (0–255). Pixels at I_min become this value."},
                {"name": "Max", "detail": "Target maximum intensity (0–255). Pixels at I_max become this value."},
            ],
            "when_to_use": "Standardising contrast across multiple captures, preparing images for ML, visualisation.",
            "limitations": "Global only. A single outlier pixel can compress the rest of the range. No non-linear adjustment.",
        },
        "equalize": {
            "name": "Equalize Histogram",
            "how_it_works": "Computes the histogram, builds the cumulative distribution function (CDF), and remaps each pixel: new_value = CDF(old_value) × 255.",
            "effect": "Maximises global contrast automatically. The histogram becomes approximately flat. Dark images become brighter.",
            "params_detail": [],
            "when_to_use": "Low-contrast images, backlit scenes, automatic contrast improvement without tuning any parameters.",
            "limitations": "Global only — cannot handle uneven illumination. May amplify noise in flat regions. For non-uniform lighting use CLAHE.",
        },
        "color_conversion": {
            "name": "Color Conversion",
            "how_it_works": "Extracts a single channel from a colour image or converts to grayscale. RGB channels are split directly. HSV and HLS channels are computed via colour space conversion.",
            "effect": "Grayscale produces a standard monochrome image. R/G/B show the contribution of each sensor channel. H/S/V and Lightness reveal colour properties independent of brightness.",
            "params_detail": [
                {"name": "Mode", "detail": "Grayscale: standard BGR→Gray weights. Red/Green/Blue: single channel. Hue (HSV): colour type (0–179). Saturation (HSV): colour purity. Value (HSV): brightness. Lightness (HLS): perceived brightness."},
            ],
            "when_to_use": "Grayscale for all downstream processing that expects monochrome. Red/Green/Blue to isolate coloured features. Hue to segment by colour regardless of shading. Saturation to find vivid vs dull regions.",
            "limitations": "Hue values wrap (red appears at both 0 and 179). Saturation and Value are sensitive to lighting. Single-channel extraction discards information.",
        },
        "invert": {
            "name": "Invert",
            "how_it_works": "Applies a bitwise NOT: new_pixel = 255 − old_pixel (for uint8). Every channel is inverted independently.",
            "effect": "Creates a photographic negative. White becomes black, black becomes white.",
            "params_detail": [],
            "when_to_use": "Inverting binary masks, reversing polarity for display, preparing images for operations that expect dark-on-bright.",
            "limitations": "No parameters. Output data type matches input.",
        },
        "resize": {
            "name": "Resize",
            "how_it_works": "Scales the image using the selected interpolation method. Factor mode multiplies both dimensions. Width and Height modes preserve aspect ratio. Longest Side scales so the larger dimension equals the given value.",
            "effect": "Changes image resolution. Useful for speeding up downstream processing or matching sizes for image arithmetic.",
            "params_detail": [
                {"name": "Mode", "detail": "Factor: uniform scale (0.5 = half size). Width: set width, auto height. Height: set height, auto width. Longest Side: fit within a square limit."},
                {"name": "Value", "detail": "Scale factor (Factor mode) or target pixel count (Width/Height/Longest Side modes)."},
                {"name": "Interpolation", "detail": "Nearest (blocky, fast). Linear (smooth, good default). Cubic (sharper, slower). Lanczos (best quality, slowest)."},
            ],
            "when_to_use": "Downscaling for performance in high-speed inspection, upscaling for display, matching image sizes before arithmetic operations.",
            "limitations": "Upscaling cannot add real detail. Nearest interpolation creates jagged edges. Lanczos is computationally expensive.",
        },
        "rotate_flip": {
            "name": "Rotate / Flip",
            "how_it_works": "Applies lossless 90° increments via cv2.rotate or axis reflection via cv2.flip.",
            "effect": "Rotate 90° CW/CCW turns the image a quarter turn. 180° flips upside-down. Flip Horizontal mirrors left↔right. Flip Vertical mirrors top↔bottom. Flip Both does both simultaneously.",
            "params_detail": [
                {"name": "Mode", "detail": "90° CW (clockwise), 90° CCW (counter-clockwise), 180° (half-turn), Flip H (left-right mirror), Flip V (top-bottom mirror), Flip Both (180° mirror)."},
            ],
            "when_to_use": "Correcting camera orientation, augmenting training data, mirroring for UI display.",
            "limitations": "90° rotations change width and height. Flipping does not. Non-90° rotation requires the Resize filter with a custom warp.",
        },
    },
}

TOOL_HELP["Histogram"] = {
    "name": "Histogram",
    "category": "Vision Tools",
    "purpose": (
        "Computes the image histogram and statistical descriptors (mean, std dev, min, max, median). "
        "Auto-detects the bit depth (8/10/12/14/16-bit) and selects appropriate bin count."
    ),
    "inputs": [
        {"name": "input image", "type": "VisionImage / ndarray",
         "description": "Input image for histogram analysis."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Visual plot of the histogram as an image."},
        {"name": "results", "type": "ToolResults",
         "description": "Results containing mean, std_dev, min_val, max_val, median_val, bins."},
    ],
    "params": [
        {"name": "bin_mode", "default": '"Auto"',
         "description": "'Auto' selects bin count based on bit depth (256 for 8-bit, 4096 for 12-bit, etc.)."},
        {"name": "num_bins", "default": "256",
         "description": "Custom bin count when bin_mode is not Auto."},

    ],
    "notes": (
        "The histogram image output is a plot (not the raw data). Use the results "
        "output to access statistical values in downstream tools."
    ),
    "see_also": ["Filters", "Blob Detector"],
}

TOOL_HELP["Pattern Matcher"] = {
    "name": "Pattern Matcher",
    "category": "Vision Tools",
    "purpose": (
        "Industrial multi-scale template matching with rotation and scale invariance. "
        "Searches for a reference pattern (template) in the input image and returns matches "
        "with position, angle, scale, and score. Supports user-defined origin point for "
        "fixture calibration."
    ),
    "inputs": [
        {"name": "input image", "type": "VisionImage / ndarray",
         "description": "Search image. Must have been trained (template captured) before matching."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Annotated image with match bounding boxes, origins, and labels."},
        {"name": "count", "type": "int",
         "description": "Number of matches found."},
        {"name": "trained", "type": "bool",
         "description": "True if a template has been trained."},
        {"name": "results", "type": "ToolResults[PatternMatchResult]",
         "description": "Collection of match results."},
    ],
    "params": [
        {"name": "train_roi", "default": "[0, 0, 0, 0, 0]",
         "description": "ROI for capturing the reference template."},
        {"name": "search_roi", "default": "[0, 0, 0, 0, 0]",
         "description": "ROI for searching. Zero = full image."},
        {"name": "score_threshold", "default": "0.7",
         "description": "Minimum match score (0.0-1.0). Lower = more tolerant."},
        {"name": "angle_start", "default": "0",
         "description": "Start of rotation search range (degrees)."},
        {"name": "angle_end", "default": "360",
         "description": "End of rotation search range (degrees)."},
        {"name": "angle_step", "default": "1",
         "description": "Rotation angle step (degrees). Smaller = more precise but slower."},
        {"name": "scale_start", "default": "0.9",
         "description": "Start of scale search range (fraction of template size)."},
        {"name": "scale_end", "default": "1.1",
         "description": "End of scale search range."},
        {"name": "scale_step", "default": "0.05",
         "description": "Scale step. Smaller = more precise but slower."},
        {"name": "pyramid_levels", "default": "2",
         "description": "Image pyramid depth for coarse-to-fine search."},
        {"name": "max_matches", "default": "5",
         "description": "Maximum number of matches to return."},
        {"name": "is_trained", "default": "False",
         "description": "Whether a template has been captured."},
        {"name": "origin_x", "default": "0.0",
         "description": "User-defined origin X offset from template center."},
        {"name": "origin_y", "default": "0.0",
         "description": "User-defined origin Y offset from template center."},
        {"name": "act_as_fixture", "default": "False",
         "description": "If True, the match position/angle is applied as a fixture transform on the output image."},
    ],
    "result": {
        "class": "PatternMatchResult",
        "fields": [
            {"name": "id", "type": "int", "description": "Match index."},
            {"name": "x", "type": "float", "description": "Match center X (image pixels)."},
            {"name": "y", "type": "float", "description": "Match center Y (image pixels)."},
            {"name": "angle", "type": "float", "description": "Rotation angle of the match (degrees)."},
            {"name": "scale", "type": "float", "description": "Scale of the match relative to the template."},
            {"name": "score", "type": "float", "description": "Match quality score (0.0-1.0)."},
        ],
    },
    "notes": (
        "Training captures the reference template. The search then finds all instances of that template "
        "in the search ROI. The origin point lets you define a specific reference location (e.g., a "
        "corner of the part) rather than the center of the template. "
        "When act_as_fixture is enabled, the output image's fixture (x, y, angle) is set from the "
        "best match, so downstream tools operate in the part's coordinate system."
    ),
    "see_also": ["Blob Detector", "Python Script"],
}

TOOL_HELP["Barcode Decoder"] = {
    "name": "Barcode Decoder",
    "category": "Vision Tools",
    "purpose": (
        "Detects and decodes QR codes, 1D barcodes (UPC, EAN, Code128, etc.), DataMatrix codes, "
        "and ArUco markers. Supports multiple code types simultaneously in Auto mode, with "
        "deduplication of overlapping detections."
    ),
    "inputs": [
        {"name": "input image", "type": "VisionImage / ndarray",
         "description": "Input image containing codes to decode."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Annotated image with found-code overlays (bounding boxes, centers, labels)."},
        {"name": "count", "type": "int",
         "description": "Number of codes detected and decoded."},
        {"name": "results", "type": "ToolResults[CodeResult]",
         "description": "Collection of decoded code results."},
    ],
    "params": [
        {"name": "code_type", "default": '"Auto"',
         "description": "Detection mode: Auto, QR Code, Barcode 1D, DataMatrix, or ArUco."},
        {"name": "roi", "default": "[0, 0, 0, 0, 0]",
         "description": "Search region [x, y, width, height, angle]."},
        {"name": "qr_params.eps_x", "default": "0.1",
         "description": "QR code X-axis noise tolerance."},
        {"name": "qr_params.eps_y", "default": "0.1",
         "description": "QR code Y-axis noise tolerance."},
        {"name": "datamatrix_params.timeout_ms", "default": "500",
         "description": "DataMatrix decode timeout in milliseconds."},
        {"name": "datamatrix_params.shrink", "default": "2",
         "description": "Downsample factor for DataMatrix scanning. Higher = faster but may miss small codes."},
        {"name": "datamatrix_params.max_count", "default": "1",
         "description": "Max codes to find. 1 = stop as soon as one is found."},
        {"name": "datamatrix_params.enable_sharpening", "default": "True",
         "description": "Apply unsharp mask before DataMatrix detection."},
        {"name": "datamatrix_params.enable_adaptive_thresh", "default": "False",
         "description": "Apply adaptive threshold before DataMatrix detection."},
        {"name": "barcode_1d_params", "default": "—",
         "description": "No configurable parameters for 1D detection."},
        {"name": "aruco_params.dict_name", "default": '"DICT_6X6_250"',
         "description": "ArUco dictionary."},
        {"name": "aruco_params.cornerRefinementMethod", "default": "0",
         "description": "0=NONE, 1=SUBPIX, 2=CONTOUR, 3=APRILTAG."},
        {"name": "preprocessing.strategy", "default": '"no_clahe_then_fallback"',
         "description": "CLAHE strategy: no_clahe_then_fallback, always_clahe, or always_no_clahe."},
    ],
    "result": {
        "class": "CodeResult",
        "fields": [
            {"name": "id", "type": "int", "description": "Result index."},
            {"name": "x", "type": "float", "description": "Center X coordinate (image pixels)."},
            {"name": "y", "type": "float", "description": "Center Y coordinate (image pixels)."},
            {"name": "angle", "type": "float", "description": "Rotation of the detected code (degrees)."},
            {"name": "text", "type": "str", "description": "Decoded text/string content."},
            {"name": "format", "type": "str", "description": "Code format (QR, Barcode 1D, DataMatrix, ArUco:dict)."},
            {"name": "quality", "type": "float", "description": "Decode quality score."},
        ],
    },
    "notes": (
        "In Auto mode, all detector backends run with IoU-based deduplication. "
        "Select a specific code type to avoid running backends unnecessarily. "
        "DataMatrix detection can struggle with large full-resolution images. "
        "Use the shrink parameter to speed up full-image scans. "
        "ArUco markers support 20 dictionary types plus AprilTag variants."
    ),
    "see_also": ["Pattern Matcher"],
    "barcode_types": {
        "auto": {
            "name": "Auto",
            "how_it_works": "Runs all available detectors (QR, 1D, DataMatrix, ArUco) on the image and deduplicates overlapping detections using IoU.",
            "effect": "Finds any supported code type in a single pass.",
            "params_detail": [],
            "when_to_use": "When you don't know which code type to expect, or multiple types may appear.",
            "limitations": "Slower than selecting a specific type. May return false positives from overlapping detectors.",
        },
        "qr": {
            "name": "QR Code",
            "how_it_works": "Locates finder patterns (position detection patterns) and decodes the data module grid.",
            "effect": "Decodes QR codes with position, rotation, and content.",
            "params_detail": [
                {"name": "epsX/Y", "detail": "Noise tolerance. Higher values ignore more distortion. Typical: 0.05-0.5."},
            ],
            "when_to_use": "Tracking, product identification, URLs, serial numbers.",
            "limitations": "Requires good contrast and minimal perspective distortion. Large QR codes at distance may fail.",
        },
        "barcode_1d": {
            "name": "Barcode 1D",
            "how_it_works": "Decodes linear barcodes by detecting patterns of alternating bars and spaces. No user-configurable parameters.",
            "effect": "Decodes UPC, EAN, Code128, Code39, and other common 1D formats.",
            "params_detail": [],
            "when_to_use": "Retail products, logistics labels, industry-standard barcodes.",
            "limitations": "Performance depends on image resolution and contrast.",
        },
        "datamatrix": {
            "name": "DataMatrix",
            "how_it_works": "Locates and decodes DataMatrix codes using timing patterns and finder patterns. Supports square and rectangular variants.",
            "effect": "Decodes DataMatrix codes with position, rotation, and content.",
            "params_detail": [
                {"name": "Timeout", "detail": "Maximum decode time in ms. Increase for large images."},
                {"name": "Shrink", "detail": "Downsample factor. 1=full res, 2=half. Higher=faster but may miss small codes."},
                {"name": "Max Count", "detail": "Stop after finding this many codes. 1=fastest."},
                {"name": "Enable Sharpening", "detail": "Apply unsharp mask before detection. Helps with blurry codes."},
                {"name": "Enable Adaptive Threshold", "detail": "Use adaptive threshold. Helps with uneven lighting."},
            ],
            "when_to_use": "Electronics manufacturing, medical devices, automotive DPM marks.",
            "limitations": "Large full-resolution scans can be slow. Use Shrink and Max Count for speed.",
        },
        "aruco": {
            "name": "ArUco",
            "how_it_works": "Detects binary square markers from a predefined dictionary. Each marker has a unique ID encoded in a grid pattern.",
            "effect": "Returns marker ID, corners, and pose information.",
            "params_detail": [
                {"name": "Dictionary", "detail": "Marker dictionary (DICT_4X4_50 to DICT_7X7_1000, plus AprilTag). Larger = more markers but harder to detect."},
                {"name": "Corner Refinement", "detail": "0=None, 1=Subpixel, 2=Contour, 3=AprilTag. Higher = smoother corners."},
            ],
            "when_to_use": "Camera calibration, augmented reality, robot guidance, multi-marker tracking.",
            "limitations": "Only works with ArUco markers (binary square pattern). Larger dictionaries are more prone to false positives at low resolution.",
        },
    },
}

TOOL_HELP["Measurement"] = {
    "name": "Measurement",
    "category": "Vision Tools",
    "purpose": (
        "Performs geometric measurements on overlay primitives (points, lines, contours, "
        "circles, ROIs) using a table-driven interface. Each row defines a measurement "
        "mode (distance, angle, area, perimeter, radius, diameter, width, or height) and "
        "selects the overlay elements to measure. Results can be converted from pixels "
        "to real-world units via the pixels-per-mm calibration."
    ),
    "inputs": [
        {"name": "input image", "type": "VisionImage / ndarray",
         "description": "Input image containing overlay primitives to measure."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Annotated image with measurement overlays."},
        {"name": "count", "type": "int",
         "description": "Number of measurement results."},
        {"name": "results", "type": "ToolResults[MeasurementResult]",
         "description": "Collection of MeasurementResult items."},
    ],
    "params": [
        {"name": "overlay_measurements", "default": "[]",
         "description": "List of measurement rows. Each row has: enabled, mode, source1/2, ref_id1/2, type1/2, strategy1/2, color, thickness."},
        {"name": "pixels_per_mm", "default": "1.0",
         "description": "Calibration factor. Results are reported in mm when set, otherwise in pixels."},
        {"name": "show_overlays", "default": "True",
         "description": "Show all overlay primitives in the viewer."},
        {"name": "show_labels", "default": "True",
         "description": "Show measurement value labels."},
    ],
    "result": {
        "class": "MeasurementResult",
        "fields": [
            {"name": "id", "type": "int", "description": "Result index."},
            {"name": "type", "type": "str", "description": "Measurement type (distance, angle, area, perimeter, radius, diameter, width, height)."},
            {"name": "value", "type": "float", "description": "Measured value in calibrated units."},
            {"name": "unit", "type": "str", "description": "Unit (mm, mm², deg, or px)."},
            {"name": "x1, y1", "type": "float", "description": "First reference point in image pixels."},
            {"name": "x2, y2", "type": "float", "description": "Second reference point in image pixels."},
            {"name": "angle", "type": "float", "description": "Angle in degrees (0 = horizontal right)."},
            {"name": "label", "type": "str", "description": "Descriptive label."},
        ],
    },
    "notes": (
        "Measurements operate on overlay primitives generated by other tools (edge detection, "
        "blob detector, pattern matcher, etc.). The measurement mode determines which primitives "
        "are compatible: distance works with most types, angle requires two lines, single-element "
        "modes (area, perimeter, radius, diameter, width, height) operate on one primitive. "
        "Set pixels_per_mm for real-world units; leave at 1.0 for pixel measurements."
    ),
    "see_also": ["Blob Detector", "Pattern Matcher", "Filters"],
    "measurement_modes": {
        "distance": {
            "name": "Distance",
            "how_it_works": "Measures the Euclidean distance between two points selected from overlay primitives. Each point can use a different strategy (center, closest, farthest, edge0, edge1) to determine the reference location.",
            "effect": "Returns the distance in mm (if calibrated) between two reference points.",
            "params_detail": [
                {"name": "Element 1/2", "detail": "Overlay primitive (point, axes, line, circle, ROI, contour, polygon)."},
                {"name": "Strategy", "detail": "How the point is selected: center, closest, farthest, edge0, edge1."},
            ],
            "when_to_use": "Measuring gap widths, part dimensions, feature spacing.",
            "limitations": "Accuracy depends on pixels_per_mm calibration. Requires two defined overlay primitives.",
        },
        "angle": {
            "name": "Angle",
            "how_it_works": "Computes the angle between two line primitives. Strategies determine which reference point on each line to use.",
            "effect": "Returns the angle in degrees between the two lines (0° = parallel, 90° = perpendicular).",
            "params_detail": [
                {"name": "Element 1/2", "detail": "Line overlay primitive only."},
                {"name": "Strategy", "detail": "How the line reference is selected (center, closest, farthest, edge0, edge1)."},
            ],
            "when_to_use": "Checking part alignment, perpendicularity, chamfer angles.",
            "limitations": "Both elements must be lines. Result is in degrees, not mm.",
        },
        "area": {
            "name": "Area",
            "how_it_works": "Computes the area enclosed by a contour, polygon, circle, or ROI using the shoelace formula or πr².",
            "effect": "Returns area in mm² (if calibrated) or px².",
            "params_detail": [
                {"name": "Element", "detail": "Single closed primitive (ROI, contour, polygon, circle)."},
            ],
            "when_to_use": "Measuring defect sizes, part surface area, hole areas.",
            "limitations": "Only works with closed primitives. Not for open contours.",
        },
        "perimeter": {
            "name": "Perimeter",
            "how_it_works": "Computes the total perimeter of a closed contour, polygon, or circle using arc length or 2πr.",
            "effect": "Returns the perimeter length in mm (if calibrated) or px.",
            "params_detail": [
                {"name": "Element", "detail": "Single closed primitive (contour, polygon, circle)."},
            ],
            "when_to_use": "Measuring part outlines, boundary lengths.",
            "limitations": "Only works with closed primitives.",
        },
        "radius": {
            "name": "Radius",
            "how_it_works": "Extracts the radius from a circle primitive.",
            "effect": "Returns the radius in mm (if calibrated) or px.",
            "params_detail": [
                {"name": "Element", "detail": "Circle primitive only."},
            ],
            "when_to_use": "Measuring hole radii, fillet sizes, circular features.",
            "limitations": "Only works with circle primitives.",
        },
        "diameter": {
            "name": "Diameter",
            "how_it_works": "Extracts the diameter (2 × radius) from a circle primitive.",
            "effect": "Returns the diameter in mm (if calibrated) or px.",
            "params_detail": [
                {"name": "Element", "detail": "Circle primitive only."},
            ],
            "when_to_use": "Measuring hole diameters, pin diameters, circular part dimensions.",
            "limitations": "Only works with circle primitives. Value = 2 × radius.",
        },
        "width": {
            "name": "Width",
            "how_it_works": "Measures the width of a rectangular ROI primitive along its short axis.",
            "effect": "Returns the width in mm (if calibrated) or px.",
            "params_detail": [
                {"name": "Element", "detail": "ROI primitive only."},
            ],
            "when_to_use": "Measuring rectangular part widths, gap widths in rectangular regions.",
            "limitations": "Only works with ROI primitives. Width = shorter dimension of the ROI.",
        },
        "height": {
            "name": "Height",
            "how_it_works": "Measures the height of a rectangular ROI primitive along its long axis.",
            "effect": "Returns the height in mm (if calibrated) or px.",
            "params_detail": [
                {"name": "Element", "detail": "ROI primitive only."},
            ],
            "when_to_use": "Measuring rectangular part heights, strip lengths.",
            "limitations": "Only works with ROI primitives. Height = longer dimension of the ROI.",
        },
    },
}

TOOL_HELP["Edge Detection"] = {
    "name": "Edge Detection",
    "category": "Vision Tools",
    "purpose": (
        "1D caliper tool that detects edges along a scan line projected across a ROI. "
        "Supports single-edge detection (one transition) or edge-pair detection (width between "
        "two opposite-polarity transitions). Ideal for measuring features like silicone cords, "
        "gap widths, and any edge-to-edge distance. Configurable polarity, threshold, and sub-pixel "
        "interpolation for high-accuracy results."
    ),
    "inputs": [
        {"name": "input image", "type": "VisionImage / ndarray",
         "description": "Input image containing features to detect."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Annotated image with edge overlays, direction arrow, and width line."},
        {"name": "count", "type": "int",
         "description": "Number of results (0 or 1)."},
        {"name": "results", "type": "ToolResults[EdgePairResult | SingleEdgeResult]",
         "description": "Collection of edge results. Single Edge mode: one SingleEdgeResult with "
                        "(x, y, score, polarity). Edge Pair mode: one EdgePairResult with "
                        "(x1, y1, x2, y2, width, score1, score2)."},
    ],
    "params": [
        {"name": "detect_mode", "default": '"Single Edge"',
         "description": "'Single Edge' finds one edge transition. 'Edge Pair' finds a pair of "
                        "edges and measures the width between them."},
        {"name": "edge_selector", "default": '"First"',
         "description": "Edge selection strategy for Single Edge mode: 'First', 'Last', or 'Strongest'."},
        {"name": "edge1_polarity", "default": '"Dark→Light"',
         "description": "Polarity of the first edge in Edge Pair mode: 'Dark→Light', "
                        "'Light→Dark', or 'Any'."},
        {"name": "edge2_polarity", "default": '"Light→Dark"',
         "description": "Polarity of the second edge in Edge Pair mode."},
        {"name": "single_polarity", "default": '"Any"',
         "description": "Polarity for Single Edge mode: 'Dark→Light', 'Light→Dark', or 'Any'."},
        {"name": "edge_threshold", "default": "30",
         "description": "Minimum gradient magnitude to consider as an edge (1-255)."},
        {"name": "projection_half_height", "default": "2",
         "description": "Number of pixels to average perpendicularly to the scan direction. "
                        "Higher values smooth noise but reduce spatial resolution."},
        {"name": "min_edge_width", "default": "3",
         "description": "Minimum distance in pixels between the two edges (Edge Pair mode)."},
        {"name": "max_edge_width", "default": "9999",
         "description": "Maximum distance in pixels between the two edges (Edge Pair mode)."},
        {"name": "subpixel", "default": "True",
         "description": "If True, edge positions are interpolated using parabolic peak fitting "
                        "for sub-pixel accuracy."},
        {"name": "roi", "default": "[0, 0, 0, 0, 0]",
         "description": "Region of interest [x, y, width, height, angle] in fixture coords."},
        {"name": "show_search_roi", "default": "True",
         "description": "Draw the ROI outline on the output image."},
        {"name": "show_direction_arrow", "default": "True",
         "description": "Draw a directional arrow indicating scan direction along the ROI."},
        {"name": "show_edge_lines", "default": "True",
         "description": "Draw transverse lines at each detected edge."},
        {"name": "show_edge_points", "default": "True",
         "description": "Draw points at the center of each detected edge."},
        {"name": "show_labels", "default": "True",
         "description": "Show text labels with coordinates and polarity."},
        {"name": "show_width_line", "default": "True",
         "description": "Draw the width line connecting both edges with a label (Edge Pair mode)."},
    ],
    "result": {
        "class": "EdgePairResult / SingleEdgeResult",
        "fields": [
            {"name": "id", "type": "int", "description": "Result index."},
            {"name": "x, y", "type": "float", "description": "Edge center coordinates (image pixels) — for SingleEdgeResult."},
            {"name": "x1, y1, x2, y2", "type": "float", "description": "Edge pair center coordinates — for EdgePairResult."},
            {"name": "width", "type": "float", "description": "Width in pixels between edge pair."},
            {"name": "score", "type": "float", "description": "Edge strength (gradient magnitude)."},
            {"name": "polarity", "type": "str", "description": "Edge polarity: 'Dark→Light', 'Light→Dark', or 'Any'."},
        ],
    },
    "notes": (
        "The algorithm projects a band of pixels perpendicular to the scan axis into a 1D intensity "
        "profile, then finds edges by gradient polarity. This is orders of magnitude faster than "
        "running a full 2D edge detector (Canny/Sobel) on the entire ROI. "
        "Rotation is handled via affine warping during ROI extraction. "
        "The scan direction is always along the ROI's long axis. Arrow points from start to end. "
        "Sub-pixel interpolation uses parabolic fitting around the peak gradient sample."
    ),
    "see_also": ["Blob Detector", "Measurement", "Pattern Matcher"],
}

# ── Logic / Custom ─────────────────────────────────────────────────────────

TOOL_HELP["Python Script"] = {
    "name": "Python Script",
    "category": "Logic",
    "purpose": (
        "Runs user-defined Python code as part of the vision pipeline. Supports dynamic "
        "input/output ports, a built-in debugger (breakpoints, step-over, step-into, variable "
        "inspection), and access to the full VisionImage API including overlays and fixtures."
    ),
    "inputs": [],
    "outputs": [],
    "params": [
        {"name": "script_code", "default": '""',
         "description": "Embedded Python code. The template shows the expected CustomTool class structure."},
        {"name": "script_path", "default": '""',
         "description": "Path to an external Python file (alternative to embedded code)."},
        {"name": "input_ports", "default": "[]",
         "description": "List of input port names. Each port becomes an input in the script."},
        {"name": "output_ports", "default": "[]",
         "description": "List of output port names. The script sets these via self.set_output()."},
    ],
    "notes": (
        "The script must define a CustomTool subclass with __init__() and process(inputs) methods. "
        "Access inputs by name via the inputs dict. Set outputs with self.set_output(name, value). "
        "Draw overlays with self.draw_*() methods. "
        "The integrated debugger supports breakpoints (click line numbers), watch expressions, "
        "step-over (skip into function calls), step-into, and step-out. "
        "Breakpoints persist while the config window is open and are auto-cleaned when it closes."
    ),
    "see_also": ["Block"],
}

TOOL_HELP["Block"] = {
    "name": "Block",
    "category": "Logic",
    "purpose": (
        "Groups a sub-workflow of tools into a single reusable block. Allows creating "
        "encapsulated processing pipelines with defined inputs and outputs, enabling "
        "modular design and reuse of common vision routines."
    ),
    "inputs": [],
    "outputs": [],
    "params": [
        {"name": "port_metadata", "default": "{}",
         "description": "Dict with 'inputs' and 'outputs' lists defining the block's external interface."},
    ],
    "notes": (
        "Double-click a Block to edit its internal workflow. The block's inputs and outputs "
        "map to the external ports shown in the workflow tree. Blocks can be nested (a block "
        "inside another block) for hierarchical pipeline organization."
    ),
    "see_also": ["Python Script"],
}

TOOL_HELP["Image Calculator"] = {
    "name": "Image Calculator",
    "category": "Vision Tools",
    "purpose": (
        "Performs arithmetic, bitwise, and blending operations between two images or between "
        "an image and a constant. Operations execute sequentially in a pipeline, each receiving "
        "the result of the previous operation. The tool has two input ports (image_A and image_B) — operations "
        "that need only one image use Image A. When sizes differ between A and B, Image B is "
        "automatically resized to match A."
    ),
    "inputs": [
        {"name": "image_A", "type": "VisionImage / ndarray",
         "description": "Primary input image. Used for all operations."},
        {"name": "image_B", "type": "VisionImage / ndarray",
         "description": "Secondary input image (optional). Used by binary operations."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Result of the sequential operation pipeline."},
    ],
    "params": [
        {"name": "ops", "default": "[]",
         "description": "Ordered list of operation dicts. Each has 'enabled', 'type', and type-specific params."},
    ],
    "result": None,
    "notes": (
        "Operations execute from top to bottom. Unchecked steps are skipped. "
        "Image B is reused as supplied in every step — it is not overwritten by intermediate results. "
        "For single-image operations (Bitwise NOT, Add/Multiply Constant), only Image A is used. "
        "Integer overflow is handled by clipping to [0, 255]. Division uses a small epsilon to avoid division by zero."
    ),
    "see_also": ["Filters", "Python Script"],
    "calc_ops": {
        "add": {
            "name": "Add (A + B)",
            "how_it_works": "Element-wise addition of A and B. Result = clamp(A + B, 0, 255).",
            "effect": "Brighter image. Combining two exposures or adding a pattern to an image.",
            "params_detail": [],
            "when_to_use": "Blending exposures, adding a calibration pattern, creating composite images.",
            "limitations": "Clipping at 255 loses information in bright regions. Consider Weighted Add for controlled blending.",
        },
        "subtract": {
            "name": "Subtract",
            "how_it_works": "Element-wise subtraction. Normal: A − B. Reverse: B − A. Clamped to [0, 255].",
            "effect": "Darker image. Highlights differences between two images (background subtraction).",
            "params_detail": [
                {"name": "Reverse (B − A)", "detail": "Swap the operands. Useful when B contains brighter features of interest."},
            ],
            "when_to_use": "Background subtraction, defect detection (difference from reference), motion detection.",
            "limitations": "Lossy for signed differences (only positive values survive). Use Abs Difference to preserve magnitude regardless of polarity.",
        },
        "abs_diff": {
            "name": "Abs Difference",
            "how_it_works": "Element-wise |A − B| using cv2.absdiff. Always positive.",
            "effect": "Highlights any difference between two images regardless of direction.",
            "params_detail": [],
            "when_to_use": "Change detection, alignment verification, comparing a part to its reference.",
            "limitations": "Not suitable when the sign of the difference matters (e.g. under vs over exposure).",
        },
        "multiply": {
            "name": "Multiply (A * B)",
            "how_it_works": "Element-wise multiply: Result = (A × B) / 255. Clamped to [0, 255].",
            "effect": "Darker image. Can be used as a mask: white region of B passes A, black region blocks A.",
            "params_detail": [],
            "when_to_use": "Applying a mask, vignette correction, modulating one image by another.",
            "limitations": "Values are always scaled down. For brighter results use Add with a constant offset.",
        },
        "divide": {
            "name": "Divide",
            "how_it_works": "Element-wise division: A / B or B / A, multiplied by a scale factor. A small epsilon prevents division by zero.",
            "effect": "Ratio image. Useful for flat-field correction or normalising by a reference.",
            "params_detail": [
                {"name": "Reverse (B / A)", "detail": "Swap numerator and denominator."},
                {"name": "Scale", "detail": "Multiplier applied to the quotient. Typical: 1.0."},
            ],
            "when_to_use": "Flat-field correction, shading compensation, colour ratio analysis.",
            "limitations": "Very sensitive to noise in dark (near-zero) regions. Set Scale to keep values in [0, 255] range.",
        },
        "max": {
            "name": "Max (A, B)",
            "how_it_works": "Element-wise maximum per pixel and channel.",
            "effect": "Brightest regions from both images are preserved.",
            "params_detail": [],
            "when_to_use": "Combining focus stacks, keeping the best-lit regions from multiple captures.",
            "limitations": "No blending — the transition between images is abrupt at pixel boundaries.",
        },
        "min": {
            "name": "Min (A, B)",
            "how_it_works": "Element-wise minimum per pixel and channel.",
            "effect": "Darkest regions from both images are preserved.",
            "params_detail": [],
            "when_to_use": "Removing hot pixels, combining dark frames, keeping shadow detail.",
            "limitations": "No blending — hard transitions at pixel boundaries.",
        },
        "weighted_add": {
            "name": "Weighted Add",
            "how_it_works": "Result = α × A + β × B + γ. Uses cv2.addWeighted.",
            "effect": "Alpha-blended composite where α controls the visibility of A and β controls B. Gamma is a global brightness offset.",
            "params_detail": [
                {"name": "Alpha (A weight)", "detail": "Contribution of Image A. 0 = invisible, 1 = full. Typical: 0.5."},
                {"name": "Beta (B weight)", "detail": "Contribution of Image B. 0 = invisible, 1 = full. Typical: 0.5."},
                {"name": "Gamma (offset)", "detail": "Constant added to every pixel after mix. Typical: 0."},
            ],
            "when_to_use": "Transparent overlays (α + β = 1), fading between exposures, HIS display, adding annotations with adjustable opacity.",
            "limitations": "Values outside [0, 255] clip. Use small α+β to keep the result in range.",
        },
        "bitwise_and": {
            "name": "Bitwise AND",
            "how_it_works": "Pixel-wise binary AND of A and B. For uint8: operation is per-channel.",
            "effect": "Keeps only the features present in both images. Darkens the result.",
            "params_detail": [],
            "when_to_use": "Masking: B is a binary mask and A is the image. Only pixels white in both survive.",
            "limitations": "Grayscale input is treated as a single-channel bitmask. For soft masking use Multiply.",
        },
        "bitwise_or": {
            "name": "Bitwise OR",
            "how_it_works": "Pixel-wise binary OR of A and B.",
            "effect": "Combines bright features from both images. Result is as bright or brighter than either input.",
            "params_detail": [],
            "when_to_use": "Union of two masks, combining segmentation results, superimposing binary regions.",
            "limitations": "Not suitable for blending photographic content — use Add or Weighted Add instead.",
        },
        "bitwise_xor": {
            "name": "Bitwise XOR",
            "how_it_works": "Pixel-wise binary exclusive-OR of A and B.",
            "effect": "Highlights exactly where A and B differ. Good for change detection.",
            "params_detail": [],
            "when_to_use": "Comparing two binary masks, detecting edges of differences, toggle display.",
            "limitations": "Photographic content yields unintelligible results. Best on binary images.",
        },
        "bitwise_not": {
            "name": "Bitwise NOT",
            "how_it_works": "Inverts every bit: result = 255 − pixel (for uint8). Single-image operation (uses A only).",
            "effect": "Photographic negative. White ↔ black.",
            "params_detail": [],
            "when_to_use": "Inverting masks, reversing polarity before binary morphology, preparing inputs for downstream tools that expect a given polarity.",
            "limitations": "Single-image only. Image B is ignored.",
        },
        "add_constant": {
            "name": "Add Constant",
            "how_it_works": "Adds a fixed integer value to every pixel. Single-image operation (uses A only).",
            "effect": "Uniformly brightens or darkens the image depending on sign of Value.",
            "params_detail": [
                {"name": "Value", "detail": "Integer to add (0–255). 0 = no change."},
            ],
            "when_to_use": "Brightness adjustment, shifting image range before thresholding, adding a uniform bias.",
            "limitations": "Clipped at [0, 255]. Negative values are not supported directly (use Subtract with a constant image).",
        },
        "multiply_constant": {
            "name": "Multiply Constant",
            "how_it_works": "Multiplies every pixel by a fixed float value. Single-image operation (uses A only).",
            "effect": "Scales brightness and contrast. Values < 1 darken, values > 1 brighten.",
            "params_detail": [
                {"name": "Value", "detail": "Multiplier. 0.5 = half brightness, 2.0 = double. Clamped to [0, 255]."},
            ],
            "when_to_use": "Contrast adjustment, exposure simulation, gain correction.",
            "limitations": "Clipped at [0, 255]. Very large multipliers produce saturated (white) images.",
        },
    },
}

# ── Classification ─────────────────────────────────────────────────────────

TOOL_HELP["Classifier"] = {
    "name": "Classifier",
    "category": "Classification",
    "purpose": (
        "Multi-class classifier using deep embedding similarity. "
        "Trains a projection head with triplet loss on frozen EfficientNet-B0 features, "
        "then classifies via k-NN (k=3) cosine similarity on a memory bank of all training "
        "embeddings. Supports rejection of unknown samples via per-class percentile thresholds."
    ),
    "inputs": [
        {"name": "input image", "type": "VisionImage",
         "description": "Image to classify. ROI is used if configured."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Input with classification overlay (PASS/FAIL + score)."},
        {"name": "class_label", "type": "str",
         "description": "Human-readable class label of the predicted class."},
        {"name": "score", "type": "float",
         "description": "Cosine similarity to the class centroid (0–1)."},
        {"name": "pass", "type": "bool",
         "description": "True if winner matches expected class and score >= threshold."},
    ],
    "params": [
        {"name": "threshold", "default": 0.65, "description": "Minimum score for PASS."},
        {"name": "expected_class", "default": "", "description": "Expected class label. Empty = any class accepted."},
        {"name": "k_nn", "default": 3, "description": "Number of nearest neighbors for majority vote."},
        {"name": "roi", "default": "—", "description": "Region of interest [x, y, w, h, angle]."},
        {"name": "output_mode", "default": "Full Image", "description": "Full image with overlays or cropped ROI."},
    ],
    "notes": (
        "Recording mode captures ROI images assigned to the currently selected class tab. "
        "Each class needs at least 5 images. Train requires at least 2 classes. "
        "The EfficientNet-B0 backbone is frozen; only the 128D projection head is trained "
        "via triplet loss with hard mining. Inference runs at ~10ms per image on CPU."
    ),
}

TOOL_HELP["Anomaly Detector"] = {
    "name": "Anomaly Detector",
    "category": "Analysis",
    "purpose": (
        "Unsupervised defect detection using PatchCore. Given a set of good (nominal) "
        "images, it builds a memory bank of patch-level features from an ONNX backbone "
        "(ResNet18-Layer3). At inference, per-patch distances to the nearest neighbor "
        "in the memory bank produce an anomaly score and a pixel-level heatmap. "
        "Template-matching alignment compensates for ±20px fixture drift."
    ),
    "inputs": [
        {"name": "input image", "type": "VisionImage",
         "description": "Pipeline input image with optional fixture transform."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Output with anomaly overlays (PASS/FAIL, score). "
                        "When Show Heatmap is on, the per-patch anomaly heatmap "
                        "(JET colormap) is blended into the ROI region."},
        {"name": "score", "type": "float",
         "description": "Anomaly score — maximum per-patch distance to the "
                        "nearest neighbor in the coreset memory bank. "
                        "Higher = more anomalous."},
        {"name": "pass", "type": "bool",
         "description": "True when score < threshold (good part). "
                        "False when score >= threshold (anomaly detected)."},
    ],
    "params": [
        {"name": "threshold", "default": 1.0, "description": "Multiplier applied to the calibrated threshold (percentile 99 of training scores). 1.0 = calibrated default."},
        {"name": "max_train_samples", "default": 20, "description": "Maximum images to capture during recording (auto-stop)."},
        {"name": "show_labels", "default": True, "description": "Show PASS/FAIL and score overlay."},
        {"name": "show_heatmap", "default": True, "description": "Blend the anomaly heatmap into the output image."},
        {"name": "alignment_enabled", "default": True, "description": "Enable template-matching alignment (compensates ±20px fixture drift)."},
        {"name": "roi", "default": "—", "description": "Region of interest [x, y, w, h, angle]."},
        {"name": "output_mode", "default": "Full Image", "description": "Full image with overlays or cropped ROI."},
    ],
    "backbone_types": {
        "fast": {
            "name": "fast",
            "how_it_works": "ResNet18-Layer2 at 224\u00d7224px, stride 8, 128 channels, 28\u00d728 grid (784 patches).",
            "effect": "Fastest option (~0.5\u00d7 speed vs balanced). 784 patches cover 8\u00d78 px each. Good for high-speed inspection with basic defect detection.",
            "when_to_use": "When throughput is critical and defects are large enough (>= 16\u00d716 px).",
            "limitations": "Fewer channels (128) reduce discriminative power vs 256-channel backbones."
        },
        "balanced": {
            "name": "balanced",
            "how_it_works": "EfficientNet-B0 multi-layer at 224\u00d7224px, 256 channels, 28\u00d728 grid (784 patches).",
            "effect": "Best speed/quality trade-off (1\u00d7 baseline). 256-channel multi-scale features from EfficientNet-B0 block3 (stride 8) + block4 (stride 16).",
            "when_to_use": "Default for most use cases. Good general-purpose anomaly detection.",
            "limitations": "28\u00d728 grid limits localization precision for very small defects."
        },
        "quality": {
            "name": "quality",
            "how_it_works": "ResNet18-Layer3 at 448\u00d7448px, stride 16, 256 channels, 28\u00d728 grid (784 patches).",
            "effect": "Deeper semantic features from Layer3 (~3\u00d7 speed). 448px input gives the backbone more context per patch.",
            "when_to_use": "When defect discrimination is more important than speed. Good for subtle appearance defects.",
            "limitations": "3\u00d7 slower than balanced. Same 28\u00d728 grid as balanced, so no improvement in localization precision."
        },
        "detail": {
            "name": "detail",
            "how_it_works": "ResNet18-Layer2 at 448\u00d7448px, stride 8, 128 channels, 56\u00d756 grid (3136 patches).",
            "effect": "4\u00d7 more patches than balanced (~4\u00d7 speed). Each patch covers only 8\u00d78 px of the original image, enabling detection of very small defects. Early-layer features (textural).",
            "when_to_use": "For finding small defects (8\u201316 px) where localization precision matters.",
            "limitations": "128 channels (vs 256) means less descriptive power per patch. Layer2 features are more textural, may produce more false positives on textured surfaces."
        },
        "hd": {
            "name": "hd",
            "how_it_works": "ResNet18-Layer3 at 896\u00d7896px, stride 16, 256 channels, 56\u00d756 grid (3136 patches).",
            "effect": "Maximum detail: 4\u00d7 more patches than balanced AND 256 channels per patch (~12\u00d7 speed). Combines fine spatial granularity with deep semantic features.",
            "when_to_use": "When detection quality is the top priority and speed is not a concern.",
            "limitations": "12\u00d7 slower than balanced. Requires significant RAM (~2 GB per inference)."
        }
    },
    "notes": (
        "Algorithm: PatchCore with selectable ONNX backbone. "
        "Training is fully unsupervised \u2014 only good parts needed. "
        "Memory bank uses greedy farthest-point coreset (~200 patches). "
        "Threshold auto-calibrated at percentile 99 of training scores. "
        "The anomaly heatmap is built by interpolating the patch-score "
        "grid to the original image size, applying Gaussian blur (kernel 21), "
        "and blending with a JET colormap (max alpha 0.7). "
        "Alignment uses normalized cross-correlation on the central 50% patch "
        "with a maximum shift of plus-minus 30px. "
        "Training runs in a background thread (QThread) to keep the UI responsive. "
        "Typical fit time: ~4s for 20 images on Core i5 with balanced backbone."
    ),
}



TOOL_HELP["OCR"] = {
    "name": "OCR",
    "category": "Vision Tools",
    "purpose": "Recognizes text in a cropped ROI using PP-OCRv5 (ONNX Runtime). Pure recognition — no text detection. The user must define a Search ROI around the text region.",
    "inputs": [{"name": "input image", "type": "VisionImage / ndarray", "description": "Input image containing the text region."}],
    "outputs": [
        {"name": "output image", "type": "VisionImage", "description": "Input with detection overlays."},
        {"name": "text", "type": "str", "description": "Recognized text (Read/Locate) or found text (Verify)."},
        {"name": "confidence", "type": "float", "description": "Global confidence [0, 1]."},
        {"name": "count", "type": "int", "description": "Number of text regions (always 1 in pure recognition mode)."},
    ],
    "params": [
        {"name": "mode", "default": "Read", "description": "Read: free text. Verify: compare against expected text. Locate: per-character bounding boxes."},
        {"name": "model_type", "default": "Server", "description": "Server: higher accuracy. Mobile: faster."},
        {"name": "expected_text", "default": "", "description": "Expected text for Verify mode."},
        {"name": "min_region_confidence", "default": 0.5, "description": "Minimum recognition confidence to accept the text."},
        {"name": "roi", "default": "—", "description": "Search region [x, y, w, h, angle]."},
    ],
    "ocr_modes": {
        "read": {"name": "Read", "how_it_works": "Runs the ROI through PP-OCRv5 SVTR recognition. Returns recognized text.",
                 "params_detail": []},
        "verify": {"name": "Verify", "how_it_works": "Reads text then compares using Levenshtein distance.",
                   "params_detail": [{"name": "Expected Text", "detail": "Text expected in the ROI."}]},
        "locate": {"name": "Locate", "how_it_works": "Same as Read with per-character bounding box estimation.",
                   "params_detail": []},
    },
    "notes": (
        "Pure recognition engine (no detection). The ROI must be tightly cropped around the text. "
        "Uses PP-OCRv5 SVTR model via ONNX Runtime. The Chinese mixed model supports "
        "simplified Chinese, traditional Chinese, English, and Japanese. "
        "Model loaded once at startup via preload_engine() — no runtime disk I/O. "
        "No internet required after initial model download."
    ),
}


# ── HMI Project ─────────────────────────────────────────────────────────────

TOOL_HELP["HMI Project"] = {
    "name": "HMI Project",
    "category": "HMI",
    "purpose": (
        "The HMI (Human Machine Interface) system runs a Python script in a "
        "background thread that controls job execution and updates widget properties "
        "in real time. The <code>proj</code> object gives you full access to the project, "
        "its slots, and all widgets."
    ),
    "inputs": [],
    "outputs": [],
    "params": [
        {"name": "run_slot(name)", "default": "—", "description": "Execute a job slot by name. Calls <code>slot.run(advance=True)</code>. Returns immediately if <code>_stop</code> is set."},
        {"name": "log(msg, level='info')", "default": "—", "description": "Append a timestamped message to the log panel. Level is typically <code>'info'</code> or <code>'error'</code>."},
        {"name": "set_slot_status(name, status)", "default": "—", "description": "Set a slot's status text. Use <code>'pass'</code>, <code>'fail'</code>, or <code>'idle'</code>."},
        {"name": "sleep(seconds)", "default": "—", "description": "Interruptible sleep — checks <code>_stop</code> every 0.1 s and returns early if stop is requested."},
        {"name": "help()", "default": "—", "description": "Print this API reference to the log panel."},
        {"name": "proj", "default": "—", "description": "The <code>HMIProject</code> instance. Access slots, layout, and all widget properties."},
        {"name": "proj.widgets[name]", "default": "—", "description": "Dict-like access to any widget by its unique name. Returns the widget dict. Mutate properties directly: <code>proj.widgets['My counter']['min'] = 0</code>."},
        {"name": "proj.slots", "default": "—", "description": "List of all configured job slots. Each slot has <code>.name</code>, <code>.job_path</code>, <code>.controller</code>, <code>.run()</code>."},
        {"name": "proj.layout", "default": "—", "description": "Raw layout dict with keys <code>rows</code>, <code>cols</code>, <code>widgets</code>."},
        {"name": "proj.project_id", "default": "—", "description": "Project identifier string (read/write via <code>set_project_id()</code>)."},
        {"name": "_stop", "default": "—", "description": "A <code>threading.Event</code>. When set by the Stop button, loops should check <code>if _stop.is_set(): break</code>."},
        {"name": "proj.input(msg)", "default": "—", "description": "Prompt operator for text input. Shows modal on all connected frontends. Blocks script until response or timeout. Returns entered string or <code>None</code>."},
        {"name": "proj.confirm(msg, options=None)", "default": "—", "description": "Prompt operator with buttons. Default options <code>['Yes','No','Cancel']</code> returns <code>True</code>/<code>False</code>/<code>None</code>. Custom options returns selected string."},
    ],
    "notes": (
        "The script runs in a separate thread within the HMI worker process. "
        "You have full access to Python standard library modules.<br><br>"
        "<b>Typical loop:</b><br><br>"
        "<code>while True:</code><br>"
        "<code>&nbsp;&nbsp;&nbsp;&nbsp;if _stop.is_set(): break</code><br>"
        "<code>&nbsp;&nbsp;&nbsp;&nbsp;run_slot('Slot1')</code><br>"
        "<code>&nbsp;&nbsp;&nbsp;&nbsp;sleep(1)</code><br><br>"
        "<b>Widget example:</b><br><br>"
        "<code>w = proj.widgets['My counter']</code><br>"
        "<code>w['min'] = 0; w['max'] = 50</code><br>"
        "<code>proj.widgets['Title']['label'] = 'Hello'</code><br>"
        "<code>proj.widgets['Title']['bg_color'] = [0, 100, 200]</code><br>"
        "<code>proj.widgets['Title']['font_size'] = 24</code><br><br>"
        "<b>Interactive prompts (blocking):</b><br><br>"
        "<code>val = proj.input('Enter batch size:')</code><br>"
        "<code>if val is not None:</code><br>"
        "<code>&nbsp;&nbsp;&nbsp;&nbsp;log(f'Batch: {val}')</code><br><br>"
        "<code>ok = proj.confirm('Continue?')</code><br>"
        "<code>if ok:</code><br>"
        "<code>&nbsp;&nbsp;&nbsp;&nbsp;run_slot('Slot1')</code><br>"
        "<code>elif ok is False:</code><br>"
        "<code>&nbsp;&nbsp;&nbsp;&nbsp;log('Skipped', level='warning')</code><br>"
        "<code>else:</code><br>"
        "<code>&nbsp;&nbsp;&nbsp;&nbsp;log('Cancelled')</code>"
    ),
    "see_also": ["ImageViewer", "MeasureLabel", "StatusIndicator", "TextLabel", "LogPanel"],
}

# ── HMI Scripting ──────────────────────────────────────────────────────────

TOOL_HELP["HMI Scripting"] = {
    "name": "HMI Scripting",
    "category": "HMI",
    "purpose": (
        "The HMI Script runs in a background thread within the worker process. "
        "It has a dedicated namespace with functions to control job execution, "
        "update widget properties, prompt the operator, and interact with "
        "job tools directly — setting inputs before <code>run_slot()</code> "
        "and reading outputs after execution."
    ),
    "inputs": [],
    "outputs": [],
    "params": [
        {"name": "run_slot(name)", "default": "—", "description": "Execute a job slot by name. Calls <code>slot.run(advance=True)</code>. Returns immediately if <code>_stop</code> is set."},
        {"name": "log(msg, level='info')", "default": "—", "description": "Append a timestamped message to the log panel. Levels: <code>info</code>, <code>error</code>, <code>warning</code>, <code>debug</code>."},
        {"name": "set_slot_status(name, status)", "default": "—", "description": "Set a slot's status text. Use <code>'pass'</code>, <code>'fail'</code>, <code>'pending'</code> or <code>'idle'</code>."},
        {"name": "sleep(seconds)", "default": "—", "description": "Interruptible sleep — checks <code>_stop</code> every 0.1 s and returns early if stop is requested."},
        {"name": "help()", "default": "—", "description": "Print this API reference to the log panel."},
        {"name": "proj", "default": "—", "description": "The <code>HMIProject</code> instance. Access slots, layout, and all widget properties. Read/write widget configs: <code>proj.widgets['name']['min'] = 0</code>."},
        {"name": "proj.widgets[name]", "default": "—", "description": "Dict-like access to any widget by its unique name. Returns the widget dict. Mutate properties directly."},
        {"name": "proj.get_widget_value(name)", "default": "—", "description": "Evaluate a widget's field + condition. Returns <code>{'value': ..., 'status': 'pass'|'fail'|'error'}</code>."},
        {"name": "proj.slots", "default": "—", "description": "List of <code>HMISlot</code>. Each slot has <code>.name</code>, <code>.job_path</code>, <code>.controller</code> (the <code>WorkflowController</code>), and <code>.run()</code>."},
        {"name": "proj.input(msg, ...)", "default": "—", "description": "Prompt operator for text input (blocking). Optional: <code>width</code>, <code>height</code>, <code>font_size</code> in px."},
        {"name": "proj.confirm(msg, options=None)", "default": "—", "description": "Prompt operator with buttons. Default <code>['Yes','No','Cancel']</code> returns <code>True</code>/<code>False</code>/<code>None</code>. Custom options returns selected string."},
        {"name": "_stop", "default": "—", "description": "A <code>threading.Event</code>. Set when Stop is pressed. Long-running loops must check <code>if _stop.is_set(): break</code>."},
    ],
    "notes": (
        "<h3>Script Environment</h3>"
        "<p>The script executes in a dedicated thread within the HMI worker process. "
        "The namespace includes all functions listed above plus the <code>proj</code> object. "
        "Standard Python modules are available. The <code>_stop</code> event is set when "
        "the user presses Stop — long-running loops must check it.</p>"
        "<h3>Interacting with Job Tools</h3>"
        "<p>Each slot has a <code>WorkflowController</code> that holds the loaded job. "
        "You can set tool inputs <b>before</b> calling <code>run_slot()</code> and read "
        "tool outputs <b>after</b> execution:</p>"
        "<pre>"
        "# Get a slot's controller\n"
        "ctrl = proj.slots[0].controller\n"
        "if ctrl is None:\n"
        "    log('Slot not loaded', level='error')\n"
        "    return\n\n"
        "# Set a tool's input before running\n"
        "tool = ctrl.workflow.get_tool('Camera')\n"
        "tool.set_input('image', snapshot)\n\n"
        "# Run the job (connections between tools are respected)\n"
        "run_slot('Slot1')\n\n"
        "# Read a tool's output after execution\n"
        "tool = ctrl.workflow.get_tool('Measurement')\n"
        "value = tool.get_output('value')\n"
        "log(f'Measured: {value}')\n\n"
        "# Check if the job had errors\n"
        "if ctrl.last_error:\n"
        "    log(f'Job error: {ctrl.last_error}', level='error')"
        "</pre>"
        "<h3>Reading Widget Values</h3>"
        "<p>If a MeasureLabel widget is bound to a slot/output, you can evaluate "
        "it (including its condition) with a single call:</p>"
        "<pre>"
        "result = proj.get_widget_value('My widget')\n"
        "if result['status'] == 'pass':\n"
        "    log(f'Value {result[\"value\"]} OK')\n"
        "else:\n"
        "    log(f'Value {result[\"value\"]} FAIL', level='error')"
        "</pre>"
        "<h3>Typical Loop</h3>"
        "<pre>"
        "while True:\n"
        "    if _stop.is_set():\n"
        "        break\n\n"
        "    # Inject external data into a tool\n"
        "    ctrl = proj.slots[0].controller\n"
        "    src = ctrl.workflow.get_tool('Source')\n"
        "    src.set_input('image', acquire_frame())\n\n"
        "    run_slot('Slot1')\n\n"
        "    # Read pass/fail from the last tool\n"
        "    ctrl = proj.slots[0].controller\n"
        "    result = ctrl.workflow.get_tool('Classifier').get_output('result')\n"
        "    status = 'pass' if result > 0.5 else 'fail'\n"
        "    set_slot_status('Slot1', status)\n\n"
        "    sleep(1)"
        "</pre>"
        "<h3>Interactive Prompts</h3>"
        "<pre>"
        "# Text input\n"
        "batch = proj.input('Enter batch ID:')\n"
        "if batch is not None:\n"
        "    log(f'Batch: {batch}')\n"
        "    run_slot('Slot1')\n\n"
        "# Confirmation\n"
        "ok = proj.confirm('Continue?')\n"
        "if ok:\n"
        "    log('Proceeding')\n"
        "elif ok is False:\n"
        "    log('Skipped', level='warning')\n"
        "else:\n"
        "    log('Cancelled')\n\n"
        "# Custom options\n"
        "mode = proj.confirm('Select mode:', ['Auto', 'Manual'])\n"
        "if mode == 'Auto':\n"
        "    run_slot('Slot1')"
        "</pre>"
        "<h3>Widget Properties</h3>"
        "<p>Read/write any widget property directly. Common writable properties:</p>"
        "<pre>"
        "w = proj.widgets['Title']\n"
        "w['label'] = 'New Title'\n"
        "w['bg_color'] = [0, 100, 200]\n"
        "w['font_size'] = 24\n"
        "w['halign'] = 'center'\n\n"
        "# Numeric conditions\n"
        "w = proj.widgets['Counter']\n"
        "w['min'] = 0\n"
        "w['max'] = 100\n"
        "w['condition'] = 'between'\n\n"
        "# Text conditions\n"
        "w['value_type'] = 'text'\n"
        "w['condition'] = 'contains'\n"
        "w['compare_value'] = 'OK'"
        "</pre>"
    ),
    "see_also": ["HMI Project", "ImageViewer", "MeasureLabel", "StatusIndicator", "LogPanel"],
}

TOOL_HELP["ImageViewer"] = {
    "name": "ImageViewer",
    "category": "HMI",
    "purpose": "Displays an image output from a job slot. Shows the tool's output image with overlays.",
    "inputs": [],
    "outputs": [],
    "params": [
        {"name": "slot", "default": "—", "description": "Name of the job slot whose output to display."},
        {"name": "output", "default": "—", "description": "Output port name (e.g. <code>'ToolName > output image'</code>)."},
        {"name": "label", "default": "—", "description": "Optional display label shown on the viewer."},
    ],
}

TOOL_HELP["MeasureLabel"] = {
    "name": "MeasureLabel",
    "category": "HMI",
    "purpose": "Displays a numeric value extracted from a tool output, with optional range coloring.",
    "inputs": [],
    "outputs": [],
    "params": [
        {"name": "slot", "default": "—", "description": "Name of the job slot."},
        {"name": "output", "default": "—", "description": "Output port name (e.g. <code>'ToolName > results'</code>)."},
        {"name": "field", "default": "[0].value", "description": "Field expression to extract a scalar. Use <code>[N]</code> for index, <code>.name</code> for attribute. Example: <code>[0].area</code>, <code>Count</code>."},
        {"name": "label", "default": "—", "description": "Optional display label above the value."},
        {"name": "min", "default": "0", "description": "Lower bound for 'between' condition."},
        {"name": "max", "default": "100", "description": "Upper bound for 'between' condition."},
        {"name": "value_type", "default": "numeric", "description": "One of <code>'numeric'</code> or <code>'text'</code>."},
        {"name": "condition", "default": "between", "description": "Numeric: <code>between</code>, <code>></code>, <code>>=</code>, <code><</code>, <code><=</code>, <code>=</code>, <code>!=</code>. Text: <code>equals</code>, <code>!equals</code>, <code>contains</code>, <code>!contains</code>, <code>regex</code>."},
        {"name": "compare_value", "default": "—", "description": "Value to compare against for non-between conditions (numeric and text)."},
        {"name": "regex", "default": "—", "description": "Regex pattern for <code>condition='regex'</code>."},
        {"name": "font_size", "default": "14", "description": "Font size in pixels for the value display."},
        {"name": "halign", "default": "center", "description": "Horizontal alignment: <code>'left'</code>, <code>'center'</code>, <code>'right'</code>."},
        {"name": "label_layout", "default": "stacked", "description": "Label position: <code>'stacked'</code> (above value) or <code>'inline'</code> (same line as value)."},
    ],
    "notes": (
        "The <code>field</code> expression supports:<br>"
        "<code>[0]</code> — index into a list<br>"
        "<code>.area</code> — attribute access<br>"
        '<code>["area"]</code> — dict key access<br><br>'
        "<b>Examples:</b><br>"
        "<code>[0].area</code> — first blob's area<br>"
        "<code>Count</code> — ToolResults.Count property<br>"
        "<code>[0].x</code> — first blob's X coordinate<br><br>"
        "<b>Condition examples:</b><br>"
        "<code>w['value_type']='numeric'; w['condition']='between'; w['min']=0; w['max']=50</code><br>"
        "<code>w['condition']='>'; w['compare_value']='10'</code><br>"
        "<code>w['value_type']='text'; w['condition']='contains'; w['compare_value']='OK'</code><br>"
        "<code>w['condition']='regex'; w['regex']='^OK.*'</code>"
    ),
    "see_also": ["HMI Project", "StatusIndicator", "TextLabel"],
}

TOOL_HELP["LedIndicator"] = {
    "name": "LedIndicator",
    "category": "HMI",
    "purpose": "Displays a colored circle that reflects a slot's status set via <code>set_slot_status()</code>. Supports pass (green), fail (red), pending (yellow), and idle (gray) states with configurable colors.",
    "inputs": [],
    "outputs": [],
    "params": [
        {"name": "slot", "default": "—", "description": "Name of the job slot whose status to reflect."},
        {"name": "label", "default": "—", "description": "Optional display label below the LED."},
        {"name": "color_pass", "default": "#4ec9b0", "description": "Hex color for <code>'pass'</code> state."},
        {"name": "color_fail", "default": "#c62828", "description": "Hex color for <code>'fail'</code> state."},
        {"name": "color_pending", "default": "#ffa726", "description": "Hex color for <code>'pending'</code> state."},
        {"name": "color_idle", "default": "#888888", "description": "Hex color for <code>'idle'</code> state."},
    ],
    "notes": (
        "Status is set from the HMI script:<br><br>"
        "<code>set_slot_status('Slot1', 'pass')</code><br>"
        "<code>set_slot_status('Slot1', 'fail')</code><br>"
        "<code>set_slot_status('Slot1', 'pending')</code>"
    ),
    "see_also": ["HMI Project", "StatusIndicator", "TextLabel"],
}

TOOL_HELP["StatusIndicator"] = {
    "name": "StatusIndicator",
    "category": "HMI",
    "purpose": "Shows a PASS / FAIL indicator based on a slot's status, set via <code>set_slot_status()</code>.",
    "inputs": [],
    "outputs": [],
    "params": [
        {"name": "slot", "default": "—", "description": "Name of the job slot whose status to reflect."},
        {"name": "label", "default": "—", "description": "Optional display label."},
        {"name": "color_pass", "default": "#2e7d32", "description": "Hex color for <code>'pass'</code> state."},
        {"name": "color_fail", "default": "#c62828", "description": "Hex color for <code>'fail'</code> state."},
        {"name": "color_pending", "default": "#c1c100", "description": "Hex color for <code>'pending'</code> state."},
        {"name": "color_idle", "default": "#888888", "description": "Hex color for <code>'idle'</code> state."},
    ],
    "notes": (
        "Status is set from the HMI script:<br><br>"
        "<code>set_slot_status('Slot1', 'pass')</code><br>"
        "<code>set_slot_status('Slot1', 'fail')</code><br>"
        "<code>set_slot_status('Slot1', 'pending')</code>"
    ),
    "see_also": ["HMI Project", "LedIndicator", "MeasureLabel"],
}

TOOL_HELP["TextLabel"] = {
    "name": "TextLabel",
    "category": "HMI",
    "purpose": "Displays static or script-updated text with full styling control.",
    "inputs": [],
    "outputs": [],
    "params": [
        {"name": "label", "default": "Label", "description": "The text content to display."},
        {"name": "bg_color", "default": "[45, 45, 45]", "description": "Background color as <code>[R, G, B]</code> (0–255 each)."},
        {"name": "text_color", "default": "[204, 204, 204]", "description": "Text color as <code>[R, G, B]</code>."},
        {"name": "font_size", "default": "14", "description": "Font size in pixels."},
        {"name": "halign", "default": "center", "description": "Horizontal alignment: <code>'left'</code>, <code>'center'</code>, <code>'right'</code>."},
        {"name": "valign", "default": "center", "description": "Vertical alignment: <code>'top'</code>, <code>'center'</code>, <code>'bottom'</code>."},
    ],
    "see_also": ["HMI Project", "MeasureLabel", "LogPanel"],
}

TOOL_HELP["LogPanel"] = {
    "name": "LogPanel",
    "category": "HMI",
    "purpose": "Displays log messages from the HMI script in a scrollable panel.",
    "inputs": [],
    "outputs": [],
    "params": [
        {"name": "font_size", "default": "14", "description": "Font size in pixels for log entries."},
    ],
    "notes": (
        "Log messages are added from the HMI script:<br><br>"
        "<code>log('Job started')</code><br>"
        "<code>log('Error: measurement failed', level='error')</code>"
    ),
    "see_also": ["HMI Project", "TextLabel"],
}

TOOL_HELP["Shape Match"] = {
    "name": "Shape Match",
    "category": "Vision Tools",
    "purpose": (
        "Gradient-orientation shape matching for rotation, scale and translation. "
        "The native kernel proposes hypotheses, NCC/ECC refines fully visible "
        "parts, and a gradient/ICP solver handles occlusion. The score is the "
        "fraction of template gradient mass matched by the image."
    ),
    "inputs": [
        {"name": "input image", "type": "VisionImage / ndarray",
         "description": "Search image. Train a template before running."},
    ],
    "outputs": [
        {"name": "output image", "type": "VisionImage",
         "description": "Annotated image with found region and origin."},
        {"name": "count", "type": "int",
         "description": "Number of matches found."},
        {"name": "trained", "type": "bool",
         "description": "True if a shape model is loaded."},
        {"name": "warnings", "type": "list",
         "description": "Candidate rejection and solver warnings."},
        {"name": "results", "type": "ToolResults[ShapeMatchResult]",
         "description": "Matches with x, y, angle, scale, score, solver, "
                        "ncc_score, visible_fraction, ecc_cc and warnings."},
    ],
    "params": [
        {"name": "train_roi", "default": "[0, 0, 0, 0, 0]",
         "description": "Training ROI in fixture coordinates. Supports angle. "
                        "Zero width/height = whole captured image."},
        {"name": "search_roi", "default": "[0, 0, 0, 0, 0]",
         "description": "Search ROI in fixture coordinates. Zero = whole image."},
        {"name": "angle_start", "default": "-30",
         "description": "Start of rotation search (degrees)."},
        {"name": "angle_end", "default": "30",
         "description": "End of rotation search (degrees)."},
        {"name": "scale_start", "default": "0.9",
         "description": "Start of scale search (fraction of template size)."},
        {"name": "scale_end", "default": "1.1",
         "description": "End of scale search."},
        {"name": "min_score", "default": "0.45",
         "description": "Minimum gradient alignment score (0-1). Lower = more "
                        "tolerant to occlusion."},
        {"name": "max_matches", "default": "1",
         "description": "Maximum number of instances to return. 0 = All (capped at 100)."},
        {"name": "max_features", "default": "800",
         "description": "Maximum number of template edge features."},
        {"name": "origin_x", "default": "0",
         "description": "Reported origin X relative to the template center. "
                        "0 = template center."},
        {"name": "origin_y", "default": "0",
         "description": "Reported origin Y relative to the template center. "
                        "0 = template center."},
        {"name": "act_as_fixture", "default": "False",
         "description": "Use the best match origin as the output fixture."},
        {"name": "show_search_roi", "default": "True",
         "description": "Draw the search ROI in the output image."},
        {"name": "show_found_region", "default": "True",
         "description": "Draw the found template region."},
        {"name": "show_found_origin", "default": "True",
         "description": "Draw axes at the reported origin."},
        {"name": "show_labels", "default": "True",
         "description": "Draw match labels."},
        {"name": "show_overlays", "default": "True",
         "description": "Render overlays in the viewer."},
        {"name": "copy_overlays_to_input", "default": "True",
         "description": "Copy generated overlays to the input image."},
    ],
    "notes": (
        "Capture a train image, draw the Training ROI (rotation supported), and "
        "press Train ShapeMatch. The Shape Preview shows the template edges, "
        "selected feature points and their gradient orientation. Matching "
        "requires the native <code>_falco_kernel</code> extension. Fully visible "
        "parts use ECC; occluded parts use the explicit occlusion solver."
    ),
    "see_also": ["Pattern Matcher", "Measurement"],
}
