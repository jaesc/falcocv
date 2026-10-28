import numpy as np
import cv2
from typing import Dict, List, Optional, Any
from falcocv.core.overlay_layer import OverlayLayer, OverlayType, OverlayElement


class VisionImage:
    """
    High-performance image representation with vector-based overlay system.

    This class decouples image data from visual overlays by using a lightweight
    vector primitive architecture. Overlays are stored as mathematical primitives
    (rectangles, circles, lines, etc.) and rendered on-demand only when explicitly
    requested, ensuring maximum memory efficiency and rendering performance.

    Attributes:
        data: The raw numpy array containing image pixel data
        x, y, angle: Fixture transformation parameters (position and rotation)
        scale_x, scale_y, scale_z: Scale factors for coordinate transformation
        offset_z: Z-axis offset for 3D operations
        metadata: Dictionary for storing arbitrary metadata
        _overlay_layers: Dictionary of overlay layers (name -> OverlayLayer)
        _layer_order: List maintaining the rendering order of visible overlay layers
        _world_matrix: 4x4 world transformation matrix (Image → World)
        _world_homography: 3x3 homography when set via point correspondences
        _world_valid: Flag indicating whether world calibration has been performed
    """
    def __init__(self, data, x=0.0, y=0.0, angle=0.0, scale_x=1.0, scale_y=1.0, scale_z=1.0, offset_z=0.0):
        if isinstance(data, VisionImage):
            raise TypeError("VisionImage constructor expects a numpy array, not another VisionImage.")
        self.data = data

        self._angle = float(angle)
        self._sin_angle = 0.0
        self._cos_angle = 1.0
        self._update_angle_cache()

        self.x = x
        self.y = y
        self.scale_x = scale_x
        self.scale_y = scale_y
        self.scale_z = scale_z
        self.offset_z = offset_z

        # World coordinate mapping (Image → World)
        # 4x4 matrix for future 3D extrinsics compatibility; 3x3 homography for 2D perspective
        self._world_matrix = np.eye(4, dtype=np.float64)
        self._world_homography = None  # 3x3 matrix when set via findHomography
        self._world_valid = False
        self._world_src_points = np.empty((0, 2), dtype=np.float64)
        self._world_dst_points = np.empty((0, 2), dtype=np.float64)

        self.metadata = {}
        self._3d_data = None
        self._3d_cache_key = None
        self._overlay_layers: Dict[str, OverlayLayer] = {}
        self._layer_order: List[str] = []
        self._gray_cache = None
        self._gray_data_id = None

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = float(value)
        self._update_angle_cache()

    def _update_angle_cache(self):
        """Precompute sin/cos of current angle to avoid redundant trig calls."""
        rad = np.deg2rad(self._angle)
        self._cos_angle = np.cos(rad)
        self._sin_angle = np.sin(rad)

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def overlay_layers(self) -> Dict[str, 'OverlayLayer']:
        """Public accessor for overlay layers dictionary."""
        return self._overlay_layers

    @property
    def gray(self) -> np.ndarray:
        data_id = id(self.data)
        if self._gray_data_id != data_id:
            if self.data.ndim == 2:
                self._gray_cache = self.data
            else:
                self._gray_cache = cv2.cvtColor(self.data, cv2.COLOR_BGR2GRAY)
            self._gray_data_id = data_id
        return self._gray_cache

    def _ensure_layer(self, layer_name: str, source_tool: str = "") -> OverlayLayer:
        if layer_name not in self._overlay_layers:
            layer = OverlayLayer(name=layer_name, source_tool=source_tool)
            self._overlay_layers[layer_name] = layer
            self._layer_order.append(layer_name)
        return self._overlay_layers[layer_name]

    def add_overlay_layer(self, layer: OverlayLayer) -> None:
        name = layer.name
        if name not in self._overlay_layers:
            self._overlay_layers[name] = layer
            self._layer_order.append(name)

    def get_overlay_layer(self, layer_name: str, create: bool = True, source_tool: str = "") -> Optional[OverlayLayer]:
        """Return an overlay layer, optionally claiming ownership of it.

        Passing ``source_tool`` means "the elements I am about to add belong to
        this tool", so an existing (possibly inherited) layer is re-tagged.
        Without that, ``OverlayLayer.add_roi()/add_contour()/add_text()/...``
        would stamp the name of whichever tool created the layer, and
        ``BaseTool._copy_overlays_to_input()`` — which filters by
        ``element.source_tool`` — would silently drop the new elements.
        """
        if layer_name in self._overlay_layers:
            layer = self._overlay_layers[layer_name]
            if source_tool:
                layer.source_tool = source_tool
            return layer
        if create:
            return self._ensure_layer(layer_name, source_tool)
        return None

    def remove_overlay_layer(self, layer_name: str) -> None:
        if layer_name in self._overlay_layers:
            del self._overlay_layers[layer_name]
            self._layer_order.remove(layer_name)

    def clear_all_overlays(self) -> None:
        self._overlay_layers.clear()
        self._layer_order.clear()

    def list_layers(self) -> List[str]:
        return [name for name in self._layer_order if self._overlay_layers[name].visible]

    def render_overlays(self, layers: Optional[List[str]] = None) -> np.ndarray:
        if not self.list_layers():
            return self.data

        if self.data.ndim == 2:
            result_img = cv2.cvtColor(self.data, cv2.COLOR_GRAY2BGR)
        else:
            result_img = self.data.copy()

        # Draw directly on the image copy for each visible element
        for layer_name, layer in self._overlay_layers.items():
            if not layer.visible:
                continue
            if layers is not None and layer_name not in layers:
                continue
            for elem in layer.get_visible_elements():
                if elem.element_type == OverlayType.CONTOUR:
                    contour = np.array(elem.data["contour"], dtype=np.int32)
                    color = tuple(elem.data.get("color", (0, 255, 0)))
                    thickness = elem.data.get("thickness", 2)
                    fill = elem.data.get("fill", False)
                    if fill:
                        cv2.drawContours(result_img, [contour], -1, color, -1)
                    else:
                        cv2.drawContours(result_img, [contour], -1, color, thickness)
                elif elem.element_type == OverlayType.CIRCLE:
                    center = tuple(map(int, elem.data["center"]))
                    radius = int(elem.data["radius"])
                    color = tuple(elem.data.get("color", (0, 255, 0)))
                    thickness = elem.data.get("thickness", 2)
                    fill = elem.data.get("fill", False)
                    cv2.circle(result_img, center, radius, color, -1 if fill else thickness)
                elif elem.element_type == OverlayType.TEXT:
                    text = str(elem.data["text"])
                    position = tuple(map(int, elem.data["position"]))
                    color = tuple(elem.data.get("color", (255, 255, 0)))
                    font_scale = elem.data.get("font_scale", 0.5)
                    thickness = elem.data.get("thickness", 1)
                    cv2.putText(result_img, text, position, 0, font_scale, color, thickness)
                elif elem.element_type == OverlayType.POINT:
                    cx, cy = int(elem.data["center"][0]), int(elem.data["center"][1])
                    color = tuple(elem.data.get("color", (0, 0, 255)))
                    r = int(elem.data.get("radius", 4)) // 2
                    cv2.line(result_img, (cx - r, cy), (cx + r, cy), color, 2)
                    cv2.line(result_img, (cx, cy - r), (cx, cy + r), color, 2)
                elif elem.element_type == OverlayType.LINE:
                    start = tuple(map(int, elem.data["start"]))
                    end = tuple(map(int, elem.data["end"]))
                    color = tuple(elem.data.get("color", (0, 0, 255)))
                    thickness = elem.data.get("thickness", 2)
                    cv2.line(result_img, start, end, color, thickness)
                elif elem.element_type == OverlayType.AXES:
                    cx, cy = int(elem.data["center"][0]), int(elem.data["center"][1])
                    angle_rad = np.deg2rad(elem.data.get("angle", 0))
                    length = int(elem.data.get("length", 30))
                    color_x = tuple(elem.data.get("color_x", (0, 0, 255)))
                    color_y = tuple(elem.data.get("color_y", (0, 255, 0)))
                    thickness = elem.data.get("thickness", 2)
                    ex = int(cx + length * np.cos(angle_rad))
                    ey = int(cy + length * np.sin(angle_rad))
                    perp_rad = angle_rad + np.pi / 2
                    ey2 = int(cy + length * np.sin(perp_rad))
                    ex2 = int(cx + length * np.cos(perp_rad))
                    cv2.line(result_img, (cx, cy), (ex, ey), color_x, thickness)
                    cv2.line(result_img, (cx, cy), (ex2, ey2), color_y, thickness)
                elif elem.element_type == OverlayType.POLYGON:
                    pts = np.array(elem.data.get("points", []), dtype=np.int32).reshape((-1, 1, 2))
                    color = tuple(elem.data.get("color", (0, 255, 0)))
                    thickness = elem.data.get("thickness", 2)
                    fill = elem.data.get("fill", False)
                    if len(pts) > 0:
                        cv2.polylines(result_img, [pts], True, color, -1 if fill else thickness)
                elif elem.element_type == OverlayType.ROI:
                    roi_data = elem.data["roi"]
                    x, y, w, h = roi_data[:4]
                    angle = roi_data[4] if len(roi_data) > 4 else 0
                    color = tuple(elem.data.get("color", (255, 128, 0)))
                    thickness = elem.data.get("thickness", 2)
                    fill = elem.data.get("fill", False)
                    fill_image = elem.data.get("fill_image")
                    fill_alpha = elem.data.get("fill_alpha", 0.6)
                    if fill_image is not None and fill:
                        # ROI filled with image (alpha-blend)
                        bmp = fill_image
                        if abs(angle) < 0.5:
                            x1, y1 = int(x), int(y)
                            x2, y2 = x1 + int(w), y1 + int(h)
                            x1 = max(0, x1); y1 = max(0, y1)
                            x2 = min(result_img.shape[1], x2)
                            y2 = min(result_img.shape[0], y2)
                            patch = result_img[y1:y2, x1:x2]
                            bmp_cropped = bmp[:y2-y1, :x2-x1]
                            if patch.shape[:2] == bmp_cropped.shape[:2]:
                                result_img[y1:y2, x1:x2] = (
                                    patch * (1 - fill_alpha) + bmp_cropped * fill_alpha
                                ).astype(np.uint8)
                        else:
                            cx, cy = x + w / 2, y + h / 2
                            M = cv2.getRotationMatrix2D((int(cx), int(cy)), angle, 1.0)
                            rotated = cv2.warpAffine(
                                bmp, M, (result_img.shape[1], result_img.shape[0]),
                                flags=cv2.INTER_LINEAR,
                            )
                            mask = (rotated > 0).astype(np.float32)
                            result_img[:] = (
                                result_img * (1 - mask * fill_alpha)
                                + rotated * fill_alpha
                            ).astype(np.uint8)
                        if thickness > 0:
                            x1, y1 = int(x), int(y)
                            x2, y2 = x1 + int(w), y1 + int(h)
                            cv2.rectangle(result_img, (x1, y1), (x2, y2), color, thickness)
                    elif angle == 0:
                        if fill:
                            cv2.rectangle(result_img, (int(x), int(y)), (int(x + w), int(y + h)), color, -1)
                        else:
                            cv2.rectangle(result_img, (int(x), int(y)), (int(x + w), int(y + h)), color, thickness)
                    else:
                        # Rotated ROI: compute box points and draw as contour
                        cx, cy = x + w / 2, y + h / 2
                        rect = ((cx, cy), (w, h), angle)
                        box = cv2.boxPoints(rect)
                        box = np.int32(box)
                        if fill:
                            cv2.drawContours(result_img, [box], -1, color, -1)
                        else:
                            cv2.drawContours(result_img, [box], -1, color, thickness)
        
        return result_img

    def draw_roi(self, roi_data, color=(255, 128, 0), thickness=2, layer_name: Optional[str] = None, fill: bool = False):
        """
        Add a rectangular ROI overlay to the image.
        
        Args:
            roi_data: List [x, y, width, height, angle] or [x, y, width, height]
            color: BGR color tuple (e.g., (255, 128, 0) for orange)
            thickness: Line thickness in pixels (use -1 for filled)
            layer_name: Name of the overlay layer (defaults to "default")
            fill: Whether to fill the rectangle
        """
        layer_name = layer_name or "default"
        layer = self._ensure_layer(layer_name)
        layer.add_roi(roi_data, color, thickness, fill=fill)

    def draw_contour(self, contour, color=(0, 255, 0), thickness=2, layer_name: Optional[str] = None, fill: bool = False):
        """
        Add a contour overlay to the image.
        
        Args:
            contour: Numpy array of contour points
            color: BGR color tuple
            thickness: Line thickness in pixels
            layer_name: Name of the overlay layer (defaults to "default")
            fill: Whether to fill the contour
        """
        layer_name = layer_name or "default"
        layer = self._ensure_layer(layer_name)
        layer.add_contour(contour, color, thickness, fill=fill)

    def draw_circle(self, center, radius, color=(0, 255, 0), thickness=2, layer_name: Optional[str] = None, fill: bool = False):
        """
        Add a circle overlay to the image.
        
        Args:
            center: Tuple (x, y) for circle center
            radius: Circle radius in pixels
            color: BGR color tuple
            thickness: Line thickness (use -1 for filled)
            layer_name: Name of the overlay layer (defaults to "default")
            fill: Whether to fill the circle
        """
        layer_name = layer_name or "default"
        layer = self._ensure_layer(layer_name)
        layer.add_circle(center, radius, color, thickness, fill=fill)

    def draw_text(self, text, position, color=(255, 255, 0), font_scale=0.5, thickness=1, layer_name: Optional[str] = None):
        """
        Add text overlay to the image.
        
        Args:
            text: Text string to display
            position: Tuple (x, y) for text position
            color: BGR color tuple
            font_scale: Font scale factor
            thickness: Text thickness
            layer_name: Name of the overlay layer (defaults to "default")
        """
        layer_name = layer_name or "default"
        layer = self._ensure_layer(layer_name)
        layer.add_text(text, position, color, font_scale, thickness)

    def draw_line(self, start, end, color=(0, 0, 255), thickness=2, layer_name: Optional[str] = None):
        """
        Add a line overlay to the image.
        
        Args:
            start: Tuple (x, y) for line start
            end: Tuple (x, y) for line end
            color: BGR color tuple
            thickness: Line thickness in pixels
            layer_name: Name of the overlay layer (defaults to "default")
        """
        layer_name = layer_name or "default"
        layer = self._ensure_layer(layer_name)
        layer.add_line(start, end, color, thickness)

    def draw_axes(self, center, angle, length=30, color_x=(0, 0, 255), color_y=(0, 255, 0), thickness=2, layer_name: Optional[str] = None):
        """
        Add coordinate axes overlay to the image.
        
        Args:
            center: Tuple (x, y) for axes center
            angle: Rotation angle in degrees
            length: Length of axis lines in pixels
            color_x: BGR color for X-axis (default: blue)
            color_y: BGR color for Y-axis (default: green)
            thickness: Line thickness
            layer_name: Name of the overlay layer (defaults to "default")
        """
        layer_name = layer_name or "default"
        layer = self._ensure_layer(layer_name)
        layer.add_axes(center, angle, length, color_x, color_y, thickness)

    def get_image_with_overlay(self):
        """Get the image with rendered overlays drawn on top."""
        return self.render_overlays()

    def get_image_without_overlay(self):
        """Get a copy of the raw image data without any overlays."""
        return self.data.copy()

    def get_combined_layers(self, layer_names: List[str]) -> np.ndarray:
        """Render and combine specific overlay layers onto the image."""
        return self.render_overlays(layers=layer_names)

    def optimize_overlay_memory(self) -> None:
        """
        Optimize overlay memory by removing empty layers and consolidating storage.
        
        This method should be called after intensive overlay operations to reclaim
        memory and improve rendering performance.
        """
        # Remove empty layers
        empty_layers = [name for name, layer in self._overlay_layers.items() 
                       if len(layer.elements) == 0]
        for name in empty_layers:
            self.remove_overlay_layer(name)
    
    def inherit_overlays_from(self, other: 'VisionImage') -> None:
        """Share overlay layers from another VisionImage instance.

        Overlays are immutable after creation, so elements are shared by
        reference for maximum performance. If mutability is ever needed,
        switch to elem.copy() in the extend() call.
        """
        for layer_name, layer in other._overlay_layers.items():
            new_layer = self._ensure_layer(layer_name, source_tool=layer.source_tool)
            new_layer.elements.extend(layer.elements)
    
    def get_overlay_stats(self) -> dict:
        """
        Get statistics about overlay storage for performance monitoring.
        
        Returns:
            Dictionary with overlay memory and complexity metrics
        """
        total_elements = sum(len(layer.elements) for layer in self._overlay_layers.values())
        total_layers = len(self._overlay_layers)
        visible_layers = len(self.list_layers())
        
        return {
            "total_layers": total_layers,
            "visible_layers": visible_layers,
            "total_elements": total_elements,
            "layers": {name: len(layer.elements) for name, layer in self._overlay_layers.items()}
        }

    def get_overlays_3d(self):
        """Return all world-space overlay elements (used by the 3D viewer)."""
        result = []
        for layer in self._overlay_layers.values():
            for elem in layer.elements:
                if elem.element_type.name.startswith("WORLD_"):
                    result.append(elem)
        return result

    def copy(self):
        new_img = VisionImage(self.data.copy(), self.x, self.y, self.angle,
                               self.scale_x, self.scale_y, self.scale_z, self.offset_z)
        new_img._world_matrix = self._world_matrix.copy()
        new_img._world_homography = self._world_homography.copy() if self._world_homography is not None else None
        new_img._world_valid = self._world_valid
        new_img._world_src_points = self._world_src_points.copy()
        new_img._world_dst_points = self._world_dst_points.copy()
        new_img.metadata = self.metadata.copy()
        new_img._3d_data = self._3d_data
        new_img._3d_cache_key = self._3d_cache_key
        for layer_name, layer in self._overlay_layers.items():
            new_img._overlay_layers[layer_name] = OverlayLayer(
                name=layer.name,
                source_tool=layer.source_tool
            )
            new_img._overlay_layers[layer_name].visible = layer.visible
            new_img._overlay_layers[layer_name].z_order = layer.z_order
            for elem in layer.elements:
                new_img._overlay_layers[layer_name].elements.append(
                    OverlayElement(
                        element_type=elem.element_type,
                        data=elem.data.copy(),
                        visible=elem.visible,
                        priority=elem.priority,
                        style=elem.style.copy(),
                        source_tool=elem.source_tool,
                        ref_id=elem.ref_id,
                    )
                )
        new_img._layer_order = self._layer_order.copy()
        return new_img

    def get_matrix(self):
        return cv2.getRotationMatrix2D((self.x, self.y), self._angle, 1.0)

    def fixture_to_image(self, roi_fixture):
        if len(roi_fixture) == 4:
            fx, fy, fw, fh = roi_fixture
            fa = 0.0
        else:
            fx, fy, fw, fh, fa = roi_fixture
        cx_f = fx + fw/2.0
        cy_f = fy + fh/2.0
        cos_a, sin_a = self._cos_angle, self._sin_angle
        abs_cx = self.x + (cx_f * cos_a - cy_f * sin_a)
        abs_cy = self.y + (cx_f * sin_a + cy_f * cos_a)
        abs_x = abs_cx - fw/2.0
        abs_y = abs_cy - fh/2.0
        abs_angle = self._angle + fa
        return [abs_x, abs_y, fw, fh, abs_angle]

    def image_to_fixture(self, roi_image):
        if len(roi_image) == 4:
            ix, iy, iw, ih = roi_image
            ia = 0.0
        else:
            ix, iy, iw, ih, ia = roi_image
        cx_i = ix + iw/2.0
        cy_i = iy + ih/2.0
        cx_f = cx_i - self.x
        cy_f = cy_i - self.y
        cos_a, sin_a = self._cos_angle, -self._sin_angle  # cos(-x)=cos(x), sin(-x)=-sin(x)
        fx = cx_f * cos_a - cy_f * sin_a
        fy = cx_f * sin_a + cy_f * cos_a
        fix_x = fx - iw/2.0
        fix_y = fy - ih/2.0
        rel_angle = ia - self._angle
        return [fix_x, fix_y, iw, ih, rel_angle]

    def get_roi_data(self, roi_rect_5):
        rx, ry, rw, rh, ra = roi_rect_5
        total_angle = self._angle + ra
        if abs(total_angle) < 1e-9:
            # The ROI is axis-aligned in image space, but the fixture may still
            # be rotated: an upright ROI drawn on a rotated fixture is stored
            # as ra = -fixture.angle, so total_angle is 0 while the fixture
            # transform still moves the ROI.  Use fixture_to_image() for the
            # origin instead of a plain translation, otherwise the crop is
            # taken from the wrong place (and a rotated fixture with x=y=0
            # would even crop at the fixture coordinates).
            abs_x, abs_y, _, _, _ = self.fixture_to_image([rx, ry, rw, rh, ra])
            ix, iy = int(abs_x), int(abs_y)
            iw, ih = int(rw), int(rh)
            h, w = self.data.shape[:2]
            ix = max(0, min(ix, w-1))
            iy = max(0, min(iy, h-1))
            iw = max(1, min(iw, w - ix))
            ih = max(1, min(ih, h - iy))
            return VisionImage(self.data[iy:iy+ih, ix:ix+iw])
        cx_f = rx + rw/2
        cy_f = ry + rh/2
        cos_a, sin_a = self._cos_angle, self._sin_angle
        abs_cx = self.x + (cx_f * cos_a - cy_f * sin_a)
        abs_cy = self.y + (cx_f * sin_a + cy_f * cos_a)
        abs_x = abs_cx - rw/2.0
        abs_y = abs_cy - rh/2.0
        abs_angle = self._angle + ra
        if abs_angle == 0:
            ix, iy = int(abs_cx - rw/2), int(abs_cy - rh/2)
            iw, ih = int(rw), int(rh)
            h_img, w_img = self.data.shape[:2]
            ix = max(0, min(ix, w_img-1))
            iy = max(0, min(iy, h_img-1))
            iw = max(1, min(iw, w_img - ix))
            ih = max(1, min(ih, h_img - iy))
            return self.data[iy:iy+ih, ix:ix+iw]
        else:
            M_rot = cv2.getRotationMatrix2D((abs_cx, abs_cy), abs_angle, 1.0)
            current = M_rot @ np.array([abs_cx, abs_cy, 1.0])
            M_rot[0, 2] += ((rw / 2.0) - current[0])
            M_rot[1, 2] += ((rh / 2.0) - current[1])
            return cv2.warpAffine(self.data, M_rot, (int(rw), int(rh)), flags=cv2.INTER_LINEAR)

    # ──────────────────────────────────────────────
    # World coordinate mapping (Image → World)
    # ──────────────────────────────────────────────

    def set_world_from_points(self, img_points, world_points, method='homography'):
        """
        Compute world transformation from point correspondences.

        Provides both manual calibration (user supplies matching points) and
        a path for automatic calibration (points from find_calibration_grid).

        Args:
            img_points: Nx2 array of image pixel coordinates (x, y).
            world_points: Nx2 or Nx3 array of real-world coordinates (mm).
            method: 'homography' (4+ pts, default) or 'affine' (3 pts).
        """
        img_pts = np.asarray(img_points, dtype=np.float64)
        world_pts = np.asarray(world_points, dtype=np.float64)
        n = len(img_pts)

        if n < 3:
            self.clear_world()
            return

        self.clear_world()

        if method == 'affine' or n == 3:
            if n > 3:
                import warnings
                warnings.warn(f"Usando solo 3 de {n} puntos para transformación afín.")
            M = cv2.getAffineTransform(
                img_pts[:3].astype(np.float32),
                world_pts[:3].astype(np.float32)
            )
            # Embed 2×3 affine into 4×4: M = [[a, b, tx], [c, d, ty]]
            self._world_matrix = np.eye(4, dtype=np.float64)
            self._world_matrix[0, 0] = M[0, 0]
            self._world_matrix[0, 1] = M[0, 1]
            self._world_matrix[0, 3] = M[0, 2]
            self._world_matrix[1, 0] = M[1, 0]
            self._world_matrix[1, 1] = M[1, 1]
            self._world_matrix[1, 3] = M[1, 2]

        elif method == 'homography' and n >= 4:
            H, mask = cv2.findHomography(
                img_pts.astype(np.float32),
                world_pts.astype(np.float32),
                cv2.RANSAC, 3.0
            )
            if H is None:
                raise ValueError(
                    "No se pudo calcular la homografía. "
                    "Verifica que los puntos no sean colineales."
                )
            self._world_homography = H
            # Build 4×4 from homography's affine component
            self._world_matrix = np.eye(4, dtype=np.float64)
            self._world_matrix[0, :] = [H[0, 0], H[0, 1], 0.0, H[0, 2]]
            self._world_matrix[1, :] = [H[1, 0], H[1, 1], 0.0, H[1, 2]]
            # H[2] = [h20, h21, 1.0] — perspective part, used by image_to_world

        else:
            raise ValueError(
                f"El método '{method}' requiere al menos 4 puntos "
                f"(recibidos: {n}). Usa method='affine' para 3 puntos."
            )

        self._world_valid = True
        self._world_src_points = img_pts.copy()
        self._world_dst_points = world_pts.copy()

    def image_to_world(self, x, y):
        """
        Transform image pixel (x, y) → world coordinates (wx, wy, wz).

        Uses homography when available (correct perspective), falls back to 4×4 matrix.
        3D-ready: returns (wx, wy, wz). In 2D mode wz is always 0.
        """
        if self._world_homography is not None:
            pt = self._world_homography @ np.array([x, y, 1.0])
            wx, wy = float(pt[0] / pt[2]), float(pt[1] / pt[2])
            return wx, wy, 0.0
        pt = self._world_matrix @ np.array([x, y, 0.0, 1.0])
        return float(pt[0]), float(pt[1]), float(pt[2])

    def world_to_image(self, wx, wy, wz=0.0):
        """
        Transform world coordinates → image pixel (x, y).

        Inverse of image_to_world. Uses inverted homography (correct perspective)
        or inverted 4×4 matrix.
        """
        if self._world_homography is not None:
            H_inv = np.linalg.inv(self._world_homography)
            pt = H_inv @ np.array([wx, wy, 1.0])
            ix, iy = float(pt[0] / pt[2]), float(pt[1] / pt[2])
            return ix, iy
        M_inv = np.linalg.inv(self._world_matrix)
        pt = M_inv @ np.array([wx, wy, wz, 1.0])
        return float(pt[0]), float(pt[1])

    @property
    def world_matrix(self):
        """Return the 4×4 world transformation matrix (extrinsics in future 3D mode)."""
        return self._world_matrix.copy()

    def clear_world(self):
        """Reset world calibration to identity."""
        self._world_matrix = np.eye(4, dtype=np.float64)
        self._world_homography = None
        self._world_valid = False
        self._world_src_points = np.empty((0, 2), dtype=np.float64)
        self._world_dst_points = np.empty((0, 2), dtype=np.float64)

    @staticmethod
    def find_calibration_grid(image, pattern='checkerboard', grid_size=(7, 7),
                               square_size=1.0):
        """
        Automatically detect calibration points in an image.

        Detects a calibration pattern (checkerboard or circle grid) and generates
        corresponding real-world coordinates. The result can be passed directly
        to set_world_from_points().

        Args:
            image: numpy array (grayscale or BGR) containing the calibration pattern.
            pattern: 'checkerboard' (default) or 'circles'.
            grid_size: (columns, rows) of interior corners / circle centers.
            square_size: Spacing between grid points in world units (mm).

        Returns:
            (img_points, world_points) tuple ready for set_world_from_points,
            or None if the pattern was not detected.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

        if pattern == 'checkerboard':
            ret, corners = cv2.findChessboardCorners(gray, grid_size, None)
            if not ret:
                return None
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            img_pts = corners.reshape(-1, 2)

        elif pattern == 'circles':
            flags = cv2.CALIB_CB_SYMMETRIC_GRID
            ret, centers = cv2.findCirclesGrid(gray, grid_size, None, flags)
            if not ret:
                flags = cv2.CALIB_CB_ASYMMETRIC_GRID
                ret, centers = cv2.findCirclesGrid(gray, grid_size, None, flags)
            if not ret:
                return None
            img_pts = centers.reshape(-1, 2)

        else:
            raise ValueError(f"Patrón desconocido: '{pattern}'. Usa 'checkerboard' o 'circles'.")

        # Generate world points on Z=0 plane matching grid layout
        world_pts = []
        for row in range(grid_size[1]):
            for col in range(grid_size[0]):
                world_pts.append([col * square_size, row * square_size])

        return img_pts, np.array(world_pts, dtype=np.float64)

    def draw_world_grid(self, spacing, range_x, range_y,
                        color=(200, 200, 0), thickness=1, layer_name='world_grid'):
        """
        Draw a world-coordinate grid overlay on the image.

        Each grid line is computed in world coordinates and projected back to image
        pixels via world_to_image(), so the grid follows the perspective of the
        calibrated scene.

        Args:
            spacing: Distance between grid lines in world units (mm).
            range_x: (min, max) in world X.
            range_y: (min, max) in world Y.
            color: BGR color tuple.
            thickness: Line thickness in pixels.
            layer_name: Overlay layer name.
        """
        self._ensure_layer(layer_name, source_tool='world')
        x_min, x_max = range_x
        y_min, y_max = range_y

        # Vertical lines (constant X)
        x = x_min
        while x <= x_max + 1e-9:
            p1 = self.world_to_image(x, y_min)
            p2 = self.world_to_image(x, y_max)
            self.draw_line(p1, p2, color=color, thickness=thickness, layer_name=layer_name)
            x += spacing

        # Horizontal lines (constant Y)
        y = y_min
        while y <= y_max + 1e-9:
            p1 = self.world_to_image(x_min, y)
            p2 = self.world_to_image(x_max, y)
            self.draw_line(p1, p2, color=color, thickness=thickness, layer_name=layer_name)
            y += spacing

    @staticmethod
    def from_array(arr):
        if isinstance(arr, VisionImage):
            return arr
        return VisionImage(arr)