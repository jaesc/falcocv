import cv2
import numpy as np
import math
import base64
import threading
from falcocv.core.tool_base import BaseTool, CallbackList
from falcocv.core.tool_base import ToolResult, ToolResults
from falcocv.core.tool_registry import register_tool
from falcocv.core.roi_utils import extract_roi
from falcocv.core.vision_image import VisionImage
from falcocv.core.pointcloud_utils import propagate_3d


# --- Background cache builder -------------------------------------------
class _CacheWorker:
    """Builds the template cache in a background thread."""
    def __init__(self, template_gray, params, template_size, existing_cache=None):
        self.finished = CallbackList()
        self._template_gray = template_gray
        self._params = params
        self._template_size = template_size
        self._existing_cache = existing_cache or {}
        self._cache = {}
        self._cache_params = {}
        self.stop_requested = False

    def run(self):
        try:
            levels  = int(self._params["pyramid_levels"])
            angle_step = max(0.1, float(self._params["angle_step"]))
            angles  = list(np.arange(float(self._params["angle_start"]),
                                     float(self._params["angle_end"]) + 0.001, angle_step))
            scales  = np.arange(self._params["scale_start"],
                                self._params["scale_end"] + 0.001,
                                max(0.01, self._params["scale_step"])).tolist()

            # Compute expected key set, rounding floats for stability
            expected = set()
            for s in scales:
                for a in angles:
                    for l in range(levels):
                        expected.add((round(s, 6), round(a, 6), l))

            # Start with existing cache, remove stale keys
            cache = dict(self._existing_cache)
            stale = [k for k in cache if k not in expected]
            for k in stale:
                del cache[k]

            # Only compute missing keys
            for s in scales:
                if self.stop_requested: return
                for a in angles:
                    if self.stop_requested: return
                    key_0 = (round(s, 6), round(a, 6), 0)
                    if key_0 not in cache:
                        t_base = _transform(self._template_gray, a, s)
                        cache[key_0] = t_base
                    else:
                        t_base = cache[key_0]
                    curr = t_base
                    for l in range(1, levels):
                        if self.stop_requested: return
                        key = (round(s, 6), round(a, 6), l)
                        if key not in cache:
                            curr = cv2.pyrDown(curr)
                            cache[key] = curr
                        else:
                            curr = cache[key]

            self._cache = cache

            cache_params = {
                "angle_start":   self._params["angle_start"],
                "angle_end":     self._params["angle_end"],
                "angle_step":    self._params["angle_step"],
                "scale_start":   self._params["scale_start"],
                "scale_end":     self._params["scale_end"],
                "scale_step":    self._params["scale_step"],
                "pyramid_levels": self._params["pyramid_levels"],
                "template_size": self._template_size,
            }
            self._cache_params = cache_params
            if not self.stop_requested:
                self.finished.emit(cache, cache_params)
        except Exception:
            self._cache = {}
            self._cache_params = {}
            if not self.stop_requested:
                self.finished.emit({}, {})


def _transform(template, angle, scale):
    """Rotate + scale a template image, auto-expanding the canvas."""
    if angle == 0 and scale == 1.0:
        return template
    h, w = template.shape[:2]
    center = (w / 2.0, h / 2.0)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    cos_v, sin_v = abs(M[0, 0]), abs(M[0, 1])
    new_w = int((h * sin_v) + (w * cos_v))
    new_h = int((h * cos_v) + (w * sin_v))
    M[0, 2] += (new_w / 2.0) - center[0]
    M[1, 2] += (new_h / 2.0) - center[1]
    return cv2.warpAffine(template, M, (new_w, new_h), flags=cv2.INTER_LINEAR)
# -------------------------------------------------------------------------


class PatternMatchResult(ToolResult):
    def __init__(self, id, x, y, angle, scale, score):
        super().__init__(id=id, x=x, y=y, angle=angle, scale=scale, score=score)


@register_tool(prefix="[PATT]", default_name="Pattern Matcher", group="Vision Tools")
class PatternMatcher(BaseTool):
    """
    Industrial Multi-Scale Pattern Matcher with User-Defined Origin.
    """
    def __init__(self, name="Pattern Matcher"):
        super().__init__(name)
        self.params = {
            "train_roi": [0, 0, 0, 0, 0],
            "search_roi": [0, 0, 0, 0, 0],
            "score_threshold": 0.7,
            "angle_start": -10,
            "angle_end": 10,
            "angle_step": 1,
            "scale_start": 1.0,
            "scale_end": 1.0,
            "scale_step": 0.1,
            "pyramid_levels": 3,
            "max_matches": 1,
            "show_labels": True,
            "show_search_roi": True,
            "show_overlays": True,
            "show_found_region": True,
            "show_found_origin": True,
            "train_image_b64": None,
            "template_b64": None,
            "is_trained": False,
            "origin_x": 0.0, # Relative to template center
            "origin_y": 0.0,
            "act_as_fixture": False,
            "copy_overlays_to_input": True,
        }
        self.inputs = {
            "input image": None,
        }
        self.outputs = {
            "output image": None,
            "count": 0,
            "trained": False,
            "results": ToolResults(),
        }
        self._static_train_image = None
        self._static_train_fixture = {'x': 0.0, 'y': 0.0, 'angle': 0.0}
        self._template_gray = None
        self._template_color = None
        self._template_size = (0, 0)
        self._template_cache = {}  # {(scale, angle, level): image}
        self._cache_params = {}    # To detect when cache needs rebuild
        self._cache_ready = False  # False while a background rebuild is in progress
        self._cache_worker = None  # Active worker (kept to prevent GC)
        self._cache_thread = None  # Thread for async rebuild

    @property
    def is_trained(self):
        return self._template_gray is not None

    def save_train_image(self):
        src_obj = self.inputs.get("input image")
        if src_obj is None: return

        from falcocv.core.vision_image import VisionImage
        src = src_obj.data if isinstance(src_obj, VisionImage) else src_obj

        if isinstance(src, np.ndarray):
            self._static_train_image = src.copy()
            _, buffer = cv2.imencode('.png', src)
            self.params["train_image_b64"] = base64.b64encode(buffer).decode('utf-8')
            # Store fixture parameters for coordinate conversion
            if isinstance(src_obj, VisionImage):
                self._static_train_fixture = {
                    'x': src_obj.x, 'y': src_obj.y, 'angle': src_obj.angle
                }
            else:
                self._static_train_fixture = {'x': 0.0, 'y': 0.0, 'angle': 0.0}
            # Save fixture data to params for serialization
            self.params["_static_train_fixture"] = dict(self._static_train_fixture)
            self.clear_training(keep_image=True)
            self.data_changed.emit()

    def clear_training(self, keep_image=False):
        self._template_gray = None
        self._template_color = None
        self._template_size = (0, 0)
        self.outputs["trained"] = False
        self.outputs["results"] = ToolResults()
        self.outputs["count"] = 0
        if not keep_image:
            self._static_train_image = None
            self.params["train_image_b64"] = None
        self._static_train_fixture = {'x': 0.0, 'y': 0.0, 'angle': 0.0}
        self.params["_static_train_fixture"] = dict(self._static_train_fixture)
        self.params["template_b64"] = None
        self.params["is_trained"] = False
        self.params["origin_x"] = 0.0
        self.params["origin_y"] = 0.0
        self.data_changed.emit()

    def deserialize(self, data: dict, job_manager=None):
        """Restore tool state from saved job data, rebuilding internal state from base64 images."""
        super().deserialize(data, job_manager)
        self._restore_state_from_params()

    def _restore_state_from_params(self):
        """Rebuild internal state (_static_train_image, _template_gray, etc.) from params base64 data."""
        # Restore training image
        if self.params.get("train_image_b64") is not None:
            try:
                img_data = base64.b64decode(self.params["train_image_b64"])
                nparr = np.frombuffer(img_data, np.uint8)
                self._static_train_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                self._static_train_fixture = self.params.get("_static_train_fixture", {'x': 0.0, 'y': 0.0, 'angle': 0.0})
            except Exception:
                self._static_train_image = None
        else:
            self._static_train_image = None

        # Restore template
        if self.params.get("template_b64") is not None and self.params.get("is_trained"):
            try:
                img_data = base64.b64decode(self.params["template_b64"])
                nparr = np.frombuffer(img_data, np.uint8)
                roi = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if roi is not None:
                    self._template_color = roi.copy()
                    self._template_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
                    self._template_size = (self._template_gray.shape[1], self._template_gray.shape[0])
                    self.outputs["trained"] = True
                    self._rebuild_cache(async_=False)
                else:
                    self._clear_template_state()
            except Exception:
                self._clear_template_state()
        else:
            self._clear_template_state()

    def _clear_template_state(self):
        """Reset template-related internal state."""
        self._template_gray = None
        self._template_color = None
        self._template_size = (0, 0)
        self._template_cache = {}
        self._cache_ready = False
        self._cache_params = {}
        self.outputs["trained"] = False

    def get_config_widget(self):
        from .config import PatternMatcherConfig
        return PatternMatcherConfig(self)

    def train_from_roi(self, x, y, w, h, a=0):
        """Train from a ROI in fixture coordinates (current config API)."""
        self.params["train_roi"] = [x, y, w, h, a]
        self.train_pattern()

    def train_pattern(self):
        img = self._static_train_image
        if img is None:
            raise ValueError("No training image captured.")
        if img.size == 0:
            raise ValueError("Training image is empty.")

        roi_data = self.params["train_roi"]
        if len(roi_data) == 4:
            x, y, w, h = roi_data
            a = 0
        elif len(roi_data) == 5:
            x, y, w, h, a = roi_data
        else:
            raise ValueError(f"train_roi must have 4 or 5 elements, got {len(roi_data)}: {roi_data}")

        if w <= 0 or h <= 0:
            roi = img
        else:
            # Convert fixture-relative ROI to image-space using stored fixture
            fx = self._static_train_fixture
            if abs(fx.get('x', 0)) > 0.01 or abs(fx.get('y', 0)) > 0.01 or abs(fx.get('angle', 0)) > 0.01:
                from falcocv.core.vision_image import VisionImage
                tmp = VisionImage(img, x=fx['x'], y=fx['y'], angle=fx['angle'])
                abs_roi = tmp.fixture_to_image([x, y, w, h, a])
                x, y, w, h, a = abs_roi
            # Compute intersection of requested ROI with image bounds
            img_h, img_w = img.shape[:2]
            x1 = max(int(x), 0)
            y1 = max(int(y), 0)
            x2 = min(int(x + w), img_w)
            y2 = min(int(y + h), img_h)
            if x2 <= x1 or y2 <= y1:
                raise ValueError(f"ROI is outside image bounds (img: {img_w}x{img_h}).")
            x = x1
            y = y1
            w = x2 - x1
            h = y2 - y1
            # Affine extraction (Rotated ROI)
            center = (x + w / 2, y + h / 2)
            M = cv2.getRotationMatrix2D(center, a, 1.0)
            roi = cv2.warpAffine(img, M, (img_w, img_h), flags=cv2.INTER_LINEAR)
            roi = roi[y:y + h, x:x + w]

        if roi is None or roi.size == 0:
            raise ValueError(
                f"Extracted ROI is empty. ROI coords: x={x}, y={y}, w={w}, h={h}, "
                f"image size: {img.shape[1]}x{img.shape[0]}."
            )
        self._template_color = roi.copy()
        self._template_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
        self._template_size = (self._template_gray.shape[1], self._template_gray.shape[0])

        _, buffer = cv2.imencode('.png', roi)
        self.params["template_b64"] = base64.b64encode(buffer).decode('utf-8')
        self.params["is_trained"] = True
        self.outputs["trained"] = True
        self._cache_ready = False
        self._cache_params = {}
        self._template_cache = {}    # Force full cache rebuild (existing_cache reuses old)
        self._rebuild_cache(async_=False)
        self.data_changed.emit()

    def _rebuild_cache(self, async_=True):
        """Starts an asynchronous cache rebuild on a background thread."""
        if not self.is_trained:
            self._template_cache = {}
            self._cache_ready = False
            return

        # Stop any running worker before starting a new one
        if self._cache_worker is not None:
            self._cache_worker.stop_requested = True
            if self._cache_thread is not None and self._cache_thread.is_alive():
                self._cache_thread.join(timeout=5)

        self._cache_ready = False
        worker = _CacheWorker(
            self._template_gray,
            dict(self.params),
            self._template_size,
            existing_cache=self._template_cache,
        )
        worker.finished.connect(self._on_cache_built)
        self._cache_worker = worker
        if async_:
            self._cache_thread = threading.Thread(target=worker.run, daemon=True)
            self._cache_thread.start()
        else:
            # Synchronous rebuild - process immediately in current thread
            worker.run()
            self._on_cache_built(worker._cache, worker._cache_params, emit_signal=False)

    def _on_cache_built(self, cache, cache_params, emit_signal=True):
        """Slot called when the background worker finishes building the cache."""
        self._template_cache = cache
        self._cache_params = cache_params
        self._cache_ready = bool(cache)
        # Emit signal for async mode; sync mode will emit at end of run()
        if emit_signal and self._cache_ready:
            self.data_changed.emit()

    def _transformed_template(self, template, angle, scale):
        return _transform(template, angle, scale)

    def run(self, advance=False):
        if self._static_train_image is None and self.params["train_image_b64"] is not None:
            try:
                img_data = base64.b64decode(self.params["train_image_b64"])
                nparr = np.frombuffer(img_data, np.uint8)
                self._static_train_image = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
            except: pass

        if self._template_gray is None and self.params.get("template_b64") is not None:
            try:
                img_data = base64.b64decode(self.params["template_b64"])
                nparr = np.frombuffer(img_data, np.uint8)
                roi = cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
                if roi is not None:
                    self._template_color = roi.copy()
                    self._template_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if roi.ndim == 3 else roi.copy()
                    self._template_size = (self._template_gray.shape[1], self._template_gray.shape[0])
            except: pass

        img_input = self.inputs.get("input image")
        self.outputs["output image"] = None
        self.outputs["count"] = 0
        self.outputs["results"] = ToolResults()
        self.outputs["trained"] = self.is_trained
        if img_input is None: self.data_changed.emit(); return

        # Ensure we work with a VisionImage so overlays and fixturing metadata
        # are preserved when producing outputs. If the input is already a
        # VisionImage, reuse and copy it; otherwise convert from the numpy
        # array.
        from falcocv.core.vision_image import VisionImage
        if isinstance(img_input, VisionImage):
            v_img = img_input
        else:
            v_img = VisionImage.from_array(img_input)
        # Create output VisionImage sharing the input data (no copy) with its own overlay storage
        vision_img = VisionImage(v_img.data, x=v_img.x, y=v_img.y, angle=v_img.angle,
                                  scale_x=v_img.scale_x, scale_y=v_img.scale_y,
                                   scale_z=v_img.scale_z, offset_z=v_img.offset_z)
        vision_img.inherit_overlays_from(v_img)
        propagate_3d(v_img, vision_img, regenerate=False)
        # Pre-create default layer with source_tool so new overlays are tagged correctly
        vision_img.get_overlay_layer("default", source_tool=self.name)
        _layer = vision_img.get_overlay_layer("default")
        _layer.source_tool = self.name
        _layer.elements = [e for e in _layer.elements if e.source_tool != self.name]

        roi = extract_roi(v_img, self.params["search_roi"])
        roi_img = roi["roi_src"]
        abs_x_roi, abs_y_roi = roi["ax"], roi["ay"]

        if self.params["show_search_roi"]:
            # Draw search ROI in absolute coordinates (after fixture transformation)
            vision_img.get_overlay_layer("default").add_roi([abs_x_roi, abs_y_roi, roi["aw"], roi["ah"], 0], color=(255, 128, 0), thickness=2)

        if not self.is_trained: self.outputs["output image"] = vision_img; self.data_changed.emit(); return

        # Search
        levels = int(self.params["pyramid_levels"])

        # Trigger cache rebuild if parameters changed; skip matching this frame
        current_cache_keys = ["angle_start", "angle_end", "angle_step",
                              "scale_start", "scale_end", "scale_step", "pyramid_levels"]
        needs_rebuild = (
            not self._cache_ready
            or any(self.params[k] != self._cache_params.get(k) for k in current_cache_keys)
            or self._cache_params.get("template_size") != self._template_size
        )
        if needs_rebuild:
            # Synchronous rebuild ensures dependent tools get correct output on first pass
            self._rebuild_cache(async_=False)

        if roi_img is None or roi_img.size == 0:
            self.outputs["output image"] = vision_img
            self.data_changed.emit()
            return
        # Extract raw data for OpenCV
        roi_raw = roi_img.data if isinstance(roi_img, VisionImage) else roi_img
        search_gray = cv2.cvtColor(roi_raw, cv2.COLOR_BGR2GRAY) if roi_raw.ndim == 3 else roi_raw.copy()
        search_pyr = [search_gray]
        for _ in range(levels - 1): search_pyr.append(cv2.pyrDown(search_pyr[-1]))

        fine_step  = max(0.1, float(self.params["angle_step"]))
        coarse_step = fine_step * 4.0
        angles_fine   = list(np.arange(float(self.params["angle_start"]),
                                       float(self.params["angle_end"]) + 0.001, fine_step))
        angles_coarse = list(np.arange(float(self.params["angle_start"]),
                                       float(self.params["angle_end"]) + 0.001,
                                       max(fine_step, coarse_step)))
        scales = np.arange(self.params["scale_start"],
                           self.params["scale_end"] + 0.001,
                           max(0.01, self.params["scale_step"])).tolist()
        threshold = float(self.params["score_threshold"])
        limit     = int(self.params["max_matches"]) if self.params["max_matches"] > 0 else 100

        top_idx  = levels - 1
        top_img  = search_pyr[top_idx]
        coarse_threshold = threshold * 0.6  # Permissive during coarse sweep

        # ---- Paso 1+2: Coarse sweep → fine sweep around promising angles ----
        # Use minMaxLoc (single best per (s,a)) for speed.
        # We ALWAYS expand all coarse angles to fine neighbors so we never miss
        # a pattern whose true angle falls between two coarse samples.
        promising_angles = set(angles_coarse)  # Always expand around all coarse angles
        coarse_candidates = []
        for s in scales:
            for a in angles_coarse:
                t = self._template_cache.get((s, a, top_idx))
                if t is None or t.shape[0] > top_img.shape[0] or t.shape[1] > top_img.shape[1]: continue
                res = cv2.matchTemplate(top_img, t, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)  # Single best per (s,a)
                if max_val >= threshold * 0.55:  # Permissive: catch weak hits to expand around them
                    coarse_candidates.append({"x": max_loc[0], "y": max_loc[1],
                                              "score": max_val, "angle": a, "scale": s})

        # Expand ALL coarse angles to their fine neighbors
        fine_angles = set()
        n_steps = int(math.ceil(coarse_step / fine_step))
        for pa in promising_angles:
            for da in (i * fine_step for i in range(-n_steps, n_steps + 1)):
                candidate_angle = round((pa + da) / fine_step) * fine_step
                if int(self.params["angle_start"]) <= candidate_angle <= int(self.params["angle_end"]):
                    fine_angles.add(candidate_angle)

        # Fine sweep on the intermediate angles (not already covered by coarse)
        candidates = list(coarse_candidates)
        for s in scales:
            for a in sorted(fine_angles):
                if a in angles_coarse: continue  # Already covered
                t = self._template_cache.get((s, a, top_idx))
                if t is None or t.shape[0] > top_img.shape[0] or t.shape[1] > top_img.shape[1]: continue
                res = cv2.matchTemplate(top_img, t, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                if max_val >= threshold * 0.55:
                    candidates.append({"x": max_loc[0], "y": max_loc[1],
                                       "score": max_val, "angle": a, "scale": s})

        # Early NMS at top-pyramid level
        candidates.sort(key=lambda x: x["score"], reverse=True)
        pruned = []
        for c in candidates:
            if not any(abs(c["x"] - p["x"]) < 3 and abs(c["y"] - p["y"]) < 3 for p in pruned):
                pruned.append(c)
                if len(pruned) >= limit * 5: break

        # ---- Pyramid refinement (with early-exit on score drop) -------------
        refined = []
        early_exit_threshold = threshold * 0.85

        def get_subpixel_offset(y1, y2, y3):
            # Parabolic fit peak offset: delta = 0.5 * (y1 - y3) / (y1 - 2*y2 + y3)
            denom = (y1 - 2*y2 + y3)
            return 0.5 * (y1 - y3) / denom if abs(denom) > 1e-6 else 0.0

        for cand in pruned:
            curr_x, curr_y, curr_score = cand["x"], cand["y"], cand["score"]
            curr_a, curr_s = cand["angle"], cand["scale"]
            ok = True
            last_res = None
            last_sx, last_sy = 0, 0

            for l in range(top_idx, -1, -1):
                search_img = search_pyr[l]
                t = self._template_cache.get((curr_s, curr_a, l))
                if t is None: ok = False; break
                tw, th = t.shape[1], t.shape[0]

                # Dynamic search window around current candidate
                win = 3 if l < top_idx else 0
                sx = max(0, int(curr_x - win)); sy = max(0, int(curr_y - win))
                ex = min(search_img.shape[1], int(curr_x + tw + win))
                ey = min(search_img.shape[0], int(curr_y + th + win))
                if ex - sx < tw or ey - sy < th: ok = False; break

                res = cv2.matchTemplate(search_img[sy:ey, sx:ex], t, cv2.TM_CCOEFF_NORMED)
                _, max_val, _, max_loc = cv2.minMaxLoc(res)
                curr_x, curr_y, curr_score = sx + max_loc[0], sy + max_loc[1], max_val

                if curr_score < early_exit_threshold: ok = False; break
                if l > 0:
                    curr_x *= 2; curr_y *= 2
                else:
                    last_res = res
                    last_sx, last_sy = sx, sy

            if ok and curr_score >= threshold and last_res is not None:
                # 1. Sub-pixel Refinement (X, Y)
                h_r, w_r = last_res.shape
                # Local coords within last_res
                lx, ly = int(curr_x - last_sx), int(curr_y - last_sy)

                final_x, final_y = float(curr_x), float(curr_y)
                if 0 < lx < w_r - 1:
                    final_x += get_subpixel_offset(last_res[ly, lx-1], last_res[ly, lx], last_res[ly, lx+1])
                if 0 < ly < h_r - 1:
                    final_y += get_subpixel_offset(last_res[ly-1, lx], last_res[ly, lx], last_res[ly+1, lx])

                # 2. Sub-degree Refinement (Angle)
                # We need scores at best_angle - step and best_angle + step at the same position/scale
                step = float(self.params["angle_step"])
                a_left, a_right = curr_a - step, curr_a + step
                s_left, s_right = 0.0, 0.0

                # Quick re-checks at level 0 for angle interpolation
                for ang, score_ptr in [(a_left, "s_left"), (a_right, "s_right")]:
                    t_ang = self._template_cache.get((curr_s, ang, 0))
                    if t_ang is not None:
                        # Extract 1x1 at the exact found location (integer part is fine for angle score)
                        ix, iy = int(final_x), int(final_y)
                        if 0 <= ix < search_pyr[0].shape[1] - t_ang.shape[1] and \
                           0 <= iy < search_pyr[0].shape[0] - t_ang.shape[0]:
                            patch = search_pyr[0][iy : iy + t_ang.shape[0], ix : ix + t_ang.shape[1]]
                            res_ang = cv2.matchTemplate(patch, t_ang, cv2.TM_CCOEFF_NORMED)
                            if score_ptr == "s_left": s_left = float(res_ang[0,0])
                            else: s_right = float(res_ang[0,0])

                final_a = float(curr_a)
                if s_left > 0 and s_right > 0:
                    final_a += get_subpixel_offset(s_left, curr_score, s_right) * step

                refined.append({"x": final_x, "y": final_y, "score": curr_score,
                                 "angle": final_a, "scale": curr_s,
                                 "w": self._template_size[0] * curr_s,
                                 "h": self._template_size[1] * curr_s})

        refined.sort(key=lambda x: x["score"], reverse=True)
        final_results, kept = ToolResults(), []
        for r in refined:
            too_close = False
            for k in kept:
                if math.sqrt((r["x"] - k["x"])**2 + (r["y"] - k["y"])**2) < 20: too_close = True; break
            if not too_close:
                kept.append(r);
                if len(kept) >= limit: break

        # Origin Offset
        ox, oy = self.params["origin_x"], self.params["origin_y"]
        tw, th = self._template_size

        for idx, r in enumerate(kept):
            rad = np.deg2rad(-r["angle"])
            cos_a, sin_a = np.cos(rad), np.sin(rad)

            def transform_pt(px, py):
                tx, ty = (px - tw/2) * r["scale"], (py - th/2) * r["scale"]
                rx, ry = tx * cos_a - ty * sin_a, tx * sin_a + ty * cos_a
                # Use absolute ROI offsets (converted from fixture coordinates)
                return (rx + r["x"] + abs_x_roi + (tw*r["scale"])/2, ry + r["y"] + abs_y_roi + (th*r["scale"])/2)

            # Final result is the transformed defined origin
            cx, cy = transform_pt(tw/2 + ox, th/2 + oy)
            final_results.append(PatternMatchResult(id=idx, x=cx, y=cy, angle=r["angle"], scale=r["scale"], score=r["score"]))

            ref = f"match_{idx}"
            pts = [transform_pt(0,0), transform_pt(tw,0), transform_pt(tw,th), transform_pt(0,th)]
            if self.params["show_found_region"]:
                vision_img.get_overlay_layer("default").add_contour(pts, color=(0, 255, 0), thickness=2, ref_id=ref)

            if self.params["show_found_origin"]:
                vision_img.get_overlay_layer("default").add_axes(
                    (int(cx), int(cy)), angle=r["angle"], length=30,
                    color_x=(0, 0, 255), color_y=(0, 255, 0), thickness=2, ref_id=ref
                )

            if self.params["show_labels"]:
                vision_img.get_overlay_layer("default").add_text(f"#{idx} {r['score']:.2f}", position=(cx + 10, cy - 10), color=(0, 255, 255), font_scale=0.5)

        # Fixturing: If enabled and we have at least one match, output fixturized image
        if self.params.get("act_as_fixture") and len(final_results) > 0:
            best = final_results[0]
            # Apply fixture transform to the already-overlay-drawn image
            vision_img.x = best.x
            vision_img.y = best.y
            vision_img.angle = best.angle

        self.outputs["output image"] = vision_img

        self.outputs["count"] = len(final_results)
        self.outputs["results"] = final_results
        self._copy_overlays_to_input(vision_img, v_img)
        self.data_changed.emit()


