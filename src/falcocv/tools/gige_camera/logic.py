# Logic module for GigeCameraLoader.

import cv2
import numpy as np
import threading
import os
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool
from falcocv.core.config import ConfigManager
from falcocv.core.vision_image import VisionImage

from falcocv.core.tool_base import CallbackList

# ---------------------------------------------------------------------------
# Harvesters is a runtime dependency; the guard keeps the rest of the app
# usable if an environment was installed without it.
# ---------------------------------------------------------------------------
try:
    from harvesters.core import Harvester
    import genicam.gentl
    _HARVESTERS_AVAILABLE = True
except ImportError:
    _HARVESTERS_AVAILABLE = False

_CAMERA_POOL = {}  # { "serial": {"h": Harvester, "ia": ImageAcquirer} }
_POOL_LOCK = threading.Lock()

@register_tool(prefix="[GIGE]", default_name="GigE Camera", group="Acquisition")
class GigeCameraLoader(BaseTool):
    """
    Acquires frames from a GigE Vision or USB3 Vision camera using Harvesters + GenTL.
    Supports persistent Custom GenICam features and Serial Number binding.
    """
    connection_changed = CallbackList()

    def __init__(self, name="GigE Camera"):
        super().__init__(name)
        self.inputs = {}
        self.outputs = {"output image": None}
        
        # Load CTI path from global config
        config = ConfigManager()
        default_cti = config.get_gige_cti_path()

        self.params = {
            "camera_type":        "Hardware",   # "Hardware" | "Simulation"
            "cti_path":           default_cti,   # Path to CTI file
            "camera_serial":      "",
            "trigger_mode":       "SoftwareTrigger",  # "FreeRun" | "SoftwareTrigger" | "HardwareTrigger"
            "exposure_us":        20000.0,
            "output_format":      "BGR",           # "BGR" | "Gray"
            "custom_features":    {},              # { "FeatureName": "ValueAsString" }
        }

        # Runtime state (not serialised)
        self._h = None
        self._ia = None
        self._stream_lock = threading.Lock()
        self._last_frame: np.ndarray | None = None
        self._discovered: list = []   # list of dicts {"vendor", "model", "serial", "id"}

    # ------------------------------------------------------------------
    # Public helpers
    # ------------------------------------------------------------------
    def discover(self) -> list:
        """Discover available cameras. Returns list of pure python dicts."""
        if not _HARVESTERS_AVAILABLE:
            return []
            
        if self.params["camera_type"] == "Simulation":
            return []

        cti = self.params.get("cti_path", "")
        if not os.path.exists(cti):
            raise FileNotFoundError(f"CTI file not found: {cti}")

        h = Harvester()
        h.add_file(cti)
        h.update()
        
        results = []
        for d in h.device_info_list:
            vendor = getattr(d, 'vendor', 'Unknown')
            model = getattr(d, 'model', 'Camera')
            # Extract serial number safely, fallback to id_
            serial = getattr(d, 'serial_number', getattr(d, 'id_', str(len(results))))
            results.append({
                "vendor": vendor,
                "model": model,
                "serial": serial,
                "id": getattr(d, 'id_', serial)
            })
            
        h.reset()
        self._discovered = results
        return results

    def is_connected(self) -> bool:
        return self._ia is not None or (self.params["camera_type"] == "Simulation" and getattr(self, "_sim_running", False))

    def should_persist_connection(self) -> bool:
        """Return True if this camera should stay connected across job reloads."""
        return self.params.get("camera_type") == "Hardware" and self.is_connected()

    def connection_key(self) -> tuple:
        """Unique key used by the main window to identify persistent cameras."""
        return (self.params.get("camera_serial", ""), self.params.get("cti_path", ""))

    def connect(self):
        """Open the camera and start streaming."""
        self.disconnect()

        if self.params["camera_type"] == "Simulation":
            self._sim_running = True
            self._sim_frame_count = 0
            self.connection_changed.emit(True)
            return

        target_serial = self.params.get("camera_serial", "")
        cti = self.params.get("cti_path", "")

        if not _HARVESTERS_AVAILABLE:
            raise RuntimeError("harvesters is not installed. Run: pip install harvesters")

        if not os.path.exists(cti):
            raise FileNotFoundError(f"CTI file not found: {cti}")

        # Single acquisition of _POOL_LOCK — check pool, create if needed, done.
        # This eliminates the TOCTOU window between two separate lock acquisitions.
        with _POOL_LOCK:
            # 1. Check pool first (fast path — no Harvester creation needed)
            if target_serial and target_serial in _CAMERA_POOL:
                entry = _CAMERA_POOL[target_serial]
                ia = entry.get("ia")
                stale_entry = True
                if ia is not None:
                    try:
                        if getattr(ia, "is_destroyed", False):
                            stale_entry = True
                        else:
                            _ = ia.remote_device.node_map
                            stale_entry = False
                    except Exception:
                        stale_entry = True

                if not stale_entry:
                    self._h = entry.get("h")
                    self._ia = ia
                    self._apply_camera_params()
                    self.connection_changed.emit(True)
                    return
                else:
                    del _CAMERA_POOL[target_serial]

            # 2. Not in pool (or stale) — create Harvester and connect
            h = Harvester()
            h.add_file(cti)
            h.update()

            if len(h.device_info_list) == 0:
                h.reset()
                raise RuntimeError("No cameras found by GenTL producer")

            # Find camera by serial
            idx = 0
            if target_serial:
                for i, d in enumerate(h.device_info_list):
                    s = getattr(d, 'serial_number', getattr(d, 'id_', str(i)))
                    if s == target_serial:
                        idx = i
                        break
            else:
                d = h.device_info_list[0]
                target_serial = getattr(d, 'serial_number', getattr(d, 'id_', "0"))
                self.params["camera_serial"] = target_serial

            self._h = h
            self._ia = self._h.create(idx)
            self._apply_camera_params()
            self._ia.start()
            _CAMERA_POOL[target_serial] = {"h": self._h, "ia": self._ia}

        self.connection_changed.emit(True)

    def disconnect(self):
        """Stop streaming and release the camera."""
        if self.params["camera_type"] == "Simulation":
            self._sim_running = False
            self._last_frame = None
            self.connection_changed.emit(False)
            return

        # Acquire with timeout to avoid blocking the UI if _grab_frame is mid-fetch
        if not self._stream_lock.acquire(timeout=1.5):
            print("WARNING: _stream_lock timeout in disconnect — forcing disconnect")
        try:
            with _POOL_LOCK:
                serial = self.params.get("camera_serial", "")
                if serial in _CAMERA_POOL and _CAMERA_POOL[serial]["h"] == self._h:
                    del _CAMERA_POOL[serial]

            if self._ia is not None:
                try:
                    self._ia.stop()
                    if hasattr(self._ia, "destroy"):
                        self._ia.destroy()
                except Exception:
                    pass
                self._ia = None
            if self._h is not None:
                try:
                    if hasattr(self._h, "reset"):
                        self._h.reset()
                except Exception:
                    pass
                self._h = None
            self._last_frame = None
        finally:
            self._stream_lock.release()
        self.connection_changed.emit(False)

    def _apply_camera_params(self):
        """Push all configured parameters to the camera."""
        if self._ia is None:
            return
        node_map = self._ia.remote_device.node_map
        
        # 1. Trigger
        try:
            if self.params["trigger_mode"] == "SoftwareTrigger":
                if hasattr(node_map, "TriggerMode"): node_map.TriggerMode.value = "On"
                if hasattr(node_map, "TriggerSource"): node_map.TriggerSource.value = "Software"
            elif self.params["trigger_mode"] == "HardwareTrigger":
                if hasattr(node_map, "TriggerMode"): node_map.TriggerMode.value = "On"
                if hasattr(node_map, "TriggerSource"): node_map.TriggerSource.value = "Line1"
            else:
                if hasattr(node_map, "TriggerMode"): node_map.TriggerMode.value = "Off"
        except Exception as e:
            print(f"Trigger param err: {type(e).__name__}: {e}")

        # 2. Exposure
        try:
            if hasattr(node_map, "ExposureAuto"): node_map.ExposureAuto.value = "Off"
            if hasattr(node_map, "ExposureTime"): node_map.ExposureTime.value = float(self.params["exposure_us"])
            elif hasattr(node_map, "ExposureTimeAbs"): node_map.ExposureTimeAbs.value = float(self.params["exposure_us"])
        except Exception as e:
            print(f"Exposure param err: {type(e).__name__}: {e}")

        # 3. Custom Features
        customs = self.params.get("custom_features", {})
        for feat, val in customs.items():
            if hasattr(node_map, feat):
                try:
                    node = getattr(node_map, feat)
                    if not node.is_writable():
                        continue
                    # Safely convert
                    ptype = node.node.principal_interface_type
                    if ptype == "IInteger":
                        node.value = int(val)
                    elif ptype == "IFloat":
                        node.value = float(val)
                    elif ptype == "IBoolean":
                        node.value = str(val).strip().lower() in ("true", "1", "yes")
                    else:
                        node.value = str(val) # String, Enumeration, etc.
                except Exception as e:
                    print(f"Custom Feature '{feat}' err: {type(e).__name__}: {e}")

    def _grab_frame(self, override_software_trigger=False) -> np.ndarray | None:
        """Grab one frame from the stream and convert to BGR/Gray numpy."""
        if self.params["camera_type"] == "Simulation":
            # Simulation: Fast and efficient
            arr = np.zeros((480, 640), dtype=np.uint8)
            self._sim_frame_count += 1
            x = (self._sim_frame_count * 5) % 640
            cv2.circle(arr, (x, 240), 50, 255, -1)
            cv2.putText(arr, f"SIM FRAME {self._sim_frame_count}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, 200, 1)
            return arr

        # Keep the critical section under lock: _ia check, trigger, fetch.
        # Data conversion (reshape, copy, cvtColor) runs outside the lock.
        arr = None
        self._stream_lock.acquire()
        try:
            if self._ia is None:
                return None

            is_sw_trigger = (self.params["trigger_mode"] == "SoftwareTrigger" or override_software_trigger)
            if is_sw_trigger:
                try:
                    node_map = self._ia.remote_device.node_map
                    if hasattr(node_map, "TriggerSoftware"):
                        node_map.TriggerSoftware.execute()
                except Exception as e:
                    print(f"TriggerSoftware err: {e}")

            MAX_DRAIN = 30
            for _ in range(MAX_DRAIN):
                try:
                    with self._ia.fetch(timeout=0.001) as buffer:
                        component = buffer.payload.components[0]
                        if component.num_components_per_pixel == 1:
                            arr = component.data.reshape(component.height, component.width).copy()
                        else:
                            arr = component.data.reshape(component.height, component.width, component.num_components_per_pixel).copy()
                except Exception:
                    break

            if arr is None:
                try:
                    with self._ia.fetch(timeout=1.0) as buffer:
                        component = buffer.payload.components[0]
                        if component.num_components_per_pixel == 1:
                            arr = component.data.reshape(component.height, component.width).copy()
                        else:
                            arr = component.data.reshape(component.height, component.width, component.num_components_per_pixel).copy()
                except Exception as e:
                    print(f"Fetch err: {e}")
                    return None
        finally:
            self._stream_lock.release()

        if arr is None:
            return None

        if len(arr.shape) == 3 and arr.shape[2] == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        return arr

    def _override_freerun(self, enable: bool):
        """Temporarily force FreeRun or restore configured trigger."""
        if not self.is_connected() or self._ia is None:
            return
        node_map = self._ia.remote_device.node_map
        try:
            if enable:
                if hasattr(node_map, "TriggerMode"):
                    node_map.TriggerMode.value = "Off"
            else:
                self._apply_camera_params()
        except Exception as e:
            print(f"Override FreeRun err: {e}")

    def snap(self):
        """Connect if needed, grab one frame, restore trigger mode and emit."""
        if not self.is_connected():
            self.connect()
            if not self.is_connected():
                return None

        if self.params["trigger_mode"] != "SoftwareTrigger":
            try:
                nm = self._ia.remote_device.node_map
                if hasattr(nm, "TriggerMode"):
                    nm.TriggerMode.value = "On"
                if hasattr(nm, "TriggerSource"):
                    nm.TriggerSource.value = "Software"
            except Exception:
                pass

        frame = self._grab_frame(override_software_trigger=True)

        if self.params["trigger_mode"] != "SoftwareTrigger":
            self._apply_camera_params()

        if frame is not None:
            self._last_frame = frame
            self.outputs["output image"] = VisionImage(frame.copy())
            self.mark_dirty()
            return self.outputs["output image"]
        return None

    def start_preview(self):
        """Connect if needed and switch to FreeRun for live UI preview."""
        if not self.is_connected():
            self.connect()
            if not self.is_connected():
                raise RuntimeError("Could not connect camera for live preview")
        self._override_freerun(True)

    def stop_preview(self):
        """Restore configured trigger mode after live preview."""
        self._override_freerun(False)

    def grab_preview_frame(self) -> np.ndarray | None:
        """Grab a single frame without triggering the workflow."""
        return self._grab_frame(override_software_trigger=False)

    def run(self, advance=False):
        if not self.is_connected():
            self.connect()

        if advance or self._last_frame is None:
            frame = self._grab_frame()
            if frame is None:
                raise RuntimeError("Timeout or error grabbing frame")
            self._last_frame = frame

        self.outputs["output image"] = VisionImage(self._last_frame.copy())
        self.data_changed.emit()

    def get_config_widget(self):
        from .config import GigeCameraConfig
        return GigeCameraConfig(self)

    def serialize(self, job_manager=None) -> dict:
        return super().serialize(job_manager)

    def deserialize(self, data: dict, job_manager=None):
        super().deserialize(data, job_manager)
