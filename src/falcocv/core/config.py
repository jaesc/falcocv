import json
import os
from falcocv.core.config_paths import state_dir

class ConfigManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._init_config()
        return cls._instance

    def _init_config(self):
        config_dir = state_dir()
        self.config_path = os.path.join(config_dir, "config.json")
        self.defaults = {
            "window": {
                "x": 100,
                "y": 100,
                "width": 1200,
                "height": 800,
                "maximized": False,
                "editor_splitter": [400, 600]
            },
            "colors": {
                "tool_input": "#4EC9B0",
                "tool_output": "#2196F3",
                "block_input": "#2E7D32",
                "block_output": "#1565C0"
            },
            "recent_files": [],
            "gige_camera": {
                "cti_path": ""
            },
            "depth_viewer": {
                "colormap": "TURBO",
                "vmin": None,
                "vmax": None,
                "use_colormap": True,
            },
            "viewer3d": {
                "point_size": 3.0,
            },
            "toolbox_visible": True,
            "show_overlays": True,
            "tool_splitters": {},
            "resolution_mode": "auto",
            "theme": "dark",
            "web_port": 8000,
            "debug_max_depth": 10,
        }
        self._tool_splitter_defaults = {
            "EdgeDetection":    [25, 75],
            "PatternMatcher":   [25, 75],
            "BarcodeDecoder":   [25, 75],
            "ImageLoader":      [20, 80],
            "HistogramTool":    [25, 75],
            "BlobDetector":     [25, 75],
            "FiltersTool":      [25, 75],
            "MeasurementTool":  [25, 75],
            "GigeCamera":       [35, 65],
            "PythonScriptTool": [55, 45, 70, 30],
            "Block":            [30, 70],
        }
        self.data = self.load()

    def get_recent_files(self):
        return self.data.get("recent_files", [])

    def add_recent_file(self, path):
        path = os.path.abspath(path)
        recents = self.get_recent_files()
        if path in recents:
            recents.remove(path)
        recents.insert(0, path)
        self.data["recent_files"] = recents[:10]
        self.save()

    def load(self):
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r") as f:
                    return {**self.defaults, **json.load(f)}
            except:
                return self.defaults
        return self.defaults

    def save(self):
        with open(self.config_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def get_editor_splitter(self):
        return self.data["window"].get("editor_splitter", [400, 600])

    def set_editor_splitter(self, sizes):
        self.data["window"]["editor_splitter"] = sizes
        self.save()

    def get_color(self, key):
        return self.data["colors"].get(key, self.defaults["colors"][key])

    def get_gige_cti_path(self):
        return self.data.get("gige_camera", {}).get("cti_path", "")

    def set_gige_cti_path(self, path):
        if "gige_camera" not in self.data:
            self.data["gige_camera"] = {}
        self.data["gige_camera"]["cti_path"] = path
        self.save()

    def get_show_overlays(self) -> bool:
        return self.data.get("show_overlays", True)

    def set_show_overlays(self, visible: bool) -> None:
        self.data["show_overlays"] = visible
        self.save()

    def get_toolbox_visible(self) -> bool:
        return self.data.get("toolbox_visible", True)

    def set_toolbox_visible(self, visible: bool) -> None:
        self.data["toolbox_visible"] = visible
        self.save()

    def get_resolution_mode(self) -> str:
        return self.data.get("resolution_mode", "auto")

    def set_resolution_mode(self, mode: str) -> None:
        if mode in ("auto", "high", "low"):
            self.data["resolution_mode"] = mode
            self.save()

    def get_theme(self) -> str:
        return self.data.get("theme", "dark")

    def set_theme(self, name: str) -> None:
        self.data["theme"] = name
        self.save()

    def get_depth_viewer(self) -> dict:
        base = dict(self.defaults.get("depth_viewer", {}))
        base.update(self.data.get("depth_viewer", {}))
        return base

    def set_depth_viewer(self, updates: dict):
        cur = self.get_depth_viewer()
        cur.update(updates)
        self.data["depth_viewer"] = cur
        self.save()

    def get_3d_point_size(self) -> float:
        return float(self.data.get("viewer3d", {}).get("point_size", 3.0))

    def set_3d_point_size(self, size: float):
        if "viewer3d" not in self.data:
            self.data["viewer3d"] = {}
        self.data["viewer3d"]["point_size"] = float(size)
        self.save()

    def get_tool_splitter_default(self, class_name):
        return self._tool_splitter_defaults.get(class_name, [25, 75])

    def get_tool_splitter(self, class_name):
        return self.data.get("tool_splitters", {}).get(class_name)

    def set_tool_splitter(self, class_name, sizes):
        self.data.setdefault("tool_splitters", {})[class_name] = sizes
        self.save()

    def get_web_port(self) -> int:
        return int(self.data.get("web_port", 8000))

    def set_web_port(self, port: int):
        self.data["web_port"] = int(port)
        self.save()

    def reset_view_defaults(self):
        self.data["tool_splitters"] = {}
        self.data["toolbox_visible"] = self.defaults["toolbox_visible"]
        self.data["show_overlays"] = self.defaults["show_overlays"]
        self.data["depth_viewer"] = dict(self.defaults["depth_viewer"])
        self.data["viewer3d"] = dict(self.defaults["viewer3d"])
        self.data["resolution_mode"] = self.defaults["resolution_mode"]
        self.data["window"] = dict(self.defaults["window"])
        self.save()
