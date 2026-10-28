# Logic module for ImageLoader.

import os
import cv2
from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool

@register_tool(prefix="[LOAD]", default_name="Image Loader", group="Acquisition")
class ImageLoader(BaseTool):
    def __init__(self, name="Image Loader"):
        super().__init__(name)
        self.outputs = {"output image": None}
        self.params = {
            "mode": "File", # "File" or "Folder"
            "file_path": "",
            "folder_path": "",
            "sort_by": "Name", # "Name" or "Date"
            "current_index": 0,
            "delay": 0
        }
        self._image_files = []

    def _refresh_folder(self):
        self._image_files = []
        path = self.params["folder_path"]
        if not path or not os.path.exists(path):
            return

        valid_exts = {".png", ".jpg", ".jpeg", ".bmp"}
        for f in os.listdir(path):
            if os.path.splitext(f)[1].lower() in valid_exts:
                self._image_files.append(os.path.join(path, f))

        if self.params["sort_by"] == "Name":
            self._image_files.sort()
        elif self.params["sort_by"] == "Date":
            self._image_files.sort(key=os.path.getmtime)

        # Fix index if it went out of bounds
        if self.params["current_index"] >= len(self._image_files):
            self.params["current_index"] = 0

    def run(self, advance=False):
        # Industrial Rule: Always clear outputs at the start of run
        self.outputs["output image"] = None
        img_path = None
        
        if self.params["mode"] == "File":
            img_path = self.params["file_path"]
            if not img_path:
                raise FileNotFoundError("No file path specified.")
        elif self.params["mode"] == "Folder":
            if not self._image_files:
                self._refresh_folder()
                
            if self._image_files:
                if advance:
                    self.params["current_index"] += 1
                    if self.params["current_index"] >= len(self._image_files):
                        self.params["current_index"] = 0
                img_path = self._image_files[self.params["current_index"]]
            else:
                raise FileNotFoundError(f"No images found in folder: {self.params['folder_path']}")

        if img_path and os.path.exists(img_path):
            img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                raise RuntimeError(f"Failed to decode image: {img_path}")
            # PNG (and similar) may load as BGRA; vision pipeline expects 3-channel BGR.
            if img.ndim == 3 and img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            elif img.ndim == 3 and img.shape[2] == 2:
                # Grayscale + alpha (some PNGs): drop alpha
                gray = img[:, :, 0]
                img = gray
            
            # Wrap in VisionImage
            from falcocv.core.vision_image import VisionImage
            self.outputs["output image"] = VisionImage(img)
        else:
            raise FileNotFoundError(f"Image path does not exist: {img_path}")
            
        self.data_changed.emit()

    def serialize(self, job_manager=None) -> dict:
        data = super().serialize(job_manager)
        # Remove transient runtime state that shouldn't persist in the job
        data["params"] = {k: v for k, v in data["params"].items() if k != "current_index"}
        return data

    def deserialize(self, data: dict, job_manager=None):
        super().deserialize(data, job_manager)
        self.params["current_index"] = 0  # Reset to first image on load
        self._image_files = []
        if self.params.get("mode") == "Folder":
            self._refresh_folder()

    def get_config_widget(self):
        from .config import ImageLoaderConfig
        return ImageLoaderConfig(self)
