import json
import zipfile
import io
import os
import numpy as np

class JobManager:
    def __init__(self):
        self.assets_to_save = {} # {asset_id: numpy_array}
        self.loaded_assets = {}  # {asset_id: numpy_array}
        self._current_job_path = None

    def add_asset(self, array: np.ndarray) -> str:
        """Add a numpy array as an asset and return its generated ID."""
        import uuid
        asset_id = f"asset_{uuid.uuid4().hex}.npy"
        self.assets_to_save[asset_id] = array
        return asset_id

    def get_asset(self, asset_id: str) -> np.ndarray:
        """Retrieve an asset by its ID."""
        return self.loaded_assets.get(asset_id)

    def save_job(self, workflow, filepath: str):
        """Serialize workflow and assets into an uncompressed ZIP file (.cvj)."""
        self.assets_to_save.clear()
        
        # 1. Serialize topology (this process populates self.assets_to_save)
        topology = workflow.serialize(job_manager=self)
        
        # 2. Write to ZIP
        with zipfile.ZipFile(filepath, 'w', zipfile.ZIP_STORED) as zf:
            # Write JSON
            zf.writestr("workflow.json", json.dumps(topology, indent=2))
            
            # Write Assets
            for asset_id, array in self.assets_to_save.items():
                if array is not None:
                    # Save numpy array to a bytes buffer
                    buf = io.BytesIO()
                    np.save(buf, array)
                    # Write buffer to zip
                    zf.writestr(f"assets/{asset_id}", buf.getvalue())
                    
        self._current_job_path = filepath
        self.assets_to_save.clear()

    def load_job(self, workflow, filepath: str):
        """Deserialize workflow and assets from a ZIP file (.cvj)."""
        self.loaded_assets.clear()
        
        with zipfile.ZipFile(filepath, 'r') as zf:
            # 1. Load Assets into RAM
            asset_names = [n for n in zf.namelist() if n.startswith("assets/") and n.endswith(".npy")]
            for name in asset_names:
                asset_id = name.split("/")[-1]
                with zf.open(name) as f:
                    buf = io.BytesIO(f.read())
                    array = np.load(buf)
                    self.loaded_assets[asset_id] = array
                    
            # 2. Load JSON Topology
            with zf.open("workflow.json") as f:
                topology = json.loads(f.read().decode('utf-8'))
                
            # 3. Reconstruct Workflow
            workflow.deserialize(topology, job_manager=self)
            
        self._current_job_path = filepath
        
    def get_current_job_path(self):
        return self._current_job_path
