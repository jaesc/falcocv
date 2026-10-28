"""Install a built wheel into a throwaway venv and run ``falcocv self-test``.

Usage:
    python scripts/wheel_smoke.py dist/falcocv-*.whl
"""

from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import tempfile
import venv
from pathlib import Path


def _venv_python(root: Path) -> Path:
    if os.name == "nt":
        return root / "Scripts" / "python.exe"
    return root / "bin" / "python"


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    wheels = [Path(p).resolve() for p in (args or sorted(glob.glob("dist/*.whl")))]
    if not wheels:
        print("No wheel found. Build one with: python -m build --wheel")
        return 1
    wheel = wheels[-1]

    with tempfile.TemporaryDirectory(prefix="falcocv-wheel-") as tmp:
        root = Path(tmp)
        env_dir = root / "venv"
        print(f"Creating throwaway venv: {env_dir}")
        venv.EnvBuilder(with_pip=True, system_site_packages=True).create(env_dir)
        python = _venv_python(env_dir)

        print(f"Installing {wheel}")
        subprocess.check_call(
            [str(python), "-m", "pip", "install", "--no-deps", str(wheel)],
            cwd=root,
        )

        env = {
            **os.environ,
            "FALCOCV_CONFIG_DIR": str(root / "config"),
        }
        print("Running: falcocv self-test")
        result = subprocess.run(
            [str(python), "-m", "falcocv", "self-test"],
            cwd=root,
            env=env,
        )
        if result.returncode != 0:
            print(f"[FAIL] self-test returned {result.returncode}")
            return result.returncode

        # Headless job: create a small pipeline with the installed wheel and
        # execute it through the real CLI.
        make_job = root / "make_job.py"
        make_job.write_text(
            """
import os
import cv2
import numpy as np
from falcocv.core.job_manager import JobManager
from falcocv.core.workflow import Workflow
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.image_loader.logic import ImageLoader

root = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(root, "blobs.png")
image = np.zeros((240, 320, 3), dtype=np.uint8)
for i in range(3):
    x = 30 + i * 90
    cv2.rectangle(image, (x, 70), (x + 50, 140), (255, 255, 255), -1)
assert cv2.imwrite(image_path, image)

workflow = Workflow()
loader = ImageLoader("Loader")
blob = BlobDetector("Blob")
loader.params.update({"mode": "File", "file_path": image_path})
blob.params.update({
    "threshold_mode": "Fixed",
    "threshold_value": 128,
    "polarity": "Bright Blobs",
    "area_min": 50,
    "area_max": 100000,
    "show_contours": False,
    "show_centroids": False,
    "show_labels": False,
    "show_search_roi": False,
})
workflow.add_tool(loader)
workflow.add_tool(blob)
workflow.connect(loader, "output image", blob, "input image")
JobManager().save_job(workflow, os.path.join(root, "job.cvj"))
""",
            encoding="utf-8",
        )
        subprocess.check_call([str(python), str(make_job)], cwd=root, env=env)
        results_path = root / "results.json"
        headless = subprocess.run(
            [
                str(python),
                "-m",
                "falcocv",
                "run",
                str(root / "job.cvj"),
                "--output",
                str(results_path),
            ],
            cwd=root,
            env=env,
        )
        if headless.returncode != 0:
            print(f"[FAIL] headless run returned {headless.returncode}")
            return headless.returncode
        data = json.loads(results_path.read_text(encoding="utf-8"))
        if not data.get("ok") or data.get("tools", [{}])[0].get("name") != "Loader":
            print(f"[FAIL] unexpected headless report: {data}")
            return 1
        print("[PASS] wheel headless run")
    print("[PASS] wheel smoke test")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
