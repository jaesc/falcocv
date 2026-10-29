"""Render two annotated result images for the docs.

Usage:
    python examples/render_demo_result.py
"""

from __future__ import annotations

from pathlib import Path

import cv2

from falcocv.core.job_manager import JobManager
from falcocv.core.workflow import Workflow
from falcocv.core.workflow_controller import WorkflowController
from falcocv.tools.image_loader.logic import ImageLoader

ROOT = Path(__file__).resolve().parent
JOB = ROOT / "demo_inspection.cvj"

CASES = {
    "result_ok.jpg": "02_ok.jpg",
    "result_ng.jpg": "11_ng_missing_hole.jpg",
}


def main() -> int:
    for output, image in CASES.items():
        workflow = Workflow()
        JobManager().load_job(workflow, str(JOB))
        loader = next(t for t in workflow.tools if isinstance(t, ImageLoader))
        loader.params.update({"mode": "File", "file_path": str(ROOT / "images" / image)})
        controller = WorkflowController(workflow)
        controller.set_scheduler(lambda callback: callback())
        controller.run(advance=False)

        last = workflow.tools[-1]
        rendered = last.outputs["output image"].render_overlays()
        cv2.imwrite(str(ROOT / output), rendered, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        print(f"{image} -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
