"""Helpers for headless workflow integration tests."""

import cv2
import numpy as np

from falcocv.core.workflow import Workflow
from falcocv.core.workflow_controller import WorkflowController


def make_blob_image(path, count=3):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for i in range(count):
        x = 30 + i * 90
        cv2.rectangle(image, (x, 70), (x + 50, 140), (255, 255, 255), -1)
    assert cv2.imwrite(str(path), image)
    return str(path)


def make_depth_plane(path, width=96, height=64):
    import tifffile

    rows = np.arange(height, dtype=np.float32)[:, None]
    cols = np.arange(width, dtype=np.float32)[None, :]
    depth = 100.0 + 0.5 * rows + 0.25 * cols
    tifffile.imwrite(str(path), depth.astype(np.float32))
    return str(path)


def run_sync(workflow: Workflow) -> WorkflowController:
    controller = WorkflowController(workflow)
    controller.set_scheduler(lambda fn: fn())
    controller.run(advance=False)
    return controller


def tool_by_type(workflow, type_name):
    return next(t for t in workflow.tools if type(t).__name__ == type_name)
