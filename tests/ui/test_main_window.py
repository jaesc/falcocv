"""Main window smoke tests: add, open config, run and persist."""

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.ui


def _synthetic_image(path):
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    for x in (40, 140, 240):
        cv2.rectangle(image, (x, 80), (x + 50, 140), (255, 255, 255), -1)
    assert cv2.imwrite(str(path), image)
    return path


def _tool_by_type(workflow, type_name):
    return next(t for t in workflow.tools if type(t).__name__ == type_name)


def test_add_tool_and_open_config_via_double_click(main_window, qtbot):
    window = main_window
    window.editor.handle_add_tool("ImageLoader")
    qtbot.wait(50)

    tree = window.editor.tree
    item = tree.topLevelItem(tree.topLevelItemCount() - 1)
    tool = tree.get_tool_from_item(item)
    assert tool is not None

    # Emit the view signal directly: the real mouse double-click does not
    # deliver reliably on the offscreen platform.
    tree.itemDoubleClicked.emit(item, 0)
    qtbot.waitUntil(lambda: tool._config_window is not None, timeout=3000)

    dialog = tool._config_window
    assert type(dialog._config_widget).__name__ == "ImageLoaderConfig"
    dialog.close()
    assert tool._config_window is None


def test_image_loader_filters_blob_pipeline(main_window, qtbot, tmp_path):
    window = main_window
    image_path = _synthetic_image(tmp_path / "blobs.png")

    window.editor.handle_add_tool("ImageLoader")
    window.editor.handle_add_tool("FiltersTool")
    window.editor.handle_add_tool("BlobDetector")

    loader = _tool_by_type(window.workflow, "ImageLoader")
    filters = _tool_by_type(window.workflow, "FiltersTool")
    blob = _tool_by_type(window.workflow, "BlobDetector")

    loader.params.update({"mode": "File", "file_path": str(image_path)})
    blob.params.update(
        {
            "threshold_mode": "Fixed",
            "threshold_value": 128,
            "polarity": "Bright Blobs",
            "area_min": 50,
            "area_max": 100000,
            "show_contours": False,
            "show_centroids": False,
            "show_labels": False,
            "show_search_roi": False,
        }
    )

    window.workflow.connect(loader, "output image", filters, "input image")
    window.workflow.connect(filters, "output image", blob, "input image")

    window.editor.controller.run(advance=False)
    qtbot.wait(50)

    assert window.workflow.last_error is None
    assert filters.outputs["output image"] is not None
    assert blob.outputs["count"] == 3


def test_block_config_restores_active_editor(main_window, qtbot):
    window = main_window
    from falcocv.ui.widgets.toolbox import Toolbox

    original_editor = Toolbox._active_editor
    window.editor.handle_add_tool("Block")
    qtbot.wait(50)

    tree = window.editor.tree
    item = tree.topLevelItem(tree.topLevelItemCount() - 1)
    tool = tree.get_tool_from_item(item)
    tree.itemDoubleClicked.emit(item, 0)
    qtbot.waitUntil(lambda: tool._config_window is not None, timeout=3000)

    dialog = tool._config_window
    assert Toolbox._active_editor is dialog._config_widget.editor

    dialog.close()
    assert Toolbox._active_editor is original_editor


def test_save_and_load_workflow(main_window, tmp_path):
    from falcocv.core.job_manager import JobManager
    from falcocv.core.workflow import Workflow

    window = main_window
    window.editor.handle_add_tool("ImageLoader")
    window.editor.handle_add_tool("FiltersTool")

    loader = _tool_by_type(window.workflow, "ImageLoader")
    filters = _tool_by_type(window.workflow, "FiltersTool")
    window.workflow.connect(loader, "output image", filters, "input image")

    path = tmp_path / "ui_job.cvj"
    window.job_manager.save_job(window.workflow, str(path))
    assert path.exists()

    loaded = Workflow()
    JobManager().load_job(loaded, str(path))
    assert [type(t).__name__ for t in loaded.tools] == ["ImageLoader", "FiltersTool"]
    assert len(loaded.connections) == 1
