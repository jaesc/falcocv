"""Installation self-test used by CI and support.

Run with::

    falcocv self-test

It exercises the same import/startup paths as the real application without
opening a visible window: every tool is registered, every configuration
widget and the help data are imported, a synthetic workflow runs end-to-end and
a job is saved/loaded.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import threading
import traceback

# Must be set before PyQt6 is imported.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# Do not read or write the user's real configuration while self-testing.
_SELF_TEST_CONFIG_DIR = tempfile.mkdtemp(prefix="falcocv-self-test-config-")
os.environ["FALCOCV_CONFIG_DIR"] = _SELF_TEST_CONFIG_DIR


def _log_unhandled(exc_type, exc_value, exc_tb):
    try:
        print("=== UNHANDLED PYTHON EXCEPTION ===", file=sys.stderr, flush=True)
        traceback.print_exception(exc_type, exc_value, exc_tb)
        sys.stderr.flush()
    except Exception:
        pass


sys.excepthook = _log_unhandled


def _thread_excepthook(args):
    _log_unhandled(args.exc_type, args.exc_value, args.exc_traceback)


threading.excepthook = _thread_excepthook


def _fail(message: str) -> None:
    print(f"[FAIL] {message}", flush=True)


def _pass(message: str) -> None:
    print(f"[PASS] {message}", flush=True)


def _skip(message: str) -> None:
    print(f"[SKIP] {message}", flush=True)


def _check_configs() -> bool:
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])

    import falcocv.tools  # noqa: F401  (populate registry)
    from falcocv.core.tool_registry import create_tool, get_all_tools
    from falcocv.ui.tool_help import TOOL_HELP
    from falcocv.ui.widgets.help_dialog import ToolHelpDialog  # noqa: F401
    from falcocv.ui.widgets.tool_config_dialog import ToolConfigDialog

    ok = True
    for reg in get_all_tools():
        try:
            tool = create_tool(reg.tool_type)
            dialog = ToolConfigDialog(tool)
            dialog.show()
            app.processEvents()
            if dialog._config_widget is None:
                raise AssertionError("config widget is None")
            dialog.close()
            app.processEvents()
            if tool._config_window is not None:
                raise AssertionError("_config_window was not cleared")
            _pass(f"config: {reg.tool_type}")
        except Exception as exc:
            ok = False
            _fail(f"config: {reg.tool_type}: {type(exc).__name__}: {exc}")
            traceback.print_exc()

    if not TOOL_HELP:
        ok = False
        _fail("help data is empty (falcocv.ui.tool_help)")
    else:
        _pass(f"help data: {len(TOOL_HELP)} entries")

    return ok


def _check_pipeline() -> bool:
    import cv2
    import numpy as np

    from falcocv.core.job_manager import JobManager
    from falcocv.core.workflow import Workflow
    from falcocv.core.workflow_controller import WorkflowController
    from falcocv.tools.blob_detector.logic import BlobDetector
    from falcocv.tools.filters.logic import FiltersTool
    from falcocv.tools.image_loader.logic import ImageLoader

    try:
        with tempfile.TemporaryDirectory(prefix="falcocv-self-test-") as tmp:
            image_path = os.path.join(tmp, "blobs.png")
            image = np.zeros((240, 320, 3), dtype=np.uint8)
            for i in range(3):
                x = 30 + i * 90
                cv2.rectangle(image, (x, 70), (x + 50, 140), (255, 255, 255), -1)
            if not cv2.imwrite(image_path, image):
                raise RuntimeError("could not write the synthetic test image")

            workflow = Workflow()
            loader = ImageLoader("Loader")
            filters = FiltersTool("Filters")
            blob = BlobDetector("Blob")
            loader.params.update({"mode": "File", "file_path": image_path})
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
            for tool in (loader, filters, blob):
                workflow.add_tool(tool)
            workflow.connect(loader, "output image", filters, "input image")
            workflow.connect(filters, "output image", blob, "input image")

            controller = WorkflowController(workflow)
            controller.set_scheduler(lambda fn: fn())
            controller.run()
            if workflow.last_error:
                raise RuntimeError(workflow.last_error)
            if blob.outputs["count"] != 3:
                raise AssertionError(f"expected 3 blobs, got {blob.outputs['count']}")
            _pass("pipeline: ImageLoader -> Filters -> BlobDetector (3 blobs)")

            job_path = os.path.join(tmp, "self-test.cvj")
            JobManager().save_job(workflow, job_path)
            loaded = Workflow()
            JobManager().load_job(loaded, job_path)
            if len(loaded.tools) != 3:
                raise AssertionError("job did not round-trip its tools")
            _pass("job: save/load round-trip")
        return True
    except Exception as exc:
        _fail(f"pipeline: {type(exc).__name__}: {exc}")
        traceback.print_exc()
        return False


def _check_models() -> None:
    try:
        from falcocv.core.models import missing_models

        missing = missing_models()
    except Exception as exc:
        _skip(f"models: could not check ({type(exc).__name__}: {exc})")
        return
    if missing:
        _skip(
            "models: {} missing; the application will download them on first "
            "run".format(", ".join(missing))
        )
    else:
        _pass("models: all present")


def main(argv=None) -> int:
    from falcocv.version import __version__

    print(f"FalcoCV self-test {__version__}")
    print(f"module: {sys.modules.get('falcocv').__file__ if 'falcocv' in sys.modules else '?'}")

    ok = True
    try:
        import falcocv  # noqa: F401
        _pass(f"import: {falcocv.__file__}")
    except Exception as exc:
        _fail(f"import: {type(exc).__name__}: {exc}")
        return 1

    ok = _check_configs() and ok
    ok = _check_pipeline() and ok
    _check_models()

    print("SELF-TEST:", "PASS" if ok else "FAIL")
    shutil.rmtree(_SELF_TEST_CONFIG_DIR, ignore_errors=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
