#!/usr/bin/env python3
"""Generate the screenshots used by the README.

The script intentionally uses the real Qt widgets instead of drawing mock
windows: the Vision Editor is loaded with the demo job, the HMI screens are
rendered by HMI Designer, and the 3D image is a framebuffer capture of the
point-cloud viewer.  Run it from the repository root:

    python scripts/capture_readme_screenshots.py

Outputs are written to ``docs/images``.  Existing screenshots can be replaced
by hand at any time; this script is only the reproducible starting point.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from PyQt6.QtCore import QPointF, QRectF, Qt
from PyQt6.QtGui import (
    QColor,
    QFont,
    QGuiApplication,
    QImage,
    QPainter,
    QPen,
)
from PyQt6.QtWidgets import QApplication


IMAGE_WIDTH = 1600


def _wait(app: QApplication, seconds: float = 0.4) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def _save_image(image: QImage, path: Path, width: int = IMAGE_WIDTH) -> None:
    """Save a PNG, downscaling high-DPI screen grabs to README width."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if image.width() > width:
        height = round(image.height() * width / image.width())
        image = image.scaled(
            width,
            height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
    if not image.save(str(path), "PNG"):
        raise RuntimeError(f"could not save {path}")


def _grab_window(app: QApplication, widget) -> QImage:
    """Grab a top-level Qt window without depending on its desktop position."""

    widget.show()
    widget.raise_()
    widget.activateWindow()
    _wait(app, 0.8)
    screen = QGuiApplication.primaryScreen()
    if screen is None:
        raise RuntimeError("no Qt screen available")
    pixmap = screen.grabWindow(widget.winId())
    if pixmap.isNull():
        pixmap = widget.grab()
    image = pixmap.toImage()
    if image.isNull():
        raise RuntimeError(f"empty screenshot for {widget!r}")
    return image


def _prepare_main_window():
    """Populate the globals that falcocv.main.main() normally injects."""

    import falcocv.tools  # noqa: F401  (registers the tools)
    import falcocv.main as main_module
    from falcocv.core.config import ConfigManager
    from falcocv.core.job_manager import JobManager
    from falcocv.core.workflow import Workflow
    from falcocv.core.workflow_controller import ControllerState
    from falcocv.ui.widgets.job_config_dialog import JobConfigDialog
    from falcocv.ui.widgets.toolbox import Toolbox
    from falcocv.ui.widgets.workflow_editor import WorkflowEditor

    main_module.ConfigManager = ConfigManager
    main_module.Workflow = Workflow
    main_module.ControllerState = ControllerState
    main_module.JobManager = JobManager
    main_module.WorkflowEditor = WorkflowEditor
    main_module.Toolbox = Toolbox
    main_module.JobConfigDialog = JobConfigDialog

    # The editor screenshot does not need the background web server.  Keeping it
    # off also makes the script safe to run while `falcocv-server` is running.
    main_module.MainWindow._web_start = lambda self: None
    return main_module


def capture_vision_editor(app: QApplication, output_dir: Path) -> Path:
    main_module = _prepare_main_window()
    window = main_module.MainWindow()
    window.resize(1600, 1000)
    window.showNormal()
    window.open_job_from_path(str((ROOT / "examples" / "demo_inspection.cvj").resolve()))
    window.editor.controller.run(advance=False)
    _wait(app, 1.8)
    path = output_dir / "vision-editor.png"
    _save_image(_grab_window(app, window), path)
    window.close()
    _wait(app, 0.2)
    return path


def _synthetic_point_cloud() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(12)
    count = 260_000
    x = rng.uniform(-80.0, 80.0, count)
    y = rng.uniform(-40.0, 40.0, count)
    mask = np.ones(count, dtype=bool)
    for cx in (-24.0, 24.0):
        mask &= (x - cx) ** 2 + (y + 1.0) ** 2 > 11.0**2
    x = x[mask]
    y = y[mask]
    z = 0.8 * np.sin(x / 25.0) * np.cos(y / 18.0) + rng.normal(0.0, 0.12, len(x))
    points = np.column_stack((x, y, z)).astype(np.float32)
    scaled = (z - z.min()) / (z.max() - z.min() + 1e-6)
    colors = np.column_stack(
        (40 + 180 * scaled, 140 + 80 * (1.0 - scaled), 220 - 80 * scaled)
    ).astype(np.uint8)
    return (
        points,
        colors,
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        np.array([160.0, 80.0, 8.0], dtype=np.float32),
    )


def capture_3d(app: QApplication, output_dir: Path) -> Path:
    from falcocv.ui.widgets.viewer3d import Viewer3D

    points, colors, center, size = _synthetic_point_cloud()
    widget = Viewer3D()
    widget.resize(1400, 900)
    widget.show()
    widget.set_data(points, colors, bbox_center=center, bbox_size=size)
    widget.camera.distance = 175.0
    widget.set_point_size(2.0)
    widget.set_roi(center, np.array([120.0, 60.0, 5.0], dtype=np.float32))
    widget.show_roi(True)
    _wait(app, 1.8)
    framebuffer = widget.grabFramebuffer()
    widget.close()
    _wait(app, 0.2)
    if framebuffer.isNull():
        raise RuntimeError("could not grab the Viewer3D framebuffer")

    image = QImage(1600, 1000, QImage.Format.Format_ARGB32)
    image.fill(QColor("#111318"))
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    painter.setPen(QColor("#e8edf2"))
    title = QFont("Segoe UI")
    title.setPixelSize(26)
    title.setWeight(QFont.Weight.DemiBold)
    painter.setFont(title)
    painter.drawText(QRectF(44, 24, 1100, 42), Qt.AlignmentFlag.AlignLeft, "3D Point Cloud  |  Plane Fit  |  ROI")
    painter.setPen(QColor("#8b96a3"))
    subtitle = QFont("Segoe UI")
    subtitle.setPixelSize(14)
    painter.setFont(subtitle)
    painter.drawText(
        QRectF(46, 62, 1200, 26),
        Qt.AlignmentFlag.AlignLeft,
        "Synthetic bracket cloud  |  world coordinates  |  Z-up  |  240k points",
    )

    target = QRectF(44, 104, 1512, 824)
    scaled = framebuffer.scaled(
        int(target.width()),
        int(target.height()),
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )
    # QOpenGLWidget framebuffers report a 2x device-pixel ratio on high-DPI
    # screens; draw them as normal pixels in the composite image.
    scaled.setDevicePixelRatio(1.0)
    x = target.x() + (target.width() - scaled.width()) / 2
    y = target.y() + (target.height() - scaled.height()) / 2
    painter.drawImage(QPointF(x, y), scaled)
    painter.setPen(QPen(QColor("#2a313a"), 1))
    painter.drawRoundedRect(QRectF(x - 1, y - 1, scaled.width() + 2, scaled.height() + 2), 7, 7)

    painter.setPen(QColor("#8b96a3"))
    footer = QFont("Segoe UI")
    footer.setPixelSize(13)
    painter.setFont(footer)
    painter.drawText(
        QRectF(46, 950, 1510, 24),
        Qt.AlignmentFlag.AlignLeft,
        "Drag to orbit  |  wheel to zoom  |  right-drag to pan  |  3D ROI shown in world coordinates",
    )
    painter.end()
    path = output_dir / "vision-3d.png"
    _save_image(image, path)
    return path


def _load_demo_project():
    from falcocv.core.hmi import HMIProject

    return HMIProject.load(str(ROOT / "examples" / "demo_dashboard.hmi"))


def _select_tab(dialog, title: str) -> None:
    if dialog._tabs is None:
        return
    for index in range(dialog._tabs.count()):
        if dialog._tabs.tabText(index).startswith(title):
            dialog._tabs.setCurrentIndex(index)
            return
    raise RuntimeError(f"HMI Designer tab {title!r} not found")


def capture_hmi_designer(app: QApplication, output_dir: Path) -> Path:
    from falcocv.ui.widgets.hmi_designer_dialog import HMIDesignerDialog

    project = _load_demo_project()
    dialog = HMIDesignerDialog(project)
    dialog.resize(1600, 1000)
    dialog.showNormal()
    _select_tab(dialog, "HMI Layout")
    _wait(app, 1.0)
    path = output_dir / "hmi-designer.png"
    _save_image(_grab_window(app, dialog), path)
    dialog._modified = False
    dialog.close()
    _wait(app, 0.2)
    return path


def capture_hmi_servers(app: QApplication, output_dir: Path) -> Path:
    from falcocv.ui.widgets.hmi_designer_dialog import HMIDesignerDialog

    project = _load_demo_project()
    project.servers_config = {
        "version": 1,
        "servers": [
            {
                "id": "modbus_plc",
                "type": "modbus_tcp_server",
                "enabled": False,
                "host": "0.0.0.0",
                "port": 502,
                "unit_id": 1,
                "word_order": "big",
                "mappings": [
                    {"address": "HOLDING:40001", "tag": "Line.Speed", "access": "rw"},
                    {"address": "COIL:00001", "tag": "Line.Start", "access": "rw"},
                ],
            },
            {
                "id": "opcua_line",
                "type": "opcua_server",
                "enabled": False,
                "endpoint": "opc.tcp://0.0.0.0:4840/falcocv",
                "namespace": "urn:falcocv",
                "mappings": [
                    {"address": "ns=2;s=Line.Speed", "tag": "Line.Speed", "access": "rw"},
                    {"address": "ns=2;s=Line.Start", "tag": "Line.Start", "access": "rw"},
                ],
            },
        ],
    }
    dialog = HMIDesignerDialog(project)
    dialog.resize(1600, 1000)
    dialog.showNormal()
    _select_tab(dialog, "Servers")
    _wait(app, 1.0)
    path = output_dir / "hmi-servers.png"
    _save_image(_grab_window(app, dialog), path)
    dialog._modified = False
    dialog.close()
    _wait(app, 0.2)
    return path


def _draw_preview_result_image(image: QImage, project) -> None:
    """Place the demo result in the preview's ImageViewer cell.

    HMIPreviewWidget deliberately draws a neutral grid for image viewers.  For
    the README dashboard, put the real demo result in that cell so the capture
    shows the relationship between the workflow and the HMI.
    """

    result_path = ROOT / "examples" / "result_ok.jpg"
    if not result_path.is_file():
        return
    result = QImage(str(result_path))
    if result.isNull():
        return

    layout = project.layout
    rows = max(1, int(layout.get("rows", 20)))
    cols = max(1, int(layout.get("cols", 30)))
    cell_w = image.width() / cols
    cell_h = image.height() / rows
    for widget in layout.get("widgets", []):
        if widget.get("type") != "ImageViewer":
            continue
        x, y, w, h = (
            widget.get("x", 0),
            widget.get("y", 0),
            widget.get("w", 4),
            widget.get("h", 3),
        )
        # Match HMIPreviewWidget: outer cell inset 8, image inset another 4.
        target = QRectF(
            x * cell_w + 12,
            y * cell_h + 12,
            max(1, w * cell_w - 24),
            max(1, h * cell_h - 24),
        )
        scaled = result.scaled(
            int(target.width()),
            int(target.height()),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter = QPainter(image)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        painter.setBrush(QColor("#0b0d10"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(target.adjusted(-2, -2, 2, 2), 4, 4)
        draw_x = target.x() + (target.width() - scaled.width()) / 2
        draw_y = target.y() + (target.height() - scaled.height()) / 2
        painter.drawImage(QPointF(draw_x, draw_y), scaled)
        painter.end()
        return


def _capture_hmi_preview(app: QApplication, output_dir: Path, name: str = "hmi-preview.png") -> Path:
    from falcocv.ui.widgets.hmi_preview import HMIPreviewWidget

    project = _load_demo_project()
    # These are design-time sample states; the live worker replaces them when
    # the project is deployed.
    for widget in project.layout.get("widgets", []):
        if widget.get("type") == "StatusIndicator":
            widget["passed"] = True
        if widget.get("type") == "MeasureLabel":
            widget["status"] = "pass"
    widget = HMIPreviewWidget(project)
    widget.resize(1600, 1000)
    widget.show()
    _wait(app, 1.0)
    image = widget.grab().toImage()
    image.setDevicePixelRatio(1.0)
    _draw_preview_result_image(image, project)
    path = output_dir / name
    _save_image(image, path)
    widget.close()
    _wait(app, 0.2)
    return path


def capture_hmi_dashboard(app: QApplication, output_dir: Path) -> Path:
    """Render the demo dashboard with representative runtime values.

    HMIPreviewWidget is the same renderer used by HMI Designer's Preview tab.
    It deliberately does not spin up a worker, which keeps README generation
    deterministic; replace the resulting file with a live browser screenshot
    when a deployed demo worker is available.
    """

    return _capture_hmi_preview(app, output_dir, "hmi-dashboard.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "docs" / "images",
        help="directory for README images (default: docs/images)",
    )
    args = parser.parse_args()

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from falcocv.ui.style import apply_qt_theme

    apply_qt_theme(app)

    outputs: list[Path] = []
    outputs.append(capture_vision_editor(app, args.output_dir))
    outputs.append(capture_3d(app, args.output_dir))
    outputs.append(capture_hmi_designer(app, args.output_dir))
    outputs.append(capture_hmi_servers(app, args.output_dir))
    outputs.append(capture_hmi_dashboard(app, args.output_dir))

    for path in outputs:
        print(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
