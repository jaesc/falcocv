"""Regression: renaming a tool must not rebuild the tree from itemChanged.

Rebuilding the tree while Qt is committing the inline editor deletes the item
being edited and used to abort the process (Qt ``qFatal``).
"""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytestmark = pytest.mark.ui

from falcocv.core.workflow import Workflow
from falcocv.tools.image_loader import ImageLoader


@pytest.fixture(scope="module")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_rename_updates_tool_without_rebuilding_tree(qapp):
    from falcocv.ui.widgets.workflow_tree import WorkflowTree

    workflow = Workflow()
    tool = ImageLoader("Image Loader_0")
    workflow.add_tool(tool)

    tree = WorkflowTree()
    tree.set_workflow(workflow)
    item = tree.topLevelItem(0)
    try:
        assert tree.get_tool_from_item(item) is tool

        item.setText(0, "[LOAD] Renamed Loader")

        assert tool.name == "Renamed Loader"
        assert tree.topLevelItem(0) is item
    finally:
        tree.deleteLater()
