"""Dirty tracking for edits made inside a Block config editor."""

import pytest

from falcocv.core.tool_base import BaseTool
from falcocv.tools.block.logic import Block

pytestmark = pytest.mark.ui


class _NoopTool(BaseTool):
    def __init__(self, name):
        super().__init__(name)
        self.inputs = {}
        self.outputs = {}

    def run(self, advance=False):
        pass


def _add_block(window, qtbot):
    window.editor.handle_add_tool("Block")
    qtbot.wait(20)
    tree = window.editor.tree
    item = tree.topLevelItem(tree.topLevelItemCount() - 1)
    block = tree.get_tool_from_item(item)
    assert isinstance(block, Block)
    # handle_add_tool() bypasses the tree signal that reconnects tool signals.
    window._connect_tool_signals()
    return block, item


def _open_block(window, qtbot, block, item):
    window.editor.tree.itemDoubleClicked.emit(item, 0)
    qtbot.waitUntil(lambda: block._config_window is not None, timeout=3000)
    return block._config_window._config_widget


def test_nested_parameter_edit_marks_job_modified(main_window, qtbot):
    window = main_window
    block, item = _add_block(window, qtbot)
    nested = _NoopTool("nested")
    block.workflow.add_tool(nested)
    _open_block(window, qtbot, block, item)
    window._modified = False
    qtbot.wait(20)

    nested.mark_dirty()
    qtbot.wait(20)

    assert window._modified is True


def test_nested_direct_data_changed_marks_job_modified(main_window, qtbot):
    window = main_window
    block, item = _add_block(window, qtbot)
    nested = _NoopTool("nested")
    block.workflow.add_tool(nested)
    _open_block(window, qtbot, block, item)
    window._modified = False
    qtbot.wait(20)

    # ROI editors and PortTable emit data_changed directly, without mark_dirty.
    nested.data_changed.emit()
    qtbot.wait(20)

    assert window._modified is True


def test_nested_connection_change_marks_job_modified(main_window, qtbot):
    window = main_window
    block, item = _add_block(window, qtbot)
    first = _NoopTool("first")
    second = _NoopTool("second")
    block.workflow.add_tool(first)
    block.workflow.add_tool(second)
    config = _open_block(window, qtbot, block, item)
    window._modified = False
    qtbot.wait(20)

    config.editor.tree.connection_changed.emit(
        first, "output image", second, "input image"
    )
    qtbot.wait(20)

    assert window._modified is True


def test_nested_run_does_not_mark_job_modified(main_window, qtbot):
    window = main_window
    block, item = _add_block(window, qtbot)
    block.workflow.add_tool(_NoopTool("nested"))
    config = _open_block(window, qtbot, block, item)
    window._modified = False
    qtbot.wait(20)

    with qtbot.waitSignal(config.editor.engine.workflow_finished, timeout=5000):
        config.editor.run_workflow(advance=False)
    qtbot.wait(50)

    assert window._modified is False


def test_block_config_close_with_dirty_nested_controller_marks_job(
    main_window, qtbot
):
    window = main_window
    block, item = _add_block(window, qtbot)
    config = _open_block(window, qtbot, block, item)
    dialog = block._config_window
    window._modified = False
    qtbot.wait(20)

    # No tool signal: only the close-time safety net can propagate this.
    config.editor.controller.mark_dirty()
    dialog.close()
    qtbot.wait(20)

    assert window._modified is True
