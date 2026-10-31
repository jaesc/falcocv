"""Structural regressions for the workflow tree and the Block config editor.

The Block config embeds a second WorkflowEditor.  It must render the existing
sub-workflow on construction and must never lose tools when the view is
refreshed, reordered or edited while the config is open.
"""

import pytest
from PyQt6.QtCore import QMimeData, QPointF, Qt
from PyQt6.QtGui import QDropEvent

pytestmark = pytest.mark.ui

from falcocv.core.workflow import Workflow
from falcocv.tools.block.config import BlockConfig
from falcocv.tools.block.logic import Block
from falcocv.tools.image_loader.logic import ImageLoader
from falcocv.ui.widgets.tool_config_dialog import ToolConfigDialog
from falcocv.ui.widgets.workflow_editor import WorkflowEditor
from falcocv.ui.widgets.workflow_tree import _TOOL_MIME


def _workflow(*names):
    workflow = Workflow()
    for name in names:
        workflow.add_tool(ImageLoader(name))
    return workflow


def _tree_tools(editor):
    return [
        editor.tree.get_tool_from_item(editor.tree.topLevelItem(i))
        for i in range(editor.tree.topLevelItemCount())
    ]


class _DropEvent(QDropEvent):
    """A drop event with an explicit source (Qt sets it only for real drags)."""

    def __init__(self, source, mime):
        super().__init__(
            QPointF(5, 5),
            Qt.DropAction.CopyAction | Qt.DropAction.MoveAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self._source = source

    def source(self):
        return self._source


def test_editor_renders_an_existing_workflow(qtbot):
    workflow = _workflow("a", "b", "c")
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)

    assert editor.tree.topLevelItemCount() == 3
    assert _tree_tools(editor) == workflow.tools


def test_block_config_reopen_keeps_its_tools(qtbot):
    block = Block("Block")
    for name in ("a", "b", "c"):
        block.workflow.add_tool(ImageLoader(name))

    dialog = ToolConfigDialog(block)
    qtbot.addWidget(dialog)
    config = dialog._config_widget
    assert isinstance(config, BlockConfig)
    assert config.editor.tree.topLevelItemCount() == 5  # 2 headers + 3 tools

    dialog.close()
    assert block._config_window is None

    # Reopening used to construct an empty tree even though the workflow still
    # contained every tool.
    reopened = ToolConfigDialog(block)
    qtbot.addWidget(reopened)
    assert len(block.workflow.tools) == 3
    assert reopened._config_widget.editor.tree.topLevelItemCount() == 5
    reopened.close()


def test_on_reorder_never_deletes_tools_missing_from_the_view(qtbot):
    workflow = _workflow("a", "b", "c")
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)

    editor.tree.takeTopLevelItem(0)
    assert editor.tree.topLevelItemCount() == 2

    editor.on_reorder()

    assert [t.name for t in workflow.tools] == ["a", "b", "c"]
    assert editor.tree.topLevelItemCount() == 3
    assert _tree_tools(editor) == workflow.tools


def test_render_error_keeps_the_tool_visible(qtbot):
    workflow = _workflow("good", "bad")
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)

    bad = workflow.tools[1]

    def boom(_name):
        raise RuntimeError("broken output")

    bad.get_output = boom
    editor.tree.set_workflow(workflow)

    assert editor.tree.topLevelItemCount() == 2
    bad_item = next(
        editor.tree.topLevelItem(i)
        for i in range(editor.tree.topLevelItemCount())
        if editor.tree.get_tool_from_item(editor.tree.topLevelItem(i)) is bad
    )
    assert bad_item.text(1) == "Render error"
    assert _tree_tools(editor) == workflow.tools


def test_set_workflow_is_atomic(qtbot, monkeypatch):
    workflow = _workflow("a", "b")
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)
    old_tools = list(workflow.tools)

    def boom(_workflow):
        raise RuntimeError("cannot build")

    monkeypatch.setattr(editor.tree, "_build_items", boom)
    editor.tree.set_workflow(_workflow("x"))

    # The old tree and the old workflow reference are still intact.
    assert editor.tree.topLevelItemCount() == 2
    assert _tree_tools(editor) == old_tools
    assert editor.tree._workflow is workflow


def test_cross_tree_move_keeps_every_other_tool(qtbot):
    source_wf = _workflow("a", "b", "c")
    target_wf = _workflow("d")
    source = WorkflowEditor(source_wf)
    target = WorkflowEditor(target_wf)
    qtbot.addWidget(source)
    qtbot.addWidget(target)

    moved = source_wf.tools[2]
    assert target.tree._move_tool_between(source.tree, moved.uuid, 0)

    assert [t.name for t in source_wf.tools] == ["a", "b"]
    assert [t.name for t in target_wf.tools] == ["c", "d"]
    assert _tree_tools(source) == source_wf.tools
    assert _tree_tools(target) == target_wf.tools


def test_drop_event_accepts_as_copy(qtbot):
    source_wf = _workflow("a", "b")
    target_wf = _workflow("d")
    source = WorkflowEditor(source_wf)
    target = WorkflowEditor(target_wf)
    qtbot.addWidget(source)
    qtbot.addWidget(target)
    source.show()
    target.show()
    qtbot.wait(20)

    mime = QMimeData()
    mime.setData(_TOOL_MIME, source_wf.tools[1].uuid.encode("utf-8"))
    event = _DropEvent(source.tree, mime)
    target.tree.dropEvent(event)

    assert event.isAccepted()
    assert event.dropAction() == Qt.DropAction.CopyAction
    assert [t.name for t in source_wf.tools] == ["a"]
    assert [t.name for t in target_wf.tools] == ["b", "d"]


def test_drop_event_reorders_within_the_same_tree(qtbot):
    workflow = _workflow("a", "b", "c")
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)
    editor.show()
    qtbot.wait(20)

    mime = QMimeData()
    mime.setData(_TOOL_MIME, workflow.tools[2].uuid.encode("utf-8"))
    event = _DropEvent(editor.tree, mime)
    editor.tree.dropEvent(event)

    assert event.isAccepted()
    assert event.dropAction() == Qt.DropAction.CopyAction
    assert [t.name for t in workflow.tools] == ["c", "a", "b"]
    assert editor.tree.topLevelItemCount() == 3


def test_move_a_tool_into_a_block_workflow(qtbot):
    main_wf = Workflow()
    block = Block("Block")
    moved = ImageLoader("moved")
    main_wf.add_tool(block)
    main_wf.add_tool(moved)
    main = WorkflowEditor(main_wf)
    inner = WorkflowEditor(block.workflow)
    qtbot.addWidget(main)
    qtbot.addWidget(inner)

    assert inner.tree._move_tool_between(main.tree, moved.uuid, 0)
    assert main_wf.tools == [block]
    assert block.workflow.tools == [moved]
    assert inner.tree.topLevelItemCount() == 3  # 2 headers + tool


def test_moving_a_block_into_its_own_workflow_is_refused(qtbot):
    main_wf = Workflow()
    block = Block("Block")
    main_wf.add_tool(block)
    main = WorkflowEditor(main_wf)
    inner = WorkflowEditor(block.workflow)
    qtbot.addWidget(main)
    qtbot.addWidget(inner)

    assert not inner.tree._move_tool_between(main.tree, block.uuid, 0)
    assert main_wf.tools == [block]
    assert block.workflow.tools == []
