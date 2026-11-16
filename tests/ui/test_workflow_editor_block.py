"""WorkflowEditor: multi-selection and Generate Block."""

import pytest
from PyQt6.QtWidgets import QAbstractItemView, QDialog, QMessageBox

import falcocv.ui.widgets.generate_block_dialog as dialog_module
from falcocv.core.port_types import IMAGE, RESULTS
from falcocv.core.tool_base import BaseTool
from falcocv.core.workflow import Workflow
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.block.logic import Block
from falcocv.tools.filters.logic import FiltersTool
from falcocv.tools.image_loader.logic import ImageLoader
from falcocv.ui.widgets.workflow_editor import WorkflowEditor


pytestmark = pytest.mark.ui


class _Sink(BaseTool):
    input_port_types = {"results": RESULTS, "input image": IMAGE}
    output_port_types = {}

    def __init__(self, name):
        super().__init__(name)
        self.inputs = {"results": None, "input image": None}
        self.outputs = {}

    def run(self, advance=False):
        pass


def _workflow():
    loader = ImageLoader("loader")
    filters = FiltersTool("filters")
    blob = BlobDetector("blob")
    sink = _Sink("sink")
    workflow = Workflow()
    for tool in (loader, filters, blob, sink):
        workflow.add_tool(tool)
    workflow.connect(loader, "output image", filters, "input image")
    workflow.connect(filters, "output image", blob, "input image")
    workflow.connect(blob, "results", sink, "results")
    return workflow, loader, filters, blob, sink


def _select(editor, *tools):
    editor.tree.clearSelection()
    for index in range(editor.tree.topLevelItemCount()):
        item = editor.tree.topLevelItem(index)
        if editor.tree.get_tool_from_item(item) in tools:
            item.setSelected(True)


def _fake_dialog(monkeypatch, keep_originals=True, name="Group"):
    class FakeDialog:
        def __init__(self, tool_names, default_name="Block", parent=None):
            self.tool_names = list(tool_names)
            self._name = name
            self._keep = keep_originals

        def exec(self):
            return QDialog.DialogCode.Accepted

        def block_name(self):
            return self._name

        @property
        def keep_originals(self):
            return self._keep

    monkeypatch.setattr(dialog_module, "GenerateBlockDialog", FakeDialog)


def test_tree_uses_extended_selection_and_workflow_order(qtbot):
    workflow, loader, filters, blob, sink = _workflow()
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)

    assert (
        editor.tree.selectionMode()
        == QAbstractItemView.SelectionMode.ExtendedSelection
    )

    _select(editor, blob, loader)
    assert editor.tree.selected_tools() == [loader, blob]


def test_context_menu_offers_generate_block_for_multi_selection(
    qtbot, monkeypatch
):
    from PyQt6.QtWidgets import QMenu

    workflow, loader, filters, blob, sink = _workflow()
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)
    _select(editor, filters, blob)

    seen = []

    def fake_exec(menu, *args, **kwargs):
        seen.append([action.text() for action in menu.actions()])

    monkeypatch.setattr(QMenu, "exec", fake_exec)
    item = editor.tree.topLevelItem(1)
    editor.tree.show_context_menu(editor.tree.visualItemRect(item).center())

    assert seen
    assert "Generate Block from 2 tools..." in seen[0]
    assert "Delete 2 tools" in seen[0]


def test_delete_key_emits_all_selected_tools(qtbot):
    from PyQt6.QtCore import QEvent, Qt
    from PyQt6.QtGui import QKeyEvent

    workflow, loader, filters, blob, sink = _workflow()
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)
    _select(editor, filters, blob)

    seen = []
    editor.tree.tools_deleted.connect(lambda tools: seen.append(list(tools)))

    event = QKeyEvent(
        QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier
    )
    editor.tree.keyPressEvent(event)

    assert seen == [[filters, blob]]


def test_generate_block_dialog_defaults_to_copy(qtbot):
    from falcocv.ui.widgets.generate_block_dialog import GenerateBlockDialog

    dialog = GenerateBlockDialog(["a", "b"], "Group")
    qtbot.addWidget(dialog)

    assert dialog.keep_originals is True
    assert dialog.copy_radio.isChecked() is True
    assert dialog.block_name() == "Group"

    dialog.move_radio.setChecked(True)
    assert dialog.keep_originals is False
    assert dialog.copy_note.isEnabled() is False
    assert dialog.move_note.isEnabled() is True


def test_generate_block_copy_keeps_originals(qtbot, monkeypatch):
    workflow, loader, filters, blob, sink = _workflow()
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)
    _fake_dialog(monkeypatch, keep_originals=True)

    editor.on_generate_block([filters, blob])

    assert len(workflow.tools) == 5
    block = workflow.tools[3]
    assert workflow.tools == [loader, filters, blob, block, sink]
    assert isinstance(block, Block)
    assert block.name == "Group"
    assert block.workflow.tools[0] is not filters
    assert block.workflow.tools[1] is not blob
    assert workflow.connections[0] == (
        loader, "output image", filters, "input image"
    )


def test_generate_block_move_removes_originals(qtbot, monkeypatch):
    workflow, loader, filters, blob, sink = _workflow()
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)
    _fake_dialog(monkeypatch, keep_originals=False)

    editor.on_generate_block([filters, blob])

    block = workflow.tools[1]
    assert workflow.tools == [loader, block, sink]
    assert isinstance(block, Block)
    assert block.workflow.tools == [filters, blob]
    assert filters not in workflow.tools
    assert blob not in workflow.tools


def test_generate_block_rejects_non_contiguous_selection(qtbot, monkeypatch):
    workflow, loader, filters, blob, sink = _workflow()
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)
    seen = []
    monkeypatch.setattr(
        QMessageBox,
        "information",
        lambda *args, **kwargs: seen.append(args),
    )

    editor.on_generate_block([loader, blob])

    assert seen
    assert not any(isinstance(tool, Block) for tool in workflow.tools)


def test_delete_selected_tools_in_one_step(qtbot):
    workflow, loader, filters, blob, sink = _workflow()
    editor = WorkflowEditor(workflow)
    qtbot.addWidget(editor)

    editor.on_deletions([filters, blob])

    assert workflow.tools == [loader, sink]
    assert workflow.connections == []
