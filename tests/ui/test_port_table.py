"""Shared port editor: adding, renaming and typing reach the tool."""

from falcocv.core.port_types import ANY, IMAGE, PLANE, RESULTS
from falcocv.tools.python_script.logic import PythonScriptTool
from falcocv.ui.widgets.port_table import PortTable


def test_port_table_adds_and_types(qtbot):
    tool = PythonScriptTool()
    table = PortTable(tool, "inputs")
    qtbot.addWidget(table)

    before = len(tool.inputs)
    table.add_port()
    assert len(tool.inputs) == before + 1

    name = table._names[-1]
    assert tool.get_input_port_type(name) == ANY

    row = table._names.index(name)
    combo = table.table.cellWidget(row, 1)
    combo.setCurrentIndex(combo.findData(PLANE))
    assert tool.get_input_port_type(name) == PLANE


def test_port_table_rename_keeps_the_kind(qtbot):
    tool = PythonScriptTool()
    table = PortTable(tool, "outputs")
    qtbot.addWidget(table)

    row = table._names.index("output image")
    item = table.table.item(row, 0)
    item.setText("frame")

    assert "frame" in tool.outputs
    assert "output image" not in tool.outputs
    assert tool.get_output_port_type("frame") == IMAGE


def test_port_table_offers_results_kind(qtbot):
    tool = PythonScriptTool()
    table = PortTable(tool, "outputs")
    qtbot.addWidget(table)

    combo = table.table.cellWidget(0, 1)
    assert combo.findData(RESULTS) >= 0


def test_port_table_removes_a_port(qtbot):
    tool = PythonScriptTool()
    table = PortTable(tool, "inputs")
    qtbot.addWidget(table)

    table.add_port()
    name = table._names[-1]
    assert name in tool.inputs

    row = table._names.index(name)
    table.table.selectRow(row)
    table.remove_selected()

    assert name not in tool.inputs
    assert name not in tool.port_metadata["inputs"]
