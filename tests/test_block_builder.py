"""Block builder: group a contiguous range of workflow tools."""

import pytest

from falcocv.core.port_types import IMAGE, RESULTS
from falcocv.core.tool_base import BaseTool
from falcocv.core.workflow import Workflow
from falcocv.tools.block.builder import BlockBuildError, build_block_from_tools
from falcocv.tools.blob_detector.logic import BlobDetector
from falcocv.tools.filters.logic import FiltersTool
from falcocv.tools.image_loader.logic import ImageLoader


class _Sink(BaseTool):
    input_port_types = {"results": RESULTS, "input image": IMAGE}

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


def test_move_block_rewires_external_connections():
    workflow, loader, filters, blob, sink = _workflow()

    block = build_block_from_tools(
        workflow, [filters, blob], "Group", keep_originals=False
    )

    assert workflow.tools == [loader, block, sink]
    assert block.workflow.tools == [filters, blob]
    assert filters in block.workflow.tools
    assert blob in block.workflow.tools

    # External edges became typed Block ports.
    assert block.inputs == {"output image": None}
    assert block.outputs == {"results": None}
    assert block.get_input_port_type("output image") == IMAGE
    assert block.get_output_port_type("results") == RESULTS

    assert workflow.connections == [
        (loader, "output image", block, "output image"),
        (block, "results", sink, "results"),
    ]
    assert block.workflow.connections == [
        (filters, "output image", blob, "input image"),
        (block, "output image", filters, "input image"),
        (blob, "results", block, "results"),
    ]


def test_copy_block_keeps_originals_and_uses_new_tools():
    workflow, loader, filters, blob, sink = _workflow()
    before_connections = list(workflow.connections)

    block = build_block_from_tools(
        workflow, [filters, blob], "Group", keep_originals=True
    )

    assert workflow.tools == [loader, filters, blob, block, sink]
    assert workflow.connections == before_connections
    assert len(block.workflow.tools) == 2
    assert block.workflow.tools[0] is not filters
    assert block.workflow.tools[1] is not blob
    assert block.workflow.tools[0].uuid != filters.uuid
    assert block.workflow.tools[1].uuid != blob.uuid

    assert block.inputs == {"output image": None}
    assert block.outputs == {"results": None}
    assert block.workflow.connections == [
        (block.workflow.tools[0], "output image", block.workflow.tools[1], "input image"),
        (block, "output image", block.workflow.tools[0], "input image"),
        (block.workflow.tools[1], "results", block, "results"),
    ]


def test_non_contiguous_selection_is_rejected():
    workflow, _loader, filters, _blob, _sink = _workflow()

    with pytest.raises(BlockBuildError, match="not contiguous"):
        build_block_from_tools(
            workflow, [filters, _sink], "Group", keep_originals=True
        )


def test_block_name_collision_gets_a_suffix():
    workflow, _loader, filters, blob, _sink = _workflow()
    workflow.add_tool(_Sink("Group"))

    block = build_block_from_tools(
        workflow, [filters, blob], "Group", keep_originals=True
    )

    assert block.name == "Group_2"


def test_fan_out_uses_one_port_per_source_output():
    workflow = Workflow()
    loader = ImageLoader("loader")
    filters = FiltersTool("filters")
    blob = BlobDetector("blob")
    consumer = _Sink("consumer")
    sink = _Sink("sink")
    for tool in (loader, filters, blob, consumer, sink):
        workflow.add_tool(tool)
    workflow.connect(loader, "output image", filters, "input image")
    workflow.connect(loader, "output image", consumer, "results")
    workflow.connect(filters, "output image", blob, "input image")
    workflow.connect(blob, "results", sink, "results")

    block = build_block_from_tools(
        workflow, [filters, blob, consumer], "Group", keep_originals=False
    )

    assert block.inputs == {"output image": None}
    assert [c for c in workflow.connections if c[2] is block] == [
        (loader, "output image", block, "output image")
    ]
    # One Block port feeds both internal consumers with different input names.
    assert len([
        c for c in block.workflow.connections
        if c[0] is block
    ]) == 2


def test_block_serializes_its_ports_and_nested_workflow():
    from falcocv.tools.block.logic import Block

    workflow, _loader, filters, blob, _sink = _workflow()
    block = build_block_from_tools(
        workflow, [filters, blob], "Group", keep_originals=False
    )

    restored = Block("restored")
    restored.deserialize(block.serialize())

    assert restored.inputs == block.inputs
    assert restored.outputs == block.outputs
    assert restored.port_metadata == block.port_metadata
    assert len(restored.workflow.tools) == 2
    assert any(c[0] is restored for c in restored.workflow.connections)
    assert any(c[2] is restored for c in restored.workflow.connections)


def test_move_inside_a_block_rewires_the_parent_ports():
    from falcocv.tools.block.logic import Block

    parent = Block("parent")
    parent.inputs = {"in": None}
    parent.outputs = {"out": None}
    parent.port_metadata = {"inputs": {"in": IMAGE}, "outputs": {"out": RESULTS}}

    filters = FiltersTool("filters")
    blob = BlobDetector("blob")
    parent.workflow.add_tool(filters)
    parent.workflow.add_tool(blob)
    parent.workflow.connect(parent, "in", filters, "input image")
    parent.workflow.connect(filters, "output image", blob, "input image")
    parent.workflow.connect(blob, "results", parent, "out")

    block = build_block_from_tools(
        parent.workflow, [filters, blob], "Group", keep_originals=False
    )

    assert parent.workflow.tools == [block]
    assert parent.workflow.connections == [
        (parent, "in", block, "in"),
        (block, "results", parent, "out"),
    ]
    assert block.inputs == {"in": None}
    assert block.outputs == {"results": None}
