"""A new workflow run must not reuse data from the previous run."""

import numpy as np

from falcocv.core.tool_base import BaseTool
from falcocv.core.vision_image import VisionImage
from falcocv.core.workflow import Workflow
from falcocv.tools.block.logic import Block
from falcocv.tools.filters.logic import FiltersTool


class SourceTool(BaseTool):
    def __init__(self, name="Source"):
        super().__init__(name)
        self.inputs = {}
        self.outputs = {"image": None}
        self.image = VisionImage(np.zeros((8, 8, 3), np.uint8))
        self.fail = False
        self.executions = 0

    def run(self, advance=False):
        self.executions += 1
        if self.fail:
            raise RuntimeError("boom")
        self.outputs["image"] = self.image


class ConsumerTool(BaseTool):
    def __init__(self, name="Consumer"):
        super().__init__(name)
        self.inputs = {"image": None}
        self.outputs = {"result": None}
        self.seen = []

    def run(self, advance=False):
        self.seen.append(self.inputs["image"])
        self.outputs["result"] = self.inputs["image"]


def test_unconnected_input_is_cleared_and_fails_the_guard():
    workflow = Workflow()
    filters = FiltersTool("Filters")
    workflow.add_tool(filters)
    filters.inputs["input image"] = VisionImage(np.zeros((8, 8, 3), np.uint8))

    workflow.run()

    assert filters.inputs["input image"] is None
    assert filters.last_error == "Input image is missing"
    assert filters.outputs["output image"] is None


def test_connected_input_receives_the_current_output():
    first = VisionImage(np.full((8, 8, 3), 10, np.uint8))
    second = VisionImage(np.full((8, 8, 3), 20, np.uint8))
    source = SourceTool("Source")
    consumer = ConsumerTool("Consumer")
    source.image = first
    workflow = Workflow()
    workflow.add_tool(source)
    workflow.add_tool(consumer)
    workflow.connect(source, "image", consumer, "image")

    workflow.run()
    assert consumer.seen[-1] is first

    source.image = second
    workflow.run()
    assert consumer.seen[-1] is second


def test_failed_source_does_not_leak_its_previous_output():
    source = SourceTool("Source")
    consumer = ConsumerTool("Consumer")
    workflow = Workflow()
    workflow.add_tool(source)
    workflow.add_tool(consumer)
    workflow.connect(source, "image", consumer, "image")
    workflow.stop_on_error = False

    workflow.run()
    assert consumer.seen[-1] is source.image

    source.fail = True
    workflow.run()

    assert source.last_error == "RuntimeError: boom"
    assert consumer.seen[-1] is None


def test_tool_before_its_source_never_sees_a_previous_run_output():
    source = SourceTool("Source")
    consumer = ConsumerTool("Consumer")
    workflow = Workflow()
    # Manual order is the contract: the consumer runs before the source.
    workflow.add_tool(consumer)
    workflow.add_tool(source)
    workflow.connect(source, "image", consumer, "image")

    workflow.run()
    workflow.run()

    assert consumer.seen == [None, None]
    assert source.outputs["image"] is source.image


def test_skipped_tool_outputs_are_cleared():
    first = SourceTool("First")
    first.fail = True
    second = SourceTool("Second")
    second.outputs["image"] = second.image
    workflow = Workflow()
    workflow.add_tool(first)
    workflow.add_tool(second)
    workflow.stop_on_error = True

    workflow.run()

    assert second.executions == 0
    assert second.outputs["image"] is None


def test_block_child_data_is_cleared_even_when_the_block_is_disabled():
    child = ConsumerTool("Child")
    child.inputs["image"] = VisionImage(np.zeros((8, 8, 3), np.uint8))
    block = Block("Block")
    block.workflow.add_tool(child)
    block.enabled = False

    workflow = Workflow()
    workflow.add_tool(block)
    workflow.run()

    assert child.inputs["image"] is None
    assert child.seen == []


def test_clear_run_data_keeps_params_and_port_keys():
    source = SourceTool("Source")
    source.params = {"name": "keep", "value": 3}
    workflow = Workflow()
    workflow.add_tool(source)

    workflow.run()

    assert source.params == {"name": "keep", "value": 3}
    assert list(source.inputs) == []
    assert list(source.outputs) == ["image"]
