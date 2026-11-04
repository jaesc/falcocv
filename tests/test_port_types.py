"""Port kinds and connection filtering."""

import pytest

from falcocv.core.port_types import (
    ANY,
    IMAGE,
    PLANE,
    RESULTS,
    connection_candidates,
    normalize_kind,
    parent_input_candidates,
    ports_compatible,
    rename_port,
)
from falcocv.core.tool_base import BaseTool
from falcocv.core.workflow import Workflow


class _ImageSource(BaseTool):
    output_port_types = {"output image": IMAGE}

    def __init__(self, name="Image Source"):
        super().__init__(name)
        self.inputs = {}
        self.outputs = {"output image": None}

    def run(self, advance=False):
        pass


class _PlaneSource(BaseTool):
    output_port_types = {"plane": PLANE}

    def __init__(self, name="Plane Source"):
        super().__init__(name)
        self.inputs = {}
        self.outputs = {"plane": None}

    def run(self, advance=False):
        pass


class _Consumer(BaseTool):
    input_port_types = {"input image": IMAGE, "reference plane": PLANE}

    def __init__(self, name="Consumer"):
        super().__init__(name)
        self.inputs = {"input image": None, "reference plane": None}
        self.outputs = {}

    def run(self, advance=False):
        pass


def test_normalize_kind_accepts_labels_and_legacy_values():
    assert normalize_kind("Image") == IMAGE
    assert normalize_kind("plane") == PLANE
    assert normalize_kind(None) == ANY
    assert normalize_kind("Number") == ANY
    assert normalize_kind("String") == ANY
    assert normalize_kind("anything else") == ANY


def test_ports_compatible_follows_the_typed_input_rule():
    assert ports_compatible(IMAGE, IMAGE)
    assert ports_compatible(PLANE, PLANE)
    assert not ports_compatible(IMAGE, PLANE)
    assert not ports_compatible(PLANE, IMAGE)
    # An untyped input accepts anything.
    assert ports_compatible(IMAGE, ANY)
    assert ports_compatible(PLANE, ANY)
    # An untyped output only feeds untyped inputs.
    assert ports_compatible(ANY, ANY)
    assert not ports_compatible(ANY, IMAGE)


def test_connection_candidates_respect_order_and_type():
    image_source = _ImageSource("Image Source")
    plane_source = _PlaneSource("Plane Source")
    consumer = _Consumer()

    workflow = Workflow()
    workflow.add_tool(image_source)
    workflow.add_tool(plane_source)
    workflow.add_tool(consumer)

    image_candidates = connection_candidates(
        workflow, consumer, "input image"
    )
    assert image_candidates == [(image_source, "output image")]

    plane_candidates = connection_candidates(
        workflow, consumer, "reference plane"
    )
    assert plane_candidates == [(plane_source, "plane")]


def test_connection_candidates_ignore_later_tools():
    consumer = _Consumer("Early Consumer")
    image_source = _ImageSource("Late Source")

    workflow = Workflow()
    workflow.add_tool(consumer)
    workflow.add_tool(image_source)

    assert connection_candidates(workflow, consumer, "input image") == []


def test_parent_inputs_are_typed_and_always_available():
    from falcocv.tools.block.logic import Block

    block = Block("Parent")
    block.port_metadata = {
        "inputs": {"image": IMAGE, "plane": PLANE},
        "outputs": {},
    }
    block.inputs = {"image": None, "plane": None}

    consumer = _Consumer("Inner")
    inner = Workflow(parent_tool=block)
    inner.add_tool(consumer)

    image_candidates = parent_input_candidates(inner, consumer, "input image")
    assert image_candidates == [(block, "image")]

    plane_candidates = parent_input_candidates(
        inner, consumer, "reference plane"
    )
    assert plane_candidates == [(block, "plane")]


def test_block_reports_port_types_from_metadata():
    from falcocv.tools.block.logic import Block

    block = Block("Parent")
    block.port_metadata = {
        "inputs": {"image": "Image", "plain": "Number"},
        "outputs": {"result": PLANE},
    }
    assert block.get_input_port_type("image") == IMAGE
    assert block.get_input_port_type("plain") == ANY
    assert block.get_output_port_type("result") == PLANE
    assert block.get_output_port_type("missing") == ANY


GOLDEN_IMAGE_PORTS = {
    "AnomalyDetector": ({"input image"}, {"output image"}),
    "BarcodeDecoder": ({"input image"}, {"output image"}),
    "BlobDetector": ({"input image"}, {"output image", "binary mask"}),
    "Classifier": ({"input image"}, {"output image"}),
    "DepthRangeLoader": (set(), {"output image"}),
    "EdgeDetection": ({"input image"}, {"output image"}),
    "FiltersTool": ({"input image"}, {"output image"}),
    "FixtureOverlay": ({"input image"}, {"output image"}),
    "GigeCameraLoader": (set(), {"output image"}),
    "HeightBlobs3D": ({"input image"}, {"output image", "binary mask"}),
    "HistogramTool": ({"input image"}, {"output image"}),
    "ImageCalculatorTool": ({"image_A", "image_B"}, {"output image"}),
    "ImageLoader": (set(), {"output image"}),
    "MeasurementTool": ({"input image"}, {"output image"}),
    "OCR": ({"input image"}, {"output image"}),
    "PlaneFitter": ({"input image"}, {"output image"}),
    "ShapeMatch": ({"input image"}, {"output image"}),
    "TemplateMatch": ({"input image"}, {"output image"}),
}

GOLDEN_PLANE_PORTS = {
    "PlaneFitter": (set(), {"plane"}),
}


def _registered_tools():
    import falcocv.tools  # noqa: F401  (registers every tool)
    from falcocv.core.tool_registry import get_all_tools

    return {reg.tool_type: reg.cls for reg in get_all_tools()}


def test_every_tool_with_results_declares_results_kind():
    classes = _registered_tools()
    for name, tool_class in classes.items():
        tool = tool_class()
        if "results" in tool.outputs:
            assert tool.get_output_port_type("results") == RESULTS, name


def test_every_tool_declares_its_image_and_plane_ports():
    classes = _registered_tools()
    checked = set()
    for name, (inputs, outputs) in GOLDEN_IMAGE_PORTS.items():
        tool = classes[name]()
        image_inputs = {
            port for port, kind in tool.input_port_types.items()
            if normalize_kind(kind) == IMAGE
        }
        image_outputs = {
            port for port, kind in tool.output_port_types.items()
            if normalize_kind(kind) == IMAGE
        }
        assert image_inputs == inputs, name
        assert image_outputs == outputs, name
        for port in tool.input_port_types:
            assert port in tool.inputs, f"{name}: {port} is not an input"
        for port in tool.output_port_types:
            assert port in tool.outputs, f"{name}: {port} is not an output"
        checked.add(name)

    for name, (inputs, outputs) in GOLDEN_PLANE_PORTS.items():
        tool = classes[name]()
        plane_outputs = {
            port for port, kind in tool.output_port_types.items()
            if normalize_kind(kind) == PLANE
        }
        assert plane_outputs == outputs, name
        checked.add(name)

    # Block has dynamic ports typed through port_metadata.
    assert "Block" in classes

    # Typed non-image ports: the plane datum and the structured results.
    height_blobs = classes["HeightBlobs3D"]()
    assert height_blobs.get_input_port_type("reference plane") == PLANE
    assert height_blobs.get_output_port_type("results") == RESULTS
    plane_fitter = classes["PlaneFitter"]()
    assert plane_fitter.get_output_port_type("plane") == PLANE

    assert set(GOLDEN_IMAGE_PORTS) | set(GOLDEN_PLANE_PORTS) == checked


def test_undeclared_ports_are_not_treated_as_images():
    from falcocv.tools.filters.logic import FiltersTool

    tool = FiltersTool()
    assert tool.get_output_port_type("output image") == IMAGE
    assert tool.get_output_port_type("count") == ANY
    assert tool.get_input_port_type("unknown port") == ANY


def test_python_script_types_its_ports_through_metadata():
    from falcocv.tools.python_script.logic import PythonScriptTool

    tool = PythonScriptTool()
    assert tool.get_input_port_type("input image") == IMAGE
    assert tool.get_output_port_type("output image") == IMAGE

    name = tool.add_input_port(kind=PLANE)
    assert tool.get_input_port_type(name) == PLANE

    tool.set_input_ports(["input image", name])
    assert tool.get_input_port_type(name) == PLANE


def test_python_script_ports_round_trip():
    from falcocv.tools.python_script.logic import PythonScriptTool

    tool = PythonScriptTool()
    tool.add_input_port("reference plane", kind=PLANE)

    clone = PythonScriptTool()
    clone.deserialize(tool.serialize())

    assert list(clone.inputs) == ["input image", "reference plane"]
    assert clone.get_input_port_type("reference plane") == PLANE
    assert clone.get_output_port_type("output image") == IMAGE


def test_rename_port_follows_connections():
    image_source = _ImageSource("Source")
    consumer = _Consumer("Sink")
    workflow = Workflow()
    workflow.add_tool(image_source)
    workflow.add_tool(consumer)
    workflow.connect(image_source, "output image", consumer, "input image")

    rename_port(consumer, "inputs", "input image", "frame")
    workflow.rename_port(consumer, "inputs", "input image", "frame")
    assert workflow.connections == [
        (image_source, "output image", consumer, "frame")
    ]

    rename_port(image_source, "outputs", "output image", "frame_out")
    workflow.rename_port(image_source, "outputs", "output image", "frame_out")
    assert workflow.connections == [
        (image_source, "frame_out", consumer, "frame")
    ]


def test_remove_port_drops_its_connections():
    image_source = _ImageSource("Source")
    consumer = _Consumer("Sink")
    workflow = Workflow()
    workflow.add_tool(image_source)
    workflow.add_tool(consumer)
    workflow.connect(image_source, "output image", consumer, "input image")

    workflow.remove_port_connections(consumer, "inputs", "input image")
    assert workflow.connections == []

    workflow.connect(image_source, "output image", consumer, "input image")
    workflow.remove_port_connections(image_source, "outputs", "output image")
    assert workflow.connections == []
