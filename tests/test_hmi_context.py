"""Runtime object injection from the HMI script into job Python Script tools."""

from falcocv.core.hmi import HMIProject
from falcocv.tools.python_script.logic import PythonScriptTool


class _DummyController:
    def __init__(self):
        self.workflow = type("DummyWorkflow", (), {"engine_context": None})()
        self.seen_context = None

    def run(self, advance=True):
        self.seen_context = self.workflow.engine_context


def test_run_slot_injects_hmi_context_into_workflow():
    project = HMIProject()
    slot = project.add_slot(name="Slot1")
    controller = _DummyController()
    slot._controller = controller
    marker = object()
    project.context["external"] = marker

    project.run_slot("Slot1")

    assert controller.seen_context is project.context
    assert controller.seen_context["external"] is marker


def test_hmi_script_namespace_exposes_context():
    project = HMIProject()
    project.script = "context['marker'] = 42"
    project._run_script_single()
    assert project.context["marker"] == 42


def test_python_script_receives_and_refreshes_engine_context():
    tool = PythonScriptTool()
    tool.params["script_code"] = """
class CustomTool:
    def __init__(self, engine_context):
        self.engine = engine_context or {}

    def process(self, inputs):
        return {"output image": None}
"""
    first = {"source": "first"}
    second = {"source": "second"}

    tool.engine_context = first
    tool.run()
    assert tool._instance.engine is first

    tool.engine_context = second
    tool.run()
    assert tool._instance.engine is second
