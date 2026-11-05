"""Runtime object injection from the HMI script into job Python Script tools."""

from falcocv.core.hmi import HMIProject
from falcocv.tools.python_script.logic import PythonScriptTool


def _dummy_comms_config():
    return {
        "version": 1,
        "devices": [
            {
                "id": "dummy",
                "driver": "dummy",
                "tags": [
                    {
                        "name": "trigger",
                        "type": "bool",
                        "access": "rw",
                        "default": False,
                    },
                    {
                        "name": "count",
                        "type": "int",
                        "access": "rw",
                        "default": 0,
                    },
                ],
            }
        ],
    }


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


def test_default_python_script_template_compiles_and_exposes_log():
    from falcocv.tools.python_script.config import DEFAULT_TEMPLATE

    assert "mecheye" not in DEFAULT_TEMPLATE
    assert 'self.engine.get("log")' in DEFAULT_TEMPLATE
    assert "engine_context or {}" in DEFAULT_TEMPLATE
    compile(DEFAULT_TEMPLATE, "<default-template>", "exec")


def test_hmi_project_round_trips_comms_config(tmp_path):
    project = HMIProject()
    project.comms_config = _dummy_comms_config()
    path = tmp_path / "project.hmi"

    project.save(path)
    restored = HMIProject.load(path)

    assert restored.comms_config["devices"][0]["id"] == "dummy"
    assert restored.comms is None


def test_hmi_project_start_runtime_injects_comms_into_context():
    project = HMIProject()
    project.comms_config = _dummy_comms_config()

    project.start_runtime()
    try:
        assert project.comms is not None
        assert project.context["comms"] is project.comms
        assert project.comms.read("dummy.trigger") is False
        project.comms.write("dummy.trigger", True)
        assert project.comms.read("dummy.trigger") is True
    finally:
        project.stop_runtime()

    assert project.comms is None
    assert "comms" not in project.context


def test_hmi_script_namespace_exposes_comms():
    project = HMIProject()
    project.comms_config = _dummy_comms_config()
    project.start_runtime()
    try:
        project.script = "comms.write('dummy.count', 42)"
        project._run_script_single()
        assert project.comms.read("dummy.count") == 42
    finally:
        project.stop_runtime()


def test_job_python_script_receives_comms_through_context():
    project = HMIProject()
    project.comms_config = _dummy_comms_config()
    project.start_runtime()
    try:
        tool = PythonScriptTool()
        tool.params["script_code"] = """
class CustomTool:
    def __init__(self, engine_context):
        self.engine = engine_context or {}

    def process(self, inputs):
        comms = self.engine.get("comms")
        comms.write("dummy.count", 99)
        return {"output image": None}
"""
        tool.engine_context = project.context
        tool.run()
        assert project.comms.read("dummy.count") == 99
    finally:
        project.stop_runtime()


def test_hmi_project_without_comms_starts_without_runtime():
    project = HMIProject()
    project.start_runtime()
    assert project.comms is None
    project.stop_runtime()


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
