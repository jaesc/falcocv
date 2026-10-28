# Logic module for Block.

from falcocv.core.tool_base import BaseTool
from falcocv.core.tool_registry import register_tool
from falcocv.core.workflow import Workflow

@register_tool(prefix="[BLCK]", default_name="Block", group="Logic")
class Block(BaseTool):
    def __init__(self, name="Block"):
        super().__init__(name)
        self.workflow = Workflow(parent_tool=self)
        self.inputs = {}
        self.outputs = {}
        self.port_metadata = {"inputs": {}, "outputs": {}}

    def run(self, advance=False):
        self.workflow.run(advance=advance)
        if self.workflow.last_error:
            raise RuntimeError(self.workflow.last_error)

    def execute(self, advance=False):
        super().execute(advance=advance)
        self.data_changed.emit()

    def serialize(self, job_manager=None) -> dict:
        data = super().serialize(job_manager)
        data["type"] = "Block"
        data["port_metadata"] = self.port_metadata
        data["workflow"] = self.workflow.serialize(job_manager)
        return data

    def deserialize(self, data: dict, job_manager=None):
        super().deserialize(data, job_manager)
        self.port_metadata = data.get("port_metadata", {"inputs": {}, "outputs": {}})
        for in_name in self.port_metadata.get("inputs", {}):
            self.inputs[in_name] = None
        for out_name in self.port_metadata.get("outputs", {}):
            self.outputs[out_name] = None
            
        if "workflow" in data:
            self.workflow.deserialize(data["workflow"], job_manager)

    def get_config_widget(self):
        from .config import BlockConfig
        return BlockConfig(self)
