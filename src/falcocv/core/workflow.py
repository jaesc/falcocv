from falcocv.core.tool_registry import TOOL_PREFIXES

_TOOL_MODULE_MAP: dict[str, str] = {
    "ImageLoader": "falcocv.tools.image_loader",
    "FiltersTool": "falcocv.tools.filters",
    "Block": "falcocv.tools.block",
    "ToolBlock": "falcocv.tools.block",
    "GigeCameraLoader": "falcocv.tools.gige_camera",
    "PythonScriptTool": "falcocv.tools.python_script",
    "BlobDetector": "falcocv.tools.blob_detector",
    "HistogramTool": "falcocv.tools.histogram",
    "PatternMatcher": "falcocv.tools.pattern_matcher",
    "ShapeMatch": "falcocv.tools.shape_match",
    "DepthRangeLoader": "falcocv.tools.depth_range_loader",
    "BarcodeDecoder": "falcocv.tools.barcode_decoder",
    "MeasurementTool": "falcocv.tools.measurement",
    "EdgeDetection": "falcocv.tools.edge_detection",
    "AnomalyDetector": "falcocv.tools.anomaly_detector",
    "Classifier": "falcocv.tools.classifier",
    "ImageCalculatorTool": "falcocv.tools.image_calculator",
    "OCR": "falcocv.tools.ocr",
    "FixtureOverlay": "falcocv.tools.fixture_overlay",
    "PlaneFitter": "falcocv.tools.plane_fitter",
}


class Workflow:
    def __init__(self, parent_tool=None):
        self.parent_tool = parent_tool
        self.tools = []
        self.connections = [] # List of (source_tool, output_name, target_tool, input_name)
        self.last_execution_time_ms = 0.0
        self.stop_on_error: bool = False
        self.last_error: str | None = None
        self.engine_context = None
        self.hmi_layout: dict = {}

    def add_tool(self, tool):
        self.tools.append(tool)

    def get_tool(self, tool_name):
        return next((t for t in self.tools if t.name == tool_name), None)

    def get_output(self, tool_name, output_name):
        tool = self.get_tool(tool_name)
        return tool.get_output(output_name) if tool else None

    def remove_tool(self, tool):
        if tool in self.tools:
            self.tools.remove(tool)
            # Remove associated connections
            self.connections = [c for c in self.connections if c[0] != tool and c[2] != tool]

    def connect(self, src_tool, out_name, dst_tool, in_name):
        # Remove existing connection to this input if any
        self.remove_connection(dst_tool, in_name)
        self.connections.append((src_tool, out_name, dst_tool, in_name))

    def remove_connection(self, dst_tool, in_name):
        self.connections = [c for c in self.connections if not (c[2] == dst_tool and c[3] == in_name)]

    def run(self, advance=False):
        import time
        start_ns = time.perf_counter_ns()
        self.last_error = None
        
        # Optimize: Group connections by destination tool
        conns_by_dst = {}
        for src, out_name, dst, in_name in self.connections:
            if dst not in conns_by_dst: conns_by_dst[dst] = []
            conns_by_dst[dst].append((src, out_name, in_name))

        for tool in self.tools:
            # 1. Update inputs from pre-grouped connections
            if tool in conns_by_dst:
                for src, out_name, in_name in conns_by_dst[tool]:
                    if self.parent_tool and src == self.parent_tool:
                        tool.set_input(in_name, self.parent_tool.inputs.get(out_name))
                    elif src is not None:
                        tool.set_input(in_name, src.get_output(out_name))
            
            # 2. Inject context
            if self.engine_context:
                tool.engine_context = self.engine_context

            # 3. Execute if enabled
            if getattr(tool, 'enabled', True):
                try:
                    tool.execute(advance=advance)
                except Exception as e:
                    import traceback
                    err_msg = f"{type(e).__name__}: {e}"
                    tool.last_error = err_msg
                    self.last_error = f"{tool.name}: {err_msg}"
                    print(f"Workflow error in {tool.name}:\n{traceback.format_exc()}")
                    if self.stop_on_error:
                        break
                    continue
            else:
                # Disabled tools do not execute and should not provide stale outputs
                for output_name in list(tool.outputs.keys()):
                    tool.outputs[output_name] = None
                tool.last_error = None
                tool.last_execution_time_ms = 0.0

            if tool.last_error:
                self.last_error = f"{tool.name}: {tool.last_error}"
                if self.stop_on_error:
                    break
        
        # 4. Push results to parent if this is a sub-workflow
        if self.parent_tool:
            if self.parent_tool in conns_by_dst:
                for src, out_name, in_name in conns_by_dst[self.parent_tool]:
                    self.parent_tool.outputs[in_name] = src.get_output(out_name)
                    
        end_ns = time.perf_counter_ns()
        self.last_execution_time_ms = (end_ns - start_ns) / 1e6

    def serialize(self, job_manager=None) -> dict:
        tools_data = [t.serialize(job_manager) for t in self.tools]
        conns_data = []
        for src, out_name, dst, in_name in self.connections:
            src_id = "PARENT" if src == self.parent_tool else src.uuid
            dst_id = "PARENT" if dst == self.parent_tool else dst.uuid
            conns_data.append({
                "src": src_id,
                "out_name": out_name,
                "dst": dst_id,
                "in_name": in_name
            })
        return {
            "tools": tools_data,
            "connections": conns_data,
            "stop_on_error": self.stop_on_error,
            "hmi_layout": self.hmi_layout,
        }

    def create_tool_from_data(self, data: dict, job_manager=None):
        """Factory method to create a tool instance from serialized data."""
        from falcocv.core.tool_registry import get_registration, TOOL_PREFIXES

        t_type = data.get("type")
        if not t_type:
            return None

        if t_type == "ToolBlock":
            t_type = "Block"

        reg = get_registration(t_type)
        if reg is None:
            module_path = _TOOL_MODULE_MAP.get(t_type)
            if module_path:
                import importlib
                importlib.import_module(module_path)
                reg = get_registration(t_type)
        if reg is None:
            return None

        tool = reg.cls()
        tool.deserialize(data, job_manager)
        if not tool.prefix:
            tool.prefix = TOOL_PREFIXES.get(t_type, "")
        return tool

    def deserialize(self, data: dict, job_manager=None):
        self.tools.clear()
        self.connections.clear()
        self.stop_on_error = data.get("stop_on_error", False)
        self.hmi_layout = data.get("hmi_layout", {})
        
        tool_map = {}
        for t_data in data.get("tools", []):
            tool = self.create_tool_from_data(t_data, job_manager)
            if tool:
                self.add_tool(tool)
                tool_map[tool.uuid] = tool
        
        for c_data in data.get("connections", []):
            src_id = c_data["src"]
            dst_id = c_data["dst"]
            
            src_tool = self.parent_tool if src_id == "PARENT" else tool_map.get(src_id)
            dst_tool = self.parent_tool if dst_id == "PARENT" else tool_map.get(dst_id)
            
            if src_tool and dst_tool:
                self.connect(src_tool, c_data["out_name"], dst_tool, c_data["in_name"])
