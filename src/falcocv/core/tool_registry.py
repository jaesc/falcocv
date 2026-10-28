from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolReg:
    cls: type
    prefix: str
    default_name: str
    tool_type: str
    group: str = ""
    config_cls: type | None = None


_TOOL_REGISTRY: dict[str, ToolReg] = {}


def register_tool(*, prefix: str, default_name: str = "", group: str = "", config_cls: type | None = None):
    def _inner(tool_cls: type) -> type:
        type_name = tool_cls.__name__
        _TOOL_REGISTRY[type_name] = ToolReg(
            cls=tool_cls,
            prefix=prefix,
            default_name=default_name or type_name,
            tool_type=type_name,
            group=group,
            config_cls=config_cls,
        )
        tool_cls.prefix = prefix
        return tool_cls
    return _inner


def create_tool(tool_type: str, **init_kwargs: Any):
    reg = _TOOL_REGISTRY.get(tool_type)
    if reg is None:
        return None
    return reg.cls(**init_kwargs)


def get_registration(tool_type: str) -> ToolReg | None:
    return _TOOL_REGISTRY.get(tool_type)


def get_registration_by_class(cls: type) -> ToolReg | None:
    return _TOOL_REGISTRY.get(cls.__name__)


def get_all_tools() -> list[ToolReg]:
    return list(_TOOL_REGISTRY.values())


def get_tool_count() -> int:
    return len(_TOOL_REGISTRY)


TOOL_PREFIXES: dict[str, str] = {
    "ImageLoader":         "[LOAD]",
    "FiltersTool":         "[FILT]",
    "AnomalyDetector":     "[ANOM]",
    "Block":               "[BLCK]",
    "GigeCameraLoader":    "[GIGE]",
    "PythonScriptTool":    "[PYTH]",
    "BlobDetector":        "[BLOB]",
    "HistogramTool":       "[HIST]",
    "PatternMatcher":      "[PATT]",
    "ShapeMatch":          "[SHAP]",
    "DepthRangeLoader":    "[DPTH]",
    "BarcodeDecoder":      "[BDEC]",
    "MeasurementTool":     "[MEAS]",
    "EdgeDetection":       "[EDGE]",
    "ImageCalculatorTool": "[CALC]",
    "Classifier":          "[CLAS]",
    "OCR":        "[IOCR]",
    "FixtureOverlay": "[FIXT]",
    "PlaneFitter": "[PLAN]",
}
