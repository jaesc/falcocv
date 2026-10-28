"""Contract tests for every registered tool (no GUI, no models)."""

import pytest

import falcocv.tools  # noqa: F401  (populate the registry)
from falcocv.core.tool_registry import create_tool, get_all_tools

pytestmark = pytest.mark.integration

TOOLS = get_all_tools()


@pytest.mark.parametrize("reg", TOOLS, ids=lambda r: r.tool_type)
def test_tool_serialization_roundtrip(reg):
    tool = create_tool(reg.tool_type)
    assert tool is not None

    data = tool.serialize()
    assert isinstance(data, dict)
    assert data.get("type")

    clone = create_tool(reg.tool_type)
    clone.deserialize(data)

    assert clone.name == tool.name
    assert isinstance(clone.params, dict)
    roundtrip = clone.serialize()
    assert roundtrip.get("type") == data.get("type")


@pytest.mark.parametrize("reg", TOOLS, ids=lambda r: r.tool_type)
def test_tool_config_schema(reg):
    tool = create_tool(reg.tool_type)
    schema = tool.get_config_schema()
    assert isinstance(schema, dict)
    assert schema.get("type") == "object"
