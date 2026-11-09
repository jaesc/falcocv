"""Contract tests for every registered tool (no GUI, no models).

A new tool has to touch several lookups, and forgetting one fails silently at
runtime (the tool disappears from a loaded job or its config dialog breaks).
These tests derive the expected set from the registry, so adding a tool is
enough to exercise them:

* ``@register_tool`` metadata (prefix, default name, toolbox group);
* the ``falcocv.tools`` package import (what populates the toolbox);
* ``_TOOL_MODULE_MAP`` (loading a ``.cvj`` in a cold process/worker);
* ``TOOL_PREFIXES`` (prefix restored when a serialized job carries none);
* ``tools/<pkg>/config.py`` exported by ``get_config_widget()``;
* the port contract in ``tests/test_port_types.py``.
"""

import ast
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import falcocv.tools  # noqa: F401  (populate the registry)
from falcocv.core.tool_registry import (
    TOOL_PREFIXES,
    create_tool,
    get_all_tools,
    get_registration,
)
from falcocv.core.workflow import _TOOL_MODULE_MAP

pytestmark = pytest.mark.integration

TOOLS = get_all_tools()
TOOLS_DIR = Path(__file__).resolve().parents[2] / "src" / "falcocv" / "tools"

# "ToolBlock" is the serialized name of Block before the rename.  Job files in
# the field still use it, so the lazy map must keep it as an alias.
LEGACY_TYPE_ALIASES = {"ToolBlock": "Block"}


# --------------------------------------------------------------------------- #
# Filesystem helpers
# --------------------------------------------------------------------------- #

def _tool_packages() -> set[str]:
    """Every ``falcocv.tools.<pkg>`` package that ships a ``logic.py``."""
    return {f"falcocv.tools.{path.parent.name}" for path in TOOLS_DIR.glob("*/logic.py")}


def _imported_tool_modules() -> set[str]:
    tree = ast.parse((TOOLS_DIR / "__init__.py").read_text(encoding="utf-8-sig"))
    modules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("falcocv.tools."):
                    modules.add(alias.name)
    return modules


def _config_class_name(logic_path: Path) -> str:
    """The config widget class imported lazily by ``get_config_widget()``."""
    tree = ast.parse(logic_path.read_text(encoding="utf-8-sig"))
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "get_config_widget":
            for child in ast.walk(node):
                if (
                    isinstance(child, ast.ImportFrom)
                    and child.level == 1
                    and child.module == "config"
                ):
                    assert len(child.names) == 1, f"{logic_path}: import one class only"
                    return child.names[0].name
    return ""


# --------------------------------------------------------------------------- #
# Serialization
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Registration sites
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("reg", TOOLS, ids=lambda r: r.tool_type)
def test_registration_metadata_is_complete(reg):
    assert reg.tool_type == reg.cls.__name__
    assert reg.prefix, f"{reg.tool_type}: @register_tool(prefix=...) is empty"
    assert reg.cls.prefix == reg.prefix
    assert reg.default_name, f"{reg.tool_type}: default_name is empty"
    assert reg.group, f"{reg.tool_type}: @register_tool(group=...) is empty"


@pytest.mark.parametrize("reg", TOOLS, ids=lambda r: r.tool_type)
def test_prefix_table_matches_the_registration(reg):
    assert TOOL_PREFIXES[reg.tool_type] == reg.prefix, (
        f"{reg.tool_type}: TOOL_PREFIXES disagrees with @register_tool(prefix=...)"
    )


def test_prefix_table_covers_the_registry_exactly():
    registered = {reg.tool_type for reg in TOOLS}
    missing = sorted(registered - set(TOOL_PREFIXES))
    stale = sorted(set(TOOL_PREFIXES) - registered)
    assert not missing, f"TOOL_PREFIXES is missing: {missing}"
    assert not stale, f"TOOL_PREFIXES has entries for unknown tools: {stale}"


def test_tools_package_imports_every_tool_module():
    assert _imported_tool_modules() == _tool_packages(), (
        "falcocv/tools/__init__.py must import every tools/<pkg>/logic.py module, "
        "otherwise `import falcocv.tools` does not register the tool for the toolbox"
    )


def test_registry_and_lazy_map_cover_every_tool_package():
    """Every registered class belongs to a package, and the lazy map points there."""
    module_of: dict[str, str] = {}
    for package in sorted(_tool_packages()):
        logic_path = f"{package}.logic"
        module = importlib.import_module(logic_path)
        for value in vars(module).values():
            if not isinstance(value, type) or value.__module__ != logic_path:
                continue
            reg = get_registration(value.__name__)
            if reg is not None and reg.cls is value:
                module_of[value.__name__] = package

    registered = {reg.tool_type for reg in TOOLS}
    assert set(module_of) == registered, (
        "a tools/<pkg>/logic.py defines a @register_tool class that is not in the "
        f"registry: {sorted(set(module_of) - registered)}"
    )
    wrong = {
        tool_type: (_TOOL_MODULE_MAP.get(tool_type), module_path)
        for tool_type, module_path in module_of.items()
        if _TOOL_MODULE_MAP.get(tool_type) != module_path
    }
    assert not wrong, f"_TOOL_MODULE_MAP points to the wrong module: {wrong}"


def test_lazy_import_map_has_no_stale_entries():
    allowed = {reg.tool_type for reg in TOOLS} | set(LEGACY_TYPE_ALIASES)
    stale = sorted(set(_TOOL_MODULE_MAP) - allowed)
    assert not stale, f"_TOOL_MODULE_MAP has entries for unknown tools: {stale}"

    for alias, target in LEGACY_TYPE_ALIASES.items():
        assert _TOOL_MODULE_MAP[alias] == _TOOL_MODULE_MAP[target]


def test_every_mapped_module_imports():
    broken = []
    for tool_type, module_path in _TOOL_MODULE_MAP.items():
        try:
            importlib.import_module(module_path)
        except Exception as exc:  # noqa: BLE001 - report every broken entry
            broken.append(f"{tool_type} -> {module_path}: {exc}")
    assert not broken, broken


@pytest.mark.parametrize("reg", TOOLS, ids=lambda r: r.tool_type)
def test_tool_ships_a_config_widget(reg):
    package_dir = Path(importlib.import_module(reg.cls.__module__).__file__).parent

    assert callable(getattr(reg.cls, "get_config_widget", None)), (
        f"{reg.tool_type}: define get_config_widget() so the editor can open its config"
    )

    config_path = package_dir / "config.py"
    assert config_path.is_file(), f"{reg.tool_type}: missing {config_path.name}"

    class_name = _config_class_name(package_dir / "logic.py")
    assert class_name, (
        f"{reg.tool_type}: get_config_widget() must import its widget from .config"
    )
    defined = {
        node.name
        for node in ast.parse(config_path.read_text(encoding="utf-8-sig")).body
        if isinstance(node, ast.ClassDef)
    }
    assert class_name in defined, (
        f"{reg.tool_type}: {config_path.name} does not define {class_name}"
    )


def test_every_tool_deserializes_in_a_cold_process():
    """A serialized workflow loads in a fresh interpreter, as the worker does.

    Also proves ``deserialize()`` needs no GUI and no model download.
    Map coverage is enforced by ``test_registry_and_lazy_map_cover_every_tool_package``:
    a full-payload load cannot isolate a missing entry, because importing any
    tool module runs ``falcocv.tools.__init__`` and registers everything.
    """
    expected = [reg.tool_type for reg in TOOLS]
    payload = json.dumps({"tools": [{"type": name} for name in expected], "connections": []})
    code = (
        "import json, os\n"
        "from falcocv.core.workflow import Workflow\n"
        "workflow = Workflow()\n"
        "workflow.deserialize(json.loads(os.environ['FALCOCV_TEST_PAYLOAD']))\n"
        "loaded = [tool.__class__.__name__ for tool in workflow.tools]\n"
        "expected = json.loads(os.environ['FALCOCV_TEST_EXPECTED'])\n"
        "assert loaded == expected, loaded\n"
        "without_prefix = [t.name for t in workflow.tools if not t.prefix]\n"
        "assert not without_prefix, f'no prefix (TOOL_PREFIXES): {without_prefix}'\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(TOOLS_DIR.parents[1]), env.get("PYTHONPATH", "")))
    )
    env["FALCOCV_TEST_PAYLOAD"] = payload
    env["FALCOCV_TEST_EXPECTED"] = json.dumps(expected)

    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    assert result.returncode == 0, result.stderr
