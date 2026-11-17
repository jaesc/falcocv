"""Create a Block tool from a contiguous range of workflow tools.

The workflow is an ordered pipeline.  Grouping non-contiguous tools into a
Block would move them relative to the tools left outside, so this module only
accepts a contiguous selection.
"""

from __future__ import annotations

import copy
import uuid

from falcocv.core.port_types import ANY, normalize_kind
from falcocv.core.tool_base import BaseTool
from falcocv.core.workflow import Workflow
from falcocv.tools.block.logic import Block


class BlockBuildError(ValueError):
    """Raised when the selected tools cannot be grouped into a Block."""


def _unique_name(base_name: str, existing: set[str]) -> str:
    base = str(base_name or "Block").strip() or "Block"
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _unique_port(base_name: str, existing: set[str]) -> str:
    base = str(base_name or "port").strip() or "port"
    if base not in existing:
        return base
    index = 2
    while f"{base}_{index}" in existing:
        index += 1
    return f"{base}_{index}"


def _port_kind(tool: BaseTool, direction: str, name: str) -> str:
    getter = getattr(
        tool,
        "get_output_port_type" if direction == "outputs" else "get_input_port_type",
        None,
    )
    if getter is None:
        return ANY
    return normalize_kind(getter(name))


def _reassign_uuids(tool: BaseTool) -> None:
    """Give a copied tool (and any nested tools) fresh UUIDs."""
    tool.uuid = str(uuid.uuid4())
    child = getattr(tool, "workflow", None)
    if child is not None:
        for nested in child.tools:
            _reassign_uuids(nested)


def _validate_selection(
    source_workflow: Workflow, tools: list[BaseTool]
) -> tuple[list[BaseTool], list[int]]:
    if len(tools) < 2:
        raise BlockBuildError("Select at least two tools.")

    unique: list[BaseTool] = []
    seen: set[int] = set()
    for tool in tools:
        if tool is source_workflow.parent_tool:
            raise BlockBuildError("The parent Block cannot be selected.")
        if not any(tool is candidate for candidate in source_workflow.tools):
            raise BlockBuildError(f"{tool.name} is not part of this workflow.")
        if id(tool) in seen:
            continue
        seen.add(id(tool))
        unique.append(tool)

    ordered = [
        tool for tool in source_workflow.tools if id(tool) in seen
    ]
    if len(ordered) != len(unique):
        raise BlockBuildError("The selection contains tools in an invalid state.")

    indices = [source_workflow.tools.index(tool) for tool in ordered]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise BlockBuildError(
            "The selected tools are not contiguous in the workflow. "
            "Select a contiguous range, without unselected tools in between."
        )
    return ordered, indices


def build_block_from_tools(
    source_workflow: Workflow,
    tools: list[BaseTool],
    name: str,
    keep_originals: bool = True,
) -> Block:
    """Group ``tools`` into a new :class:`Block`.

    ``keep_originals=True`` copies the tools into the Block and leaves the
    source workflow untouched.  ``False`` moves them into the Block, replacing
    the selection and rewiring its external connections to Block ports.

    The Block is inserted into ``source_workflow`` and returned.
    """
    ordered, indices = _validate_selection(source_workflow, tools)

    existing_names = {tool.name for tool in source_workflow.tools}
    block = Block(_unique_name(name, existing_names))
    block_workflow = block.workflow

    # Copy or move the tools into the block.
    if keep_originals:
        copied: list[BaseTool] = []
        copy_map: dict[int, BaseTool] = {}
        for tool in ordered:
            data = copy.deepcopy(tool.serialize())
            data.pop("uuid", None)
            clone = source_workflow.create_tool_from_data(data)
            if clone is None:
                raise BlockBuildError(f"Could not copy {tool.name}.")
            _reassign_uuids(clone)
            copied.append(clone)
            copy_map[id(tool)] = clone
        block_workflow.tools = copied
    else:
        copy_map = {id(tool): tool for tool in ordered}
        block_workflow.tools = list(ordered)

    selected_ids = set(copy_map)
    internal_connections = []
    incoming: dict[tuple[int, str], dict] = {}
    outgoing: dict[tuple[int, str], dict] = {}

    for src, out_name, dst, in_name in source_workflow.connections:
        src_selected = src is not None and id(src) in selected_ids
        dst_selected = dst is not None and id(dst) in selected_ids

        if src_selected and dst_selected:
            internal_connections.append(
                (
                    copy_map[id(src)],
                    out_name,
                    copy_map[id(dst)],
                    in_name,
                )
            )
        elif dst_selected and src is not None:
            key = (id(src), out_name)
            entry = incoming.setdefault(
                key,
                {"src": src, "out_name": out_name, "targets": []},
            )
            entry["targets"].append((dst, in_name))
        elif src_selected and dst is not None:
            key = (id(src), out_name)
            entry = outgoing.setdefault(
                key,
                {"src": src, "out_name": out_name, "targets": []},
            )
            entry["targets"].append((dst, in_name))

    block_workflow.connections = list(internal_connections)

    used_inputs: set[str] = set()
    used_outputs: set[str] = set()
    rewired = []

    for entry in incoming.values():
        src = entry["src"]
        out_name = entry["out_name"]
        preferred = out_name
        if preferred in used_inputs:
            preferred = f"{src.name}.{out_name}"
        port = _unique_port(preferred, used_inputs)
        used_inputs.add(port)

        kind = _port_kind(src, "outputs", out_name)
        block.port_metadata.setdefault("inputs", {})[port] = kind
        block.inputs[port] = None

        for dst, in_name in entry["targets"]:
            target = copy_map.get(id(dst), dst)
            block_workflow.connections.append((block, port, target, in_name))

        if not keep_originals:
            rewired.append((src, out_name, block, port))

    for entry in outgoing.values():
        src = entry["src"]
        out_name = entry["out_name"]
        preferred = out_name
        if preferred in used_outputs:
            preferred = f"{src.name}.{out_name}"
        port = _unique_port(preferred, used_outputs)
        used_outputs.add(port)

        kind = _port_kind(src, "outputs", out_name)
        block.port_metadata.setdefault("outputs", {})[port] = kind
        block.outputs[port] = None

        source_copy = copy_map[id(src)]
        block_workflow.connections.append((source_copy, out_name, block, port))

        if not keep_originals:
            for dst, in_name in entry["targets"]:
                rewired.append((block, port, dst, in_name))

    if keep_originals:
        source_workflow.tools.insert(max(indices) + 1, block)
    else:
        start = min(indices)
        source_workflow.connections = [
            connection
            for connection in source_workflow.connections
            if not (
                (connection[0] is not None and id(connection[0]) in selected_ids)
                or (connection[2] is not None and id(connection[2]) in selected_ids)
            )
        ] + rewired
        source_workflow.tools = [
            tool for tool in source_workflow.tools if id(tool) not in selected_ids
        ]
        source_workflow.tools.insert(start, block)

    return block
