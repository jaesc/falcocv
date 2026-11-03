"""Port kinds and connection compatibility for the workflow graph.

Every tool declares the kind of its image and plane ports; anything else stays
untyped (``ANY``).  The kind is never inferred from the port name: a tool that
adds a new image or plane port must declare it, and the tests fail otherwise.

Rules:

- a typed input only accepts outputs of its own kind;
- an untyped input accepts anything;
- an untyped output only feeds untyped inputs.
"""

from __future__ import annotations

IMAGE = "image"
PLANE = "plane"
ANY = "any"

PORT_LABELS = {IMAGE: "Image", PLANE: "Plane", ANY: "Any"}
LABEL_TO_PORT = {label: kind for kind, label in PORT_LABELS.items()}

# Legacy block metadata used display labels and extra kinds.  Older values map
# to the closest kind we have now; Number/String/Boolean behaved as untyped.
_ALIASES = {
    "image": IMAGE,
    "img": IMAGE,
    "plane": PLANE,
    "any": ANY,
    "number": ANY,
    "string": ANY,
    "boolean": ANY,
    "": ANY,
}


def normalize_kind(value) -> str:
    """Return the canonical kind for a declared value or metadata entry."""
    if value is None:
        return ANY
    return _ALIASES.get(str(value).strip().lower(), ANY)


def port_label(kind: str) -> str:
    return PORT_LABELS.get(normalize_kind(kind), PORT_LABELS[ANY])


def ports_compatible(out_kind, in_kind) -> bool:
    """Whether an output of ``out_kind`` may feed an input of ``in_kind``."""
    in_kind = normalize_kind(in_kind)
    out_kind = normalize_kind(out_kind)
    if in_kind == ANY:
        return True
    return out_kind == in_kind


def connection_candidates(workflow, dst_tool, in_name):
    """``(tool, out_name)`` pairs that may feed ``dst_tool.in_name``.

    Only tools that run before ``dst_tool`` are offered; the workflow order is
    the execution order, so this also makes cycles impossible.  Parent-block
    inputs are handled separately (they come from outside the level).
    """
    in_kind = dst_tool.get_input_port_type(in_name)
    tools = list(getattr(workflow, "tools", []))
    try:
        limit = tools.index(dst_tool)
    except ValueError:
        limit = len(tools)
    candidates = []
    for tool in tools[:limit]:
        for out_name in tool.outputs:
            if ports_compatible(tool.get_output_port_type(out_name), in_kind):
                candidates.append((tool, out_name))
    return candidates


def parent_input_candidates(workflow, dst_tool, in_name):
    """Parent-block inputs usable as sources for ``dst_tool.in_name``."""
    parent = getattr(workflow, "parent_tool", None)
    if parent is None or dst_tool is parent:
        return []
    in_kind = dst_tool.get_input_port_type(in_name)
    return [
        (parent, name)
        for name in parent.inputs
        if ports_compatible(parent.get_input_port_type(name), in_kind)
    ]


def _port_dict(tool, direction):
    return tool.inputs if direction == "inputs" else tool.outputs


def add_port(tool, direction, name, kind=ANY):
    """Create a port in ``tool`` and register its kind."""
    name = str(name or "").strip()
    target = _port_dict(tool, direction)
    if not name or name in target:
        return None
    target[name] = None
    tool.port_metadata.setdefault(direction, {})[name] = normalize_kind(kind)
    return name


def remove_port(tool, direction, name):
    """Remove a port and its metadata entry."""
    _port_dict(tool, direction).pop(name, None)
    tool.port_metadata.setdefault(direction, {}).pop(name, None)


def set_port_type(tool, direction, name, kind):
    """Change the kind of an existing port."""
    if name in _port_dict(tool, direction):
        tool.port_metadata.setdefault(direction, {})[name] = normalize_kind(
            kind
        )


def rename_port(tool, direction, old, new):
    """Rename a port, keeping its kind.  Returns the final name."""
    new = str(new or "").strip()
    target = _port_dict(tool, direction)
    if not new or new == old or old not in target or new in target:
        return old
    items = [
        (new, value) if key == old else (key, value)
        for key, value in target.items()
    ]
    target.clear()
    target.update(items)
    metadata = tool.port_metadata.setdefault(direction, {})
    kind = metadata.pop(old, ANY)
    metadata[new] = kind
    return new


def unique_port_name(tool, direction, prefix="Port"):
    target = _port_dict(tool, direction)
    index = 1
    while f"{prefix}_{index}" in target:
        index += 1
    return f"{prefix}_{index}"
