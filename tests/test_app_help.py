"""Application help pages: structure, links and coverage."""

from falcocv.ui.app_help import APP_HELP
from falcocv.ui.tool_help import TOOL_HELP

GUIDE_CATEGORIES = {"Getting Started", "Guides", "Reference"}


def test_guides_are_merged_into_tool_help():
    for key in APP_HELP:
        assert TOOL_HELP[key] is APP_HELP[key]


def test_every_guide_has_the_required_structure():
    for key, entry in APP_HELP.items():
        assert entry["kind"] == "guide", key
        assert entry["name"] == key, key
        assert entry["category"] in GUIDE_CATEGORIES, key
        assert entry["purpose"].strip(), key
        assert entry["sections"], key
        for section in entry["sections"]:
            assert section.get("title", "").strip(), key
            assert section.get("html", "").strip(), key


def test_guide_names_are_unique():
    assert len(APP_HELP) == len({entry["name"] for entry in APP_HELP.values()})


def test_see_also_targets_resolve():
    for key, entry in APP_HELP.items():
        for target in entry.get("see_also", []):
            assert target in TOOL_HELP, f"{key} -> {target}"


def test_guides_cover_the_main_user_topics():
    expected = {
        "Getting Started",
        "Your First Workflow",
        "Files, Folders & Logs",
        "Troubleshooting",
        "Vision Editor",
        "Running Jobs",
        "HMI Designer",
        "Web Dashboard & HMI",
        "Communications",
        "Custom Tools & Debugger",
        "3D & Depth",
        "User Libraries",
        "Command Line",
    }
    assert expected <= set(APP_HELP)


def test_every_registered_tool_has_a_help_page():
    import falcocv.tools  # noqa: F401
    from falcocv.core.tool_registry import get_all_tools
    from falcocv.ui.tool_help import TYPE_TO_HELP_KEY

    for reg in get_all_tools():
        key = TYPE_TO_HELP_KEY.get(reg.tool_type, reg.default_name)
        assert key in TOOL_HELP, (
            f"{reg.tool_type} (default name {reg.default_name!r}) has no help page"
        )
