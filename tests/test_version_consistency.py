"""The package version has a single source of truth."""

import re
import tomllib
from pathlib import Path

from falcocv.version import __version__

_ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_reads_the_version_from_version_module():
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = data["project"]

    assert "version" not in project
    assert "version" in project.get("dynamic", [])

    metadata = data["tool"]["scikit-build"]["metadata"]["version"]
    assert metadata["provider"] == "scikit_build_core.metadata.regex"
    assert metadata["input"] == "src/falcocv/version.py"

    source = (_ROOT / metadata["input"]).read_text(encoding="utf-8")
    match = re.search(metadata["regex"], source)
    assert match is not None, "the metadata regex no longer matches version.py"
    assert match.group("value") == __version__
