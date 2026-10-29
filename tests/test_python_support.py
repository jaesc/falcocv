"""Guard the supported Python range used by the published wheel."""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_only_python_313_is_supported():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert data["project"]["requires-python"] == ">=3.13,<3.14"
    assert data["tool"]["cibuildwheel"]["build"] == "cp313-*"
