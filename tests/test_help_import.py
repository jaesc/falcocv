"""Packaging regression: help data must live inside the falcocv package.

The application is launched through the ``falcocv`` console script, whose
``sys.path`` does not include the repository root.  A top-level ``docs``
package therefore works under pytest (cwd = repo root) but fails in a real
launch and in an installed wheel.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def test_help_dialog_imports_from_clean_cwd(tmp_path):
    env = os.environ.copy()
    # Deliberately *replace* PYTHONPATH: if the repository root leaks into
    # sys.path, the legacy top-level ``docs`` package would make this test
    # pass for the wrong reason.
    env["PYTHONPATH"] = str(SRC)

    code = (
        "from falcocv.ui.tool_help import TOOL_HELP, TYPE_TO_HELP_KEY; "
        "from falcocv.ui.widgets.help_dialog import ToolHelpDialog; "
        "assert 'Image Loader' in TOOL_HELP; "
        "assert TYPE_TO_HELP_KEY['ImageLoader'] == 'Image Loader'; "
        "print('ok')"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert "ok" in result.stdout
