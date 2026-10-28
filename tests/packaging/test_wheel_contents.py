"""Packaging smoke test: the built wheel must contain its runtime data."""

import glob
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
WHEELS = sorted(glob.glob(str(ROOT / "dist" / "*.whl")))


@pytest.mark.skipif(not WHEELS, reason="no wheel in dist/ (build it first)")
def test_wheel_contains_runtime_files():
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "check_wheel.py"), WHEELS[-1]],
        cwd=ROOT,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "[PASS]" in result.stdout
