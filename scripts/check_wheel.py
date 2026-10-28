"""Validate the contents of a built FalcoCV wheel.

Usage:
    python scripts/check_wheel.py dist/falcocv-*.whl
"""

from __future__ import annotations

import glob
import sys
import zipfile
from pathlib import Path


def check_wheel(path: Path) -> list[str]:
    errors: list[str] = []
    with zipfile.ZipFile(path) as wheel:
        names = wheel.namelist()
        metadata_name = next(
            (name for name in names if name.endswith(".dist-info/METADATA")), None
        )
        metadata = (
            wheel.read(metadata_name).decode("utf-8", "replace")
            if metadata_name
            else ""
        )

    def has(name: str) -> bool:
        return name in names

    def has_prefix(prefix: str) -> bool:
        return any(name.startswith(prefix) for name in names)

    if not has("falcocv/ui/tool_help.py"):
        errors.append("missing falcocv/ui/tool_help.py (help data)")
    if not has("falcocv/self_test.py"):
        errors.append("missing falcocv/self_test.py")
    if not has("falcocv/run_job.py"):
        errors.append("missing falcocv/run_job.py (headless runner)")
    if not has("falcocv/core/results.py"):
        errors.append("missing falcocv/core/results.py (result contract)")
    if not has("falcocv/tools/shape_match/logic.py"):
        errors.append("missing falcocv/tools/shape_match/logic.py")
    if not any("_falco_kernel" in name for name in names):
        errors.append("missing the native _falco_kernel extension")
    if has_prefix("docs/"):
        errors.append("legacy top-level docs/ package must not be packaged")
    if has_prefix("models/") or has_prefix("falcocv/models/"):
        errors.append("models/ must not be packaged (use the models release)")
    if not any(name.startswith("falcocv/_falco_kernel") for name in names):
        errors.append("kernel is not inside the falcocv package")

    def fields(prefix: str) -> list[str]:
        return [
            line.split(":", 1)[1].strip().lower()
            for line in metadata.splitlines()
            if line.lower().startswith(prefix)
        ]

    if not metadata:
        errors.append("missing wheel METADATA")
    else:
        requirements = fields("requires-dist:")
        if not any(item.startswith("harvesters") for item in requirements):
            errors.append("harvesters must be a runtime dependency")
        if any(item == "gige" for item in fields("provides-extra:")):
            errors.append("gige extra must not exist (harvesters is installed by default)")

    return errors


def main(argv=None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    paths = [Path(p) for p in args] if args else [
        Path(p) for p in sorted(glob.glob("dist/*.whl"))
    ]
    if not paths:
        print("No wheel found. Build one with: python -m build --wheel")
        return 1

    failed = False
    for path in paths:
        errors = check_wheel(path)
        if errors:
            failed = True
            print(f"[FAIL] {path}")
            for error in errors:
                print(f"  - {error}")
        else:
            print(f"[PASS] {path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
