"""FalcoCV core package — workflow engine, tool base, Python core."""

import os
import sys
from pathlib import Path

# Minimal pybind11 kernel (_falco_kernel.pyd) lives in cpp/bin/ during
# development and inside the falcocv package in installed wheels.
_bin = str(Path(__file__).resolve().parents[3] / "cpp" / "bin")
if os.path.isdir(_bin):
    if _bin not in sys.path:
        sys.path.insert(0, _bin)
    _old_path = os.environ.get("PATH", "")
    if _bin not in _old_path.split(os.pathsep):
        os.environ["PATH"] = _bin + os.pathsep + _old_path

# Make ``from _falco_kernel import ...`` work for installed wheels too.
# Prefer the freshly built development kernel when it is available.
try:
    import _falco_kernel as _kernel
except ImportError:
    try:
        from falcocv import _falco_kernel as _kernel
    except ImportError:
        _kernel = None
if _kernel is not None:
    sys.modules.setdefault("_falco_kernel", _kernel)
