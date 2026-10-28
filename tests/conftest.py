import atexit
import os
import shutil
import sys
import tempfile

# UI tests must never open real windows or touch the user's config.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
_test_config_dir = tempfile.mkdtemp(prefix="falcocv-tests-")
os.environ["FALCOCV_CONFIG_DIR"] = _test_config_dir
# main.py writes its crash log under APPDATA/XDG_CONFIG_HOME.
if sys.platform == "win32":
    os.environ["APPDATA"] = _test_config_dir
else:
    os.environ["XDG_CONFIG_HOME"] = _test_config_dir
atexit.register(shutil.rmtree, _test_config_dir, True)

_repo = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# Source layout: make ``import falcocv`` available without installing.
sys.path.insert(0, os.path.join(_repo, "src"))
# The native kernel lives in cpp/bin during development.
sys.path.insert(0, os.path.join(_repo, "cpp", "bin"))

try:
    import _falco_kernel  # noqa: F401
except ImportError:
    # Editable/wheel installs put the kernel inside the falcocv package.
    from falcocv import _falco_kernel as _kernel
    sys.modules.setdefault("_falco_kernel", _kernel)
