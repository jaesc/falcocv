from .logic import PythonScriptTool

try:
    from .config import PythonScriptConfig
except ImportError:
    pass
