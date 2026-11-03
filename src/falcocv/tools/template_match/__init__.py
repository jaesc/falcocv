from .logic import TemplateMatch

try:
    from .config import TemplateMatchConfig
except ImportError:
    pass
