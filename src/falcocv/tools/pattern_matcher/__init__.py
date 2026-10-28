from .logic import PatternMatcher

try:
    from .config import PatternMatcherConfig
except ImportError:
    pass
