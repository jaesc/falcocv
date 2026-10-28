from .logic import AnomalyDetector

try:
    from .config import AnomalyDetectorConfig
except ImportError:
    pass
