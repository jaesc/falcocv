from .logic import BlobDetector

try:
    from .config import BlobDetectorConfig
except ImportError:
    pass
