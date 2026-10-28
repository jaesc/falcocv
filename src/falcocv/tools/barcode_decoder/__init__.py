from .logic import BarcodeDecoder, CodeResult

try:
    from .config import BarcodeDecoderConfig
except ImportError:
    pass
