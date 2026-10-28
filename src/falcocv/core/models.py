"""Model resolution and first-run download for FalcoCV.

Models are not shipped in the wheel/sdist (PyPI file-size limits). They are
downloaded on first run to the user data directory:

    Windows: %APPDATA%/FalcoCV/models
    Linux:   $XDG_DATA_HOME/FalcoCV/models (or ~/.local/share/FalcoCV/models)

During development, an existing ``<repo>/models`` directory is used as-is so
working offline is possible.

The download base URL can be overridden with ``FALCOCV_MODELS_URL`` (useful for
mirrors / release candidates) and ``FALCOCV_MODELS_DIR`` can point at an
existing local bundle. The expected file names and sizes are kept in
``MODEL_FILES``; a size mismatch triggers a re-download.
"""
from __future__ import annotations

import os
import shutil
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable

# GitHub Release that hosts the model bundle. Override with FALCOCV_MODELS_URL.
DEFAULT_MODELS_URL = "https://github.com/jaesc/falcocv/releases/download/models-v1"
_USER_AGENT = "FalcoCV-model-downloader/1.0"

MODEL_FILES: dict[str, int] = {
    "EfficientNet-B0.onnx": 558030,
    "EfficientNet-B0.onnx.data": 15990784,
    "EfficientNet-B0-MultiLayer.onnx": 425118,
    "EfficientNet-B0-MultiLayer.onnx.data": 3452160,
    "ResNet18-Layer2.onnx": 2734432,
    "ResNet18-Layer3.onnx": 11131054,
    "ch_PP-OCRv5_rec_mobile.onnx": 16631306,
    "ch_PP-OCRv5_rec_server.onnx": 84577022,
    "ppocrv5_dict.txt": 74012,
}


MODEL_SIZE_ALIASES: dict[str, set[int]] = {
    # Git on Windows may check the dictionary out with CRLF.
    "ppocrv5_dict.txt": {74012, 92395},
}


def _size_ok(name: str, size: int) -> bool:
    expected = MODEL_FILES[name]
    return size in MODEL_SIZE_ALIASES.get(name, {expected})


class ModelDownloadError(RuntimeError):
    pass


def _dev_models_dir() -> Path | None:
    """Return the source-tree models directory when running from a checkout."""
    candidate = Path(__file__).resolve().parents[3] / "models"
    return candidate if candidate.is_dir() else None


def models_dir() -> Path:
    env = os.environ.get("FALCOCV_MODELS_DIR")
    if env:
        return Path(env).expanduser().resolve()

    # Development checkout: keep using the bundled local models.
    dev = _dev_models_dir()
    if dev is not None and any(dev.iterdir()):
        return dev

    try:
        from falcocv.core.environment import APP_DATA
        base = APP_DATA
    except Exception:
        base = Path.home() / ".falcocv"
    return (base / "models").resolve()


def model_path(name: str) -> Path:
    return models_dir() / name


def missing_models() -> list[str]:
    root = models_dir()
    missing = []
    for name, expected_size in MODEL_FILES.items():
        path = root / name
        if not path.is_file() or not _size_ok(name, path.stat().st_size):
            missing.append(name)
    return missing


def _download_one(url: str, dest: Path, expected_size: int,
                  progress: Callable[[int, int], None] | None = None) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as response, open(tmp, "wb") as fh:
        total = int(response.headers.get("Content-Length", expected_size) or expected_size)
        done = 0
        while True:
            chunk = response.read(1024 * 256)
            if not chunk:
                break
            fh.write(chunk)
            done += len(chunk)
            if progress is not None:
                progress(done, total)
    size = tmp.stat().st_size
    if expected_size and size != expected_size:
        tmp.unlink(missing_ok=True)
        raise ModelDownloadError(
            f"{dest.name}: expected {expected_size} bytes, got {size}"
        )
    tmp.replace(dest)


def ensure_models(progress: Callable[[str, int, int, int, int], None] | None = None,
                  retries: int = 3) -> Path:
    """Ensure every model file exists, downloading missing/corrupt ones.

    ``progress`` receives ``(filename, downloaded, total, index, count)``.
    Raises ``ModelDownloadError`` if a file cannot be downloaded after retries.
    """
    root = models_dir()
    pending = []
    for name, size in MODEL_FILES.items():
        path = root / name
        if not path.is_file() or not _size_ok(name, path.stat().st_size):
            pending.append(name)

    if not pending:
        return root

    base_url = os.environ.get("FALCOCV_MODELS_URL", DEFAULT_MODELS_URL).rstrip("/")
    if not base_url:
        raise ModelDownloadError("FALCOCV_MODELS_URL is empty")

    for index, name in enumerate(pending, start=1):
        url = f"{base_url}/{name}"
        last_error: Exception | None = None
        for attempt in range(1, retries + 1):
            try:
                def _cb(done: int, total: int, _name=name, _index=index):
                    if progress is not None:
                        progress(_name, done, total, _index, len(pending))

                _download_one(url, root / name, MODEL_FILES[name], _cb)
                last_error = None
                break
            except Exception as exc:  # network, HTTP, size mismatch
                last_error = exc
                if attempt < retries:
                    time.sleep(attempt)
        if last_error is not None:
            raise ModelDownloadError(
                f"Could not download {name!r} from {url}: {last_error}"
            ) from last_error
    return root


def download_models_cli() -> int:
    """Console entry point: falcocv download-models."""
    root = ensure_models()
    print(f"Models ready at: {root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(download_models_cli())
