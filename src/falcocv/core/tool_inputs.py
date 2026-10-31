"""Validation helpers shared by vision tools."""

from __future__ import annotations

import numpy as np

from falcocv.core.vision_image import VisionImage


def image_input_error(value, label: str = "Input image") -> str | None:
    """Return an error message when ``value`` is not a usable image, else None.

    Tools receive ``VisionImage`` objects or raw numpy arrays.  A missing,
    empty or non-image input must fail the tool instead of silently producing
    an empty output.
    """
    if value is None:
        return f"{label} is missing"
    data = value.data if isinstance(value, VisionImage) else value
    if data is None:
        return f"{label} is missing"
    if not isinstance(data, np.ndarray):
        return f"{label} is not a valid image"
    if data.ndim not in (2, 3) or data.size == 0:
        return f"{label} is not a valid image"
    return None
