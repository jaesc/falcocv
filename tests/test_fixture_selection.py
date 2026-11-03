"""Partial fixture application: choose which components a match anchors."""

import numpy as np
import pytest

from falcocv.core.vision_image import VisionImage


def _image():
    return VisionImage(
        np.zeros((20, 20), dtype=np.uint8),
        x=1.0,
        y=2.0,
        angle=3.0,
    )


def test_apply_fixture_copies_all_components():
    image = _image()
    image.apply_fixture(10.0, 20.0, 30.0)
    assert image.x == 10.0
    assert image.y == 20.0
    assert image.angle == 30.0


def test_apply_fixture_keeps_unselected_components():
    image = _image()
    image.apply_fixture(10.0, 20.0, 30.0, use_y=False)
    assert image.x == 10.0
    assert image.y == 2.0
    assert image.angle == 30.0


def test_apply_fixture_can_skip_everything():
    image = _image()
    image.apply_fixture(
        10.0, 20.0, 30.0, use_x=False, use_y=False, use_angle=False
    )
    assert image.x == 1.0
    assert image.y == 2.0
    assert image.angle == 3.0


def test_apply_fixture_applies_the_angle_offset_and_wraps():
    image = _image()
    image.apply_fixture(0.0, 0.0, 170.0, angle_offset=20.0)
    assert image.angle == pytest.approx(-170.0)


def test_apply_fixture_ignores_the_offset_when_the_angle_is_not_used():
    image = _image()
    image.apply_fixture(0.0, 0.0, 170.0, use_angle=False, angle_offset=20.0)
    assert image.angle == 3.0


@pytest.mark.parametrize(
    "module_name, class_name",
    [
        ("falcocv.tools.shape_match.logic", "ShapeMatch"),
        ("falcocv.tools.template_match.logic", "TemplateMatch"),
        ("falcocv.tools.barcode_decoder.logic", "BarcodeDecoder"),
    ],
)
def test_tools_default_to_all_components(module_name, class_name):
    import importlib

    module = importlib.import_module(module_name)
    tool = getattr(module, class_name)()
    assert tool.params["fixture_use_x"] is True
    assert tool.params["fixture_use_y"] is True
    assert tool.params["fixture_use_angle"] is True
