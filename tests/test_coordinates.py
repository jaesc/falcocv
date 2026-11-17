"""Coordinate calibration/fixture helpers."""

import numpy as np

from falcocv.core.coordinates import (
    fixture_to_world_xy,
    fixture_transform,
    world_to_fixture_xy,
)


class _Image:
    def __init__(self, x=0.0, y=0.0, angle=0.0, scale_x=1.0, scale_y=1.0):
        self.x = x
        self.y = y
        self.angle = angle
        self.scale_x = scale_x
        self.scale_y = scale_y


def test_identity_fixture_returns_world_coordinates():
    transform = fixture_transform(_Image(), 100)
    x, y = world_to_fixture_xy([12.0], [-4.0], transform)

    assert transform is None
    assert x[0] == 12.0
    assert y[0] == -4.0


def test_fixture_round_trip():
    image = _Image(x=30.0, y=40.0, angle=27.0, scale_x=0.5, scale_y=0.25)
    transform = fixture_transform(image, 200)
    world = np.array([[12.0, -4.0], [80.0, 20.0]], dtype=np.float64)

    fx, fy = world_to_fixture_xy(world[:, 0], world[:, 1], transform)
    wx, wy = fixture_to_world_xy(fx, fy, transform)

    assert np.allclose(wx, world[:, 0])
    assert np.allclose(wy, world[:, 1])


def test_fixture_origin_maps_to_zero():
    image = _Image(x=30.0, y=40.0, angle=-15.0, scale_x=0.25, scale_y=0.25)
    transform = fixture_transform(image, 100)
    origin_x = (100 - 1 - image.x) * image.scale_x
    origin_y = image.y * image.scale_y

    fx, fy = world_to_fixture_xy([origin_x], [origin_y], transform)

    assert fx[0] == 0.0
    assert fy[0] == 0.0
