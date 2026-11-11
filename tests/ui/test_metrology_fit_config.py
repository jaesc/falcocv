"""Radius limits in the shared metrology fit config."""

import pytest

from falcocv.tools.circle_fit.config import CircleFitConfig
from falcocv.tools.circle_fit.logic import CircleFit
from falcocv.tools.line_fit.config import LineFitConfig
from falcocv.tools.line_fit.logic import LineFit


pytestmark = pytest.mark.ui


def test_circle_fit_config_edits_the_radius_limits(qtbot):
    tool = CircleFit()
    config = CircleFitConfig(tool)
    qtbot.addWidget(config)

    config.spin_min_radius.setValue(12.5)
    config.spin_max_radius.setValue(80.0)

    assert tool.params["min_radius"] == pytest.approx(12.5)
    assert tool.params["max_radius"] == pytest.approx(80.0)


def test_line_fit_config_has_no_radius_limits(qtbot):
    tool = LineFit()
    config = LineFitConfig(tool)
    qtbot.addWidget(config)

    assert not hasattr(config, "spin_min_radius")
    assert not hasattr(config, "spin_max_radius")
