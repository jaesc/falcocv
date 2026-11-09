"""The fixture axis checkboxes follow the 'act as fixture' toggle."""

import pytest

from falcocv.tools.barcode_decoder.config import BarcodeDecoderConfig
from falcocv.tools.barcode_decoder.logic import BarcodeDecoder
from falcocv.tools.shape_match.config import ShapeMatchConfig
from falcocv.tools.shape_match.logic import ShapeMatch
from falcocv.tools.template_match.config import TemplateMatchConfig
from falcocv.tools.template_match.logic import TemplateMatch

CASES = [
    (ShapeMatch, ShapeMatchConfig),
    (TemplateMatch, TemplateMatchConfig),
    (BarcodeDecoder, BarcodeDecoderConfig),
]


@pytest.mark.parametrize(
    "tool_cls, config_cls", CASES, ids=lambda item: item.__name__
)
def test_fixture_axes_are_enabled_only_with_the_toggle(
    qtbot, tool_cls, config_cls
):
    tool = tool_cls()
    config = config_cls(tool)
    qtbot.addWidget(config)

    assert len(config.fixture_axes) == 3
    assert all(box.isChecked() for box in config.fixture_axes)
    assert not any(box.isEnabled() for box in config.fixture_axes)

    config.chk_fixture.setChecked(True)
    assert all(box.isEnabled() for box in config.fixture_axes)

    config.fixture_axes[1].setChecked(False)
    assert tool.params["fixture_use_y"] is False
    config.fixture_axes[1].setChecked(True)
    assert tool.params["fixture_use_y"] is True


@pytest.mark.parametrize(
    "tool_cls, config_cls",
    [(ShapeMatch, ShapeMatchConfig), (TemplateMatch, TemplateMatchConfig)],
    ids=["shape", "template"],
)
def test_match_fixture_is_a_dedicated_section(qtbot, tool_cls, config_cls):
    config = config_cls(tool_cls())
    qtbot.addWidget(config)

    assert config.chk_fixture.text() == "Act as Fixture"
    assert config.chk_fixture.parent().title() == "Fixture"
