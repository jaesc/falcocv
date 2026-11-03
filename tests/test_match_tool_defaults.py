"""Default search ranges for the template and shape matchers."""

from falcocv.tools.shape_match.logic import ShapeMatch
from falcocv.tools.template_match.logic import TemplateMatch


def test_shape_match_default_search_ranges():
    tool = ShapeMatch()

    assert tool.params["angle_start"] == -3.0
    assert tool.params["angle_end"] == 3.0
    assert tool.params["scale_start"] == 1.0
    assert tool.params["scale_end"] == 1.0


def test_template_match_default_search_ranges():
    tool = TemplateMatch()

    assert tool.params["angle_start"] == -3
    assert tool.params["angle_end"] == 3
    assert tool.params["scale_start"] == 1.0
    assert tool.params["scale_end"] == 1.0
