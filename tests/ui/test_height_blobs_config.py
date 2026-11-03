"""Height Blobs 3D config: tabs, mask view and shared record selector."""

from falcocv.tools.height_blobs.config import HeightBlobs3DConfig
from falcocv.tools.height_blobs.logic import HeightBlobs3D


def test_config_has_parameters_and_results_tabs(qtbot):
    tool = HeightBlobs3D()
    config = HeightBlobs3DConfig(tool)
    qtbot.addWidget(config)

    assert config.tabs.count() == 2
    assert config.tabs.tabText(0) == "Parameters"
    assert config.tabs.tabText(1) == "Results"


def test_binary_mask_hides_the_roi(qtbot):
    tool = HeightBlobs3D()
    config = HeightBlobs3DConfig(tool)
    qtbot.addWidget(config)

    index = config.record_selector.findData("binary mask")
    assert index >= 0
    config.record_selector.setCurrentIndex(index)

    assert config.viewer._roi_data is None
    assert not config.roi_editor.btn_draw.isChecked()


def test_record_selector_is_wide_enough(qtbot):
    tool = HeightBlobs3D()
    config = HeightBlobs3DConfig(tool)
    qtbot.addWidget(config)

    assert config.record_selector.minimumWidth() >= 285


def test_parameter_page_fits_a_narrow_panel(qtbot):
    tool = HeightBlobs3D()
    config = HeightBlobs3DConfig(tool)
    qtbot.addWidget(config)
    config.resize(1200, 800)
    config.show()
    qtbot.wait(20)

    assert config.params_page.minimumSizeHint().width() <= 240
    page = config.tabs.widget(0)
    assert page.horizontalScrollBar().isVisible() is False
