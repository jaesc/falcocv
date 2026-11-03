from PyQt6.QtWidgets import QApplication, QScrollArea

from falcocv.ui.widgets.toolbox import Toolbox


def _process_events():
    QApplication.processEvents()


def test_toolbox_shows_vertical_scrollbar_at_low_height(qtbot):
    toolbox = Toolbox(embedded=True)
    qtbot.addWidget(toolbox)
    scroll = toolbox.findChild(QScrollArea, "ToolboxScroll")
    assert scroll is not None

    toolbox.resize(250, 120)
    toolbox.show()
    _process_events()

    bar = scroll.verticalScrollBar()
    assert bar.maximum() > 0
    assert bar.isVisible()

    toolbox.resize(250, 1200)
    _process_events()

    assert bar.maximum() == 0
    assert not bar.isVisible()
