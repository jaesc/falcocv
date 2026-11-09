from falcocv.ui.metrology_config import MetrologyFitConfigBase


class CircleFitConfig(MetrologyFitConfigBase):
    TOOL_KEY = "CircleFit"
    IS_CIRCLE = True
    FIT_LABEL = "Show Fitted Circle"
    RESULT_NOUN = "circle"
    MAX_RESULTS = 10
