from falcocv.ui.metrology_config import MetrologyFitConfigBase


class LineFitConfig(MetrologyFitConfigBase):
    TOOL_KEY = "LineFit"
    FIT_LABEL = "Show Fitted Line"
    RESULT_NOUN = "line"
    MAX_RESULTS = 10
