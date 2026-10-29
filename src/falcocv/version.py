"""FalcoCV release versioning.

Single source of truth for version numbers.
Updated by the build script and git tags.
"""

import datetime

__version__ = "1.0.1"
__build__ = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
__semver__ = f"{__version__}+{__build__}"

RELEASE_LABEL = f"v{__version__}"


def as_full_label(build_date: str = "") -> str:
    label = f"FalcoCV {RELEASE_LABEL}"
    if build_date:
        label += f" ({build_date})"
    return label
