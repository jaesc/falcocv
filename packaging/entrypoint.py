"""Entry script for the FalcoCV standalone build.

The PyInstaller spec builds two bootloaders (windowed and console) and the
build script copies them to the final names.  Dispatching on the executable
name keeps a single analysis pass, which is what keeps the release build fast.
"""

from __future__ import annotations

import os
import sys


def _dispatch() -> int:
    name = os.path.basename(sys.executable).lower()

    if name.startswith("falcocvserver"):
        from falcocv.web_runner import main

        return main()
    if name.startswith("falcocvcli"):
        from falcocv.cli import main

        return main()
    if name.startswith("falcocvhmi"):
        from falcocv.hmi_designer import main

        main()
        return 0

    from falcocv.main import main

    main()
    return 0


if __name__ == "__main__":
    raise SystemExit(_dispatch())
