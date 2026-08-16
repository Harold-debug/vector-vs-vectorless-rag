#!/usr/bin/env python3
"""Launch the Gradio UI without relying on editable installs.

macOS/iCloud marks editable .pth files under Documents as UF_HIDDEN;
Python 3.13 skips those files, which causes `No module named vvvrag`.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from vvvrag.cli import main

if __name__ == "__main__":
    # Default to `ui` when no args are given.
    if len(sys.argv) == 1:
        sys.argv.append("ui")
    main()
