#!/usr/bin/env python
"""Development launcher for the standalone YOLO service."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from yolo_service.app import main  # noqa: E402, I001


if __name__ == "__main__":
    main()
