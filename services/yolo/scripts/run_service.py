#!/usr/bin/env python
"""Development launcher for the standalone YOLO service."""

from __future__ import annotations

import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = SERVICE_ROOT.parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from yolo_service.app import main  # noqa: E402, I001


if __name__ == "__main__":
    main()
