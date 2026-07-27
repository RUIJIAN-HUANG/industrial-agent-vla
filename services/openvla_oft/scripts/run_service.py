#!/usr/bin/env python
"""Development starter for the OpenVLA-OFT Arm_B service."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from openvla_oft.app import main

if __name__ == "__main__":
    os.environ.setdefault("OPENVLA_OFT_USE_MOCK", "1")
    main()
