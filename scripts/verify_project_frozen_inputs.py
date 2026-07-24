#!/usr/bin/env python3
"""Verify team-frozen diagrams and the archived v1.0 planning snapshot."""

from __future__ import annotations

from pathlib import Path

from frozen_input_verifier import verify_files


PROJECT_FROZEN_INPUTS = {
    "docs/assets/system-architecture-frozen.png": {
        "size": 266_080,
        "sha256": "78BF2B0A1AE5710093E3521EA2A4603537CE14FF826EB3E6D3FD9ED100B249C7",
    },
    "docs/assets/team-roles-frozen.png": {
        "size": 80_761,
        "sha256": "7077AD854067C7110D861F1402FFF33AF5A52A5091AD4A5DB68E5520F99DDDFE",
    },
    "docs/source/XH-202607_initial_plan_v1.0.docx": {
        "size": 472_181,
        "sha256": "4360A1D56F3A48DA83680FF63C15D06FB6F9D893E76EF25A6249493990FC60AB",
    },
}


def main() -> int:
    return verify_files(
        repo_root=Path(__file__).resolve().parents[1],
        label="Project-frozen input",
        expected_files=PROJECT_FROZEN_INPUTS,
    )


if __name__ == "__main__":
    raise SystemExit(main())
