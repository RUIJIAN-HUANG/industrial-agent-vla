#!/usr/bin/env python3
"""Verify the two official competition PDFs, the only official sources of truth."""

from __future__ import annotations

from pathlib import Path

from frozen_input_verifier import verify_files


OFFICIAL_SOURCES = {
    "docs/official/XH-202607_competition_spec.pdf": {
        "size": 308_874,
        "sha256": "FDC21B1C0EDAA48BD2CDE22E5B103F458F5106759ACD4D9C65236549D4695D25",
    },
    "docs/official/XH-202607_official_QA.pdf": {
        "size": 325_783,
        "sha256": "0A381757E35EE402E954CCB34CA0A5453DE4119AABEED1165AFD66666FC05731",
    },
}


def main() -> int:
    return verify_files(
        repo_root=Path(__file__).resolve().parents[1],
        label="Official source",
        expected_files=OFFICIAL_SOURCES,
    )


if __name__ == "__main__":
    raise SystemExit(main())
