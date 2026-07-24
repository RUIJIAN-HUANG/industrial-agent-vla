#!/usr/bin/env python3
"""Verify that immutable competition baselines still match their frozen bytes."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path


BASELINES = {
    "docs/official/XH-202607_competition_spec.pdf": {
        "size": 308_874,
        "sha256": "FDC21B1C0EDAA48BD2CDE22E5B103F458F5106759ACD4D9C65236549D4695D25",
    },
    "docs/official/XH-202607_official_QA.pdf": {
        "size": 325_783,
        "sha256": "0A381757E35EE402E954CCB34CA0A5453DE4119AABEED1165AFD66666FC05731",
    },
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    failures: list[str] = []

    for relative_path, expected in BASELINES.items():
        path = repo_root / relative_path
        if not path.is_file():
            failures.append(f"MISSING {relative_path}")
            continue

        actual_size = path.stat().st_size
        if actual_size != expected["size"]:
            failures.append(
                f"SIZE {relative_path}: expected {expected['size']}, got {actual_size}"
            )

        actual_hash = sha256(path)
        if actual_hash != expected["sha256"]:
            failures.append(
                f"SHA256 {relative_path}: expected {expected['sha256']}, got {actual_hash}"
            )

    if failures:
        print("Immutable baseline verification FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"Immutable baseline verification passed ({len(BASELINES)} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
