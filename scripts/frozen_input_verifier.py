"""Shared SHA-256 verifier for versioned competition inputs."""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Mapping


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def verify_files(
    *,
    repo_root: Path,
    label: str,
    expected_files: Mapping[str, Mapping[str, int | str]],
) -> int:
    failures: list[str] = []

    for relative_path, expected in expected_files.items():
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
        print(f"{label} verification FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)
        return 1

    print(f"{label} verification passed ({len(expected_files)} files).")
    return 0
