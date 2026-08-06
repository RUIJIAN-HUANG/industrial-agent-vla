#!/usr/bin/env python3
"""Fail CI when repository-only files violate the competition repository policy."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_TRACKED_FILE_BYTES = 10 * 1024 * 1024

REQUIRED_FILES = (
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
    "CONTRIBUTING.md",
    "README.md",
    "configs/README.md",
    "data/README.md",
    "docs/README.md",
    "docs/architecture/ADR-0003-yolo-scoring-sidecar.md",
    "docs/architecture/agent-framework.md",
    "docs/project-management/data-collection-and-five-member-execution-guide.md",
    "docs/repository-structure.md",
    "experiments/README.md",
    "models/MANIFEST.md",
    "models/README.md",
    "reports/evidence-index.md",
    "reports/README.md",
    "schemas/README.md",
    "scripts/README.md",
    "services/README.md",
    "services/openvla_oft/README.md",
    "services/pi05/README.md",
    "services/yolo/README.md",
    "simulation/README.md",
    "src/README.md",
)

FORBIDDEN_SUFFIXES = {
    ".bag",
    ".ckpt",
    ".db3",
    ".engine",
    ".h5",
    ".hdf5",
    ".mcap",
    ".mkv",
    ".mov",
    ".mp4",
    ".npy",
    ".npz",
    ".onnx",
    ".pb",
    ".pem",
    ".pt",
    ".pth",
    ".safetensors",
    ".tflite",
}

ALLOWED_BINARY_FIXTURES = {
    Path("tests/fixtures/golden_episode_v1/episode.h5"),
}

FORBIDDEN_EXACT_NAMES = {
    ".env",
    ".pypirc",
    ".DS_Store",
    "Thumbs.db",
    "id_dsa",
    "id_ecdsa",
    "id_ed25519",
    "id_rsa",
}

FORBIDDEN_DIRECTORY_NAMES = {
    ".cache",
    ".idea",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".venv",
    ".vscode",
    "__pycache__",
    "artifacts",
    "checkpoints",
    "logs",
    "mlruns",
    "outputs",
    "runs",
    "wandb",
}

SECRET_FILE_PREFIXES = (
    "credentials",
    "secret",
)


def _tracked_files() -> tuple[Path, ...]:
    result = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "-z"],
        check=True,
        capture_output=True,
    )
    paths = result.stdout.decode("utf-8").split("\0")
    return tuple(Path(path) for path in paths if path)


def _is_secret_filename(path: Path) -> bool:
    name = path.name.lower()
    if name in {item.lower() for item in FORBIDDEN_EXACT_NAMES}:
        return True
    return any(name.startswith(prefix) for prefix in SECRET_FILE_PREFIXES)


def find_policy_violations() -> list[str]:
    violations: list[str] = []

    for relative in REQUIRED_FILES:
        if not (ROOT / relative).is_file():
            violations.append(f"missing required repository file: {relative}")

    for relative in _tracked_files():
        absolute = ROOT / relative
        lowered_parts = {part.lower() for part in relative.parts[:-1]}

        if not absolute.is_file():
            violations.append(
                f"tracked path is missing or not a regular file: {relative}"
            )
            continue

        if (
            relative.suffix.lower() in FORBIDDEN_SUFFIXES
            and relative not in ALLOWED_BINARY_FIXTURES
        ):
            violations.append(
                f"forbidden binary/artifact extension is tracked: {relative}"
            )

        if _is_secret_filename(relative):
            violations.append(f"credential or secret-like file is tracked: {relative}")

        forbidden_directories = lowered_parts & FORBIDDEN_DIRECTORY_NAMES
        if forbidden_directories:
            names = ", ".join(sorted(forbidden_directories))
            violations.append(
                f"generated/private directory is tracked ({names}): {relative}"
            )

        size = absolute.stat().st_size
        if size > MAX_TRACKED_FILE_BYTES:
            mib = size / (1024 * 1024)
            violations.append(
                f"tracked file exceeds 10 MiB ({mib:.1f} MiB): {relative}"
            )

    return violations


def main() -> int:
    try:
        violations = find_policy_violations()
    except (OSError, subprocess.CalledProcessError, UnicodeDecodeError) as exc:
        print(f"[FAIL] repository hygiene check could not run: {exc}", file=sys.stderr)
        return 2

    if violations:
        print("[FAIL] repository hygiene violations:", file=sys.stderr)
        for violation in violations:
            print(f"  - {violation}", file=sys.stderr)
        return 1

    tracked_count = len(_tracked_files())
    print(
        "[PASS] repository hygiene: "
        f"{tracked_count} tracked files, no forbidden artifacts or missing markers"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
