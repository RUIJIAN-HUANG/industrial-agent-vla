"""Build the immutable local OpenVLA-OFT checkpoint manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=Path)
    parser.add_argument(
        "--output",
        default="checkpoint.manifest.json",
        help="Manifest path relative to checkpoint_dir.",
    )
    args = parser.parse_args()
    root = args.checkpoint_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"checkpoint_dir does not exist: {root}")
    output = (root / args.output).resolve()
    try:
        output.relative_to(root)
    except ValueError:
        parser.error("--output must stay inside checkpoint_dir")

    files = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path == output or ".git" in path.parts:
            continue
        files.append(
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
        )
    if not files:
        parser.error("checkpoint_dir contains no checkpoint files")
    output.write_text(
        json.dumps(
            {"schema_version": "1.0", "files": files},
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"manifest={output}")
    print(f"checkpoint_sha=sha256:{sha256_file(output)}")
    norm_stats = root / "dataset_statistics.json"
    if norm_stats.is_file():
        print(f"norm_stats_sha=sha256:{sha256_file(norm_stats)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
