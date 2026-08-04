"""Convert one Canonical HDF5 episode to OpenVLA-OFT RLDS-style files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SERVICE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from openvla_oft.canonical import load_openvla_arm_b_steps  # noqa: E402
from openvla_oft.rlds import (  # noqa: E402
    load_rlds_style_episode,
    write_rlds_style_episode,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--episode",
        required=True,
        type=Path,
        help="Canonical episode directory containing episode.h5 and structure.json.",
    )
    parser.add_argument(
        "--split-registry",
        required=True,
        type=Path,
        help="Verified Split Registry; the Episode must be assigned to Train.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Absent destination directory for metadata.json, steps.jsonl, arrays.npz.",
    )
    args = parser.parse_args()

    from industrial_agent.data import SplitRegistry

    split_registry = SplitRegistry.load(args.split_registry)
    steps = load_openvla_arm_b_steps(
        args.episode,
        split_registry=split_registry,
    )
    export_dir = write_rlds_style_episode(steps, args.output_dir)
    summary = load_rlds_style_episode(export_dir)["metadata"]
    print(json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
