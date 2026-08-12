"""Pure helpers shared by the formal V2 keyboard collection entry."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from typing import Any

from industrial_agent.data import CanonicalRecorder, EpisodeMetadata
from industrial_agent.image_cas import ImageCas, ImageCasConfig
from simulation.v2_collection_preflight import (
    CollectionPreflight,
    CollectionSplit,
    build_collection_preflight,
)


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
DEFAULT_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visible V2 keyboard collection with Canonical recording."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--episode-root", type=Path, required=True)
    parser.add_argument("--cas-root", type=Path, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-scene", type=Path, required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--scene-seed", type=int, required=True)
    parser.add_argument(
        "--split",
        choices=tuple(item.value for item in CollectionSplit),
        required=True,
    )
    parser.add_argument("--franka-usd")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--translation-step-m", type=float, default=0.005)
    parser.add_argument("--rotation-step-deg", type=float, default=2.0)
    parser.add_argument("--headless", action="store_true")
    return parser


def git_identity(repository_root: Path = REPOSITORY_ROOT) -> tuple[str, bool]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repository_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return head, not bool(status.strip())


def preflight_from_args(
    args: argparse.Namespace,
    *,
    git_sha: str,
    worktree_clean: bool,
) -> CollectionPreflight:
    if not 1 <= args.max_actions <= 5000:
        raise ValueError("--max-actions must be in [1, 5000]")
    if not 0.0 < args.translation_step_m <= 0.01:
        raise ValueError("--translation-step-m must be in (0, 0.01]")
    if not 0.0 < args.rotation_step_deg <= 5.0:
        raise ValueError("--rotation-step-deg must be in (0, 5]")
    return build_collection_preflight(
        config_path=args.config,
        episode_root=args.episode_root,
        cas_root=args.cas_root,
        episode_id=args.episode_id,
        task_id=args.task_id,
        instruction=args.instruction,
        scene_seed=args.scene_seed,
        split=args.split,
        headless=args.headless,
        git_sha=git_sha,
        worktree_clean=worktree_clean,
    )


def create_recorder(
    preflight: CollectionPreflight,
) -> tuple[ImageCas, CanonicalRecorder]:
    image_cas = ImageCas(ImageCasConfig(root=preflight.cas_root))
    image_cas.assert_ready(writable=True)
    metadata = EpisodeMetadata(
        episode_id=preflight.episode_id,
        task_id=preflight.task_id,
        instruction=preflight.instruction,
        scene_seed=preflight.scene_seed,
        git_sha=preflight.git_sha,
        scene_config_sha256=preflight.scene_config_sha256,
        scene_id=preflight.scene_id,
    )
    return image_cas, CanonicalRecorder(
        preflight.episode_root,
        metadata,
        image_cas=image_cas,
    )


def write_result(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str) + "\n",
        encoding="utf-8",
    )


def preflight_payload(preflight: CollectionPreflight) -> dict[str, Any]:
    payload = asdict(preflight)
    payload["split"] = preflight.split.value
    return payload
