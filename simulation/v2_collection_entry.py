"""Pure helpers shared by the formal V2 keyboard collection entry."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import subprocess
from typing import Any

from industrial_agent.data import CanonicalV2EpisodeMetadata, CanonicalV2Recorder
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
    parser.add_argument("--frozen-collection-sha")
    parser.add_argument("--expected-scene-config-sha256")
    parser.add_argument("--openpi-root", type=Path)
    parser.add_argument(
        "--split",
        choices=tuple(item.value for item in CollectionSplit),
        required=True,
    )
    parser.add_argument("--franka-usd")
    parser.add_argument("--max-actions", type=int, default=500)
    parser.add_argument("--translation-step-m", type=float, default=0.05)
    parser.add_argument("--fine-translation-step-m", type=float, default=0.005)
    parser.add_argument("--rotation-step-deg", type=float, default=5.0)
    parser.add_argument(
        "--replay-episode",
        type=Path,
        help=(
            "Strict Canonical V2 episode used as the action source. "
            "When set, keyboard input is disabled and the recorded task actions "
            "are replayed automatically."
        ),
    )
    parser.add_argument(
        "--trajectory-profile",
        choices=("baseline", "diverse_low", "approach_curve"),
        default="baseline",
        help=(
            "Replay profile. baseline preserves the source actions; "
            "diverse_low applies a smooth, small lift-path variation; "
            "approach_curve adds a small pre-grasp approach arc."
        ),
    )
    parser.add_argument(
        "--trajectory-seed",
        type=int,
        default=0,
        help="Deterministic seed for the selected trajectory profile.",
    )
    parser.add_argument(
        "--trajectory-variant",
        type=int,
        default=0,
        help="Explicit variant index for non-baseline replay profiles.",
    )
    parser.add_argument(
        "--lift-mm",
        type=float,
        default=None,
        help=(
            "Explicit diverse_low lift amplitude in millimetres. "
            "When set, this overrides the seed-selected amplitude."
        ),
    )
    parser.add_argument(
        "--final-y-offset-mm",
        type=float,
        default=0.0,
        help="Smooth final placement correction along bin-local Y, in millimetres.",
    )
    parser.add_argument(
        "--final-z-offset-mm",
        type=float,
        default=0.0,
        help="Smooth final placement correction along bin-local Z, in millimetres.",
    )
    parser.add_argument(
        "--ik-backend",
        choices=("pink", "lula"),
        default="pink",
        help="Live IK backend; Pink adds null-space posture regularization.",
    )
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
    if not 0.0 < args.translation_step_m <= 0.05:
        raise ValueError("--translation-step-m must be in (0, 0.05]")
    if not 0.0 < args.fine_translation_step_m <= args.translation_step_m:
        raise ValueError(
            "--fine-translation-step-m must be positive and no larger than "
            "--translation-step-m"
        )
    if not 0.0 < args.rotation_step_deg <= 5.0:
        raise ValueError("--rotation-step-deg must be in (0, 5]")
    if args.trajectory_profile != "baseline" and args.replay_episode is None:
        raise ValueError("a trajectory profile requires --replay-episode")
    if args.lift_mm is not None and not 0.0 < args.lift_mm <= 5.0:
        raise ValueError("--lift-mm must be in (0, 5] millimetres")
    if args.lift_mm is not None and args.trajectory_profile != "diverse_low":
        raise ValueError("--lift-mm requires --trajectory-profile diverse_low")
    if (
        args.trajectory_profile == "approach_curve"
        and not 1 <= args.trajectory_variant <= 4
    ):
        raise ValueError("approach_curve requires --trajectory-variant in [1, 4]")
    if args.trajectory_profile != "approach_curve" and args.trajectory_variant != 0:
        raise ValueError("--trajectory-variant is only supported with approach_curve")
    if abs(args.final_y_offset_mm) > 20.0 or abs(args.final_z_offset_mm) > 20.0:
        raise ValueError("final placement offsets must be within +/-20 millimetres")
    if (
        args.final_y_offset_mm != 0.0 or args.final_z_offset_mm != 0.0
    ) and args.trajectory_profile != "diverse_low":
        raise ValueError(
            "final placement offsets require --trajectory-profile diverse_low"
        )
    openpi_sha: str | None = None
    openpi_clean: bool | None = None
    if args.openpi_root is not None:
        openpi_sha, openpi_clean = git_identity(args.openpi_root.expanduser().resolve())
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
        frozen_collection_sha=args.frozen_collection_sha,
        expected_scene_config_sha256=args.expected_scene_config_sha256,
        openpi_git_sha=openpi_sha,
        openpi_worktree_clean=openpi_clean,
    )


def create_recorder(
    preflight: CollectionPreflight,
) -> tuple[ImageCas, CanonicalV2Recorder]:
    image_cas = ImageCas(ImageCasConfig(root=preflight.cas_root))
    image_cas.assert_ready(writable=True)
    metadata = CanonicalV2EpisodeMetadata(
        episode_id=preflight.episode_id,
        task_id=preflight.task_id,
        instruction=preflight.instruction,
        scene_seed=preflight.scene_seed,
        git_sha=preflight.git_sha,
        scene_config_sha256=preflight.scene_config_sha256,
        scene_id=preflight.scene_id,
    )
    return image_cas, CanonicalV2Recorder(
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
