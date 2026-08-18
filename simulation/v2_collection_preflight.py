"""Fail-closed preflight for one V2 manual collection session."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Any

from simulation.v2_collection_state import V2CollectionContract


_CANONICAL_SCHEMA_VERSION = "1.0"
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")


class CollectionSplit(str, Enum):
    PRACTICE = "practice"
    TEST = "test"
    TRAIN = "train"
    VALIDATION = "validation"


class CollectionPreflightError(ValueError):
    """A collection attempt that must be rejected before Isaac starts."""


@dataclass(frozen=True)
class CollectionPreflight:
    config_path: Path
    episode_root: Path
    cas_root: Path
    episode_id: str
    episode_dir: Path
    task_id: str
    instruction: str
    scene_seed: int
    split: CollectionSplit
    schema_version: str
    scene_id: str
    git_sha: str
    scene_config_sha256: str
    training_allowed: bool
    full_task_required: bool


def _non_blank(value: Any, name: str, *, maximum: int = 16_384) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CollectionPreflightError(f"{name} must be non-empty")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise CollectionPreflightError(
            f"{name} exceeds {maximum} characters"
        )
    return normalized


def _safe_output_root(value: str | Path, name: str) -> Path:
    source = Path(value).expanduser()

    if source.exists() and source.is_symlink():
        raise CollectionPreflightError(
            f"{name} must not be a symbolic link"
        )

    return source.resolve()


def build_collection_preflight(
    *,
    config_path: str | Path,
    episode_root: str | Path,
    cas_root: str | Path,
    episode_id: str,
    task_id: str,
    instruction: str,
    scene_seed: int,
    split: CollectionSplit | str,
    headless: bool,
    git_sha: str,
    worktree_clean: bool,
) -> CollectionPreflight:
    """Validate immutable identity before constructing any Isaac object."""

    if headless:
        raise CollectionPreflightError(
            "manual collection requires a visible Isaac GUI"
        )

    if not worktree_clean:
        raise CollectionPreflightError(
            "formal collection requires a clean Git worktree"
        )

    if (
        not isinstance(git_sha, str)
        or _GIT_SHA.fullmatch(git_sha) is None
    ):
        raise CollectionPreflightError(
            "git_sha must be exactly 40 hexadecimal characters"
        )

    if (
        not isinstance(episode_id, str)
        or _SAFE_EPISODE_ID.fullmatch(episode_id) is None
    ):
        raise CollectionPreflightError(
            "episode_id must match ^[A-Za-z0-9._-]{1,128}$"
        )

    if (
        isinstance(scene_seed, bool)
        or not isinstance(scene_seed, int)
        or scene_seed < 0
    ):
        raise CollectionPreflightError(
            "scene_seed must be a non-negative integer"
        )

    try:
        normalized_split = CollectionSplit(split)
    except ValueError as exc:
        raise CollectionPreflightError(
            "split must be practice, test, train, or validation"
        ) from exc

    source_config = Path(config_path).expanduser()
    if source_config.is_symlink():
        raise CollectionPreflightError(
            "config_path must not be a symbolic link"
        )
    config = source_config.resolve()
    if not config.is_file():
        raise CollectionPreflightError(
            f"scene config does not exist: {config}"
        )

    raw_config = config.read_bytes()
    try:
        payload = json.loads(raw_config.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CollectionPreflightError(
            "scene config must be valid UTF-8 JSON"
        ) from exc

    if not isinstance(payload, dict):
        raise CollectionPreflightError(
            "scene config root must be an object"
        )

    contract = V2CollectionContract.from_config(payload)

    collection = payload.get("collection")
    if not isinstance(collection, dict):
        raise CollectionPreflightError(
            "collection config must be an object"
        )
    if collection.get("mode") != "manual_keyboard":
        raise CollectionPreflightError(
            "collection mode must be manual_keyboard"
        )
    if collection.get("online_gt_allowed") is not False:
        raise CollectionPreflightError(
            "online ground truth must remain disabled"
        )

    episodes = _safe_output_root(episode_root, "episode_root")
    cas = _safe_output_root(cas_root, "cas_root")
    episode_dir = episodes / episode_id

    if episode_dir.exists() or episode_dir.is_symlink():
        raise CollectionPreflightError(
            f"episode already exists and will not be overwritten: "
            f"{episode_dir}"
        )

    return CollectionPreflight(
        config_path=config,
        episode_root=episodes,
        cas_root=cas,
        episode_id=episode_id,
        episode_dir=episode_dir,
        task_id=_non_blank(task_id, "task_id", maximum=4096),
        instruction=_non_blank(instruction, "instruction"),
        scene_seed=scene_seed,
        split=normalized_split,
        schema_version=_CANONICAL_SCHEMA_VERSION,
        scene_id=contract.scene_id,
        git_sha=git_sha.lower(),
        scene_config_sha256=f"sha256:{sha256(raw_config).hexdigest()}",
        training_allowed=normalized_split is CollectionSplit.TRAIN,
        full_task_required=normalized_split
        in {CollectionSplit.TRAIN, CollectionSplit.VALIDATION},
    )
