from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_agent.data import (
    CanonicalEpisodeReader,
    DataLeakageError,
    DatasetSplit,
    SplitAssignmentError,
    SplitRegistry,
    SplitRegistryIntegrityError,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
GOLDEN_EPISODE = REPO_ROOT / "tests" / "fixtures" / "golden_episode_v1"
GOLDEN_EPISODE_ID = "golden_episode_v1"


def _assign(
    registry: SplitRegistry,
    episode_id: str,
    split: str,
    *,
    scene_seed: int,
    scenario_group_id: str | None = None,
    parent_episode_id: str | None = None,
) -> None:
    registry.assign_episode(
        episode_id,
        split,
        scenario_group_id=scenario_group_id or f"group-{episode_id}",
        scene_seed=scene_seed,
        asset_variant="asset-v1",
        camera_seed=scene_seed + 100,
        lighting_seed=scene_seed + 200,
        parent_episode_id=parent_episode_id,
    )


def test_registry_round_trip_preserves_assignments_and_digest(tmp_path: Path) -> None:
    registry = SplitRegistry()
    _assign(registry, "train-001", "train", scene_seed=11)
    _assign(registry, "val-001", "val", scene_seed=22)
    _assign(registry, "test-001", "test", scene_seed=33)

    path = registry.save(tmp_path / "splits" / "split_registry_v1.json")
    loaded = SplitRegistry.load(path)

    assert loaded.registry_sha256 == registry.registry_sha256
    assert loaded.get_split("train-001") is DatasetSplit.TRAIN
    assert loaded.get_split("val-001") is DatasetSplit.VAL
    assert loaded.get_split("test-001") is DatasetSplit.TEST


def test_exact_reassignment_is_idempotent_but_changes_are_forbidden() -> None:
    registry = SplitRegistry()
    _assign(registry, "episode-001", "train", scene_seed=11)
    original_digest = registry.registry_sha256

    _assign(registry, "episode-001", "train", scene_seed=11)
    assert registry.registry_sha256 == original_digest

    with pytest.raises(SplitAssignmentError, match="reassignment is forbidden"):
        _assign(registry, "episode-001", "val", scene_seed=11)


def test_save_cannot_overwrite_an_existing_assignment(tmp_path: Path) -> None:
    path = tmp_path / "split_registry_v1.json"
    original = SplitRegistry()
    _assign(original, "episode-001", "train", scene_seed=11)
    original.save(path)

    replacement = SplitRegistry()
    _assign(replacement, "episode-001", "val", scene_seed=22)
    with pytest.raises(SplitAssignmentError, match="overwrite"):
        replacement.save(path)

    assert SplitRegistry.load(path).get_split("episode-001") is DatasetSplit.TRAIN


def test_tampered_registry_fails_sha256_verification(tmp_path: Path) -> None:
    registry = SplitRegistry()
    _assign(registry, "episode-001", "train", scene_seed=11)
    path = registry.save(tmp_path / "split_registry_v1.json")

    document = json.loads(path.read_text(encoding="utf-8"))
    document["assignments"]["episode-001"]["split"] = "test"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(SplitRegistryIntegrityError, match="SHA-256 mismatch"):
        SplitRegistry.load(path)


def test_frozen_group_and_scene_seed_cannot_cross_splits() -> None:
    grouped = SplitRegistry()
    _assign(
        grouped,
        "episode-a",
        "train",
        scene_seed=11,
        scenario_group_id="scene-group",
    )
    with pytest.raises(SplitAssignmentError, match="same frozen group key"):
        _assign(
            grouped,
            "episode-b",
            "val",
            scene_seed=11,
            scenario_group_id="scene-group",
        )

    seeded = SplitRegistry()
    _assign(seeded, "episode-a", "train", scene_seed=11)
    with pytest.raises(SplitAssignmentError, match="scene_seed 11"):
        _assign(seeded, "episode-b", "test", scene_seed=11)


def test_recovery_episode_must_share_parent_split() -> None:
    registry = SplitRegistry()
    _assign(registry, "parent-001", "train", scene_seed=11)

    with pytest.raises(SplitAssignmentError, match="parent episode must share"):
        _assign(
            registry,
            "recovery-001",
            "val",
            scene_seed=22,
            parent_episode_id="parent-001",
        )


def test_training_reader_allows_registered_train_episode() -> None:
    registry = SplitRegistry()
    _assign(registry, GOLDEN_EPISODE_ID, "train", scene_seed=20260803)

    with CanonicalEpisodeReader(
        GOLDEN_EPISODE,
        split_registry=registry,
        is_training=True,
    ) as reader:
        assert reader.split_assignment is not None
        assert reader.split_assignment.split is DatasetSplit.TRAIN
        assert len(tuple(reader.iter_valid_actions())) == 1


@pytest.mark.parametrize("split", ["val", "test"])
def test_training_reader_blocks_registered_val_and_test(
    split: str,
) -> None:
    registry = SplitRegistry()
    _assign(registry, GOLDEN_EPISODE_ID, split, scene_seed=20260803)

    with pytest.raises(DataLeakageError, match=f"split '{split}'"):
        CanonicalEpisodeReader(
            GOLDEN_EPISODE,
            split_registry=registry,
            is_training=True,
        )


def test_training_reader_requires_registered_episode() -> None:
    with pytest.raises(DataLeakageError, match="requires a verified split registry"):
        CanonicalEpisodeReader(GOLDEN_EPISODE, is_training=True)

    registry = SplitRegistry()
    with pytest.raises(DataLeakageError, match="unregistered episode"):
        CanonicalEpisodeReader(
            GOLDEN_EPISODE,
            split_registry=registry,
            is_training=True,
        )
