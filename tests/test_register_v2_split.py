from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_agent.data import DatasetSplit, SplitRegistry
from scripts.pi05 import register_v2_split as registration


def _formal_result(
    root: Path,
    *,
    episode_id: str,
    collection_split: str,
    scene_seed: int,
) -> tuple[Path, Path, dict[str, object]]:
    episode_dir = root / "episodes" / episode_id
    episode_dir.mkdir(parents=True)
    result = {
        "status": "PASS",
        "outcome": "SUCCEEDED",
        "episode_path": str(episode_dir),
        "preflight": {
            "episode_id": episode_id,
            "episode_dir": str(episode_dir),
            "split": collection_split,
            "scene_seed": scene_seed,
        },
    }
    path = root / f"{episode_id}-result.json"
    path.write_text(json.dumps(result), encoding="utf-8")
    manifest = {
        "metadata": {
            "episode_id": episode_id,
            "scene_id": "single_bin_manual_industrial_v2",
            "scene_seed": scene_seed,
            "scene_config_sha256": f"sha256:{'a' * 64}",
            "outcome": "SUCCEEDED",
        }
    }
    return path, episode_dir, manifest


def test_registers_train_mother_from_matching_collection_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, manifest = _formal_result(
        tmp_path,
        episode_id="bin01-train-m01",
        collection_split="train",
        scene_seed=0,
    )
    monkeypatch.setattr(registration, "_validated_manifest", lambda _: manifest)
    registry_path = tmp_path / "split_registry_v1.json"

    receipt = registration.register_collection_result(
        result_json=result,
        registry_path=registry_path,
        split="train",
        scenario_group_id="bin01-finished01-m01",
    )

    loaded = SplitRegistry.load(registry_path)
    assignment = loaded.get_assignment("bin01-train-m01")
    assert assignment.split is DatasetSplit.TRAIN
    assert assignment.scene_seed == 0
    assert receipt["registry_sha256"] == loaded.registry_sha256


def test_maps_validation_collection_to_val_registry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, manifest = _formal_result(
        tmp_path,
        episode_id="bin01-val-m04",
        collection_split="validation",
        scene_seed=1,
    )
    monkeypatch.setattr(registration, "_validated_manifest", lambda _: manifest)
    registry_path = tmp_path / "split_registry_v1.json"

    registration.register_collection_result(
        result_json=result,
        registry_path=registry_path,
        split="val",
        scenario_group_id="bin01-finished01-m04",
    )

    assert (
        SplitRegistry.load(registry_path).get_split("bin01-val-m04") is DatasetSplit.VAL
    )


def test_rejects_mismatched_split_before_registry_is_written(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result, _, manifest = _formal_result(
        tmp_path,
        episode_id="bin01-val-m04",
        collection_split="train",
        scene_seed=1,
    )
    monkeypatch.setattr(registration, "_validated_manifest", lambda _: manifest)
    registry_path = tmp_path / "split_registry_v1.json"

    with pytest.raises(registration.SplitRegistrationError, match="split mismatch"):
        registration.register_collection_result(
            result_json=result,
            registry_path=registry_path,
            split="val",
            scenario_group_id="bin01-finished01-m04",
        )

    assert not registry_path.exists()


def test_derived_episode_inherits_registered_parent_group(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mother_result, _, mother_manifest = _formal_result(
        tmp_path,
        episode_id="bin01-train-m01",
        collection_split="train",
        scene_seed=0,
    )
    derived_result, _, derived_manifest = _formal_result(
        tmp_path,
        episode_id="bin01-train-m01-derived-001",
        collection_split="train",
        scene_seed=0,
    )
    manifests = iter((mother_manifest, derived_manifest))
    monkeypatch.setattr(registration, "_validated_manifest", lambda _: next(manifests))
    registry_path = tmp_path / "split_registry_v1.json"
    registration.register_collection_result(
        result_json=mother_result,
        registry_path=registry_path,
        split="train",
        scenario_group_id="bin01-finished01-m01",
    )

    registration.register_collection_result(
        result_json=derived_result,
        registry_path=registry_path,
        split="train",
        parent_episode_id="bin01-train-m01",
    )

    registry = SplitRegistry.load(registry_path)
    mother = registry.get_assignment("bin01-train-m01")
    derived = registry.get_assignment("bin01-train-m01-derived-001")
    assert derived.parent_episode_id == mother.episode_id
    assert derived.split is mother.split
    assert derived.group_key == mother.group_key
