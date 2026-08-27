from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_agent.contracts import ActionStep
from simulation import generate_v2_replay_batch as batch


def _source_actions() -> tuple[ActionStep, ...]:
    actions: list[ActionStep] = []
    for index in range(26):
        gripper = 0.0 if 8 <= index <= 18 else 1.0
        values = [0.001, 0.0, 0.0005, 0.0, 0.0, 0.0, gripper]
        actions.append(ActionStep.from_sequence(values, duration_ms=100))
    return tuple(actions)


def _source(tmp_path: Path) -> batch.SourceEpisode:
    return batch.SourceEpisode(
        path=(tmp_path / "source-episode").resolve(),
        episode_id="w01-mother-001",
        task_id="W01_TO_S14",
        instruction="请将扳手 W01 放置到料箱的 S14 格子中。",
        scene_config_sha256=f"sha256:{'a' * 64}",
        hdf5_sha256=f"sha256:{'b' * 64}",
        actions=_source_actions(),
        arm_ids=("Arm_A",) * len(_source_actions()),
    )


def test_variant_schedule_is_deterministic_and_contains_both_profiles() -> None:
    first = batch.build_variant_specs(
        base_seed=700,
        diverse_low_count=3,
        approach_curve_count=4,
    )
    second = batch.build_variant_specs(
        base_seed=700,
        diverse_low_count=3,
        approach_curve_count=4,
    )

    assert first == second
    assert [item.profile for item in first] == ["diverse_low"] * 3 + [
        "approach_curve"
    ] * 4
    assert [item.final_y_offset_mm for item in first[:3]] == [0.0, -2.0, 2.0]
    assert [item.variant for item in first[3:]] == [1, 2, 3, 4]


def test_source_metadata_rejects_failed_episode() -> None:
    with pytest.raises(batch.ReplayBatchError, match="outcome SUCCEEDED"):
        batch._validate_source_metadata(
            {
                "outcome": "FAILED",
                "task_id": "W01_TO_S14",
                "instruction": "请将扳手 W01 放置到料箱的 S14 格子中。",
            }
        )


def test_action_hash_includes_duration() -> None:
    action = ActionStep.from_sequence([0, 0, 0, 0, 0, 0, 1], duration_ms=100)
    slower = ActionStep.from_sequence([0, 0, 0, 0, 0, 0, 1], duration_ms=200)

    assert batch.action_sha256([action]) != batch.action_sha256([slower])


def test_approach_curve_is_bounded_to_guard_safe_amplitude() -> None:
    import numpy as np

    source = list(_source_actions())
    varied = batch._diversify_replay_actions(
        source,
        profile="approach_curve",
        seed=901,
        variant=1,
    )

    base_positions = np.vstack(
        [np.zeros(3), np.cumsum(np.asarray([a.values[:3] for a in source]), axis=0)]
    )
    varied_positions = np.vstack(
        [np.zeros(3), np.cumsum(np.asarray([a.values[:3] for a in varied]), axis=0)]
    )
    displacement = np.max(np.abs(varied_positions - base_positions), axis=0)

    assert np.any(np.abs(varied_positions - base_positions) > 0.0)
    assert displacement[0] <= 0.0005 + 1e-12
    assert displacement[1] <= 0.0005 + 1e-12
    assert displacement[2] == 0.0
    assert [a.values[6] for a in varied] == [a.values[6] for a in source]
    for original, changed in zip(source, varied):
        original_xyz = np.asarray(original.values[:3], dtype=float)
        changed_xyz = np.asarray(changed.values[:3], dtype=float)
        if np.linalg.norm(original_xyz) > 1e-12:
            assert np.linalg.norm(changed_xyz - original_xyz) <= (
                0.1 * np.linalg.norm(original_xyz) + 1e-12
            )


def test_approach_curve_skips_zero_pregrasp_steps() -> None:
    source = list(_source_actions())
    source[3] = ActionStep.from_sequence(
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, source[3].values[6]],
        duration_ms=source[3].duration_ms,
    )

    varied = batch._diversify_replay_actions(
        source,
        profile="approach_curve",
        seed=902,
        variant=2,
    )

    assert len(varied) == len(source)
    assert varied[3].values[:3] == source[3].values[:3]
    assert any(a.values[:3] != b.values[:3] for a, b in zip(varied, source))


def test_dual_arm_variation_preserves_one_ordered_arm_boundary() -> None:
    arm_a = list(_source_actions())
    arm_b = list(_source_actions())
    actions = arm_a + arm_b
    arm_ids = ["Arm_A"] * len(arm_a) + ["Arm_B"] * len(arm_b)

    varied = batch._diversify_replay_actions(
        actions,
        profile="diverse_low",
        seed=903,
        final_y_offset_mm=2.0,
        arm_ids=arm_ids,
    )

    assert len(varied) == len(actions)
    assert [action.values[6] for action in varied] == [
        action.values[6] for action in actions
    ]
    import numpy as np

    np.testing.assert_allclose(
        varied[len(arm_a) - 1].values[:3],
        arm_a[-1].values[:3],
        rtol=0.0,
        atol=1e-12,
    )
    assert any(
        changed.values[:3] != original.values[:3]
        for changed, original in zip(varied, actions)
    )


def test_generate_w01_batch_writes_hashed_configs_manifest_and_commands(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    scene_config = tmp_path / "scene.json"
    scene_config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(batch, "load_source_episode", lambda *args, **kwargs: source)

    manifest_path = batch.generate_batch(
        source_episode=source.path,
        output_dir=tmp_path / "batch",
        episode_root=tmp_path / "episodes",
        cas_root=tmp_path / "cas",
        artifact_root=tmp_path / "artifacts",
        scene_output_root=tmp_path / "scenes",
        scene_config=scene_config,
        base_seed=900,
        diverse_low_count=2,
        approach_curve_count=4,
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["source"]["task_id"] == "W01_TO_S14"
    assert manifest["counts"] == {"planned": 6, "accepted": 0, "rejected": 0}
    assert manifest["deduplication"] == {
        "reference_roots": [],
        "existing_unique_action_hashes": 0,
        "existing_episode_count": 0,
    }
    hashes = {item["planned_action_sha256"] for item in manifest["trajectories"]}
    assert len(hashes) == 6
    for item in manifest["trajectories"]:
        config_path = manifest_path.parent / item["config_file"]
        assert batch._sha256_file(config_path) == item["config_sha256"]
        config = json.loads(config_path.read_text(encoding="utf-8"))
        assert config["run"]["task_id"] == "W01_TO_S14"
        assert config["trajectory"]["seed"] == item["seed"]
        assert config["trajectory"]["final_offset_mm"] == item["final_offset_mm"]
    commands = (manifest_path.parent / batch.COMMANDS_FILENAME).read_text(
        encoding="utf-8"
    )
    assert commands.count("simulation/run_v2_keyboard_collection.py") == 6
    assert "--task-id' 'W01_TO_S14" in commands
    assert "finalize" in commands


def test_generation_rejects_a_duplicate_of_the_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    scene_config = tmp_path / "scene.json"
    scene_config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(batch, "load_source_episode", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        batch,
        "_diversify_replay_actions",
        lambda actions, **kwargs: list(actions),
    )

    with pytest.raises(batch.ReplayBatchError, match="duplicate trajectory rejected"):
        batch.generate_batch(
            source_episode=source.path,
            output_dir=tmp_path / "batch",
            episode_root=tmp_path / "episodes",
            cas_root=tmp_path / "cas",
            artifact_root=tmp_path / "artifacts",
            scene_output_root=tmp_path / "scenes",
            scene_config=scene_config,
            diverse_low_count=1,
            approach_curve_count=0,
        )


def test_generation_rejects_a_duplicate_from_an_existing_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    scene_config = tmp_path / "scene.json"
    scene_config.write_text("{}\n", encoding="utf-8")
    spec = batch.build_variant_specs(
        base_seed=1000,
        diverse_low_count=1,
        approach_curve_count=0,
    )[0]
    candidate = batch._diversify_replay_actions(
        list(source.actions),
        profile=spec.profile,
        seed=spec.seed,
        variant=spec.variant,
        lift_mm=spec.lift_mm,
        final_y_offset_mm=spec.final_y_offset_mm,
        final_z_offset_mm=spec.final_z_offset_mm,
    )
    candidate_sha = batch.action_sha256(candidate)
    existing_root = tmp_path / "existing"
    existing_root.mkdir()
    monkeypatch.setattr(batch, "load_source_episode", lambda *args, **kwargs: source)
    monkeypatch.setattr(
        batch,
        "load_existing_action_hashes",
        lambda roots: {candidate_sha: (str(existing_root / "episode-001"),)},
    )

    with pytest.raises(batch.ReplayBatchError, match="existing episodes"):
        batch.generate_batch(
            source_episode=source.path,
            output_dir=tmp_path / "batch",
            episode_root=tmp_path / "episodes",
            cas_root=tmp_path / "cas",
            artifact_root=tmp_path / "artifacts",
            scene_output_root=tmp_path / "scenes",
            scene_config=scene_config,
            diverse_low_count=1,
            approach_curve_count=0,
            reject_against_roots=[existing_root],
        )


def test_finalize_rejects_missing_failed_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    scene_config = tmp_path / "scene.json"
    scene_config.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(batch, "load_source_episode", lambda *args, **kwargs: source)
    manifest_path = batch.generate_batch(
        source_episode=source.path,
        output_dir=tmp_path / "batch",
        episode_root=tmp_path / "episodes",
        cas_root=tmp_path / "cas",
        artifact_root=tmp_path / "artifacts",
        scene_output_root=tmp_path / "scenes",
        scene_config=scene_config,
        diverse_low_count=1,
        approach_curve_count=1,
    )

    result = batch.finalize_batch(manifest_path)

    assert result["status"] == "REJECTED"
    assert result["training_ready"] is False
    assert result["counts"] == {"planned": 2, "accepted": 0, "rejected": 2}
    assert all(item["status"] == "REJECTED" for item in result["trajectories"])


def test_approach_curve_cardinality_is_fail_closed() -> None:
    with pytest.raises(batch.ReplayBatchError, match="approach_curve_count"):
        batch.build_variant_specs(
            base_seed=1,
            diverse_low_count=0,
            approach_curve_count=5,
        )
