import argparse
from pathlib import Path

import numpy as np
import pytest

from industrial_agent.contracts import ActionStep
from industrial_agent.data import CanonicalV2Recorder
from scripts.pi05.canonical_v2 import CanonicalV2Reader
from simulation.canonical_recorder_bridge import CanonicalRecorderBridge
from simulation.v2_collection_entry import (
    build_parser,
    create_recorder,
    preflight_from_args,
)


CAMERA_IDS = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")


class _StaticRgbPipeline:
    def __init__(self, references) -> None:
        self._references = references

    def capture_references(self):
        return self._references


def _args(tmp_path: Path) -> argparse.Namespace:
    return build_parser().parse_args(
        [
            "--episode-root",
            str(tmp_path / "episodes"),
            "--cas-root",
            str(tmp_path / "cas"),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--output-scene",
            str(tmp_path / "scene.usda"),
            "--episode-id",
            "v2-practice-001",
            "--task-id",
            "P01_TO_S11",
            "--instruction",
            "把P01放到S11中",
            "--scene-seed",
            "7",
            "--split",
            "practice",
        ]
    )


def test_cli_builds_visible_practice_preflight(tmp_path: Path) -> None:
    assert _args(tmp_path).rotation_step_deg == 5.0
    result = preflight_from_args(_args(tmp_path), git_sha="a" * 40, worktree_clean=True)
    assert result.scene_id == "single_bin_manual_industrial_v2"
    assert result.training_allowed is False
    assert result.full_task_required is False


def test_cli_accepts_optional_replay_episode(tmp_path: Path) -> None:
    args = _args(tmp_path)
    assert args.replay_episode is None

    replay_dir = tmp_path / "golden-episode"
    parsed = build_parser().parse_args(
        [
            "--episode-root",
            str(tmp_path / "episodes"),
            "--cas-root",
            str(tmp_path / "cas"),
            "--artifact-dir",
            str(tmp_path / "artifacts"),
            "--output-scene",
            str(tmp_path / "scene.usda"),
            "--episode-id",
            "v2-replay-001",
            "--task-id",
            "P01_TO_S11",
            "--instruction",
            "把P01放到S11中",
            "--scene-seed",
            "7",
            "--split",
            "practice",
            "--replay-episode",
            str(replay_dir),
        ]
    )
    assert parsed.replay_episode == replay_dir


def test_cli_creates_canonical_v2_recorder(tmp_path: Path) -> None:
    preflight = preflight_from_args(
        _args(tmp_path),
        git_sha="a" * 40,
        worktree_clean=True,
    )
    _, recorder = create_recorder(preflight)
    try:
        assert isinstance(recorder, CanonicalV2Recorder)
        assert recorder._h5.attrs["canonical_schema_version"] == "2.0"
        assert "schema_version" not in recorder._h5.attrs
    finally:
        recorder.abort()


def test_formal_entry_writes_ten_actions_and_v2_reader_recovers_them(
    tmp_path: Path,
) -> None:
    preflight = preflight_from_args(
        _args(tmp_path),
        git_sha="a" * 40,
        worktree_clean=True,
    )
    image_cas, recorder = create_recorder(preflight)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    references = {
        camera_id: image_cas.write_rgb(frame, camera_id=camera_id)
        for camera_id in CAMERA_IDS
    }
    states = {
        "Arm_A": np.asarray([0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "Arm_B": np.asarray([0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0], dtype=np.float32),
    }
    bridge = CanonicalRecorderBridge(
        recorder=recorder,
        rgb_pipeline=_StaticRgbPipeline(references),
        state_source=lambda: states,
        timestamp_origin_ns=1_000_000_000,
    )
    expected: list[np.ndarray] = []

    bridge.record_initial(physics_tick=0)
    latest_tick = 0
    for index in range(10):
        values = np.asarray(
            [index / 1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(index % 2)],
            dtype=np.float32,
        )
        expected.append(values)
        bridge.record_action(
            ActionStep.from_sequence(values, duration_ms=100),
            arm_id="Arm_A",
            subtask_id="P01_TO_S11",
            chunk_id=f"practice-{index:02d}",
            physics_tick=latest_tick,
        )
        if index == 9:
            break
        next_tick = latest_tick + 12
        for physics_tick in range(latest_tick + 1, next_tick + 1):
            bridge.observe_physics_tick(
                physics_tick,
                render_due=physics_tick % 4 == 0,
            )
        latest_tick = next_tick

    episode_path = bridge.save(outcome="SUCCEEDED")

    with CanonicalV2Reader(episode_path) as reader:
        recovered = tuple(reader.iter_action_7d())
        assert reader.episode_id == "v2-practice-001"
        assert len(recovered) == 10
        np.testing.assert_array_equal(np.stack(recovered), np.stack(expected))
        assert reader.state_7d("Arm_A").shape == (55, 7)


def test_cli_rejects_oversized_motion_step(tmp_path: Path) -> None:
    args = _args(tmp_path)
    args.translation_step_m = 0.051
    with pytest.raises(ValueError, match="translation-step"):
        preflight_from_args(args, git_sha="a" * 40, worktree_clean=True)
