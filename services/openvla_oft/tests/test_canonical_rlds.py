from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from industrial_agent.data import CanonicalRecorder, EpisodeMetadata
from industrial_agent.image_cas import ImageCas, ImageCasConfig

from openvla_oft.canonical import load_openvla_arm_b_steps
from openvla_oft.exceptions import ServiceError
from openvla_oft.rlds import (
    build_rlds_episode,
    summarize_rlds_style_export,
    write_rlds_style_episode,
)

CAMERA_IDS = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")


def _metadata(episode_id: str = "arm-b-episode") -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_id=episode_id,
        task_id="golden-task-arm-b",
        instruction="transport the full bin from HANDOFF_CENTER to FINISHED_01",
        scene_seed=20260804,
        git_sha="a" * 40,
        scene_config_sha256=f"sha256:{'b' * 64}",
    )


def _image(value: int) -> np.ndarray:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[..., 0] = value
    frame[..., 1] = 32
    return frame


def _create_episode(
    tmp_path: Path,
    *,
    arm_id: str = "Arm_B",
    executor: str = "openvla_oft",
    subtask_id: str = "S02_ARM_B_TRANSPORT",
) -> Path:
    image_cas = ImageCas(ImageCasConfig(root=tmp_path / "cas"))
    recorder = CanonicalRecorder(
        tmp_path / "episodes",
        _metadata(),
        image_cas=image_cas,
    )
    timestamp_base = 1_000_000_000
    for sequence_id, physics_tick in enumerate((0, 4, 8, 12)):
        for camera_id in CAMERA_IDS:
            reference = image_cas.write_rgb(
                _image(24 + sequence_id),
                camera_id=camera_id,
            )
            recorder.add_frame(
                camera_id=camera_id,
                timestamp_ns=timestamp_base + physics_tick,
                physics_tick=physics_tick,
                sequence_id=sequence_id,
                image_reference=reference,
            )
    for sequence_id, physics_tick in enumerate((0, 2, 4, 6, 8, 10, 12)):
        for state_arm_id in ("Arm_A", "Arm_B"):
            gripper = 1.0 if state_arm_id == "Arm_B" else 0.0
            recorder.add_state(
                arm_id=state_arm_id,
                timestamp_ns=timestamp_base + 100 + physics_tick,
                physics_tick=physics_tick,
                sequence_id=sequence_id,
                state_7d=[
                    float(sequence_id),
                    0.1,
                    0.2,
                    0.0,
                    0.0,
                    0.0,
                    gripper,
                ],
            )
    recorder.add_action_chunk(
        arm_id=arm_id,
        executor=executor,
        subtask_id=subtask_id,
        chunk_id="chunk-arm-b-001",
        start_timestamp_ns=timestamp_base + 200,
        start_physics_tick=0,
        start_sequence_id=0,
        actions=[
            [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0],
            [0.00, 0.01, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )
    return recorder.save_episode(outcome="SUCCEEDED")


def test_load_openvla_arm_b_steps_reads_hdf5_pixels_and_lineage(tmp_path: Path) -> None:
    episode_path = _create_episode(tmp_path)

    steps = load_openvla_arm_b_steps(episode_path)

    assert len(steps) == 2
    assert steps[0].is_first is True
    assert steps[0].is_last is False
    assert steps[1].is_last is True
    assert steps[1].is_terminal is True
    assert steps[0].image.shape == (720, 1280, 3)
    assert steps[0].image.dtype == np.uint8
    assert int(steps[1].image[0, 0, 0]) == 27
    assert steps[1].state_7d[0] == pytest.approx(6.0)
    assert steps[1].source.camera_id == "CAM_B_TOP"
    assert steps[1].source.camera_sequence_id == 3
    assert steps[1].source.state_sequence_id == 6
    assert steps[1].source.action_sequence_id == 1

    sample = steps[0].to_training_sample()
    assert sample["robot_role"] == "arm_b_openvla"
    assert sample["model_input"]["wrist_image"] is None
    assert np.asarray(sample["model_input"]["full_image"]).shape == (720, 1280, 3)


def test_build_rlds_episode_preserves_step_flags_and_sources(tmp_path: Path) -> None:
    steps = load_openvla_arm_b_steps(_create_episode(tmp_path))

    episode = build_rlds_episode(steps)

    assert episode["episode_id"] == "arm-b-episode"
    assert episode["steps"][0]["is_first"] is True
    assert episode["steps"][0]["is_last"] is False
    assert episode["steps"][1]["is_last"] is True
    assert episode["steps"][1]["metadata"]["camera_id"] == "CAM_B_TOP"
    assert episode["steps"][1]["observation"]["image"].shape == (720, 1280, 3)


def test_write_rlds_style_episode_writes_metadata_jsonl_and_arrays(
    tmp_path: Path,
) -> None:
    steps = load_openvla_arm_b_steps(_create_episode(tmp_path))
    output_dir = tmp_path / "rlds_export"

    write_rlds_style_episode(steps, output_dir)

    summary = summarize_rlds_style_export(output_dir)
    assert summary["schema_version"] == "openvla_oft_rlds_style_v1"
    assert summary["step_count"] == 2
    assert summary["arrays"]["images"]["shape"] == [2, 720, 1280, 3]
    records = [
        json.loads(line)
        for line in (output_dir / "steps.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert records[0]["image_array"] == "arrays.npz:images[0]"
    assert records[1]["is_terminal"] is True
    arrays = np.load(output_dir / "arrays.npz")
    assert arrays["images"].shape == (2, 720, 1280, 3)
    assert arrays["states"].shape == (2, 7)
    assert arrays["actions"].shape == (2, 7)


def test_load_openvla_arm_b_steps_rejects_arm_a_only_episode(tmp_path: Path) -> None:
    episode_path = _create_episode(
        tmp_path,
        arm_id="Arm_A",
        executor="pi05",
        subtask_id="S01_ARM_A_PACK_HANDOFF",
    )

    with pytest.raises(ServiceError, match="no valid Arm_B"):
        load_openvla_arm_b_steps(episode_path)


def test_write_rlds_style_episode_rejects_existing_non_empty_output(
    tmp_path: Path,
) -> None:
    steps = load_openvla_arm_b_steps(_create_episode(tmp_path))
    output_dir = tmp_path / "rlds_export"
    output_dir.mkdir()
    (output_dir / "old.txt").write_text("old", encoding="utf-8")

    with pytest.raises(ServiceError, match="empty or absent"):
        write_rlds_style_episode(steps, output_dir)
