from __future__ import annotations

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from industrial_agent.data import (
    CanonicalV2EpisodeMetadata,
    CanonicalV2Recorder,
    PaddingPolicy,
    PaddingStrategy,
)
from industrial_agent.image_cas import ImageCas, ImageCasConfig
from scripts.pi05.canonical_v2 import CanonicalV2Reader


CAMERA_IDS = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
ARM_IDS = ("Arm_A", "Arm_B")


def _metadata(**overrides: object) -> CanonicalV2EpisodeMetadata:
    values: dict[str, object] = {
        "episode_id": "v2-p01-000001",
        "task_id": "P01_TO_S11",
        "instruction": "把P01放到S11中",
        "scene_seed": 20260819,
        "git_sha": "a" * 40,
        "scene_config_sha256": f"sha256:{'b' * 64}",
    }
    values.update(overrides)
    return CanonicalV2EpisodeMetadata(**values)  # type: ignore[arg-type]


def _recorder(tmp_path: Path) -> tuple[CanonicalV2Recorder, ImageCas]:
    image_cas = ImageCas(ImageCasConfig(root=tmp_path / "cas"))
    recorder = CanonicalV2Recorder(
        tmp_path / "episodes",
        _metadata(),
        image_cas=image_cas,
    )
    return recorder, image_cas


def _record_complete_episode(tmp_path: Path) -> Path:
    recorder, image_cas = _recorder(tmp_path)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[20:30, 40:50] = (10, 20, 30)
    references = {
        camera_id: image_cas.write_rgb(frame, camera_id=camera_id)
        for camera_id in CAMERA_IDS
    }
    with recorder:
        for camera_id in CAMERA_IDS:
            recorder.add_frame(
                camera_id=camera_id,
                timestamp_ns=1_000_000_000,
                physics_tick=0,
                sequence_id=0,
                image_reference=references[camera_id],
            )
        for arm_id in ARM_IDS:
            recorder.add_state(
                arm_id=arm_id,
                timestamp_ns=1_000_000_000,
                physics_tick=0,
                sequence_id=0,
                state_7d=[0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0.375],
            )
        recorder.add_action(
            arm_id="Arm_A",
            executor="pi05",
            subtask_id="P01_TO_S11",
            chunk_id="manual-p01-000001",
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
                action_7d=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        return recorder.save_episode(outcome="SUCCEEDED")


def test_v2_recorder_writes_reader_valid_episode(tmp_path: Path) -> None:
    episode_path = _record_complete_episode(tmp_path)
    manifest = json.loads(
        (episode_path / "structure.json").read_text(encoding="utf-8")
    )

    assert manifest["canonical_schema_version"] == "2.0"
    assert "schema_version" not in manifest
    assert manifest["metadata"]["scene_id"] == "single_bin_manual_industrial_v2"
    assert manifest["metadata"]["padding_policy"] == {
        "strategy": "none",
        "target_length": None,
    }
    assert manifest["streams"]["actions"]["valid_count"] == 1
    with h5py.File(episode_path / "episode.h5", "r") as h5:
        assert h5.attrs["canonical_schema_version"] == "2.0"
        assert "schema_version" not in h5.attrs
        assert h5["actions/valid_mask"][:].tolist() == [True]
    with CanonicalV2Reader(episode_path) as reader:
        assert len(tuple(reader.iter_action_7d())) == 1
        np.testing.assert_equal(reader.state_7d("Arm_A")[0, 6], 0.375)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("scene_id", "single_bin_static_handoff_v1", "scene_id"),
        ("task_id", "P02_TO_S21", "task_id"),
        ("instruction", "把P02放到S21中", "instruction"),
    ],
)
def test_v2_metadata_rejects_non_frozen_identity(
    field: str,
    value: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _metadata(**{field: value})


def test_v2_recorder_rejects_padding_policy(tmp_path: Path) -> None:
    image_cas = ImageCas(ImageCasConfig(root=tmp_path / "cas"))
    policy = PaddingPolicy(
        strategy=PaddingStrategy.REPEAT_LAST_MASKED,
        target_length=10,
    )

    with pytest.raises(ValueError, match="forbids padding"):
        CanonicalV2Recorder(
            tmp_path / "episodes",
            _metadata(),
            image_cas=image_cas,
            padding_policy=policy,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("arm_id", "Arm_B"),
        ("executor", "openvla_oft"),
        ("subtask_id", "P02_TO_S21"),
    ],
)
def test_v2_recorder_rejects_wrong_action_identity(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    recorder, _ = _recorder(tmp_path)
    kwargs = {
        "arm_id": "Arm_A",
        "executor": "pi05",
        "subtask_id": "P01_TO_S11",
        "chunk_id": "manual-p01",
        "timestamp_ns": 1_000_000_000,
        "physics_tick": 0,
        "sequence_id": 0,
        "action_7d": [0.0] * 7,
    }
    kwargs[field] = value

    with recorder, pytest.raises(ValueError, match="Canonical V2 actions require"):
        recorder.add_action(**kwargs)  # type: ignore[arg-type]


def test_v2_action_chunk_preserves_all_rows_without_padding(tmp_path: Path) -> None:
    recorder, _ = _recorder(tmp_path)
    actions = np.zeros((3, 7), dtype=np.float32)

    with recorder:
        result = recorder.add_action_chunk(
            arm_id="Arm_A",
            executor="pi05",
            subtask_id="P01_TO_S11",
            chunk_id="manual-p01",
            start_timestamp_ns=1_000_000_000,
            start_physics_tick=0,
            start_sequence_id=0,
            actions=actions,
        )
        assert result.values.shape == (3, 7)
        assert result.valid_mask.tolist() == [True, True, True]
        assert recorder._h5["actions/valid_mask"][:].tolist() == [True, True, True]
