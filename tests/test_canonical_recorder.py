from __future__ import annotations

import json
from pathlib import Path

import h5py
from jsonschema import Draft202012Validator
import numpy as np
import pytest

from industrial_agent.data import (
    CanonicalEpisodeReader,
    CanonicalRecorder,
    EpisodeMetadata,
    OfflineEpisodeReplay,
    PaddingPolicy,
    PaddingStrategy,
)
from industrial_agent.image_cas import ImageCas, ImageCasConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERA_IDS = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
ARM_IDS = ("Arm_A", "Arm_B")


def _metadata(episode_id: str = "episode-001") -> EpisodeMetadata:
    return EpisodeMetadata(
        episode_id=episode_id,
        task_id="task-001",
        instruction="将四个红色零件装箱并完成交接",
        scene_seed=7,
        git_sha="a" * 40,
        scene_config_sha256=f"sha256:{'b' * 64}",
    )


def _cas(tmp_path: Path) -> ImageCas:
    return ImageCas(ImageCasConfig(root=tmp_path / "cas"))


def _references(image_cas: ImageCas) -> dict[str, object]:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    return {
        camera_id: image_cas.write_rgb(frame, camera_id=camera_id)
        for camera_id in CAMERA_IDS
    }


def _record_complete_episode(
    tmp_path: Path,
    *,
    episode_id: str = "episode-001",
    padding_policy: PaddingPolicy | None = None,
) -> Path:
    image_cas = _cas(tmp_path)
    references = _references(image_cas)
    recorder = CanonicalRecorder(
        tmp_path / "episodes",
        _metadata(episode_id),
        image_cas=image_cas,
        padding_policy=padding_policy,
    )
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
            state_7d=[0.1, 0.2, 0.3, 0.01, -0.02, 0.03, 1.0],
        )
    recorder.add_action_chunk(
        arm_id="Arm_A",
        executor="pi05",
        subtask_id="S01_ARM_A_PACK_HANDOFF",
        chunk_id="chunk-001",
        start_timestamp_ns=1_000_000_000,
        start_physics_tick=0,
        start_sequence_id=0,
        actions=[[0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 1.0]],
    )
    return recorder.save_episode(outcome="SUCCEEDED")


def test_recorder_round_trip_and_manifest_schema(tmp_path: Path) -> None:
    episode_path = _record_complete_episode(
        tmp_path,
        padding_policy=PaddingPolicy(
            PaddingStrategy.ZERO_MASKED,
            target_length=2,
        ),
    )
    manifest = json.loads((episode_path / "structure.json").read_text(encoding="utf-8"))
    schema = json.loads(
        (REPO_ROOT / "schemas" / "canonical-episode.schema.json").read_text(
            encoding="utf-8"
        )
    )
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(manifest)

    assert manifest["metadata"]["wrist_image"] is None
    assert manifest["metadata"]["offline_gt_included"] is False
    assert manifest["metadata"]["frequency_contract"] == {
        "physics_hz": 120,
        "control_hz": 60,
        "render_hz": 30,
        "model_inference_hz": 10,
    }
    assert manifest["streams"]["actions"]["count"] == 2
    assert manifest["streams"]["actions"]["valid_count"] == 1

    with CanonicalEpisodeReader(episode_path) as reader:
        for camera_id in CAMERA_IDS:
            assert reader.camera_frames(camera_id).shape == (1, 720, 1280, 3)
        for arm_id in ARM_IDS:
            state = reader.state_stream(arm_id)
            assert state["state_7d"].shape == (1, 7)
        actions = OfflineEpisodeReplay(reader).actions()
        assert len(actions) == 1
        assert actions[0].arm_id == "Arm_A"
        assert actions[0].executor == "pi05"
        assert actions[0].duration_ms == 100
        assert actions[0].action_7d == pytest.approx(
            (0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 1.0)
        )

    with h5py.File(episode_path / "episode.h5", "r") as h5:
        assert set(h5) == {"cameras", "robot_state", "actions"}
        assert h5.attrs["wrist_image"] == "null"
        assert h5["actions/valid_mask"][:].tolist() == [True, False]


def test_invalid_shapes_and_wrong_executor_fail_closed(tmp_path: Path) -> None:
    image_cas = _cas(tmp_path)
    recorder = CanonicalRecorder(
        tmp_path / "episodes",
        _metadata(),
        image_cas=image_cas,
    )
    bad_reference = image_cas.write_rgb(
        np.zeros((480, 640, 3), dtype=np.uint8),
        camera_id="CAM_A_TOP",
    )
    with pytest.raises(ValueError, match="1280|dimensions"):
        recorder.add_frame(
            camera_id="CAM_A_TOP",
            timestamp_ns=1,
            physics_tick=0,
            sequence_id=0,
            image_reference=bad_reference,
        )
    with pytest.raises(ValueError, match=r"shape \[7\]"):
        recorder.add_state(
            arm_id="Arm_A",
            timestamp_ns=1,
            physics_tick=0,
            sequence_id=0,
            state_7d=[0.0] * 6,
        )
    with pytest.raises(ValueError, match="requires executor"):
        recorder.add_action(
            arm_id="Arm_A",
            executor="openvla_oft",
            subtask_id="S01_ARM_A_PACK_HANDOFF",
            chunk_id="chunk-bad",
            timestamp_ns=1,
            physics_tick=0,
            sequence_id=0,
            action_7d=[0.0] * 7,
        )
    recorder.abort()


def test_incomplete_episode_is_never_published(tmp_path: Path) -> None:
    episodes = tmp_path / "episodes"
    recorder = CanonicalRecorder(
        episodes,
        _metadata(),
        image_cas=_cas(tmp_path),
    )
    with pytest.raises(ValueError, match="no RGB frames"):
        recorder.save_episode(outcome="SUCCEEDED")
    assert not (episodes / "episode-001").exists()
    recorder.abort()
    assert not list(episodes.glob("*.tmp"))


def test_camera_sync_mismatch_blocks_publication(tmp_path: Path) -> None:
    image_cas = _cas(tmp_path)
    references = _references(image_cas)
    recorder = CanonicalRecorder(
        tmp_path / "episodes",
        _metadata(),
        image_cas=image_cas,
    )
    for index, camera_id in enumerate(CAMERA_IDS):
        recorder.add_frame(
            camera_id=camera_id,
            timestamp_ns=1_000_000_000 + index,
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
            state_7d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
    recorder.add_action(
        arm_id="Arm_A",
        executor="pi05",
        subtask_id="S01_ARM_A_PACK_HANDOFF",
        chunk_id="chunk-001",
        timestamp_ns=1_000_000_000,
        physics_tick=0,
        sequence_id=0,
        action_7d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    )
    with pytest.raises(ValueError, match="synchronized timestamps"):
        recorder.save_episode(outcome="SUCCEEDED")
    recorder.abort()


def test_reader_rejects_tampered_hdf5(tmp_path: Path) -> None:
    episode_path = _record_complete_episode(tmp_path)
    with (episode_path / "episode.h5").open("ab") as stream:
        stream.write(b"tamper")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        CanonicalEpisodeReader(episode_path)


def test_each_stream_enforces_its_frozen_physics_tick_grid(tmp_path: Path) -> None:
    image_cas = _cas(tmp_path)
    references = _references(image_cas)
    recorder = CanonicalRecorder(
        tmp_path / "episodes",
        _metadata(),
        image_cas=image_cas,
    )

    with pytest.raises(ValueError, match="stride 4"):
        recorder.add_frame(
            camera_id="CAM_A_TOP",
            timestamp_ns=1,
            physics_tick=2,
            sequence_id=0,
            image_reference=references["CAM_A_TOP"],
        )
    with pytest.raises(ValueError, match="stride 2"):
        recorder.add_state(
            arm_id="Arm_A",
            timestamp_ns=1,
            physics_tick=1,
            sequence_id=0,
            state_7d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        )
    with pytest.raises(ValueError, match="100"):
        recorder.add_action(
            arm_id="Arm_A",
            executor="pi05",
            subtask_id="S01_ARM_A_PACK_HANDOFF",
            chunk_id="chunk-bad-rate",
            timestamp_ns=1,
            physics_tick=0,
            sequence_id=0,
            action_7d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            duration_ms=50,
        )
    recorder.abort()
