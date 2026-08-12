from __future__ import annotations

import json
from pathlib import Path
import shutil

import h5py
from jsonschema import Draft202012Validator
import numpy as np
import pytest

from industrial_agent.data import (
    CanonicalEpisodeReader,
    CanonicalRecorder,
    EpisodeMetadata,
    OfflineEpisodeReplay,
)
from industrial_agent.image_cas import ImageCas, ImageCasConfig


REPO_ROOT = Path(__file__).resolve().parents[1]
CAMERA_IDS = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
ARM_IDS = ("Arm_A", "Arm_B")
GOLDEN_EPISODE = REPO_ROOT / "tests" / "fixtures" / "golden_episode_v1"


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


def test_golden_episode_round_trip_and_manifest_schema() -> None:
    episode_path = GOLDEN_EPISODE
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
    assert manifest["metadata"]["padding_policy"] == {
        "strategy": "none",
        "target_length": None,
    }
    assert manifest["streams"]["actions"]["count"] == 1
    assert manifest["streams"]["actions"]["valid_count"] == 1

    with CanonicalEpisodeReader(episode_path) as reader:
        for camera_id in CAMERA_IDS:
            frames = reader.camera_frames(camera_id)
            assert frames.shape == (3, 720, 1280, 3)
            assert frames.dtype == np.uint8
            assert np.count_nonzero(frames) > 0
        for arm_id in ARM_IDS:
            state = reader.state_stream(arm_id)
            assert state["state_7d"].shape == (6, 7)
            assert set(np.unique(state["state_7d"][:, 6])).issubset({0.0, 1.0})
        actions = OfflineEpisodeReplay(reader).actions()
        assert len(actions) == 1
        assert actions[0].arm_id == "Arm_A"
        assert actions[0].executor == "pi05"
        assert actions[0].duration_ms == 100
        assert actions[0].action_7d == pytest.approx(
            (0.006, -0.002, -0.004, 0.0, 0.01, -0.01, 0.0)
        )

    with h5py.File(episode_path / "episode.h5", "r") as h5:
        assert set(h5) == {"cameras", "robot_state", "actions"}
        assert h5.attrs["wrist_image"] == "null"
        assert h5["actions/valid_mask"][:].tolist() == [True]


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
    episode_path = tmp_path / "golden_episode_v1"
    shutil.copytree(GOLDEN_EPISODE, episode_path)
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

# V2_SCENE_ID_COMPATIBILITY_TESTS
def test_episode_metadata_accepts_only_audited_scene_ids() -> None:
    v1 = _metadata("scene-v1")
    assert v1.scene_id == "single_bin_static_handoff_v1"

    v2 = EpisodeMetadata(
        episode_id="scene-v2",
        task_id="task-001",
        instruction="在 V2 场景执行人工采集",
        scene_seed=7,
        git_sha="a" * 40,
        scene_config_sha256=f"sha256:{'b' * 64}",
        scene_id="single_bin_manual_industrial_v2",
    )
    assert v2.scene_id == "single_bin_manual_industrial_v2"

    with pytest.raises(ValueError, match="scene_id"):
        EpisodeMetadata(
            episode_id="scene-unknown",
            task_id="task-001",
            instruction="拒绝未经审计的场景",
            scene_seed=7,
            git_sha="a" * 40,
            scene_config_sha256=f"sha256:{'b' * 64}",
            scene_id="unknown_scene_v99",
        )


def test_canonical_schema_accepts_v1_v2_and_rejects_unknown_scene() -> None:
    schema = json.loads(
        (REPO_ROOT / "schemas" / "canonical-episode.schema.json").read_text(
            encoding="utf-8"
        )
    )
    manifest = json.loads(
        (GOLDEN_EPISODE / "structure.json").read_text(encoding="utf-8")
    )
    validator = Draft202012Validator(schema)

    for scene_id in (
        "single_bin_static_handoff_v1",
        "single_bin_manual_industrial_v2",
    ):
        candidate = json.loads(json.dumps(manifest))
        candidate["metadata"]["scene_id"] = scene_id
        validator.validate(candidate)

    unknown = json.loads(json.dumps(manifest))
    unknown["metadata"]["scene_id"] = "unknown_scene_v99"
    errors = list(validator.iter_errors(unknown))
    assert errors
    assert any(tuple(error.path) == ("metadata", "scene_id") for error in errors)
