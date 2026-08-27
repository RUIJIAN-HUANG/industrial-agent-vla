from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from industrial_agent.image_cas import ImageCas, ImageCasConfig
from scripts.pi05.canonical_v2 import CanonicalV2Reader
from simulation.v2_collection_recorder import (
    ARM_IDS,
    CAMERA_IDS,
    V2CollectionIdentity,
    V2CollectionRecorder,
)


def _writer(
    tmp_path: Path,
    *,
    task_id: str = "P01_TO_S11",
    instruction: str = "请将轴件 P01 放置到料箱的 S11 格子中。",
) -> tuple[V2CollectionRecorder, ImageCas]:
    image_cas = ImageCas(ImageCasConfig(root=tmp_path / "cas"))
    identity = V2CollectionIdentity(
        episode_id="v2-collection-000001",
        scene_seed=19,
        git_sha="a" * 40,
        scene_config_sha256=f"sha256:{'b' * 64}",
        task_id=task_id,
        instruction=instruction,
    )
    return (
        V2CollectionRecorder(
            tmp_path / "episodes",
            identity,
            image_cas=image_cas,
        ),
        image_cas,
    )


def _images(image_cas: ImageCas) -> dict[str, object]:
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[0, 0] = (1, 2, 3)
    return {
        camera_id: image_cas.write_rgb(frame, camera_id=camera_id)
        for camera_id in CAMERA_IDS
    }


def _states() -> dict[str, list[float]]:
    return {arm_id: [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0] for arm_id in ARM_IDS}


def test_v2_collection_boundary_writes_reader_valid_episode(tmp_path: Path) -> None:
    writer, image_cas = _writer(tmp_path)
    with writer:
        writer.record_camera_bundle(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            images=_images(image_cas),
        )
        writer.record_state_bundle(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            states=_states(),
        )
        writer.record_action(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            chunk_id="manual-p01-0",
            action_7d=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        episode_path = writer.finalize(outcome="SUCCEEDED")

    with CanonicalV2Reader(episode_path) as reader:
        assert len(tuple(reader.iter_action_7d())) == 1


def test_v2_collection_boundary_preserves_w01_identity(tmp_path: Path) -> None:
    writer, image_cas = _writer(
        tmp_path,
        task_id="W01_TO_S14",
        instruction="请将扳手 W01 放置到料箱的 S14 格子中。",
    )
    with writer:
        writer.record_camera_bundle(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            images=_images(image_cas),
        )
        writer.record_state_bundle(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            states=_states(),
        )
        writer.record_action(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            chunk_id="manual-w01-0",
            action_7d=[0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        )
        episode_path = writer.finalize(outcome="SUCCEEDED")

    with CanonicalV2Reader(episode_path) as reader:
        assert reader.manifest["metadata"]["task_id"] == "W01_TO_S14"
        assert (
            reader.manifest["metadata"]["instruction"]
            == "请将扳手 W01 放置到料箱的 S14 格子中。"
        )
        assert reader._h5["actions/subtask_id"][0].decode() == "W01_TO_S14"


def test_v2_collection_rejects_action_without_exact_tick_observation(
    tmp_path: Path,
) -> None:
    writer, _ = _writer(tmp_path)
    with writer, pytest.raises(ValueError, match="three-camera bundle"):
        writer.record_action(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            chunk_id="manual-p01-0",
            action_7d=np.zeros(7, dtype=np.float32),
        )


def test_v2_collection_rejects_incomplete_camera_bundle(tmp_path: Path) -> None:
    writer, image_cas = _writer(tmp_path)
    images = _images(image_cas)
    del images["CAM_B_TOP"]

    with writer, pytest.raises(ValueError, match="images must contain exactly"):
        writer.record_camera_bundle(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            images=images,
        )


def test_v2_collection_rejects_incomplete_state_bundle(tmp_path: Path) -> None:
    writer, _ = _writer(tmp_path)
    states = _states()
    del states["Arm_B"]

    with writer, pytest.raises(ValueError, match="states must contain exactly"):
        writer.record_state_bundle(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            states=states,
        )


@pytest.mark.parametrize("gripper", [-1.0, 0.5, 0.999])
def test_v2_collection_rejects_non_binary_action_gripper(
    tmp_path: Path,
    gripper: float,
) -> None:
    writer, image_cas = _writer(tmp_path)
    with writer:
        writer.record_camera_bundle(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            images=_images(image_cas),
        )
        writer.record_state_bundle(
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            states=_states(),
        )
        with pytest.raises(ValueError, match="exactly 0.0 or 1.0"):
            writer.record_action(
                timestamp_ns=1_000_000_000,
                physics_tick=0,
                sequence_id=0,
                chunk_id="manual-p01-invalid",
                action_7d=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper],
            )
