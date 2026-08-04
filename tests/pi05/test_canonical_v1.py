from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from industrial_agent.data import (
    CanonicalRecorder,
    EpisodeMetadata,
    SplitRegistry,
)
from industrial_agent.image_cas import ImageCas, ImageCasConfig
from scripts.pi05.canonical_v1 import (
    CanonicalPi05StateMapper,
    CanonicalV1Error,
    EXPECTED_ROBOT_ROLE,
    load_rgb_image,
    map_state,
    read_canonical_dataset,
    read_canonical_episode,
)


CAMERA_IDS = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
ARM_IDS = ("Arm_A", "Arm_B")


def _sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def build_episode(
    output_root: Path,
    episode_id: str = "train-a-000001",
    *,
    arm_id: str = "Arm_A",
    executor: str = "pi05",
    scene_seed: int = 101,
    fallback: bool = False,
    valid_mask: bool = True,
    sentinel: float = 0.1,
) -> Path:
    """Create one compact authoritative HDF5 Episode for role-E tests."""

    image_cas = ImageCas(ImageCasConfig(root=output_root / f"cas-{episode_id}"))
    recorder = CanonicalRecorder(
        output_root,
        EpisodeMetadata(
            episode_id=episode_id,
            task_id="golden-task-v1",
            instruction="将四个红色零件装箱并完成交接",
            scene_seed=scene_seed,
            git_sha="a" * 40,
            scene_config_sha256=f"sha256:{'b' * 64}",
        ),
        image_cas=image_cas,
    )
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    frame[..., 0] = 64
    frame[200:300, 300:420] = (220, 20, 20)
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
                is_fallback=fallback if camera_id == "CAM_A_TOP" else False,
            )
        for current_arm in ARM_IDS:
            recorder.add_state(
                arm_id=current_arm,
                timestamp_ns=1_000_000_000,
                physics_tick=0,
                sequence_id=0,
                state_7d=[sentinel, 0.2, 0.3, 0.01, -0.02, 0.03, 1.0],
            )
        recorder.add_action(
            arm_id=arm_id,
            executor=executor,
            subtask_id=(
                "S01_ARM_A_PACK_HANDOFF" if arm_id == "Arm_A" else "S02_ARM_B_TRANSPORT"
            ),
            chunk_id=f"{episode_id}-chunk",
            timestamp_ns=1_000_000_000,
            physics_tick=0,
            sequence_id=0,
            action_7d=[0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 1.0],
        )
        if not valid_mask:
            recorder._h5["actions/valid_mask"][0] = False
        return recorder.save_episode(outcome="SUCCEEDED")


def build_registry(entries: list[tuple[Path, str]]) -> SplitRegistry:
    registry = SplitRegistry()
    for index, (episode_path, split) in enumerate(entries):
        manifest = json.loads(
            (episode_path / "structure.json").read_text(encoding="utf-8")
        )
        metadata = manifest["metadata"]
        registry.assign_episode(
            metadata["episode_id"],
            split,
            scenario_group_id=f"group-{metadata['episode_id']}",
            scene_seed=int(metadata["scene_seed"]),
            asset_variant=f"asset-{index}",
            camera_seed=10_000 + index,
            lighting_seed=20_000 + index,
        )
    return registry


def refresh_hdf5_sha(episode_path: Path) -> None:
    structure_path = episode_path / "structure.json"
    structure = json.loads(structure_path.read_text(encoding="utf-8"))
    structure["storage"]["sha256"] = _sha256(episode_path / "episode.h5")
    structure_path.write_text(
        json.dumps(structure, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_authoritative_hdf5_reader_projects_arm_a_sample(tmp_path: Path) -> None:
    episode_path = build_episode(tmp_path)
    registry = build_registry([(episode_path, "train")])

    episode = read_canonical_episode(episode_path, split_registry=registry)

    assert episode.robot_role == EXPECTED_ROBOT_ROLE
    assert episode.split == "train"
    assert len(episode.steps) == 1
    step = episode.steps[0]
    assert step.physics_tick == 0
    assert step.action_sequence_id == 0
    assert step.camera_sequence_id == 0
    assert step.state_sequence_id == 0
    assert step.state_7d.shape == (7,)
    assert step.action_7d.shape == (7,)
    assert load_rgb_image(step, episode_id=episode.episode_id).shape == (
        720,
        1280,
        3,
    )


def test_state_mapper_uses_frozen_state_7d_without_reconstruction(
    tmp_path: Path,
) -> None:
    episode_path = build_episode(tmp_path, sentinel=0.4321)
    registry = build_registry([(episode_path, "train")])
    episode = read_canonical_episode(episode_path, split_registry=registry)
    mapped = map_state(CanonicalPi05StateMapper(), episode, episode.steps[0])
    assert mapped.dtype == np.float32
    assert mapped == pytest.approx(episode.steps[0].state_7d)
    assert mapped[0] == pytest.approx(0.4321)


def test_external_split_registry_is_required(tmp_path: Path) -> None:
    episode_path = build_episode(tmp_path)
    with pytest.raises(CanonicalV1Error, match="SplitRegistry is required"):
        read_canonical_episode(episode_path)


def test_split_is_derived_from_registry_not_episode_metadata(tmp_path: Path) -> None:
    episode_path = build_episode(tmp_path, episode_id="val-a-000001")
    registry = build_registry([(episode_path, "val")])
    episode = read_canonical_episode(episode_path, split_registry=registry)
    assert episode.split == "val"
    assert "split" not in episode.meta
    assert "robot_role" not in episode.meta


def test_valid_arm_b_action_is_rejected_by_role_e_projection(tmp_path: Path) -> None:
    episode_path = build_episode(
        tmp_path,
        episode_id="arm-b-000001",
        arm_id="Arm_B",
        executor="openvla_oft",
    )
    registry = build_registry([(episode_path, "train")])
    with pytest.raises(CanonicalV1Error, match="non-Arm_A/pi05"):
        read_canonical_episode(episode_path, split_registry=registry)


def test_missing_exact_camera_tick_fails_without_nearest_fallback(
    tmp_path: Path,
) -> None:
    episode_path = build_episode(tmp_path)
    with h5py.File(episode_path / "episode.h5", "r+") as h5:
        for camera_id in CAMERA_IDS:
            h5[f"cameras/{camera_id}/physics_tick"][0] = 4
    refresh_hdf5_sha(episode_path)
    registry = build_registry([(episode_path, "train")])
    with pytest.raises(CanonicalV1Error, match="no CAM_A_TOP sample"):
        read_canonical_episode(episode_path, split_registry=registry)


def test_missing_exact_state_tick_fails_without_nearest_fallback(
    tmp_path: Path,
) -> None:
    episode_path = build_episode(tmp_path)
    with h5py.File(episode_path / "episode.h5", "r+") as h5:
        for arm_id in ARM_IDS:
            h5[f"robot_state/{arm_id}/physics_tick"][0] = 2
    refresh_hdf5_sha(episode_path)
    registry = build_registry([(episode_path, "train")])
    with pytest.raises(CanonicalV1Error, match="no Arm_A state"):
        read_canonical_episode(episode_path, split_registry=registry)


def test_cam_a_fallback_frame_is_rejected(tmp_path: Path) -> None:
    episode_path = build_episode(tmp_path, fallback=True)
    registry = build_registry([(episode_path, "train")])
    with pytest.raises(CanonicalV1Error, match="fallback frames are forbidden"):
        read_canonical_episode(episode_path, split_registry=registry)


def test_masked_only_episode_is_rejected_by_authoritative_reader(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="at least one valid action"):
        build_episode(tmp_path, valid_mask=False)


def test_dataset_enumeration_accepts_only_hdf5_contract(tmp_path: Path) -> None:
    episode_path = build_episode(tmp_path)
    registry = build_registry([(episode_path, "train")])
    episodes = read_canonical_dataset(tmp_path, split_registry=registry)
    assert [episode.episode_id for episode in episodes] == ["train-a-000001"]

    old_root = tmp_path / "old-format"
    old_root.mkdir()
    (old_root / "meta.json").write_text("{}", encoding="utf-8")
    (old_root / "steps.jsonl").write_text("{}\n", encoding="utf-8")
    assert [episode.episode_id for episode in episodes] == ["train-a-000001"]
