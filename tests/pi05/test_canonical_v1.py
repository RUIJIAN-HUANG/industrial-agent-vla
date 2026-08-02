from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from PIL import Image

from scripts.pi05.canonical_v1 import (
    CANONICAL_QUATERNION_ORDER_XYZW,
    CANONICAL_TCP_POSE_ORDER,
    LIBRARY_QUATERNION_ORDER_WXYZ,
    CanonicalPi05StateMapper,
    CanonicalV1Error,
    map_state,
    quaternion_xyzw_to_rotation_vector,
    read_canonical_episode,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refresh_checksums(episode_dir: Path) -> None:
    files = [episode_dir / "meta.json", episode_dir / "steps.jsonl"]
    files.extend(sorted((episode_dir / "rgb" / "CAM_A_TOP").glob("*")))
    lines = [
        f"{_sha256(path)}  {path.relative_to(episode_dir).as_posix()}"
        for path in files
        if path.is_file()
    ]
    (episode_dir / "checksums.sha256").write_text("\n".join(lines) + "\n")


def read_steps(episode_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in (episode_dir / "steps.jsonl").read_text().splitlines()
    ]


def write_steps(episode_dir: Path, steps: list[dict[str, Any]]) -> None:
    (episode_dir / "steps.jsonl").write_text(
        "\n".join(json.dumps(step, allow_nan=True) for step in steps) + "\n",
        encoding="utf-8",
    )
    refresh_checksums(episode_dir)


def build_episode(
    root: Path,
    episode_id: str = "train-a-000001",
    *,
    split: str = "train",
    eligible: bool = True,
    step_count: int = 12,
    sentinel: float = 1.0,
    valid: Callable[[int], bool] | None = None,
) -> Path:
    episode_dir = root / episode_id
    image_dir = episode_dir / "rgb" / "CAM_A_TOP"
    image_dir.mkdir(parents=True)
    meta = {
        "schema_version": "1.0",
        "episode_id": episode_id,
        "scenario_group_id": f"scenario-{episode_id}",
        "split": split,
        "scene_seed": 20260803,
        "asset_variant": "assets-v1",
        "task_id": "pack_handoff_v1",
        "instruction": f"instruction for {episode_id}",
        "robot_role": "arm_a_pi05",
        "scene_config_sha256": "a" * 64,
        "controller_version": "b" * 40,
        "recorder_version": "c" * 40,
        "camera_ids": ["CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP"],
        "control_hz": 60,
        "render_hz": 30,
        "started_at": "2026-08-03T00:00:00+08:00",
        "ended_at": "2026-08-03T00:01:00+08:00",
        "outcome": "success",
        "dataset_failure_label": None,
        "parent_episode_id": None,
        "eligible_for_imitation": eligible,
    }
    (episode_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8"
    )
    steps: list[dict[str, Any]] = []
    for index in range(step_count):
        relative = f"rgb/CAM_A_TOP/{index:06d}.png"
        Image.new(
            "RGB",
            (1280, 720),
            color=(index % 256, (index * 3) % 256, (index * 7) % 256),
        ).save(episode_dir / relative)
        steps.append(
            {
                "step_index": index,
                "timestamp_ns": 1_000_000_000 + index * 100_000_000,
                "observation_id": f"{episode_id}-obs-{index}",
                "rgb": {"CAM_A_TOP": relative},
                "wrist_image": None,
                "joint_position": [sentinel + index, sentinel + index + 0.25],
                "joint_velocity": [0.1, 0.2],
                "tcp_pose": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "gripper_state": 1.0,
                "robot": {
                    "arm_a": {
                        "retreated": False,
                        "gripper_open": index % 2 == 0,
                    },
                    "arm_b": {"retreated": True},
                },
                "action_7d": [
                    sentinel + index / 100.0,
                    0.002,
                    0.003,
                    0.004,
                    0.005,
                    0.006,
                    1.0,
                ],
                "action_duration_s": 0.1,
                "agent_state": "EXECUTING",
                "operation_phase": "ARM_A_PACKING",
                "handoff_token": "A_ONLY",
                "safety_flags": [],
                "valid_for_training": valid(index) if valid else True,
            }
        )
    write_steps(episode_dir, steps)
    return episode_dir


def mutate_meta(episode_dir: Path, key: str, value: Any) -> None:
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    meta[key] = value
    (episode_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    refresh_checksums(episode_dir)


def delete_meta(episode_dir: Path, key: str) -> None:
    meta = json.loads((episode_dir / "meta.json").read_text(encoding="utf-8"))
    del meta[key]
    (episode_dir / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    refresh_checksums(episode_dir)


def assert_context(error: CanonicalV1Error, field: str) -> None:
    text = str(error)
    assert "episode_id=" in text
    assert "step_index=" in text
    assert f"field='{field}'" in text


def test_valid_canonical_v1_preserves_raw_fields(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    episode = read_canonical_episode(episode_dir)
    assert episode.episode_id == episode_dir.name
    assert episode.split == "train"
    assert len(episode.steps) == 12
    step = episode.steps[0]
    assert step.joint_position.tolist() == [1.0, 1.25]
    assert step.joint_velocity.tolist() == [0.1, 0.2]
    assert step.tcp_pose.shape == (7,)
    assert step.action_7d.shape == (7,)
    assert step.action_7d.dtype == np.float32
    assert step.cam_a_top_relative_path.startswith("rgb/CAM_A_TOP/")


def test_wrist_image_must_be_present_and_null(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    steps = read_steps(episode_dir)
    del steps[0]["wrist_image"]
    write_steps(episode_dir, steps)
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, "wrist_image")


@pytest.mark.parametrize(
    ("key", "value", "field"),
    [
        ("robot_role", "arm_b_openvla", "robot_role"),
        ("split", "validation", "split"),
        ("schema_version", "0.9", "schema_version"),
    ],
)
def test_invalid_episode_metadata_is_rejected(
    tmp_path: Path, key: str, value: Any, field: str
) -> None:
    episode_dir = build_episode(tmp_path)
    mutate_meta(episode_dir, key, value)
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, field)


@pytest.mark.parametrize(
    "field",
    [
        "schema_version",
        "episode_id",
        "scenario_group_id",
        "split",
        "scene_seed",
        "asset_variant",
        "task_id",
        "instruction",
        "robot_role",
        "scene_config_sha256",
        "controller_version",
        "recorder_version",
        "camera_ids",
        "control_hz",
        "render_hz",
        "started_at",
        "ended_at",
        "outcome",
        "dataset_failure_label",
        "parent_episode_id",
        "eligible_for_imitation",
    ],
)
def test_every_required_meta_field_is_enforced(tmp_path: Path, field: str) -> None:
    episode_dir = build_episode(tmp_path)
    delete_meta(episode_dir, field)
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, field)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("schema_version", None),
        ("episode_id", None),
        ("scenario_group_id", 1),
        ("split", []),
        ("scene_seed", True),
        ("asset_variant", {}),
        ("task_id", 1),
        ("instruction", False),
        ("robot_role", []),
        ("scene_config_sha256", 1),
        ("controller_version", None),
        ("recorder_version", []),
        ("camera_ids", "CAM_A_TOP"),
        ("control_hz", True),
        ("render_hz", 30.0),
        ("started_at", 1),
        ("ended_at", []),
        ("outcome", {}),
        ("dataset_failure_label", []),
        ("parent_episode_id", False),
        ("eligible_for_imitation", 1),
    ],
)
def test_every_required_meta_field_type_is_enforced(
    tmp_path: Path, field: str, value: Any
) -> None:
    episode_dir = build_episode(tmp_path)
    mutate_meta(episode_dir, field, value)
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, field)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("scene_seed", True),
        ("scene_seed", -1),
        ("scene_config_sha256", "bad"),
        ("controller_version", "main"),
        ("recorder_version", "abc"),
        ("camera_ids", []),
        ("camera_ids", ["CAM_A_TOP", "CAM_A_TOP"]),
        ("camera_ids", ["CAM_B_TOP"]),
        ("control_hz", 30),
        ("render_hz", 60),
        ("started_at", "2026-08-03"),
        ("ended_at", "2026-08-02T00:00:00+08:00"),
        ("outcome", "invalid"),
        ("dataset_failure_label", 3),
        ("parent_episode_id", False),
    ],
)
def test_invalid_complete_meta_values_are_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    episode_dir = build_episode(tmp_path)
    mutate_meta(episode_dir, field, value)
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, field)


def test_episode_id_must_match_directory(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    mutate_meta(episode_dir, "episode_id", "different-episode")
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, "episode_id")


def test_eligible_false_is_valid_but_has_no_training_steps(tmp_path: Path) -> None:
    episode = read_canonical_episode(build_episode(tmp_path, eligible=False))
    assert episode.eligible_for_imitation is False
    assert episode.training_steps == ()
    with pytest.raises(CanonicalV1Error, match="not eligible"):
        _ = episode.imitation_steps


def test_valid_false_is_excluded_without_becoming_a_bad_step(tmp_path: Path) -> None:
    episode = read_canonical_episode(
        build_episode(tmp_path, valid=lambda index: index != 3)
    )
    assert len(episode.steps) == 12
    assert [step.step_index for step in episode.training_steps] == [
        index for index in range(12) if index != 3
    ]
    with pytest.raises(CanonicalV1Error, match="rejects the entire Episode"):
        _ = episode.imitation_steps


def test_frozen_quaternion_orders_and_production_mapper(tmp_path: Path) -> None:
    assert CANONICAL_TCP_POSE_ORDER == ("x", "y", "z", "qx", "qy", "qz", "qw")
    assert CANONICAL_QUATERNION_ORDER_XYZW == ("qx", "qy", "qz", "qw")
    assert LIBRARY_QUATERNION_ORDER_WXYZ == ("qw", "qx", "qy", "qz")
    episode = read_canonical_episode(build_episode(tmp_path))
    mapper = CanonicalPi05StateMapper()
    state = map_state(mapper, episode, episode.steps[0])
    assert state.dtype == np.float32
    assert state.tolist() == pytest.approx([0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0])
    steps = read_steps(episode.root)
    steps[0]["gripper_state"] = -999.0
    write_steps(episode.root, steps)
    reread = read_canonical_episode(episode.root)
    assert map_state(mapper, reread, reread.steps[0])[6] == 1.0
    assert map_state(mapper, reread, reread.steps[1])[6] == 0.0


@pytest.mark.parametrize(
    ("quaternion", "expected"),
    [
        ([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0]),
        ([0.0, 0.0, 0.0, -1.0], [0.0, 0.0, 0.0]),
        ([1.0, 0.0, 0.0, 0.0], [np.pi, 0.0, 0.0]),
    ],
)
def test_quaternion_xyzw_shortest_rotation_vector(
    quaternion: list[float], expected: list[float]
) -> None:
    result = quaternion_xyzw_to_rotation_vector(
        np.asarray(quaternion), episode_id="episode", step_index=3
    )
    assert result.dtype == np.float32
    assert result.tolist() == pytest.approx(expected, abs=1e-6)


def test_zero_norm_quaternion_has_full_context() -> None:
    with pytest.raises(CanonicalV1Error) as captured:
        quaternion_xyzw_to_rotation_vector(
            np.zeros(4), episode_id="episode", step_index=4
        )
    assert_context(captured.value, "tcp_pose.quaternion_xyzw")


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_quaternion_has_full_context(bad_value: float) -> None:
    quaternion = np.asarray([0.0, bad_value, 0.0, 1.0])
    with pytest.raises(CanonicalV1Error) as captured:
        quaternion_xyzw_to_rotation_vector(
            quaternion, episode_id="episode", step_index=5
        )
    assert_context(captured.value, "tcp_pose.quaternion_xyzw")


@pytest.mark.parametrize("value", [None, 1, 0.5, "open"])
def test_arm_a_gripper_open_must_be_controller_boolean(
    tmp_path: Path, value: Any
) -> None:
    episode_dir = build_episode(tmp_path)
    steps = read_steps(episode_dir)
    if value is None:
        del steps[0]["robot"]["arm_a"]["gripper_open"]
    else:
        steps[0]["robot"]["arm_a"]["gripper_open"] = value
    write_steps(episode_dir, steps)
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, "robot.arm_a.gripper_open")


def test_non_contiguous_step_is_rejected(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    steps = read_steps(episode_dir)
    steps[4]["step_index"] = 8
    write_steps(episode_dir, steps)
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, "step_index")


def test_non_monotonic_timestamp_is_rejected(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    steps = read_steps(episode_dir)
    steps[5]["timestamp_ns"] = steps[4]["timestamp_ns"]
    write_steps(episode_dir, steps)
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, "timestamp_ns")


@pytest.mark.parametrize("duplicate", [False, True])
def test_observation_id_must_be_present_and_unique(
    tmp_path: Path, duplicate: bool
) -> None:
    episode_dir = build_episode(tmp_path)
    steps = read_steps(episode_dir)
    if duplicate:
        steps[1]["observation_id"] = steps[0]["observation_id"]
    else:
        del steps[1]["observation_id"]
    write_steps(episode_dir, steps)
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, "observation_id")


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    (episode_dir / "rgb/CAM_A_TOP/000002.png").unlink()
    with pytest.raises(CanonicalV1Error, match="does not exist"):
        read_canonical_episode(episode_dir)


def test_bad_image_is_rejected(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    (episode_dir / "rgb/CAM_A_TOP/000002.png").write_bytes(b"not an image")
    refresh_checksums(episode_dir)
    with pytest.raises(CanonicalV1Error, match="cannot be decoded"):
        read_canonical_episode(episode_dir)


def test_wrong_image_size_is_rejected(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    Image.new("RGB", (640, 480)).save(episode_dir / "rgb/CAM_A_TOP/000002.png")
    refresh_checksums(episode_dir)
    with pytest.raises(CanonicalV1Error, match="1280x720"):
        read_canonical_episode(episode_dir)


def test_non_rgb_image_is_rejected(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    Image.new("L", (1280, 720)).save(episode_dir / "rgb/CAM_A_TOP/000002.png")
    refresh_checksums(episode_dir)
    with pytest.raises(CanonicalV1Error, match="mode must be RGB"):
        read_canonical_episode(episode_dir)


def test_image_path_escape_is_rejected(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    steps = read_steps(episode_dir)
    steps[0]["rgb"]["CAM_A_TOP"] = "../outside.png"
    write_steps(episode_dir, steps)
    with pytest.raises(CanonicalV1Error, match="escapes"):
        read_canonical_episode(episode_dir)


def test_checksum_mismatch_is_rejected(tmp_path: Path) -> None:
    episode_dir = build_episode(tmp_path)
    image = episode_dir / "rgb/CAM_A_TOP/000002.png"
    image.write_bytes(image.read_bytes() + b"tamper")
    with pytest.raises(CanonicalV1Error, match="SHA-256 mismatch"):
        read_canonical_episode(episode_dir)


@pytest.mark.parametrize("mode", ["missing", "empty", "malformed"])
def test_checksum_sidecar_is_required_and_strict(tmp_path: Path, mode: str) -> None:
    episode_dir = build_episode(tmp_path)
    sidecar = episode_dir / "checksums.sha256"
    if mode == "missing":
        sidecar.unlink()
    elif mode == "empty":
        sidecar.write_text("", encoding="utf-8")
    else:
        sidecar.write_text("not-a-sha  meta.json\n", encoding="utf-8")
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, "checksums.sha256")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("action_7d", [0.0] * 6),
        ("action_7d", [0.0, 0.0, 0.0, float("nan"), 0.0, 0.0, 1.0]),
        ("action_7d", [0.0, 0.0, 0.0, float("inf"), 0.0, 0.0, 1.0]),
        ("tcp_pose", [0.0] * 6),
        ("tcp_pose", [0.0, 0.0, 0.0, 0.0, 0.0, float("nan"), 1.0]),
    ],
)
def test_invalid_physical_vectors_are_rejected(
    tmp_path: Path, field: str, value: list[float]
) -> None:
    episode_dir = build_episode(tmp_path)
    steps = read_steps(episode_dir)
    steps[2][field] = value
    write_steps(episode_dir, steps)
    with pytest.raises(CanonicalV1Error) as captured:
        read_canonical_episode(episode_dir)
    assert_context(captured.value, field)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("wrist_image", {"path": "rgb/CAM_WRIST/000000.png"}),
        ("arm_b_state", [0.0, 1.0]),
        ("offline_gt", "offline_gt/train-a-000001.json"),
    ],
)
def test_forbidden_wrist_arm_b_and_gt_are_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    episode_dir = build_episode(tmp_path)
    steps = read_steps(episode_dir)
    steps[1][field] = value
    write_steps(episode_dir, steps)
    with pytest.raises(CanonicalV1Error, match="forbidden"):
        read_canonical_episode(episode_dir)


@pytest.mark.parametrize("legacy_name", ["steps.parquet", "steps.hdf5", "front_rgb"])
def test_legacy_formats_are_explicitly_rejected(
    tmp_path: Path, legacy_name: str
) -> None:
    episode_dir = build_episode(tmp_path)
    legacy = episode_dir / legacy_name
    legacy.mkdir() if legacy_name == "front_rgb" else legacy.write_bytes(b"legacy")
    with pytest.raises(CanonicalV1Error, match=r"meta.json \+ steps.jsonl"):
        read_canonical_episode(episode_dir)
