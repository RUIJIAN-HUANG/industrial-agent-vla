from __future__ import annotations

import pytest

from openvla_oft.dataset import build_model_input, build_training_sample
from openvla_oft.exceptions import ServiceError


def _image(camera_id: str = "CAM_B_TOP") -> dict[str, object]:
    digest = "a" * 64
    return {
        "uri": f"cas://sha256/{digest}",
        "image_sha256": f"sha256:{digest}",
        "camera_id": camera_id,
        "width": 1280,
        "height": 720,
    }


def _canonical_step() -> dict[str, object]:
    return {
        "episode_id": "episode-0001",
        "task_id": "episode-0001:S02_ARM_B_TRANSPORT",
        "subtask_id": "S02_ARM_B_TRANSPORT",
        "step_id": 3,
        "robot_role": "arm_b_openvla",
        "camera": {
            "arm_b_rgb": _image(),
            "wrist_image": None,
        },
        "robot": {
            "arm_b": {
                "state": [0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 0.5],
                "tcp_pose_m_rad": [0.4, 0.0, 0.4, 0.0, 0.0, 0.0],
                "gripper_open": False,
            }
        },
        "action": [
            [0.0, 0.015, 0.0, 0.0, 0.0, 0.0, -0.75],
            [0.0, 0.010, 0.010, 0.0, 0.0, 0.0, 0.75],
        ],
    }


def test_build_training_sample_accepts_frozen_arm_b_step():
    sample = build_training_sample(
        _canonical_step(),
        task_description="transport the bin after handoff_ready",
    )

    assert sample["robot_role"] == "arm_b_openvla"
    assert sample["model_input"]["full_image"]["camera_id"] == "CAM_B_TOP"
    assert sample["model_input"]["wrist_image"] is None
    assert len(sample["model_input"]["state"]) == 7
    assert len(sample["action"]) == 2


def test_build_model_input_derives_state_from_tcp_pose_when_state_absent():
    step = _canonical_step()
    robot = step["robot"]
    assert isinstance(robot, dict)
    arm_b = robot["arm_b"]
    assert isinstance(arm_b, dict)
    del arm_b["state"]
    arm_b["gripper_open"] = True

    model_input = build_model_input(
        step,
        task_description="transport the bin after handoff_ready",
    )

    assert model_input["state"] == [0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 1.0]


def test_build_training_sample_rejects_non_null_wrist_image():
    step = _canonical_step()
    camera = step["camera"]
    assert isinstance(camera, dict)
    camera["wrist_image"] = _image("CAM_B_WRIST")

    with pytest.raises(ServiceError, match="wrist_image must be null"):
        build_training_sample(step, task_description="transport")


def test_build_training_sample_rejects_arm_a_role():
    step = _canonical_step()
    step["subtask_id"] = "S01_ARM_A_PACK_HANDOFF"

    with pytest.raises(ServiceError, match="S02 Arm_B"):
        build_training_sample(step, task_description="transport")


def test_build_training_sample_rejects_wrong_camera():
    step = _canonical_step()
    camera = step["camera"]
    assert isinstance(camera, dict)
    camera["arm_b_rgb"] = _image("CAM_HANDOFF")

    with pytest.raises(ServiceError, match="CAM_B_TOP"):
        build_training_sample(step, task_description="transport")


def test_build_training_sample_rejects_missing_arm_b_rgb_even_with_full_image():
    step = _canonical_step()
    camera = step["camera"]
    assert isinstance(camera, dict)
    camera["full_image"] = _image("CAM_HANDOFF")
    del camera["arm_b_rgb"]

    with pytest.raises(ServiceError, match="camera.arm_b_rgb"):
        build_training_sample(step, task_description="transport")


def test_build_training_sample_rejects_digest_mismatch():
    step = _canonical_step()
    camera = step["camera"]
    assert isinstance(camera, dict)
    image = camera["arm_b_rgb"]
    assert isinstance(image, dict)
    image["image_sha256"] = f"sha256:{'b' * 64}"

    with pytest.raises(ServiceError, match="digest must match"):
        build_training_sample(step, task_description="transport")


def test_build_training_sample_rejects_state_with_extra_values():
    step = _canonical_step()
    robot = step["robot"]
    assert isinstance(robot, dict)
    arm_b = robot["arm_b"]
    assert isinstance(arm_b, dict)
    arm_b["state"] = [0.0] * 8

    with pytest.raises(ServiceError, match="exactly 7"):
        build_training_sample(step, task_description="transport")


def test_build_training_sample_rejects_missing_episode_id():
    step = _canonical_step()
    del step["episode_id"]

    with pytest.raises(ServiceError, match="episode_id"):
        build_training_sample(step, task_description="transport")


def test_build_training_sample_rejects_invalid_action_shape():
    step = _canonical_step()
    step["action"] = [[0.0, 0.0]]

    with pytest.raises(ServiceError, match="exactly 7"):
        build_training_sample(step, task_description="transport")
