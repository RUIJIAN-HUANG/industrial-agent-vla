"""Canonical episode helpers for OpenVLA-OFT Arm_B training inputs."""

from __future__ import annotations

import re
from typing import Any, Mapping

from .exceptions import ServiceError
from .utils import finite_vector, validate_action_matrix

ARM_B_SUBTASK_ID = "S02_ARM_B_TRANSPORT"
ARM_B_ROLE = "arm_b_openvla"
ARM_B_CAMERA_ID = "CAM_B_TOP"
FROZEN_IMAGE_SIZE = (1280, 720)
OPENVLA_MODEL_INPUT_KEYS = frozenset(
    {"task_description", "full_image", "wrist_image", "state"}
)
_CAS_URI_PATTERN = re.compile(r"^cas://sha256/([0-9a-fA-F]{64})$")
_SHA256_PATTERN = re.compile(r"^sha256:([0-9a-fA-F]{64})$")


def build_training_sample(
    step: Mapping[str, Any],
    *,
    task_description: str,
    max_chunk_steps: int = 32,
) -> dict[str, Any]:
    """Build one OpenVLA-OFT training sample from a canonical Arm_B step.

    This is intentionally stricter than a generic loader.  It accepts only the
    frozen Arm_B transport role, the single top-view camera, no wrist image,
    a 7-D Arm_B state, and an ``N x 7`` canonical action chunk.
    """

    if not isinstance(step, Mapping):
        raise _bad_sample("canonical step must be an object")
    _validate_arm_b_role(step)
    model_input = build_model_input(step, task_description=task_description)
    action = _extract_action(step, max_chunk_steps=max_chunk_steps)
    task_id = _non_empty_string(step.get("task_id"), "task_id")
    episode_id = _non_empty_string(step.get("episode_id"), "episode_id")
    return {
        "task_id": task_id,
        "episode_id": episode_id,
        "step_id": _extract_step_id(step),
        "robot_role": ARM_B_ROLE,
        "model_input": model_input,
        "action": action,
    }


def build_model_input(
    step: Mapping[str, Any],
    *,
    task_description: str,
) -> dict[str, Any]:
    """Return the exact ``/v1/infer`` ``model_input`` for Arm_B OpenVLA-OFT."""

    if not isinstance(task_description, str) or not task_description.strip():
        raise _bad_sample("task_description must be a non-empty string")
    full_image = _extract_arm_b_image(step)
    _reject_wrist_image(step)
    state = _extract_arm_b_state(step)
    model_input = {
        "task_description": task_description,
        "full_image": full_image,
        "wrist_image": None,
        "state": state,
    }
    if set(model_input) != OPENVLA_MODEL_INPUT_KEYS:
        raise AssertionError("OpenVLA model_input keys drifted")
    return model_input


def _validate_arm_b_role(step: Mapping[str, Any]) -> None:
    subtask_id = step.get("subtask_id")
    if subtask_id is not None and subtask_id != ARM_B_SUBTASK_ID:
        raise _bad_sample("OpenVLA-OFT training samples must use S02 Arm_B transport")
    robot_role = step.get("robot_role")
    if robot_role is not None and robot_role != ARM_B_ROLE:
        raise _bad_sample(
            "OpenVLA-OFT training samples must use robot_role=arm_b_openvla"
        )


def _extract_arm_b_image(step: Mapping[str, Any]) -> dict[str, Any]:
    camera = step.get("camera")
    if not isinstance(camera, Mapping):
        raise _bad_sample("canonical step.camera must be an object")
    image = camera.get("arm_b_rgb")
    return _image_reference(image, field_name="camera.arm_b_rgb")


def _reject_wrist_image(step: Mapping[str, Any]) -> None:
    camera = step.get("camera")
    if isinstance(camera, Mapping) and camera.get("wrist_image") is not None:
        raise _bad_sample(
            "frozen Isaac scene has no wrist camera; wrist_image must be null"
        )
    if step.get("wrist_image") is not None:
        raise _bad_sample(
            "frozen Isaac scene has no wrist camera; wrist_image must be null"
        )


def _extract_arm_b_state(step: Mapping[str, Any]) -> list[float]:
    robot = step.get("robot")
    if not isinstance(robot, Mapping):
        raise _bad_sample("canonical step.robot must be an object")
    arm_b = robot.get("arm_b")
    if not isinstance(arm_b, Mapping):
        raise _bad_sample("canonical step.robot.arm_b must be an object")
    state = arm_b.get("state")
    if state is not None:
        values = finite_vector(state, "robot.arm_b.state", min_length=7)
        if len(values) != 7:
            raise _bad_sample("robot.arm_b.state must contain exactly 7 values")
        return values
    tcp_pose = finite_vector(
        arm_b.get("tcp_pose_m_rad"),
        "robot.arm_b.tcp_pose_m_rad",
        min_length=6,
    )
    gripper_open = arm_b.get("gripper_open")
    if not isinstance(gripper_open, bool):
        raise _bad_sample(
            "robot.arm_b.gripper_open is required when robot.arm_b.state is absent"
        )
    gripper_norm = 1.0 if gripper_open else 0.0
    return [*tcp_pose[:6], gripper_norm]


def _extract_action(
    step: Mapping[str, Any],
    *,
    max_chunk_steps: int,
) -> list[list[float]]:
    action = step.get("action")
    if action is None:
        action_chunk = step.get("action_chunk")
        if isinstance(action_chunk, Mapping):
            steps = action_chunk.get("steps")
            if isinstance(steps, list):
                action = [
                    item.get("values") if isinstance(item, Mapping) else item
                    for item in steps
                ]
    return validate_action_matrix(action, max_steps=max_chunk_steps)


def _image_reference(value: Any, *, field_name: str) -> dict[str, Any]:
    required = {"uri", "image_sha256", "camera_id", "width", "height"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise _bad_sample(f"{field_name} must be an ImageReference")
    uri = value["uri"]
    image_sha256 = value["image_sha256"]
    uri_match = _CAS_URI_PATTERN.fullmatch(uri) if isinstance(uri, str) else None
    sha_match = (
        _SHA256_PATTERN.fullmatch(image_sha256)
        if isinstance(image_sha256, str)
        else None
    )
    if uri_match is None or sha_match is None:
        raise _bad_sample(
            f"{field_name} must use cas://sha256/<digest> and sha256:<digest>"
        )
    if uri_match.group(1).casefold() != sha_match.group(1).casefold():
        raise _bad_sample(f"{field_name}.uri digest must match image_sha256")
    if value["camera_id"] != ARM_B_CAMERA_ID:
        raise _bad_sample(f"{field_name}.camera_id must be {ARM_B_CAMERA_ID}")
    if (value["width"], value["height"]) != FROZEN_IMAGE_SIZE:
        raise _bad_sample(f"{field_name} must be 1280x720")
    return dict(value)


def _extract_step_id(step: Mapping[str, Any]) -> int:
    value = step.get("step_id", step.get("step_index", 0))
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _bad_sample("step_id must be a non-negative integer")
    return value


def _non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise _bad_sample(f"{field_name} must be a non-empty string")
    return value


def _bad_sample(message: str) -> ServiceError:
    return ServiceError("DATA_3001_INVALID_SAMPLE", message, retryable=False)
