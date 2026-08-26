from __future__ import annotations

from simulation.run_pi05_isaac_closed_loop import (
    _safety_policy_from_config,
    build_observation,
    build_task_state,
)
from industrial_agent.v2_observation import V2ObservationGateway
from industrial_agent.v2_task_profile import require_formal_v2_task


def test_build_task_state_contains_no_ground_truth_fields() -> None:
    task = build_task_state(require_formal_v2_task("P01_TO_S11"))
    assert task == {
        "task_id": "P01_TO_S11",
        "target_object_id": "P01",
        "target_slot_id": "S11",
        "status": "ACTIVE",
        "terminal": False,
        "terminal_confidence": 0.0,
        "verification_votes": 0,
    }


def test_build_observation_is_accepted_by_online_gateway() -> None:
    observation = build_observation(
        camera={
            "full_image": {
                "uri": f"cas://sha256/{'a' * 64}",
                "image_sha256": f"sha256:{'a' * 64}",
                "camera_id": "CAM_A_TOP",
                "width": 1280,
                "height": 720,
            },
            "arm_a_rgb": {
                "uri": f"cas://sha256/{'a' * 64}",
                "image_sha256": f"sha256:{'a' * 64}",
                "camera_id": "CAM_A_TOP",
                "width": 1280,
                "height": 720,
            },
            "handoff_rgb": {
                "uri": f"cas://sha256/{'b' * 64}",
                "image_sha256": f"sha256:{'b' * 64}",
                "camera_id": "CAM_HANDOFF",
                "width": 1280,
                "height": 720,
            },
            "arm_b_rgb": {
                "uri": f"cas://sha256/{'c' * 64}",
                "image_sha256": f"sha256:{'c' * 64}",
                "camera_id": "CAM_B_TOP",
                "width": 1280,
                "height": 720,
            },
            "wrist_image": None,
        },
        robot={
            "active_arm": "Arm_A",
            "arm_a": {
                "state": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "tcp_pose_m_rad": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0],
                "retreated": False,
                "gripper_open": True,
                "stationary": True,
            },
            "arm_b": {
                "state": [0.4, 0.4, 0.5, 0.0, 0.0, 0.0, 1.0],
                "tcp_pose_m_rad": [0.4, 0.4, 0.5, 0.0, 0.0, 0.0],
                "retreated": True,
                "gripper_open": True,
                "stationary": True,
            },
        },
        task=build_task_state(require_formal_v2_task("P01_TO_S11")),
        observation_id="closed-loop-obs-000001",
        timestamp_ms=1,
    )
    result = V2ObservationGateway().ingest_online(observation)
    assert result.observation_id == "closed-loop-obs-000001"
    assert result.data["camera"]["wrist_image"] is None


def test_safety_policy_is_loaded_from_agent_config() -> None:
    policy = _safety_policy_from_config(
        {
            "safety": {
                "axis_abs_limits": [0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0],
                "workspace_by_arm": {
                    "Arm_A": {"min_m": [0.0, -0.6, 0.0], "max_m": [0.7, 0.45, 1.0]},
                    "Arm_B": {"min_m": [0.0, -0.25, 0.0], "max_m": [0.7, 0.6, 1.0]},
                },
                "max_chunk_steps": 32,
            }
        }
    )
    assert policy.axis_abs_limits == (0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0)
    assert policy.max_chunk_steps == 32
