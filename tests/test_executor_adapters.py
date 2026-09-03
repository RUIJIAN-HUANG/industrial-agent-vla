from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from industrial_agent.contracts import Postcondition, TaskSchema
from industrial_agent.errors import ExecutorError, FailureCode
from industrial_agent.executor import (
    ExecutionContext,
    Pi05Adapter,
    build_executors_from_config,
)
from industrial_agent.observation import ObservationGateway
from tests.test_contracts_and_observation import raw_observation


CHECKPOINT_SHA = f"sha256:{'1' * 64}"
NORM_STATS_SHA = f"sha256:{'2' * 64}"


class EchoTransport:
    def __init__(self, *, health_overrides: Mapping[str, Any] | None = None):
        self.calls: list[tuple[str, Mapping[str, Any], int]] = []
        self.health_overrides = dict(health_overrides or {})

    def request(
        self, route: str, payload: Mapping[str, Any], timeout_ms: int
    ) -> Mapping[str, Any]:
        self.calls.append((route, payload, timeout_ms))
        if route == "/health":
            response: dict[str, Any] = {
                "schema_version": "1.0",
                "service": "pi05",
                "status": "ready",
                "checkpoint_sha": CHECKPOINT_SHA,
                "norm_stats_sha": NORM_STATS_SHA,
                "supported_task_types": [
                    "pick_place",
                    "visual_manipulation",
                    "instruction_interaction",
                ],
                "supported_action_contracts": ["1.0"],
            }
            response.update(self.health_overrides)
            return response
        if route == "/v1/cancel":
            return {"status": "cancelled"}
        return {
            **{
                key: payload[key]
                for key in (
                    "schema_version",
                    "request_id",
                    "trace_id",
                    "episode_id",
                    "task_id",
                    "subtask_id",
                    "step_id",
                    "observation_id",
                    "executor",
                    "checkpoint_sha",
                    "norm_stats_sha",
                )
            },
            "status": "ok",
            "action_chunk": {
                "contract_version": "1.0",
                "chunk_id": "canonical-chunk",
                "task_id": payload["task_id"],
                "executor": "pi05",
                "action_space": "ee_delta_pose_gripper",
                "frame": "robot_base",
                "translation_unit": "m",
                "rotation_unit": "rad",
                "gripper_unit": "normalized",
                "steps": [
                    {
                        "values": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                        "duration_ms": 137,
                    }
                ],
            },
            "timing": {"queue_ms": 1, "inference_ms": 2, "total_ms": 3},
        }


def _observation() -> Any:
    raw = raw_observation()

    def image(camera_id: str, digest_char: str) -> dict[str, object]:
        digest = digest_char * 64
        return {
            "uri": f"cas://sha256/{digest}",
            "image_sha256": f"sha256:{digest}",
            "camera_id": camera_id,
            "width": 1280,
            "height": 720,
        }

    raw["camera"] = {
        "full_image": image("CAM_A_TOP", "a"),
        "arm_a_rgb": image("CAM_A_TOP", "b"),
        "handoff_rgb": image("CAM_HANDOFF", "c"),
        "arm_b_rgb": image("CAM_B_TOP", "d"),
        "wrist_image": None,
    }
    robot = raw["robot"]
    assert isinstance(robot, dict)
    robot["arm_a"] = {
        "tcp_pose_m_rad": [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
        "state": [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.375],
        "retreated": False,
        "gripper_open": False,
        "stationary": True,
    }
    robot["arm_b"] = {
        "tcp_pose_m_rad": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0],
        "state": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
        "retreated": True,
        "gripper_open": False,
        "stationary": True,
    }
    return ObservationGateway().ingest_online(raw)


def _task(arm_id: str = "Arm_A") -> TaskSchema:
    return TaskSchema(
        task_id="parent:subtask",
        instruction="execute semantic action",
        task_type="pick_place",
        metadata={"subtask_id": "subtask", "arm_id": arm_id},
        postconditions=(
            Postcondition(kind="field_equals", path="task.status", expected="done"),
        ),
    )


def _context(arm_id: str | None = None) -> ExecutionContext:
    return ExecutionContext(
        run_id="episode-1",
        strategy_attempt=1,
        replan_index=0,
        step_id=4,
        arm_id=arm_id,
    )


def test_pi05_routes_each_arm_to_its_camera_and_state() -> None:
    observation = _observation()
    for arm_id, camera_id, state_x, gripper_opening in (
        ("Arm_A", "CAM_A_TOP", 0.5, 0.375),
        ("Arm_B", "CAM_B_TOP", 0.4, 0.0),
    ):
        transport = EchoTransport()
        adapter = Pi05Adapter(
            transport,
            checkpoint_sha=CHECKPOINT_SHA,
            norm_stats_sha=NORM_STATS_SHA,
        )
        result = adapter.plan(_task(arm_id), observation, _context(arm_id))
        assert result.executor == "pi05"
        payload = transport.calls[-1][1]
        assert payload["arm_id"] == arm_id
        model_input = payload["model_input"]
        assert isinstance(model_input, Mapping)
        assert model_input["arm_id"] == arm_id
        model_observation = model_input["observation"]
        assert isinstance(model_observation, Mapping)
        camera = model_observation["camera"]
        robot = model_observation["robot"]
        assert isinstance(camera, Mapping) and isinstance(robot, Mapping)
        assert camera["full_image"]["camera_id"] == camera_id
        assert robot["state"][0] == state_x
        assert robot["state"][6] == gripper_opening


def test_pi05_context_arm_overrides_task_metadata() -> None:
    transport = EchoTransport()
    adapter = Pi05Adapter(
        transport,
        checkpoint_sha=CHECKPOINT_SHA,
        norm_stats_sha=NORM_STATS_SHA,
    )
    adapter.plan(_task("Arm_A"), _observation(), _context("Arm_B"))
    assert transport.calls[-1][1]["arm_id"] == "Arm_B"


def test_pi05_health_requires_matching_identity() -> None:
    adapter = Pi05Adapter(
        EchoTransport(health_overrides={"service": "wrong"}),
        checkpoint_sha=CHECKPOINT_SHA,
        norm_stats_sha=NORM_STATS_SHA,
    )
    assert adapter.health() is False


def test_executor_factory_only_accepts_pi05(tmp_path: Path) -> None:
    config = json.loads(
        (
            Path(__file__).resolve().parents[1] / "configs" / "agent.default.json"
        ).read_text(encoding="utf-8")
    )
    config["executors"]["pi05"]["checkpoint_sha"] = CHECKPOINT_SHA
    config["executors"]["pi05"]["norm_stats_sha"] = NORM_STATS_SHA
    calls: list[tuple[str, str]] = []

    def factory(name: str, url: str) -> EchoTransport:
        calls.append((name, url))
        return EchoTransport()

    executors = build_executors_from_config(config, factory)
    assert [item.descriptor.name for item in executors] == ["pi05"]
    assert calls == [("pi05", "http://127.0.0.1:8101")]

    config["executors"]["retired"] = config["executors"]["pi05"]
    with pytest.raises(ValueError, match="unsupported executors"):
        build_executors_from_config(config, factory)


def test_pi05_rejects_invalid_arm_before_transport() -> None:
    transport = EchoTransport()
    adapter = Pi05Adapter(
        transport,
        checkpoint_sha=CHECKPOINT_SHA,
        norm_stats_sha=NORM_STATS_SHA,
    )
    with pytest.raises(ExecutorError) as caught:
        adapter.plan(_task("Arm_C"), _observation(), _context())
    assert caught.value.code is FailureCode.INVALID_TASK
    assert not transport.calls
