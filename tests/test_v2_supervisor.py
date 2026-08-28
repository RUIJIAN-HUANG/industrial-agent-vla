from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from industrial_agent.contracts import ActionChunk, ActionStep, TaskSchema
from industrial_agent.environment import SafeStopReceipt
from industrial_agent.executor import ExecutionContext, ExecutorDescriptor
from industrial_agent.fsm import AgentState
from industrial_agent.perception import (
    Detection,
    MockPerceptionAgent,
    PerceptionContext,
    PerceptionError,
)
from industrial_agent.supervisor_main import build_supervisor
from industrial_agent.v2_supervisor import V2Supervisor


ROOT = Path(__file__).resolve().parents[1]
PINNED_SHA = f"sha256:{'a' * 64}"
PERCEPTION_CHECKPOINT_SHA = (
    "sha256:2a8beca3ff52f6cd7a2f81f087df71793889d7017f81156a8286f4ffb106080f"
)
CLASS_MAP_SHA = (
    "sha256:839fdb76e458f9148959e727d289a29495130ce9c868b10b57adcaab4323ba06"
)
PERCEPTION_CONFIG_SHA = (
    "sha256:a28227b8296f736280a43e5b2defb559692fe49e14f6876cf6f918321b8f1e56"
)


def _task() -> TaskSchema:
    return TaskSchema.from_dict(
        json.loads(
            (ROOT / "configs" / "task.v2.p01-to-s11.example.json").read_text(
                encoding="utf-8"
            )
        )
    )


def _config() -> dict[str, Any]:
    config = json.loads(
        (ROOT / "configs" / "agent.default.json").read_text(encoding="utf-8")
    )
    config["executors"]["pi05"]["checkpoint_sha"] = PINNED_SHA
    config["executors"]["pi05"]["norm_stats_sha"] = PINNED_SHA
    return config


def _image(camera_id: str, digest: str = "a") -> dict[str, object]:
    return {
        "uri": f"cas://sha256/{digest * 64}",
        "image_sha256": f"sha256:{digest * 64}",
        "camera_id": camera_id,
        "width": 1280,
        "height": 720,
    }


def v2_observation(*, observation_id: str = "v2-obs-1", terminal: bool = False):
    return {
        "observation_version": "2.0",
        "observation_id": observation_id,
        "timestamp_ms": 1,
        "camera": {
            "full_image": _image("CAM_A_TOP"),
            "arm_a_rgb": _image("CAM_A_TOP"),
            "handoff_rgb": _image("CAM_HANDOFF", "b"),
            "arm_b_rgb": _image("CAM_B_TOP", "c"),
            "wrist_image": None,
        },
        "objects": [],
        "robot": {
            "active_arm": "Arm_A" if not terminal else "NONE",
            "arm_a": {
                "tcp_pose_m_rad": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0],
                "state": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "retreated": terminal,
                "gripper_open": True,
                "stationary": True,
            },
            "arm_b": {
                "tcp_pose_m_rad": [0.4, 0.4, 0.5, 0.0, 0.0, 0.0],
                "state": [0.4, 0.4, 0.5, 0.0, 0.0, 0.0, 1.0],
                "retreated": True,
                "gripper_open": True,
                "stationary": True,
            },
        },
        "safety": {
            "emergency_stop": False,
            "protective_stop": False,
            "system_fault": None,
        },
        "task": {
            "task_id": "P01_TO_S11",
            "target_object_id": "P01",
            "target_slot_id": "S11",
            "status": "SUCCEEDED" if terminal else "ACTIVE",
            "terminal": terminal,
            "terminal_confidence": 0.9 if terminal else 0.0,
            "verification_votes": 2 if terminal else 0,
        },
        "quality": {"confidence": 1.0},
    }


class _Executor:
    descriptor = ExecutorDescriptor(
        name="pi05",
        task_types=frozenset({"pick_place"}),
        action_contract_version="1.0",
        checkpoint_sha=PINNED_SHA,
        norm_stats_sha=PINNED_SHA,
    )

    def __init__(self) -> None:
        self.observations: list[Any] = []

    def health(self) -> bool:
        return True

    def plan(
        self, task: TaskSchema, observation: Any, context: ExecutionContext
    ) -> ActionChunk:
        self.observations.append(observation)
        return ActionChunk(
            contract_version="1.0",
            chunk_id=f"chunk-{context.step_id}",
            task_id=task.task_id,
            executor="pi05",
            steps=(ActionStep.from_sequence([0, 0, 0, 0, 0, 0, 1]),),
        )

    def cancel(self, task_id: str, reason: str) -> None:
        pass


def _hex_detection(context: PerceptionContext) -> tuple[Detection, ...]:
    return (
        Detection(
            detection_id=f"p01-{context.step_id}",
            class_id=2,
            class_name="hex_nut",
            confidence=0.91,
            bbox_xyxy=(120.0, 80.0, 220.0, 180.0),
            camera_id=context.image.camera_id,
            image_width=context.image.width,
            image_height=context.image.height,
        ),
    )


def _perception(
    detector: Any = _hex_detection,
) -> MockPerceptionAgent:
    return MockPerceptionAgent(
        checkpoint_sha=PERCEPTION_CHECKPOINT_SHA,
        class_map_sha=CLASS_MAP_SHA,
        config_sha=PERCEPTION_CONFIG_SHA,
        detector=detector,
    )


class _Environment:
    def __init__(self, *, terminal_after_step: bool = True) -> None:
        self.terminal_after_step = terminal_after_step
        self.steps = 0
        self.stops = 0
        self.observations = 0

    def observe(self) -> Mapping[str, Any]:
        self.observations += 1
        return v2_observation(
            observation_id=f"v2-observe-{self.observations}",
            terminal=self.steps > 0 and self.terminal_after_step,
        )

    def step(self, action: ActionStep, **kwargs: Any) -> Mapping[str, Any]:
        assert kwargs["arm_id"] == "Arm_A"
        assert kwargs["control_token"] == "A_ONLY"
        self.steps += 1
        return v2_observation(
            observation_id=f"v2-obs-{self.steps}",
            terminal=self.terminal_after_step,
        )

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        self.stops += 1
        return SafeStopReceipt(True, True, True, True, f"stop-{self.stops}")


class _Transport:
    def request(self, route: str, payload: Mapping[str, Any], timeout_ms: int):
        raise AssertionError(f"unexpected call during composition: {route}")


def test_v2_supervisor_completes_sensor_closed_loop() -> None:
    supervisor = V2Supervisor.from_config(
        _Executor(),
        _config(),
        perception=_perception(),
    )
    environment = _Environment()
    result = supervisor.run(_task(), environment)
    assert result.success is True
    assert result.state is AgentState.SUCCEEDED
    assert result.executor_history == ("pi05",)
    assert result.control_token_history == ("NONE", "A_ONLY", "NONE")
    assert environment.steps == 1


def test_v2_supervisor_stops_when_decision_budget_is_exhausted() -> None:
    config = _config()
    config["recovery"]["max_decisions_per_task"] = 1
    supervisor = V2Supervisor.from_config(
        _Executor(),
        config,
        perception=_perception(),
    )
    environment = _Environment(terminal_after_step=False)
    result = supervisor.run(_task(), environment)
    assert result.success is False
    assert result.state is AgentState.SAFE_STOPPED
    assert environment.stops == 1


def test_production_builder_wires_pi05_and_shadow_yolo_sidecar() -> None:
    calls: list[tuple[str, str]] = []

    def factory(name: str, url: str) -> _Transport:
        calls.append((name, url))
        return _Transport()

    supervisor = build_supervisor(_config(), transport_factory=factory)
    assert isinstance(supervisor, V2Supervisor)
    assert calls == [
        ("pi05", "http://127.0.0.1:8101"),
        ("yolo", "http://127.0.0.1:8103"),
    ]


def test_v2_supervisor_runs_yolo_sidecar_without_polluting_pi05_observation() -> None:
    captured: list[PerceptionContext] = []

    def detector(context: PerceptionContext) -> tuple[Detection, ...]:
        captured.append(context)
        return _hex_detection(context)

    executor = _Executor()
    supervisor = V2Supervisor.from_config(
        executor,
        _config(),
        perception=_perception(detector),
    )
    result = supervisor.run(_task(), _Environment())

    assert result.success is True
    assert [context.allowed_class_names for context in captured] == [("hex_nut",)]
    assert [context.subtask_id for context in captured] == ["P01_TO_S11"]
    assert executor.observations
    assert "detections" not in executor.observations[0].data["camera"]["arm_a_rgb"]
    requested = [
        event for event in result.events if event.event_type == "perception.requested"
    ]
    assert requested[0].payload["allowed_class_names"] == ["hex_nut"]
    locked = [
        event
        for event in result.events
        if event.event_type == "perception.target_locked"
    ]
    assert locked[0].payload["target_lock"]["object_id"] == "P01"
    assert locked[0].payload["target_lock"]["slot_id"] == "S11"
    assert locked[0].payload["target_lock"]["slot_index"] == 1
    assert locked[0].payload["target_lock"]["detection"]["zone_id"] == "A"
    assert all(
        event.payload["control_path_impact"] == "none"
        for event in result.events
        if event.event_type.startswith("perception.")
    )


def test_v2_supervisor_yolo_failure_does_not_block_vla_execution() -> None:
    def failing_detector(context: PerceptionContext) -> tuple[Detection, ...]:
        del context
        raise PerceptionError(
            FailureCode.PERCEPTION_TIMEOUT,
            "sidecar timeout",
            retryable=True,
        )

    executor = _Executor()
    supervisor = V2Supervisor.from_config(
        executor,
        _config(),
        perception=_perception(failing_detector),
    )
    environment = _Environment()
    result = supervisor.run(_task(), environment)

    assert result.success is True
    assert environment.steps == 1
    assert len(executor.observations) == 1
    failed = [
        event for event in result.events if event.event_type == "perception.failed"
    ]
    assert failed
    assert failed[0].payload["vla_recovery_untouched"] is True
    assert failed[0].payload["control_path_impact"] == "none"


def test_production_builder_rejects_abolished_v1() -> None:
    legacy = json.loads(
        (ROOT / "configs" / "agent.v1.legacy.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="V1 is abolished"):
        build_supervisor(legacy, transport_factory=lambda _name, _url: _Transport())
