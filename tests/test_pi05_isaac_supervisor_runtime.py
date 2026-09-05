from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from industrial_agent.contracts import ActionChunk, ActionStep, Observation, TaskSchema
from industrial_agent.safety import ActionSafetyValidator, SafetyPolicy
from industrial_agent.environment import SafeStopReceipt
from industrial_agent.errors import FailureCode
from industrial_agent.fsm import AgentState
from industrial_agent.isaac_runtime import IsaacMainThreadGate
from simulation.pi05_isaac_supervisor_runtime import (
    _ActionRecorder,
    _RecordingEnvironment,
    Task3WorkspaceGraceSafety,
    run_supervisor_runtime,
    with_decision_budget,
)
from tests.test_v2_observation import v2_observation


ROOT = Path(__file__).resolve().parents[1]
PINNED_SHA = f"sha256:{'a' * 64}"


def _config() -> dict[str, Any]:
    config = json.loads(
        (ROOT / "configs" / "agent.default.json").read_text(encoding="utf-8")
    )
    config["executors"]["pi05"]["checkpoint_sha"] = PINNED_SHA
    config["executors"]["pi05"]["norm_stats_sha"] = PINNED_SHA
    return config


def _task() -> TaskSchema:
    return TaskSchema.from_dict(
        json.loads(
            (ROOT / "configs" / "task.v2.p01-to-s11.example.json").read_text(
                encoding="utf-8"
            )
        )
    )


def _task3() -> TaskSchema:
    return TaskSchema.from_dict(
        json.loads(
            (ROOT / "configs" / "task.v2.bin01-to-finished01.example.json").read_text(
                encoding="utf-8"
            )
        )
    )


def _action_chunk(values: list[float]) -> ActionChunk:
    return ActionChunk(
        contract_version="1.0",
        chunk_id="task3-test-chunk",
        task_id="BIN01_TO_FINISHED01",
        executor="pi05",
        steps=(ActionStep.from_sequence(values),),
    )


def _task3_observation(x: float) -> Observation:
    payload = v2_observation(observation_id="task3-observation")
    payload["robot"]["arm_a"]["tcp_pose_m_rad"][0] = x
    return Observation(
        observation_id="task3-observation",
        timestamp_ms=1,
        data=payload,
    )


def test_task3_workspace_grace_clamps_within_margin_and_rejects_beyond() -> None:
    safety = Task3WorkspaceGraceSafety(ActionSafetyValidator(SafetyPolicy()))
    observation = _task3_observation(0.679)

    clamped = safety.validate_and_limit(
        _action_chunk([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        observation,
        arm_id="Arm_A",
        control_token="A_ONLY",
    )
    assert clamped.accepted is True
    assert clamped.chunk is not None
    assert clamped.chunk.steps[0].values[0] == pytest.approx(0.020)
    assert "dx" in clamped.limited_axes

    rejected = safety.validate_and_limit(
        _action_chunk([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),
        _task3_observation(0.69),
        arm_id="Arm_A",
        control_token="A_ONLY",
    )
    assert rejected.accepted is False
    assert rejected.code is FailureCode.ACTION_WORKSPACE_BREACH


def test_task3_grace_does_not_change_p01_validator_path() -> None:
    safety = ActionSafetyValidator(SafetyPolicy())
    chunk = ActionChunk(
        contract_version="1.0",
        chunk_id="p01-test-chunk",
        task_id="P01_TO_S11",
        executor="pi05",
        steps=(ActionStep.from_sequence([0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]),),
    )
    observation = _task3_observation(0.69)
    decision = safety.validate_and_limit(
        chunk,
        observation,
        arm_id="Arm_A",
        control_token="A_ONLY",
    )
    assert decision.accepted is False
    assert decision.code is FailureCode.ACTION_WORKSPACE_BREACH


class _Transport:
    def __init__(
        self,
        *,
        health_ok: bool = True,
        infer_error: BaseException | None = None,
        out_of_bounds: bool = False,
    ) -> None:
        self.health_ok = health_ok
        self.infer_error = infer_error
        self.out_of_bounds = out_of_bounds
        self.routes: list[str] = []

    def request(
        self,
        route: str,
        payload: Mapping[str, Any],
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        del timeout_ms
        self.routes.append(route)
        if route == "/health":
            if not self.health_ok:
                raise OSError("π0.5 unavailable")
            return {
                "schema_version": "1.0",
                "service": "pi05",
                "status": "ready",
                "checkpoint_sha": PINNED_SHA,
                "norm_stats_sha": PINNED_SHA,
                "supported_task_types": [
                    "pick_place",
                    "visual_manipulation",
                    "instruction_interaction",
                ],
                "supported_action_contracts": ["1.0"],
            }
        if route == "/v1/cancel":
            return {"status": "cancelled"}
        if self.infer_error is not None:
            raise self.infer_error
        values = [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]
        if self.out_of_bounds:
            values[0] = 0.5
        return {
            "schema_version": "1.0",
            "request_id": payload["request_id"],
            "trace_id": payload["trace_id"],
            "episode_id": payload["episode_id"],
            "task_id": payload["task_id"],
            "subtask_id": payload["subtask_id"],
            "step_id": payload["step_id"],
            "observation_id": payload["observation_id"],
            "executor": "pi05",
            "checkpoint_sha": PINNED_SHA,
            "norm_stats_sha": PINNED_SHA,
            "status": "ok",
            "timing": {"queue_ms": 1, "inference_ms": 1, "total_ms": 2},
            "action_chunk": {
                "contract_version": "1.0",
                "chunk_id": f"chunk-{payload['step_id']}",
                "task_id": payload["task_id"],
                "executor": "pi05",
                "action_space": "ee_delta_pose_gripper",
                "frame": "robot_base",
                "translation_unit": "m",
                "rotation_unit": "rad",
                "gripper_unit": "normalized",
                "steps": [{"values": values, "duration_ms": 100}],
            },
        }


class _Environment:
    def __init__(
        self,
        *,
        fail_step: bool = False,
        near_workspace_edge: bool = False,
    ) -> None:
        self.steps = 0
        self.stops = 0
        self.fail_step = fail_step
        self.near_workspace_edge = near_workspace_edge

    def observe(self) -> Mapping[str, Any]:
        observation = v2_observation(observation_id=f"observation-{self.steps}")
        if self.near_workspace_edge:
            observation["robot"]["arm_a"]["tcp_pose_m_rad"][0] = 0.69
            observation["robot"]["arm_a"]["state"][0] = 0.69
        return observation

    def step(self, action: ActionStep, **kwargs: Any) -> Mapping[str, Any]:
        del action, kwargs
        if self.fail_step:
            raise RuntimeError("controller rejected action")
        self.steps += 1
        return self.observe()

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        assert reason
        self.stops += 1
        return SafeStopReceipt(True, True, True, True, f"stop-{self.stops}")


def _run(
    environment: _Environment,
    transport: _Transport,
    *,
    max_steps: int,
) -> Any:
    gate = IsaacMainThreadGate()
    try:
        return run_supervisor_runtime(
            config=_config(),
            task=_task(),
            environment=environment,
            gate=gate,
            max_steps=max_steps,
            transport_factory=lambda _name, _url: transport,
        )
    finally:
        gate.close("test complete")


def test_budget_is_copied_and_real_v2_supervisor_path_records_commands() -> None:
    config = _config()
    original = deepcopy(config)
    runtime_config = with_decision_budget(config, 2)
    assert config == original
    assert runtime_config["recovery"]["max_decisions_per_task"] == 2

    environment = _Environment()
    transport = _Transport()
    report = _run(environment, transport, max_steps=2)

    assert report.run_result.state is AgentState.SAFE_STOPPED
    assert report.run_result.success is False
    assert report.run_result.failure_code is FailureCode.RECOVERY_EXHAUSTED
    assert environment.steps == 2
    assert environment.stops == 1
    assert transport.routes == ["/health", "/v1/infer", "/v1/infer"]
    assert [record.observation_id for record in report.actions] == [
        "observation-0",
        "observation-1",
    ]
    assert [record.command_id for record in report.actions] == [
        report.actions[0].command_id,
        report.actions[1].command_id,
    ]
    assert report.actions[0].execution_result["status"] == "ACKED"


@pytest.mark.parametrize(
    ("transport", "failure_code"),
    [
        (_Transport(health_ok=False), FailureCode.EXECUTOR_UNAVAILABLE),
        (
            _Transport(infer_error=TimeoutError("inference timed out")),
            FailureCode.EXECUTOR_TIMEOUT,
        ),
        (_Transport(out_of_bounds=True), FailureCode.ACTION_WORKSPACE_BREACH),
    ],
)
def test_service_timeout_and_action_boundary_failures_stop_once(
    transport: _Transport,
    failure_code: FailureCode,
) -> None:
    environment = _Environment(near_workspace_edge=transport.out_of_bounds)
    report = _run(environment, transport, max_steps=1)
    assert report.run_result.success is False
    assert report.run_result.failure_code is failure_code
    assert report.safe_stop_confirmed is True
    assert environment.stops == 1


def test_action_failure_is_recorded_and_safe_stopped() -> None:
    environment = _Environment(fail_step=True)
    report = _run(environment, _Transport(), max_steps=1)
    assert report.run_result.success is False
    assert report.run_result.state is AgentState.SAFE_STOPPED
    assert environment.stops == 1
    assert report.actions[0].command_id
    assert report.actions[0].execution_result["status"] == "FAILED"
    assert report.actions[0].execution_result["error_type"] == "RuntimeError"


def test_sha_mismatch_setup_is_stopped_and_raised() -> None:
    config = _config()
    config["executors"]["pi05"]["checkpoint_sha"] = "not-a-sha"
    environment = _Environment()
    gate = IsaacMainThreadGate()
    try:
        with pytest.raises(ValueError, match="sha256"):
            run_supervisor_runtime(
                config=config,
                task=_task(),
                environment=environment,
                gate=gate,
                max_steps=1,
                transport_factory=lambda _name, _url: _Transport(),
            )
    finally:
        gate.close("test complete")
    assert environment.stops == 1


def test_duplicate_commands_and_safe_stops_are_idempotent() -> None:
    delegate = _Environment()
    wrapper = _RecordingEnvironment(delegate, recorder=_ActionRecorder())
    action = ActionStep.from_sequence([0, 0, 0, 0, 0, 0, 0])
    kwargs = {
        "arm_id": "Arm_A",
        "control_token": "A_ONLY",
        "command_id": "duplicate-command",
        "expected_observation_id": "observation-0",
        "expected_state_digest": "sha256:test",
    }
    wrapper.step(action, **kwargs)
    with pytest.raises(RuntimeError, match="duplicate command_id"):
        wrapper.step(action, **kwargs)
    first = wrapper.safe_stop("first")
    second = wrapper.safe_stop("second")
    assert first is second
    assert delegate.steps == 1
    assert delegate.stops == 1


def test_static_task_state_never_reports_success_without_provider() -> None:
    report = _run(_Environment(), _Transport(), max_steps=1)
    assert report.run_result.success is False
    assert report.run_result.failure_code is FailureCode.RECOVERY_EXHAUSTED
