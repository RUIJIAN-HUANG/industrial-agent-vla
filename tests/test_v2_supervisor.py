from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import pytest

from industrial_agent.contracts import ActionChunk, ActionStep, TaskSchema
from industrial_agent.environment import SafeStopReceipt
from industrial_agent.executor import ExecutionContext, ExecutorDescriptor
from industrial_agent.fsm import AgentState
from industrial_agent.supervisor_main import build_supervisor
from industrial_agent.v2_supervisor import V2Supervisor
from tests.test_v2_observation import v2_observation


ROOT = Path(__file__).resolve().parents[1]
PINNED_SHA = f"sha256:{'a' * 64}"


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


class _Executor:
    descriptor = ExecutorDescriptor(
        name="pi05",
        task_types=frozenset({"pick_place"}),
        action_contract_version="1.0",
        checkpoint_sha=PINNED_SHA,
        norm_stats_sha=PINNED_SHA,
    )

    def health(self) -> bool:
        return True

    def plan(
        self, task: TaskSchema, observation: Any, context: ExecutionContext
    ) -> ActionChunk:
        return ActionChunk(
            contract_version="1.0",
            chunk_id=f"chunk-{context.step_id}",
            task_id=task.task_id,
            executor="pi05",
            steps=(ActionStep.from_sequence([0, 0, 0, 0, 0, 0, 1]),),
        )

    def cancel(self, task_id: str, reason: str) -> None:
        pass


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
    supervisor = V2Supervisor.from_config(_Executor(), _config())
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
    supervisor = V2Supervisor.from_config(_Executor(), config)
    environment = _Environment(terminal_after_step=False)
    result = supervisor.run(_task(), environment)
    assert result.success is False
    assert result.state is AgentState.SAFE_STOPPED
    assert environment.stops == 1


def test_production_builder_wires_only_pi05() -> None:
    calls: list[tuple[str, str]] = []

    def factory(name: str, url: str) -> _Transport:
        calls.append((name, url))
        return _Transport()

    supervisor = build_supervisor(_config(), transport_factory=factory)
    assert isinstance(supervisor, V2Supervisor)
    assert calls == [("pi05", "http://127.0.0.1:8101")]


def test_production_builder_rejects_abolished_v1() -> None:
    legacy = json.loads(
        (ROOT / "configs" / "agent.v1.legacy.json").read_text(encoding="utf-8")
    )
    with pytest.raises(ValueError, match="V1 is abolished"):
        build_supervisor(legacy, transport_factory=lambda _name, _url: _Transport())
