from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from industrial_agent.contracts import (
    ActionChunk,
    ActionStep,
    Postcondition,
    TaskSchema,
)
from industrial_agent.environment import SafeStopReceipt
from industrial_agent.executor import ExecutionContext, ExecutorDescriptor
from industrial_agent.fsm import AgentState
from industrial_agent.supervisor_main import build_supervisor
from industrial_agent.v2_supervisor import V2Supervisor, V2TaskPlanner
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


def _bin_task() -> TaskSchema:
    return TaskSchema(
        task_id="BIN01_TO_FINISHED01",
        instruction="把Bin_01搬到FINISHED_01",
        task_type="visual_manipulation",
        target_object="Bin_01",
        target_location="FINISHED_01",
        constraints={"scene_id": "single_bin_manual_industrial_v2"},
        metadata={"profile_id": "single_bin_manual_industrial_v2"},
        postconditions=(
            Postcondition(
                kind="object_in_zone",
                object_id="Bin_01",
                zone_id="FINISHED_01",
            ),
        ),
    )


def test_bin_task_plans_ordered_arm_a_then_arm_b_handoff() -> None:
    plan = V2TaskPlanner().plan(_bin_task(), "run-bin")
    assert [(item.sequence, item.arm_id) for item in plan.subtasks] == [
        (1, "Arm_A"),
        (2, "Arm_B"),
    ]
    assert plan.subtasks[1].depends_on == ("BIN01_TO_HANDOFF_CENTER",)


def _config() -> dict[str, Any]:
    config = json.loads(
        (ROOT / "configs" / "agent.default.json").read_text(encoding="utf-8")
    )
    config["executors"]["pi05"]["checkpoint_sha"] = PINNED_SHA
    config["executors"]["pi05"]["norm_stats_sha"] = PINNED_SHA
    return config


class _Executor:
    def __init__(self) -> None:
        self.arm_calls: list[str] = []
        self.instructions: list[str] = []
        self.subtask_ids: list[str] = []
        self.model_task_ids: list[str | None] = []
        self.model_subtask_ids: list[str | None] = []
        self.model_instructions: list[str | None] = []

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
        self.arm_calls.append(context.arm_id)
        self.instructions.append(context.original_instruction or task.instruction)
        self.subtask_ids.append(str(task.metadata.get("subtask_id")))
        self.model_task_ids.append(context.model_task_id)
        self.model_subtask_ids.append(context.model_subtask_id)
        self.model_instructions.append(context.model_instruction)
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


def _bin_observation(observation_id: str, *, token: str, terminal: bool = False):
    raw = v2_observation(observation_id=observation_id, terminal=terminal)
    raw["task"] = {
        "task_id": "BIN01_TO_FINISHED01",
        "target_object_id": "Bin_01",
        "target_slot_id": None,
        "status": "SUCCEEDED" if terminal else "ACTIVE",
        "terminal": terminal,
        "terminal_confidence": 0.9 if terminal else 0.0,
        "verification_votes": 2 if terminal else 0,
    }
    raw["robot"]["active_arm"] = {
        "A_ONLY": "Arm_A",
        "HANDOFF_VERIFY": "NONE",
        "B_ONLY": "Arm_B",
        "NONE": "NONE",
    }[token]
    raw["robot"]["arm_a"]["retreated"] = token != "A_ONLY"
    raw["robot"]["arm_b"]["retreated"] = True
    return raw


class _BinHandoffEnvironment:
    def __init__(self) -> None:
        self.initial_observed = False
        self.verification_reads = 0
        self.steps: list[tuple[str, str]] = []
        self.final_reads = 0

    def observe(self) -> Mapping[str, Any]:
        if not self.initial_observed:
            self.initial_observed = True
            return _bin_observation("bin-initial", token="A_ONLY")
        if len(self.steps) == 1:
            self.verification_reads += 1
            token = "B_ONLY" if self.verification_reads >= 2 else "HANDOFF_VERIFY"
            return _bin_observation(
                f"bin-handoff-verify-{self.verification_reads}", token=token
            )
        self.final_reads += 1
        return _bin_observation(
            f"bin-final-verify-{self.final_reads}", token="NONE", terminal=True
        )

    def step(self, action: ActionStep, **kwargs: Any) -> Mapping[str, Any]:
        self.steps.append((kwargs["arm_id"], kwargs["control_token"]))
        if len(self.steps) == 1:
            return _bin_observation("bin-at-handoff", token="HANDOFF_VERIFY")
        return _bin_observation("bin-at-finished", token="NONE", terminal=True)

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        return SafeStopReceipt(True, True, True, True, "bin-stop")


def test_v2_supervisor_completes_sensor_closed_loop() -> None:
    supervisor = V2Supervisor.from_config(_Executor(), _config())
    environment = _Environment()
    result = supervisor.run(_task(), environment)
    assert result.success is True
    assert result.state is AgentState.SUCCEEDED
    assert result.executor_history == ("pi05",)
    assert result.control_token_history == ("A_ONLY", "NONE")
    assert environment.steps == 1


def test_v2_supervisor_polls_without_motion_during_verified_dual_arm_handoff() -> None:
    executor = _Executor()
    supervisor = V2Supervisor.from_config(executor, _config())
    environment = _BinHandoffEnvironment()

    result = supervisor.run(_bin_task(), environment)

    assert result.success is True
    assert result.control_token_history == (
        "A_ONLY",
        "HANDOFF_VERIFY",
        "B_ONLY",
        "NONE",
    )
    assert environment.steps == [("Arm_A", "A_ONLY"), ("Arm_B", "B_ONLY")]
    assert environment.verification_reads == 2
    assert executor.arm_calls == ["Arm_A", "Arm_B"]
    assert executor.instructions == [
        "把Bin_01搬到HANDOFF_CENTER",
        "把Bin_01搬到FINISHED_01",
    ]
    assert executor.subtask_ids == [
        "BIN01_TO_HANDOFF_CENTER",
        "BIN01_HANDOFF_TO_FINISHED01",
    ]
    assert executor.model_task_ids == [
        "BIN01_TO_FINISHED01",
        "BIN01_TO_FINISHED01",
    ]
    assert executor.model_subtask_ids == [
        "BIN01_TO_FINISHED01",
        "BIN01_TO_FINISHED01",
    ]
    assert executor.model_instructions == [
        "把Bin_01搬到FINISHED_01",
        "把Bin_01搬到FINISHED_01",
    ]


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
