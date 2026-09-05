"""V2 Supervisor runtime bridge for the π0.5 Isaac closed loop.

This module contains the platform-neutral part of the role-E Isaac entry
point.  Isaac API calls remain inside :class:`IsaacExecutionEnvironment` and
are marshalled by ``IsaacMainThreadGate``; this module only composes the
formal V2 Supervisor with a recording/idempotent environment boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass, replace
import logging
from threading import Event, Lock
from typing import Any, TYPE_CHECKING

from industrial_agent.contracts import ActionChunk, ActionStep, TaskSchema
from industrial_agent.environment import ExecutionEnvironment, SafeStopReceipt
from industrial_agent.errors import FailureCode
from industrial_agent.executor import (
    ExecutionContext,
    Executor,
    ProcessTransport,
)
from industrial_agent.run_result import RunResult
from industrial_agent.safety import (
    AXIS_NAMES,
    ActionSafetyValidator,
    SafetyDecision,
)
from industrial_agent.supervisor_main import build_supervisor
from industrial_agent.v2_supervisor import V2Supervisor

if TYPE_CHECKING:
    from industrial_agent.isaac_runtime import IsaacMainThreadGate


logger = logging.getLogger(__name__)


_TASK3_ID = "BIN01_TO_FINISHED01"
_TASK3_WORKSPACE_GRACE_XY_M = 0.030
_TASK3_WORKSPACE_GRACE_Z_LOWER_M = 0.010
_TASK3_WORKSPACE_GRACE_Z_UPPER_M = 0.005
_TASK3_WORKSPACE_CLAMP_INSET_M = 0.001
_TASK3_WORKSPACE_MAX_CONSECUTIVE_CLAMPS = 8


class Task3WorkspaceGraceSafety:
    """Apply the formal task-three boundary grace before safety validation.

    The frozen workspace remains the nominal execution envelope.  A task-three
    action that predicts a small overshoot is projected back inside that
    envelope, while an overshoot beyond the configured grace is still rejected.
    The underlying validator keeps all action, token, frame, and finite-value
    checks.  This class is intentionally kept in the role-E simulation bridge;
    the framework safety policy is not changed for P01/W01 or other callers.
    """

    def __init__(self, delegate: ActionSafetyValidator) -> None:
        self._delegate = delegate
        self._consecutive_clamps = 0

        policy = delegate.policy
        expanded_a_min = tuple(
            value - margin
            for value, margin in zip(
                policy.arm_a_workspace_min_m,
                (
                    _TASK3_WORKSPACE_GRACE_XY_M,
                    _TASK3_WORKSPACE_GRACE_XY_M,
                    _TASK3_WORKSPACE_GRACE_Z_LOWER_M,
                ),
            )
        )
        expanded_a_max = tuple(
            value + margin
            for value, margin in zip(
                policy.arm_a_workspace_max_m,
                (
                    _TASK3_WORKSPACE_GRACE_XY_M,
                    _TASK3_WORKSPACE_GRACE_XY_M,
                    _TASK3_WORKSPACE_GRACE_Z_UPPER_M,
                ),
            )
        )
        expanded_b_min = tuple(
            value - margin
            for value, margin in zip(
                policy.arm_b_workspace_min_m,
                (
                    _TASK3_WORKSPACE_GRACE_XY_M,
                    _TASK3_WORKSPACE_GRACE_XY_M,
                    _TASK3_WORKSPACE_GRACE_Z_LOWER_M,
                ),
            )
        )
        expanded_b_max = tuple(
            value + margin
            for value, margin in zip(
                policy.arm_b_workspace_max_m,
                (
                    _TASK3_WORKSPACE_GRACE_XY_M,
                    _TASK3_WORKSPACE_GRACE_XY_M,
                    _TASK3_WORKSPACE_GRACE_Z_UPPER_M,
                ),
            )
        )
        expanded_policy = replace(
            policy,
            arm_a_workspace_min_m=expanded_a_min,
            arm_a_workspace_max_m=expanded_a_max,
            arm_b_workspace_min_m=expanded_b_min,
            arm_b_workspace_max_m=expanded_b_max,
        )
        self._expanded_delegate = ActionSafetyValidator(expanded_policy)

    @property
    def policy(self) -> Any:
        return self._delegate.policy

    def _workspace(self, arm_id: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
        policy = self._delegate.policy
        if arm_id == "Arm_A":
            return policy.arm_a_workspace_min_m, policy.arm_a_workspace_max_m
        if arm_id == "Arm_B":
            return policy.arm_b_workspace_min_m, policy.arm_b_workspace_max_m
        raise ValueError(f"unsupported arm_id for task-three workspace: {arm_id!r}")

    def _grace(self, axis: int, *, lower: bool) -> float:
        if axis < 2:
            return _TASK3_WORKSPACE_GRACE_XY_M
        return (
            _TASK3_WORKSPACE_GRACE_Z_LOWER_M
            if lower
            else _TASK3_WORKSPACE_GRACE_Z_UPPER_M
        )

    def validate_and_limit(
        self,
        chunk: ActionChunk,
        observation: Any,
        *,
        arm_id: str,
        control_token: str,
    ) -> SafetyDecision:
        try:
            chunk.validate_contract()
        except Exception:
            return self._expanded_delegate.validate_and_limit(
                chunk,
                observation,
                arm_id=arm_id,
                control_token=control_token,
            )

        workspace_min, workspace_max = self._workspace(arm_id)
        robot = observation.data.get("robot", {})
        arm_key = {"Arm_A": "arm_a", "Arm_B": "arm_b"}.get(arm_id)
        arm_state = robot.get(arm_key) if isinstance(robot, Mapping) else None
        pose = (
            arm_state.get("tcp_pose_m_rad") if isinstance(arm_state, Mapping) else None
        )
        if (
            not isinstance(pose, (list, tuple))
            or len(pose) < 3
            or any(not isinstance(value, (int, float)) for value in pose[:3])
        ):
            return self._expanded_delegate.validate_and_limit(
                chunk,
                observation,
                arm_id=arm_id,
                control_token=control_token,
            )

        projected = [float(value) for value in pose[:3]]
        for axis, current in enumerate(projected):
            expanded_min = workspace_min[axis] - self._grace(axis, lower=True)
            expanded_max = workspace_max[axis] + self._grace(axis, lower=False)
            if current < expanded_min or current > expanded_max:
                return SafetyDecision(
                    False,
                    FailureCode.ACTION_WORKSPACE_BREACH,
                    f"current {AXIS_NAMES[axis][1:]}={current:.6f}m exceeds "
                    f"task-three grace for {arm_id} robot_base workspace",
                )
        adjusted_steps: list[ActionStep] = []
        limited_axes: set[str] = set()
        first_step_clamped = False
        for step_index, step in enumerate(chunk.steps):
            if step.has_non_finite():
                return self._expanded_delegate.validate_and_limit(
                    chunk,
                    observation,
                    arm_id=arm_id,
                    control_token=control_token,
                )
            values = list(step.values)
            for index, limit in enumerate(self._delegate.policy.axis_abs_limits):
                bounded = min(limit, max(-limit, values[index]))
                if bounded != values[index]:
                    values[index] = bounded
                    limited_axes.add(AXIS_NAMES[index])

            step_clamped = False
            for axis in range(3):
                target = projected[axis] + values[axis]
                if target < workspace_min[axis]:
                    overshoot = workspace_min[axis] - target
                    if overshoot > self._grace(axis, lower=True):
                        return SafetyDecision(
                            False,
                            FailureCode.ACTION_WORKSPACE_BREACH,
                            f"projected {AXIS_NAMES[axis][1:]}={target:.6f}m "
                            f"exceeds task-three grace for {arm_id} robot_base workspace",
                        )
                    target = workspace_min[axis] + _TASK3_WORKSPACE_CLAMP_INSET_M
                    values[axis] = target - projected[axis]
                    limited_axes.add(AXIS_NAMES[axis])
                    step_clamped = True
                elif target > workspace_max[axis]:
                    overshoot = target - workspace_max[axis]
                    if overshoot > self._grace(axis, lower=False):
                        return SafetyDecision(
                            False,
                            FailureCode.ACTION_WORKSPACE_BREACH,
                            f"projected {AXIS_NAMES[axis][1:]}={target:.6f}m "
                            f"exceeds task-three grace for {arm_id} robot_base workspace",
                        )
                    target = workspace_max[axis] - _TASK3_WORKSPACE_CLAMP_INSET_M
                    values[axis] = target - projected[axis]
                    limited_axes.add(AXIS_NAMES[axis])
                    step_clamped = True
                projected[axis] = target
            if step_index == 0:
                first_step_clamped = step_clamped
            adjusted_steps.append(
                ActionStep.from_sequence(values, duration_ms=step.duration_ms)
            )

        if first_step_clamped:
            self._consecutive_clamps += 1
        else:
            self._consecutive_clamps = 0
        if self._consecutive_clamps > _TASK3_WORKSPACE_MAX_CONSECUTIVE_CLAMPS:
            return SafetyDecision(
                False,
                FailureCode.ACTION_WORKSPACE_BREACH,
                "task-three workspace grace exceeded consecutive clamp budget",
            )

        adjusted = replace(chunk, steps=tuple(adjusted_steps))
        decision = self._expanded_delegate.validate_and_limit(
            adjusted,
            observation,
            arm_id=arm_id,
            control_token=control_token,
        )
        if decision.accepted and limited_axes:
            logger.warning(
                "task-three workspace grace applied arm=%s chunk_id=%s "
                "axes=%s consecutive=%d",
                arm_id,
                chunk.chunk_id,
                ",".join(sorted(limited_axes)),
                self._consecutive_clamps,
            )
            return replace(
                decision,
                reason="accepted with task-three workspace grace",
                limited_axes=tuple(sorted(set(decision.limited_axes) | limited_axes)),
            )
        return decision


@dataclass(frozen=True)
class ActionExecutionRecord:
    """Auditable summary of one Supervisor command attempt."""

    decision_index: int
    observation_id: str
    command_id: str
    chunk_id: str | None
    action_7d: tuple[float, ...]
    execution_result: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible action record."""

        return {
            "decision_index": self.decision_index,
            "observation_id": self.observation_id,
            "command_id": self.command_id,
            "chunk_id": self.chunk_id,
            "action_7d": list(self.action_7d),
            "execution_result": dict(self.execution_result),
        }


@dataclass(frozen=True)
class SupervisorRuntimeReport:
    """Supervisor result plus execution-boundary evidence."""

    run_result: RunResult
    actions: tuple[ActionExecutionRecord, ...]
    safe_stop_receipt: SafeStopReceipt

    @property
    def safe_stop_confirmed(self) -> bool:
        """Whether the controller returned a complete physical stop receipt."""

        return self.safe_stop_receipt.confirmed

    def action_dicts(self) -> list[dict[str, Any]]:
        """Return action records ready for the runner's result JSON."""

        return [record.to_dict() for record in self.actions]


def with_decision_budget(
    config: Mapping[str, Any],
    max_steps: int,
) -> dict[str, Any]:
    """Copy ``config`` and replace only the V2 decision budget.

    The Supervisor reads its budget from ``recovery.max_decisions_per_task``.
    A deep copy is intentional: callers commonly reuse the loaded agent
    configuration for a later direct-mode run in the same process.
    """

    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    if not isinstance(config, Mapping):
        raise TypeError("config must be a mapping")

    runtime_config = deepcopy(dict(config))
    recovery = runtime_config.get("recovery")
    if not isinstance(recovery, Mapping):
        raise ValueError("V2 config recovery must be an object")
    runtime_config["recovery"] = dict(recovery)
    runtime_config["recovery"]["max_decisions_per_task"] = max_steps
    return runtime_config


class _ActionRecorder:
    """Collect action metadata shared by executor and environment wrappers."""

    def __init__(self) -> None:
        self._planned: dict[int, tuple[str, tuple[float, ...]]] = {}
        self._records: list[ActionExecutionRecord] = []
        self._lock = Lock()

    def record_plan(self, context: ExecutionContext, chunk: ActionChunk) -> None:
        with self._lock:
            self._planned[context.step_id] = (
                chunk.chunk_id,
                tuple(chunk.steps[0].values),
            )

    def begin(
        self,
        action: ActionStep,
        *,
        observation_id: str,
        command_id: str,
    ) -> int:
        step_id = _command_step_id(command_id)
        with self._lock:
            planned = self._planned.get(step_id)
            chunk_id = planned[0] if planned is not None else None
            record = ActionExecutionRecord(
                decision_index=len(self._records),
                observation_id=observation_id,
                command_id=command_id,
                chunk_id=chunk_id,
                action_7d=tuple(action.values),
                execution_result={"status": "STARTED"},
            )
            self._records.append(record)
            return len(self._records) - 1

    def finish(self, index: int, result: Mapping[str, Any]) -> None:
        with self._lock:
            record = self._records[index]
            self._records[index] = ActionExecutionRecord(
                decision_index=record.decision_index,
                observation_id=record.observation_id,
                command_id=record.command_id,
                chunk_id=record.chunk_id,
                action_7d=record.action_7d,
                execution_result=dict(result),
            )

    def snapshot(self) -> tuple[ActionExecutionRecord, ...]:
        with self._lock:
            return tuple(self._records)


def _command_step_id(command_id: str) -> int:
    marker = "-command-"
    suffix = command_id.rsplit(marker, 1)[-1] if marker in command_id else ""
    try:
        return int(suffix)
    except ValueError:
        return -1


def _execution_success_result(raw_result: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {"status": "ACKED"}
    observation_id = raw_result.get("observation_id")
    if isinstance(observation_id, str) and observation_id:
        result["observation_id"] = observation_id
    return result


class _RecordingExecutor:
    """Delegate π0.5 calls while retaining the chunk for command auditing."""

    def __init__(
        self,
        delegate: Executor,
        recorder: _ActionRecorder,
        stop_event: Event | None = None,
    ) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self._stop_event = stop_event
        self.descriptor = delegate.descriptor

    def health(self) -> bool:
        return self._delegate.health()

    def plan(
        self,
        task: TaskSchema,
        observation: Any,
        context: ExecutionContext,
    ) -> ActionChunk:
        if self._stop_event is not None and self._stop_event.is_set():
            self._delegate.cancel(task.task_id, "operator safe-stop requested")
            raise RuntimeError("operator safe-stop requested before π0.5 inference")
        chunk = self._delegate.plan(task, observation, context)
        if self._stop_event is not None and self._stop_event.is_set():
            self._delegate.cancel(task.task_id, "operator safe-stop requested")
            raise RuntimeError("operator safe-stop requested during π0.5 inference")
        self._recorder.record_plan(context, chunk)
        return chunk

    def cancel(self, task_id: str, reason: str) -> None:
        self._delegate.cancel(task_id, reason)


class _RecordingEnvironment:
    """Add command idempotency and bounded execution evidence to an environment."""

    def __init__(
        self,
        delegate: ExecutionEnvironment,
        recorder: _ActionRecorder,
        stop_event: Event | None = None,
    ) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self._stop_event = stop_event
        self._seen_commands: set[str] = set()
        self._stop_receipt: SafeStopReceipt | None = None
        self._lock = Lock()

    def observe(self) -> Mapping[str, Any]:
        if self._stop_event is not None and self._stop_event.is_set():
            raise RuntimeError("operator safe-stop requested before observation")
        return self._delegate.observe()

    def step(
        self,
        action: ActionStep,
        *,
        arm_id: str,
        control_token: str,
        command_id: str,
        expected_observation_id: str,
        expected_state_digest: str,
    ) -> Mapping[str, Any]:
        if self._stop_event is not None and self._stop_event.is_set():
            raise RuntimeError("operator safe-stop requested before action execution")
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("command_id must be a non-empty string")
        with self._lock:
            if command_id in self._seen_commands:
                logger.error(
                    "duplicate Isaac command rejected command_id=%s", command_id
                )
                raise RuntimeError(f"duplicate command_id rejected: {command_id}")
            self._seen_commands.add(command_id)

        record_index = self._recorder.begin(
            action,
            observation_id=expected_observation_id,
            command_id=command_id,
        )
        try:
            raw_result = self._delegate.step(
                action,
                arm_id=arm_id,
                control_token=control_token,
                command_id=command_id,
                expected_observation_id=expected_observation_id,
                expected_state_digest=expected_state_digest,
            )
        except Exception as exc:
            logger.exception("Isaac action failed command_id=%s", command_id)
            self._recorder.finish(
                record_index,
                {
                    "status": "FAILED",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        if not isinstance(raw_result, Mapping):
            error = TypeError("Isaac action result must be an object")
            self._recorder.finish(
                record_index,
                {
                    "status": "FAILED",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            raise error
        self._recorder.finish(record_index, _execution_success_result(raw_result))
        return raw_result

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        with self._lock:
            if self._stop_receipt is not None:
                logger.info("duplicate safe-stop suppressed reason=%s", reason)
                return self._stop_receipt
            try:
                receipt = self._delegate.safe_stop(reason)
            except Exception as exc:
                logger.exception("Isaac safe-stop failed reason=%s", reason)
                receipt = SafeStopReceipt(
                    controller_ack=False,
                    buffers_cleared=False,
                    arm_a_stopped=False,
                    arm_b_stopped=False,
                    stop_epoch=f"runtime-stop-failed-{type(exc).__name__}",
                )
            if not isinstance(receipt, SafeStopReceipt):
                logger.error(
                    "Isaac safe-stop returned invalid receipt type=%s",
                    type(receipt).__name__,
                )
                receipt = SafeStopReceipt(
                    controller_ack=False,
                    buffers_cleared=False,
                    arm_a_stopped=False,
                    arm_b_stopped=False,
                    stop_epoch="runtime-stop-invalid-receipt",
                )
            self._stop_receipt = receipt
            return receipt

    @property
    def stop_receipt(self) -> SafeStopReceipt | None:
        with self._lock:
            return self._stop_receipt


def run_supervisor_runtime(
    *,
    config: Mapping[str, Any],
    task: TaskSchema,
    environment: ExecutionEnvironment,
    gate: IsaacMainThreadGate,
    max_steps: int,
    idle_callback: Callable[[], None] | None = None,
    transport_factory: Callable[[str, str], ProcessTransport] | None = None,
    stop_event: Event | None = None,
) -> SupervisorRuntimeReport:
    """Run the formal V2 Supervisor through the Isaac owner-thread gate.

    The function always requests one idempotent safe stop before returning,
    including the expected decision-budget exhaustion path.  Exceptions from
    Supervisor setup or execution are logged, stopped, and re-raised so the
    outer Isaac runner can write its complete failure JSON.
    """

    runtime_config = with_decision_budget(config, max_steps)
    recorder = _ActionRecorder()
    recording_environment = _RecordingEnvironment(
        environment,
        recorder,
        stop_event=stop_event,
    )

    try:
        supervisor: V2Supervisor = build_supervisor(
            runtime_config,
            transport_factory=transport_factory,
        )
        if task.task_id == _TASK3_ID:
            supervisor.safety = Task3WorkspaceGraceSafety(supervisor.safety)
            logger.info(
                "task-three workspace grace enabled xy_mm=%.1f "
                "z_lower_mm=%.1f z_upper_mm=%.1f",
                _TASK3_WORKSPACE_GRACE_XY_M * 1000.0,
                _TASK3_WORKSPACE_GRACE_Z_LOWER_M * 1000.0,
                _TASK3_WORKSPACE_GRACE_Z_UPPER_M * 1000.0,
            )
        supervisor.executor = _RecordingExecutor(
            supervisor.executor,
            recorder,
            stop_event=stop_event,
        )
        run_result = gate.run_worker_until_complete(
            lambda: supervisor.run(task, recording_environment),
            idle_callback=idle_callback,
        )
        if not isinstance(run_result, RunResult):
            raise TypeError("V2 Supervisor returned a non-RunResult value")
    except BaseException:
        logger.exception("V2 Supervisor runtime failed")
        recording_environment.safe_stop("V2 Supervisor runtime exception")
        raise

    stop_reason = (
        "V2 Supervisor task completed; revoke motion before process exit"
        if run_result.success
        else f"V2 Supervisor stopped: {run_result.message}"
    )
    receipt = recording_environment.safe_stop(stop_reason)
    if not receipt.confirmed:
        logger.error("V2 Supervisor safe-stop was not confirmed")
    return SupervisorRuntimeReport(
        run_result=run_result,
        actions=recorder.snapshot(),
        safe_stop_receipt=receipt,
    )


def run_v2_supervisor_runtime(
    *,
    config: Mapping[str, Any],
    task: TaskSchema,
    environment: ExecutionEnvironment,
    gate: IsaacMainThreadGate,
    max_steps: int,
    idle_callback: Callable[[], None] | None = None,
    transport_factory: Callable[[str, str], ProcessTransport] | None = None,
    stop_event: Event | None = None,
) -> SupervisorRuntimeReport:
    """Explicit V2-named alias for :func:`run_supervisor_runtime`."""

    return run_supervisor_runtime(
        config=config,
        task=task,
        environment=environment,
        gate=gate,
        max_steps=max_steps,
        idle_callback=idle_callback,
        transport_factory=transport_factory,
        stop_event=stop_event,
    )


__all__ = [
    "ActionExecutionRecord",
    "SupervisorRuntimeReport",
    "Task3WorkspaceGraceSafety",
    "run_supervisor_runtime",
    "run_v2_supervisor_runtime",
    "with_decision_budget",
]
