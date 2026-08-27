"""V2 Supervisor runtime bridge for the π0.5 Isaac closed loop.

This module contains the platform-neutral part of the role-E Isaac entry
point.  Isaac API calls remain inside :class:`IsaacExecutionEnvironment` and
are marshalled by ``IsaacMainThreadGate``; this module only composes the
formal V2 Supervisor with a recording/idempotent environment boundary.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
import logging
from threading import Lock
from typing import Any, TYPE_CHECKING

from industrial_agent.contracts import ActionChunk, ActionStep, TaskSchema
from industrial_agent.environment import ExecutionEnvironment, SafeStopReceipt
from industrial_agent.executor import (
    ExecutionContext,
    Executor,
    ProcessTransport,
)
from industrial_agent.orchestrator import RunResult
from industrial_agent.supervisor_main import build_supervisor
from industrial_agent.v2_supervisor import V2Supervisor

if TYPE_CHECKING:
    from industrial_agent.isaac_runtime import IsaacMainThreadGate


logger = logging.getLogger(__name__)


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
            self._planned[context.step_id] = (chunk.chunk_id, tuple(chunk.steps[0].values))

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

    def __init__(self, delegate: Executor, recorder: _ActionRecorder) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self.descriptor = delegate.descriptor

    def health(self) -> bool:
        return self._delegate.health()

    def plan(
        self,
        task: TaskSchema,
        observation: Any,
        context: ExecutionContext,
    ) -> ActionChunk:
        chunk = self._delegate.plan(task, observation, context)
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
    ) -> None:
        self._delegate = delegate
        self._recorder = recorder
        self._seen_commands: set[str] = set()
        self._stop_receipt: SafeStopReceipt | None = None
        self._lock = Lock()

    def observe(self) -> Mapping[str, Any]:
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
        if not isinstance(command_id, str) or not command_id:
            raise ValueError("command_id must be a non-empty string")
        with self._lock:
            if command_id in self._seen_commands:
                logger.error("duplicate Isaac command rejected command_id=%s", command_id)
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
) -> SupervisorRuntimeReport:
    """Run the formal V2 Supervisor through the Isaac owner-thread gate.

    The function always requests one idempotent safe stop before returning,
    including the expected decision-budget exhaustion path.  Exceptions from
    Supervisor setup or execution are logged, stopped, and re-raised so the
    outer Isaac runner can write its complete failure JSON.
    """

    runtime_config = with_decision_budget(config, max_steps)
    recorder = _ActionRecorder()
    recording_environment = _RecordingEnvironment(environment, recorder)

    try:
        supervisor: V2Supervisor = build_supervisor(
            runtime_config,
            transport_factory=transport_factory,
        )
        supervisor.executor = _RecordingExecutor(supervisor.executor, recorder)
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
    )


__all__ = [
    "ActionExecutionRecord",
    "SupervisorRuntimeReport",
    "run_supervisor_runtime",
    "run_v2_supervisor_runtime",
    "with_decision_budget",
]
