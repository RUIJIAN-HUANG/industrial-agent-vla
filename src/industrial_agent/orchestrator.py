"""Supervisor Agent: FSM, routing, execution, verification, and recovery."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import isfinite
from typing import Any, Mapping, Sequence
from uuid import uuid4

from .contracts import (
    ActionStep,
    Observation,
    SubtaskStatus,
    TaskPlan,
    TaskSchema,
)
from .environment import ExecutionEnvironment
from .errors import AgentError, FailureCode
from .executor import (
    ExecutionContext,
    Executor,
    ExecutorRouter,
    is_pinned_artifact_digest,
)
from .fsm import AgentFSM, AgentState, StateTransition
from .observation import ObservationGateway
from .planner import SemanticTaskPlanner
from .safety import ActionSafetyValidator, SafetyPolicy, safety_state_failure
from .telemetry import EventRecord, EventSink, MemoryStore, RunMemory
from .verifier import PostconditionVerifier, VerificationResult, Verdict


@dataclass(frozen=True)
class RunResult:
    run_id: str
    task_id: str
    state: AgentState
    success: bool
    failure_code: FailureCode
    message: str
    executor_history: tuple[str, ...]
    replan_counts: dict[str, int]
    switch_count: int
    transitions: tuple[StateTransition, ...]
    verification: VerificationResult | None
    task_plan: dict[str, Any]
    events: tuple[EventRecord, ...]


class IndustrialAgent:
    """Lightweight total Agent with bounded, auditable recovery."""

    def __init__(
        self,
        executors: Sequence[Executor],
        *,
        gateway: ObservationGateway | None = None,
        safety: ActionSafetyValidator | None = None,
        verifier: PostconditionVerifier | None = None,
        events: EventSink | None = None,
        memory_store: MemoryStore | None = None,
        planner: SemanticTaskPlanner | None = None,
        verification_frames: int = 3,
        executor_timeout_ms: int = 15_000,
        max_decisions_per_strategy_attempt: int = 8,
    ):
        if verification_frames < 1 or verification_frames > 9:
            raise ValueError("verification_frames must be in [1, 9]")
        if not 1 <= max_decisions_per_strategy_attempt <= 100:
            raise ValueError("max_decisions_per_strategy_attempt must be in [1, 100]")
        self.router = ExecutorRouter(executors)
        self.gateway = gateway or ObservationGateway()
        self.safety = safety or ActionSafetyValidator()
        self.verifier = verifier or PostconditionVerifier()
        self.events = events or EventSink()
        self.memory_store = memory_store or MemoryStore()
        self.planner = planner or SemanticTaskPlanner()
        self.verification_frames = verification_frames
        self.executor_timeout_ms = executor_timeout_ms
        self.max_decisions_per_strategy_attempt = max_decisions_per_strategy_attempt

        self._fsm = AgentFSM()
        self._queue: deque[ActionStep] = deque()
        self._run_id = ""
        self._task_id = ""
        self._memory: RunMemory | None = None
        self._plan: TaskPlan | None = None

    @classmethod
    def from_config(
        cls,
        executors: Sequence[Executor],
        config: Mapping[str, Any],
        *,
        gateway: ObservationGateway | None = None,
        verifier: PostconditionVerifier | None = None,
        events: EventSink | None = None,
        memory_store: MemoryStore | None = None,
        planner: SemanticTaskPlanner | None = None,
    ) -> IndustrialAgent:
        """Build the core from the versioned JSON-compatible configuration."""

        if not isinstance(config, Mapping):
            raise ValueError("agent config must be an object")
        version = config.get("config_version")
        if not isinstance(version, str) or version.split(".", 1)[0] != "1":
            raise ValueError(f"unsupported config_version: {version!r}")

        def required_int(
            mapping: Mapping[str, Any],
            key: str,
            *,
            minimum: int,
            maximum: int,
        ) -> int:
            value = mapping.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not minimum <= value <= maximum
            ):
                raise ValueError(f"{key} must be an integer in [{minimum}, {maximum}]")
            return value

        recovery = config.get("recovery")
        if not isinstance(recovery, Mapping):
            raise ValueError("recovery config must be an object")
        frozen_recovery = {
            "max_replans_per_subtask_strategy": 1,
            "max_switches_per_run": 1,
            "allow_switch_back": False,
            "clear_action_queue_on_recovery": True,
        }
        mismatches = {
            key: {"expected": expected, "actual": recovery.get(key)}
            for key, expected in frozen_recovery.items()
            if recovery.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"recovery invariants are frozen: {mismatches}")

        raw_safety = config.get("safety")
        if not isinstance(raw_safety, Mapping):
            raise ValueError("safety config must be an object")

        def float_tuple(key: str, size: int) -> tuple[float, ...]:
            value = raw_safety.get(key)
            if (
                not isinstance(value, (list, tuple))
                or len(value) != size
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for item in value
                )
            ):
                raise ValueError(f"safety.{key} must contain {size} numbers")
            result = tuple(float(item) for item in value)
            if not all(isfinite(item) for item in result):
                raise ValueError(f"safety.{key} values must be finite")
            return result

        axis_limits = float_tuple("axis_abs_limits", 7)
        workspace_min = float_tuple("workspace_min_m", 3)
        workspace_max = float_tuple("workspace_max_m", 3)
        if any(low >= high for low, high in zip(workspace_min, workspace_max)):
            raise ValueError("each workspace_min_m value must be below workspace_max_m")
        if any(limit <= 0 for limit in axis_limits):
            raise ValueError("all safety.axis_abs_limits values must be positive")
        if axis_limits[6] > 1.0:
            raise ValueError("gripper axis limit cannot exceed normalized range 1.0")
        policy = SafetyPolicy(
            axis_abs_limits=axis_limits,  # type: ignore[arg-type]
            workspace_min_m=workspace_min,  # type: ignore[arg-type]
            workspace_max_m=workspace_max,  # type: ignore[arg-type]
            max_chunk_steps=required_int(
                raw_safety, "max_chunk_steps", minimum=1, maximum=32
            ),
        )
        action_contract = raw_safety.get("action_contract_version")
        if action_contract != "1.0":
            raise ValueError(
                "safety.action_contract_version must match core contract 1.0"
            )

        raw_executors = config.get("executors")
        if not isinstance(raw_executors, Mapping):
            raise ValueError("executors config must be an object")
        required_executor_names = {"openvla_oft", "pi05"}
        if set(raw_executors) != required_executor_names:
            raise ValueError(
                "config.executors must declare exactly "
                f"{sorted(required_executor_names)}"
            )
        enabled_names: set[str] = set()
        for name, raw in raw_executors.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"config.executors.{name} must be an object")
            enabled = raw.get("enabled")
            if not isinstance(enabled, bool):
                raise ValueError(f"config.executors.{name}.enabled must be a boolean")
            if enabled:
                enabled_names.add(name)
        if not enabled_names:
            raise ValueError("at least one executor must be explicitly enabled")
        provided_names: set[str] = set()
        for executor in executors:
            descriptor = executor.descriptor
            name = descriptor.name
            if name in provided_names:
                raise ValueError(f"executor descriptor name is duplicated: {name!r}")
            provided_names.add(name)

            expected = raw_executors.get(name)
            if not isinstance(expected, Mapping):
                raise ValueError(
                    f"executor {name!r} is not declared in config.executors"
                )
            if descriptor.action_contract_version != action_contract:
                raise ValueError(
                    f"executor {name!r} action_contract_version mismatch: "
                    f"expected {action_contract!r}, got "
                    f"{descriptor.action_contract_version!r}"
                )

            for field in ("checkpoint_sha", "norm_stats_sha"):
                expected_value = expected.get(field)
                actual_value = getattr(descriptor, field)
                if expected_value == "REPLACE_WITH_PINNED_SHA":
                    raise ValueError(
                        f"executor {name!r} config.{field} is still an unsafe placeholder"
                    )
                if not is_pinned_artifact_digest(expected_value):
                    raise ValueError(
                        f"executor {name!r} config.{field} must match "
                        "'sha256:<64 hexadecimal characters>'"
                    )
                if actual_value != expected_value:
                    raise ValueError(
                        f"executor {name!r} {field} mismatch: "
                        f"expected {expected_value!r}, got {actual_value!r}"
                    )
        if provided_names != enabled_names:
            missing = sorted(enabled_names - provided_names)
            unexpected = sorted(provided_names - enabled_names)
            raise ValueError(
                "enabled executor set mismatch: "
                f"missing={missing}, unexpected={unexpected}"
            )

        return cls(
            executors,
            gateway=gateway,
            safety=ActionSafetyValidator(policy),
            verifier=verifier,
            events=events,
            memory_store=memory_store,
            planner=planner,
            verification_frames=required_int(
                config, "verification_frames", minimum=1, maximum=9
            ),
            executor_timeout_ms=required_int(
                config, "executor_timeout_ms", minimum=1, maximum=300_000
            ),
            max_decisions_per_strategy_attempt=required_int(
                recovery,
                "max_decisions_per_strategy_attempt",
                minimum=1,
                maximum=100,
            ),
        )

    def _emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        self.events.emit(
            run_id=self._run_id,
            task_id=self._task_id,
            event_type=event_type,
            state=self._fsm.state,
            payload=payload,
        )

    def _transition(self, target: AgentState, reason: str) -> None:
        record = self._fsm.transition(target, reason)
        self._emit(
            "fsm.transition",
            {
                "from": record.previous.value,
                "to": record.current.value,
                "reason": reason,
            },
        )

    def _clear_queue(self, reason: str) -> None:
        removed = len(self._queue)
        self._queue.clear()
        self._emit("action_queue.cleared", {"reason": reason, "removed_steps": removed})

    def _result(
        self,
        code: FailureCode,
        message: str,
        verification: VerificationResult | None,
        event_start: int,
    ) -> RunResult:
        assert self._memory is not None
        return RunResult(
            run_id=self._run_id,
            task_id=self._task_id,
            state=self._fsm.state,
            success=self._fsm.state is AgentState.SUCCEEDED,
            failure_code=code,
            message=message,
            executor_history=tuple(self._memory.executor_history),
            replan_counts=dict(self._memory.replan_counts),
            switch_count=self._memory.switch_count,
            transitions=tuple(self._fsm.history),
            verification=verification,
            task_plan=self._plan.to_dict() if self._plan is not None else {},
            events=tuple(self.events.events[event_start:]),
        )

    def _fail(
        self,
        code: FailureCode,
        message: str,
        verification: VerificationResult | None,
        event_start: int,
    ) -> RunResult:
        assert self._memory is not None
        self._memory.last_failure_code = code.value
        self._clear_queue("terminal_failure")
        if self._fsm.state is not AgentState.FAILED:
            self._transition(AgentState.FAILED, message)
        self._emit("run.failed", {"failure_code": code.value, "message": message})
        return self._result(code, message, verification, event_start)

    def _safe_stop(
        self,
        environment: ExecutionEnvironment,
        code: FailureCode,
        message: str,
        verification: VerificationResult | None,
        event_start: int,
    ) -> RunResult:
        assert self._memory is not None
        self._memory.last_failure_code = code.value
        self._clear_queue("immediate_safe_stop")
        stop_error: str | None = None
        try:
            environment.safe_stop(message)
        except Exception as exc:
            stop_error = str(exc)
        finally:
            if self._fsm.state is not AgentState.SAFE_STOPPED:
                self._transition(AgentState.SAFE_STOPPED, message)
            self._emit(
                "run.safe_stopped",
                {
                    "failure_code": code.value,
                    "message": message,
                    "safe_stop_error": stop_error,
                },
            )
        return self._result(code, message, verification, event_start)

    def _check_system_fault(
        self,
        observation: Observation,
        environment: ExecutionEnvironment,
        verification: VerificationResult | None,
        event_start: int,
    ) -> RunResult | None:
        fault = safety_state_failure(observation)
        if fault is None:
            return None
        code, reason = fault
        return self._safe_stop(environment, code, reason, verification, event_start)

    def _recover(
        self,
        *,
        task: TaskSchema,
        current: Executor,
        code: FailureCode,
        reason: str,
        excluded: set[str],
        local_replans: dict[str, int],
    ) -> tuple[Executor | None, bool]:
        """Return `(executor_or_none, should_continue)`.

        `None, True` means select a different executor on the next loop.
        """

        assert self._memory is not None
        name = current.descriptor.name
        self._memory.last_failure_code = code.value
        replans = local_replans.get(name, 0)
        if replans < 1:
            local_replans[name] = replans + 1
            self._memory.replan_counts[name] = (
                self._memory.replan_counts.get(name, 0) + 1
            )
            self._clear_queue("replan")
            current.cancel(task.task_id, reason)
            self._transition(
                AgentState.REPLANNING,
                f"{name}: one bounded replan after {code.value}",
            )
            self._emit(
                "recovery.replan",
                {
                    "executor": name,
                    "replan_index": local_replans[name],
                    "trigger_code": code.value,
                },
            )
            self._transition(AgentState.OBSERVING, "refresh observation for replan")
            return current, True
        if self._memory.switch_count < 1:
            self._memory.switch_count += 1
            excluded.add(name)
            self._clear_queue("executor_switch")
            current.cancel(task.task_id, reason)
            self._transition(
                AgentState.SWITCHING,
                f"switch away from {name} after replan exhausted",
            )
            self._emit(
                "recovery.switch",
                {
                    "from_executor": name,
                    "switch_index": self._memory.switch_count,
                    "trigger_code": code.value,
                    "no_switch_back": True,
                },
            )
            self._transition(AgentState.OBSERVING, "refresh observation for switch")
            return None, True
        return current, False

    def run(self, task: TaskSchema, environment: ExecutionEnvironment) -> RunResult:
        """Plan semantic subtasks and execute them with bounded local recovery."""

        self._fsm = AgentFSM()
        self._queue.clear()
        self._plan = None
        self._run_id = str(uuid4())
        self._task_id = task.task_id
        self.gateway.reset()
        self._memory = self.memory_store.create(self._run_id, task.task_id)
        event_start = len(self.events.events)
        verification: VerificationResult | None = None
        self._emit("run.started", {"task_schema_version": task.schema_version})
        self._transition(AgentState.VALIDATING_TASK, "validate TaskSchema")
        try:
            task.validate()
        except AgentError as exc:
            return self._fail(
                exc.code,
                str(exc),
                verification=verification,
                event_start=event_start,
            )
        self._transition(AgentState.PLANNING, "build semantic TaskPlan")
        try:
            self._plan = self.planner.plan(task, self._run_id)
            self._plan.validate()
        except AgentError as exc:
            return self._fail(exc.code, str(exc), verification, event_start)
        except (TypeError, ValueError) as exc:
            return self._fail(
                FailureCode.INVALID_TASK,
                f"semantic planner rejected task constraints: {exc}",
                verification,
                event_start,
            )
        self._memory.plan_id = self._plan.plan_id
        self._emit(
            "task_plan.created",
            {
                "plan_id": self._plan.plan_id,
                "subtask_count": len(self._plan.subtasks),
                "semantic_only": True,
            },
        )

        subtask_index = 0
        subtask = self._plan.subtasks[subtask_index]
        subtask.status = SubtaskStatus.READY
        active_task = subtask.as_task(task)
        self._memory.active_subtask_id = subtask.subtask_id
        self._transition(
            AgentState.OBSERVING, "plan accepted; perceive current subtask"
        )

        current: Executor | None = None
        excluded: set[str] = set()
        local_replans: dict[str, int] = {}
        strategy_attempt = 0
        subtask_iterations = 0
        decisions_in_strategy_attempt = 0
        last_observation: Observation | None = None
        action_has_executed = False

        def reject_observation(
            exc: AgentError,
            phase: str,
        ) -> RunResult:
            if action_has_executed and exc.code in {
                FailureCode.OBSERVATION_INVALID,
                FailureCode.OBSERVATION_GT_FORBIDDEN,
            }:
                return self._safe_stop(
                    environment,
                    exc.code,
                    f"unsafe online observation {phase} after action: {exc}",
                    verification,
                    event_start,
                )
            return self._fail(exc.code, str(exc), verification, event_start)

        while True:
            try:
                last_observation = self.gateway.ingest_online(environment.observe())
            except AgentError as exc:
                subtask.status = SubtaskStatus.FAILED
                return reject_observation(exc, "during control loop")
            except Exception as exc:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    FailureCode.SYSTEM_FAULT,
                    f"environment observe failed: {exc}",
                    verification,
                    event_start,
                )
            self._memory.last_observation_id = last_observation.observation_id
            stopped = self._check_system_fault(
                last_observation, environment, verification, event_start
            )
            if stopped is not None:
                subtask.status = SubtaskStatus.FAILED
                return stopped

            if subtask.status is SubtaskStatus.READY and subtask.preconditions:
                precondition_frames = [last_observation]
                try:
                    while len(precondition_frames) < self.verification_frames:
                        frame = self.gateway.ingest_online(environment.observe())
                        last_observation = frame
                        self._memory.last_observation_id = frame.observation_id
                        stopped = self._check_system_fault(
                            frame, environment, verification, event_start
                        )
                        if stopped is not None:
                            subtask.status = SubtaskStatus.FAILED
                            return stopped
                        precondition_frames.append(frame)
                except AgentError as exc:
                    subtask.status = SubtaskStatus.FAILED
                    return reject_observation(exc, "during precondition check")
                except Exception as exc:
                    subtask.status = SubtaskStatus.FAILED
                    return self._safe_stop(
                        environment,
                        FailureCode.SYSTEM_FAULT,
                        f"precondition observation failed: {exc}",
                        verification,
                        event_start,
                    )
                precondition_task = TaskSchema(
                    task_id=active_task.task_id,
                    instruction=active_task.instruction,
                    task_type=active_task.task_type,
                    postconditions=subtask.preconditions,
                )
                precondition_result = self.verifier.verify(
                    precondition_task, precondition_frames
                )
                self._emit(
                    "subtask.preconditions_checked",
                    {
                        "subtask_id": subtask.subtask_id,
                        "verdict": precondition_result.verdict.value,
                        "frames": len(precondition_frames),
                    },
                )
                if precondition_result.verdict is not Verdict.PASS:
                    subtask.status = SubtaskStatus.FAILED
                    return self._fail(
                        precondition_result.code,
                        f"preconditions are {precondition_result.verdict.value} "
                        f"for {subtask.subtask_id}",
                        precondition_result,
                        event_start,
                    )

            # A repeat-until workflow must be a no-op when its observable
            # postcondition already holds (for example, a bin is already full).
            if subtask.repeat_until_postcondition and subtask_iterations == 0:
                precheck_frames = [last_observation]
                try:
                    while len(precheck_frames) < self.verification_frames:
                        frame = self.gateway.ingest_online(environment.observe())
                        last_observation = frame
                        self._memory.last_observation_id = frame.observation_id
                        stopped = self._check_system_fault(
                            frame, environment, verification, event_start
                        )
                        if stopped is not None:
                            subtask.status = SubtaskStatus.FAILED
                            return stopped
                        precheck_frames.append(frame)
                except AgentError as exc:
                    subtask.status = SubtaskStatus.FAILED
                    return reject_observation(exc, "during repeat precheck")
                except Exception as exc:
                    subtask.status = SubtaskStatus.FAILED
                    return self._safe_stop(
                        environment,
                        FailureCode.SYSTEM_FAULT,
                        f"repeat precheck observation failed: {exc}",
                        verification,
                        event_start,
                    )
                verification = self.verifier.verify(active_task, precheck_frames)
                self._emit(
                    "subtask.prechecked",
                    {
                        "subtask_id": subtask.subtask_id,
                        "verdict": verification.verdict.value,
                        "frames": len(precheck_frames),
                    },
                )
                if verification.verdict is Verdict.PASS:
                    subtask.status = SubtaskStatus.VERIFIED
                    self._emit(
                        "subtask.verified",
                        {
                            "subtask_id": subtask.subtask_id,
                            "iterations": 0,
                            "no_action_required": True,
                        },
                    )
                    if subtask_index + 1 == len(self._plan.subtasks):
                        self._transition(
                            AgentState.SUCCEEDED,
                            "repeat postcondition already satisfied",
                        )
                        self._memory.last_failure_code = FailureCode.NONE.value
                        self._emit(
                            "run.succeeded",
                            {"executor": None, "no_action_required": True},
                        )
                        return self._result(
                            FailureCode.NONE,
                            "task plan already satisfied by online observations",
                            verification,
                            event_start,
                        )
                    self._transition(
                        AgentState.ADVANCING_SUBTASK,
                        "advance without action after repeat precheck",
                    )
                    subtask_index += 1
                    subtask = self._plan.subtasks[subtask_index]
                    verified_ids = {
                        item.subtask_id
                        for item in self._plan.subtasks
                        if item.status is SubtaskStatus.VERIFIED
                    }
                    if not set(subtask.depends_on).issubset(verified_ids):
                        subtask.status = SubtaskStatus.FAILED
                        return self._fail(
                            FailureCode.INVALID_TASK,
                            f"dependencies not verified for {subtask.subtask_id}",
                            verification,
                            event_start,
                        )
                    subtask.status = SubtaskStatus.READY
                    active_task = subtask.as_task(task)
                    self._memory.active_subtask_id = subtask.subtask_id
                    current = None
                    local_replans = {}
                    strategy_attempt = 0
                    subtask_iterations = 0
                    decisions_in_strategy_attempt = 0
                    self._transition(
                        AgentState.OBSERVING,
                        "re-perceive before executing next subtask",
                    )
                    continue

            self._transition(
                AgentState.SELECTING_EXECUTOR,
                "select or retain a healthy compatible strategy",
            )
            if current is None:
                try:
                    current = self.router.select(active_task, frozenset(excluded))
                except AgentError as exc:
                    subtask.status = SubtaskStatus.FAILED
                    return self._fail(exc.code, str(exc), verification, event_start)
                strategy_attempt += 1
                name = current.descriptor.name
                self._memory.active_executor = name
                self._memory.executor_history.append(name)
                self._memory.replan_counts.setdefault(name, 0)
                subtask.status = SubtaskStatus.RUNNING
                self._emit(
                    "executor.selected",
                    {
                        "executor": name,
                        "subtask_id": subtask.subtask_id,
                        "strategy_attempt": strategy_attempt,
                        "checkpoint_sha": current.descriptor.checkpoint_sha,
                        "norm_stats_sha": current.descriptor.norm_stats_sha,
                    },
                )
            name = current.descriptor.name
            self._transition(AgentState.EXECUTING, f"request action chunk from {name}")
            context = ExecutionContext(
                run_id=self._run_id,
                strategy_attempt=strategy_attempt,
                replan_index=local_replans.get(name, 0),
                step_id=subtask_iterations,
                timeout_ms=self.executor_timeout_ms,
            )
            try:
                chunk = current.plan(active_task, last_observation, context)
                if chunk.task_id != active_task.task_id or chunk.executor != name:
                    raise AgentError(
                        FailureCode.ACTION_CONTRACT_INVALID,
                        "action task_id/executor does not match active strategy",
                    )
                decision = self.safety.validate_and_limit(chunk, last_observation)
                if not decision.accepted or decision.chunk is None:
                    raise AgentError(decision.code, decision.reason)
            except AgentError as exc:
                current, should_continue = self._recover(
                    task=active_task,
                    current=current,
                    code=exc.code,
                    reason=str(exc),
                    excluded=excluded,
                    local_replans=local_replans,
                )
                if should_continue:
                    decisions_in_strategy_attempt = 0
                    continue
                subtask.status = SubtaskStatus.FAILED
                return self._fail(
                    FailureCode.RECOVERY_EXHAUSTED,
                    f"recovery exhausted after {exc.code.value}: {exc}",
                    verification,
                    event_start,
                )
            except Exception as exc:
                current, should_continue = self._recover(
                    task=active_task,
                    current=current,
                    code=FailureCode.EXECUTOR_RUNTIME,
                    reason=str(exc),
                    excluded=excluded,
                    local_replans=local_replans,
                )
                if should_continue:
                    decisions_in_strategy_attempt = 0
                    continue
                subtask.status = SubtaskStatus.FAILED
                return self._fail(
                    FailureCode.RECOVERY_EXHAUSTED,
                    f"unexpected executor error; recovery exhausted: {exc}",
                    verification,
                    event_start,
                )

            subtask_iterations += 1
            decisions_in_strategy_attempt += 1
            if decision.limited_axes:
                self._emit(
                    "safety.action_limited",
                    {
                        "chunk_id": decision.chunk.chunk_id,
                        "limited_axes": list(decision.limited_axes),
                    },
                )
            # Receding-horizon policy: never execute an action generated before
            # the newest observation. Remaining chunk steps are intentionally
            # discarded and the VLA is queried again after verification.
            self._queue.append(decision.chunk.steps[0])
            self._emit(
                "action_chunk.accepted",
                {
                    "chunk_id": decision.chunk.chunk_id,
                    "subtask_id": subtask.subtask_id,
                    "subtask_iteration": subtask_iterations,
                    "proposed_steps": len(decision.chunk.steps),
                    "executed_steps": 1,
                    "discarded_steps": len(decision.chunk.steps) - 1,
                    "execution_policy": "receding_horizon_one_step",
                    "contract_version": decision.chunk.contract_version,
                },
            )
            try:
                while self._queue:
                    action = self._queue.popleft()
                    raw_observation = environment.step(action)
                    action_has_executed = True
                    last_observation = self.gateway.ingest_online(raw_observation)
                    self._memory.last_observation_id = last_observation.observation_id
                    stopped = self._check_system_fault(
                        last_observation, environment, verification, event_start
                    )
                    if stopped is not None:
                        subtask.status = SubtaskStatus.FAILED
                        return stopped
            except AgentError as exc:
                if exc.code in {
                    FailureCode.OBSERVATION_INVALID,
                    FailureCode.OBSERVATION_GT_FORBIDDEN,
                }:
                    subtask.status = SubtaskStatus.FAILED
                    return self._safe_stop(
                        environment,
                        exc.code,
                        f"unsafe online observation after action: {exc}",
                        verification,
                        event_start,
                    )
                current, should_continue = self._recover(
                    task=active_task,
                    current=current,
                    code=exc.code,
                    reason=str(exc),
                    excluded=excluded,
                    local_replans=local_replans,
                )
                if should_continue:
                    decisions_in_strategy_attempt = 0
                    continue
                subtask.status = SubtaskStatus.FAILED
                return self._fail(
                    FailureCode.RECOVERY_EXHAUSTED,
                    f"execution/recovery exhausted: {exc}",
                    verification,
                    event_start,
                )
            except Exception as exc:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    FailureCode.SYSTEM_FAULT,
                    f"environment execution failed: {exc}",
                    verification,
                    event_start,
                )
            self._memory.completed_chunk_ids.append(decision.chunk.chunk_id)

            self._transition(AgentState.VERIFYING, "verify subtask postconditions")
            frames = [last_observation]
            try:
                while len(frames) < self.verification_frames:
                    frame = self.gateway.ingest_online(environment.observe())
                    self._memory.last_observation_id = frame.observation_id
                    stopped = self._check_system_fault(
                        frame, environment, verification, event_start
                    )
                    if stopped is not None:
                        subtask.status = SubtaskStatus.FAILED
                        return stopped
                    frames.append(frame)
            except AgentError as exc:
                subtask.status = SubtaskStatus.FAILED
                return reject_observation(exc, "during verification")
            except Exception as exc:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    FailureCode.SYSTEM_FAULT,
                    f"verification observation failed: {exc}",
                    verification,
                    event_start,
                )
            verification = self.verifier.verify(active_task, frames)
            self._emit(
                "verification.completed",
                {
                    "subtask_id": subtask.subtask_id,
                    "verdict": verification.verdict.value,
                    "failure_code": verification.code.value,
                    "frames": len(frames),
                    "conditions": [
                        {
                            "kind": item.kind,
                            "verdict": item.verdict.value,
                            "pass_votes": item.pass_votes,
                            "fail_votes": item.fail_votes,
                            "uncertain_votes": item.uncertain_votes,
                            "required_votes": item.required_votes,
                        }
                        for item in verification.conditions
                    ],
                },
            )
            if verification.verdict is Verdict.PASS:
                subtask.status = SubtaskStatus.VERIFIED
                self._emit(
                    "subtask.verified",
                    {
                        "subtask_id": subtask.subtask_id,
                        "iterations": subtask_iterations,
                    },
                )
                if subtask_index + 1 == len(self._plan.subtasks):
                    self._transition(
                        AgentState.SUCCEEDED, "all subtasks and postconditions passed"
                    )
                    self._memory.last_failure_code = FailureCode.NONE.value
                    self._emit("run.succeeded", {"executor": name})
                    return self._result(
                        FailureCode.NONE,
                        "task plan completed and verified",
                        verification,
                        event_start,
                    )
                self._transition(
                    AgentState.ADVANCING_SUBTASK,
                    "advance after verified current subtask",
                )
                subtask_index += 1
                subtask = self._plan.subtasks[subtask_index]
                verified_ids = {
                    item.subtask_id
                    for item in self._plan.subtasks
                    if item.status is SubtaskStatus.VERIFIED
                }
                if not set(subtask.depends_on).issubset(verified_ids):
                    subtask.status = SubtaskStatus.FAILED
                    return self._fail(
                        FailureCode.INVALID_TASK,
                        f"dependencies not verified for {subtask.subtask_id}",
                        verification,
                        event_start,
                    )
                subtask.status = SubtaskStatus.READY
                active_task = subtask.as_task(task)
                self._memory.active_subtask_id = subtask.subtask_id
                current = None
                local_replans = {}
                strategy_attempt = 0
                subtask_iterations = 0
                decisions_in_strategy_attempt = 0
                self._transition(
                    AgentState.OBSERVING,
                    "re-perceive before executing next subtask",
                )
                continue

            if (
                subtask.repeat_until_postcondition
                and verification.verdict is Verdict.FAIL
                and subtask_iterations < subtask.max_iterations
            ):
                self._transition(
                    AgentState.ADVANCING_SUBTASK,
                    "repeat semantic subtask until its postcondition",
                )
                self._emit(
                    "subtask.iteration_incomplete",
                    {
                        "subtask_id": subtask.subtask_id,
                        "iteration": subtask_iterations,
                        "max_iterations": subtask.max_iterations,
                    },
                )
                self._transition(
                    AgentState.OBSERVING,
                    "re-perceive before next semantic loop iteration",
                )
                continue
            if (
                subtask.repeat_until_postcondition
                and subtask_iterations >= subtask.max_iterations
            ):
                subtask.status = SubtaskStatus.FAILED
                return self._fail(
                    FailureCode.RECOVERY_EXHAUSTED,
                    f"semantic loop reached max_iterations={subtask.max_iterations}",
                    verification,
                    event_start,
                )

            if decisions_in_strategy_attempt < self.max_decisions_per_strategy_attempt:
                self._emit(
                    "closed_loop.redecision",
                    {
                        "subtask_id": subtask.subtask_id,
                        "executor": name,
                        "verdict": verification.verdict.value,
                        "decision_index": decisions_in_strategy_attempt,
                        "decision_budget": self.max_decisions_per_strategy_attempt,
                        "next_observation_required": True,
                    },
                )
                self._transition(
                    AgentState.OBSERVING,
                    "postcondition not met; re-observe before next VLA decision",
                )
                continue

            current, should_continue = self._recover(
                task=active_task,
                current=current,
                code=verification.code,
                reason=verification.verdict.value,
                excluded=excluded,
                local_replans=local_replans,
            )
            if should_continue:
                decisions_in_strategy_attempt = 0
                continue
            subtask.status = SubtaskStatus.FAILED
            return self._fail(
                FailureCode.RECOVERY_EXHAUSTED,
                f"postcondition {verification.verdict.value}; recovery exhausted",
                verification,
                event_start,
            )
