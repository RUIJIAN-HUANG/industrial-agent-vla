"""Formal V2 π0.5 closed-loop Supervisor for either configured arm."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import replace
from threading import Lock
from typing import Any
from uuid import uuid4

from .contracts import Postcondition, Subtask, SubtaskStatus, TaskPlan, TaskSchema
from .environment import ExecutionEnvironment, execution_guard_digest
from .errors import AgentError, FailureCode
from .executor import ExecutionContext, Executor
from .fsm import AgentFSM, AgentState
from .run_result import RunResult
from .safety import ActionSafetyValidator, SafetyPolicy, safety_state_failure
from .v2_observation import V2ObservationGateway
from .v2_task_profile import (
    V2_FORMAL_TASK_IDS,
    V2_PROFILE_ID,
    V2_SCENE_ID,
    require_formal_v2_task,
)


V2_CONTROL_TOKEN = "A_ONLY"
BIN_HANDOFF_TASK_ID = "BIN01_TO_FINISHED01"


def _tuple3(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain exactly 3 numbers")
    try:
        return tuple(float(item) for item in value)  # type: ignore[return-value]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain exactly 3 numbers") from exc


def v2_safety_policy_from_config(config: Mapping[str, Any]) -> SafetyPolicy:
    raw = config.get("safety")
    if not isinstance(raw, Mapping):
        raise ValueError("V2 config safety must be an object")
    limits = raw.get("axis_abs_limits")
    if not isinstance(limits, (list, tuple)) or len(limits) != 7:
        raise ValueError("V2 safety.axis_abs_limits must contain 7 numbers")
    workspace = raw.get("workspace_by_arm")
    if not isinstance(workspace, Mapping):
        raise ValueError("V2 safety.workspace_by_arm must be an object")
    arm_a = workspace.get("Arm_A")
    arm_b = workspace.get("Arm_B")
    if not isinstance(arm_a, Mapping) or not isinstance(arm_b, Mapping):
        raise ValueError("V2 safety requires Arm_A and parked Arm_B workspaces")
    return SafetyPolicy(
        axis_abs_limits=tuple(float(item) for item in limits),  # type: ignore[arg-type]
        arm_a_workspace_min_m=_tuple3(arm_a.get("min_m"), "Arm_A.min_m"),
        arm_a_workspace_max_m=_tuple3(arm_a.get("max_m"), "Arm_A.max_m"),
        arm_b_workspace_min_m=_tuple3(arm_b.get("min_m"), "Arm_B.min_m"),
        arm_b_workspace_max_m=_tuple3(arm_b.get("max_m"), "Arm_B.max_m"),
        max_chunk_steps=int(raw.get("max_chunk_steps", 32)),
    )


class V2TaskPlanner:
    """Map one frozen V2 user choice to an ordered π0.5 execution plan."""

    def plan(self, task: TaskSchema, run_id: str) -> TaskPlan:
        task.validate()
        spec = require_formal_v2_task(task.task_id)
        if task.instruction != spec.instruction:
            raise ValueError("task instruction does not match the frozen V2 catalog")
        if task.target_object != spec.target_object:
            raise ValueError("task target_object does not match the frozen V2 catalog")
        expected_target = spec.target_slot or spec.target_zone
        if task.target_location != expected_target:
            raise ValueError(
                "task target_location does not match the frozen V2 catalog"
            )
        if task.metadata.get("profile_id") != V2_PROFILE_ID:
            raise ValueError(f"task metadata.profile_id must be {V2_PROFILE_ID!r}")
        if task.constraints.get("scene_id") != V2_SCENE_ID:
            raise ValueError(f"task constraints.scene_id must be {V2_SCENE_ID!r}")
        if spec.task_id == BIN_HANDOFF_TASK_ID:
            handoff_subtask = Subtask(
                subtask_id="BIN01_TO_HANDOFF_CENTER",
                sequence=1,
                instruction="把Bin_01搬到HANDOFF_CENTER",
                task_type=task.task_type,
                preconditions=(),
                postconditions=(
                    Postcondition(
                        kind="object_in_zone",
                        object_id="Bin_01",
                        zone_id="HANDOFF_CENTER",
                    ),
                ),
                assigned_executor="pi05",
                arm_id="Arm_A",
                repeat_until_postcondition=True,
                max_iterations=100,
                status=SubtaskStatus.READY,
            )
            finish_subtask = Subtask(
                subtask_id="BIN01_HANDOFF_TO_FINISHED01",
                sequence=2,
                instruction=spec.instruction,
                task_type=task.task_type,
                preconditions=handoff_subtask.postconditions,
                postconditions=task.postconditions,
                depends_on=(handoff_subtask.subtask_id,),
                assigned_executor="pi05",
                arm_id="Arm_B",
                repeat_until_postcondition=True,
                max_iterations=100,
                status=SubtaskStatus.PENDING,
            )
            subtasks = [handoff_subtask, finish_subtask]
        else:
            subtasks = [
                Subtask(
                    subtask_id=spec.task_id,
                    sequence=1,
                    instruction=spec.instruction,
                    task_type=task.task_type,
                    preconditions=(),
                    postconditions=task.postconditions,
                    assigned_executor="pi05",
                    arm_id=spec.active_arm,
                    repeat_until_postcondition=True,
                    max_iterations=100,
                    status=SubtaskStatus.READY,
                )
            ]
        plan = TaskPlan(
            plan_id=f"v2-plan-{uuid4().hex}",
            episode_id=run_id,
            task_id=task.task_id,
            subtasks=subtasks,
        )
        plan.validate()
        return plan


class V2Supervisor:
    """Single-policy V2 Supervisor with bounded, ordered execution."""

    def __init__(
        self,
        executor: Executor,
        *,
        safety: ActionSafetyValidator,
        executor_timeout_ms: int,
        max_decisions: int,
        verification_frames: int,
        terminal_min_confidence: float,
        terminal_required_votes: int,
    ) -> None:
        if executor.descriptor.name != "pi05":
            raise ValueError("formal V2 Supervisor requires the pi05 executor")
        if executor_timeout_ms < 1 or max_decisions < 1 or verification_frames < 1:
            raise ValueError("V2 timeout and decision budget must be positive")
        if not 0 <= terminal_min_confidence <= 1:
            raise ValueError("V2 terminal_min_confidence must be in [0, 1]")
        if terminal_required_votes < 1:
            raise ValueError("V2 terminal_required_votes must be positive")
        self.executor = executor
        self.safety = safety
        self.executor_timeout_ms = executor_timeout_ms
        self.max_decisions = max_decisions
        self.verification_frames = verification_frames
        self.terminal_min_confidence = terminal_min_confidence
        self.terminal_required_votes = terminal_required_votes
        self.planner = V2TaskPlanner()
        self._run_lock = Lock()

    @classmethod
    def from_config(
        cls,
        executor: Executor,
        config: Mapping[str, Any],
    ) -> "V2Supervisor":
        if str(config.get("config_version", "")).split(".", 1)[0] != "2":
            raise ValueError("V1 is abolished; Supervisor requires a V2 config")
        if config.get("profile_id") != V2_PROFILE_ID:
            raise ValueError(f"V2 config profile_id must be {V2_PROFILE_ID!r}")
        if config.get("scene_id") != V2_SCENE_ID:
            raise ValueError(f"V2 config scene_id must be {V2_SCENE_ID!r}")
        if frozenset(config.get("formal_task_ids", ())) != V2_FORMAL_TASK_IDS:
            raise ValueError(
                "V2 config formal_task_ids do not match the frozen catalog"
            )
        verification = config.get("verification")
        recovery = config.get("recovery")
        if not isinstance(verification, Mapping) or not isinstance(recovery, Mapping):
            raise ValueError("V2 config requires verification and recovery objects")
        if verification.get("frames") != 3:
            raise ValueError("formal V2 verification requires exactly 3 frames")
        return cls(
            executor,
            safety=ActionSafetyValidator(v2_safety_policy_from_config(config)),
            executor_timeout_ms=int(config.get("executor_timeout_ms", 0)),
            max_decisions=int(recovery.get("max_decisions_per_task", 0)),
            verification_frames=int(verification.get("frames", 0)),
            terminal_min_confidence=float(verification.get("min_confidence", -1)),
            terminal_required_votes=int(verification.get("required_votes", 0)),
        )

    def run(self, task: TaskSchema, environment: ExecutionEnvironment) -> RunResult:
        run_id = f"v2-run-{uuid4().hex}"
        if not self._run_lock.acquire(blocking=False):
            busy_fsm = AgentFSM()
            busy_fsm.transition(
                AgentState.VALIDATING_TASK, "check V2 Supervisor availability"
            )
            busy_fsm.transition(AgentState.FAILED, "V2 Supervisor is already running")
            return self._result(
                run_id,
                task.task_id,
                busy_fsm,
                False,
                FailureCode.AGENT_BUSY,
                "V2 Supervisor is already running",
                None,
                (),
                ("NONE",),
            )
        fsm = AgentFSM()
        gateway = V2ObservationGateway()
        plan: TaskPlan | None = None
        executor_history: list[str] = []
        token_history: list[str] = []
        motion_started = False
        try:
            fsm.transition(AgentState.VALIDATING_TASK, "validate frozen V2 task")
            try:
                plan = self.planner.plan(task, run_id)
            except (AgentError, TypeError, ValueError) as exc:
                fsm.transition(AgentState.FAILED, "V2 task validation failed")
                code = (
                    exc.code
                    if isinstance(exc, AgentError)
                    else FailureCode.INVALID_TASK
                )
                return self._result(
                    run_id,
                    task.task_id,
                    fsm,
                    False,
                    code,
                    str(exc),
                    plan,
                    executor_history,
                    token_history,
                )
            fsm.transition(
                AgentState.PLANNING,
                f"V2 task mapped to {len(plan.subtasks)} ordered pi05 subtask(s)",
            )
            if not self.executor.health():
                fsm.transition(AgentState.FAILED, "pi05 health check failed")
                return self._result(
                    run_id,
                    task.task_id,
                    fsm,
                    False,
                    FailureCode.EXECUTOR_UNAVAILABLE,
                    "π0.5 is not ready",
                    plan,
                    executor_history,
                    token_history,
                )
            fsm.transition(AgentState.OBSERVING, "read initial V2 observation")
            try:
                observation = gateway.ingest_online(environment.observe())
                action_count = 0
                handoff_verification_reads = 0
                while action_count < self.max_decisions:
                    safety_failure = safety_state_failure(observation)
                    if safety_failure:
                        code, reason = safety_failure
                        return self._stop_result(
                            environment,
                            run_id,
                            task.task_id,
                            fsm,
                            code,
                            reason,
                            plan,
                            executor_history,
                            token_history,
                        )
                    terminal, terminal_failure = self._terminal_state(observation, task)
                    if terminal_failure:
                        return self._stop_result(
                            environment,
                            run_id,
                            task.task_id,
                            fsm,
                            FailureCode.POSTCONDITION_FAILED,
                            terminal_failure,
                            plan,
                            executor_history,
                            token_history,
                        )
                    if terminal:
                        verified, verify_message = self._verify_terminal_window(
                            environment, gateway, observation, task
                        )
                        if not verified:
                            return self._stop_result(
                                environment,
                                run_id,
                                task.task_id,
                                fsm,
                                FailureCode.VERIFICATION_UNCERTAIN,
                                verify_message,
                                plan,
                                executor_history,
                                token_history,
                            )
                        fsm.transition(
                            AgentState.SUCCEEDED, "V2 terminal evidence accepted"
                        )
                        for subtask in plan.subtasks:
                            subtask.status = SubtaskStatus.VERIFIED
                        self._append_control_token(token_history, "NONE")
                        return self._result(
                            run_id,
                            task.task_id,
                            fsm,
                            True,
                            FailureCode.NONE,
                            "V2 task completed",
                            plan,
                            executor_history,
                            token_history,
                        )

                    control_token = self._observation_control_token(task, observation)
                    self._append_control_token(token_history, control_token)
                    if control_token == "HANDOFF_VERIFY":
                        handoff_verification_reads += 1
                        if handoff_verification_reads > self.verification_frames * 2:
                            return self._stop_result(
                                environment,
                                run_id,
                                task.task_id,
                                fsm,
                                FailureCode.VERIFICATION_UNCERTAIN,
                                "handoff verification did not reach B_ONLY within the fresh-frame budget",
                                plan,
                                executor_history,
                                token_history,
                            )
                        observation = gateway.ingest_online(environment.observe())
                        continue

                    active_subtask = self._active_subtask(task, plan, observation)
                    active_subtask.status = SubtaskStatus.RUNNING
                    arm_id = active_subtask.arm_id or "Arm_A"
                    expected_token = "A_ONLY" if arm_id == "Arm_A" else "B_ONLY"
                    if control_token != expected_token:
                        raise ValueError(
                            f"online control token {control_token!r} does not match {arm_id}"
                        )
                    fsm.transition(
                        AgentState.ASSIGNING_ROLE,
                        f"grant {control_token} to pi05",
                    )
                    fsm.transition(
                        AgentState.EXECUTING, "request and execute one 7D action"
                    )
                    context = ExecutionContext(
                        run_id=run_id,
                        strategy_attempt=0,
                        replan_index=action_count,
                        step_id=action_count,
                        timeout_ms=self.executor_timeout_ms,
                        original_instruction=active_subtask.instruction,
                        arm_id=arm_id,
                    )
                    target_location = next(
                        (
                            condition.zone_id
                            for condition in active_subtask.postconditions
                            if condition.zone_id is not None
                        ),
                        task.target_location,
                    )
                    execution_task = replace(
                        task,
                        instruction=active_subtask.instruction,
                        target_location=target_location,
                        postconditions=active_subtask.postconditions,
                        metadata={
                            **dict(task.metadata),
                            "subtask_id": active_subtask.subtask_id,
                            "arm_id": arm_id,
                        },
                    )
                    chunk = self.executor.plan(execution_task, observation, context)
                    executor_history.append("pi05")
                    decision = self.safety.validate_and_limit(
                        chunk,
                        observation,
                        arm_id=arm_id,
                        control_token=control_token,
                    )
                    if not decision.accepted or decision.chunk is None:
                        return self._stop_result(
                            environment,
                            run_id,
                            task.task_id,
                            fsm,
                            decision.code,
                            decision.reason,
                            plan,
                            executor_history,
                            token_history,
                        )
                    action = decision.chunk.steps[0]
                    motion_started = True
                    next_raw = environment.step(
                        action,
                        arm_id=arm_id,
                        control_token=control_token,
                        command_id=f"{run_id}-command-{action_count:06d}",
                        expected_observation_id=observation.observation_id,
                        expected_state_digest=execution_guard_digest(observation.data),
                    )
                    action_count += 1
                    fsm.transition(AgentState.VERIFYING, "ingest next sensor frame")
                    observation = gateway.ingest_online(next_raw)
                    terminal, terminal_failure = self._terminal_state(observation, task)
                    if terminal_failure:
                        return self._stop_result(
                            environment,
                            run_id,
                            task.task_id,
                            fsm,
                            FailureCode.POSTCONDITION_FAILED,
                            terminal_failure,
                            plan,
                            executor_history,
                            token_history,
                        )
                    if terminal:
                        verified, verify_message = self._verify_terminal_window(
                            environment, gateway, observation, task
                        )
                        if not verified:
                            return self._stop_result(
                                environment,
                                run_id,
                                task.task_id,
                                fsm,
                                FailureCode.VERIFICATION_UNCERTAIN,
                                verify_message,
                                plan,
                                executor_history,
                                token_history,
                            )
                        fsm.transition(
                            AgentState.SUCCEEDED, "V2 terminal evidence accepted"
                        )
                        for subtask in plan.subtasks:
                            subtask.status = SubtaskStatus.VERIFIED
                        self._append_control_token(token_history, "NONE")
                        return self._result(
                            run_id,
                            task.task_id,
                            fsm,
                            True,
                            FailureCode.NONE,
                            "V2 task completed",
                            plan,
                            executor_history,
                            token_history,
                        )
                    if action_count < self.max_decisions:
                        fsm.transition(AgentState.REPLANNING, "terminal not reached")
                        fsm.transition(AgentState.OBSERVING, "continue V2 closed loop")

                return self._stop_result(
                    environment,
                    run_id,
                    task.task_id,
                    fsm,
                    FailureCode.RECOVERY_EXHAUSTED,
                    f"V2 decision budget exhausted after {self.max_decisions} actions",
                    plan,
                    executor_history,
                    token_history,
                )
            except (
                AgentError,
                OSError,
                RuntimeError,
                TimeoutError,
                TypeError,
                ValueError,
            ) as exc:
                code = (
                    exc.code
                    if isinstance(exc, AgentError)
                    else FailureCode.SYSTEM_FAULT
                )
                if motion_started or fsm.state not in {
                    AgentState.PLANNING,
                    AgentState.FAILED,
                }:
                    return self._stop_result(
                        environment,
                        run_id,
                        task.task_id,
                        fsm,
                        code,
                        str(exc),
                        plan,
                        executor_history,
                        token_history,
                    )
                if fsm.state != AgentState.FAILED:
                    fsm.transition(
                        AgentState.FAILED, "V2 execution failed before motion"
                    )
                return self._result(
                    run_id,
                    task.task_id,
                    fsm,
                    False,
                    code,
                    str(exc),
                    plan,
                    executor_history,
                    token_history,
                )
        finally:
            self._run_lock.release()

    @staticmethod
    def _active_subtask(task: TaskSchema, plan: TaskPlan, observation: Any) -> Subtask:
        """Select the current relay leg from the authoritative active-arm lease."""

        if task.task_id != BIN_HANDOFF_TASK_ID:
            return plan.subtasks[0]
        robot = observation.data.get("robot")
        active_arm = robot.get("active_arm") if isinstance(robot, Mapping) else None
        if active_arm == "Arm_A":
            return plan.subtasks[0]
        if active_arm == "Arm_B":
            plan.subtasks[0].status = SubtaskStatus.VERIFIED
            return plan.subtasks[1]
        raise ValueError(
            "BIN01_TO_FINISHED01 requires an active Arm_A or Arm_B lease "
            "before requesting an action"
        )

    @staticmethod
    def _observation_control_token(task: TaskSchema, observation: Any) -> str:
        robot = observation.data.get("robot")
        active_arm = robot.get("active_arm") if isinstance(robot, Mapping) else None
        if active_arm == "Arm_A":
            return "A_ONLY"
        if active_arm == "Arm_B":
            return "B_ONLY"
        if active_arm == "NONE" and task.task_id == BIN_HANDOFF_TASK_ID:
            raw_task = observation.data.get("task")
            if isinstance(raw_task, Mapping) and raw_task.get("terminal") is True:
                return "NONE"
            return "HANDOFF_VERIFY"
        if active_arm == "NONE":
            return "NONE"
        raise ValueError("online observation has no valid active-arm lease")

    @staticmethod
    def _append_control_token(history: list[str], token: str) -> None:
        if token not in {"A_ONLY", "HANDOFF_VERIFY", "B_ONLY", "NONE"}:
            raise ValueError(f"invalid V2 control token: {token!r}")
        if not history or history[-1] != token:
            history.append(token)

    def _terminal_state(
        self, observation: Any, task: TaskSchema
    ) -> tuple[bool, str | None]:
        raw = observation.data.get("task")
        if not isinstance(raw, Mapping) or raw.get("task_id") != task.task_id:
            return False, "online task identity changed during V2 execution"
        if raw.get("status") == "FAILED":
            return False, "online V2 task-state provider reported FAILED"
        if raw.get("terminal") is not True:
            return False, None
        confidence = float(raw.get("terminal_confidence", 0))
        votes = int(raw.get("verification_votes", 0))
        if confidence < self.terminal_min_confidence:
            return False, "terminal evidence confidence is below the frozen threshold"
        if votes < self.terminal_required_votes:
            return False, "terminal evidence does not have enough verification votes"
        return True, None

    def _verify_terminal_window(
        self,
        environment: ExecutionEnvironment,
        gateway: V2ObservationGateway,
        first_observation: Any,
        task: TaskSchema,
    ) -> tuple[bool, str]:
        """Independently require a fresh multi-frame terminal window."""

        frames = [first_observation]
        for _ in range(self.verification_frames - 1):
            frames.append(gateway.ingest_online(environment.observe()))
        pass_votes = 0
        for frame in frames:
            safety_failure = safety_state_failure(frame)
            if safety_failure:
                return False, safety_failure[1]
            terminal, terminal_failure = self._terminal_state(frame, task)
            if terminal_failure:
                return False, terminal_failure
            if terminal:
                pass_votes += 1
        if pass_votes < self.terminal_required_votes:
            return (
                False,
                (
                    f"V2 terminal verification received {pass_votes}/"
                    f"{self.verification_frames} passing fresh frames; "
                    f"required {self.terminal_required_votes}"
                ),
            )
        return True, "V2 terminal verification passed"

    def _stop_result(
        self,
        environment: ExecutionEnvironment,
        run_id: str,
        task_id: str,
        fsm: AgentFSM,
        code: FailureCode,
        message: str,
        plan: TaskPlan | None,
        executor_history: list[str],
        token_history: list[str],
    ) -> RunResult:
        try:
            receipt = environment.safe_stop(message)
            confirmed = receipt.confirmed
        except (AgentError, OSError, RuntimeError, TimeoutError, TypeError, ValueError):
            confirmed = False
        target = AgentState.SAFE_STOPPED if confirmed else AgentState.SAFE_STOP_FAILED
        fsm.force_safety_terminal(target, message)
        self._append_control_token(token_history, "NONE")
        return self._result(
            run_id,
            task_id,
            fsm,
            False,
            code,
            message,
            plan,
            executor_history,
            token_history,
        )

    @staticmethod
    def _result(
        run_id: str,
        task_id: str,
        fsm: AgentFSM,
        success: bool,
        code: FailureCode,
        message: str,
        plan: TaskPlan | None,
        executor_history: Any,
        token_history: Any,
    ) -> RunResult:
        return RunResult(
            run_id=run_id,
            task_id=task_id,
            state=fsm.state,
            success=success,
            failure_code=code,
            message=message,
            executor_history=tuple(executor_history),
            control_token_history=tuple(token_history),
            replan_counts={"pi05": max(0, len(executor_history) - 1)},
            transitions=tuple(fsm.history),
            verification=None,
            task_plan=plan.to_dict() if plan is not None else {},
            events=(),
        )


__all__ = [
    "V2_CONTROL_TOKEN",
    "V2Supervisor",
    "V2TaskPlanner",
    "v2_safety_policy_from_config",
]
