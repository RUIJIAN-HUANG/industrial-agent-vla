"""Supervisor Agent: fixed lifecycle, execution, verification, and recovery."""

from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass, replace
from math import isfinite
from queue import Empty, Queue
from threading import Lock, Thread
from typing import Any, Callable, Mapping, Sequence, TypeVar
from uuid import uuid4

from .contracts import (
    ActionChunk,
    ActionStep,
    Observation,
    SubtaskStatus,
    TaskPlan,
    TaskSchema,
)
from .environment import (
    ExecutionEnvironment,
    PreWriteStateStaleError,
    SafeStopReceipt,
    execution_guard_digest,
)
from .errors import AgentError, FailureCode
from .executor import (
    EXECUTOR_CONFIG_FIELDS,
    ExecutionContext,
    Executor,
    ExecutorRegistry,
    is_pinned_artifact_digest,
)
from .fsm import AgentFSM, AgentState, StateTransition
from .image_cas import ImageCasConfig
from .lifecycle import (
    ARM_A_PACK_HANDOFF_SUBTASK_ID,
    ARM_B_TRANSPORT_SUBTASK_ID,
    ControlToken,
    FixedDualVLAPlanner,
    FixedLifecycle,
    FixedTaskProfile,
    FROZEN_TOKEN_SEQUENCE,
    HANDOFF_CANDIDATE_CHECKED_EVENT_TYPE,
    HANDOFF_READY_EVENT_TYPE,
    HANDOFF_VERIFIED_EVENT_TYPE,
)
from .observation import FROZEN_IMAGE_HEIGHT, FROZEN_IMAGE_WIDTH, ObservationGateway
from .perception import (
    DetectionEvidenceSink,
    DetectionPacket,
    ImageReference,
    PerceptionAgent,
    PerceptionContext,
    PerceptionError,
    PerceptionMode,
    PERCEPTION_CONFIG_FIELDS,
)
from .safety import ActionSafetyValidator, SafetyPolicy, safety_state_failure
from .sync_contract import canonical_state_7d
from .telemetry import EventRecord, EventSink, MemoryStore, RunMemory
from .verifier import PostconditionVerifier, VerificationResult, Verdict

T = TypeVar("T")


def _invoke_with_hard_deadline(
    operation: Callable[[], T],
    timeout_ms: int,
) -> tuple[bool, T | None, BaseException | None]:
    """Run an untrusted adapter call behind a daemon watchdog."""

    result_queue: Queue[tuple[bool, Any]] = Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put((True, operation()))
        except BaseException as exc:  # keep adapter failures in the worker
            result_queue.put((False, exc))

    thread = Thread(target=worker, daemon=True)
    thread.start()
    thread.join(timeout_ms / 1_000)
    if thread.is_alive():
        return (
            False,
            None,
            TimeoutError(f"operation exceeded hard deadline of {timeout_ms} ms"),
        )
    try:
        succeeded, value = result_queue.get_nowait()
    except Empty:
        return False, None, RuntimeError("adapter worker exited without a result")
    if succeeded:
        return True, value, None
    assert isinstance(value, BaseException)
    return False, None, value


@dataclass(frozen=True)
class RunResult:
    run_id: str
    task_id: str
    state: AgentState
    success: bool
    failure_code: FailureCode
    message: str
    executor_history: tuple[str, ...]
    control_token_history: tuple[str, ...]
    replan_counts: dict[str, int]
    transitions: tuple[StateTransition, ...]
    verification: VerificationResult | None
    task_plan: dict[str, Any]
    events: tuple[EventRecord, ...]


class IndustrialAgent:
    """Fixed four-Agent supervisor with bounded, auditable recovery."""

    def __init__(
        self,
        executors: Sequence[Executor],
        *,
        gateway: ObservationGateway | None = None,
        safety: ActionSafetyValidator | None = None,
        verifier: PostconditionVerifier | None = None,
        events: EventSink | None = None,
        memory_store: MemoryStore | None = None,
        perception: PerceptionAgent | None = None,
        perception_evidence: DetectionEvidenceSink | None = None,
        perception_mode: PerceptionMode | str = PerceptionMode.SHADOW_SCORE,
        require_perception: bool | None = None,
        verification_frames: int = 3,
        executor_timeout_ms: int = 15_000,
        perception_timeout_ms: int = 5_000,
        max_perception_attempts: int = 1,
        perception_confidence_threshold: float = 0.25,
        perception_iou_threshold: float = 0.45,
        max_decisions_per_strategy_attempt: int = 8,
        task_profile: FixedTaskProfile | None = None,
        require_durable_handoff: bool = True,
        safe_stop_timeout_ms: int = 2_000,
    ):
        if verification_frames < 1 or verification_frames > 9:
            raise ValueError("verification_frames must be in [1, 9]")
        if not 1 <= max_decisions_per_strategy_attempt <= 100:
            raise ValueError("max_decisions_per_strategy_attempt must be in [1, 100]")
        if not 1 <= perception_timeout_ms <= 120_000:
            raise ValueError("perception_timeout_ms must be in [1, 120000]")
        if not 1 <= safe_stop_timeout_ms <= 30_000:
            raise ValueError("safe_stop_timeout_ms must be in [1, 30000]")
        try:
            normalized_perception_mode = PerceptionMode(perception_mode)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"unsupported perception_mode: {perception_mode!r}"
            ) from exc
        if max_perception_attempts != 1:
            raise ValueError(
                f"{normalized_perception_mode.value} uses exactly one "
                "failure-non-gating sidecar attempt"
            )
        for name, value in (
            ("perception_confidence_threshold", perception_confidence_threshold),
            ("perception_iou_threshold", perception_iou_threshold),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise ValueError(f"{name} must be a finite number in [0, 1]")
        if require_perception is not None and not isinstance(require_perception, bool):
            raise ValueError("require_perception must be boolean or None")
        requested_perception = True
        if require_perception is False:
            raise ValueError(
                "require_perception=False cannot weaken the fixed four-Agent topology"
            )
        if requested_perception and perception is None:
            raise ValueError(
                "the fixed four-Agent topology requires an injected YOLO "
                "PerceptionAgent"
            )
        profile = task_profile or FixedTaskProfile()
        profile.validate_frozen()
        executor_names = {item.descriptor.name for item in executors}
        required_names = {
            profile.primary_executor,
            profile.collaborative_executor,
        }
        if executor_names != required_names:
            raise ValueError(
                "fixed dual-VLA topology requires exactly both executors: "
                f"expected={sorted(required_names)}, "
                f"provided={sorted(executor_names)}"
            )
        if verification_frames != profile.handoff_verification_frames:
            raise ValueError(
                "fixed dual-VLA handoff requires exactly "
                f"{profile.handoff_verification_frames} verification frames"
            )
        if perception is not None and perception.descriptor.name != "yolo":
            raise ValueError("the independent perception Agent must be named 'yolo'")
        self.executors = ExecutorRegistry(executors)
        self.gateway = gateway or ObservationGateway()
        self.safety = safety or ActionSafetyValidator()
        self.verifier = verifier or PostconditionVerifier()
        self.events = events or EventSink()
        if require_durable_handoff is not True:
            raise ValueError(
                "fixed dual-VLA handoff cannot disable durable event persistence"
            )
        if not self.events.durable:
            raise ValueError("durable handoff requires an fsync-backed EventSink path")
        self.require_durable_handoff = True
        self.memory_store = memory_store or MemoryStore()
        self.topology_mode = "FIXED_DUAL_VLA_SERIAL"
        self.task_profile = profile
        self.planner = FixedDualVLAPlanner(profile)
        self.perception = perception
        self.perception_evidence = perception_evidence or DetectionEvidenceSink()
        self.perception_mode = normalized_perception_mode
        self.perception_required = requested_perception or perception is not None
        self.verification_frames = verification_frames
        self.executor_timeout_ms = executor_timeout_ms
        self.perception_timeout_ms = perception_timeout_ms
        self.max_perception_attempts = max_perception_attempts
        self.perception_confidence_threshold = float(perception_confidence_threshold)
        self.perception_iou_threshold = float(perception_iou_threshold)
        self.max_decisions_per_strategy_attempt = max_decisions_per_strategy_attempt
        self.safe_stop_timeout_ms = safe_stop_timeout_ms

        self._fsm = AgentFSM()
        self._queue: deque[ActionStep] = deque()
        self._run_id = ""
        self._task_id = ""
        self._memory: RunMemory | None = None
        self._plan: TaskPlan | None = None
        self._lifecycle: FixedLifecycle | None = None
        self._perception_disabled_for_run = False
        self._perception_quarantined = False
        self._quarantined_executors: set[str] = set()
        self._run_lock = Lock()

    @property
    def current_control_token(self) -> str:
        """Expose the authoritative lease to the controller boundary."""

        lifecycle = self._lifecycle
        return (
            lifecycle.token.value if lifecycle is not None else ControlToken.NONE.value
        )

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
        perception: PerceptionAgent | None = None,
        perception_evidence: DetectionEvidenceSink | None = None,
        require_perception: bool | None = None,
    ) -> IndustrialAgent:
        """Build the core from the versioned JSON-compatible configuration."""

        if not isinstance(config, Mapping):
            raise ValueError("agent config must be an object")
        version = config.get("config_version")
        if not isinstance(version, str) or version.split(".", 1)[0] != "1":
            raise ValueError(f"unsupported config_version: {version!r}")

        if "routing" in config:
            raise ValueError(
                "routing is obsolete in FIXED_DUAL_VLA_SERIAL mode; "
                "executor ownership comes from lifecycle.task_profile"
            )
        expected_top_level_keys = {
            "config_version",
            "verification_frames",
            "executor_timeout_ms",
            "image_cas",
            "lifecycle",
            "recovery",
            "safety",
            "telemetry",
            "perception",
            "executors",
        }
        if set(config) != expected_top_level_keys:
            raise ValueError(
                f"agent config must contain exactly {sorted(expected_top_level_keys)}"
            )
        ImageCasConfig.from_mapping(config.get("image_cas"))
        raw_lifecycle = config.get("lifecycle")
        if not isinstance(raw_lifecycle, Mapping):
            raise ValueError("lifecycle config must be an object")
        if raw_lifecycle.get("mode") != "FIXED_DUAL_VLA_SERIAL":
            raise ValueError("lifecycle.mode must be 'FIXED_DUAL_VLA_SERIAL'")
        if raw_lifecycle.get("supervisor_nlp") is not False:
            raise ValueError(
                "lifecycle.supervisor_nlp is frozen false; π0.5 handles language"
            )
        token_sequence = raw_lifecycle.get("token_sequence")
        expected_token_sequence = [token.value for token in FROZEN_TOKEN_SEQUENCE]
        if token_sequence != expected_token_sequence:
            raise ValueError(
                f"lifecycle.token_sequence must be {expected_token_sequence!r}"
            )
        profile = FixedTaskProfile.from_mapping(raw_lifecycle.get("task_profile", {}))

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

        def required_float(
            mapping: Mapping[str, Any],
            key: str,
            *,
            minimum: float,
            maximum: float,
        ) -> float:
            value = mapping.get(key)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not isfinite(float(value))
                or not minimum <= float(value) <= maximum
            ):
                raise ValueError(
                    f"{key} must be a finite number in [{minimum}, {maximum}]"
                )
            return float(value)

        recovery = config.get("recovery")
        if not isinstance(recovery, Mapping):
            raise ValueError("recovery config must be an object")
        frozen_recovery = {
            "max_replans_per_phase": 1,
            "max_switches_per_run": 0,
            "clear_action_queue_on_recovery": True,
        }
        allowed_recovery_keys = {
            *frozen_recovery,
            "max_decisions_per_strategy_attempt",
        }
        unexpected_recovery_keys = set(recovery) - allowed_recovery_keys
        if unexpected_recovery_keys:
            raise ValueError(
                "recovery invariants are frozen; obsolete fields are forbidden: "
                f"{sorted(unexpected_recovery_keys)}"
            )
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
        expected_safety_keys = {
            "action_contract_version",
            "axis_abs_limits",
            "workspace_by_arm",
            "max_chunk_steps",
            "safe_stop_timeout_ms",
        }
        if set(raw_safety) != expected_safety_keys:
            raise ValueError(
                f"safety config must contain exactly {sorted(expected_safety_keys)}"
            )

        def float_tuple(
            mapping: Mapping[str, Any],
            key: str,
            size: int,
            *,
            path: str = "safety",
        ) -> tuple[float, ...]:
            value = mapping.get(key)
            if (
                not isinstance(value, (list, tuple))
                or len(value) != size
                or any(
                    isinstance(item, bool) or not isinstance(item, (int, float))
                    for item in value
                )
            ):
                raise ValueError(f"{path}.{key} must contain {size} numbers")
            result = tuple(float(item) for item in value)
            if not all(isfinite(item) for item in result):
                raise ValueError(f"{path}.{key} values must be finite")
            return result

        axis_limits = float_tuple(raw_safety, "axis_abs_limits", 7)
        raw_workspace = raw_safety.get("workspace_by_arm")
        if not isinstance(raw_workspace, Mapping) or set(raw_workspace) != {
            "Arm_A",
            "Arm_B",
        }:
            raise ValueError("safety.workspace_by_arm must define Arm_A and Arm_B")

        workspace_bounds: dict[str, tuple[tuple[float, ...], tuple[float, ...]]] = {}
        for arm_id in ("Arm_A", "Arm_B"):
            raw_arm_workspace = raw_workspace.get(arm_id)
            if (
                not isinstance(raw_arm_workspace, Mapping)
                or set(raw_arm_workspace) != {"frame", "min_m", "max_m"}
                or raw_arm_workspace.get("frame") != "robot_base"
            ):
                raise ValueError(
                    f"safety.workspace_by_arm.{arm_id} must contain "
                    "frame='robot_base', min_m and max_m"
                )
            lower = float_tuple(
                raw_arm_workspace,
                "min_m",
                3,
                path=f"safety.workspace_by_arm.{arm_id}",
            )
            upper = float_tuple(
                raw_arm_workspace,
                "max_m",
                3,
                path=f"safety.workspace_by_arm.{arm_id}",
            )
            if any(low >= high for low, high in zip(lower, upper)):
                raise ValueError(
                    f"each {arm_id} workspace min_m value must be below max_m"
                )
            workspace_bounds[arm_id] = (lower, upper)
        if any(limit <= 0 for limit in axis_limits):
            raise ValueError("all safety.axis_abs_limits values must be positive")
        if axis_limits[6] > 1.0:
            raise ValueError("gripper axis limit cannot exceed normalized range 1.0")
        policy = SafetyPolicy(
            axis_abs_limits=axis_limits,  # type: ignore[arg-type]
            arm_a_workspace_min_m=workspace_bounds["Arm_A"][0],  # type: ignore[arg-type]
            arm_a_workspace_max_m=workspace_bounds["Arm_A"][1],  # type: ignore[arg-type]
            arm_b_workspace_min_m=workspace_bounds["Arm_B"][0],  # type: ignore[arg-type]
            arm_b_workspace_max_m=workspace_bounds["Arm_B"][1],  # type: ignore[arg-type]
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
        required_executor_names = {
            profile.primary_executor,
            profile.collaborative_executor,
        }
        if set(raw_executors) != required_executor_names:
            raise ValueError(
                "config.executors must declare exactly "
                f"{sorted(required_executor_names)}"
            )
        enabled_names: set[str] = set()
        for name, raw in raw_executors.items():
            if not isinstance(raw, Mapping):
                raise ValueError(f"config.executors.{name} must be an object")
            if set(raw) != EXECUTOR_CONFIG_FIELDS:
                raise ValueError(
                    f"config.executors.{name} must contain exactly "
                    f"{sorted(EXECUTOR_CONFIG_FIELDS)}"
                )
            base_url = raw.get("base_url")
            if not isinstance(base_url, str) or not base_url.startswith(
                ("http://", "https://")
            ):
                raise ValueError(
                    f"config.executors.{name}.base_url must be an HTTP(S) URL"
                )
            enabled = raw.get("enabled")
            if not isinstance(enabled, bool):
                raise ValueError(f"config.executors.{name}.enabled must be a boolean")
            if enabled:
                enabled_names.add(name)
        if enabled_names != required_executor_names:
            raise ValueError(
                "fixed dual-VLA topology requires both executors enabled: "
                f"expected={sorted(required_executor_names)}, "
                f"enabled={sorted(enabled_names)}"
            )
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

        raw_perception = config.get("perception")
        if not isinstance(raw_perception, Mapping):
            raise ValueError("config.perception must be an object")
        if raw_perception.get("required") is not True:
            raise ValueError(
                "config.perception.required is frozen true for the four-Agent topology"
            )
        try:
            perception_mode = PerceptionMode(raw_perception.get("mode"))
        except ValueError as exc:
            raise ValueError("config.perception.mode must be 'SHADOW_SCORE'") from exc
        evidence_jsonl_path = raw_perception.get("evidence_jsonl_path")
        if not isinstance(evidence_jsonl_path, str) or not evidence_jsonl_path.strip():
            raise ValueError(
                "config.perception.evidence_jsonl_path must be a non-empty path"
            )
        if require_perception is False:
            raise ValueError("require_perception=False cannot weaken the frozen config")
        if perception is None:
            raise ValueError(
                "the configured four-Agent topology requires an injected YOLO "
                "PerceptionAgent"
            )
        base_url = raw_perception.get("base_url")
        if not isinstance(base_url, str) or not base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError("config.perception.base_url must be an HTTP(S) URL")
        descriptor = perception.descriptor
        if descriptor.name != "yolo":
            raise ValueError("config.perception requires the Agent named 'yolo'")
        if set(raw_perception) != PERCEPTION_CONFIG_FIELDS:
            raise ValueError(
                "config.perception must contain exactly "
                f"{sorted(PERCEPTION_CONFIG_FIELDS)}"
            )
        expected_contract = raw_perception.get("detection_contract_version")
        if descriptor.detection_contract_version != expected_contract:
            raise ValueError(
                "perception detection_contract_version mismatch: "
                f"expected {expected_contract!r}, got "
                f"{descriptor.detection_contract_version!r}"
            )
        for field in ("checkpoint_sha", "class_map_sha", "config_sha"):
            expected_value = raw_perception.get(field)
            actual_value = getattr(descriptor, field)
            if expected_value == "REPLACE_WITH_PINNED_SHA":
                raise ValueError(
                    f"config.perception.{field} is still an unsafe placeholder"
                )
            if not is_pinned_artifact_digest(expected_value):
                raise ValueError(
                    f"config.perception.{field} must match "
                    "'sha256:<64 hexadecimal characters>'"
                )
            if actual_value != expected_value:
                raise ValueError(
                    f"perception {field} mismatch: "
                    f"expected {expected_value!r}, got {actual_value!r}"
                )

        raw_telemetry = config.get("telemetry")
        if not isinstance(raw_telemetry, Mapping) or set(raw_telemetry) != {
            "event_jsonl_path",
            "require_durable_handoff",
        }:
            raise ValueError(
                "config.telemetry must contain exactly event_jsonl_path and "
                "require_durable_handoff"
            )
        event_jsonl_path = raw_telemetry.get("event_jsonl_path")
        if not isinstance(event_jsonl_path, str) or not event_jsonl_path.strip():
            raise ValueError(
                "config.telemetry.event_jsonl_path must be a non-empty path"
            )
        if raw_telemetry.get("require_durable_handoff") is not True:
            raise ValueError(
                "config.telemetry.require_durable_handoff must remain true"
            )

        return cls(
            executors,
            gateway=gateway,
            safety=ActionSafetyValidator(policy),
            verifier=verifier,
            events=events or EventSink(event_jsonl_path),
            memory_store=memory_store,
            perception=perception,
            perception_evidence=(
                perception_evidence
                if perception_evidence is not None
                else DetectionEvidenceSink(evidence_jsonl_path)
            ),
            perception_mode=perception_mode,
            require_perception=True,
            verification_frames=required_int(
                config, "verification_frames", minimum=1, maximum=9
            ),
            executor_timeout_ms=required_int(
                config, "executor_timeout_ms", minimum=1, maximum=300_000
            ),
            perception_timeout_ms=required_int(
                raw_perception, "timeout_ms", minimum=1, maximum=120_000
            ),
            max_perception_attempts=required_int(
                raw_perception, "max_attempts", minimum=1, maximum=10
            ),
            perception_confidence_threshold=required_float(
                raw_perception,
                "confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            perception_iou_threshold=required_float(
                raw_perception,
                "iou_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
            max_decisions_per_strategy_attempt=required_int(
                recovery,
                "max_decisions_per_strategy_attempt",
                minimum=1,
                maximum=100,
            ),
            task_profile=profile,
            require_durable_handoff=True,
            safe_stop_timeout_ms=required_int(
                raw_safety,
                "safe_stop_timeout_ms",
                minimum=1,
                maximum=30_000,
            ),
        )

    def _emit(self, event_type: str, payload: dict[str, Any] | None = None) -> None:
        event_payload = dict(payload or {})
        if self._lifecycle is not None:
            event_payload.setdefault(
                "control_token",
                self._lifecycle.token.value,
            )
        self.events.emit(
            run_id=self._run_id,
            task_id=self._task_id,
            event_type=event_type,
            state=self._fsm.state,
            payload=event_payload,
        )

    def _record_control_token(
        self,
        previous: ControlToken | None,
        current: ControlToken,
        reason: str,
    ) -> None:
        assert self._memory is not None
        self._memory.control_token = current.value
        if (
            not self._memory.control_token_history
            or self._memory.control_token_history[-1] != current.value
        ):
            self._memory.control_token_history.append(current.value)
        self._emit(
            "control_token.changed",
            {
                "previous_token": previous.value if previous is not None else None,
                "next_token": current.value,
                "reason": reason,
            },
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
            control_token_history=tuple(self._memory.control_token_history),
            replan_counts=dict(self._memory.replan_counts),
            transitions=tuple(self._fsm.history),
            verification=verification,
            task_plan=self._plan.to_dict() if self._plan is not None else {},
            events=self.events.events_for_run(self._run_id),
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
        # A terminal run must never leave the controller lease at A_ONLY,
        # HANDOFF_VERIFY or B_ONLY, even when no physical action occurred and
        # therefore no hardware safe-stop is necessary.
        if (
            self._lifecycle is not None
            and self._lifecycle.token is not ControlToken.NONE
        ):
            previous, current = self._lifecycle.safe_stop()
            self._memory.control_token = current.value
            if (
                not self._memory.control_token_history
                or self._memory.control_token_history[-1] != current.value
            ):
                self._memory.control_token_history.append(current.value)
            self._emit(
                "control_token.changed",
                {
                    "previous_token": previous.value,
                    "next_token": current.value,
                    "reason": f"terminal failure without motion: {message}",
                },
            )
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
        telemetry_errors: list[str] = []
        if self._lifecycle is not None:
            previous, current = self._lifecycle.safe_stop()
            self._memory.control_token = current.value
            if (
                not self._memory.control_token_history
                or self._memory.control_token_history[-1] != current.value
            ):
                self._memory.control_token_history.append(current.value)
            try:
                self._emit(
                    "control_token.changed",
                    {
                        "previous_token": previous.value,
                        "next_token": current.value,
                        "reason": message,
                    },
                )
            except Exception as exc:
                telemetry_errors.append(f"control_token.changed: {exc}")
        removed_steps = len(self._queue)
        self._queue.clear()
        try:
            self._emit(
                "action_queue.cleared",
                {
                    "reason": "immediate_safe_stop",
                    "removed_steps": removed_steps,
                },
            )
        except Exception as exc:
            telemetry_errors.append(f"action_queue.cleared: {exc}")
        stop_error: str | None = None
        receipt_payload: dict[str, Any] | None = None
        stopped, receipt_value, stop_call_error = _invoke_with_hard_deadline(
            lambda: environment.safe_stop(message),
            self.safe_stop_timeout_ms,
        )
        if not stopped:
            stop_error = str(stop_call_error)
        elif not isinstance(receipt_value, SafeStopReceipt):
            stop_error = "safe_stop must return a SafeStopReceipt"
        elif not receipt_value.confirmed:
            stop_error = "controller returned an unconfirmed safe-stop receipt"
            receipt_payload = {
                "controller_ack": receipt_value.controller_ack,
                "buffers_cleared": receipt_value.buffers_cleared,
                "arm_a_stopped": receipt_value.arm_a_stopped,
                "arm_b_stopped": receipt_value.arm_b_stopped,
                "stop_epoch": receipt_value.stop_epoch,
            }
        else:
            receipt_payload = {
                "controller_ack": receipt_value.controller_ack,
                "buffers_cleared": receipt_value.buffers_cleared,
                "arm_a_stopped": receipt_value.arm_a_stopped,
                "arm_b_stopped": receipt_value.arm_b_stopped,
                "stop_epoch": receipt_value.stop_epoch,
            }
            observed, raw_stopped_state, observe_error = _invoke_with_hard_deadline(
                environment.observe,
                self.safe_stop_timeout_ms,
            )
            if not observed:
                stop_error = f"post-stop sensor confirmation failed: {observe_error}"
            else:
                try:
                    assert raw_stopped_state is not None
                    stopped_observation = self.gateway.ingest_online(raw_stopped_state)
                    robot = stopped_observation.data.get("robot")
                    if not isinstance(robot, Mapping):
                        raise ValueError("post-stop robot state is missing")
                    arm_a = robot.get("arm_a")
                    arm_b = robot.get("arm_b")
                    if (
                        robot.get("active_arm") != "NONE"
                        or not isinstance(arm_a, Mapping)
                        or arm_a.get("stationary") is not True
                        or not isinstance(arm_b, Mapping)
                        or arm_b.get("stationary") is not True
                    ):
                        raise ValueError(
                            "post-stop sensors do not confirm both arms stationary"
                        )
                except Exception as exc:
                    stop_error = f"post-stop sensor confirmation rejected: {exc}"
        target_state = (
            AgentState.SAFE_STOP_FAILED
            if stop_error is not None
            else AgentState.SAFE_STOPPED
        )
        if self._fsm.state is not target_state:
            transition = self._fsm.force_safety_terminal(target_state, message)
            try:
                self._emit(
                    "fsm.transition",
                    {
                        "from": transition.previous.value,
                        "to": transition.current.value,
                        "reason": message,
                    },
                )
            except Exception as exc:
                telemetry_errors.append(f"fsm.transition: {exc}")
        terminal_event = (
            "run.safe_stop_failed"
            if target_state is AgentState.SAFE_STOP_FAILED
            else "run.safe_stopped"
        )
        try:
            self._emit(
                terminal_event,
                {
                    "failure_code": code.value,
                    "message": message,
                    "safe_stop_error": stop_error,
                    "stop_confirmed": stop_error is None,
                    "stop_receipt": receipt_payload,
                    "telemetry_errors": list(telemetry_errors),
                },
            )
        except Exception as exc:
            telemetry_errors.append(f"{terminal_event}: {exc}")
        if stop_error is not None:
            self._memory.last_failure_code = FailureCode.SYSTEM_FAULT.value
            return self._result(
                FailureCode.SYSTEM_FAULT,
                f"{message}; physical safe-stop confirmation failed: {stop_error}",
                verification,
                event_start,
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
        if fault is not None:
            code, reason = fault
            return self._safe_stop(
                environment,
                code,
                reason,
                verification,
                event_start,
            )
        if self._lifecycle is not None:
            consistency_failure = self._fixed_observation_consistency_failure(
                observation
            )
            if consistency_failure is not None:
                return self._safe_stop(
                    environment,
                    FailureCode.SAFETY_REJECTED,
                    f"dual-arm observation interlock failed: {consistency_failure}",
                    verification,
                    event_start,
                )
            token_failure = self._fixed_token_interlock_failure(observation)
            if token_failure is not None:
                return self._safe_stop(
                    environment,
                    FailureCode.SAFETY_REJECTED,
                    f"control-token interlock failed: {token_failure}",
                    verification,
                    event_start,
                )
        return None

    @staticmethod
    def _fixed_observation_consistency_failure(
        observation: Observation,
    ) -> str | None:
        """Reject contradictory sensor summaries before they can vote."""

        camera = observation.data.get("camera")
        robot = observation.data.get("robot")
        task_state = observation.data.get("task", {})
        quality = observation.data.get("quality")
        if not isinstance(camera, Mapping):
            return "camera state is missing or invalid"
        allowed_camera_keys = {
            "full_image",
            "arm_a_rgb",
            "handoff_rgb",
            "arm_b_rgb",
            "wrist_image",
        }
        unknown_camera_keys = set(camera) - allowed_camera_keys
        if unknown_camera_keys:
            return f"camera contains unknown fields: {sorted(unknown_camera_keys)}"
        required_phase_cameras = {
            "full_image",
            "arm_a_rgb",
            "handoff_rgb",
            "arm_b_rgb",
        }
        missing_phase_cameras = required_phase_cameras - set(camera)
        if missing_phase_cameras:
            return (
                "camera is missing frozen phase streams: "
                f"{sorted(missing_phase_cameras)}"
            )
        expected_camera_ids = {
            "arm_a_rgb": "CAM_A_TOP",
            "handoff_rgb": "CAM_HANDOFF",
            "arm_b_rgb": "CAM_B_TOP",
        }
        for image_key in allowed_camera_keys:
            if image_key not in camera or camera[image_key] is None:
                continue
            raw_image = camera[image_key]
            required_image_keys = {
                "uri",
                "image_sha256",
                "camera_id",
                "width",
                "height",
            }
            if (
                not isinstance(raw_image, Mapping)
                or set(raw_image) != required_image_keys
            ):
                return (
                    f"camera.{image_key} must contain exactly "
                    f"{sorted(required_image_keys)}"
                )
            try:
                image_reference = ImageReference(
                    uri=raw_image["uri"],
                    image_sha256=raw_image["image_sha256"],
                    camera_id=raw_image["camera_id"],
                    width=raw_image["width"],
                    height=raw_image["height"],
                )
            except (AgentError, TypeError, ValueError) as exc:
                return f"camera.{image_key} is invalid: {exc}"
            expected_camera_id = expected_camera_ids.get(image_key)
            if (
                expected_camera_id is not None
                and image_reference.camera_id != expected_camera_id
            ):
                return (
                    f"camera.{image_key}.camera_id must be "
                    f"{expected_camera_id!r}, got {image_reference.camera_id!r}"
                )
            if image_key == "full_image" and image_reference.camera_id not in {
                "CAM_A_TOP",
                "CAM_HANDOFF",
                "CAM_B_TOP",
            }:
                return "camera.full_image must reference a frozen RGB camera"
            if image_key in required_phase_cameras and (
                image_reference.width,
                image_reference.height,
            ) != (FROZEN_IMAGE_WIDTH, FROZEN_IMAGE_HEIGHT):
                return (
                    f"camera.{image_key} must use frozen "
                    f"{FROZEN_IMAGE_WIDTH}x{FROZEN_IMAGE_HEIGHT} resolution"
                )
        if not isinstance(robot, Mapping):
            return "robot state is missing or invalid"
        if set(robot) != {"active_arm", "arm_a", "arm_b"}:
            return "robot must contain exactly active_arm, arm_a and arm_b"
        if not isinstance(task_state, Mapping):
            return "task sensor summary is invalid"
        allowed_task_keys = {
            "packed_part_count",
            "bin_at_handoff",
            "arm_a_retreated",
            "arm_b_retreated",
            "bin_at_finished",
            "bin_speed_m_s",
            "status",
        }
        unknown_task_keys = set(task_state) - allowed_task_keys
        if unknown_task_keys:
            return f"task contains unknown fields: {sorted(unknown_task_keys)}"
        packed_part_count = task_state.get("packed_part_count")
        if (
            isinstance(packed_part_count, bool)
            or not isinstance(packed_part_count, int)
            or not 0 <= packed_part_count <= 6
        ):
            return "task.packed_part_count must be an integer in [0, 6]"
        for field_name in ("bin_at_handoff", "bin_at_finished"):
            if not isinstance(task_state.get(field_name), bool):
                return f"task.{field_name} must be boolean"
        if task_state["bin_at_handoff"] and task_state["bin_at_finished"]:
            return "bin_at_handoff and bin_at_finished cannot both be true"
        bin_speed_m_s = task_state.get("bin_speed_m_s")
        if (
            isinstance(bin_speed_m_s, bool)
            or not isinstance(bin_speed_m_s, (int, float))
            or not isfinite(float(bin_speed_m_s))
            or float(bin_speed_m_s) < 0.0
        ):
            return "task.bin_speed_m_s must be a finite non-negative number"
        if not isinstance(quality, Mapping):
            return "quality state is missing or invalid"
        quality_confidence = quality.get("confidence")
        if (
            isinstance(quality_confidence, bool)
            or not isinstance(quality_confidence, (int, float))
            or not isfinite(float(quality_confidence))
            or not 0.0 <= float(quality_confidence) <= 1.0
        ):
            return "quality.confidence must be a finite number in [0, 1]"
        active_arm = robot.get("active_arm")
        if active_arm not in {"Arm_A", "Arm_B", "NONE"}:
            return f"robot.active_arm is invalid: {active_arm!r}"
        for arm_key, summary_key in (
            ("arm_a", "arm_a_retreated"),
            ("arm_b", "arm_b_retreated"),
        ):
            arm_state = robot.get(arm_key)
            if not isinstance(arm_state, Mapping):
                return f"robot.{arm_key} state is missing"
            if set(arm_state) != {
                "tcp_pose_m_rad",
                "state",
                "retreated",
                "gripper_open",
                "stationary",
            }:
                return (
                    f"robot.{arm_key} must contain exactly "
                    "tcp_pose_m_rad, state, retreated, gripper_open and stationary"
                )
            tcp_pose = arm_state.get("tcp_pose_m_rad")
            state = arm_state.get("state")
            for field_name, values, expected_length in (
                ("tcp_pose_m_rad", tcp_pose, 6),
                ("state", state, 7),
            ):
                if (
                    not isinstance(values, (list, tuple))
                    or len(values) != expected_length
                    or any(
                        isinstance(item, bool)
                        or not isinstance(item, (int, float))
                        or not isfinite(float(item))
                        for item in values
                    )
                ):
                    return (
                        f"robot.{arm_key}.{field_name} must contain exactly "
                        f"{expected_length} finite numbers"
                    )
            retreated = arm_state.get("retreated")
            if not isinstance(retreated, bool):
                return f"robot.{arm_key}.retreated must be boolean"
            gripper_open = arm_state.get("gripper_open")
            if not isinstance(gripper_open, bool):
                return f"robot.{arm_key}.gripper_open must be boolean"
            try:
                expected_state = canonical_state_7d(tcp_pose, gripper_open)
            except (TypeError, ValueError) as exc:
                return f"robot.{arm_key} cannot produce canonical state_7d: {exc}"
            if any(
                abs(float(actual) - expected) > 1e-9
                for actual, expected in zip(state, expected_state)
            ):
                return (
                    f"robot.{arm_key}.state must equal tcp_pose_m_rad plus "
                    "controller-confirmed gripper state"
                )
            stationary = arm_state.get("stationary")
            if not isinstance(stationary, bool):
                return f"robot.{arm_key}.stationary must be boolean"
            summary = task_state.get(summary_key)
            if summary is not None and not isinstance(summary, bool):
                return f"task.{summary_key} must be boolean when present"
            if summary is not None and summary is not retreated:
                return (
                    f"task.{summary_key}={summary!r} conflicts with "
                    f"robot.{arm_key}.retreated={retreated!r}"
                )
        return None

    @staticmethod
    def _fixed_arm_interlock_failure(
        observation: Observation,
        *,
        arm_id: str,
    ) -> str | None:
        """Cross-check active-arm and opposite-arm retreat sensor state."""

        robot = observation.data.get("robot")
        if not isinstance(robot, Mapping):
            return "robot state is missing or invalid"
        active_arm = robot.get("active_arm")
        if active_arm not in {arm_id, "NONE"}:
            return f"robot.active_arm={active_arm!r} conflicts with {arm_id}"

        current_key = "arm_a" if arm_id == "Arm_A" else "arm_b"
        opposite_key = "arm_b" if arm_id == "Arm_A" else "arm_a"
        current_state = robot.get(current_key)
        opposite_state = robot.get(opposite_key)
        if not isinstance(current_state, Mapping):
            return f"robot.{current_key} state is missing"
        if not isinstance(opposite_state, Mapping):
            return f"robot.{opposite_key} state is missing"
        if opposite_state.get("retreated") is not True:
            return f"robot.{opposite_key}.retreated must be true before {arm_id} acts"
        if opposite_state.get("stationary") is not True:
            return f"robot.{opposite_key}.stationary must be true before {arm_id} acts"
        return None

    @staticmethod
    def _preexecution_state_change_failure(
        planning_observation: Observation,
        execution_observation: Observation,
        *,
        arm_id: str,
    ) -> str | None:
        """Invalidate a chunk if critical state changed during model latency."""

        if planning_observation.data.get("safety") != execution_observation.data.get(
            "safety"
        ):
            return "safety state changed after inference started"
        planning_robot = planning_observation.data.get("robot")
        execution_robot = execution_observation.data.get("robot")
        if not isinstance(planning_robot, Mapping) or not isinstance(
            execution_robot, Mapping
        ):
            return "robot state is missing"
        if planning_robot.get("active_arm") != execution_robot.get("active_arm"):
            return "robot.active_arm changed during inference"
        for arm_key in ("arm_a", "arm_b"):
            planning_arm = planning_robot.get(arm_key)
            execution_arm = execution_robot.get(arm_key)
            if not isinstance(planning_arm, Mapping) or not isinstance(
                execution_arm, Mapping
            ):
                return f"robot.{arm_key} state is missing"
            for field_name in ("retreated", "gripper_open", "stationary"):
                if planning_arm.get(field_name) != execution_arm.get(field_name):
                    return f"robot.{arm_key}.{field_name} changed during inference"
            for field_name, tolerance in (
                ("tcp_pose_m_rad", 1e-4),
                ("state", 1e-3),
            ):
                before = planning_arm.get(field_name)
                after = execution_arm.get(field_name)
                if not isinstance(before, (list, tuple)) or not isinstance(
                    after, (list, tuple)
                ):
                    return f"robot.{arm_key}.{field_name} is missing"
                if len(before) != len(after) or any(
                    abs(float(left) - float(right)) > tolerance
                    for left, right in zip(before, after)
                ):
                    return f"robot.{arm_key}.{field_name} changed during inference"
        return None

    @staticmethod
    def _preexecution_semantic_change_reason(
        planning_observation: Observation,
        execution_observation: Observation,
    ) -> str | None:
        """Detect meaningful scene drift without reacting to confidence noise."""

        before_task = planning_observation.data.get("task")
        after_task = execution_observation.data.get("task")
        if not isinstance(before_task, Mapping) or not isinstance(after_task, Mapping):
            return "task state is missing"
        for field_name in (
            "packed_part_count",
            "bin_at_handoff",
            "bin_at_finished",
            "status",
        ):
            if before_task.get(field_name) != after_task.get(field_name):
                return f"task.{field_name} changed during inference"
        try:
            if (
                abs(
                    float(before_task.get("bin_speed_m_s", 0.0))
                    - float(after_task.get("bin_speed_m_s", 0.0))
                )
                > 0.005
            ):
                return "task.bin_speed_m_s changed beyond tolerance"
        except (TypeError, ValueError):
            return "task.bin_speed_m_s is invalid"

        def discrete_objects(observation: Observation) -> tuple[tuple[Any, Any], ...]:
            objects = observation.data.get("objects")
            if not isinstance(objects, (list, tuple)):
                return ()
            return tuple(
                sorted(
                    (
                        item.get("object_id"),
                        item.get("zone_id"),
                    )
                    for item in objects
                    if isinstance(item, Mapping)
                )
            )

        if discrete_objects(planning_observation) != discrete_objects(
            execution_observation
        ):
            return "object identity or zone changed during inference"
        return None

    def _fixed_token_interlock_failure(
        self,
        observation: Observation,
    ) -> str | None:
        """Bind every observed arm state to the currently held control token."""

        assert self._lifecycle is not None
        robot = observation.data.get("robot")
        if not isinstance(robot, Mapping):
            return "robot state is missing or invalid"
        arm_a = robot.get("arm_a")
        arm_b = robot.get("arm_b")
        if not isinstance(arm_a, Mapping) or not isinstance(arm_b, Mapping):
            return "both arm states are required"
        active_arm = robot.get("active_arm")
        arm_a_retreated = arm_a.get("retreated")
        arm_b_retreated = arm_b.get("retreated")
        arm_a_stationary = arm_a.get("stationary")
        arm_b_stationary = arm_b.get("stationary")
        token = self._lifecycle.token

        if token is ControlToken.A_ONLY:
            if active_arm not in {self.task_profile.arm_a_id, "NONE"}:
                return f"A_ONLY forbids robot.active_arm={active_arm!r}"
            if arm_b_retreated is not True:
                return "A_ONLY requires Arm_B retreated"
        elif token is ControlToken.HANDOFF_VERIFY:
            if active_arm != "NONE":
                return "HANDOFF_VERIFY requires robot.active_arm='NONE'"
            if arm_a_retreated is not True or arm_b_retreated is not True:
                return "HANDOFF_VERIFY requires both arms retreated"
            if arm_a_stationary is not True or arm_b_stationary is not True:
                return "HANDOFF_VERIFY requires both arms stationary"
        elif token is ControlToken.B_ONLY:
            if active_arm not in {self.task_profile.arm_b_id, "NONE"}:
                return f"B_ONLY forbids robot.active_arm={active_arm!r}"
            if arm_a_retreated is not True:
                return "B_ONLY requires Arm_A retreated"
        elif token is ControlToken.NONE:
            if active_arm != "NONE":
                return "NONE requires robot.active_arm='NONE'"
            if arm_a_stationary is not True or arm_b_stationary is not True:
                return "NONE requires both arms stationary"
        return None

    @staticmethod
    def _image_reference(
        observation: Observation,
        *,
        camera_key: str = "full_image",
    ) -> ImageReference:
        """Resolve the phase image shared by the current VLA and YOLO."""

        camera = observation.data.get("camera")
        if not isinstance(camera, Mapping):
            raise PerceptionError(
                FailureCode.OBSERVATION_INVALID,
                "camera observation must be an object for YOLO perception",
                retryable=True,
            )
        if camera_key not in camera:
            raise PerceptionError(
                FailureCode.OBSERVATION_INVALID,
                f"camera.{camera_key} is required; phase perception has no fallback",
                retryable=False,
            )
        raw_image = camera[camera_key]
        if not isinstance(raw_image, Mapping):
            raise PerceptionError(
                FailureCode.OBSERVATION_INVALID,
                f"camera.{camera_key} must be an immutable image reference",
                retryable=False,
            )
        expected_camera_ids = {
            "arm_a_rgb": "CAM_A_TOP",
            "handoff_rgb": "CAM_HANDOFF",
            "arm_b_rgb": "CAM_B_TOP",
        }
        try:
            image = ImageReference(
                uri=raw_image.get("uri"),
                image_sha256=raw_image.get("image_sha256"),
                camera_id=raw_image.get("camera_id"),
                width=raw_image.get("width"),
                height=raw_image.get("height"),
            )
        except (TypeError, ValueError, AgentError) as exc:
            raise PerceptionError(
                FailureCode.OBSERVATION_INVALID,
                f"full-image identity is invalid for YOLO perception: {exc}",
                retryable=True,
            ) from exc
        expected_camera_id = expected_camera_ids.get(camera_key)
        if expected_camera_id is not None and image.camera_id != expected_camera_id:
            raise PerceptionError(
                FailureCode.OBSERVATION_INVALID,
                f"camera.{camera_key}.camera_id must be {expected_camera_id!r}",
                retryable=False,
            )
        if camera_key in {"full_image", "arm_a_rgb", "handoff_rgb", "arm_b_rgb"} and (
            image.width,
            image.height,
        ) != (FROZEN_IMAGE_WIDTH, FROZEN_IMAGE_HEIGHT):
            raise PerceptionError(
                FailureCode.OBSERVATION_INVALID,
                f"camera.{camera_key} must use frozen "
                f"{FROZEN_IMAGE_WIDTH}x{FROZEN_IMAGE_HEIGHT} resolution",
                retryable=False,
            )
        return image

    @staticmethod
    def _verification_frame_identity_failure(
        frames: Sequence[Observation],
        *,
        camera_key: str,
    ) -> str | None:
        """Require genuinely distinct camera frames for a voting quorum."""

        timestamps = [frame.timestamp_ms for frame in frames]
        if any(
            current <= previous for previous, current in zip(timestamps, timestamps[1:])
        ):
            return "verification timestamps must be strictly increasing"
        image_hashes: list[str] = []
        for frame in frames:
            camera = frame.data.get("camera")
            if not isinstance(camera, Mapping):
                return "camera observation is missing"
            raw_image = camera.get(camera_key)
            if not isinstance(raw_image, Mapping):
                return f"camera.{camera_key} must be an image reference"
            expected_camera_id = {
                "handoff_rgb": "CAM_HANDOFF",
                "arm_b_rgb": "CAM_B_TOP",
            }.get(camera_key)
            if raw_image.get("camera_id") != expected_camera_id:
                return f"camera.{camera_key}.camera_id must be {expected_camera_id!r}"
            image_sha = raw_image.get("image_sha256")
            if not isinstance(image_sha, str) or not image_sha:
                return f"camera.{camera_key} image SHA is missing"
            image_hashes.append(image_sha)
        if len(image_hashes) != len(set(image_hashes)):
            return "verification image SHA values must be unique"
        return None

    def _perceive_for_vla(
        self,
        *,
        task: TaskSchema,
        observation: Observation,
        environment: ExecutionEnvironment,
        verification: VerificationResult | None,
        event_start: int,
        action_has_executed: bool,
        step_id: int,
    ) -> tuple[Observation, DetectionPacket | None, RunResult | None]:
        """Sample YOLO once for scoring; never gate or mutate VLA recovery."""

        del environment, verification, event_start, action_has_executed
        if self.perception is None:
            return observation, None, None
        if self._perception_disabled_for_run:
            self._emit(
                "perception.skipped",
                {
                    "agent": self.perception.descriptor.name,
                    "reason": "sidecar disabled after failed health/deadline",
                    "control_path_impact": "none",
                },
            )
            return observation, None, None
        self._transition(
            AgentState.PERCEIVING,
            "sample same-frame YOLO scoring sidecar",
        )
        descriptor = self.perception.descriptor
        subtask_id = str(task.metadata.get("subtask_id", task.task_id))
        image: ImageReference | None = None
        try:
            phase_camera_key = {
                ARM_A_PACK_HANDOFF_SUBTASK_ID: "arm_a_rgb",
                ARM_B_TRANSPORT_SUBTASK_ID: "arm_b_rgb",
            }.get(subtask_id, "full_image")
            image = self._image_reference(
                observation,
                camera_key=phase_camera_key,
            )
            context = PerceptionContext(
                run_id=self._run_id,
                task_id=task.task_id,
                subtask_id=subtask_id,
                step_id=step_id,
                observation_id=observation.observation_id,
                image=image,
                timeout_ms=self.perception_timeout_ms,
                # Empty means run the complete frozen class map for mAP.
                allowed_class_names=(),
                confidence_threshold=self.perception_confidence_threshold,
                iou_threshold=self.perception_iou_threshold,
            )
            self._emit(
                "perception.requested",
                {
                    "agent": descriptor.name,
                    "perception_mode": self.perception_mode.value,
                    "subtask_id": subtask_id,
                    "step_id": step_id,
                    "attempt": 1,
                    "attempt_budget": 1,
                    "observation_id": observation.observation_id,
                    "image_sha256": image.image_sha256,
                    "camera_id": image.camera_id,
                    "phase_camera_key": phase_camera_key,
                    "image_width": image.width,
                    "image_height": image.height,
                    "checkpoint_sha": descriptor.checkpoint_sha,
                    "class_map_sha": descriptor.class_map_sha,
                    "config_sha": descriptor.config_sha,
                    "confidence_threshold": self.perception_confidence_threshold,
                    "iou_threshold": self.perception_iou_threshold,
                    "control_path_impact": "none",
                },
            )
            completed, packet_value, detector_error = _invoke_with_hard_deadline(
                lambda: self.perception.detect(context),
                self.perception_timeout_ms,
            )
            if not completed:
                if isinstance(detector_error, TimeoutError):
                    self._perception_disabled_for_run = True
                    self._perception_quarantined = True
                    Thread(
                        target=lambda: self.perception.cancel(
                            context.task_id,
                            "supervisor hard deadline exceeded",
                        ),
                        daemon=True,
                    ).start()
                    raise PerceptionError(
                        FailureCode.PERCEPTION_TIMEOUT,
                        str(detector_error),
                        retryable=False,
                    )
                if isinstance(detector_error, PerceptionError):
                    raise detector_error
                raise PerceptionError(
                    FailureCode.PERCEPTION_UNAVAILABLE,
                    f"YOLO adapter failed: {detector_error}",
                    retryable=False,
                )
            packet = packet_value
            if not isinstance(packet, DetectionPacket):
                raise PerceptionError(
                    FailureCode.PERCEPTION_BAD_RESPONSE,
                    "YOLO Agent must return a DetectionPacket",
                )
            revision_mismatches = {
                key: {
                    "expected": getattr(descriptor, key),
                    "actual": getattr(packet, key),
                }
                for key in (
                    "checkpoint_sha",
                    "class_map_sha",
                    "config_sha",
                    "detection_contract_version",
                )
                if getattr(packet, key) != getattr(descriptor, key)
            }
            if revision_mismatches:
                raise PerceptionError(
                    FailureCode.PERCEPTION_REVISION_MISMATCH,
                    f"YOLO packet deployment identity mismatch: {revision_mismatches}",
                )
            try:
                packet.validate_against(
                    observation_id=observation.observation_id,
                    image=image,
                    descriptor=descriptor,
                )
            except (AgentError, TypeError, ValueError) as exc:
                raise PerceptionError(
                    FailureCode.PERCEPTION_BAD_RESPONSE,
                    f"YOLO packet frame/contract validation failed: {exc}",
                ) from exc
            expected_correlation = {
                "trace_id": self._run_id,
                "episode_id": self._run_id,
                "task_id": task.task_id,
                "subtask_id": subtask_id,
                "step_id": step_id,
            }
            mismatches = {
                key: {
                    "expected": expected,
                    "actual": getattr(packet, key),
                }
                for key, expected in expected_correlation.items()
                if getattr(packet, key) != expected
            }
            if mismatches:
                raise PerceptionError(
                    FailureCode.PERCEPTION_BAD_RESPONSE,
                    f"YOLO packet correlation mismatch: {mismatches}",
                )
            packet_data = packet.to_dict()
            try:
                self.perception_evidence.record_packet(
                    packet,
                    mode=self.perception_mode,
                )
            except Exception as exc:
                self._emit(
                    "perception.evidence_write_failed",
                    {
                        "packet_id": packet.packet_id,
                        "observation_id": packet.observation_id,
                        "image_sha256": packet.image_sha256,
                        "message": str(exc),
                        "control_path_impact": "none",
                    },
                )
            self._emit(
                "perception.completed",
                {
                    "agent": descriptor.name,
                    "perception_mode": self.perception_mode.value,
                    "packet_id": packet.packet_id,
                    "trace_id": packet.trace_id,
                    "subtask_id": subtask_id,
                    "step_id": step_id,
                    "attempt": 1,
                    "observation_id": packet.observation_id,
                    "image_sha256": packet.image_sha256,
                    "camera_id": packet.camera_id,
                    "checkpoint_sha": packet.checkpoint_sha,
                    "class_map_sha": packet.class_map_sha,
                    "config_sha": packet.config_sha,
                    "detection_count": len(packet.detections),
                    "detections": packet_data["detections"],
                    "timing": packet_data["timing"],
                    "raw_packet": packet_data,
                    "same_frame_verified": True,
                    "empty_detection_is_valid": not packet.detections,
                    "control_path_impact": "none",
                },
            )
            return observation, packet, None
        except AgentError as exc:
            code = exc.code
            message = str(exc)
        except Exception as exc:
            code = FailureCode.PERCEPTION_UNAVAILABLE
            message = f"unexpected YOLO Agent error: {exc}"

        try:
            self.perception_evidence.record_failure(
                trace_id=self._run_id,
                task_id=task.task_id,
                subtask_id=subtask_id,
                step_id=step_id,
                observation_id=observation.observation_id,
                image=image,
                descriptor=descriptor,
                failure_code=code,
                message=message,
                mode=self.perception_mode,
            )
        except Exception as exc:
            self._emit(
                "perception.evidence_write_failed",
                {
                    "observation_id": observation.observation_id,
                    "image_sha256": (image.image_sha256 if image is not None else None),
                    "message": str(exc),
                    "control_path_impact": "none",
                },
            )
        self._emit(
            "perception.failed",
            {
                "agent": descriptor.name,
                "perception_mode": self.perception_mode.value,
                "subtask_id": subtask_id,
                "step_id": step_id,
                "attempt": 1,
                "attempt_budget": 1,
                "observation_id": observation.observation_id,
                "image_sha256": image.image_sha256 if image is not None else None,
                "checkpoint_sha": descriptor.checkpoint_sha,
                "class_map_sha": descriptor.class_map_sha,
                "config_sha": descriptor.config_sha,
                "failure_code": code.value,
                "message": message,
                "will_retry": False,
                "vla_recovery_untouched": True,
                "control_path_impact": "none",
            },
        )
        return observation, None, None

    def _recover(
        self,
        *,
        task: TaskSchema,
        current: Executor,
        code: FailureCode,
        reason: str,
        local_replans: dict[str, int],
    ) -> tuple[Executor | None, bool]:
        """Retry the same lifecycle-assigned VLA once, then stop the phase."""

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
            cancelled, _, cancel_error = _invoke_with_hard_deadline(
                lambda: current.cancel(task.task_id, reason),
                min(self.executor_timeout_ms, 1_000),
            )
            if not cancelled:
                raise AgentError(
                    FailureCode.EXECUTOR_CANCELLED,
                    f"{name} cancellation failed or timed out: {cancel_error}",
                )
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
        self._emit(
            "recovery.phase_exhausted",
            {
                "executor": name,
                "trigger_code": code.value,
                "switch_allowed": False,
                "reason": "fixed lifecycle never substitutes one VLA for the other",
            },
        )
        return current, False

    def _observe_with_deadline(
        self, environment: ExecutionEnvironment
    ) -> Mapping[str, Any]:
        completed, value, error = _invoke_with_hard_deadline(
            environment.observe,
            self.executor_timeout_ms,
        )
        if not completed:
            raise AgentError(
                FailureCode.SYSTEM_FAULT,
                "environment observation deadline exceeded or failed: "
                f"{error}; controller state is unknown",
            )
        if not isinstance(value, Mapping):
            raise AgentError(
                FailureCode.OBSERVATION_INVALID,
                "environment.observe must return an observation object",
            )
        return value

    def _step_with_deadline(
        self,
        environment: ExecutionEnvironment,
        action: ActionStep,
        *,
        arm_id: str,
        control_token: str,
        command_id: str,
        expected_observation_id: str,
        expected_state_digest: str,
    ) -> Mapping[str, Any]:
        completed, value, error = _invoke_with_hard_deadline(
            lambda: environment.step(
                action,
                arm_id=arm_id,
                control_token=control_token,
                command_id=command_id,
                expected_observation_id=expected_observation_id,
                expected_state_digest=expected_state_digest,
            ),
            self.executor_timeout_ms,
        )
        if not completed:
            if isinstance(error, PreWriteStateStaleError):
                raise error
            raise AgentError(
                FailureCode.SYSTEM_FAULT,
                "controller command acknowledgement deadline exceeded or failed: "
                f"{error}; execution outcome is unknown",
            )
        if not isinstance(value, Mapping):
            raise AgentError(
                FailureCode.OBSERVATION_INVALID,
                "environment.step must return an observation object",
            )
        return value

    def run(self, task: TaskSchema, environment: ExecutionEnvironment) -> RunResult:
        """Run one task; reject overlapping calls on this stateful instance."""

        if not self._run_lock.acquire(blocking=False):
            raise AgentError(
                FailureCode.AGENT_BUSY,
                "IndustrialAgent already has an active workcell control run",
            )
        self._memory = None
        event_start = len(self.events.events)
        try:
            try:
                return self._run_once(task, environment)
            except BaseException as exc:
                if not isinstance(exc, Exception):
                    interrupt_reason = f"supervisor interrupted by {type(exc).__name__}"
                    if self._memory is not None:
                        self._safe_stop(
                            environment,
                            FailureCode.SYSTEM_FAULT,
                            interrupt_reason,
                            verification=None,
                            event_start=event_start,
                        )
                    else:
                        _invoke_with_hard_deadline(
                            lambda: environment.safe_stop(interrupt_reason),
                            self.safe_stop_timeout_ms,
                        )
                    raise
                if self._memory is None:
                    initialization_error = str(exc)
                    stopped, receipt, stop_error = _invoke_with_hard_deadline(
                        lambda: environment.safe_stop(
                            "unhandled supervisor failure before run memory: "
                            f"{initialization_error}"
                        ),
                        self.safe_stop_timeout_ms,
                    )
                    stop_confirmed = (
                        stopped
                        and isinstance(receipt, SafeStopReceipt)
                        and receipt.confirmed
                    )
                    raise AgentError(
                        FailureCode.SYSTEM_FAULT,
                        "supervisor initialization failed: "
                        f"{initialization_error}; "
                        f"emergency stop confirmed={stop_confirmed}; "
                        f"stop_error={stop_error}",
                    ) from exc
                return self._safe_stop(
                    environment,
                    FailureCode.SYSTEM_FAULT,
                    f"unhandled supervisor failure: {exc}",
                    verification=None,
                    event_start=event_start,
                )
        finally:
            self._run_lock.release()

    def _run_once(
        self,
        task: TaskSchema,
        environment: ExecutionEnvironment,
    ) -> RunResult:
        """Plan semantic subtasks and execute them with bounded local recovery."""

        self._fsm = AgentFSM()
        self._queue.clear()
        self._plan = None
        self._run_id = str(uuid4())
        self._task_id = task.task_id
        self._perception_disabled_for_run = self._perception_quarantined
        self.gateway.reset()
        self._memory = self.memory_store.create(self._run_id, task.task_id)
        self._lifecycle = FixedLifecycle(self.task_profile)
        self._memory.control_token = self._lifecycle.token.value
        self._memory.control_token_history.append(self._lifecycle.token.value)
        event_start = len(self.events.events)
        verification: VerificationResult | None = None
        self._emit(
            "run.started",
            {
                "task_schema_version": task.schema_version,
                "perception_gate_enabled": False,
                "perception_sidecar_enabled": self.perception is not None,
                "perception_mode": self.perception_mode.value,
                "perception_required": self.perception_required,
                "perception_agent": (
                    self.perception.descriptor.name
                    if self.perception is not None
                    else None
                ),
                "topology_mode": self.topology_mode,
                "supervisor_nlp": False,
            },
        )
        self._emit(
            "control_token.initialized",
            {
                "next_token": self._lifecycle.token.value,
                "reason": "fixed task profile accepted",
            },
        )
        self._transition(AgentState.VALIDATING_TASK, "validate TaskSchema")
        try:
            task.validate()
            if task.instruction != self.task_profile.arm_a_instruction:
                raise AgentError(
                    FailureCode.INVALID_TASK,
                    "task instruction must exactly match the frozen Arm_A "
                    "deployment instruction",
                )
            if any(
                condition.required_votes != self.task_profile.handoff_required_votes
                for condition in task.postconditions
            ):
                raise AgentError(
                    FailureCode.INVALID_TASK,
                    "fixed task postconditions must use three-frame/two-vote "
                    f"quorum (required_votes="
                    f"{self.task_profile.handoff_required_votes})",
                )
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
            expected_assignments = (
                (
                    ARM_A_PACK_HANDOFF_SUBTASK_ID,
                    self.task_profile.primary_executor,
                ),
                (
                    ARM_B_TRANSPORT_SUBTASK_ID,
                    self.task_profile.collaborative_executor,
                ),
            )
            actual_assignments = tuple(
                (item.subtask_id, item.assigned_executor)
                for item in self._plan.subtasks
            )
            if actual_assignments != expected_assignments:
                raise AgentError(
                    FailureCode.INVALID_TASK,
                    "fixed dual-VLA plan was mutated: "
                    f"expected={expected_assignments!r}, "
                    f"actual={actual_assignments!r}",
                )
            for planned_subtask in self._plan.subtasks:
                required_executor = self.task_profile.executor_for_subtask(
                    planned_subtask.subtask_id
                )
                if required_executor in self._quarantined_executors:
                    raise AgentError(
                        FailureCode.EXECUTOR_UNAVAILABLE,
                        f"{required_executor} is quarantined after a previous "
                        "hard timeout; restart its isolated service before reuse",
                    )
                self.executors.select_exact(
                    required_executor,
                    planned_subtask.as_task(task),
                )
                self._emit(
                    "executor.preflight_ready",
                    {
                        "executor": required_executor,
                        "subtask_id": planned_subtask.subtask_id,
                    },
                )
            perception_health_error: str | None = None
            try:
                perception_ready = (
                    not self._perception_quarantined
                    and self.perception is not None
                    and self.perception.health()
                )
            except Exception as exc:
                perception_ready = False
                perception_health_error = str(exc)
            self._emit(
                "perception.preflight",
                {
                    "agent": (
                        self.perception.descriptor.name
                        if self.perception is not None
                        else None
                    ),
                    "ready": perception_ready,
                    "error": perception_health_error,
                    "quarantined": self._perception_quarantined,
                    "control_path_impact": "none",
                },
            )
            if not perception_ready:
                self._perception_disabled_for_run = True
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
            if action_has_executed:
                return self._safe_stop(
                    environment,
                    exc.code,
                    f"online observation {phase} failed after physical action: {exc}",
                    verification,
                    event_start,
                )
            return self._fail(exc.code, str(exc), verification, event_start)

        def terminal_failure(
            code: FailureCode,
            message: str,
            terminal_verification: VerificationResult | None,
        ) -> RunResult:
            """Physically stop whenever a failed run has already moved a robot."""

            if action_has_executed:
                return self._safe_stop(
                    environment,
                    code,
                    message,
                    terminal_verification,
                    event_start,
                )
            return self._fail(
                code,
                message,
                terminal_verification,
                event_start,
            )

        def collect_verification_frames(
            initial_frames: Sequence[Observation],
        ) -> tuple[list[Observation], RunResult | None]:
            frames = list(initial_frames)
            try:
                while len(frames) < self.verification_frames:
                    frame = self.gateway.ingest_online(
                        self._observe_with_deadline(environment)
                    )
                    self._memory.last_observation_id = frame.observation_id
                    stopped = self._check_system_fault(
                        frame,
                        environment,
                        verification,
                        event_start,
                    )
                    if stopped is not None:
                        subtask.status = SubtaskStatus.FAILED
                        return frames, stopped
                    frames.append(frame)
            except AgentError as exc:
                subtask.status = SubtaskStatus.FAILED
                return frames, reject_observation(exc, "during verification")
            except Exception as exc:
                subtask.status = SubtaskStatus.FAILED
                return frames, self._safe_stop(
                    environment,
                    FailureCode.SYSTEM_FAULT,
                    f"verification observation failed: {exc}",
                    verification,
                    event_start,
                )
            return frames, None

        def start_handoff_verification(reason: str) -> RunResult | None:
            assert self._lifecycle is not None
            try:
                previous, current_token = self._lifecycle.begin_handoff_verification()
            except AgentError as exc:
                return self._safe_stop(
                    environment,
                    exc.code,
                    str(exc),
                    verification,
                    event_start,
                )
            self._record_control_token(previous, current_token, reason)
            self._emit(
                "handoff.verification_started",
                {
                    "bin_id": self.task_profile.bin_id,
                    "handoff_zone": self.task_profile.handoff_zone,
                    "required_frames": (self.task_profile.handoff_verification_frames),
                    "required_votes": self.task_profile.handoff_required_votes,
                    "actuator_access": "NONE",
                },
            )
            return None

        while True:
            try:
                last_observation = self.gateway.ingest_online(
                    self._observe_with_deadline(environment)
                )
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
                        frame = self.gateway.ingest_online(
                            self._observe_with_deadline(environment)
                        )
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
                    return terminal_failure(
                        precondition_result.code,
                        f"preconditions are {precondition_result.verdict.value} "
                        f"for {subtask.subtask_id}",
                        precondition_result,
                    )

            # A repeat-until workflow must be a no-op when its observable
            # postcondition already holds (for example, a bin is already full).
            if subtask.repeat_until_postcondition and subtask_iterations == 0:
                precheck_frames = [last_observation]
                try:
                    while len(precheck_frames) < self.verification_frames:
                        frame = self.gateway.ingest_online(
                            self._observe_with_deadline(environment)
                        )
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
                        return terminal_failure(
                            FailureCode.INVALID_TASK,
                            f"dependencies not verified for {subtask.subtask_id}",
                            verification,
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

            last_observation, perception_packet, perception_terminal = (
                self._perceive_for_vla(
                    task=active_task,
                    observation=last_observation,
                    environment=environment,
                    verification=verification,
                    event_start=event_start,
                    action_has_executed=action_has_executed,
                    step_id=subtask_iterations,
                )
            )
            if perception_terminal is not None:
                subtask.status = SubtaskStatus.FAILED
                return perception_terminal
            self._transition(
                AgentState.ASSIGNING_ROLE,
                "assign or retain the lifecycle-owned VLA role",
            )
            if current is None:
                try:
                    required_executor = self.task_profile.executor_for_subtask(
                        subtask.subtask_id
                    )
                    self._lifecycle.authorize(
                        subtask.subtask_id,
                        required_executor,
                    )
                    current = self.executors.select_exact(
                        required_executor,
                        active_task,
                    )
                except AgentError as exc:
                    subtask.status = SubtaskStatus.FAILED
                    if exc.code is FailureCode.SAFETY_REJECTED:
                        return self._safe_stop(
                            environment,
                            exc.code,
                            str(exc),
                            verification,
                            event_start,
                        )
                    return terminal_failure(
                        exc.code,
                        str(exc),
                        verification,
                    )
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
                        "arm_id": (
                            self.task_profile.arm_a_id
                            if name == self.task_profile.primary_executor
                            else self.task_profile.arm_b_id
                        ),
                        "selection_mode": "fixed_lifecycle",
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
                original_instruction=(
                    task.instruction
                    if name == self.task_profile.primary_executor
                    else active_task.instruction
                ),
            )
            planning_observation = last_observation
            try:
                executor_task = TaskSchema.from_dict(active_task.to_dict())
                executor_observation = Observation(
                    observation_id=planning_observation.observation_id,
                    timestamp_ms=planning_observation.timestamp_ms,
                    data=deepcopy(dict(planning_observation.data)),
                    observation_version=planning_observation.observation_version,
                )
                completed, chunk_value, executor_error = _invoke_with_hard_deadline(
                    lambda: current.plan(
                        executor_task,
                        executor_observation,
                        context,
                    ),
                    self.executor_timeout_ms,
                )
                if not completed:
                    if isinstance(executor_error, TimeoutError):
                        self._quarantined_executors.add(name)
                        Thread(
                            target=lambda: current.cancel(
                                active_task.task_id,
                                "supervisor hard deadline exceeded",
                            ),
                            daemon=True,
                        ).start()
                        raise AgentError(
                            FailureCode.EXECUTOR_TIMEOUT,
                            str(executor_error),
                        )
                    if isinstance(executor_error, AgentError):
                        raise executor_error
                    raise AgentError(
                        FailureCode.EXECUTOR_RUNTIME,
                        f"executor adapter failed: {executor_error}",
                    )
                if not isinstance(chunk_value, ActionChunk):
                    raise AgentError(
                        FailureCode.EXECUTOR_BAD_RESPONSE,
                        "executor must return an ActionChunk",
                    )
                chunk = chunk_value
                if chunk.task_id != active_task.task_id or chunk.executor != name:
                    raise AgentError(
                        FailureCode.ACTION_CONTRACT_INVALID,
                        "action task_id/executor does not match active strategy",
                    )
                command_arm_id = (
                    self.task_profile.arm_a_id
                    if name == self.task_profile.primary_executor
                    else self.task_profile.arm_b_id
                )
            except AgentError as exc:
                if exc.code in {
                    FailureCode.ACTION_CONTRACT_INVALID,
                    FailureCode.ACTION_NON_FINITE,
                    FailureCode.ACTION_WORKSPACE_BREACH,
                    FailureCode.SAFETY_REJECTED,
                    FailureCode.EXECUTOR_TIMEOUT,
                }:
                    subtask.status = SubtaskStatus.FAILED
                    return self._safe_stop(
                        environment,
                        exc.code,
                        (
                            f"{name} was quarantined after a hard timeout: {exc}"
                            if exc.code is FailureCode.EXECUTOR_TIMEOUT
                            else f"unsafe action rejected from {name}: {exc}"
                        ),
                        verification,
                        event_start,
                    )
                current, should_continue = self._recover(
                    task=active_task,
                    current=current,
                    code=exc.code,
                    reason=str(exc),
                    local_replans=local_replans,
                )
                if should_continue:
                    decisions_in_strategy_attempt = 0
                    continue
                subtask.status = SubtaskStatus.FAILED
                return terminal_failure(
                    FailureCode.RECOVERY_EXHAUSTED,
                    f"recovery exhausted after {exc.code.value}: {exc}",
                    verification,
                )
            except Exception as exc:
                current, should_continue = self._recover(
                    task=active_task,
                    current=current,
                    code=FailureCode.EXECUTOR_RUNTIME,
                    reason=str(exc),
                    local_replans=local_replans,
                )
                if should_continue:
                    decisions_in_strategy_attempt = 0
                    continue
                subtask.status = SubtaskStatus.FAILED
                return terminal_failure(
                    FailureCode.RECOVERY_EXHAUSTED,
                    f"unexpected executor error; recovery exhausted: {exc}",
                    verification,
                )

            # Close the inference-time TOCTOU window.  The robot must still be
            # safe after model latency, and the controller receives the exact
            # fresh observation id it must atomically revalidate.
            try:
                execution_observation = self.gateway.ingest_online(
                    self._observe_with_deadline(environment)
                )
                self._memory.last_observation_id = execution_observation.observation_id
            except AgentError as exc:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    exc.code,
                    f"pre-execution observation rejected: {exc}",
                    verification,
                    event_start,
                )
            except Exception as exc:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    FailureCode.SYSTEM_FAULT,
                    f"pre-execution observation failed: {exc}",
                    verification,
                    event_start,
                )
            stopped = self._check_system_fault(
                execution_observation,
                environment,
                verification,
                event_start,
            )
            if stopped is not None:
                subtask.status = SubtaskStatus.FAILED
                return stopped
            state_change_failure = self._preexecution_state_change_failure(
                planning_observation,
                execution_observation,
                arm_id=command_arm_id,
            )
            if state_change_failure is not None:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    FailureCode.SAFETY_REJECTED,
                    f"action invalidated during VLA inference: {state_change_failure}",
                    verification,
                    event_start,
                )
            semantic_change_reason = self._preexecution_semantic_change_reason(
                planning_observation,
                execution_observation,
            )
            if semantic_change_reason is not None:
                current, should_continue = self._recover(
                    task=active_task,
                    current=current,
                    code=FailureCode.OBSERVATION_INVALID,
                    reason=semantic_change_reason,
                    local_replans=local_replans,
                )
                if should_continue:
                    last_observation = execution_observation
                    decisions_in_strategy_attempt = 0
                    continue
                subtask.status = SubtaskStatus.FAILED
                return terminal_failure(
                    FailureCode.RECOVERY_EXHAUSTED,
                    "scene changed repeatedly during VLA inference: "
                    f"{semantic_change_reason}",
                    verification,
                )
            last_observation = execution_observation
            command_token = self._lifecycle.token.value
            decision = self.safety.validate_and_limit(
                chunk,
                last_observation,
                arm_id=command_arm_id,
                control_token=command_token,
            )
            if not decision.accepted or decision.chunk is None:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    decision.code,
                    f"unsafe action rejected from {name}: {decision.reason}",
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
            try:
                self._lifecycle.authorize(subtask.subtask_id, name)
            except AgentError as exc:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    exc.code,
                    str(exc),
                    verification,
                    event_start,
                )
            interlock_failure = self._fixed_arm_interlock_failure(
                last_observation,
                arm_id=command_arm_id,
            )
            if interlock_failure is not None:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    FailureCode.SAFETY_REJECTED,
                    f"dual-arm interlock rejected action: {interlock_failure}",
                    verification,
                    event_start,
                )
            if decision.chunk.chunk_id in self._memory.completed_chunk_ids:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    FailureCode.ACTION_CONTRACT_INVALID,
                    "executor repeated an already executed chunk_id: "
                    f"{decision.chunk.chunk_id!r}",
                    verification,
                    event_start,
                )
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
                    "arm_id": command_arm_id,
                    "control_token": command_token,
                    "perception_packet_id": (
                        perception_packet.packet_id
                        if perception_packet is not None
                        else None
                    ),
                },
            )
            try:
                while self._queue:
                    action = self._queue.popleft()
                    raw_observation = self._step_with_deadline(
                        environment,
                        action,
                        arm_id=command_arm_id,
                        control_token=command_token,
                        command_id=(
                            f"{self._run_id}:{decision.chunk.chunk_id}:"
                            f"{subtask_iterations}"
                        ),
                        expected_observation_id=last_observation.observation_id,
                        expected_state_digest=execution_guard_digest(
                            last_observation.data
                        ),
                    )
                    action_has_executed = True
                    last_observation = self.gateway.ingest_online(raw_observation)
                    self._memory.last_observation_id = last_observation.observation_id
                    stopped = self._check_system_fault(
                        last_observation, environment, verification, event_start
                    )
                    if stopped is not None:
                        subtask.status = SubtaskStatus.FAILED
                        return stopped
            except PreWriteStateStaleError as exc:
                # The adapter's typed contract proves that the command was
                # durably aborted before any controller write. Do not claim an
                # execution or stop the cell; discard the chunk and ask the
                # lifecycle-owned VLA for one bounded replan.
                subtask_iterations = max(0, subtask_iterations - 1)
                self._emit(
                    "execution.prewrite_state_stale",
                    {
                        "executor": name,
                        "subtask_id": subtask.subtask_id,
                        "chunk_id": decision.chunk.chunk_id,
                        "failure_code": exc.code.value,
                        "hardware_write_attempted": exc.hardware_write_attempted,
                    },
                )
                current, should_continue = self._recover(
                    task=active_task,
                    current=current,
                    code=exc.code,
                    reason=str(exc),
                    local_replans=local_replans,
                )
                if should_continue:
                    decisions_in_strategy_attempt = 0
                    continue
                subtask.status = SubtaskStatus.FAILED
                return terminal_failure(
                    FailureCode.RECOVERY_EXHAUSTED,
                    "pre-write scene state changed repeatedly; recovery exhausted: "
                    f"{exc}",
                    verification,
                )
            except AgentError as exc:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    exc.code,
                    "environment rejected an issued action or returned an unsafe "
                    f"post-action observation: {exc}",
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
            handoff_subtask = (
                self._lifecycle is not None
                and subtask.subtask_id == ARM_A_PACK_HANDOFF_SUBTASK_ID
            )
            initial_frames: list[Observation] = [last_observation]
            if handoff_subtask and self._lifecycle.token is ControlToken.A_ONLY:
                candidate_task = replace(
                    active_task,
                    postconditions=tuple(
                        replace(condition, required_votes=1)
                        for condition in active_task.postconditions
                    ),
                )
                candidate = self.verifier.verify(
                    candidate_task,
                    [last_observation],
                )
                self._emit(
                    HANDOFF_CANDIDATE_CHECKED_EVENT_TYPE,
                    {
                        "verdict": candidate.verdict.value,
                        "observation_id": last_observation.observation_id,
                        "actuator_access": "Arm_A",
                    },
                )
                if candidate.verdict is Verdict.PASS:
                    stopped = start_handoff_verification(
                        "single-frame handoff candidate passed; lock both arms"
                    )
                    if stopped is not None:
                        subtask.status = SubtaskStatus.FAILED
                        return stopped
                    # All quorum frames must be captured after actuator lockout.
                    initial_frames = []

            frames, verification_terminal = collect_verification_frames(initial_frames)
            if verification_terminal is not None:
                return verification_terminal
            verification = self.verifier.verify(active_task, frames)
            if (
                handoff_subtask
                and verification.verdict is Verdict.PASS
                and self._lifecycle.token is ControlToken.A_ONLY
            ):
                stopped = start_handoff_verification(
                    "three-frame candidate passed; repeat quorum under arm lockout"
                )
                if stopped is not None:
                    subtask.status = SubtaskStatus.FAILED
                    return stopped
                frames, verification_terminal = collect_verification_frames(())
                if verification_terminal is not None:
                    return verification_terminal
                verification = self.verifier.verify(active_task, frames)
            if self._lifecycle is not None:
                verification_camera_key = (
                    "handoff_rgb" if handoff_subtask else "arm_b_rgb"
                )
                frame_identity_failure = self._verification_frame_identity_failure(
                    frames,
                    camera_key=verification_camera_key,
                )
                if frame_identity_failure is not None:
                    subtask.status = SubtaskStatus.FAILED
                    return self._safe_stop(
                        environment,
                        FailureCode.OBSERVATION_INVALID,
                        "verification frame identity rejected: "
                        f"{frame_identity_failure}",
                        verification,
                        event_start,
                    )
            self._emit(
                "verification.completed",
                {
                    "subtask_id": subtask.subtask_id,
                    "verdict": verification.verdict.value,
                    "failure_code": verification.code.value,
                    "frames": len(frames),
                    "composite_pass_votes": verification.composite_pass_votes,
                    "composite_fail_votes": verification.composite_fail_votes,
                    "composite_uncertain_votes": (
                        verification.composite_uncertain_votes
                    ),
                    "composite_required_votes": (verification.composite_required_votes),
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
                if self._lifecycle is not None:
                    if subtask.subtask_id == ARM_A_PACK_HANDOFF_SUBTASK_ID:
                        interlock_failure = self._fixed_arm_interlock_failure(
                            frames[-1],
                            arm_id=self.task_profile.arm_b_id,
                        )
                        if interlock_failure is not None:
                            subtask.status = SubtaskStatus.FAILED
                            return self._safe_stop(
                                environment,
                                FailureCode.SAFETY_REJECTED,
                                "handoff interlock rejected B_ONLY grant: "
                                f"{interlock_failure}",
                                verification,
                                event_start,
                            )
                        if self._lifecycle.token is not ControlToken.HANDOFF_VERIFY:
                            subtask.status = SubtaskStatus.FAILED
                            return self._safe_stop(
                                environment,
                                FailureCode.SAFETY_REJECTED,
                                "handoff quorum passed outside HANDOFF_VERIFY",
                                verification,
                                event_start,
                            )
                        self._emit(
                            HANDOFF_VERIFIED_EVENT_TYPE,
                            {
                                "quorum_passed": True,
                                "grants_b_only": False,
                                "stable_frames": len(frames),
                                "required_frames": (
                                    self.task_profile.handoff_verification_frames
                                ),
                                "required_votes": (
                                    self.task_profile.handoff_required_votes
                                ),
                                "bin_id": self.task_profile.bin_id,
                                "handoff_zone": self.task_profile.handoff_zone,
                                "arm_a_retreated": True,
                                "oracle_coordinates_used": False,
                            },
                        )
                        self._emit(
                            HANDOFF_READY_EVENT_TYPE,
                            {
                                "bin_id": self.task_profile.bin_id,
                                "from_arm": self.task_profile.arm_a_id,
                                "to_arm": self.task_profile.arm_b_id,
                                "durable_ack": True,
                                "grants_b_only": True,
                            },
                        )
                        previous, current_token = self._lifecycle.grant_arm_b()
                        self._record_control_token(
                            previous,
                            current_token,
                            "durable handoff.ready grants Arm B exclusive control",
                        )
                    elif subtask.subtask_id == ARM_B_TRANSPORT_SUBTASK_ID:
                        final_interlock_failure = (
                            self._fixed_observation_consistency_failure(frames[-1])
                        )
                        if final_interlock_failure is None:
                            robot = frames[-1].data.get("robot")
                            assert isinstance(robot, Mapping)
                            arm_a = robot.get("arm_a")
                            arm_b = robot.get("arm_b")
                            if (
                                robot.get("active_arm") != "NONE"
                                or not isinstance(arm_a, Mapping)
                                or arm_a.get("retreated") is not True
                                or not isinstance(arm_b, Mapping)
                                or arm_b.get("retreated") is not True
                            ):
                                final_interlock_failure = (
                                    "terminal success requires active_arm=NONE "
                                    "and both arms retreated"
                                )
                        if final_interlock_failure is not None:
                            subtask.status = SubtaskStatus.FAILED
                            return self._safe_stop(
                                environment,
                                FailureCode.SAFETY_REJECTED,
                                "terminal dual-arm interlock failed: "
                                f"{final_interlock_failure}",
                                verification,
                                event_start,
                            )
                        previous, current_token = self._lifecycle.complete()
                        self._record_control_token(
                            previous,
                            current_token,
                            "same bin verified at finished station",
                        )
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
                    return terminal_failure(
                        FailureCode.INVALID_TASK,
                        f"dependencies not verified for {subtask.subtask_id}",
                        verification,
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

            if handoff_subtask and self._lifecycle.token is ControlToken.HANDOFF_VERIFY:
                subtask.status = SubtaskStatus.FAILED
                return self._safe_stop(
                    environment,
                    verification.code,
                    "handoff evidence lost quorum after both arms were locked",
                    verification,
                    event_start,
                )

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
                return terminal_failure(
                    FailureCode.RECOVERY_EXHAUSTED,
                    f"semantic loop reached max_iterations={subtask.max_iterations}",
                    verification,
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
                local_replans=local_replans,
            )
            if should_continue:
                decisions_in_strategy_attempt = 0
                continue
            subtask.status = SubtaskStatus.FAILED
            return terminal_failure(
                FailureCode.RECOVERY_EXHAUSTED,
                f"postcondition {verification.verdict.value}; recovery exhausted",
                verification,
            )
