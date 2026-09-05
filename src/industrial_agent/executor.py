"""Executor protocol and independent-process VLA adapter skeletons.

The adapters deliberately do not import either model repository. A production
deployment runs each model in its own pinned environment/process and injects a
transport implementation (HTTP, Unix socket, gRPC, etc.) here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable
from uuid import uuid4

from .contracts import (
    ACTION_CONTRACT_VERSION,
    PI05_EXECUTOR_NAME,
    ActionChunk,
    ActionStep,
    Observation,
    TaskSchema,
)
from .errors import ContractError, ExecutorError, FailureCode
from .observation import FROZEN_IMAGE_HEIGHT, FROZEN_IMAGE_WIDTH
from .sync_contract import canonical_observed_state_7d

ARTIFACT_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-fA-F]{64}")
CAS_IMAGE_URI_PATTERN = re.compile(r"cas://sha256/([0-9a-fA-F]{64})")
PI05_TASK_TYPES = frozenset(
    {"pick_place", "visual_manipulation", "instruction_interaction"}
)
EXECUTOR_CONFIG_FIELDS = frozenset(
    {
        "enabled",
        "base_url",
        "checkpoint_sha",
        "norm_stats_sha",
    }
)
EXECUTOR_HEALTH_RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "service",
        "service_version",
        "status",
        "uptime_ms",
        "checkpoint_sha",
        "norm_stats_sha",
        "supported_task_types",
        "supported_action_contracts",
        "queue",
        "device",
        "time_ms",
    }
)


@dataclass(frozen=True)
class ExecutorDescriptor:
    """Runtime capability identity; `/health` is the JSON transport contract."""

    name: str
    task_types: frozenset[str]
    action_contract_version: str
    checkpoint_sha: str
    norm_stats_sha: str

    def __post_init__(self) -> None:
        _require_pinned_artifact_digest(self.checkpoint_sha, "checkpoint_sha")
        _require_pinned_artifact_digest(self.norm_stats_sha, "norm_stats_sha")


@dataclass(frozen=True)
class ExecutionContext:
    """Supervisor-local call context assembled into `/v1/infer` envelopes."""

    run_id: str
    strategy_attempt: int
    replan_index: int
    step_id: int = 0
    timeout_ms: int = 15_000
    original_instruction: str | None = None
    arm_id: str | None = None
    # Model-facing identity is deliberately separate from Supervisor workflow
    # stages.  For a multi-stage task these fields remain the frozen formal
    # task identity and prompt for every π0.5 call.
    model_task_id: str | None = None
    model_subtask_id: str | None = None
    model_instruction: str | None = None


@runtime_checkable
class Executor(Protocol):
    descriptor: ExecutorDescriptor

    def health(self) -> bool:
        """Return readiness without loading a model in the supervisor process."""

    def plan(
        self, task: TaskSchema, observation: Observation, context: ExecutionContext
    ) -> ActionChunk:
        """Return a versioned 7-D action chunk."""

    def cancel(self, task_id: str, reason: str) -> None:
        """Cancel pending inference/execution and discard server-side context."""


@runtime_checkable
class ProcessTransport(Protocol):
    """Minimal request/reply boundary for an independently hosted model.

    Implementations must return the decoded structured body for both successful
    and contract-defined non-2xx responses so the adapter can preserve stable
    error codes. Connection failures and deadlines may raise exceptions.
    """

    def request(
        self, route: str, payload: Mapping[str, Any], timeout_ms: int
    ) -> Mapping[str, Any]: ...


def is_pinned_artifact_digest(value: Any) -> bool:
    """Return whether a value is a complete immutable SHA-256 identifier."""

    return (
        isinstance(value, str) and ARTIFACT_DIGEST_PATTERN.fullmatch(value) is not None
    )


def _require_pinned_artifact_digest(value: Any, field: str) -> str:
    if not is_pinned_artifact_digest(value):
        raise ValueError(f"{field} must match 'sha256:<64 hexadecimal characters>'")
    return value


def _is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(
        value, (str, bytes, bytearray)
    )


def _is_compatible_version(value: Any, expected: str) -> bool:
    if not isinstance(value, str):
        return False
    parts = value.split(".")
    expected_parts = expected.split(".")
    return (
        len(parts) == 2
        and len(expected_parts) == 2
        and all(part.isdigit() for part in parts)
        and parts[0] == expected_parts[0]
    )


def _phase_vla_inputs(
    observation: Observation,
    *,
    arm_key: str,
    camera_key: str,
) -> tuple[Any, Any, Any, Any]:
    """Return only phase RGB/proprio fields; never lifecycle summaries."""

    camera = observation.data.get("camera")
    robot = observation.data.get("robot")
    if not isinstance(camera, Mapping) or not isinstance(robot, Mapping):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "camera and robot observations must be objects",
        )
    arm_state = robot.get(arm_key)
    if not isinstance(arm_state, Mapping):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"robot.{arm_key} observation must be an object",
        )
    expected_camera_id_by_key = {
        "arm_a_rgb": "CAM_A_TOP",
        "arm_b_rgb": "CAM_B_TOP",
    }
    expected_camera_id = expected_camera_id_by_key.get(camera_key)
    if expected_camera_id is None:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"unsupported fixed VLA camera key: {camera_key!r}",
        )
    if camera_key not in camera:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"camera.{camera_key} is required; production VLA inputs do not "
            "fall back to camera.full_image",
        )
    full_image = _canonical_image_reference(
        camera[camera_key],
        f"camera.{camera_key}",
        expected_camera_id=expected_camera_id,
        expected_size=(FROZEN_IMAGE_WIDTH, FROZEN_IMAGE_HEIGHT),
    )
    if camera.get("wrist_image") is not None:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "frozen three-camera profile requires camera.wrist_image=null",
        )
    tcp_pose = arm_state.get("tcp_pose_m_rad")
    try:
        state = canonical_observed_state_7d(
            tcp_pose,
            arm_state.get("state"),
            arm_state.get("gripper_open"),
        )
    except (TypeError, ValueError) as exc:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"robot.{arm_key} cannot produce canonical state_7d: {exc}",
        ) from exc
    tcp_pose = state[:6]
    return full_image, None, state, tcp_pose


def _canonical_image_reference(
    value: Any,
    field: str,
    *,
    expected_camera_id: str | None = None,
    expected_size: tuple[int, int] | None = None,
) -> dict[str, Any]:
    """Rebuild an exact image allowlist before crossing a VLA boundary."""

    required = {"uri", "image_sha256", "camera_id", "width", "height"}
    if not isinstance(value, Mapping) or set(value) != required:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{field} must contain exactly {sorted(required)}; got {actual}",
        )
    uri = value["uri"]
    image_sha256 = value["image_sha256"]
    camera_id = value["camera_id"]
    width = value["width"]
    height = value["height"]
    if not isinstance(uri, str) or not uri:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{field}.uri must be a non-empty string",
        )
    uri_match = CAS_IMAGE_URI_PATTERN.fullmatch(uri)
    if uri_match is None:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{field}.uri must be a content-addressed cas://sha256/<digest> URI",
        )
    if not is_pinned_artifact_digest(image_sha256):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{field}.image_sha256 must be a complete SHA-256 identifier",
        )
    if uri_match.group(1).casefold() != image_sha256.split(":", 1)[1].casefold():
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{field}.uri digest must match image_sha256",
        )
    if not isinstance(camera_id, str) or not camera_id:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{field}.camera_id must be a non-empty string",
        )
    if expected_camera_id is not None and camera_id != expected_camera_id:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{field}.camera_id must be {expected_camera_id!r}, got {camera_id!r}",
        )
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item < 1
        for item in (width, height)
    ):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{field}.width/height must be positive integers",
        )
    if expected_size is not None and (width, height) != expected_size:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{field} must use frozen {expected_size[0]}x{expected_size[1]} "
            f"resolution; got {width}x{height}",
        )
    return {
        "uri": uri,
        "image_sha256": image_sha256,
        "camera_id": camera_id,
        "width": width,
        "height": height,
    }


def _validate_health_response(
    response: Mapping[str, Any], descriptor: ExecutorDescriptor
) -> None:
    unknown_keys = set(response) - EXECUTOR_HEALTH_RESPONSE_FIELDS
    if unknown_keys:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "executor health response contains unknown fields: "
            f"{sorted(unknown_keys, key=str)}",
        )
    invalid_optional_fields: dict[str, Any] = {}
    if "service_version" in response and not isinstance(
        response["service_version"], str
    ):
        invalid_optional_fields["service_version"] = response["service_version"]
    for field in ("uptime_ms", "time_ms"):
        if field not in response:
            continue
        value = response[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            invalid_optional_fields[field] = value
    for field in ("queue", "device"):
        if field in response and not isinstance(response[field], Mapping):
            invalid_optional_fields[field] = response[field]
    if invalid_optional_fields:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "executor health response violates optional field schema: "
            f"{invalid_optional_fields}",
        )
    version = response.get("schema_version")
    if not _is_compatible_version(version, "1.0"):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"missing or incompatible health schema_version: {version!r}",
        )
    expected = {
        "service": descriptor.name,
        "status": "ready",
        "checkpoint_sha": descriptor.checkpoint_sha,
        "norm_stats_sha": descriptor.norm_stats_sha,
    }
    mismatches = {
        key: {"expected": value, "actual": response.get(key)}
        for key, value in expected.items()
        if response.get(key) != value
    }
    contracts = response.get("supported_action_contracts")
    if (
        not _is_sequence(contracts)
        or any(not isinstance(item, str) for item in contracts)
        or descriptor.action_contract_version not in contracts
    ):
        mismatches["supported_action_contracts"] = {
            "expected_to_contain": descriptor.action_contract_version,
            "actual": contracts,
        }
    task_types = response.get("supported_task_types")
    if (
        not _is_sequence(task_types)
        or any(not isinstance(item, str) for item in task_types)
        or not descriptor.task_types.issubset(set(task_types))
    ):
        mismatches["supported_task_types"] = {
            "expected_to_contain": sorted(descriptor.task_types),
            "actual": task_types,
        }
    if mismatches:
        raise ExecutorError(
            FailureCode.EXECUTOR_MODEL_REVISION_MISMATCH,
            f"executor health metadata mismatch: {mismatches}",
        )


_RESPONSE_COMMON_KEYS = frozenset(
    {
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
        "status",
    }
)
_RESPONSE_OPTIONAL_KEYS = frozenset({"action_chunk", "error", "timing"})
_CHUNK_KEYS = frozenset(
    {
        "contract_version",
        "chunk_id",
        "task_id",
        "executor",
        "action_space",
        "frame",
        "translation_unit",
        "rotation_unit",
        "gripper_unit",
        "steps",
    }
)
_STEP_KEYS = frozenset({"values", "duration_ms"})


def _raise_executor_status(response: Mapping[str, Any], status: str) -> None:
    raw_error = response.get("error")
    if not isinstance(raw_error, Mapping):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{status!r} response requires an error object",
        )
    code_value = raw_error.get("code")
    try:
        code = FailureCode(str(code_value))
    except ValueError as exc:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"executor returned unknown failure code: {code_value!r}",
        ) from exc
    message = raw_error.get("message")
    retryable = raw_error.get("retryable")
    if not isinstance(message, str) or not message:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "executor error.message must be a non-empty string",
        )
    if not isinstance(retryable, bool):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "executor error.retryable must be boolean",
        )
    retry_after_ms = raw_error.get("retry_after_ms")
    if retry_after_ms is not None and (
        isinstance(retry_after_ms, bool)
        or not isinstance(retry_after_ms, int)
        or retry_after_ms < 0
    ):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "executor error.retry_after_ms must be a non-negative integer",
        )
    raise ExecutorError(
        code,
        message,
        retryable=retryable,
        retry_after_ms=retry_after_ms,
    )


def _validate_timing(response: Mapping[str, Any]) -> None:
    if "timing" not in response:
        return
    timing = response["timing"]
    if not isinstance(timing, Mapping):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE, "timing must be an object"
        )
    allowed = frozenset({"queue_ms", "inference_ms", "total_ms"})
    unknown = set(timing) - allowed
    if unknown:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"timing contains unknown fields: {sorted(unknown)}",
        )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0
        for value in timing.values()
    ):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "timing values must be non-negative numbers",
        )


def _validate_response_envelope(
    response: Mapping[str, Any],
    *,
    request_id: str,
    context: ExecutionContext,
    task: TaskSchema,
    expected_task_id: str,
    expected_subtask_id: str,
    observation: Observation,
    descriptor: ExecutorDescriptor,
) -> None:
    unknown = set(response) - _RESPONSE_COMMON_KEYS - _RESPONSE_OPTIONAL_KEYS
    if unknown:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"executor response contains unknown fields: {sorted(unknown)}",
        )
    version = response.get("schema_version")
    if not _is_compatible_version(version, "1.0"):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"missing or incompatible response schema_version: {version!r}",
        )
    expected = {
        "request_id": request_id,
        "trace_id": context.run_id,
        "episode_id": context.run_id,
        "task_id": expected_task_id,
        "subtask_id": expected_subtask_id,
        "step_id": context.step_id,
        "observation_id": observation.observation_id,
        "executor": descriptor.name,
        "checkpoint_sha": descriptor.checkpoint_sha,
        "norm_stats_sha": descriptor.norm_stats_sha,
    }
    mismatches = {
        key: {"expected": value, "actual": response.get(key)}
        for key, value in expected.items()
        if response.get(key) != value
    }
    if mismatches:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"executor response envelope mismatch: {mismatches}",
        )
    _validate_timing(response)
    status = response.get("status")
    if status not in {"ok", "error", "cancelled"}:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"unsupported executor response status: {status!r}",
        )
    if status == "ok":
        if "action_chunk" not in response or "error" in response:
            raise ExecutorError(
                FailureCode.EXECUTOR_BAD_RESPONSE,
                "ok response requires action_chunk and forbids error",
            )
        return
    if "action_chunk" in response:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"{status} response must not contain action_chunk",
        )
    _raise_executor_status(response, status)


def _parse_canonical_chunk(
    response: Mapping[str, Any], *, task: TaskSchema, descriptor: ExecutorDescriptor
) -> ActionChunk:
    raw_chunk = response.get("action_chunk")
    if not isinstance(raw_chunk, Mapping):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "ok response action_chunk must be an object",
        )
    missing = _CHUNK_KEYS - set(raw_chunk)
    unknown = set(raw_chunk) - _CHUNK_KEYS
    if missing or unknown:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"action_chunk fields invalid; missing={sorted(missing)}, unknown={sorted(unknown)}",
        )
    string_fields = (
        "contract_version",
        "chunk_id",
        "task_id",
        "executor",
        "action_space",
        "frame",
        "translation_unit",
        "rotation_unit",
        "gripper_unit",
    )
    if any(not isinstance(raw_chunk[key], str) for key in string_fields):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "action_chunk metadata fields must be strings",
        )
    raw_steps = raw_chunk["steps"]
    if not _is_sequence(raw_steps):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "action_chunk.steps must be an array",
        )
    steps: list[ActionStep] = []
    try:
        for index, raw_step in enumerate(raw_steps):
            if not isinstance(raw_step, Mapping):
                raise ContractError(
                    FailureCode.ACTION_CONTRACT_INVALID,
                    f"action step {index} must be an object",
                )
            if set(raw_step) != _STEP_KEYS:
                raise ContractError(
                    FailureCode.ACTION_CONTRACT_INVALID,
                    f"action step {index} requires exactly values and duration_ms",
                )
            values = raw_step["values"]
            duration_ms = raw_step["duration_ms"]
            if not _is_sequence(values):
                raise ContractError(
                    FailureCode.ACTION_CONTRACT_INVALID,
                    f"action step {index}.values must be an array",
                )
            if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
                raise ContractError(
                    FailureCode.ACTION_CONTRACT_INVALID,
                    f"action step {index}.duration_ms must be an integer",
                )
            steps.append(ActionStep.from_sequence(values, duration_ms))
        chunk = ActionChunk(
            contract_version=raw_chunk["contract_version"],
            chunk_id=raw_chunk["chunk_id"],
            task_id=raw_chunk["task_id"],
            executor=raw_chunk["executor"],
            action_space=raw_chunk["action_space"],
            frame=raw_chunk["frame"],
            translation_unit=raw_chunk["translation_unit"],
            rotation_unit=raw_chunk["rotation_unit"],
            gripper_unit=raw_chunk["gripper_unit"],
            steps=tuple(steps),
        )
        chunk.validate_contract()
    except ContractError as exc:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            f"invalid canonical action_chunk: {exc}",
        ) from exc
    if chunk.task_id != task.task_id or chunk.executor != descriptor.name:
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "action_chunk task_id/executor does not match active request",
        )
    if not _is_compatible_version(
        chunk.contract_version, descriptor.action_contract_version
    ):
        raise ExecutorError(
            FailureCode.EXECUTOR_BAD_RESPONSE,
            "action_chunk contract version is incompatible with executor descriptor",
        )
    return chunk


class Pi05Adapter:
    """Adapter for a separately deployed openpi π0.5 policy server.

    The upstream policy API returns `policy.infer(...)[\"actions\"]`; this
    service boundary must convert that native value into a canonical
    `action_chunk` before responding.
    """

    def __init__(
        self,
        transport: ProcessTransport,
        *,
        checkpoint_sha: str,
        norm_stats_sha: str,
        task_types: frozenset[str] | None = None,
    ):
        checkpoint_sha = _require_pinned_artifact_digest(
            checkpoint_sha, "checkpoint_sha"
        )
        norm_stats_sha = _require_pinned_artifact_digest(
            norm_stats_sha, "norm_stats_sha"
        )
        self.transport = transport
        self.descriptor = ExecutorDescriptor(
            name=PI05_EXECUTOR_NAME,
            task_types=task_types or PI05_TASK_TYPES,
            action_contract_version=ACTION_CONTRACT_VERSION,
            checkpoint_sha=checkpoint_sha,
            norm_stats_sha=norm_stats_sha,
        )
        self._cancel_context_by_task: dict[str, tuple[str, str]] = {}

    def health(self) -> bool:
        try:
            response = self.transport.request("/health", {}, 1_000)
            if not isinstance(response, Mapping):
                return False
            _validate_health_response(response, self.descriptor)
            return True
        except Exception:
            return False

    def plan(
        self, task: TaskSchema, observation: Observation, context: ExecutionContext
    ) -> ActionChunk:
        arm_id = context.arm_id or task.metadata.get("arm_id", "Arm_A")
        if arm_id not in {"Arm_A", "Arm_B"}:
            raise ExecutorError(
                FailureCode.INVALID_TASK,
                f"π0.5 task arm_id must be Arm_A or Arm_B, got {arm_id!r}",
            )
        arm_key = "arm_a" if arm_id == "Arm_A" else "arm_b"
        camera_key = "arm_a_rgb" if arm_id == "Arm_A" else "arm_b_rgb"
        full_image, wrist_image, state, tcp_pose = _phase_vla_inputs(
            observation,
            arm_key=arm_key,
            camera_key=camera_key,
        )
        request_id = str(uuid4())
        # `task.metadata.subtask_id` may be an internal Supervisor phase name.
        # Never expose that workflow identity as the π0.5 model identity.
        model_task_id = context.model_task_id or task.task_id
        model_subtask_id = context.model_subtask_id or task.task_id
        model_instruction = (
            context.model_instruction
            or context.original_instruction
            or task.instruction
        )
        self._cancel_context_by_task[task.task_id] = (
            context.run_id,
            model_subtask_id,
        )
        model_input = {
            "prompt": model_instruction,
            "arm_id": arm_id,
            "observation": {
                "camera": {
                    "full_image": full_image,
                    "wrist_image": wrist_image,
                },
                "robot": {
                    "state": state,
                    "tcp_pose_m_rad": tcp_pose,
                },
            },
        }
        payload = {
            "schema_version": "1.0",
            "request_id": request_id,
            "trace_id": context.run_id,
            "episode_id": context.run_id,
            "task_id": model_task_id,
            "arm_id": arm_id,
            "subtask_id": model_subtask_id,
            "step_id": context.step_id,
            "observation_id": observation.observation_id,
            "deadline_ms": context.timeout_ms,
            "executor": self.descriptor.name,
            "checkpoint_sha": self.descriptor.checkpoint_sha,
            "norm_stats_sha": self.descriptor.norm_stats_sha,
            "expected_action_contract": ACTION_CONTRACT_VERSION,
            "model_input": model_input,
        }
        try:
            response = self.transport.request("/v1/infer", payload, context.timeout_ms)
        except TimeoutError as exc:
            raise ExecutorError(
                FailureCode.EXECUTOR_TIMEOUT, "π0.5 inference timed out"
            ) from exc
        except ExecutorError:
            raise
        except Exception as exc:
            raise ExecutorError(
                FailureCode.EXECUTOR_UNAVAILABLE, f"π0.5 transport failed: {exc}"
            ) from exc
        if not isinstance(response, Mapping):
            raise ExecutorError(
                FailureCode.EXECUTOR_BAD_RESPONSE,
                "π0.5 response must be an object",
            )
        _validate_response_envelope(
            response,
            request_id=request_id,
            context=context,
            task=task,
            expected_task_id=model_task_id,
            expected_subtask_id=model_subtask_id,
            observation=observation,
            descriptor=self.descriptor,
        )
        return _parse_canonical_chunk(
            response,
            task=replace(task, task_id=model_task_id),
            descriptor=self.descriptor,
        )

    def cancel(self, task_id: str, reason: str) -> None:
        trace_id, subtask_id = self._cancel_context_by_task.get(
            task_id, (task_id, task_id)
        )
        try:
            self.transport.request(
                "/v1/cancel",
                {
                    "schema_version": "1.0",
                    "request_id": str(uuid4()),
                    "trace_id": trace_id,
                    "episode_id": trace_id,
                    "task_id": task_id,
                    "subtask_id": subtask_id,
                    "reason": reason,
                },
                1_000,
            )
        except Exception:
            return


def build_executors_from_config(
    config: Mapping[str, Any],
    transport_factory: Callable[[str, str], ProcessTransport],
) -> tuple[Executor, ...]:
    """Build both process adapters while binding each configured deployment URL.

    The transport factory receives ``(executor_name, base_url)``. This keeps
    deployment-specific HTTP/WebSocket code outside the supervisor while ensuring
    that ``config.executors.*.base_url`` is actually consumed. Supported task
    types are frozen adapter capabilities, not deployment configuration.
    """

    raw_executors = config.get("executors")
    if not isinstance(raw_executors, Mapping):
        raise ValueError("executors config must be an object")

    adapter_types = {PI05_EXECUTOR_NAME: Pi05Adapter}
    built: list[Executor] = []
    unknown_names = set(raw_executors) - set(adapter_types)
    if unknown_names:
        raise ValueError(
            f"config.executors contains unsupported executors: {sorted(unknown_names)}"
        )
    for name, raw in raw_executors.items():
        adapter_type = adapter_types[name]
        if not isinstance(raw, Mapping):
            raise ValueError(f"config.executors.{name} must be an object")
        if set(raw) != EXECUTOR_CONFIG_FIELDS:
            raise ValueError(
                f"config.executors.{name} must contain exactly "
                f"{sorted(EXECUTOR_CONFIG_FIELDS)}; task_types are frozen in code"
            )
        enabled = raw.get("enabled")
        if not isinstance(enabled, bool):
            raise ValueError(f"config.executors.{name}.enabled must be a boolean")
        if not enabled:
            continue

        base_url = raw.get("base_url")
        if not isinstance(base_url, str) or not base_url.startswith(
            ("http://", "https://")
        ):
            raise ValueError(f"config.executors.{name}.base_url must be an HTTP(S) URL")

        artifact_values: dict[str, str] = {}
        for field in ("checkpoint_sha", "norm_stats_sha"):
            value = raw.get(field)
            if not is_pinned_artifact_digest(value):
                raise ValueError(
                    f"config.executors.{name}.{field} must match "
                    "'sha256:<64 hexadecimal characters>'"
                )
            artifact_values[field] = value

        transport = transport_factory(name, base_url)
        built.append(
            adapter_type(
                transport,
                checkpoint_sha=artifact_values["checkpoint_sha"],
                norm_stats_sha=artifact_values["norm_stats_sha"],
            )
        )
    if not built:
        raise ValueError("at least one executor must be explicitly enabled")
    return tuple(built)


class ExecutorRegistry:
    """Registry for the single lifecycle-assigned π0.5 service."""

    def __init__(
        self,
        executors: Sequence[Executor],
    ):
        names = [item.descriptor.name for item in executors]
        if len(names) != len(set(names)):
            raise ValueError("executor names must be unique")
        self._executors = {item.descriptor.name: item for item in executors}

    def select_exact(self, executor_name: str, task: TaskSchema) -> Executor:
        """Return one lifecycle-assigned executor without routing or fallback."""

        executor = self._executors.get(executor_name)
        if executor is None:
            raise ExecutorError(
                FailureCode.NO_COMPATIBLE_EXECUTOR,
                f"required lifecycle executor is unavailable: {executor_name!r}",
            )
        if task.task_type not in executor.descriptor.task_types:
            raise ExecutorError(
                FailureCode.NO_COMPATIBLE_EXECUTOR,
                f"required executor {executor_name!r} does not support "
                f"{task.task_type!r}",
            )
        if not executor.health():
            raise ExecutorError(
                FailureCode.EXECUTOR_UNAVAILABLE,
                f"required lifecycle executor is unhealthy: {executor_name!r}",
            )
        return executor
