"""Request validation and response builders for executor contracts."""

from __future__ import annotations

from time import time
from typing import Any, Mapping

from .config import looks_like_sha256
from .exceptions import ServiceError
from .utils import finite_vector

SCHEMA_VERSION = "1.0"
SERVICE_NAME = "openvla_oft"
ACTION_CONTRACT_VERSION = "1.0"
ACTION_SPACE = "ee_delta_pose_gripper"
FRAME = "robot_base"
TRANSLATION_UNIT = "m"
ROTATION_UNIT = "rad"
GRIPPER_UNIT = "normalized"

INFER_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "request_id",
        "trace_id",
        "episode_id",
        "task_id",
        "subtask_id",
        "step_id",
        "observation_id",
        "deadline_ms",
        "executor",
        "checkpoint_sha",
        "norm_stats_sha",
        "expected_action_contract",
        "model_input",
    }
)
OPENVLA_MODEL_INPUT_KEYS = frozenset(
    {"task_description", "full_image", "wrist_image", "state"}
)
CANCEL_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "request_id",
        "trace_id",
        "episode_id",
        "task_id",
        "subtask_id",
        "reason",
    }
)


def now_ms() -> int:
    return int(time() * 1000)


def build_health_response(
    config: Mapping[str, Any],
    *,
    uptime_ms: int,
    status: str = "ready",
) -> dict[str, Any]:
    artifacts = config["artifacts"]
    return {
        "schema_version": SCHEMA_VERSION,
        "service": SERVICE_NAME,
        "service_version": str(config.get("service_version", "0.1.0")),
        "status": status,
        "uptime_ms": uptime_ms,
        "checkpoint_sha": artifacts["checkpoint_sha"],
        "norm_stats_sha": artifacts["norm_stats_sha"],
        "supported_task_types": list(config.get("supported_task_types", [])),
        "supported_action_contracts": [ACTION_CONTRACT_VERSION],
        "queue": {"max_concurrent_requests": config["api"]["max_concurrent_requests"]},
        "device": {
            "mode": "mock" if config.get("mock_mode", True) else "real",
            "target": config["model"].get("device", "cuda"),
        },
        "time_ms": now_ms(),
    }


def validate_infer_request(payload: Any, config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ServiceError(
            "EXEC_2103_BAD_RESPONSE",
            "request body must be a JSON object",
        )
    unknown = set(payload) - INFER_REQUEST_KEYS
    missing = INFER_REQUEST_KEYS - set(payload)
    if unknown or missing:
        raise ServiceError(
            "EXEC_2103_BAD_RESPONSE",
            "infer fields invalid; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}",
        )
    _validate_common_envelope(payload, config)
    if payload["executor"] != SERVICE_NAME:
        raise ServiceError(
            "ROUTE_2001_NO_COMPATIBLE_EXECUTOR",
            "executor must be openvla_oft",
        )
    if payload["subtask_id"] != config["required_subtask_id"]:
        raise ServiceError(
            "SAFE_9004_ACTION_REJECTED",
            "OpenVLA-OFT may infer only after the S02 Arm_B transport "
            "subtask is active",
        )
    if payload["expected_action_contract"] != ACTION_CONTRACT_VERSION:
        raise ServiceError(
            "ACT_1201_CONTRACT_INVALID",
            "expected_action_contract must be 1.0",
        )
    deadline_ms = payload["deadline_ms"]
    if (
        isinstance(deadline_ms, bool)
        or not isinstance(deadline_ms, int)
        or deadline_ms < 1
    ):
        raise ServiceError(
            "EXEC_2103_BAD_RESPONSE",
            "deadline_ms must be a positive integer",
        )
    if deadline_ms > int(config["api"]["max_deadline_ms"]):
        raise ServiceError(
            "EXEC_2106_BACKPRESSURE",
            f"deadline_ms exceeds service maximum {config['api']['max_deadline_ms']}",
            retryable=True,
            retry_after_ms=int(config["api"]["default_deadline_ms"]),
        )
    model_input = payload["model_input"]
    if not isinstance(model_input, Mapping):
        raise ServiceError("EXEC_2103_BAD_RESPONSE", "model_input must be an object")
    unknown_model = set(model_input) - OPENVLA_MODEL_INPUT_KEYS
    missing_model = OPENVLA_MODEL_INPUT_KEYS - set(model_input)
    if unknown_model or missing_model:
        raise ServiceError(
            "EXEC_2103_BAD_RESPONSE",
            f"model_input fields invalid; missing={sorted(missing_model)}, "
            f"unknown={sorted(unknown_model)}",
        )
    if model_input["task_description"] != config["instruction"]:
        raise ServiceError(
            "TASK_1001_INVALID",
            "task_description must match the frozen Arm_B downstream instruction",
        )
    full_image = model_input["full_image"]
    if not isinstance(full_image, Mapping):
        raise ServiceError(
            "CAS_1304_METADATA_MISMATCH",
            "model_input.full_image must be an ImageReference object",
            retryable=False,
        )
    if model_input["wrist_image"] is not None:
        raise ServiceError(
            "CAS_1304_METADATA_MISMATCH",
            "frozen three-camera profile requires model_input.wrist_image=null",
            retryable=False,
        )
    state = finite_vector(model_input["state"], "model_input.state", min_length=7)
    return {
        **dict(payload),
        "model_input": {
            "task_description": model_input["task_description"],
            "full_image": dict(full_image),
            "wrist_image": None,
            "state": state,
        },
    }


def validate_cancel_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise ServiceError(
            "EXEC_2103_BAD_RESPONSE",
            "request body must be a JSON object",
        )
    unknown = set(payload) - CANCEL_REQUEST_KEYS
    missing = CANCEL_REQUEST_KEYS - set(payload)
    if unknown or missing:
        raise ServiceError(
            "EXEC_2103_BAD_RESPONSE",
            "cancel fields invalid; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}",
        )
    for key in CANCEL_REQUEST_KEYS:
        if key == "schema_version":
            continue
        if not isinstance(payload[key], str) or not payload[key]:
            raise ServiceError(
                "EXEC_2103_BAD_RESPONSE",
                f"{key} must be a non-empty string",
            )
    if not _compatible_version(payload["schema_version"]):
        raise ServiceError(
            "TASK_1002_UNSUPPORTED_VERSION",
            "schema_version must be 1.x",
        )
    return dict(payload)


def build_success_response(
    request: Mapping[str, Any],
    actions: list[list[float]],
    *,
    checkpoint_sha: str,
    norm_stats_sha: str,
    inference_ms: float,
) -> dict[str, Any]:
    return {
        **_response_envelope(request, checkpoint_sha, norm_stats_sha),
        "status": "ok",
        "action_chunk": {
            "contract_version": ACTION_CONTRACT_VERSION,
            "chunk_id": (
                f"openvla-oft:{request['trace_id']}:"
                f"{request['observation_id']}:{request['step_id']}"
            ),
            "task_id": request["task_id"],
            "executor": SERVICE_NAME,
            "action_space": ACTION_SPACE,
            "frame": FRAME,
            "translation_unit": TRANSLATION_UNIT,
            "rotation_unit": ROTATION_UNIT,
            "gripper_unit": GRIPPER_UNIT,
            "steps": [{"values": row, "duration_ms": 100} for row in actions],
        },
        "timing": {
            "queue_ms": 0.0,
            "inference_ms": round(inference_ms, 3),
            "total_ms": round(inference_ms, 3),
        },
    }


def build_error_response(
    request: Mapping[str, Any],
    error: ServiceError,
    *,
    checkpoint_sha: str,
    norm_stats_sha: str,
    status: str = "error",
) -> dict[str, Any]:
    response = {
        **_response_envelope(request, checkpoint_sha, norm_stats_sha),
        "status": status,
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
        },
    }
    if error.retry_after_ms is not None:
        response["error"]["retry_after_ms"] = error.retry_after_ms
    return response


def build_cancel_response(
    request: Mapping[str, Any],
    *,
    status: str,
    cancelled_request_ids: list[str],
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": request["request_id"],
        "trace_id": request["trace_id"],
        "task_id": request["task_id"],
        "status": status,
        "cancelled_request_ids": cancelled_request_ids,
        "server_context_cleared": True,
    }


def _response_envelope(
    request: Mapping[str, Any],
    checkpoint_sha: str,
    norm_stats_sha: str,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": str(request.get("request_id", "")),
        "trace_id": str(request.get("trace_id", "")),
        "episode_id": str(request.get("episode_id", "")),
        "task_id": str(request.get("task_id", "")),
        "subtask_id": str(request.get("subtask_id", "")),
        "step_id": (
            int(request.get("step_id", 0))
            if isinstance(request.get("step_id"), int)
            else 0
        ),
        "observation_id": str(request.get("observation_id", "")),
        "executor": SERVICE_NAME,
        "checkpoint_sha": checkpoint_sha,
        "norm_stats_sha": norm_stats_sha,
    }


def _validate_common_envelope(
    payload: Mapping[str, Any],
    config: Mapping[str, Any],
) -> None:
    for key in (
        "request_id",
        "trace_id",
        "episode_id",
        "task_id",
        "subtask_id",
        "observation_id",
    ):
        if not isinstance(payload[key], str) or not payload[key]:
            raise ServiceError(
                "EXEC_2103_BAD_RESPONSE",
                f"{key} must be a non-empty string",
            )
    if not _compatible_version(payload["schema_version"]):
        raise ServiceError(
            "TASK_1002_UNSUPPORTED_VERSION",
            "schema_version must be 1.x",
        )
    step_id = payload["step_id"]
    if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
        raise ServiceError(
            "EXEC_2103_BAD_RESPONSE",
            "step_id must be a non-negative integer",
        )
    artifacts = config["artifacts"]
    for field in ("checkpoint_sha", "norm_stats_sha"):
        if not looks_like_sha256(payload[field]):
            raise ServiceError(
                "EXEC_2105_MODEL_REVISION_MISMATCH",
                f"{field} is not pinned",
            )
        expected = artifacts[field]
        if payload[field].casefold() != expected.casefold():
            raise ServiceError(
                "EXEC_2105_MODEL_REVISION_MISMATCH",
                f"{field} does not match deployed OpenVLA-OFT artifact",
            )


def _compatible_version(value: Any) -> bool:
    return isinstance(value, str) and value.startswith("1.") and value[2:].isdigit()
