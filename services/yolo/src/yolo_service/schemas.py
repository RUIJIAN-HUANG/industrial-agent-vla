"""Validation helpers for YOLO HTTP contracts."""

from __future__ import annotations

from typing import Any, Mapping

from .exceptions import ServiceError

SCHEMA_VERSION = "1.0"
DETECTION_CONTRACT_VERSION = "1.0"

DETECT_REQUEST_KEYS = frozenset(
    {
        "schema_version",
        "request_id",
        "trace_id",
        "episode_id",
        "task_id",
        "subtask_id",
        "step_id",
        "observation_id",
        "image_sha256",
        "deadline_ms",
        "detector",
        "checkpoint_sha",
        "class_map_sha",
        "config_sha",
        "expected_detection_contract",
        "image",
        "allowed_class_names",
        "thresholds",
    }
)


def validate_detect_request(
    payload: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one /v1/detect request before inference."""

    if not isinstance(payload, Mapping):
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "request body must be a JSON object",
        )

    missing = DETECT_REQUEST_KEYS - set(payload)
    unknown = set(payload) - DETECT_REQUEST_KEYS
    if missing or unknown:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            f"detect fields invalid; missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}",
        )

    for field in (
        "request_id",
        "trace_id",
        "episode_id",
        "task_id",
        "subtask_id",
        "observation_id",
    ):
        if not isinstance(payload[field], str) or not payload[field]:
            raise ServiceError(
                "PERC_2203_BAD_RESPONSE",
                f"{field} must be a non-empty string",
            )

    if payload["schema_version"] != SCHEMA_VERSION:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "schema_version must be '1.0'",
        )
    if payload["detector"] != "yolo":
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "detector must be 'yolo'",
        )
    if payload["expected_detection_contract"] != DETECTION_CONTRACT_VERSION:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "expected_detection_contract must be '1.0'",
        )

    step_id = payload["step_id"]
    if isinstance(step_id, bool) or not isinstance(step_id, int) or step_id < 0:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "step_id must be a non-negative integer",
        )

    deadline_ms = payload["deadline_ms"]
    maximum = config["api"]["max_deadline_ms"]
    if (
        isinstance(deadline_ms, bool)
        or not isinstance(deadline_ms, int)
        or not 1 <= deadline_ms <= maximum
    ):
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            f"deadline_ms must be between 1 and {maximum}",
        )

    for field in ("checkpoint_sha", "class_map_sha", "config_sha"):
        value = payload[field]
        if not isinstance(value, str) or value.casefold() != config[field].casefold():
            raise ServiceError(
                "PERC_2205_REVISION_MISMATCH",
                f"{field} does not match the deployed YOLO service",
            )

    allowed = payload["allowed_class_names"]
    if (
        not isinstance(allowed, list)
        or any(not isinstance(item, str) or not item for item in allowed)
        or len(set(allowed)) != len(allowed)
    ):
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "allowed_class_names must contain unique non-empty strings",
        )

    thresholds = payload["thresholds"]
    if not isinstance(thresholds, Mapping):
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "thresholds must be an object",
        )
    if set(thresholds) != {"confidence", "iou"}:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "thresholds must contain only confidence and iou",
        )
    for field in ("confidence", "iou"):
        value = thresholds[field]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ServiceError(
                "PERC_2203_BAD_RESPONSE",
                f"thresholds.{field} must be a number",
            )
        if not 0 <= value <= 1:
            raise ServiceError(
                "PERC_2203_BAD_RESPONSE",
                f"thresholds.{field} must be between 0 and 1",
            )

    return dict(payload)


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


def validate_cancel_request(payload: Any) -> dict[str, Any]:
    if not isinstance(payload, Mapping) or set(payload) != CANCEL_REQUEST_KEYS:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "cancel request fields are invalid",
        )
    if payload["schema_version"] != SCHEMA_VERSION:
        raise ServiceError(
            "PERC_2203_BAD_RESPONSE",
            "schema_version must be '1.0'",
        )
    for field in CANCEL_REQUEST_KEYS - {"schema_version"}:
        if not isinstance(payload[field], str) or not payload[field]:
            raise ServiceError(
                "PERC_2203_BAD_RESPONSE",
                f"{field} must be a non-empty string",
            )
    return dict(payload)


def build_health_response(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "service": "yolo",
        "service_version": str(config["service_version"]),
        "status": "ready",
        "checkpoint_sha": config["checkpoint_sha"],
        "class_map_sha": config["class_map_sha"],
        "config_sha": config["config_sha"],
        "supported_task_types": list(config["supported_task_types"]),
        "supported_detection_contracts": [DETECTION_CONTRACT_VERSION],
        "queue": {
            "max_concurrent_requests": config["api"]["max_concurrent_requests"]
        },
        "device": {
            "mode": "mock" if config["mock_mode"] else "real",
            "target": config["model"]["device"],
        },
    }


def build_success_response(
    request: Mapping[str, Any],
    detections: list[Mapping[str, Any]],
    timing: Mapping[str, float],
) -> dict[str, Any]:
    image = request["image_reference"]
    packet = {
        "schema_version": SCHEMA_VERSION,
        "detection_contract_version": DETECTION_CONTRACT_VERSION,
        "packet_id": (
            f"yolo:{request['trace_id']}:{request['observation_id']}:"
            f"{request['step_id']}"
        ),
        "request_id": request["request_id"],
        "trace_id": request["trace_id"],
        "episode_id": request["episode_id"],
        "task_id": request["task_id"],
        "subtask_id": request["subtask_id"],
        "step_id": request["step_id"],
        "observation_id": request["observation_id"],
        "image_sha256": request["image_sha256"],
        "camera_id": image["camera_id"],
        "image_width": image["width"],
        "image_height": image["height"],
        "checkpoint_sha": request["checkpoint_sha"],
        "class_map_sha": request["class_map_sha"],
        "config_sha": request["config_sha"],
        "detections": detections,
        "timing": dict(timing),
    }
    return {
        **_response_envelope(request),
        "status": "ok",
        "detection_packet": packet,
    }


def build_error_response(
    request: Mapping[str, Any],
    error: ServiceError,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        **_response_envelope(request, config=config),
        "status": "error",
        "error": {
            "code": error.code,
            "message": error.message,
            "retryable": error.retryable,
        },
    }


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
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = config or request
    step_id = request.get("step_id", 0)
    return {
        "schema_version": SCHEMA_VERSION,
        "request_id": str(request.get("request_id", "")),
        "trace_id": str(request.get("trace_id", "")),
        "episode_id": str(request.get("episode_id", "")),
        "task_id": str(request.get("task_id", "")),
        "subtask_id": str(request.get("subtask_id", "")),
        "step_id": step_id if isinstance(step_id, int) else 0,
        "observation_id": str(request.get("observation_id", "")),
        "image_sha256": str(request.get("image_sha256", ZERO_DIGEST)),
        "detector": "yolo",
        "checkpoint_sha": str(source.get("checkpoint_sha", ZERO_DIGEST)),
        "class_map_sha": str(source.get("class_map_sha", ZERO_DIGEST)),
        "config_sha": str(source.get("config_sha", ZERO_DIGEST)),
    }


ZERO_DIGEST = f"sha256:{'0' * 64}"
