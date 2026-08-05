from __future__ import annotations

from copy import deepcopy
from time import sleep
from typing import Any

from jsonschema import Draft202012Validator

from yolo_service.model import Detection, ModelOutput
from yolo_service.routes import YoloService


class SlowModel:
    def detect(self, *args: Any, **kwargs: Any) -> ModelOutput:
        del args, kwargs
        sleep(0.05)
        return ModelOutput(detections=(), nms_ms=1.5)


class InvalidClassModel:
    def detect(self, *args: Any, **kwargs: Any) -> ModelOutput:
        del args, kwargs
        return ModelOutput(
            detections=(
                Detection(
                    class_id=0,
                    class_name="not-the-configured-class",
                    confidence=0.9,
                    bbox_xyxy=(1.0, 1.0, 10.0, 10.0),
                ),
            ),
            nms_ms=2.5,
        )


class TimedModel:
    def detect(self, *args: Any, **kwargs: Any) -> ModelOutput:
        del args, kwargs
        return ModelOutput(detections=(), nms_ms=3.75)


def test_health_matches_contract(
    service: YoloService,
    schema_store: dict[str, Any],
) -> None:
    status, response = service.health()
    Draft202012Validator(schema_store["perception-health.schema.json"]).validate(
        response
    )
    assert status == 200
    assert response["service"] == "yolo"
    assert response["status"] == "ready"


def test_mock_detect_returns_legal_empty_packet(
    service: YoloService,
    valid_detect_request: dict[str, Any],
    schema_store: dict[str, Any],
) -> None:
    status, response = service.detect(valid_detect_request)
    detect_schema = schema_store["perception-detect.schema.json"]
    packet_schema = schema_store["detection-packet.schema.json"]
    response_schema = deepcopy(detect_schema["$defs"]["response"])
    response_schema["properties"]["detection_packet"] = packet_schema
    Draft202012Validator(response_schema).validate(response)
    assert status == 200
    assert response["detection_packet"]["detections"] == []


def test_detect_rejects_revision_mismatch(
    service: YoloService,
    valid_detect_request: dict[str, Any],
) -> None:
    request = deepcopy(valid_detect_request)
    request["checkpoint_sha"] = f"sha256:{'1' * 64}"
    status, response = service.detect(request)
    assert status == 422
    assert response["error"]["code"] == "PERC_2205_REVISION_MISMATCH"


def test_detect_rejects_missing_cas_blob(
    service: YoloService,
    valid_detect_request: dict[str, Any],
) -> None:
    request = deepcopy(valid_detect_request)
    missing = "2" * 64
    request["image_sha256"] = f"sha256:{missing}"
    request["image"]["image_sha256"] = f"sha256:{missing}"
    request["image"]["uri"] = f"cas://sha256/{missing}"
    status, response = service.detect(request)
    assert status == 422
    assert response["error"]["code"] == "CAS_1301_NOT_FOUND"


def test_detect_enforces_deadline(
    config: dict[str, Any],
    valid_detect_request: dict[str, Any],
    schema_store: dict[str, Any],
) -> None:
    service = YoloService(config, model=SlowModel())
    request = deepcopy(valid_detect_request)
    request["deadline_ms"] = 1
    status, response = service.detect(request)
    assert status == 504
    assert response["error"]["code"] == "PERC_2202_TIMEOUT"

    health_status, health = service.health()
    assert health_status == 200
    assert health["status"] == "degraded"
    assert health["device"]["quarantined"] is True
    Draft202012Validator(schema_store["perception-health.schema.json"]).validate(health)

    retry_status, retry = service.detect(valid_detect_request)
    assert retry_status == 503
    assert retry["error"]["code"] == "PERC_2201_UNAVAILABLE"
    assert retry["error"]["retryable"] is False

    cancel_status, _cancel = service.cancel(
        {
            "schema_version": "1.0",
            "request_id": "cancel-after-timeout",
            "trace_id": "run-1",
            "episode_id": "run-1",
            "task_id": "task-1",
            "subtask_id": "S01_ARM_A_PACK_HANDOFF",
            "reason": "verify quarantine persistence",
        }
    )
    assert cancel_status == 200
    still_quarantined_status, _still_quarantined = service.detect(valid_detect_request)
    assert still_quarantined_status == 503

    fresh_service = YoloService(config)
    fresh_status, fresh_health = fresh_service.health()
    assert fresh_status == 200
    assert fresh_health["status"] == "ready"
    fresh_service.close()
    service.close()


def test_detect_reports_backend_nms_timing(
    config: dict[str, Any],
    valid_detect_request: dict[str, Any],
) -> None:
    service = YoloService(config, model=TimedModel())

    status, response = service.detect(valid_detect_request)

    assert status == 200
    assert response["detection_packet"]["timing"]["nms_ms"] == 3.75
    service.close()


def test_detect_rejects_model_class_identity_mismatch(
    config: dict[str, Any],
    valid_detect_request: dict[str, Any],
) -> None:
    service = YoloService(config, model=InvalidClassModel())

    status, response = service.detect(valid_detect_request)

    assert status == 422
    assert response["error"]["code"] == "PERC_2203_BAD_RESPONSE"
