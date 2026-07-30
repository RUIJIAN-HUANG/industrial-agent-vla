from __future__ import annotations

from copy import deepcopy
from time import sleep
from typing import Any

from jsonschema import Draft202012Validator

from yolo_service.routes import YoloService


class SlowModel:
    def detect(self, *args: Any, **kwargs: Any) -> list[Any]:
        del args, kwargs
        sleep(0.05)
        return []


def test_health_matches_contract(
    service: YoloService,
    schema_store: dict[str, Any],
) -> None:
    status, response = service.health()
    Draft202012Validator(
        schema_store["perception-health.schema.json"]
    ).validate(response)
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
) -> None:
    service = YoloService(config, model=SlowModel())
    request = deepcopy(valid_detect_request)
    request["deadline_ms"] = 1
    status, response = service.detect(request)
    assert status == 504
    assert response["error"]["code"] == "PERC_2202_TIMEOUT"
