from __future__ import annotations

import json
from pathlib import Path

from jsonschema import Draft202012Validator, RefResolver

REPO_ROOT = Path(__file__).resolve().parents[3]
SCHEMA_DIR = REPO_ROOT / "schemas"


def _load_schema(name: str) -> dict[str, object]:
    return json.loads((SCHEMA_DIR / name).read_text(encoding="utf-8"))


def _definition_schema(filename: str, definition: str) -> dict[str, object]:
    schema = _load_schema(filename)
    return {
        "$schema": schema["$schema"],
        "$defs": schema["$defs"],
        "$ref": f"#/$defs/{definition}",
    }


def _validator(schema: dict[str, object]) -> Draft202012Validator:
    store = {
        (SCHEMA_DIR / "action-chunk.schema.json").as_uri(): _load_schema(
            "action-chunk.schema.json"
        )
    }
    return Draft202012Validator(
        schema,
        resolver=RefResolver(
            base_uri=SCHEMA_DIR.as_uri() + "/",
            referrer=schema,
            store=store,
        ),
    )


def test_infer_request_matches_official_schema(valid_infer_request):
    schema = _definition_schema("executor-infer.schema.json", "request")

    _validator(schema).validate(valid_infer_request)


def test_infer_response_matches_official_schema(service, valid_infer_request):
    schema = _definition_schema("executor-infer.schema.json", "response")

    _, response = service.infer(valid_infer_request)
    _validator(schema).validate(response)


def test_health_response_matches_official_schema(service):
    schema = _load_schema("executor-health.schema.json")

    _, response = service.health()
    Draft202012Validator(schema).validate(response)


def test_cancel_contract_matches_official_schema(service):
    schema = _definition_schema("executor-cancel.schema.json", "response")
    request_schema = _definition_schema("executor-cancel.schema.json", "request")
    request = {
        "schema_version": "1.0",
        "request_id": "cancel-1",
        "trace_id": "run-1",
        "episode_id": "run-1",
        "task_id": "episode-0001:S02_ARM_B_TRANSPORT",
        "subtask_id": "S02_ARM_B_TRANSPORT",
        "reason": "operator requested cancellation",
    }

    _validator(request_schema).validate(request)
    _, response = service.cancel(request)
    _validator(schema).validate(response)
