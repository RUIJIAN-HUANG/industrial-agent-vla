from __future__ import annotations

import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

SERVICE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SERVICE_ROOT / "src"))

from openvla_oft.config import load_config  # noqa: E402
from openvla_oft.routes import OpenVLAOFTService  # noqa: E402


@pytest.fixture
def config() -> dict[str, Any]:
    return load_config(SERVICE_ROOT / "configs")


@pytest.fixture
def service(config: dict[str, Any]) -> OpenVLAOFTService:
    return OpenVLAOFTService(config)


@pytest.fixture
def valid_infer_request(config: dict[str, Any]) -> dict[str, Any]:
    full_digest = "a" * 64
    wrist_digest = "b" * 64
    checkpoint_sha = config["artifacts"]["checkpoint_sha"]
    norm_stats_sha = config["artifacts"]["norm_stats_sha"]
    width, height = config["image_size"]
    return {
        "schema_version": "1.0",
        "request_id": "req-1",
        "trace_id": "run-1",
        "episode_id": "run-1",
        "task_id": "episode-0001:S02_ARM_B_TRANSPORT",
        "subtask_id": "S02_ARM_B_TRANSPORT",
        "step_id": 0,
        "observation_id": "obs-1",
        "deadline_ms": 5000,
        "executor": "openvla_oft",
        "checkpoint_sha": checkpoint_sha,
        "norm_stats_sha": norm_stats_sha,
        "expected_action_contract": "1.0",
        "model_input": {
            "task_description": config["instruction"],
            "full_image": {
                "uri": f"cas://sha256/{full_digest}",
                "image_sha256": f"sha256:{full_digest}",
                "camera_id": "CAM_B_TOP",
                "width": width,
                "height": height,
            },
            "wrist_image": {
                "uri": f"cas://sha256/{wrist_digest}",
                "image_sha256": f"sha256:{wrist_digest}",
                "camera_id": "CAM_B_WRIST",
                "width": width,
                "height": height,
            },
            "state": [0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 0.5],
        },
    }


@pytest.fixture
def schema_validator() -> Any:
    loaded = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((REPO_ROOT / "schemas").glob("*.schema.json"))
    ]
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in loaded
    )

    def build(schema_name: str, definition: str | None = None) -> Draft202012Validator:
        schema = next(
            item for item in loaded if item["$id"].endswith(f"/{schema_name}")
        )
        selected = schema if definition is None else schema["$defs"][definition]
        return Draft202012Validator(selected, registry=registry)

    return build


@pytest.fixture
def copy_request() -> Any:
    def _copy(value: dict[str, Any]) -> dict[str, Any]:
        return deepcopy(value)

    return _copy
