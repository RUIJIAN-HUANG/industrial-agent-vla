from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest
from jsonschema import Draft202012Validator, ValidationError


REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_PATH = REPO_ROOT / "schemas" / "canonical-episode-v2.schema.json"
V1_MANIFEST_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "golden_episode_v1" / "structure.json"
)


def _schema() -> dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def _valid_v2_manifest() -> dict[str, Any]:
    manifest = json.loads(V1_MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["canonical_schema_version"] = "2.0"
    del manifest["schema_version"]
    manifest["metadata"].update(
        {
            "scene_id": "single_bin_manual_industrial_v2",
            "task_id": "P01_TO_S11",
            "instruction": "把P01放到S11中",
            "padding_policy": {"strategy": "none", "target_length": None},
        }
    )
    return manifest


def test_canonical_episode_v2_schema_accepts_frozen_manifest() -> None:
    schema = _schema()
    Draft202012Validator.check_schema(schema)
    Draft202012Validator(schema).validate(_valid_v2_manifest())


@pytest.mark.parametrize(
    ("stream", "arm_id", "field", "value"),
    [
        ("robot_state", "Arm_A", "dtype", "float64"),
        ("robot_state", "Arm_A", "shape", [6, 8]),
        ("robot_state", "Arm_B", "shape", [6, 6]),
        ("actions", None, "dtype", "float64"),
        ("actions", None, "shape", [1, 8]),
    ],
)
def test_canonical_episode_v2_schema_rejects_non_7d_float32_vectors(
    stream: str,
    arm_id: str | None,
    field: str,
    value: Any,
) -> None:
    manifest = copy.deepcopy(_valid_v2_manifest())
    if stream == "robot_state":
        descriptor = manifest["streams"][stream][arm_id]["datasets"]["state_7d"]
    else:
        descriptor = manifest["streams"][stream]["datasets"]["action_7d"]
    descriptor[field] = value

    with pytest.raises(ValidationError):
        Draft202012Validator(_schema()).validate(manifest)


@pytest.mark.parametrize(
    ("stream", "arm_id", "wrong_path"),
    [
        ("robot_state", "Arm_A", "/robot_state/Arm_B/state_7d"),
        ("robot_state", "Arm_B", "/robot_state/Arm_A/state_7d"),
        ("actions", None, "/robot_state/Arm_A/state_7d"),
    ],
)
def test_canonical_episode_v2_schema_rejects_cross_stream_vector_paths(
    stream: str,
    arm_id: str | None,
    wrong_path: str,
) -> None:
    manifest = copy.deepcopy(_valid_v2_manifest())
    if stream == "robot_state":
        descriptor = manifest["streams"][stream][arm_id]["datasets"]["state_7d"]
    else:
        descriptor = manifest["streams"][stream]["datasets"]["action_7d"]
    descriptor["path"] = wrong_path

    with pytest.raises(ValidationError):
        Draft202012Validator(_schema()).validate(manifest)
