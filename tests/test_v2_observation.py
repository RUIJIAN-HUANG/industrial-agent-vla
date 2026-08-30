from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from referencing import Registry, Resource

from industrial_agent.errors import FailureCode, ObservationError
from industrial_agent.v2_observation import V2ObservationGateway


def _image(camera_id: str, digest: str = "a") -> dict[str, object]:
    return {
        "uri": f"cas://sha256/{digest * 64}",
        "image_sha256": f"sha256:{digest * 64}",
        "camera_id": camera_id,
        "width": 1280,
        "height": 720,
    }


def v2_observation(*, observation_id: str = "v2-obs-1", terminal: bool = False):
    return {
        "observation_version": "2.0",
        "observation_id": observation_id,
        "timestamp_ms": 1,
        "camera": {
            "full_image": _image("CAM_A_TOP"),
            "arm_a_rgb": _image("CAM_A_TOP"),
            "handoff_rgb": _image("CAM_HANDOFF", "b"),
            "arm_b_rgb": _image("CAM_B_TOP", "c"),
            "wrist_image": None,
        },
        "objects": [],
        "robot": {
            "active_arm": "Arm_A" if not terminal else "NONE",
            "arm_a": {
                "tcp_pose_m_rad": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0],
                "state": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 1.0],
                "retreated": terminal,
                "gripper_open": True,
                "stationary": True,
            },
            "arm_b": {
                "tcp_pose_m_rad": [0.4, 0.4, 0.5, 0.0, 0.0, 0.0],
                "state": [0.4, 0.4, 0.5, 0.0, 0.0, 0.0, 1.0],
                "retreated": True,
                "gripper_open": True,
                "stationary": True,
            },
        },
        "safety": {
            "emergency_stop": False,
            "protective_stop": False,
            "system_fault": None,
        },
        "task": {
            "task_id": "P01_TO_S11",
            "target_object_id": "P01",
            "target_slot_id": "S11",
            "status": "SUCCEEDED" if terminal else "ACTIVE",
            "terminal": terminal,
            "terminal_confidence": 0.9 if terminal else 0.0,
            "verification_votes": 2 if terminal else 0,
        },
        "quality": {"confidence": 1.0},
    }


def test_v2_gateway_accepts_formal_observation() -> None:
    result = V2ObservationGateway().ingest_online(v2_observation())
    assert result.observation_version == "2.0"
    assert result.data["task"]["task_id"] == "P01_TO_S11"


def test_v2_observation_matches_machine_schema() -> None:
    root = Path(__file__).resolve().parents[1]
    schemas = [
        json.loads((root / "schemas" / name).read_text(encoding="utf-8"))
        for name in (
            "online-observation-common.schema.json",
            "online-observation-v2.schema.json",
        )
    ]
    registry = Registry().with_resources(
        (schema["$id"], Resource.from_contents(schema)) for schema in schemas
    )
    Draft202012Validator(schemas[-1], registry=registry).validate(v2_observation())


def test_v2_gateway_rejects_v1_lifecycle_fields() -> None:
    raw = v2_observation()
    raw["task"]["packed_part_count"] = 4
    with pytest.raises(ObservationError) as caught:
        V2ObservationGateway().ingest_online(raw)
    assert caught.value.code is FailureCode.OBSERVATION_INVALID


def test_v2_gateway_rejects_unfrozen_task_target() -> None:
    raw = deepcopy(v2_observation())
    raw["task"]["target_slot_id"] = "S14"
    with pytest.raises(ObservationError, match="target_slot_id"):
        V2ObservationGateway().ingest_online(raw)


def test_v2_gateway_rejects_active_arm_b() -> None:
    raw = deepcopy(v2_observation())
    raw["robot"]["active_arm"] = "Arm_B"
    with pytest.raises(ObservationError, match="Arm_A or NONE"):
        V2ObservationGateway().ingest_online(raw)


def test_v2_gateway_accepts_bin_handoff_arm_b() -> None:
    raw = v2_observation()
    raw["robot"]["active_arm"] = "Arm_B"
    raw["robot"]["arm_a"]["retreated"] = True
    raw["robot"]["arm_a"]["stationary"] = True
    raw["robot"]["arm_b"]["retreated"] = False
    raw["robot"]["arm_b"]["stationary"] = False
    raw["task"] = {
        "task_id": "BIN01_TO_FINISHED01",
        "target_object_id": "Bin_01",
        "target_slot_id": None,
        "status": "ACTIVE",
        "terminal": False,
        "terminal_confidence": 0.0,
        "verification_votes": 0,
    }
    assert V2ObservationGateway().ingest_online(raw).data["task"]["task_id"] == (
        "BIN01_TO_FINISHED01"
    )
