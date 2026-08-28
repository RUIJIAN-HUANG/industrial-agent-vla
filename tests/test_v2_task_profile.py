from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_agent.v2_task_profile import (
    V2_FORMAL_TASK_IDS,
    V2_PROFILE_ID,
    V2_SCENE_ID,
    require_formal_v2_task,
    v2_task_for_instruction,
)


ROOT = Path(__file__).resolve().parents[1]


def test_formal_v2_catalog_contains_only_current_training_tasks() -> None:
    assert V2_PROFILE_ID == V2_SCENE_ID == "single_bin_manual_industrial_v2"
    assert V2_FORMAL_TASK_IDS == {"P01_TO_S11", "W01_TO_S14"}
    assert require_formal_v2_task("P01_TO_S11").target_slot == "S11"
    assert require_formal_v2_task("W01_TO_S14").target_slot == "S14"


def test_v2_instruction_resolves_to_supervisor_task_id() -> None:
    assert v2_task_for_instruction("把W01放到S14中").task_id == "W01_TO_S14"


def test_ui_only_task_cannot_enter_formal_v2_pipeline() -> None:
    with pytest.raises(ValueError, match="has no formal data contract"):
        require_formal_v2_task("PACK_ALL_AND_FINISH")


def test_v2_json_catalog_matches_python_catalog() -> None:
    path = ROOT / "configs" / "v2-task-profile.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["profile_id"] == V2_PROFILE_ID
    assert set(payload["formal_task_ids"]) == set(V2_FORMAL_TASK_IDS)
    assert {item["task_id"] for item in payload["tasks"]} == {
        "P01_TO_S11",
        "W01_TO_S14",
        "P03_UPRIGHT_TO_S12",
        "BIN01_TO_FINISHED01",
        "PACK_ALL_AND_FINISH",
    }
