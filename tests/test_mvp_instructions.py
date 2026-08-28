from __future__ import annotations

import pytest

from industrial_agent.instructions import (
    MVP_BIN01_TO_FINISHED01,
    MVP_P01_TO_S11,
    MVP_P03_UPRIGHT_TO_S12,
    MVP_PACK_ALL_AND_FINISH,
    MVP_W01_TO_S14,
    mvp_instruction_for_task,
    mvp_instruction_options,
    mvp_task_id_for_instruction,
    normalize_mvp_instruction,
)


def test_mvp_exposes_the_human_facing_option() -> None:
    assert mvp_instruction_options() == (
        MVP_P01_TO_S11,
        MVP_W01_TO_S14,
        MVP_P03_UPRIGHT_TO_S12,
        MVP_BIN01_TO_FINISHED01,
        MVP_PACK_ALL_AND_FINISH,
    )
    assert [option.task_id for option in mvp_instruction_options()] == [
        "P01_TO_S11",
        "W01_TO_S14",
        "P03_UPRIGHT_TO_S12",
        "BIN01_TO_FINISHED01",
        "PACK_ALL_AND_FINISH",
    ]


@pytest.mark.parametrize(
    "text",
    [MVP_P01_TO_S11.display_instruction, MVP_P01_TO_S11.canonical_instruction],
)
def test_display_and_canonical_text_resolve_to_one_training_task(text: str) -> None:
    option = normalize_mvp_instruction(text)
    assert option.task_id == "P01_TO_S11"
    assert option.canonical_instruction == "把P01放到S11中"


def test_task_id_resolves_to_same_option() -> None:
    assert mvp_instruction_for_task("P01_TO_S11") is MVP_P01_TO_S11
    assert mvp_instruction_for_task("W01_TO_S14") is MVP_W01_TO_S14
    assert mvp_instruction_for_task("BIN01_TO_FINISHED01") is MVP_BIN01_TO_FINISHED01


def test_user_instruction_resolves_to_task_id_for_supervisor() -> None:
    assert mvp_task_id_for_instruction("把P01放到S11中") == "P01_TO_S11"


@pytest.mark.parametrize(
    ("task_id", "instruction"),
    [
        ("W01_TO_S14", "把W01放到S14中"),
        ("P03_UPRIGHT_TO_S12", "请将倒立的轴件 P03 翻正后，放置到料箱的 S12 格子中。"),
        ("BIN01_TO_FINISHED01", "把Bin_01搬到FINISHED_01"),
        (
            "PACK_ALL_AND_FINISH",
            "请将所有零件按指定位置装入料箱 Bin_01，再将料箱 Bin_01 搬运到成品区 FINISHED_01。",
        ),
    ],
)
def test_all_new_frozen_instructions_resolve_exactly(
    task_id: str, instruction: str
) -> None:
    option = mvp_instruction_for_task(task_id)
    assert option.canonical_instruction == instruction
    assert normalize_mvp_instruction(instruction) is option


def test_unknown_wording_is_rejected_without_fuzzy_matching() -> None:
    with pytest.raises(ValueError, match="unknown MVP instruction"):
        normalize_mvp_instruction("把轴件P01放进S11")
