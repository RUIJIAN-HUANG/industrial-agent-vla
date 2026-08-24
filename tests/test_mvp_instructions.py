from __future__ import annotations

import pytest

from industrial_agent.instructions import (
    MVP_BIN01_TO_FINISHED01,
    MVP_P01_TO_S11,
    MVP_W01_TO_S14,
    mvp_instruction_for_task,
    mvp_instruction_options,
    normalize_mvp_instruction,
)


def test_mvp_exposes_the_human_facing_option() -> None:
    assert mvp_instruction_options() == (
        MVP_P01_TO_S11,
        MVP_W01_TO_S14,
        MVP_BIN01_TO_FINISHED01,
    )
    assert MVP_P01_TO_S11.display_instruction == "帮我把螺母P01放置到料箱的S11格子中。"


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
    assert (
        mvp_instruction_for_task("BIN01_TO_FINISHED01")
        is MVP_BIN01_TO_FINISHED01
    )


def test_unknown_wording_is_rejected_without_fuzzy_matching() -> None:
    with pytest.raises(ValueError, match="unknown MVP instruction"):
        normalize_mvp_instruction("把螺母P01放进S11")
