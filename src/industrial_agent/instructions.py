"""Machine-readable MVP instruction options.

The collection UI may show a natural-language sentence, while the dataset
stores one canonical instruction.  Keeping this mapping in one module avoids
silently creating different training tasks from equivalent sentences.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class InstructionOption:
    """One selectable UI instruction and its training representation."""

    task_id: str
    display_instruction: str
    canonical_instruction: str

    def __post_init__(self) -> None:
        for field_name in (
            "task_id",
            "display_instruction",
            "canonical_instruction",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string")
            if value != value.strip():
                raise ValueError(f"{field_name} must not have surrounding whitespace")


# Frozen instruction catalog.  The display and canonical strings are identical
# because the requested frozen wording does not define separate UI prose.
MVP_P01_TO_S11 = InstructionOption(
    task_id="P01_TO_S11",
    display_instruction="把P01放到S11中",
    canonical_instruction="把P01放到S11中",
)
MVP_W01_TO_S14 = InstructionOption(
    task_id="W01_TO_S14",
    display_instruction="把W01放到S14中",
    canonical_instruction="把W01放到S14中",
)
MVP_P03_UPRIGHT_TO_S12 = InstructionOption(
    task_id="P03_UPRIGHT_TO_S12",
    display_instruction="把倒立的P03翻正后放到S12中",
    canonical_instruction="把倒立的P03翻正后放到S12中",
)
MVP_BIN01_TO_FINISHED01 = InstructionOption(
    task_id="BIN01_TO_FINISHED01",
    display_instruction="把Bin_01搬到FINISHED_01",
    canonical_instruction="把Bin_01搬到FINISHED_01",
)
MVP_PACK_ALL_AND_FINISH = InstructionOption(
    task_id="PACK_ALL_AND_FINISH",
    display_instruction="把所有零件装入Bin_01，再把Bin_01搬到FINISHED_01",
    canonical_instruction="把所有零件装入Bin_01，再把Bin_01搬到FINISHED_01",
)

MVP_INSTRUCTION_OPTIONS: tuple[InstructionOption, ...] = (
    MVP_P01_TO_S11,
    MVP_W01_TO_S14,
    MVP_P03_UPRIGHT_TO_S12,
    MVP_BIN01_TO_FINISHED01,
    MVP_PACK_ALL_AND_FINISH,
)
_BY_TASK_ID = {option.task_id: option for option in MVP_INSTRUCTION_OPTIONS}
_BY_TEXT = {
    text: option
    for option in MVP_INSTRUCTION_OPTIONS
    for text in (option.display_instruction, option.canonical_instruction)
}


def mvp_instruction_options() -> tuple[InstructionOption, ...]:
    """Return the immutable set of options exposed by the MVP UI."""

    return MVP_INSTRUCTION_OPTIONS


def mvp_instruction_for_task(task_id: str) -> InstructionOption:
    """Resolve a task ID or raise a clear error for an unknown MVP task."""

    try:
        return _BY_TASK_ID[task_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown MVP task_id: {task_id!r}") from exc


def normalize_mvp_instruction(text: str) -> InstructionOption:
    """Map the exact UI/canonical text to one training contract.

    No fuzzy matching is performed.  A new wording must be deliberately added
    as an alias or as a new versioned option instead of being guessed.
    """

    try:
        return _BY_TEXT[text]
    except (KeyError, TypeError) as exc:
        expected = ", ".join(sorted(_BY_TEXT))
        raise ValueError(f"unknown MVP instruction; use one of: {expected}") from exc


__all__ = [
    "InstructionOption",
    "MVP_INSTRUCTION_OPTIONS",
    "MVP_BIN01_TO_FINISHED01",
    "MVP_P01_TO_S11",
    "MVP_P03_UPRIGHT_TO_S12",
    "MVP_PACK_ALL_AND_FINISH",
    "MVP_W01_TO_S14",
    "mvp_instruction_for_task",
    "mvp_instruction_options",
    "normalize_mvp_instruction",
]
