"""Versioned V2 task catalog shared by the UI, Supervisor boundary and π0.5.

The V2 catalog deliberately separates the five user-visible options from the
three tasks that currently have a formal Canonical V2 data contract. A task
may be displayed before its collection/evaluation pipeline is released, but it
must not silently enter training or inference as another task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


V2_SCENE_ID = "single_bin_manual_industrial_v2"
V2_PROFILE_ID = "single_bin_manual_industrial_v2"


@dataclass(frozen=True)
class V2TaskSpec:
    task_id: str
    instruction: str
    target_object: str
    target_slot: str | None
    active_arm: str
    formal_data: bool
    target_zone: str | None = None

    def validate(self) -> None:
        if not self.task_id or not self.instruction.strip():
            raise ValueError("V2 task_id and instruction are required")
        if self.active_arm not in {"Arm_A", "Arm_B"}:
            raise ValueError(f"unsupported V2 active arm: {self.active_arm!r}")
        if self.formal_data and self.target_slot is None and self.target_zone is None:
            raise ValueError(
                "formal V2 tasks require target_slot or target_zone"
            )


V2_TASKS: tuple[V2TaskSpec, ...] = (
    V2TaskSpec(
        task_id="P01_TO_S11",
        instruction="把P01放到S11中",
        target_object="P01",
        target_slot="S11",
        active_arm="Arm_A",
        formal_data=True,
    ),
    V2TaskSpec(
        task_id="W01_TO_S14",
        instruction="把W01放到S14中",
        target_object="W01",
        target_slot="S14",
        active_arm="Arm_A",
        formal_data=True,
    ),
    V2TaskSpec(
        task_id="P03_UPRIGHT_TO_S12",
        instruction="请将倒立的轴件 P03 翻正后，放置到料箱的 S12 格子中。",
        target_object="P03",
        target_slot="S12",
        active_arm="Arm_A",
        formal_data=False,
    ),
    V2TaskSpec(
        task_id="BIN01_TO_FINISHED01",
        instruction="把Bin_01搬到FINISHED_01",
        target_object="Bin_01",
        target_slot=None,
        active_arm="Arm_A",
        formal_data=True,
        target_zone="FINISHED_01",
    ),
    V2TaskSpec(
        task_id="PACK_ALL_AND_FINISH",
        instruction="请将所有零件按指定位置装入料箱 Bin_01，再将料箱 Bin_01 搬运到成品区 FINISHED_01。",
        target_object="Bin_01",
        target_slot=None,
        active_arm="Arm_A",
        formal_data=False,
    ),
)

_BY_ID = {item.task_id: item for item in V2_TASKS}
_BY_INSTRUCTION = {item.instruction: item for item in V2_TASKS}
V2_FORMAL_TASK_IDS = frozenset(item.task_id for item in V2_TASKS if item.formal_data)


def v2_task(task_id: str) -> V2TaskSpec:
    try:
        task = _BY_ID[task_id]
    except (KeyError, TypeError) as exc:
        raise ValueError(f"unknown V2 task_id: {task_id!r}") from exc
    task.validate()
    return task


def v2_task_for_instruction(instruction: str) -> V2TaskSpec:
    try:
        task = _BY_INSTRUCTION[instruction]
    except (KeyError, TypeError) as exc:
        raise ValueError("instruction is not in the frozen V2 task catalog") from exc
    task.validate()
    return task


def require_formal_v2_task(task_id: str) -> V2TaskSpec:
    task = v2_task(task_id)
    if not task.formal_data:
        raise ValueError(
            f"V2 task {task_id!r} is frozen for UI only and has no formal data contract"
        )
    return task


def v2_task_from_mapping(value: Mapping[str, Any]) -> V2TaskSpec:
    task = V2TaskSpec(
        task_id=str(value.get("task_id", "")),
        instruction=str(value.get("instruction", "")),
        target_object=str(value.get("target_object", "")),
        target_slot=value.get("target_slot"),
        active_arm=str(value.get("active_arm", "")),
        formal_data=bool(value.get("formal_data", False)),
        target_zone=(
            str(value["target_zone"])
            if value.get("target_zone") is not None
            else None
        ),
    )
    expected = v2_task(task.task_id)
    if task != expected:
        raise ValueError(
            f"V2 task catalog entry does not match frozen task {task.task_id!r}"
        )
    return task


__all__ = [
    "V2_FORMAL_TASK_IDS",
    "V2_PROFILE_ID",
    "V2_SCENE_ID",
    "V2TaskSpec",
    "require_formal_v2_task",
    "v2_task",
    "v2_task_for_instruction",
    "v2_task_from_mapping",
]
