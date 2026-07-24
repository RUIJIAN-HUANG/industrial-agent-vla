"""Constrained semantic TaskPlan templates.

The planner emits language-level intent only. Coordinates, trajectories, and
grasp points are deliberately absent and remain the VLA's responsibility.
"""

from __future__ import annotations

from uuid import uuid4

from .contracts import Postcondition, Subtask, TaskPlan, TaskSchema


def _final_or_default(
    task: TaskSchema, default: tuple[Postcondition, ...]
) -> tuple[Postcondition, ...]:
    return task.postconditions if task.postconditions else default


class SemanticTaskPlanner:
    SUPPORTED_WORKFLOWS = frozenset(
        {
            "single",
            "place_in_designated_slot",
            "pack_until_full",
            "fill_then_move_stack",
        }
    )

    def plan(self, task: TaskSchema, episode_id: str) -> TaskPlan:
        workflow = str(task.constraints.get("workflow", "single"))
        if workflow not in self.SUPPORTED_WORKFLOWS:
            workflow = "single"
        if workflow == "place_in_designated_slot":
            subtasks = self._designated_slot(task)
        elif workflow == "pack_until_full":
            subtasks = self._pack_until_full(task)
        elif workflow == "fill_then_move_stack":
            subtasks = self._fill_then_move_stack(task)
        else:
            subtasks = [
                Subtask(
                    subtask_id="S01_EXECUTE",
                    sequence=1,
                    instruction=task.instruction,
                    task_type=task.task_type,
                    preconditions=(),
                    postconditions=task.postconditions,
                    assigned_executor=task.preferred_executor,
                )
            ]
        plan = TaskPlan(
            plan_id=str(uuid4()),
            episode_id=episode_id,
            task_id=task.task_id,
            subtasks=subtasks,
        )
        plan.validate()
        return plan

    @staticmethod
    def _designated_slot(task: TaskSchema) -> list[Subtask]:
        object_id = task.target_object or str(
            task.constraints.get("object_id", "target_object")
        )
        target_zone = task.target_location or str(
            task.constraints.get("target_zone", "designated_slot")
        )
        detected = Postcondition(
            kind="object_detected", object_id=object_id, required_votes=2
        )
        placed = Postcondition(
            kind="object_in_zone",
            object_id=object_id,
            zone_id=target_zone,
            required_votes=2,
        )
        return [
            Subtask(
                subtask_id="S01_LOCATE",
                sequence=1,
                instruction=f"识别并确认待操作物体“{object_id}”，保持环境不变。",
                task_type="object_localization",
                preconditions=(),
                postconditions=(detected,),
                assigned_executor="openvla_oft",
            ),
            Subtask(
                subtask_id="S02_PLACE",
                sequence=2,
                instruction=(
                    f"将已确认的“{object_id}”放入指定格“{target_zone}”，"
                    "动作细节由视觉动作模型决定。"
                ),
                task_type="pick_place",
                preconditions=(detected,),
                postconditions=_final_or_default(task, (placed,)),
                depends_on=("S01_LOCATE",),
            ),
        ]

    @staticmethod
    def _pack_until_full(task: TaskSchema) -> list[Subtask]:
        container_id = str(task.constraints.get("container_id", "target_container"))
        max_iterations = int(task.constraints.get("max_pack_iterations", 12))
        full = Postcondition(
            kind="field_equals",
            path="task.container_full",
            expected=True,
            required_votes=2,
        )
        return [
            Subtask(
                subtask_id="S01_PACK_UNTIL_FULL",
                sequence=1,
                instruction=(
                    f"从允许区域逐件选择可装物体并装入“{container_id}”；"
                    "每次只处理一件，传感器确认已满后立即停止继续装箱。"
                ),
                task_type="pick_place",
                preconditions=(),
                postconditions=_final_or_default(task, (full,)),
                repeat_until_postcondition=True,
                max_iterations=max_iterations,
            )
        ]

    @staticmethod
    def _fill_then_move_stack(task: TaskSchema) -> list[Subtask]:
        container_id = str(task.constraints.get("container_id", "target_container"))
        target_zone = task.target_location or str(
            task.constraints.get("target_zone", "stack_zone")
        )
        max_iterations = int(task.constraints.get("max_pack_iterations", 12))
        full = Postcondition(
            kind="field_equals",
            path="task.container_full",
            expected=True,
            required_votes=2,
        )
        moved = Postcondition(
            kind="object_in_zone",
            object_id=container_id,
            zone_id=target_zone,
            required_votes=2,
        )
        stacked = Postcondition(
            kind="field_equals",
            path="task.stacked",
            expected=True,
            required_votes=2,
        )
        return [
            Subtask(
                subtask_id="S01_FILL",
                sequence=1,
                instruction=f"检查“{container_id}”；若未满则逐件装入，确认已满后停止。",
                task_type="pick_place",
                preconditions=(),
                postconditions=(full,),
                repeat_until_postcondition=True,
                max_iterations=max_iterations,
            ),
            Subtask(
                subtask_id="S02_MOVE",
                sequence=2,
                instruction=(
                    f"仅在确认已满后，将料箱“{container_id}”搬运至“{target_zone}”。"
                ),
                task_type="visual_manipulation",
                preconditions=(full,),
                postconditions=(moved,),
                depends_on=("S01_FILL",),
            ),
            Subtask(
                subtask_id="S03_STACK",
                sequence=3,
                instruction=(
                    f"在“{target_zone}”按现场可见约束完成料箱“{container_id}”叠放。"
                ),
                task_type="visual_manipulation",
                preconditions=(moved,),
                postconditions=_final_or_default(task, (stacked,)),
                depends_on=("S02_MOVE",),
            ),
        ]
