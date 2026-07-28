"""Frozen dual-arm, dual-VLA lifecycle contracts.

This module deliberately contains no natural-language parser.  The supervisor
loads one fixed task profile, gives the original instruction to π0.5, and
advances a safety token only after observable postconditions pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Mapping
from uuid import uuid4

from .contracts import (
    OPENVLA_OFT_EXECUTOR_NAME,
    PI05_EXECUTOR_NAME,
    Postcondition,
    Subtask,
    TaskPlan,
    TaskSchema,
)
from .errors import ContractError, FailureCode


class ControlToken(str, Enum):
    """Exclusive actuator ownership for the two-arm workcell."""

    A_ONLY = "A_ONLY"
    HANDOFF_VERIFY = "HANDOFF_VERIFY"
    B_ONLY = "B_ONLY"
    NONE = "NONE"


FROZEN_TOKEN_SEQUENCE = (
    ControlToken.A_ONLY,
    ControlToken.HANDOFF_VERIFY,
    ControlToken.B_ONLY,
    ControlToken.NONE,
)

HANDOFF_CANDIDATE_CHECKED_EVENT_TYPE = "handoff.candidate_checked"
HANDOFF_VERIFIED_EVENT_TYPE = "handoff.verified"
HANDOFF_READY_EVENT_TYPE = "handoff.ready"
# Candidate checks are observations, not irreversible lifecycle milestones.
# Any run that reaches handoff emits one or more checks as retries require.
# Only the verified -> ready pair has a frozen irreversible order.
REPEATABLE_HANDOFF_EVENT_TYPES = (HANDOFF_CANDIDATE_CHECKED_EVENT_TYPE,)
FROZEN_HANDOFF_EVENT_SEQUENCE = (
    HANDOFF_VERIFIED_EVENT_TYPE,
    HANDOFF_READY_EVENT_TYPE,
)

ARM_A_PACK_HANDOFF_SUBTASK_ID = "S01_ARM_A_PACK_HANDOFF"
ARM_B_TRANSPORT_SUBTASK_ID = "S02_ARM_B_TRANSPORT"
FROZEN_SUBTASK_EXECUTOR_ASSIGNMENTS = (
    (ARM_A_PACK_HANDOFF_SUBTASK_ID, PI05_EXECUTOR_NAME),
    (ARM_B_TRANSPORT_SUBTASK_ID, OPENVLA_OFT_EXECUTOR_NAME),
)
FROZEN_SUBTASK_TOKEN_ASSIGNMENTS = (
    (ARM_A_PACK_HANDOFF_SUBTASK_ID, ControlToken.A_ONLY),
    (ARM_B_TRANSPORT_SUBTASK_ID, ControlToken.B_ONLY),
)


@dataclass(frozen=True)
class FixedTaskProfile:
    """Versioned, non-NLP workcell profile loaded by the supervisor."""

    profile_id: str = "single_bin_pack_handoff_v1"
    primary_executor: str = PI05_EXECUTOR_NAME
    collaborative_executor: str = OPENVLA_OFT_EXECUTOR_NAME
    arm_a_id: str = "Arm_A"
    arm_b_id: str = "Arm_B"
    bin_id: str = "Bin_01"
    handoff_zone: str = "HANDOFF_CENTER"
    finished_zone: str = "FINISHED_01"
    expected_part_count: int = 4
    handoff_verification_frames: int = 3
    handoff_required_votes: int = 2
    stable_bin_speed_m_s: float = 0.02
    arm_a_instruction: str = (
        "将工作区中的四个红色零件依次装入料箱；倒放零件先调整为正向。"
        "装箱完成后，将料箱放到中央交接位并返回 HOME_A。"
        "失败时重新观察后继续。"
    )
    arm_b_instruction: str = (
        "收到 handoff_ready 后，观察中央交接位，抓稳 Bin_01 并保持水平，"
        "将其搬到 FINISHED_01，松开夹爪并返回 HOME_B。"
    )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "FixedTaskProfile":
        if not isinstance(value, Mapping):
            raise ValueError("lifecycle.task_profile must be an object")
        profile = cls(
            profile_id=str(value.get("profile_id", "")),
            primary_executor=str(value.get("primary_executor", "")),
            collaborative_executor=str(value.get("collaborative_executor", "")),
            arm_a_id=str(value.get("arm_a_id", "")),
            arm_b_id=str(value.get("arm_b_id", "")),
            bin_id=str(value.get("bin_id", "")),
            handoff_zone=str(value.get("handoff_zone", "")),
            finished_zone=str(value.get("finished_zone", "")),
            expected_part_count=value.get("expected_part_count", 0),
            handoff_verification_frames=value.get("handoff_verification_frames", 0),
            handoff_required_votes=value.get("handoff_required_votes", 0),
            stable_bin_speed_m_s=value.get("stable_bin_speed_m_s", -1.0),
            arm_a_instruction=str(value.get("arm_a_instruction", "")),
            arm_b_instruction=str(value.get("arm_b_instruction", "")),
        )
        profile.validate_frozen()
        return profile

    def validate_frozen(self) -> None:
        expected = FixedTaskProfile()
        for field_name in (
            "profile_id",
            "primary_executor",
            "collaborative_executor",
            "arm_a_id",
            "arm_b_id",
            "bin_id",
            "handoff_zone",
            "finished_zone",
            "expected_part_count",
            "handoff_verification_frames",
            "handoff_required_votes",
            "stable_bin_speed_m_s",
            "arm_a_instruction",
            "arm_b_instruction",
        ):
            actual_value = getattr(self, field_name)
            expected_value = getattr(expected, field_name)
            if actual_value != expected_value:
                raise ValueError(
                    "fixed task profile cannot be changed: "
                    f"{field_name} expected {expected_value!r}, "
                    f"got {actual_value!r}"
                )

    def executor_for_subtask(self, subtask_id: str) -> str:
        mapping = {
            ARM_A_PACK_HANDOFF_SUBTASK_ID: self.primary_executor,
            ARM_B_TRANSPORT_SUBTASK_ID: self.collaborative_executor,
        }
        try:
            return mapping[subtask_id]
        except KeyError as exc:
            raise ContractError(
                FailureCode.INVALID_TASK,
                f"subtask {subtask_id!r} is not part of {self.profile_id}",
            ) from exc

    def required_token_for_subtask(self, subtask_id: str) -> ControlToken:
        mapping = dict(FROZEN_SUBTASK_TOKEN_ASSIGNMENTS)
        try:
            return mapping[subtask_id]
        except KeyError as exc:
            raise ContractError(
                FailureCode.INVALID_TASK,
                f"subtask {subtask_id!r} has no control-token assignment",
            ) from exc


class FixedDualVLAPlanner:
    """Build the only supported two-stage plan without interpreting language."""

    def __init__(self, profile: FixedTaskProfile | None = None):
        self.profile = profile or FixedTaskProfile()
        self.profile.validate_frozen()

    def plan(self, task: TaskSchema, episode_id: str) -> TaskPlan:
        profile = self.profile
        votes = profile.handoff_required_votes
        handoff_conditions = (
            Postcondition(
                kind="numeric_range",
                path="task.packed_part_count",
                minimum=float(profile.expected_part_count),
                maximum=float(profile.expected_part_count),
                required_votes=votes,
            ),
            Postcondition(
                kind="field_equals",
                path="task.bin_at_handoff",
                expected=True,
                required_votes=votes,
            ),
            Postcondition(
                kind="field_equals",
                path="task.bin_at_finished",
                expected=False,
                required_votes=votes,
            ),
            Postcondition(
                kind="numeric_range",
                path="task.bin_speed_m_s",
                minimum=0.0,
                maximum=profile.stable_bin_speed_m_s,
                required_votes=votes,
            ),
            Postcondition(
                kind="field_equals",
                path="robot.arm_a.gripper_open",
                expected=True,
                required_votes=votes,
            ),
            Postcondition(
                kind="field_equals",
                path="robot.arm_a.retreated",
                expected=True,
                required_votes=votes,
            ),
            Postcondition(
                kind="field_equals",
                path="robot.arm_b.retreated",
                expected=True,
                required_votes=votes,
            ),
        )
        transport_conditions = (
            Postcondition(
                kind="field_equals",
                path="task.bin_at_finished",
                expected=True,
                required_votes=votes,
            ),
            Postcondition(
                kind="field_equals",
                path="task.bin_at_handoff",
                expected=False,
                required_votes=votes,
            ),
            Postcondition(
                kind="numeric_range",
                path="task.bin_speed_m_s",
                minimum=0.0,
                maximum=profile.stable_bin_speed_m_s,
                required_votes=votes,
            ),
            Postcondition(
                kind="field_equals",
                path="robot.arm_b.gripper_open",
                expected=True,
                required_votes=votes,
            ),
            Postcondition(
                kind="field_equals",
                path="robot.arm_b.retreated",
                expected=True,
                required_votes=votes,
            ),
            Postcondition(
                kind="field_equals",
                path="robot.arm_a.retreated",
                expected=True,
                required_votes=votes,
            ),
        )
        first_task_type = "mock_demo" if task.task_type == "mock_demo" else "pick_place"
        second_task_type = (
            "mock_demo" if task.task_type == "mock_demo" else "visual_manipulation"
        )
        plan = TaskPlan(
            plan_id=str(uuid4()),
            episode_id=episode_id,
            task_id=task.task_id,
            subtasks=[
                Subtask(
                    subtask_id=ARM_A_PACK_HANDOFF_SUBTASK_ID,
                    sequence=1,
                    # π0.5 alone receives and interprets this preset instruction.
                    instruction=profile.arm_a_instruction,
                    task_type=first_task_type,
                    preconditions=(),
                    postconditions=handoff_conditions,
                    assigned_executor=profile.primary_executor,
                ),
                Subtask(
                    subtask_id=ARM_B_TRANSPORT_SUBTASK_ID,
                    sequence=2,
                    instruction=profile.arm_b_instruction,
                    task_type=second_task_type,
                    preconditions=handoff_conditions,
                    postconditions=transport_conditions + task.postconditions,
                    depends_on=(ARM_A_PACK_HANDOFF_SUBTASK_ID,),
                    assigned_executor=profile.collaborative_executor,
                ),
            ],
        )
        plan.validate()
        return plan


@dataclass
class FixedLifecycle:
    """Small deterministic token state machine used alongside the main FSM."""

    profile: FixedTaskProfile
    token: ControlToken = ControlToken.A_ONLY

    def authorize(self, subtask_id: str, executor_name: str) -> None:
        required_token = self.profile.required_token_for_subtask(subtask_id)
        required_executor = self.profile.executor_for_subtask(subtask_id)
        if self.token is not required_token:
            raise ContractError(
                FailureCode.SAFETY_REJECTED,
                f"{executor_name} is forbidden while token={self.token.value}; "
                f"{subtask_id} requires {required_token.value}",
            )
        if executor_name != required_executor:
            raise ContractError(
                FailureCode.SAFETY_REJECTED,
                f"{subtask_id} requires {required_executor}, got {executor_name}",
            )

    def begin_handoff_verification(self) -> tuple[ControlToken, ControlToken]:
        return self._advance(
            ControlToken.A_ONLY,
            ControlToken.HANDOFF_VERIFY,
        )

    def grant_arm_b(self) -> tuple[ControlToken, ControlToken]:
        return self._advance(
            ControlToken.HANDOFF_VERIFY,
            ControlToken.B_ONLY,
        )

    def complete(self) -> tuple[ControlToken, ControlToken]:
        return self._advance(ControlToken.B_ONLY, ControlToken.NONE)

    def safe_stop(self) -> tuple[ControlToken, ControlToken]:
        previous = self.token
        self.token = ControlToken.NONE
        return previous, self.token

    def _advance(
        self,
        expected: ControlToken,
        target: ControlToken,
    ) -> tuple[ControlToken, ControlToken]:
        if self.token is not expected:
            raise ContractError(
                FailureCode.SAFETY_REJECTED,
                f"illegal control-token transition {self.token.value}->{target.value}",
            )
        previous = self.token
        self.token = target
        return previous, target
