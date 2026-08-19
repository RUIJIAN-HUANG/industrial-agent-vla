"""Fail-closed workflow state for V2 manual keyboard collection.

This module contains no Isaac Sim imports. It validates collection progress
before the GUI entry point is allowed to publish a Canonical episode.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import Enum
from types import MappingProxyType
from typing import Any


class ControlToken(str, Enum):
    A_ONLY = "A_ONLY"
    HANDOFF_VERIFY = "HANDOFF_VERIFY"
    B_ONLY = "B_ONLY"
    NONE = "NONE"


class EpisodeOutcome(str, Enum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SAFE_STOPPED = "SAFE_STOPPED"
    SAFE_STOP_FAILED = "SAFE_STOP_FAILED"


class V2FailureCode(str, Enum):
    INACTIVE_ARM_ACTION = "V2_COLLECTION_INACTIVE_ARM_ACTION"
    PART_ORDER_VIOLATION = "V2_COLLECTION_PART_ORDER_VIOLATION"
    SLOT_MISMATCH = "V2_COLLECTION_SLOT_MISMATCH"
    PART_UNSTABLE = "V2_COLLECTION_PART_UNSTABLE"
    HANDOFF_PRECONDITION_FAILED = "V2_COLLECTION_HANDOFF_PRECONDITION_FAILED"
    FINISH_PRECONDITION_FAILED = "V2_COLLECTION_FINISH_PRECONDITION_FAILED"
    USER_SAFE_STOP = "V2_COLLECTION_USER_SAFE_STOP"
    SAFE_STOP_CONFIRMATION_FAILED = "V2_COLLECTION_SAFE_STOP_CONFIRMATION_FAILED"
    P01_ORIENTATION_EXCEEDED = "P01_ORIENTATION_EXCEEDED"
    P01_GT_NOT_FRESH = "P01_GT_NOT_FRESH"
    P01_GT_VOTE_INSUFFICIENT = "P01_GT_VOTE_INSUFFICIENT"
    P01_TERMINAL_HOLD_TOO_SHORT = "P01_TERMINAL_HOLD_TOO_SHORT"
    P01_TERMINAL_DRIFT_EXCEEDED = "P01_TERMINAL_DRIFT_EXCEEDED"
    P01_OFFLINE_GT_UNAVAILABLE = "P01_OFFLINE_GT_UNAVAILABLE"


class CollectionStateError(RuntimeError):
    """A stable, machine-readable V2 workflow rejection."""

    def __init__(self, code: V2FailureCode, message: str):
        super().__init__(message)
        self.code = code


class V2CollectionContract:
    """Frozen V2 workflow fields loaded from the audited scene config."""

    def __init__(
        self,
        *,
        scene_id: str,
        formal_part_order: tuple[str, ...],
        part_to_slot: Mapping[str, str],
        token_sequence: tuple[ControlToken, ...],
    ):
        self.scene_id = scene_id
        self.formal_part_order = formal_part_order
        self.part_to_slot = MappingProxyType(dict(part_to_slot))
        self.token_sequence = token_sequence

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> V2CollectionContract:
        scene_id = config.get("scene_id")
        if scene_id != "single_bin_manual_industrial_v2":
            raise ValueError("scene_id is not the audited V2 identity")

        collection = config.get("collection")
        if not isinstance(collection, Mapping):
            raise ValueError("collection must be an object")

        raw_order = collection.get("formal_part_order")
        if (
            not isinstance(raw_order, list)
            or not raw_order
            or not all(isinstance(item, str) and item for item in raw_order)
        ):
            raise ValueError("collection.formal_part_order is invalid")

        formal_part_order = tuple(raw_order)
        if len(set(formal_part_order)) != len(formal_part_order):
            raise ValueError("collection.formal_part_order contains duplicates")

        bin_config = config.get("bin")
        if not isinstance(bin_config, Mapping):
            raise ValueError("bin must be an object")

        slots = bin_config.get("slots")
        if not isinstance(slots, list):
            raise ValueError("bin.slots must be a list")

        part_to_slot: dict[str, str] = {}
        for slot in slots:
            if not isinstance(slot, Mapping):
                raise ValueError("each bin slot must be an object")
            part_id = slot.get("part_id")
            slot_id = slot.get("id")
            if not isinstance(part_id, str) or not isinstance(slot_id, str):
                raise ValueError("each bin slot requires string part_id and id")
            if part_id in part_to_slot:
                raise ValueError(f"duplicate slot mapping for {part_id}")
            part_to_slot[part_id] = slot_id

        if set(part_to_slot) != set(formal_part_order):
            raise ValueError(
                "formal part order and bin slot mappings must contain the same parts"
            )

        workflow = config.get("workflow")
        if not isinstance(workflow, Mapping):
            raise ValueError("workflow must be an object")

        raw_tokens = workflow.get("token_sequence")
        expected_tokens = (
            ControlToken.A_ONLY,
            ControlToken.HANDOFF_VERIFY,
            ControlToken.B_ONLY,
            ControlToken.NONE,
        )
        try:
            token_sequence = tuple(ControlToken(item) for item in raw_tokens)
        except (TypeError, ValueError) as exc:
            raise ValueError("workflow.token_sequence is invalid") from exc

        if token_sequence != expected_tokens:
            raise ValueError("workflow.token_sequence is not the frozen sequence")

        return cls(
            scene_id=scene_id,
            formal_part_order=formal_part_order,
            part_to_slot=part_to_slot,
            token_sequence=token_sequence,
        )


class V2ManualCollectionStateMachine:
    """Fail-closed progress tracker for one V2 collection attempt."""

    def __init__(self, contract: V2CollectionContract):
        self.contract = contract
        self.token = ControlToken.A_ONLY
        self.outcome: EpisodeOutcome | None = None
        self.failure_code: V2FailureCode | None = None
        self._placed_parts: list[str] = []

    @property
    def placed_parts(self) -> tuple[str, ...]:
        return tuple(self._placed_parts)

    @property
    def next_part_id(self) -> str | None:
        index = len(self._placed_parts)
        if index >= len(self.contract.formal_part_order):
            return None
        return self.contract.formal_part_order[index]

    def _require_active(self) -> None:
        if self.outcome is not None or self.token is ControlToken.NONE:
            raise RuntimeError("collection attempt is already terminal")

    def _reject(self, code: V2FailureCode, message: str) -> None:
        self.failure_code = code
        self.outcome = EpisodeOutcome.FAILED
        self.token = ControlToken.NONE
        raise CollectionStateError(code, message)

    def require_arm_action(self, arm_id: str) -> None:
        self._require_active()
        allowed = {
            ControlToken.A_ONLY: "Arm_A",
            ControlToken.B_ONLY: "Arm_B",
        }.get(self.token)

        if arm_id != allowed:
            self._reject(
                V2FailureCode.INACTIVE_ARM_ACTION,
                f"{arm_id} cannot act while token is {self.token.value}",
            )

    def record_part_placement(
        self,
        *,
        part_id: str,
        slot_id: str,
        stable: bool,
    ) -> None:
        self._require_active()
        if self.token is not ControlToken.A_ONLY:
            self._reject(
                V2FailureCode.INACTIVE_ARM_ACTION,
                "parts may only be placed during A_ONLY",
            )

        expected_part = self.next_part_id
        if part_id != expected_part:
            self._reject(
                V2FailureCode.PART_ORDER_VIOLATION,
                f"expected {expected_part}, got {part_id}",
            )

        expected_slot = self.contract.part_to_slot[part_id]
        if slot_id != expected_slot:
            self._reject(
                V2FailureCode.SLOT_MISMATCH,
                f"{part_id} must be placed in {expected_slot}, got {slot_id}",
            )

        if not stable:
            self._reject(
                V2FailureCode.PART_UNSTABLE,
                f"{part_id} is not stable in {slot_id}",
            )

        self._placed_parts.append(part_id)

    def enter_handoff_verify(
        self,
        *,
        bin_at_handoff_center: bool,
        bin_stable: bool,
        arm_a_gripper_open: bool,
        arm_a_clear: bool,
    ) -> None:
        self._require_active()
        if self.token is not ControlToken.A_ONLY:
            self._reject(
                V2FailureCode.HANDOFF_PRECONDITION_FAILED,
                "handoff verification requires A_ONLY",
            )

        all_parts_placed = tuple(self._placed_parts) == self.contract.formal_part_order
        if not all(
            (
                all_parts_placed,
                bin_at_handoff_center,
                bin_stable,
                arm_a_gripper_open,
                arm_a_clear,
            )
        ):
            self._reject(
                V2FailureCode.HANDOFF_PRECONDITION_FAILED,
                "handoff verification preconditions are not satisfied",
            )

        self.token = ControlToken.HANDOFF_VERIFY

    def activate_b_only(self) -> None:
        self._require_active()
        if self.token is not ControlToken.HANDOFF_VERIFY:
            self._reject(
                V2FailureCode.HANDOFF_PRECONDITION_FAILED,
                "B_ONLY requires HANDOFF_VERIFY",
            )
        self.token = ControlToken.B_ONLY

    def complete(
        self,
        *,
        bin_at_finished: bool,
        bin_stable: bool,
        arm_b_gripper_open: bool,
        arm_b_clear: bool,
    ) -> None:
        self._require_active()
        if self.token is not ControlToken.B_ONLY or not all(
            (
                bin_at_finished,
                bin_stable,
                arm_b_gripper_open,
                arm_b_clear,
            )
        ):
            self._reject(
                V2FailureCode.FINISH_PRECONDITION_FAILED,
                "finished-station preconditions are not satisfied",
            )

        self.token = ControlToken.NONE
        self.outcome = EpisodeOutcome.SUCCEEDED
        self.failure_code = None

    def safe_stop(self, *, confirmed: bool) -> None:
        self._require_active()
        self.token = ControlToken.NONE

        if confirmed:
            self.outcome = EpisodeOutcome.SAFE_STOPPED
            self.failure_code = V2FailureCode.USER_SAFE_STOP
        else:
            self.outcome = EpisodeOutcome.SAFE_STOP_FAILED
            self.failure_code = V2FailureCode.SAFE_STOP_CONFIRMATION_FAILED

    def fail_offline_gt(self, code: V2FailureCode) -> None:
        """Convert a provisional workflow success into a failed GT gate."""

        if self.outcome is not EpisodeOutcome.SUCCEEDED:
            raise RuntimeError("offline GT can only gate a provisional success")
        if not isinstance(code, V2FailureCode):
            raise TypeError("offline GT failure code must be V2FailureCode")
        self.outcome = EpisodeOutcome.FAILED
        self.failure_code = code
        self.token = ControlToken.NONE
