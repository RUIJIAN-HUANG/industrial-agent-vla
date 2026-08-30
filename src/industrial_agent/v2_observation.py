"""Fail-closed ingress for the formal V2 online observation contract."""

from __future__ import annotations

from copy import deepcopy
from math import isfinite
from typing import Any, Mapping

from .contracts import Observation
from .errors import FailureCode, ObservationError
from .observation import (
    ONLINE_TOP_LEVEL_ALLOWLIST,
    REQUIRED_TOP_LEVEL_FIELDS,
    _frozen_camera_failure,
    find_forbidden_online_path,
)
from .v2_task_profile import V2TaskSpec, require_formal_v2_task


V2_OBSERVATION_VERSION = "2.0"


class V2ObservationGateway:
    """Accept only fresh, sensor-derived V2 observations.

    V1 lifecycle summaries such as ``packed_part_count`` and
    ``bin_at_handoff`` are deliberately rejected by the exact V2 task-field
    allowlist.
    """

    _TASK_FIELDS = frozenset(
        {
            "task_id",
            "target_object_id",
            "target_slot_id",
            "status",
            "terminal",
            "terminal_confidence",
            "verification_votes",
        }
    )

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._seen_observation_ids: set[str] = set()
        self._last_timestamp_ms: int | None = None

    def ingest_online(self, raw: Mapping[str, Any]) -> Observation:
        if not isinstance(raw, Mapping):
            self._invalid("online V2 observation must be an object")
        forbidden = find_forbidden_online_path(raw)
        if forbidden:
            raise ObservationError(
                FailureCode.OBSERVATION_GT_FORBIDDEN,
                f"ground-truth-like field is forbidden online: {forbidden}",
            )
        unknown = set(raw) - ONLINE_TOP_LEVEL_ALLOWLIST
        if unknown:
            self._invalid(
                f"online V2 observation contains non-allowlisted fields: {sorted(unknown)}"
            )
        missing = REQUIRED_TOP_LEVEL_FIELDS - set(raw)
        if missing:
            self._invalid(
                f"online V2 observation is missing required fields: {sorted(missing)}"
            )
        if raw.get("observation_version") != V2_OBSERVATION_VERSION:
            self._invalid("formal V2 runtime requires observation_version='2.0'")

        observation_id = raw.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            self._invalid("observation_id is required")
        if observation_id in self._seen_observation_ids:
            self._invalid(
                f"observation_id must be fresh within a run: {observation_id}"
            )
        timestamp = raw.get("timestamp_ms")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
        ):
            self._invalid("timestamp_ms must be a non-negative integer")
        if self._last_timestamp_ms is not None and timestamp < self._last_timestamp_ms:
            self._invalid("timestamp_ms moved backwards within a run")

        camera_failure = _frozen_camera_failure(raw.get("camera"))
        if camera_failure:
            self._invalid(camera_failure)
        task_spec = self._validate_task(raw.get("task"))
        self._validate_robot(raw.get("robot"), task_id=task_spec.task_id)
        self._validate_safety(raw.get("safety"))
        self._validate_quality(raw.get("quality"))

        data = {
            key: deepcopy(value)
            for key, value in raw.items()
            if key not in {"observation_version", "observation_id", "timestamp_ms"}
        }
        self._seen_observation_ids.add(observation_id)
        self._last_timestamp_ms = timestamp
        return Observation(
            observation_version=V2_OBSERVATION_VERSION,
            observation_id=observation_id,
            timestamp_ms=timestamp,
            data=data,
        )

    @classmethod
    def _validate_task(cls, value: Any) -> V2TaskSpec:
        if not isinstance(value, Mapping) or set(value) != cls._TASK_FIELDS:
            cls._invalid(
                f"V2 task state must contain exactly {sorted(cls._TASK_FIELDS)}"
            )
        task_id = value.get("task_id")
        try:
            spec = require_formal_v2_task(str(task_id))
        except ValueError as exc:
            cls._invalid(str(exc))
        if value.get("target_object_id") != spec.target_object:
            cls._invalid("V2 task target_object_id does not match the frozen profile")
        if value.get("target_slot_id") != spec.target_slot:
            cls._invalid("V2 task target_slot_id does not match the frozen profile")
        status = value.get("status")
        terminal = value.get("terminal")
        confidence = value.get("terminal_confidence")
        votes = value.get("verification_votes")
        if status not in {"ACTIVE", "SUCCEEDED", "FAILED"}:
            cls._invalid("V2 task status must be ACTIVE, SUCCEEDED or FAILED")
        if not isinstance(terminal, bool):
            cls._invalid("V2 task terminal must be boolean")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            cls._invalid("V2 task terminal_confidence must be finite in [0, 1]")
        if isinstance(votes, bool) or not isinstance(votes, int) or not 0 <= votes <= 3:
            cls._invalid("V2 task verification_votes must be an integer in [0, 3]")
        if terminal != (status == "SUCCEEDED"):
            cls._invalid("V2 terminal must be true exactly when status is SUCCEEDED")
        return spec

    @classmethod
    def _validate_robot(cls, value: Any, *, task_id: str) -> None:
        if not isinstance(value, Mapping):
            cls._invalid("robot must be an object")
        active_arm = value.get("active_arm")
        allowed_active_arms = (
            {"Arm_A", "Arm_B", "NONE"}
            if task_id == "BIN01_TO_FINISHED01"
            else {"Arm_A", "NONE"}
        )
        if active_arm not in allowed_active_arms:
            if task_id == "BIN01_TO_FINISHED01":
                cls._invalid(
                    "formal V2 bin handoff permits only Arm_A, Arm_B or NONE "
                    "as robot.active_arm"
                )
            cls._invalid("formal V2 permits only Arm_A or NONE as robot.active_arm")
        for arm_key in ("arm_a", "arm_b"):
            arm = value.get(arm_key)
            if not isinstance(arm, Mapping):
                cls._invalid(f"robot.{arm_key} must be an object")
            if (
                not isinstance(arm.get("tcp_pose_m_rad"), (list, tuple))
                or len(arm["tcp_pose_m_rad"]) != 6
            ):
                cls._invalid(f"robot.{arm_key}.tcp_pose_m_rad must have 6 values")
            if (
                not isinstance(arm.get("state"), (list, tuple))
                or len(arm["state"]) != 7
            ):
                cls._invalid(f"robot.{arm_key}.state must have 7 values")
            for flag in ("retreated", "gripper_open", "stationary"):
                if not isinstance(arm.get(flag), bool):
                    cls._invalid(f"robot.{arm_key}.{flag} must be boolean")
        arm_a = value["arm_a"]
        arm_b = value["arm_b"]
        if task_id != "BIN01_TO_FINISHED01":
            if arm_b.get("retreated") is not True or arm_b.get("stationary") is not True:
                cls._invalid("Arm_B must remain retreated and stationary in formal V2")
            return
        if active_arm == "Arm_A" and (
            arm_b.get("retreated") is not True or arm_b.get("stationary") is not True
        ):
            cls._invalid("Arm_B must remain retreated and stationary while Arm_A acts")
        if active_arm == "Arm_B" and (
            arm_a.get("retreated") is not True or arm_a.get("stationary") is not True
        ):
            cls._invalid("Arm_A must remain retreated and stationary while Arm_B acts")
        if active_arm == "NONE" and not all(
            arm.get("retreated") is True and arm.get("stationary") is True
            for arm in (arm_a, arm_b)
        ):
            cls._invalid("both arms must be retreated and stationary during handoff verification")

    @classmethod
    def _validate_safety(cls, value: Any) -> None:
        if not isinstance(value, Mapping):
            cls._invalid("safety must be an object")
        if set(value) != {"emergency_stop", "protective_stop", "system_fault"}:
            cls._invalid("safety fields do not match the V2 contract")
        if not isinstance(value.get("emergency_stop"), bool) or not isinstance(
            value.get("protective_stop"), bool
        ):
            cls._invalid("safety stop fields must be boolean")
        if value.get("system_fault") is not None and not isinstance(
            value.get("system_fault"), (bool, str)
        ):
            cls._invalid("safety.system_fault must be string, boolean or null")

    @classmethod
    def _validate_quality(cls, value: Any) -> None:
        if not isinstance(value, Mapping) or set(value) != {"confidence"}:
            cls._invalid("quality must contain exactly confidence")
        confidence = value.get("confidence")
        if (
            isinstance(confidence, bool)
            or not isinstance(confidence, (int, float))
            or not isfinite(float(confidence))
            or not 0 <= float(confidence) <= 1
        ):
            cls._invalid("quality.confidence must be finite in [0, 1]")

    @staticmethod
    def _invalid(message: str) -> None:
        raise ObservationError(FailureCode.OBSERVATION_INVALID, message)


__all__ = ["V2_OBSERVATION_VERSION", "V2ObservationGateway"]
