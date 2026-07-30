"""Deterministic action validation, limiting, and safety-state handling."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite
from typing import Mapping

from .contracts import ActionChunk, ActionStep, Observation
from .errors import FailureCode


@dataclass(frozen=True)
class SafetyPolicy:
    # Per-command absolute maxima in contract dimension order.
    axis_abs_limits: tuple[float, float, float, float, float, float, float] = (
        0.05,
        0.05,
        0.05,
        0.25,
        0.25,
        0.25,
        1.0,
    )
    # Per-arm TCP envelopes in that arm's robot_base frame, metres.  The
    # asymmetry is derived from the frozen workcell layout and prevents either
    # model from using the other arm's broad/global workspace.
    arm_a_workspace_min_m: tuple[float, float, float] = (0.0, -0.60, 0.0)
    arm_a_workspace_max_m: tuple[float, float, float] = (0.70, 0.45, 1.0)
    arm_b_workspace_min_m: tuple[float, float, float] = (0.0, -0.25, 0.0)
    arm_b_workspace_max_m: tuple[float, float, float] = (0.70, 0.60, 1.0)
    max_chunk_steps: int = 32

    def __post_init__(self) -> None:
        if len(self.axis_abs_limits) != 7 or any(
            not isfinite(value) or value <= 0 for value in self.axis_abs_limits
        ):
            raise ValueError("axis_abs_limits must contain 7 positive finite values")
        if self.axis_abs_limits[6] > 1.0:
            raise ValueError("gripper safety limit cannot exceed normalized range 1.0")
        for arm_id, lower, upper in (
            ("Arm_A", self.arm_a_workspace_min_m, self.arm_a_workspace_max_m),
            ("Arm_B", self.arm_b_workspace_min_m, self.arm_b_workspace_max_m),
        ):
            if (
                len(lower) != 3
                or len(upper) != 3
                or any(not isfinite(value) for value in (*lower, *upper))
                or any(low >= high for low, high in zip(lower, upper))
            ):
                raise ValueError(
                    f"{arm_id} workspace bounds must be 3 finite increasing pairs"
                )
        if not 1 <= self.max_chunk_steps <= 32:
            raise ValueError("max_chunk_steps must be in [1, 32]")


@dataclass(frozen=True)
class SafetyDecision:
    accepted: bool
    code: FailureCode
    reason: str
    chunk: ActionChunk | None = None
    limited_axes: tuple[str, ...] = ()


AXIS_NAMES = ("dx", "dy", "dz", "dax", "day", "daz", "gripper")


def safety_state_failure(observation: Observation) -> tuple[FailureCode, str] | None:
    raw = observation.data.get("safety")
    if not isinstance(raw, Mapping):
        return FailureCode.SYSTEM_FAULT, "required safety observation is missing"
    required = {"emergency_stop", "protective_stop", "system_fault"}
    missing = required - set(raw)
    if missing:
        return (
            FailureCode.SYSTEM_FAULT,
            f"safety observation is missing fields: {sorted(missing)}",
        )
    emergency_stop = raw.get("emergency_stop")
    protective_stop = raw.get("protective_stop")
    if not isinstance(emergency_stop, bool) or not isinstance(protective_stop, bool):
        return FailureCode.SYSTEM_FAULT, "safety stop fields must be booleans"
    if emergency_stop:
        return FailureCode.EMERGENCY_STOP, "emergency stop is active"
    if protective_stop:
        return FailureCode.PROTECTIVE_STOP, "protective stop is active"
    fault = raw.get("system_fault")
    if fault is not None and not isinstance(fault, (bool, str)):
        return FailureCode.SYSTEM_FAULT, "system_fault must be string, boolean or null"
    if fault not in (None, False, "", "NONE", "none"):
        return FailureCode.SYSTEM_FAULT, f"system fault reported: {fault}"
    return None


class ActionSafetyValidator:
    def __init__(self, policy: SafetyPolicy | None = None):
        self.policy = policy or SafetyPolicy()

    def validate_and_limit(
        self,
        chunk: ActionChunk,
        observation: Observation,
        *,
        arm_id: str,
        control_token: str,
    ) -> SafetyDecision:
        try:
            chunk.validate_contract()
        except Exception as exc:
            return SafetyDecision(False, FailureCode.ACTION_CONTRACT_INVALID, str(exc))
        if len(chunk.steps) > self.policy.max_chunk_steps:
            return SafetyDecision(
                False,
                FailureCode.ACTION_CONTRACT_INVALID,
                f"chunk has {len(chunk.steps)} steps; maximum is {self.policy.max_chunk_steps}",
            )
        robot = observation.data.get("robot", {})
        if not isinstance(robot, Mapping):
            return SafetyDecision(
                False, FailureCode.OBSERVATION_INVALID, "robot must be an object"
            )
        arm_key_by_id = {"Arm_A": "arm_a", "Arm_B": "arm_b"}
        required_token_by_id = {"Arm_A": "A_ONLY", "Arm_B": "B_ONLY"}
        arm_key = arm_key_by_id.get(arm_id)
        if arm_key is None:
            return SafetyDecision(
                False,
                FailureCode.ACTION_CONTRACT_INVALID,
                f"unsupported arm_id for safety validation: {arm_id!r}",
            )
        required_token = required_token_by_id[arm_id]
        if control_token != required_token:
            return SafetyDecision(
                False,
                FailureCode.SAFETY_REJECTED,
                f"{arm_id} requires {required_token}, got {control_token!r}",
            )
        workspace_by_arm = {
            "Arm_A": (
                self.policy.arm_a_workspace_min_m,
                self.policy.arm_a_workspace_max_m,
            ),
            "Arm_B": (
                self.policy.arm_b_workspace_min_m,
                self.policy.arm_b_workspace_max_m,
            ),
        }
        workspace_min_m, workspace_max_m = workspace_by_arm[arm_id]
        arm_state = robot.get(arm_key)
        if not isinstance(arm_state, Mapping):
            return SafetyDecision(
                False,
                FailureCode.OBSERVATION_INVALID,
                f"robot.{arm_key} must be an object",
            )
        pose = arm_state.get("tcp_pose_m_rad")
        if (
            not isinstance(pose, (list, tuple))
            or len(pose) < 3
            or any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in pose[:3]
            )
            or any(not isfinite(float(value)) for value in pose[:3])
        ):
            return SafetyDecision(
                False,
                FailureCode.OBSERVATION_INVALID,
                f"robot.{arm_key}.tcp_pose_m_rad must provide "
                "at least 3 finite numbers",
            )
        projected_xyz = [float(value) for value in pose[:3]]
        limited: set[str] = set()
        safe_steps: list[ActionStep] = []
        for step in chunk.steps:
            if step.has_non_finite():
                return SafetyDecision(
                    False,
                    FailureCode.ACTION_NON_FINITE,
                    "NaN or infinity is forbidden in an action",
                )
            values = list(step.values)
            for index, limit in enumerate(self.policy.axis_abs_limits):
                bounded = min(limit, max(-limit, values[index]))
                if bounded != values[index]:
                    values[index] = bounded
                    limited.add(AXIS_NAMES[index])
            for axis in range(3):
                projected_xyz[axis] += values[axis]
                if not (
                    workspace_min_m[axis]
                    <= projected_xyz[axis]
                    <= workspace_max_m[axis]
                ):
                    return SafetyDecision(
                        False,
                        FailureCode.ACTION_WORKSPACE_BREACH,
                        f"projected {AXIS_NAMES[axis][1:]}={projected_xyz[axis]:.6f}m "
                        f"is outside {arm_id} robot_base workspace",
                    )
            safe_steps.append(
                ActionStep.from_sequence(values, duration_ms=step.duration_ms)
            )
        safe_chunk = replace(chunk, steps=tuple(safe_steps))
        return SafetyDecision(
            True,
            FailureCode.NONE,
            "accepted with limiting" if limited else "accepted",
            chunk=safe_chunk,
            limited_axes=tuple(sorted(limited)),
        )
