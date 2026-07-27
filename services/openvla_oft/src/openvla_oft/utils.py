"""Validation helpers for canonical 7-D executor actions."""

from __future__ import annotations

from math import isfinite
from typing import Any, Sequence

from .exceptions import ServiceError

DEFAULT_AXIS_ABS_LIMITS = (0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.0)


def finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServiceError("ACT_1201_CONTRACT_INVALID", f"{field_name} must be numeric")
    result = float(value)
    if not isfinite(result):
        raise ServiceError("ACT_1202_NON_FINITE", f"{field_name} must be finite")
    return result


def finite_vector(value: Any, field_name: str, *, min_length: int = 1) -> list[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ServiceError(
            "OBS_1101_INVALID",
            f"{field_name} must be an array",
            retryable=True,
        )
    if len(value) < min_length:
        raise ServiceError(
            "OBS_1101_INVALID",
            f"{field_name} must contain at least {min_length} values",
            retryable=True,
        )
    return [
        finite_float(item, f"{field_name}[{index}]") for index, item in enumerate(value)
    ]


def validate_action_matrix(
    actions: Any,
    *,
    max_steps: int,
    axis_abs_limits: Sequence[float] = DEFAULT_AXIS_ABS_LIMITS,
) -> list[list[float]]:
    if not isinstance(actions, Sequence) or isinstance(
        actions,
        (str, bytes, bytearray),
    ):
        raise ServiceError(
            "ACT_1201_CONTRACT_INVALID",
            "actions must be an N x 7 array",
        )
    if not 1 <= len(actions) <= max_steps:
        raise ServiceError(
            "ACT_1201_CONTRACT_INVALID",
            f"actions must contain 1..{max_steps} steps",
        )
    normalized: list[list[float]] = []
    for row_index, row in enumerate(actions):
        if not isinstance(row, Sequence) or isinstance(row, (str, bytes, bytearray)):
            raise ServiceError(
                "ACT_1201_CONTRACT_INVALID",
                f"actions[{row_index}] must be an array",
            )
        if len(row) != 7:
            raise ServiceError(
                "ACT_1201_CONTRACT_INVALID",
                f"actions[{row_index}] must contain exactly 7 values",
            )
        values = [
            finite_float(item, f"actions[{row_index}][{col}]")
            for col, item in enumerate(row)
        ]
        for axis, (value, limit) in enumerate(zip(values, axis_abs_limits)):
            if abs(value) > float(limit):
                raise ServiceError(
                    "ACT_1203_WORKSPACE_BREACH",
                    f"actions[{row_index}][{axis}] exceeds absolute limit {limit}",
                )
        if not -1.0 <= values[6] <= 1.0:
            raise ServiceError(
                "ACT_1201_CONTRACT_INVALID",
                "gripper command must be in [-1, 1]",
            )
        normalized.append(values)
    return normalized
