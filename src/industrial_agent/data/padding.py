"""Configurable, mask-preserving padding for offline training actions.

The frozen online ``ActionChunk`` remains variable length and contains only
real model outputs. Padding produced here is dataset-only: masked rows must
never be sent to the Supervisor, safety gateway, or robot controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_DIM = 7


class PaddingStrategy(str, Enum):
    """Supported offline tail-padding strategies."""

    NONE = "none"
    ZERO_MASKED = "zero_masked"
    REPEAT_LAST_MASKED = "repeat_last_masked"


@dataclass(frozen=True)
class PaddingPolicy:
    """Dataset-only padding policy with no hard-coded temporal horizon.

    ``NONE`` is the fail-closed default while the final per-model training-data
    contract is undecided. Masked rows are storage/training placeholders, not
    executable robot commands.
    """

    # TODO: Freeze the final training-data chunk length and episode-tail policy.
    strategy: PaddingStrategy = PaddingStrategy.NONE
    target_length: int | None = None

    def __post_init__(self) -> None:
        strategy = self.strategy
        if isinstance(strategy, str):
            try:
                strategy = PaddingStrategy(strategy)
            except ValueError as exc:
                raise ValueError(
                    f"unsupported padding strategy: {self.strategy!r}"
                ) from exc
            object.__setattr__(self, "strategy", strategy)
        if not isinstance(strategy, PaddingStrategy):
            raise TypeError("strategy must be a PaddingStrategy or its string value")

        target = self.target_length
        if target is not None and (
            isinstance(target, bool) or not isinstance(target, int) or target < 1
        ):
            raise ValueError("target_length must be a positive integer or None")
        if strategy is not PaddingStrategy.NONE and target is None:
            raise ValueError(
                f"padding strategy {strategy.value!r} requires target_length"
            )

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> PaddingPolicy:
        """Build a policy from a strict JSON-like mapping."""

        if not isinstance(value, Mapping):
            raise TypeError("padding policy must be an object")
        allowed = {"strategy", "target_length"}
        unknown = set(value) - allowed
        if unknown:
            raise ValueError(f"unknown padding policy fields: {sorted(unknown)}")
        return cls(
            strategy=value.get("strategy", PaddingStrategy.NONE),
            target_length=value.get("target_length"),
        )

    def to_dict(self) -> dict[str, str | int | None]:
        return {
            "strategy": self.strategy.value,
            "target_length": self.target_length,
        }


@dataclass(frozen=True)
class PaddingResult:
    """Padded values plus the mandatory loss/execution exclusion mask."""

    values: np.ndarray
    valid_mask: np.ndarray
    valid_length: int
    strategy: PaddingStrategy
    target_length: int

    def executable_values(self) -> np.ndarray:
        """Return only real rows; callers cannot accidentally execute padding."""

        return np.ascontiguousarray(self.values[self.valid_mask])


def _action_matrix(actions: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    if isinstance(actions, (str, bytes, bytearray)):
        raise TypeError("actions must be a numeric N x 7 array")
    try:
        matrix = np.asarray(actions, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError("actions must contain only numeric values") from exc
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] != ACTION_DIM:
        raise ValueError(f"actions must have shape [N, {ACTION_DIM}] with N >= 1")
    if not all(isfinite(float(value)) for value in matrix.flat):
        raise ValueError("actions must contain only finite values")
    gripper = matrix[:, 6]
    if np.any(gripper < -1.0) or np.any(gripper > 1.0):
        raise ValueError("action gripper values must be within [-1, 1]")
    return np.ascontiguousarray(matrix)


def pad_actions(
    actions: Sequence[Sequence[float]] | np.ndarray,
    policy: PaddingPolicy | None = None,
) -> PaddingResult:
    """Apply a dataset-only policy without truncating a real action.

    The default policy preserves the input length. A configured target shorter
    than the input is rejected rather than silently truncating model output.
    """

    resolved = policy if policy is not None else PaddingPolicy()
    if not isinstance(resolved, PaddingPolicy):
        raise TypeError("policy must be a PaddingPolicy or None")
    matrix = _action_matrix(actions)
    valid_length = int(matrix.shape[0])
    target = resolved.target_length or valid_length

    if valid_length > target:
        raise ValueError(
            f"action length {valid_length} exceeds target_length {target}; "
            "truncation is forbidden"
        )
    if resolved.strategy is PaddingStrategy.NONE and valid_length != target:
        raise ValueError(
            "padding strategy 'none' requires the action length to equal target_length"
        )

    values = np.empty((target, ACTION_DIM), dtype=np.float32)
    values[:valid_length] = matrix
    if target > valid_length:
        if resolved.strategy is PaddingStrategy.ZERO_MASKED:
            values[valid_length:] = 0.0
        elif resolved.strategy is PaddingStrategy.REPEAT_LAST_MASKED:
            values[valid_length:] = matrix[-1]
        else:
            raise AssertionError("unreachable padding strategy")

    valid_mask = np.zeros(target, dtype=np.bool_)
    valid_mask[:valid_length] = True
    values.setflags(write=False)
    valid_mask.setflags(write=False)
    return PaddingResult(
        values=values,
        valid_mask=valid_mask,
        valid_length=valid_length,
        strategy=resolved.strategy,
        target_length=target,
    )
