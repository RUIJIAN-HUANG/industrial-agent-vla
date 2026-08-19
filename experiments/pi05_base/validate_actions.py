"""Inspection helpers that preserve unknown native action semantics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActionInspection:
    shape: tuple[int, int]
    dtype: str
    horizon: int
    action_dim: int
    all_finite: bool
    non_finite_count: int
    finite_min: float | None
    finite_max: float | None
    finite_mean: float | None
    finite_abs_max_by_axis: tuple[float | None, ...]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["shape"] = list(self.shape)
        result["finite_abs_max_by_axis"] = list(self.finite_abs_max_by_axis)
        return result


def inspect_actions(actions: np.ndarray) -> ActionInspection:
    """Describe raw policy output without truncating or assigning semantics."""

    array = np.asarray(actions)
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 1:
        raise ValueError("actions must have shape [horizon,action_dim]")
    finite_mask = np.isfinite(array)
    finite_values = array[finite_mask]
    per_axis: list[float | None] = []
    for axis in range(array.shape[1]):
        values = array[:, axis]
        finite_axis = values[np.isfinite(values)]
        per_axis.append(
            float(np.max(np.abs(finite_axis))) if finite_axis.size else None
        )
    return ActionInspection(
        shape=(int(array.shape[0]), int(array.shape[1])),
        dtype=str(array.dtype),
        horizon=int(array.shape[0]),
        action_dim=int(array.shape[1]),
        all_finite=bool(np.all(finite_mask)),
        non_finite_count=int(array.size - np.count_nonzero(finite_mask)),
        finite_min=float(np.min(finite_values)) if finite_values.size else None,
        finite_max=float(np.max(finite_values)) if finite_values.size else None,
        finite_mean=float(np.mean(finite_values)) if finite_values.size else None,
        finite_abs_max_by_axis=tuple(per_axis),
    )


def require_finite_actions(actions: np.ndarray) -> ActionInspection:
    """Reject unsafe numeric output before any future simulator integration."""

    inspection = inspect_actions(actions)
    if not inspection.all_finite:
        raise ValueError(
            f"policy output contains {inspection.non_finite_count} NaN/Inf values"
        )
    return inspection
