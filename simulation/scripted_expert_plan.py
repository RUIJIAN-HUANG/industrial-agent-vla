"""Pure planning helpers for the Member-B scripted expert.

This module deliberately has no Isaac Sim imports.  It only computes bounded
Cartesian steps and the frozen first bin-slot target used by the P01 TEST gate.
Object poses are supplied by the isolated ``offline_gt`` adapter at runtime and
must never be copied into Canonical observations or episode metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class P01ExpertTuning:
    """Explicit, reviewable parameters for the first physical grasp probe."""

    tcp_grasp_offset_z_m: float = 0.105
    approach_clearance_m: float = 0.10
    max_cartesian_step_m: float = 0.02
    position_tolerance_m: float = 0.003
    close_steps: int = 3
    release_steps: int = 3

    def validate(self) -> None:
        if not 0.05 <= self.tcp_grasp_offset_z_m <= 0.18:
            raise ValueError("tcp_grasp_offset_z_m must be in [0.05, 0.18]")
        if not 0.05 <= self.approach_clearance_m <= 0.20:
            raise ValueError("approach_clearance_m must be in [0.05, 0.20]")
        if not 0.005 <= self.max_cartesian_step_m <= 0.03:
            raise ValueError("max_cartesian_step_m must be in [0.005, 0.03]")
        if not 0.0005 <= self.position_tolerance_m <= 0.005:
            raise ValueError("position_tolerance_m must be in [0.0005, 0.005]")
        if self.close_steps < 1 or self.release_steps < 1:
            raise ValueError("close_steps and release_steps must be positive")


def bounded_world_delta(
    current_world_m: Sequence[float],
    target_world_m: Sequence[float],
    *,
    max_step_m: float,
) -> np.ndarray:
    """Return one straight-line world delta no longer than ``max_step_m``."""

    current = np.asarray(current_world_m, dtype=float)
    target = np.asarray(target_world_m, dtype=float)
    if current.shape != (3,) or target.shape != (3,):
        raise ValueError("current and target positions must contain three values")
    if not np.all(np.isfinite(current)) or not np.all(np.isfinite(target)):
        raise ValueError("current and target positions must be finite")
    if not 0.0 < float(max_step_m) <= 0.03:
        raise ValueError("max_step_m must be in (0, 0.03]")
    delta = target - current
    distance = float(np.linalg.norm(delta))
    if distance <= max_step_m:
        return delta
    return delta * (float(max_step_m) / distance)


def conservative_step_limit(distance_m: float, max_step_m: float) -> int:
    """Bound the number of actions while leaving room for physics tracking."""

    if distance_m < 0.0 or max_step_m <= 0.0:
        raise ValueError("distance and step size must be non-negative/positive")
    return max(4, int(ceil(distance_m / max_step_m)) * 4 + 8)


def first_bin_slot_local_center(
    *,
    size_m: Sequence[float],
    wall_thickness_m: float,
    bottom_thickness_m: float,
    part_height_m: float,
) -> np.ndarray:
    """Return the frozen row-0/column-0 part-center target in bin-local axes."""

    size = np.asarray(size_m, dtype=float)
    if size.shape != (3,) or not np.all(np.isfinite(size)):
        raise ValueError("bin size must contain three finite values")
    if min(size) <= 0.0:
        raise ValueError("bin dimensions must be positive")
    wall = float(wall_thickness_m)
    bottom = float(bottom_thickness_m)
    part_height = float(part_height_m)
    if min(wall, bottom, part_height) <= 0.0:
        raise ValueError("wall, bottom and part height must be positive")
    interior_x = size[0] - 2.0 * wall
    interior_y = size[1] - 2.0 * wall
    if interior_x <= 0.0 or interior_y <= 0.0 or bottom >= size[2]:
        raise ValueError("bin interior dimensions are invalid")
    return np.asarray(
        [
            -interior_x / 3.0,
            -interior_y / 4.0,
            -size[2] / 2.0 + bottom + part_height / 2.0,
        ],
        dtype=float,
    )


def frozen_success_vote(votes: Sequence[bool]) -> bool:
    """Apply the leader-frozen exactly-three-fresh-frames, two-pass rule."""

    if len(votes) != 3 or any(not isinstance(vote, bool) for vote in votes):
        raise ValueError("success voting requires exactly three boolean frames")
    return sum(votes) >= 2
