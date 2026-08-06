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

    approach_clearance_m: float = 0.10
    max_cartesian_step_m: float = 0.02
    position_tolerance_m: float = 0.005
    grasp_probe_lift_m: float = 0.04
    minimum_grasp_follow_ratio: float = 0.60
    maximum_grasp_follow_error_m: float = 0.015
    physical_pinch_alignment_tolerance_m: float = 0.006
    maximum_finger_center_separation_m: float = 0.15
    minimum_finger_contact_ratio: float = 0.60
    max_rotation_step_rad: float = 0.20
    rotation_tolerance_rad: float = 0.035
    max_rotation_steps: int = 24
    transit_clearance_m: float = 0.04
    slot_work_radius_margin_m: float = 0.03
    max_actual_step_m: float = 0.06
    divergence_tolerance_m: float = 0.004
    max_consecutive_divergent_steps: int = 2
    close_steps: int = 12
    release_steps: int = 8

    def validate(self) -> None:
        if not 0.05 <= self.approach_clearance_m <= 0.20:
            raise ValueError("approach_clearance_m must be in [0.05, 0.20]")
        if not 0.005 <= self.max_cartesian_step_m <= 0.03:
            raise ValueError("max_cartesian_step_m must be in [0.005, 0.03]")
        if not 0.0005 <= self.position_tolerance_m <= 0.005:
            raise ValueError("position_tolerance_m must be in [0.0005, 0.005]")
        if not 0.02 <= self.transit_clearance_m <= 0.08:
            raise ValueError("transit_clearance_m must be in [0.02, 0.08]")
        if not 0.01 <= self.slot_work_radius_margin_m <= 0.08:
            raise ValueError("slot_work_radius_margin_m must be in [0.01, 0.08]")
        if not self.max_cartesian_step_m < self.max_actual_step_m <= 0.08:
            raise ValueError(
                "max_actual_step_m must exceed max_cartesian_step_m and be <= 0.08"
            )
        if not 0.001 <= self.divergence_tolerance_m <= 0.01:
            raise ValueError("divergence_tolerance_m must be in [0.001, 0.01]")
        if self.max_consecutive_divergent_steps < 1:
            raise ValueError("max_consecutive_divergent_steps must be positive")
        if self.close_steps < 1 or self.release_steps < 1:
            raise ValueError("close_steps and release_steps must be positive")
        if not 0.02 <= self.grasp_probe_lift_m <= 0.06:
            raise ValueError("grasp_probe_lift_m must be in [0.02, 0.06]")
        if not 0.5 <= self.minimum_grasp_follow_ratio <= 0.9:
            raise ValueError("minimum_grasp_follow_ratio must be in [0.5, 0.9]")
        if not 0.005 <= self.maximum_grasp_follow_error_m <= 0.02:
            raise ValueError("maximum_grasp_follow_error_m must be in [0.005, 0.02]")
        if not 0.001 <= self.physical_pinch_alignment_tolerance_m <= 0.01:
            raise ValueError(
                "physical_pinch_alignment_tolerance_m must be in [0.001, 0.01]"
            )
        if not 0.05 <= self.maximum_finger_center_separation_m <= 0.20:
            raise ValueError(
                "maximum_finger_center_separation_m must be in [0.05, 0.20]"
            )
        if not 0.5 <= self.minimum_finger_contact_ratio <= 0.9:
            raise ValueError("minimum_finger_contact_ratio must be in [0.5, 0.9]")
        if not 0.05 <= self.max_rotation_step_rad <= 0.30:
            raise ValueError("max_rotation_step_rad must be in [0.05, 0.30]")
        if not 0.01 <= self.rotation_tolerance_rad <= 0.05:
            raise ValueError("rotation_tolerance_rad must be in [0.01, 0.05]")
        if self.max_rotation_steps < 1:
            raise ValueError("max_rotation_steps must be positive")


def calibrated_control_target_world(
    *,
    control_frame_world_m: Sequence[float],
    physical_pinch_world_m: Sequence[float],
    desired_pinch_world_m: Sequence[float],
) -> np.ndarray:
    """Translate a physical pinch target into the controller-frame target.

    The Franka controller moves its configured control frame, which is not
    guaranteed to coincide with the midpoint between the two physical finger
    links.  Measuring both at runtime removes the fragile fixed TCP offset.
    """

    control = np.asarray(control_frame_world_m, dtype=float)
    physical = np.asarray(physical_pinch_world_m, dtype=float)
    desired = np.asarray(desired_pinch_world_m, dtype=float)
    if any(value.shape != (3,) for value in (control, physical, desired)):
        raise ValueError("control, physical pinch and desired positions must be 3-D")
    if not all(np.all(np.isfinite(value)) for value in (control, physical, desired)):
        raise ValueError("control, physical pinch and desired positions must be finite")
    return control + desired - physical


def minimum_symmetric_finger_contact_m(
    *, part_radius_m: float, minimum_contact_ratio: float
) -> float:
    """Return the per-finger opening expected when both fingers touch a part."""

    radius = float(part_radius_m)
    ratio = float(minimum_contact_ratio)
    if radius <= 0.0:
        raise ValueError("part_radius_m must be positive")
    if not 0.5 <= ratio <= 0.9:
        raise ValueError("minimum_contact_ratio must be in [0.5, 0.9]")
    return radius * ratio


def symmetric_finger_contact_report(
    finger_positions_m: Sequence[float], *, minimum_contact_m: float
) -> dict[str, object]:
    """Reject empty or one-sided closure before accepting a lift probe."""

    positions = np.asarray(finger_positions_m, dtype=float)
    threshold = float(minimum_contact_m)
    if positions.shape != (2,) or not np.all(np.isfinite(positions)):
        raise ValueError("finger_positions_m must contain two finite values")
    if np.any(positions < 0.0):
        raise ValueError("finger positions must be non-negative")
    if not 0.0 < threshold <= 0.04:
        raise ValueError("minimum_contact_m must be in (0.0, 0.04]")
    contacts = positions >= threshold
    passed = bool(np.all(contacts))
    return {
        "pass": passed,
        "reason": (
            "both fingers retained symmetric contact"
            if passed
            else "empty or one-sided gripper closure"
        ),
        "finger_positions_m": positions.tolist(),
        "minimum_contact_m": threshold,
        "contact_by_finger": contacts.tolist(),
    }


def top_down_tilt_error_rad(current_world_rotation: Sequence[Sequence[float]]) -> float:
    """Return tool-Z tilt from world down, deliberately ignoring tool yaw.

    P01 is an upright cylinder, so rotation about the vertical approach axis is
    not part of the grasp objective.  Measuring full SO(3) error would impose an
    arbitrary wrist yaw and can make an otherwise reachable grasp fail.
    """

    current = np.asarray(current_world_rotation, dtype=float)
    if current.shape != (3, 3) or not np.all(np.isfinite(current)):
        raise ValueError("current_world_rotation must be a finite 3-by-3 matrix")
    tool_z = current[:, 2]
    norm = float(np.linalg.norm(tool_z))
    if norm <= 1e-12:
        raise ValueError("tool Z axis must be non-zero")
    cosine = float(np.clip(np.dot(tool_z / norm, [0.0, 0.0, -1.0]), -1.0, 1.0))
    return float(np.arccos(cosine))


def yaw_preserving_top_down_rotation(
    current_world_rotation: Sequence[Sequence[float]],
) -> np.ndarray:
    """Build the nearest top-down target while preserving current wrist yaw.

    The returned right-handed rotation has tool Z aligned with world ``-Z``.
    Its tool X heading is the horizontal projection of the current tool X, so
    the planner corrects tilt without asking the redundant Panda wrist to turn
    toward a fixed, task-irrelevant yaw.
    """

    current = np.asarray(current_world_rotation, dtype=float)
    if current.shape != (3, 3) or not np.all(np.isfinite(current)):
        raise ValueError("current_world_rotation must be a finite 3-by-3 matrix")
    world_down = np.asarray([0.0, 0.0, -1.0], dtype=float)
    tool_x = current[:, 0].copy()
    tool_x[2] = 0.0
    if float(np.linalg.norm(tool_x)) <= 1e-9:
        projected_y = current[:, 1].copy()
        projected_y[2] = 0.0
        if float(np.linalg.norm(projected_y)) <= 1e-9:
            tool_x = np.asarray([1.0, 0.0, 0.0], dtype=float)
        else:
            projected_y /= np.linalg.norm(projected_y)
            tool_x = np.cross(projected_y, world_down)
    tool_x /= np.linalg.norm(tool_x)
    tool_y = np.cross(world_down, tool_x)
    tool_y /= np.linalg.norm(tool_y)
    return np.column_stack((tool_x, tool_y, world_down))


def grasp_follow_report(
    *,
    tcp_before_world_m: Sequence[float],
    tcp_after_world_m: Sequence[float],
    part_before_world_m: Sequence[float],
    part_after_world_m: Sequence[float],
    minimum_follow_ratio: float,
    maximum_follow_error_m: float,
) -> dict[str, float | bool | str]:
    """Verify that a closed-gripper probe lift actually carries the part."""

    tcp_before = np.asarray(tcp_before_world_m, dtype=float)
    tcp_after = np.asarray(tcp_after_world_m, dtype=float)
    part_before = np.asarray(part_before_world_m, dtype=float)
    part_after = np.asarray(part_after_world_m, dtype=float)
    if any(
        value.shape != (3,)
        for value in (tcp_before, tcp_after, part_before, part_after)
    ):
        raise ValueError("grasp verification positions must be 3-D")
    if not all(
        np.all(np.isfinite(value))
        for value in (tcp_before, tcp_after, part_before, part_after)
    ):
        raise ValueError("grasp verification positions must be finite")
    tcp_delta = tcp_after - tcp_before
    part_delta = part_after - part_before
    tcp_lift = float(tcp_delta[2])
    part_lift = float(part_delta[2])
    follow_error = float(np.linalg.norm(part_delta - tcp_delta))
    required_part_lift = max(0.01, tcp_lift * float(minimum_follow_ratio))
    passed = (
        tcp_lift >= 0.015
        and part_lift >= required_part_lift
        and follow_error <= float(maximum_follow_error_m)
    )
    reason = "part followed probe lift" if passed else "part did not follow probe lift"
    return {
        "pass": passed,
        "reason": reason,
        "tcp_lift_m": tcp_lift,
        "part_lift_m": part_lift,
        "required_part_lift_m": required_part_lift,
        "follow_error_m": follow_error,
    }


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


def bin_slot_local_centers(
    *,
    size_m: Sequence[float],
    wall_thickness_m: float,
    bottom_thickness_m: float,
    part_height_m: float,
) -> tuple[np.ndarray, ...]:
    """Return all six frozen 2-by-3 slot centers in bin-local axes."""

    first = first_bin_slot_local_center(
        size_m=size_m,
        wall_thickness_m=wall_thickness_m,
        bottom_thickness_m=bottom_thickness_m,
        part_height_m=part_height_m,
    )
    return tuple(
        np.asarray([column * abs(first[0]), row * abs(first[1]), first[2]])
        for row in (-1.0, 1.0)
        for column in (-1.0, 0.0, 1.0)
    )


def select_safest_slot_index(
    slot_tcp_world_m: Sequence[Sequence[float]],
    *,
    arm_base_world_m: Sequence[float],
    soft_work_radius_m: float,
    work_radius_margin_m: float,
) -> int:
    """Choose the reachable slot with the largest horizontal base clearance.

    The original P01 smoke hard-coded the bin corner closest to Arm_A.  For a
    downward gripper pose that target is in the Panda's cramped inner work
    region and can make redundant IK switch elbow branches.  Every returned
    candidate is still a valid frozen bin slot; this function only chooses the
    safest reachable one for the P01-only smoke.
    """

    candidates = np.asarray(slot_tcp_world_m, dtype=float)
    base = np.asarray(arm_base_world_m, dtype=float)
    if candidates.ndim != 2 or candidates.shape[1] != 3 or not len(candidates):
        raise ValueError("slot_tcp_world_m must contain one or more 3-D points")
    if base.shape != (3,) or not np.all(np.isfinite(base)):
        raise ValueError("arm_base_world_m must contain three finite values")
    if not np.all(np.isfinite(candidates)):
        raise ValueError("slot candidates must be finite")
    radius = float(soft_work_radius_m)
    margin = float(work_radius_margin_m)
    if radius <= 0.0 or margin <= 0.0 or margin >= radius:
        raise ValueError("work radius and margin are invalid")
    distances = np.linalg.norm(candidates - base, axis=1)
    reachable = np.flatnonzero(distances <= radius - margin)
    if not reachable.size:
        raise ValueError("no bin slot is inside the guarded Arm_A work radius")
    horizontal = np.linalg.norm(candidates[reachable, :2] - base[:2], axis=1)
    return int(reachable[int(np.argmax(horizontal))])


def minimum_xy_radius_along_segment(
    start_world_m: Sequence[float],
    end_world_m: Sequence[float],
    *,
    arm_base_world_m: Sequence[float],
) -> float:
    """Return the closest horizontal distance from a segment to the arm base."""

    start = np.asarray(start_world_m, dtype=float)
    end = np.asarray(end_world_m, dtype=float)
    base = np.asarray(arm_base_world_m, dtype=float)
    if any(value.shape != (3,) for value in (start, end, base)):
        raise ValueError("segment and base points must be 3-D")
    direction = end[:2] - start[:2]
    denominator = float(np.dot(direction, direction))
    if denominator <= 1e-15:
        closest = start[:2]
    else:
        fraction = float(
            np.clip(
                np.dot(base[:2] - start[:2], direction) / denominator,
                0.0,
                1.0,
            )
        )
        closest = start[:2] + direction * fraction
    return float(np.linalg.norm(closest - base[:2]))


def orthogonal_transfer_waypoints(
    start_world_m: Sequence[float],
    destination_world_m: Sequence[float],
    *,
    arm_base_world_m: Sequence[float],
    transit_clearance_m: float,
) -> tuple[np.ndarray, ...]:
    """Plan a raised L-shaped transfer on the side farther from the base."""

    start = np.asarray(start_world_m, dtype=float)
    destination = np.asarray(destination_world_m, dtype=float)
    base = np.asarray(arm_base_world_m, dtype=float)
    if any(value.shape != (3,) for value in (start, destination, base)):
        raise ValueError("transfer and base points must be 3-D")
    if not all(np.all(np.isfinite(value)) for value in (start, destination, base)):
        raise ValueError("transfer and base points must be finite")
    clearance = float(transit_clearance_m)
    if clearance <= 0.0:
        raise ValueError("transit_clearance_m must be positive")
    transit_z = max(float(start[2]), float(destination[2])) + clearance
    corners = (
        np.asarray([destination[0], start[1], transit_z], dtype=float),
        np.asarray([start[0], destination[1], transit_z], dtype=float),
    )

    def route_clearance(corner: np.ndarray) -> float:
        return min(
            minimum_xy_radius_along_segment(start, corner, arm_base_world_m=base),
            minimum_xy_radius_along_segment(corner, destination, arm_base_world_m=base),
        )

    corner = max(corners, key=route_clearance)
    over_destination = np.asarray(
        [destination[0], destination[1], transit_z], dtype=float
    )
    return corner, over_destination, destination.copy()


def motion_sample_violation(
    before_world_m: Sequence[float],
    after_world_m: Sequence[float],
    target_world_m: Sequence[float],
    *,
    max_actual_step_m: float,
    divergence_tolerance_m: float,
) -> str | None:
    """Describe a dangerous measured TCP sample, or return ``None``."""

    before = np.asarray(before_world_m, dtype=float)
    after = np.asarray(after_world_m, dtype=float)
    target = np.asarray(target_world_m, dtype=float)
    if any(value.shape != (3,) for value in (before, after, target)):
        raise ValueError("motion sample points must be 3-D")
    if not all(np.all(np.isfinite(value)) for value in (before, after, target)):
        return "non-finite TCP sample"
    actual_step = float(np.linalg.norm(after - before))
    if actual_step > float(max_actual_step_m):
        return (
            f"TCP jumped {actual_step:.6f} m; limit is {float(max_actual_step_m):.6f} m"
        )
    before_error = float(np.linalg.norm(target - before))
    after_error = float(np.linalg.norm(target - after))
    if after_error > before_error + float(divergence_tolerance_m):
        return f"TCP moved away from target: {before_error:.6f} -> {after_error:.6f} m"
    return None


def frozen_success_vote(votes: Sequence[bool]) -> bool:
    """Apply the leader-frozen exactly-three-fresh-frames, two-pass rule."""

    if len(votes) != 3 or any(not isinstance(vote, bool) for vote in votes):
        raise ValueError("success voting requires exactly three boolean frames")
    return sum(votes) >= 2
