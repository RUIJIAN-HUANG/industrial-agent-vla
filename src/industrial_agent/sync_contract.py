"""Frozen 7-D state and multi-rate synchronization contract.

This module is deliberately dependency-free so Schema adapters, model services,
and the Isaac boundary can share one source of truth without importing a model
runtime or Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isclose
from numbers import Real
from typing import Any, Sequence


STATE_7D_ORDER = (
    "x_m",
    "y_m",
    "z_m",
    "ax_rad",
    "ay_rad",
    "az_rad",
    "gripper_norm",
)

PHYSICS_HZ = 120
CONTROL_HZ = 60
RENDER_HZ = 30
MODEL_INFERENCE_HZ = 10

# Franka's two finger joints are represented as one total opening.  The
# calibrated fully-open width is 40 mm per finger (80 mm total); V2 State
# stores the measured opening as a continuous normalized value.
GRIPPER_FULLY_OPEN_WIDTH_M = 0.08


def canonical_state_7d(
    tcp_pose_m_rad: Any,
    gripper_open: Any,
) -> list[float]:
    """Build ``[x,y,z,ax,ay,az,gripper]`` in ``robot_base``.

    ``ax, ay, az`` are one rotation vector (axis multiplied by angle), never
    roll/pitch/yaw Euler angles.  A missing or ambiguous gripper readback is
    rejected instead of silently inventing model state.
    """

    if not isinstance(tcp_pose_m_rad, Sequence) or isinstance(
        tcp_pose_m_rad, (str, bytes, bytearray)
    ):
        raise TypeError("tcp_pose_m_rad must be a six-number sequence")
    if len(tcp_pose_m_rad) != 6:
        raise ValueError("tcp_pose_m_rad must contain exactly 6 values")
    if not isinstance(gripper_open, bool):
        raise TypeError("gripper_open must be a controller-confirmed boolean")

    values: list[float] = []
    for index, item in enumerate(tcp_pose_m_rad):
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise TypeError(f"tcp_pose_m_rad[{index}] must be numeric")
        value = float(item)
        if not isfinite(value):
            raise ValueError(f"tcp_pose_m_rad[{index}] must be finite")
        values.append(value)
    values.append(1.0 if gripper_open else 0.0)
    return values


def normalize_gripper_opening(
    finger_positions_m: Any,
    *,
    closed_width_m: float = 0.0,
    open_width_m: float = GRIPPER_FULLY_OPEN_WIDTH_M,
) -> float:
    """Normalize measured two-finger opening to the V2 continuous [0, 1] value.

    ``finger_positions_m`` must contain the two measured finger joint openings.
    The command/boolean path is deliberately not accepted here: V2 State must
    describe sensor readback, including partially open grippers.
    """

    try:
        if len(finger_positions_m) != 2:
            raise ValueError("finger_positions_m must contain exactly two values")
    except TypeError as exc:
        raise TypeError("finger_positions_m must be a two-number sequence") from exc
    if (
        isinstance(closed_width_m, bool)
        or not isinstance(closed_width_m, Real)
        or not isfinite(float(closed_width_m))
        or isinstance(open_width_m, bool)
        or not isinstance(open_width_m, Real)
        or not isfinite(float(open_width_m))
        or float(open_width_m) <= float(closed_width_m)
    ):
        raise ValueError("gripper calibration widths must be finite and increasing")
    opening_m = 0.0
    for index, item in enumerate(finger_positions_m):
        if isinstance(item, bool) or not isinstance(item, Real):
            raise TypeError(f"finger_positions_m[{index}] must be numeric")
        value = float(item)
        if not isfinite(value):
            raise ValueError(f"finger_positions_m[{index}] must be finite")
        opening_m += value
    normalized = (opening_m - float(closed_width_m)) / (
        float(open_width_m) - float(closed_width_m)
    )
    # Small articulation/readback overshoot is clipped at the physical bounds;
    # NaN/Infinity were rejected above and cannot be hidden by this clamp.
    return float(min(1.0, max(0.0, normalized)))


def canonical_state_7d_from_opening(
    tcp_pose_m_rad: Any,
    gripper_opening_norm: Any,
) -> list[float]:
    """Build a V2 State from a measured continuous normalized gripper opening."""

    if isinstance(gripper_opening_norm, bool) or not isinstance(
        gripper_opening_norm, Real
    ):
        raise TypeError("gripper_opening_norm must be a numeric measurement")
    opening = float(gripper_opening_norm)
    if not isfinite(opening) or not 0.0 <= opening <= 1.0:
        raise ValueError("gripper_opening_norm must be finite and within [0,1]")
    pose = canonical_state_7d(tcp_pose_m_rad, True)[:6]
    pose.append(opening)
    return pose


@dataclass(frozen=True)
class MultiRateContract:
    """Integer-ratio schedule shared by Agent, controller, renderer, and Isaac."""

    physics_hz: int = PHYSICS_HZ
    control_hz: int = CONTROL_HZ
    render_hz: int = RENDER_HZ
    model_inference_hz: int = MODEL_INFERENCE_HZ

    def __post_init__(self) -> None:
        rates = (
            self.physics_hz,
            self.control_hz,
            self.render_hz,
            self.model_inference_hz,
        )
        if any(isinstance(rate, bool) or not isinstance(rate, int) for rate in rates):
            raise TypeError("all frozen rates must be integers")
        if any(rate <= 0 for rate in rates):
            raise ValueError("all frozen rates must be positive")
        if self.physics_hz % self.control_hz:
            raise ValueError("physics_hz must be an integer multiple of control_hz")
        if self.control_hz % self.render_hz:
            raise ValueError("control_hz must be an integer multiple of render_hz")
        if self.render_hz % self.model_inference_hz:
            raise ValueError(
                "render_hz must be an integer multiple of model_inference_hz"
            )

    @property
    def physics_ticks_per_control(self) -> int:
        return self.physics_hz // self.control_hz

    @property
    def physics_ticks_per_render(self) -> int:
        return self.physics_hz // self.render_hz

    @property
    def control_ticks_per_model_step(self) -> int:
        return self.control_hz // self.model_inference_hz

    @property
    def physics_ticks_per_model_step(self) -> int:
        return self.physics_hz // self.model_inference_hz

    @property
    def render_frames_per_model_step(self) -> int:
        return self.render_hz // self.model_inference_hz

    @property
    def model_step_duration_ms(self) -> int:
        if 1000 % self.model_inference_hz:
            raise ValueError("model inference rate must divide 1000ms exactly")
        return 1000 // self.model_inference_hz

    def physics_ticks_for_duration_ms(self, duration_ms: int) -> int:
        """Return an exact physics-grid duration or fail closed.

        Physical execution must not round timing metadata because that makes
        chunk boundaries drift.  At 120Hz, integer millisecond durations must
        therefore map to a whole number of physics ticks.
        """

        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
            raise TypeError("duration_ms must be an integer")
        if duration_ms < 1:
            raise ValueError("duration_ms must be positive")
        exact_ticks = duration_ms * self.physics_hz / 1000.0
        rounded_ticks = round(exact_ticks)
        if not isclose(exact_ticks, rounded_ticks, abs_tol=1e-12, rel_tol=0.0):
            raise ValueError(
                f"duration_ms={duration_ms} is not aligned to the "
                f"{self.physics_hz}Hz physics grid"
            )
        return int(rounded_ticks)

    def control_ticks_for_duration_ms(self, duration_ms: int) -> int:
        """Return an exact control-grid duration or fail closed."""

        physics_ticks = self.physics_ticks_for_duration_ms(duration_ms)
        stride = self.physics_ticks_per_control
        if physics_ticks % stride:
            raise ValueError(
                f"duration_ms={duration_ms} is not aligned to the "
                f"{self.control_hz}Hz control grid"
            )
        return physics_ticks // stride


FROZEN_MULTI_RATE = MultiRateContract()
