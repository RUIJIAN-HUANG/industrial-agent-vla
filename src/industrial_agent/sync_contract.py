"""Frozen 7-D state and multi-rate synchronization contract.

This module is deliberately dependency-free so Schema adapters, model services,
and the Isaac boundary can share one source of truth without importing a model
runtime or Isaac Sim.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, isclose
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
