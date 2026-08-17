"""Isaac Sim 5.1 Franka backend for ``IsaacExecutionEnvironment``.

Import this module only after ``SimulationApp`` has started.  It converts the
frozen robot-base 7-D action contract into Lula or Pink inverse-kinematics
targets and writes them to the selected ``SingleArticulation``.
"""

from __future__ import annotations

from collections.abc import Callable
from math import atan2, cos, isclose, sin, sqrt
from threading import Event, Lock, get_ident
from typing import Any, Mapping

import numpy as np

from industrial_agent.contracts import ActionStep
from industrial_agent.environment import SafeStopReceipt
from industrial_agent.sync_contract import FROZEN_MULTI_RATE


_ARMS = ("Arm_A", "Arm_B")
_FINGER_JOINTS = ("panda_finger_joint1", "panda_finger_joint2")


def _quat_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Multiply wxyz quaternions."""

    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    return np.asarray(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=float,
    )


def _quat_inverse(quaternion: np.ndarray) -> np.ndarray:
    norm_squared = float(np.dot(quaternion, quaternion))
    if norm_squared <= 0.0:
        raise RuntimeError("Isaac returned a zero-norm base quaternion")
    result = quaternion.copy()
    result[1:] *= -1.0
    return result / norm_squared


def _rotvec_quaternion(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)
    axis = rotvec / angle
    return np.concatenate((np.asarray([cos(angle / 2.0)]), axis * sin(angle / 2.0)))


def _quaternion_to_rotvec(quaternion: np.ndarray) -> np.ndarray:
    """Convert a wxyz quaternion to the shortest axis-angle vector."""

    quaternion = np.asarray(quaternion, dtype=float)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise ValueError("quaternion must contain four finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise ValueError("quaternion cannot have zero norm")
    quaternion = quaternion / norm
    if quaternion[0] < 0.0:
        quaternion = -quaternion
    vector_norm = float(np.linalg.norm(quaternion[1:]))
    if vector_norm < 1e-12:
        return np.zeros(3, dtype=float)
    angle = 2.0 * atan2(vector_norm, float(quaternion[0]))
    return quaternion[1:] * (angle / vector_norm)


def _rotate_vector(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    pure = np.concatenate((np.asarray([0.0]), vector))
    return _quat_multiply(_quat_multiply(quaternion, pure), _quat_inverse(quaternion))[
        1:
    ]


def _rotation_matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""

    matrix = np.asarray(matrix, dtype=float)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise RuntimeError("Lula returned an invalid end-effector rotation matrix")

    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = 2.0 * sqrt(trace + 1.0)
        quaternion = np.asarray(
            [
                0.25 * scale,
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
            ]
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            scale = 2.0 * sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2])
            quaternion = np.asarray(
                [
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                ]
            )
        elif axis == 1:
            scale = 2.0 * sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2])
            quaternion = np.asarray(
                [
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                ]
            )
        else:
            scale = 2.0 * sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1])
            quaternion = np.asarray(
                [
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                ]
            )
    norm = float(np.linalg.norm(quaternion))
    if norm <= 0.0:
        raise RuntimeError("Lula returned a zero-norm end-effector orientation")
    return quaternion / norm


def _midpoint_tcp_offset_local(
    *,
    control_position_world_m: np.ndarray,
    control_rotation_world: np.ndarray,
    left_tip_world_m: np.ndarray,
    right_tip_world_m: np.ndarray,
) -> np.ndarray:
    """Return the rigid control-frame offset to the two-fingertip midpoint."""

    control_position = np.asarray(control_position_world_m, dtype=float)
    control_rotation = np.asarray(control_rotation_world, dtype=float)
    left_tip = np.asarray(left_tip_world_m, dtype=float)
    right_tip = np.asarray(right_tip_world_m, dtype=float)
    if any(value.shape != (3,) for value in (control_position, left_tip, right_tip)):
        raise ValueError("control and fingertip positions must be 3-D")
    if control_rotation.shape != (3, 3):
        raise ValueError("control rotation must be 3-by-3")
    if not all(
        np.all(np.isfinite(value))
        for value in (control_position, control_rotation, left_tip, right_tip)
    ):
        raise ValueError("virtual TCP calibration values must be finite")
    if not np.allclose(
        control_rotation.T @ control_rotation,
        np.eye(3),
        atol=1e-6,
        rtol=0.0,
    ):
        raise ValueError("control rotation must be orthonormal")
    midpoint_world = (left_tip + right_tip) / 2.0
    return control_rotation.T @ (midpoint_world - control_position)


def _virtual_tcp_world_position(
    control_position_world_m: np.ndarray,
    control_rotation_world: np.ndarray,
    tcp_offset_local_m: np.ndarray,
) -> np.ndarray:
    """Transform a rigid local TCP offset into its current world position."""

    return np.asarray(control_position_world_m, dtype=float) + np.asarray(
        control_rotation_world, dtype=float
    ) @ np.asarray(tcp_offset_local_m, dtype=float)


def _control_world_position_for_tcp(
    tcp_position_world_m: np.ndarray,
    tcp_orientation_world_wxyz: np.ndarray,
    tcp_offset_local_m: np.ndarray,
) -> np.ndarray:
    """Convert a desired virtual-TCP pose into the Lula control-frame position."""

    return np.asarray(tcp_position_world_m, dtype=float) - _rotate_vector(
        np.asarray(tcp_orientation_world_wxyz, dtype=float),
        np.asarray(tcp_offset_local_m, dtype=float),
    )


def _position_targets_match(controller: Any, expected_positions: np.ndarray) -> bool:
    """Confirm that the controller accepted a full-articulation hold target."""

    get_applied_action = getattr(controller, "get_applied_action", None)
    if not callable(get_applied_action):
        return False
    try:
        applied_action = get_applied_action()
        applied_positions = np.asarray(
            applied_action.joint_positions,
            dtype=float,
        )
    except (AttributeError, TypeError, ValueError):
        return False
    expected_positions = np.asarray(expected_positions, dtype=float)
    return bool(
        applied_positions.shape == expected_positions.shape
        and applied_positions.size
        and np.all(np.isfinite(applied_positions))
        and np.allclose(applied_positions, expected_positions, atol=1e-9, rtol=0.0)
    )


def _gripper_opening_m(command: float) -> float:
    """Map the frozen normalized binary command to one finger position."""

    return 0.04 if float(command) >= 0.5 else 0.0


class IsaacSimFrankaController:
    """Live dual-Franka controller with Lula FK and selectable IK."""

    def __init__(
        self,
        *,
        world: Any,
        arms: Mapping[str, Any],
        physics_dt_s: float,
        end_effector_frame_name: str = "right_gripper",
        virtual_tcp_fingertip_frame_names: tuple[str, str] | None = None,
        stationary_velocity_rad_s: float = 1e-3,
        safe_stop_action_grace_s: float = 0.25,
        ik_backend: str = "lula",
        pink_device: str = "cuda:0",
    ) -> None:
        if set(arms) != set(_ARMS):
            raise ValueError("arms must contain exactly Arm_A and Arm_B")
        if physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive")
        if not isclose(
            physics_dt_s,
            1.0 / FROZEN_MULTI_RATE.physics_hz,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                f"physics_dt_s must be exactly 1/{FROZEN_MULTI_RATE.physics_hz}"
            )
        if safe_stop_action_grace_s < 0.0:
            raise ValueError("safe_stop_action_grace_s cannot be negative")
        if ik_backend not in {"lula", "pink"}:
            raise ValueError("ik_backend must be 'lula' or 'pink'")

        try:
            from isaacsim.robot_motion.motion_generation import (
                ArticulationKinematicsSolver,
                LulaKinematicsSolver,
                interface_config_loader,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Sim 5.1 Lula motion-generation extension is unavailable"
            ) from exc

        self._world = world
        self._arms = dict(arms)
        self._physics_dt_s = float(physics_dt_s)
        self._multi_rate = FROZEN_MULTI_RATE
        self._physics_tick_index = 0
        self._tick_observer: Callable[[int, bool], None] | None = None
        self._stationary_velocity_rad_s = float(stationary_velocity_rad_s)
        self._safe_stop_action_grace_s = float(safe_stop_action_grace_s)
        self._owner_thread_id = get_ident()
        self._action_lock = Lock()
        self._action_idle = Event()
        self._action_idle.set()
        self._stop_requested = Event()
        self._stop_epoch_lock = Lock()
        self._stop_epoch = 0
        self._solvers: dict[str, Any] = {}
        self._lula_solvers: dict[str, Any] = {}
        self._lula_configs: dict[str, Mapping[str, Any]] = {}
        self._ik_backend = ik_backend
        self._pink_adapter: Any | None = None
        self._control_frame_name = end_effector_frame_name
        self._tcp_offsets_local_m: dict[str, np.ndarray] = {}
        self._tcp_definitions: dict[str, dict[str, Any]] = {}
        for arm_id, arm in self._arms.items():
            config = (
                interface_config_loader.load_supported_lula_kinematics_solver_config(
                    "Franka"
                )
            )
            self._lula_configs[arm_id] = dict(config)
            lula_solver = LulaKinematicsSolver(**config)
            self._lula_solvers[arm_id] = lula_solver
            control_solver = ArticulationKinematicsSolver(
                arm,
                lula_solver,
                end_effector_frame_name,
            )
            self._solvers[arm_id] = control_solver
            if virtual_tcp_fingertip_frame_names is None:
                self._tcp_offsets_local_m[arm_id] = np.zeros(3, dtype=float)
                self._tcp_definitions[arm_id] = {
                    "mode": "lula_frame",
                    "control_frame_name": end_effector_frame_name,
                    "tcp_frame_name": end_effector_frame_name,
                    "tcp_offset_local_m": [0.0, 0.0, 0.0],
                }
                continue

            valid_frames = set(lula_solver.get_all_frame_names())
            missing = [
                name
                for name in virtual_tcp_fingertip_frame_names
                if name not in valid_frames
            ]
            if missing:
                raise RuntimeError(
                    "Lula virtual TCP fingertip frames are unavailable: "
                    + ", ".join(missing)
                )
            base_position, base_orientation = arm.get_world_pose()
            lula_solver.set_robot_base_pose(
                np.asarray(base_position, dtype=float),
                np.asarray(base_orientation, dtype=float),
            )
            left_solver = ArticulationKinematicsSolver(
                arm,
                lula_solver,
                virtual_tcp_fingertip_frame_names[0],
            )
            right_solver = ArticulationKinematicsSolver(
                arm,
                lula_solver,
                virtual_tcp_fingertip_frame_names[1],
            )
            control_position, control_rotation = (
                control_solver.compute_end_effector_pose()
            )
            left_position, _ = left_solver.compute_end_effector_pose()
            right_position, _ = right_solver.compute_end_effector_pose()
            offset = _midpoint_tcp_offset_local(
                control_position_world_m=control_position,
                control_rotation_world=control_rotation,
                left_tip_world_m=left_position,
                right_tip_world_m=right_position,
            )
            separation_m = float(
                np.linalg.norm(
                    np.asarray(left_position, dtype=float)
                    - np.asarray(right_position, dtype=float)
                )
            )
            if not 0.0 < separation_m <= 0.20:
                raise RuntimeError(
                    "Lula fingertip separation is invalid for virtual TCP: "
                    f"{separation_m:.6f} m"
                )
            self._tcp_offsets_local_m[arm_id] = offset
            self._tcp_definitions[arm_id] = {
                "mode": "virtual_two_fingertip_midpoint",
                "control_frame_name": end_effector_frame_name,
                "fingertip_frame_names": list(virtual_tcp_fingertip_frame_names),
                "tcp_offset_local_m": offset.tolist(),
                "calibration_fingertip_separation_m": separation_m,
            }

        if self._ik_backend == "pink":
            from simulation.pink_franka_adapter import PinkFrankaAdapter

            self._pink_adapter = PinkFrankaAdapter(
                arms=self._arms,
                lula_configs=self._lula_configs,
                control_frame_name=end_effector_frame_name,
                device=pink_device,
            )

    @property
    def ik_backend(self) -> str:
        """Return the selected live inverse-kinematics backend."""

        return self._ik_backend

    @property
    def physics_tick_index(self) -> int:
        """Return the current 120 Hz episode-local physics tick."""

        return self._physics_tick_index

    def set_tick_observer(
        self,
        observer: Callable[[int, bool], None] | None,
    ) -> None:
        """Attach the canonical recorder hook before executing actions.

        The callback runs on the Isaac owner thread after every physics step.
        ``render_due`` is true only on the frozen 30 Hz render grid.  Replacing
        the callback while an action is active is rejected so one action can
        never be split across two recorders.
        """

        self._require_owner_thread()
        if observer is not None and not callable(observer):
            raise TypeError("observer must be callable or None")
        if not self._action_idle.is_set():
            raise RuntimeError("cannot replace tick observer during an action")
        self._tick_observer = observer

    def _is_owner_thread(self) -> bool:
        return get_ident() == self._owner_thread_id

    def _require_owner_thread(self) -> None:
        if not self._is_owner_thread():
            raise RuntimeError(
                "Isaac API call rejected outside the controlled runtime thread"
            )

    @staticmethod
    def _joint_names(arm: Any) -> list[str]:
        names = getattr(arm, "dof_names", None)
        if names is None:
            names = getattr(arm, "joint_names", None)
        if names is None:
            raise RuntimeError("Franka articulation exposes no DOF names")
        return [str(name) for name in names]

    def _is_stationary(self, arm_id: str) -> bool:
        velocities = np.asarray(self._arms[arm_id].get_joint_velocities(), dtype=float)
        return bool(
            velocities.size
            and np.all(np.isfinite(velocities))
            and np.max(np.abs(velocities)) <= self._stationary_velocity_rad_s
        )

    def validate_ready(self, arm_id: str) -> None:
        self._require_owner_thread()
        if arm_id not in self._arms:
            raise RuntimeError(f"unknown Isaac Franka arm: {arm_id!r}")
        if self._stop_requested.is_set():
            raise RuntimeError("Isaac Franka controller is stopped and quarantined")
        other_arm = "Arm_B" if arm_id == "Arm_A" else "Arm_A"
        if not self._is_stationary(other_arm):
            raise RuntimeError(
                f"{arm_id} controller interlock requires {other_arm} stationary"
            )

    def end_effector_pose(self, arm_id: str) -> tuple[np.ndarray, np.ndarray]:
        """Return the live world-frame virtual TCP position and rotation."""

        self._require_owner_thread()
        if arm_id not in self._arms:
            raise RuntimeError(f"unknown Isaac Franka arm: {arm_id!r}")
        arm = self._arms[arm_id]
        base_position, base_orientation = arm.get_world_pose()
        self._lula_solvers[arm_id].set_robot_base_pose(
            np.asarray(base_position, dtype=float),
            np.asarray(base_orientation, dtype=float),
        )
        control_position, rotation = self._solvers[arm_id].compute_end_effector_pose()
        control_position = np.asarray(control_position, dtype=float)
        rotation = np.asarray(rotation, dtype=float)
        if (
            control_position.shape != (3,)
            or not np.all(np.isfinite(control_position))
            or rotation.shape != (3, 3)
            or not np.all(np.isfinite(rotation))
        ):
            raise RuntimeError(f"Isaac returned an invalid TCP pose for {arm_id}")
        offset = getattr(self, "_tcp_offsets_local_m", {}).get(
            arm_id, np.zeros(3, dtype=float)
        )
        position = _virtual_tcp_world_position(
            control_position,
            rotation,
            offset,
        )
        return position, rotation

    def tcp_definition(self, arm_id: str) -> dict[str, Any]:
        """Return auditable metadata for the frame controlled as the TCP."""

        self._require_owner_thread()
        if arm_id not in self._arms:
            raise RuntimeError(f"unknown Isaac Franka arm: {arm_id!r}")
        definitions = getattr(self, "_tcp_definitions", {})
        if arm_id not in definitions:
            return {
                "mode": "lula_frame",
                "control_frame_name": getattr(
                    self, "_control_frame_name", "right_gripper"
                ),
                "tcp_offset_local_m": [0.0, 0.0, 0.0],
            }
        return dict(definitions[arm_id])

    def end_effector_pose_in_base(
        self,
        arm_id: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return TCP position and wxyz orientation in ``robot_base``."""

        world_position, world_rotation = self.end_effector_pose(arm_id)
        base_position, base_orientation = self._arms[arm_id].get_world_pose()
        base_position = np.asarray(base_position, dtype=float)
        base_orientation = np.asarray(base_orientation, dtype=float)
        inverse_base = _quat_inverse(base_orientation)
        base_frame_position = _rotate_vector(
            inverse_base,
            world_position - base_position,
        )
        world_orientation = _rotation_matrix_to_quaternion(world_rotation)
        base_frame_orientation = _quat_multiply(
            inverse_base,
            world_orientation,
        )
        norm = float(np.linalg.norm(base_frame_orientation))
        if norm <= 0.0 or not np.isfinite(norm):
            raise RuntimeError(
                f"Isaac returned an invalid base-frame TCP orientation for {arm_id}"
            )
        return base_frame_position, base_frame_orientation / norm

    def world_orientation_error_in_base(
        self,
        arm_id: str,
        target_world_rotation: np.ndarray,
    ) -> np.ndarray:
        """Return the base-frame rotvec that aligns TCP with a world target."""

        _, current_world_rotation = self.end_effector_pose(arm_id)
        target_world = _rotation_matrix_to_quaternion(target_world_rotation)
        current_world = _rotation_matrix_to_quaternion(current_world_rotation)
        delta_world = _quat_multiply(target_world, _quat_inverse(current_world))
        _, base_orientation = self._arms[arm_id].get_world_pose()
        base_orientation = np.asarray(base_orientation, dtype=float)
        delta_base = _quat_multiply(
            _quat_multiply(_quat_inverse(base_orientation), delta_world),
            base_orientation,
        )
        return _quaternion_to_rotvec(delta_base)

    def gripper_joint_positions(self, arm_id: str) -> np.ndarray:
        """Read both live finger positions in metres from the articulation."""

        self._require_owner_thread()
        if arm_id not in self._arms:
            raise RuntimeError(f"unknown Isaac Franka arm: {arm_id!r}")
        arm = self._arms[arm_id]
        names = self._joint_names(arm)
        try:
            indices = [names.index(name) for name in _FINGER_JOINTS]
        except ValueError as exc:
            raise RuntimeError(
                f"{arm_id} Franka finger joints are missing from {names!r}"
            ) from exc
        positions = np.asarray(arm.get_joint_positions(), dtype=float)
        if positions.ndim != 1 or positions.size <= max(indices):
            raise RuntimeError(f"Isaac returned invalid joint positions for {arm_id}")
        fingers = positions[indices]
        if not np.all(np.isfinite(fingers)):
            raise RuntimeError(
                f"Isaac returned non-finite finger positions for {arm_id}"
            )
        return fingers.copy()

    def action_rejection_reason(self, action: ActionStep, *, arm_id: str) -> str | None:
        """Return an IK rejection reason without moving the robot."""

        self._require_owner_thread()
        if self._stop_requested.is_set():
            return "controller is already safe-stopped"
        # Pink's QP updates its internal configuration while solving.  A Lula
        # dry-run here would reject poses that Pink can recover through the
        # redundant joint null space, defeating the selected backend.
        if getattr(self, "_ik_backend", "lula") == "pink":
            return None
        arm = self._arms[arm_id]
        solver = self._solvers[arm_id]
        lula_solver = self._lula_solvers[arm_id]
        translation = np.asarray(action.values[:3], dtype=float)
        rotation = np.asarray(action.values[3:6], dtype=float)
        control_ticks = self._multi_rate.control_ticks_for_duration_ms(
            action.duration_ms
        )
        base_position, base_orientation = arm.get_world_pose()
        base_position = np.asarray(base_position, dtype=float)
        base_orientation = np.asarray(base_orientation, dtype=float)
        lula_solver.set_robot_base_pose(base_position, base_orientation)
        control_position, current_rotation = solver.compute_end_effector_pose()
        control_position = np.asarray(control_position, dtype=float)
        current_rotation = np.asarray(current_rotation, dtype=float)
        tcp_offset_local = getattr(self, "_tcp_offsets_local_m", {}).get(
            arm_id, np.zeros(3, dtype=float)
        )
        current_position = _virtual_tcp_world_position(
            control_position,
            current_rotation,
            tcp_offset_local,
        )
        current_orientation = _rotation_matrix_to_quaternion(current_rotation)
        world_translation = _rotate_vector(base_orientation, translation)
        inverse_base_orientation = _quat_inverse(base_orientation)
        for control_index in range(1, control_ticks + 1):
            fraction = control_index / control_ticks
            target_tcp_position = current_position + world_translation * fraction
            delta_base = _rotvec_quaternion(rotation * fraction)
            delta_world = _quat_multiply(
                _quat_multiply(base_orientation, delta_base),
                inverse_base_orientation,
            )
            target_orientation = _quat_multiply(
                delta_world,
                current_orientation,
            )
            target_orientation /= sqrt(
                float(np.dot(target_orientation, target_orientation))
            )
            target_position = _control_world_position_for_tcp(
                target_tcp_position,
                target_orientation,
                tcp_offset_local,
            )
            _, success = solver.compute_inverse_kinematics(
                target_position,
                target_orientation,
            )
            if not success:
                return (
                    f"{arm_id} cannot reach this target "
                    f"(IK tick {control_index}/{control_ticks})"
                )
        return None

    def execute_action(self, action: ActionStep, *, arm_id: str) -> None:
        self._require_owner_thread()
        if not self._action_lock.acquire(blocking=False):
            raise RuntimeError("Isaac Franka controller rejected concurrent action")
        self._action_idle.clear()
        try:
            if self._stop_requested.is_set():
                raise RuntimeError(
                    "Isaac Franka controller rejected action after safe-stop"
                )
            arm = self._arms[arm_id]
            solver = self._solvers[arm_id]
            lula_solver = self._lula_solvers[arm_id]
            translation = np.asarray(action.values[:3], dtype=float)
            rotation = np.asarray(action.values[3:6], dtype=float)
            control_ticks = self._multi_rate.control_ticks_for_duration_ms(
                action.duration_ms
            )
            # The frozen canonical command is binary at the hardware boundary:
            # values >= 0.5 mean open, and values < 0.5 mean closed. This maps
            # pi0.5's 0/1 and OpenVLA-OFT's -1/+1 endpoints identically.
            finger_position_m = _gripper_opening_m(action.values[6])

            base_position, base_orientation = arm.get_world_pose()
            base_position = np.asarray(base_position, dtype=float)
            base_orientation = np.asarray(base_orientation, dtype=float)
            # Lula assumes its robot base is at the world origin unless this pose is
            # refreshed.  The frozen scene places two Frankas away from the origin,
            # so omitting it produces valid-looking but incorrect IK targets.
            lula_solver.set_robot_base_pose(base_position, base_orientation)
            control_position, current_rotation = solver.compute_end_effector_pose()
            control_position = np.asarray(control_position, dtype=float)
            current_rotation = np.asarray(current_rotation, dtype=float)
            tcp_offset_local = getattr(self, "_tcp_offsets_local_m", {}).get(
                arm_id, np.zeros(3, dtype=float)
            )
            current_position = _virtual_tcp_world_position(
                control_position,
                current_rotation,
                tcp_offset_local,
            )
            current_orientation = _rotation_matrix_to_quaternion(current_rotation)

            pink_current_position_base = None
            pink_current_orientation_base = None
            if getattr(self, "_ik_backend", "lula") == "pink":
                current_joints = np.asarray(
                    arm.get_joint_positions(), dtype=float
                )
                pink_controller = self._pink_adapter._controllers[arm_id]
                ordered_joints = current_joints[
                    pink_controller.isaac_lab_to_pink_ordering
                ]
                pink_controller.pink_configuration.update(ordered_joints)
                pink_transform = (
                    pink_controller.pink_configuration
                    .get_transform_frame_to_world(self._control_frame_name)
                )
                pink_control_position = np.asarray(
                    pink_transform.translation, dtype=float
                )
                pink_control_rotation = np.asarray(
                    pink_transform.rotation, dtype=float
                )
                pink_current_position_base = _virtual_tcp_world_position(
                    pink_control_position,
                    pink_control_rotation,
                    tcp_offset_local,
                )
                pink_current_orientation_base = (
                    _rotation_matrix_to_quaternion(pink_control_rotation)
                )
            try:
                from isaacsim.core.utils.types import ArticulationAction
            except ImportError as exc:
                raise RuntimeError(
                    "Isaac Sim 5.1 ArticulationAction is unavailable"
                ) from exc
            names = self._joint_names(arm)
            try:
                finger_indices = [names.index(name) for name in _FINGER_JOINTS]
            except ValueError as exc:
                raise RuntimeError(
                    f"{arm_id} Franka finger joints are missing from {names!r}"
                ) from exc
            if self._stop_requested.is_set():
                raise RuntimeError("control lease was revoked before simulation play")
            play = getattr(self._world, "play", None)
            if callable(play):
                play()

            # One 10Hz model delta spans exactly six 60Hz controller updates.
            # Each controller update advances two 120Hz physics ticks, while
            # every fourth global physics tick renders one 30Hz camera frame.
            # Cartesian interpolation prevents applying the full delta six
            # times and keeps ActionChunk boundaries phase aligned.
            world_translation = _rotate_vector(base_orientation, translation)
            inverse_base_orientation = _quat_inverse(base_orientation)
            for control_index in range(1, control_ticks + 1):
                if self._stop_requested.is_set():
                    raise RuntimeError(
                        "control lease was revoked during action execution"
                    )
                fraction = control_index / control_ticks
                target_tcp_position = (
                    np.asarray(current_position, dtype=float)
                    + world_translation * fraction
                )
                delta_base = _rotvec_quaternion(rotation * fraction)
                delta_world = _quat_multiply(
                    _quat_multiply(base_orientation, delta_base),
                    inverse_base_orientation,
                )
                target_orientation = _quat_multiply(
                    delta_world,
                    np.asarray(current_orientation, dtype=float),
                )
                target_orientation /= sqrt(
                    float(np.dot(target_orientation, target_orientation))
                )
                target_position = _control_world_position_for_tcp(
                    target_tcp_position,
                    target_orientation,
                    tcp_offset_local,
                )
                if getattr(self, "_ik_backend", "lula") == "pink":
                    if (
                        pink_current_position_base is None
                        or pink_current_orientation_base is None
                    ):
                        raise RuntimeError(
                            "Pink action origin was not initialized"
                        )
                    target_tcp_position_base = (
                        pink_current_position_base
                        + translation * fraction
                    )
                    target_orientation_base = _quat_multiply(
                        delta_base,
                        pink_current_orientation_base,
                    )
                    target_orientation_base /= sqrt(
                        float(
                            np.dot(
                                target_orientation_base,
                                target_orientation_base,
                            )
                        )
                    )
                    target_position_base = _control_world_position_for_tcp(
                        target_tcp_position_base,
                        target_orientation_base,
                        tcp_offset_local,
                    )
                    current_joints = np.asarray(arm.get_joint_positions(), dtype=float)
                    joint_targets = self._pink_adapter.compute(
                        arm_id=arm_id,
                        current_joint_positions=current_joints,
                        target_position_base_m=target_position_base,
                        target_orientation_base_wxyz=target_orientation_base,
                        dt_s=1.0 / self._multi_rate.control_hz,
                    )
                    ik_action = ArticulationAction(
                        joint_positions=joint_targets,
                        joint_indices=np.asarray(
                            self._pink_adapter.controlled_indices(arm_id),
                            dtype=np.int64,
                        ),
                    )
                else:
                    ik_action, success = solver.compute_inverse_kinematics(
                        target_position,
                        target_orientation,
                    )
                    if not success:
                        raise RuntimeError(
                            f"Lula IK did not converge for {arm_id} at control tick "
                            f"{control_index}/{control_ticks}"
                        )
                if self._stop_requested.is_set():
                    raise RuntimeError("control lease was revoked before IK write")
                arm.apply_action(ik_action)
                if self._stop_requested.is_set():
                    raise RuntimeError("control lease was revoked before gripper write")
                arm.apply_action(
                    ArticulationAction(
                        joint_positions=np.asarray(
                            [finger_position_m, finger_position_m], dtype=float
                        ),
                        joint_indices=np.asarray(finger_indices, dtype=np.int64),
                    )
                )

                for _ in range(self._multi_rate.physics_ticks_per_control):
                    if self._stop_requested.is_set():
                        raise RuntimeError(
                            "control lease was revoked during action execution"
                        )
                    self._physics_tick_index += 1
                    render_due = (
                        self._physics_tick_index
                        % self._multi_rate.physics_ticks_per_render
                        == 0
                    )
                    self._world.step(render=render_due)
                    observer = getattr(self, "_tick_observer", None)
                    if observer is not None:
                        observer(self._physics_tick_index, render_due)
                    if self._stop_requested.is_set():
                        raise RuntimeError(
                            "control lease was revoked during action execution"
                        )
        finally:
            self._action_idle.set()
            self._action_lock.release()

    def request_stop(self, reason: str) -> str:
        """Thread-safely revoke motion without entering an Isaac API."""

        del reason
        with self._stop_epoch_lock:
            if not self._stop_requested.is_set():
                self._stop_epoch += 1
                self._stop_requested.set()
            return f"controller-stop-{self._stop_epoch}"

    def confirm_safe_stop(
        self,
        reason: str,
        *,
        stop_epoch: str,
    ) -> SafeStopReceipt:
        """Apply hold/pause and read it back on the Isaac owner thread."""

        del reason
        self._require_owner_thread()
        with self._stop_epoch_lock:
            expected_epoch = f"controller-stop-{self._stop_epoch}"
            epoch_is_current = (
                self._stop_requested.is_set() and stop_epoch == expected_epoch
            )
        if not epoch_is_current:
            return SafeStopReceipt(
                controller_ack=False,
                buffers_cleared=False,
                arm_a_stopped=False,
                arm_b_stopped=False,
                stop_epoch=stop_epoch or expected_epoch,
            )
        # Never enter Isaac APIs concurrently with a normal world.step call.
        # The gate only schedules this method after the active owner-thread
        # action returns. The bounded wait protects direct diagnostic callers.
        if not self._action_idle.wait(timeout=self._safe_stop_action_grace_s):
            return SafeStopReceipt(
                controller_ack=False,
                buffers_cleared=False,
                arm_a_stopped=False,
                arm_b_stopped=False,
                stop_epoch=stop_epoch,
            )

        try:
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            ArticulationAction = None

        buffers_cleared = True
        for arm in self._arms.values():
            try:
                positions = np.asarray(arm.get_joint_positions(), dtype=float)
                if (
                    ArticulationAction is None
                    or not positions.size
                    or not np.all(np.isfinite(positions))
                ):
                    buffers_cleared = False
                    continue

                # Isaac Sim 5.1's ArticulationController has no public reset()
                # method. Replace any pending IK or gripper command with a
                # full-articulation hold target, then read it back from the
                # controller.
                arm.set_joint_velocities(np.zeros_like(positions))
                controller = arm.get_articulation_controller()
                controller.apply_action(
                    ArticulationAction(joint_positions=positions.copy())
                )
                buffers_cleared = buffers_cleared and _position_targets_match(
                    controller,
                    positions,
                )
            except BaseException:
                # Stopping one arm must never be skipped because the other arm
                # failed. Keep trying and return an unconfirmed receipt.
                buffers_cleared = False

        pause = getattr(self._world, "pause", None)
        is_playing = getattr(self._world, "is_playing", None)
        try:
            if not callable(pause) or not callable(is_playing):
                controller_ack = False
            else:
                pause()
                controller_ack = is_playing() is False
        except BaseException:
            controller_ack = False
        try:
            arm_a_stopped = self._is_stationary("Arm_A")
        except BaseException:
            arm_a_stopped = False
        try:
            arm_b_stopped = self._is_stationary("Arm_B")
        except BaseException:
            arm_b_stopped = False
        return SafeStopReceipt(
            controller_ack=controller_ack,
            buffers_cleared=buffers_cleared,
            arm_a_stopped=arm_a_stopped,
            arm_b_stopped=arm_b_stopped,
            stop_epoch=stop_epoch,
        )

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        """Direct owner-thread convenience wrapper used by diagnostics."""

        stop_epoch = self.request_stop(reason)
        if not self._is_owner_thread():
            return SafeStopReceipt(
                controller_ack=False,
                buffers_cleared=False,
                arm_a_stopped=False,
                arm_b_stopped=False,
                stop_epoch=stop_epoch,
            )
        return self.confirm_safe_stop(reason, stop_epoch=stop_epoch)
