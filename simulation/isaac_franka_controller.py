"""Isaac Sim 5.1 Franka backend for ``IsaacExecutionEnvironment``.

Import this module only after ``SimulationApp`` has started.  It converts the
frozen robot-base 7-D action contract into Lula inverse-kinematics targets and
writes them to the selected ``SingleArticulation``.
"""

from __future__ import annotations

from datetime import datetime, timezone
from math import ceil, cos, sin, sqrt
from typing import Any, Mapping

import numpy as np

from industrial_agent.contracts import ActionStep
from industrial_agent.environment import SafeStopReceipt


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


class IsaacSimFrankaController:
    """Live dual-Franka controller using Isaac Sim 5.1 Lula IK."""

    def __init__(
        self,
        *,
        world: Any,
        arms: Mapping[str, Any],
        physics_dt_s: float,
        end_effector_frame_name: str = "right_gripper",
        stationary_velocity_rad_s: float = 1e-3,
    ) -> None:
        if set(arms) != set(_ARMS):
            raise ValueError("arms must contain exactly Arm_A and Arm_B")
        if physics_dt_s <= 0.0:
            raise ValueError("physics_dt_s must be positive")

        try:
            from isaacsim.robot_motion.motion_generation import (
                ArticulationKinematicsSolver,
                LulaKinematicsSolver,
                load_supported_lula_kinematics_solver_config,
            )
        except ImportError as exc:
            raise RuntimeError(
                "Isaac Sim 5.1 Lula motion-generation extension is unavailable"
            ) from exc

        self._world = world
        self._arms = dict(arms)
        self._physics_dt_s = float(physics_dt_s)
        self._stationary_velocity_rad_s = float(stationary_velocity_rad_s)
        self._solvers: dict[str, Any] = {}
        self._lula_solvers: dict[str, Any] = {}
        for arm_id, arm in self._arms.items():
            config = load_supported_lula_kinematics_solver_config("Franka")
            lula_solver = LulaKinematicsSolver(**config)
            self._lula_solvers[arm_id] = lula_solver
            self._solvers[arm_id] = ArticulationKinematicsSolver(
                arm,
                lula_solver,
                end_effector_frame_name,
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
        if arm_id not in self._arms:
            raise RuntimeError(f"unknown Isaac Franka arm: {arm_id!r}")
        other_arm = "Arm_B" if arm_id == "Arm_A" else "Arm_A"
        if not self._is_stationary(other_arm):
            raise RuntimeError(
                f"{arm_id} controller interlock requires {other_arm} stationary"
            )

    def execute_action(self, action: ActionStep, *, arm_id: str) -> None:
        arm = self._arms[arm_id]
        solver = self._solvers[arm_id]
        lula_solver = self._lula_solvers[arm_id]
        translation = np.asarray(action.values[:3], dtype=float)
        rotation = np.asarray(action.values[3:6], dtype=float)
        gripper = min(1.0, max(0.0, float(action.values[6])))

        base_position, base_orientation = arm.get_world_pose()
        base_position = np.asarray(base_position, dtype=float)
        base_orientation = np.asarray(base_orientation, dtype=float)
        # Lula assumes its robot base is at the world origin unless this pose is
        # refreshed.  The frozen scene places two Frankas away from the origin,
        # so omitting it produces valid-looking but incorrect IK targets.
        lula_solver.set_robot_base_pose(base_position, base_orientation)
        current_position, current_rotation = solver.compute_end_effector_pose()
        current_orientation = _rotation_matrix_to_quaternion(current_rotation)
        target_position = np.asarray(current_position, dtype=float) + _rotate_vector(
            base_orientation, translation
        )
        delta_base = _rotvec_quaternion(rotation)
        delta_world = _quat_multiply(
            _quat_multiply(base_orientation, delta_base),
            _quat_inverse(base_orientation),
        )
        target_orientation = _quat_multiply(
            delta_world, np.asarray(current_orientation, dtype=float)
        )
        target_orientation /= sqrt(
            float(np.dot(target_orientation, target_orientation))
        )

        ik_action, success = solver.compute_inverse_kinematics(
            target_position,
            target_orientation,
        )
        if not success:
            raise RuntimeError(f"Lula IK did not converge for {arm_id}")
        arm.apply_action(ik_action)

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
        finger_position_m = 0.04 * gripper
        arm.apply_action(
            ArticulationAction(
                joint_positions=np.asarray(
                    [finger_position_m, finger_position_m], dtype=float
                ),
                joint_indices=np.asarray(finger_indices, dtype=np.int64),
            )
        )

        play = getattr(self._world, "play", None)
        if callable(play):
            play()
        step_count = max(1, ceil(action.duration_ms / 1000.0 / self._physics_dt_s))
        for _ in range(step_count):
            self._world.step(render=False)

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        del reason
        try:
            from isaacsim.core.utils.types import ArticulationAction
        except ImportError:
            ArticulationAction = None

        buffers_cleared = True
        for arm in self._arms.values():
            positions = np.asarray(arm.get_joint_positions(), dtype=float)
            if (
                ArticulationAction is None
                or not positions.size
                or not np.all(np.isfinite(positions))
            ):
                buffers_cleared = False
                continue

            # Isaac Sim 5.1's ArticulationController has no public reset()
            # method.  Replace any pending IK or gripper command with a
            # full-articulation hold target, then read it back from the
            # controller.  This keeps the receipt fail-closed without relying
            # on a private or version-specific queue API.
            arm.set_joint_velocities(np.zeros_like(positions))
            controller = arm.get_articulation_controller()
            controller.apply_action(
                ArticulationAction(joint_positions=positions.copy())
            )
            buffers_cleared = buffers_cleared and _position_targets_match(
                controller,
                positions,
            )

        pause = getattr(self._world, "pause", None)
        if callable(pause):
            pause()
            controller_ack = True
        else:
            controller_ack = False
        arm_a_stopped = self._is_stationary("Arm_A")
        arm_b_stopped = self._is_stationary("Arm_B")
        stop_epoch = datetime.now(timezone.utc).isoformat()
        return SafeStopReceipt(
            controller_ack=controller_ack,
            buffers_cleared=buffers_cleared,
            arm_a_stopped=arm_a_stopped,
            arm_b_stopped=arm_b_stopped,
            stop_epoch=stop_epoch,
        )
