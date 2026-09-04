from __future__ import annotations

from math import sqrt
from types import ModuleType
from typing import ClassVar
from threading import Event, Lock, Thread
from unittest.mock import patch
import sys
import unittest

import numpy as np

from industrial_agent.contracts import ActionStep
from industrial_agent.sync_contract import FROZEN_MULTI_RATE
from simulation.isaac_franka_controller import (
    CartesianTrackingRejected,
    GripperCloseNotSettled,
    IsaacSimFrankaController,
    _control_world_position_for_tcp,
    _gripper_opening_m,
    _midpoint_tcp_offset_local,
    _position_targets_match,
    _quaternion_to_rotvec,
    _rotate_vector,
    _rotation_matrix_to_quaternion,
    _translation_tracking_diagnostic,
    _virtual_tcp_world_position,
)


class IsaacFrankaControllerMathTests(unittest.TestCase):
    def test_tracking_rejection_retains_structured_diagnostic(self):
        diagnostic = _translation_tracking_diagnostic(
            np.asarray([0.005, 0.0, 0.0]),
            np.zeros(3),
        )
        error = CartesianTrackingRejected("Arm_B", diagnostic)

        self.assertEqual(error.arm_id, "Arm_B")
        self.assertEqual(error.diagnostic, diagnostic)
        self.assertIn("forward_progress_m=0.000000", str(error))

    def test_translation_tracking_accepts_forward_progress(self):
        diagnostic = _translation_tracking_diagnostic(
            np.asarray([0.005, 0.0, 0.0]),
            np.asarray([0.003, 0.0002, 0.0]),
        )
        self.assertTrue(diagnostic["pass"])

    def test_translation_tracking_rejects_perpendicular_motion(self):
        diagnostic = _translation_tracking_diagnostic(
            np.asarray([0.005, 0.0, 0.0]),
            np.asarray([0.0, 0.0, -0.003]),
        )
        self.assertFalse(diagnostic["pass"])
        self.assertEqual(diagnostic["forward_progress_m"], 0.0)

    def test_translation_tracking_rejects_no_progress(self):
        diagnostic = _translation_tracking_diagnostic(
            np.asarray([0.005, 0.0, 0.0]),
            np.zeros(3),
        )
        self.assertFalse(diagnostic["pass"])

    def test_translation_tracking_ignores_submillimetre_rotation_noise(self):
        diagnostic = _translation_tracking_diagnostic(
            np.asarray([-0.000020, 0.000017, 0.000030]),
            np.asarray([-0.000313, 0.0, 0.0]),
        )
        self.assertFalse(diagnostic["checked"])
        self.assertTrue(diagnostic["pass"])

    def test_identity_rotation_matrix_becomes_identity_quaternion(self):
        quaternion = _rotation_matrix_to_quaternion(np.eye(3))
        np.testing.assert_allclose(quaternion, [1.0, 0.0, 0.0, 0.0])

    def test_rotation_matrix_becomes_wxyz_quaternion(self):
        rotation_z_90 = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        quaternion = _rotation_matrix_to_quaternion(rotation_z_90)
        expected = [sqrt(0.5), 0.0, 0.0, sqrt(0.5)]
        np.testing.assert_allclose(quaternion, expected)

    def test_quaternion_rotates_base_delta_into_world(self):
        rotation_z_90 = np.asarray([sqrt(0.5), 0.0, 0.0, sqrt(0.5)])
        vector = _rotate_vector(rotation_z_90, np.asarray([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(vector, [0.0, 1.0, 0.0], atol=1e-12)

    def test_virtual_tcp_offset_is_calibrated_in_control_local_frame(self):
        rotation_z_90 = np.asarray(
            [
                [0.0, -1.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        offset = _midpoint_tcp_offset_local(
            control_position_world_m=np.asarray([1.0, 2.0, 3.0]),
            control_rotation_world=rotation_z_90,
            left_tip_world_m=np.asarray([0.98, 2.1, 3.0]),
            right_tip_world_m=np.asarray([1.02, 2.1, 3.0]),
        )
        np.testing.assert_allclose(offset, [0.1, 0.0, 0.0], atol=1e-12)
        np.testing.assert_allclose(
            _virtual_tcp_world_position(
                np.asarray([1.0, 2.0, 3.0]),
                rotation_z_90,
                offset,
            ),
            [1.0, 2.1, 3.0],
            atol=1e-12,
        )

    def test_virtual_tcp_target_is_converted_back_to_lula_control_frame(self):
        rotation_z_90 = np.asarray([sqrt(0.5), 0.0, 0.0, sqrt(0.5)])
        target = _control_world_position_for_tcp(
            np.asarray([1.0, 2.0, 3.0]),
            rotation_z_90,
            np.asarray([0.1, 0.0, 0.0]),
        )
        np.testing.assert_allclose(target, [1.0, 1.9, 3.0], atol=1e-12)


class MultiRateExecutionTests(unittest.TestCase):
    def test_preflight_rejects_unreachable_action_without_world_step(self):
        class World:
            steps = 0

        class Lula:
            @staticmethod
            def set_robot_base_pose(position, orientation):
                del position, orientation

        class Solver:
            calls = 0

            @staticmethod
            def compute_end_effector_pose():
                return np.zeros(3), np.eye(3)

            @classmethod
            def compute_inverse_kinematics(cls, position, orientation):
                del position, orientation
                cls.calls += 1
                return object(), cls.calls < 5

        class Arm:
            @staticmethod
            def get_world_pose():
                return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])

        controller = object.__new__(IsaacSimFrankaController)
        controller._world = World()
        controller._arms = {"Arm_A": Arm()}
        controller._solvers = {"Arm_A": Solver()}
        controller._lula_solvers = {"Arm_A": Lula()}
        controller._owner_thread_id = __import__("threading").get_ident()
        controller._stop_requested = Event()
        controller._multi_rate = FROZEN_MULTI_RATE

        action = ActionStep.from_sequence(
            [0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            duration_ms=100,
        )
        reason = controller.action_rejection_reason(action, arm_id="Arm_A")

        self.assertIn("IK tick 5/6", reason)
        self.assertEqual(World.steps, 0)

    def test_100ms_model_delta_is_interpolated_at_60hz(self):
        class World:
            def __init__(self):
                self.render_flags = []

            @staticmethod
            def play():
                return None

            def step(self, *, render):
                self.render_flags.append(render)

        class Lula:
            @staticmethod
            def set_robot_base_pose(position, orientation):
                del position, orientation

        class Solver:
            def __init__(self):
                self.positions = []
                self.orientations = []

            @staticmethod
            def compute_end_effector_pose():
                return np.zeros(3), np.eye(3)

            def compute_inverse_kinematics(self, position, orientation):
                self.positions.append(np.asarray(position, dtype=float))
                self.orientations.append(np.asarray(orientation, dtype=float))
                return object(), True

        class Arm:
            dof_names: ClassVar[list[str]] = [
                "panda_joint1",
                "panda_joint2",
                "panda_finger_joint1",
                "panda_finger_joint2",
            ]

            def __init__(self):
                self.actions = []

            @staticmethod
            def get_world_pose():
                return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])

            def apply_action(self, action):
                self.actions.append(action)

        world = World()
        arm = Arm()
        solver = Solver()
        controller = object.__new__(IsaacSimFrankaController)
        controller._world = world
        controller._arms = {"Arm_A": arm}
        controller._solvers = {"Arm_A": solver}
        controller._lula_solvers = {"Arm_A": Lula()}
        controller._owner_thread_id = __import__("threading").get_ident()
        controller._action_lock = Lock()
        controller._action_idle = Event()
        controller._action_idle.set()
        controller._stop_requested = Event()
        controller._multi_rate = FROZEN_MULTI_RATE
        controller._physics_tick_index = 0
        controller._tick_observer = None

        action = ActionStep.from_sequence(
            [0.06, 0.0, 0.0, 0.0, 0.0, 0.06, 1.0],
            duration_ms=100,
        )
        with patch.dict(sys.modules, _isaac_type_modules()):
            controller.execute_action(action, arm_id="Arm_A")

        self.assertEqual(len(solver.positions), 6)
        np.testing.assert_allclose(
            [position[0] for position in solver.positions],
            [0.01, 0.02, 0.03, 0.04, 0.05, 0.06],
        )
        self.assertEqual(len(world.render_flags), 12)
        self.assertEqual(sum(world.render_flags), 3)
        self.assertEqual(
            [index + 1 for index, flag in enumerate(world.render_flags) if flag],
            [4, 8, 12],
        )

    def test_tick_observer_receives_every_post_step_tick(self):
        observed = []

        class World:
            @staticmethod
            def play():
                return None

            @staticmethod
            def step(*, render):
                del render

        class Lula:
            @staticmethod
            def set_robot_base_pose(position, orientation):
                del position, orientation

        class Solver:
            @staticmethod
            def compute_end_effector_pose():
                return np.zeros(3), np.eye(3)

            @staticmethod
            def compute_inverse_kinematics(position, orientation):
                del position, orientation
                return object(), True

        class Arm:
            dof_names: ClassVar[list[str]] = [
                "panda_finger_joint1",
                "panda_finger_joint2",
            ]

            @staticmethod
            def get_world_pose():
                return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])

            @staticmethod
            def apply_action(action):
                del action

        controller = object.__new__(IsaacSimFrankaController)
        controller._world = World()
        controller._arms = {"Arm_A": Arm()}
        controller._solvers = {"Arm_A": Solver()}
        controller._lula_solvers = {"Arm_A": Lula()}
        controller._owner_thread_id = __import__("threading").get_ident()
        controller._action_lock = Lock()
        controller._action_idle = Event()
        controller._action_idle.set()
        controller._stop_requested = Event()
        controller._multi_rate = FROZEN_MULTI_RATE
        controller._physics_tick_index = 0
        controller._tick_observer = lambda tick, render: observed.append((tick, render))

        action = ActionStep.from_sequence([0, 0, 0, 0, 0, 0, 1], duration_ms=100)
        with patch.dict(sys.modules, _isaac_type_modules()):
            controller.execute_action(action, arm_id="Arm_A")

        self.assertEqual([tick for tick, _ in observed], list(range(1, 13)))
        self.assertEqual([tick for tick, render in observed if render], [4, 8, 12])

    def test_invalid_rotation_matrix_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            _rotation_matrix_to_quaternion(np.zeros((2, 2)))

    def test_pi_rotation_converts_to_finite_rotation_vector(self):
        rotvec = _quaternion_to_rotvec(np.asarray([0.0, 1.0, 0.0, 0.0]))
        np.testing.assert_allclose(rotvec, [np.pi, 0.0, 0.0])


class AppliedHoldTargetTests(unittest.TestCase):
    def test_matching_full_position_target_is_confirmed(self):
        class Controller:
            @staticmethod
            def get_applied_action():
                class Action:
                    joint_positions = np.asarray([0.1, -0.2, 0.3])

                return Action()

        self.assertTrue(
            _position_targets_match(
                Controller(),
                np.asarray([0.1, -0.2, 0.3]),
            )
        )

    def test_missing_readback_fails_closed(self):
        self.assertFalse(
            _position_targets_match(
                object(),
                np.asarray([0.1, -0.2, 0.3]),
            )
        )

    def test_partial_or_different_target_fails_closed(self):
        class Controller:
            @staticmethod
            def get_applied_action():
                class Action:
                    joint_positions = np.asarray([0.1, -0.2])

                return Action()

        self.assertFalse(
            _position_targets_match(
                Controller(),
                np.asarray([0.1, -0.2, 0.3]),
            )
        )


class GripperMappingTests(unittest.TestCase):
    def test_resolved_commands_map_to_binary_hardware_positions(self):
        self.assertEqual(_gripper_opening_m(False), 0.0)
        self.assertEqual(_gripper_opening_m(True), 0.04)

    def test_controller_latches_commands_inside_hysteresis_deadband(self):
        controller = object.__new__(IsaacSimFrankaController)
        controller._gripper_command_open_by_arm = {"Arm_A": False}
        controller._gripper_close_verified_by_arm = {"Arm_A": True}

        self.assertFalse(controller._resolve_gripper_command("Arm_A", 0.5))
        self.assertTrue(controller._resolve_gripper_command("Arm_A", 0.7))
        self.assertTrue(controller._resolve_gripper_command("Arm_A", 0.5))
        self.assertFalse(controller._resolve_gripper_command("Arm_A", 0.3))

    def test_live_finger_positions_are_read_by_joint_name(self):
        class Arm:
            dof_names: ClassVar[list[str]] = [
                "panda_joint1",
                "panda_finger_joint2",
                "panda_joint2",
                "panda_finger_joint1",
            ]

            @staticmethod
            def get_joint_positions():
                return np.asarray([0.2, 0.012, -0.3, 0.011])

        controller = object.__new__(IsaacSimFrankaController)
        controller._arms = {"Arm_A": Arm()}
        controller._owner_thread_id = __import__("threading").get_ident()
        np.testing.assert_allclose(
            controller.gripper_joint_positions("Arm_A"),
            [0.011, 0.012],
        )


class _FakeArticulationAction:
    def __init__(self, *, joint_positions=None, joint_indices=None):
        self.joint_positions = joint_positions
        self.joint_indices = joint_indices


class _FakeAppliedController:
    def __init__(self):
        self.applied = None

    def apply_action(self, action):
        self.applied = action

    def get_applied_action(self):
        return self.applied


class _FakeArm:
    def __init__(self, *, fail_velocity_write=False):
        self.positions = np.asarray([0.1, -0.2, 0.3])
        self.velocities = np.zeros(3)
        self.fail_velocity_write = fail_velocity_write
        self.velocity_write_attempted = False
        self.controller = _FakeAppliedController()

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def set_joint_velocities(self, velocities):
        self.velocity_write_attempted = True
        if self.fail_velocity_write:
            raise RuntimeError("velocity write failed")
        self.velocities = np.asarray(velocities)

    def get_articulation_controller(self):
        return self.controller


class _FakeWorld:
    def __init__(self):
        self.playing = True
        self.pause_calls = 0

    def pause(self):
        self.pause_calls += 1
        self.playing = False

    def is_playing(self):
        return self.playing


def _controller_for_stop(*, arm_a, arm_b, world=None):
    controller = object.__new__(IsaacSimFrankaController)
    controller._arms = {"Arm_A": arm_a, "Arm_B": arm_b}
    controller._world = world or _FakeWorld()
    controller._stationary_velocity_rad_s = 1e-3
    controller._safe_stop_action_grace_s = 0.0
    controller._action_idle = Event()
    controller._action_idle.set()
    controller._stop_requested = Event()
    controller._stop_epoch_lock = Lock()
    controller._stop_epoch = 0
    controller._owner_thread_id = __import__("threading").get_ident()
    return controller


def _isaac_type_modules():
    isaacsim = ModuleType("isaacsim")
    core = ModuleType("isaacsim.core")
    utils = ModuleType("isaacsim.core.utils")
    types = ModuleType("isaacsim.core.utils.types")
    types.ArticulationAction = _FakeArticulationAction
    return {
        "isaacsim": isaacsim,
        "isaacsim.core": core,
        "isaacsim.core.utils": utils,
        "isaacsim.core.utils.types": types,
    }


class _GateArm:
    dof_names: ClassVar[list[str]] = [
        "panda_joint1",
        "panda_joint2",
        "panda_finger_joint1",
        "panda_finger_joint2",
    ]

    def __init__(
        self,
        *,
        closing_progress: bool,
        contact_velocity_noise: bool = False,
    ):
        self.positions = np.asarray([0.0, 0.0, 0.04, 0.04], dtype=float)
        self.velocities = np.zeros(4, dtype=float)
        self.closing_progress = closing_progress
        self.contact_velocity_noise = contact_velocity_noise
        self.closing_commanded = False

    @staticmethod
    def get_world_pose():
        return np.zeros(3), np.asarray([1.0, 0.0, 0.0, 0.0])

    def get_joint_positions(self):
        return self.positions.copy()

    def get_joint_velocities(self):
        return self.velocities.copy()

    def apply_action(self, action):
        indices = getattr(action, "joint_indices", None)
        targets = getattr(action, "joint_positions", None)
        if indices is not None and np.array_equal(indices, [2, 3]):
            self.closing_commanded = bool(np.allclose(targets, 0.0))

    def advance_physics(self, physics_step: int) -> None:
        if not self.closing_commanded or not self.closing_progress:
            return
        if physics_step < 4:
            self.positions[2:4] = max(0.015, 0.04 - 0.008 * physics_step)
            self.velocities[2:4] = -0.01
        else:
            self.positions[2:4] = 0.015
            self.velocities[2:4] = -0.02 if self.contact_velocity_noise else 0.0


class _GateWorld:
    def __init__(self, arm: _GateArm):
        self.arm = arm
        self.physics_steps = 0

    @staticmethod
    def play():
        return None

    def step(self, *, render):
        del render
        self.physics_steps += 1
        self.arm.advance_physics(self.physics_steps)


class _GateSolver:
    def __init__(self, world: _GateWorld):
        self.world = world
        self.inverse_kinematics_at_steps: list[int] = []

    @staticmethod
    def compute_end_effector_pose():
        return np.zeros(3), np.eye(3)

    def compute_inverse_kinematics(self, position, orientation):
        del position, orientation
        self.inverse_kinematics_at_steps.append(self.world.physics_steps)
        return object(), True


class _GateLula:
    @staticmethod
    def set_robot_base_pose(position, orientation):
        del position, orientation


def _controller_for_gripper_gate(
    *,
    closing_progress: bool,
    contact_velocity_noise: bool = False,
) -> tuple[IsaacSimFrankaController, _GateWorld, _GateSolver]:
    arm = _GateArm(
        closing_progress=closing_progress,
        contact_velocity_noise=contact_velocity_noise,
    )
    world = _GateWorld(arm)
    solver = _GateSolver(world)
    controller = object.__new__(IsaacSimFrankaController)
    controller._world = world
    controller._arms = {"Arm_A": arm}
    controller._solvers = {"Arm_A": solver}
    controller._lula_solvers = {"Arm_A": _GateLula()}
    controller._owner_thread_id = __import__("threading").get_ident()
    controller._action_lock = Lock()
    controller._action_idle = Event()
    controller._action_idle.set()
    controller._stop_requested = Event()
    controller._multi_rate = FROZEN_MULTI_RATE
    controller._physics_tick_index = 0
    controller._tick_observer = None
    controller._ik_backend = "lula"
    controller._tcp_offsets_local_m = {}
    controller._gripper_settle_velocity_m_s = 1e-3
    controller._gripper_settle_control_ticks = 2
    controller._gripper_close_timeout_s = 0.1
    controller._gripper_closed_tolerance_m = 5e-4
    controller._gripper_contact_min_travel_m = 1e-3
    controller._gripper_contact_symmetry_tolerance_m = 3e-3
    controller._gripper_contact_position_delta_m = 1e-4
    controller._gripper_contact_source = None
    controller._gripper_command_open_by_arm = {"Arm_A": True}
    controller._gripper_close_verified_by_arm = {"Arm_A": False}
    controller._last_gripper_diagnostics = {}
    return controller, world, solver


class GripperCompletionGateTests(unittest.TestCase):
    def test_close_and_lift_waits_for_two_stable_contact_samples(self):
        controller, world, solver = _controller_for_gripper_gate(closing_progress=True)
        action = ActionStep.from_sequence(
            [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.31],
            duration_ms=100,
        )

        with patch.dict(sys.modules, _isaac_type_modules()):
            controller.execute_action(action, arm_id="Arm_A")

        self.assertGreaterEqual(solver.inverse_kinematics_at_steps[0], 6)
        self.assertEqual(world.physics_steps, 18)
        self.assertTrue(controller._gripper_close_verified_by_arm["Arm_A"])
        diagnostic = controller._last_gripper_diagnostics["Arm_A"]
        self.assertTrue(diagnostic["contact_confirmed"])
        self.assertEqual(diagnostic["stable_control_ticks"], 2)

    def test_contact_position_plateau_accepts_noisy_velocity_readback(self):
        controller, _, solver = _controller_for_gripper_gate(
            closing_progress=True,
            contact_velocity_noise=True,
        )
        action = ActionStep.from_sequence(
            [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.31],
            duration_ms=100,
        )

        with patch.dict(sys.modules, _isaac_type_modules()):
            controller.execute_action(action, arm_id="Arm_A")

        self.assertTrue(controller._gripper_close_verified_by_arm["Arm_A"])
        self.assertTrue(
            controller._last_gripper_diagnostics["Arm_A"]["position_plateau"]
        )
        self.assertTrue(solver.inverse_kinematics_at_steps)

    def test_unsettled_close_times_out_before_any_lift_ik(self):
        controller, _, solver = _controller_for_gripper_gate(closing_progress=False)
        action = ActionStep.from_sequence(
            [0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.31],
            duration_ms=100,
        )

        with patch.dict(sys.modules, _isaac_type_modules()):
            with self.assertRaises(GripperCloseNotSettled):
                controller.execute_action(action, arm_id="Arm_A")

        self.assertEqual(solver.inverse_kinematics_at_steps, [])


class SafeStopTests(unittest.TestCase):
    def test_failure_on_first_arm_does_not_skip_second_arm(self):
        arm_a = _FakeArm(fail_velocity_write=True)
        arm_b = _FakeArm()
        controller = _controller_for_stop(arm_a=arm_a, arm_b=arm_b)
        with patch.dict(sys.modules, _isaac_type_modules()):
            receipt = controller.safe_stop("fault")
        self.assertTrue(arm_a.velocity_write_attempted)
        self.assertTrue(arm_b.velocity_write_attempted)
        self.assertFalse(receipt.buffers_cleared)
        self.assertFalse(receipt.confirmed)

    def test_controller_ack_requires_pause_readback(self):
        class WorldWithoutReadback:
            @staticmethod
            def pause():
                return None

        controller = _controller_for_stop(
            arm_a=_FakeArm(),
            arm_b=_FakeArm(),
            world=WorldWithoutReadback(),
        )
        with patch.dict(sys.modules, _isaac_type_modules()):
            receipt = controller.safe_stop("fault")
        self.assertFalse(receipt.controller_ack)
        self.assertFalse(receipt.confirmed)

    def test_busy_owner_returns_unconfirmed_receipt_without_touching_arms(self):
        arm_a = _FakeArm()
        arm_b = _FakeArm()
        controller = _controller_for_stop(arm_a=arm_a, arm_b=arm_b)
        controller._action_idle.clear()
        with patch.dict(sys.modules, _isaac_type_modules()):
            receipt = controller.safe_stop("hung action")
        self.assertFalse(receipt.confirmed)
        self.assertFalse(arm_a.velocity_write_attempted)
        self.assertFalse(arm_b.velocity_write_attempted)

    def test_stop_epoch_is_idempotent_for_one_latched_stop(self):
        controller = _controller_for_stop(arm_a=_FakeArm(), arm_b=_FakeArm())
        with patch.dict(sys.modules, _isaac_type_modules()):
            first = controller.safe_stop("one")
            second = controller.safe_stop("two")
        self.assertEqual(first.stop_epoch, "controller-stop-1")
        self.assertEqual(second.stop_epoch, "controller-stop-1")

    def test_non_owner_safe_stop_only_signals_and_never_touches_isaac(self):
        arm_a = _FakeArm()
        arm_b = _FakeArm()
        controller = _controller_for_stop(arm_a=arm_a, arm_b=arm_b)
        result = {}

        def stop_from_worker():
            with patch.dict(sys.modules, _isaac_type_modules()):
                result["receipt"] = controller.safe_stop("watchdog")

        worker = Thread(target=stop_from_worker)
        worker.start()
        worker.join(timeout=0.5)
        self.assertFalse(worker.is_alive())
        self.assertFalse(result["receipt"].confirmed)
        self.assertFalse(arm_a.velocity_write_attempted)
        self.assertFalse(arm_b.velocity_write_attempted)
        self.assertTrue(controller._stop_requested.is_set())


if __name__ == "__main__":
    unittest.main()
