from __future__ import annotations

from math import sqrt
from types import ModuleType
from threading import Event, Lock, Thread
from unittest.mock import patch
import sys
import unittest

import numpy as np

from industrial_agent.contracts import ActionStep
from industrial_agent.sync_contract import FROZEN_MULTI_RATE
from simulation.isaac_franka_controller import (
    IsaacSimFrankaController,
    _gripper_opening_m,
    _position_targets_match,
    _rotate_vector,
    _rotation_matrix_to_quaternion,
)
from simulation.run_isaac_adapter_smoke import _quaternion_to_rotvec


class IsaacFrankaControllerMathTests(unittest.TestCase):
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


class MultiRateExecutionTests(unittest.TestCase):
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
            dof_names = [
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
    def test_both_vla_endpoint_conventions_map_to_binary_hardware_positions(self):
        for closed in (-1.0, 0.0, 0.499):
            self.assertEqual(_gripper_opening_m(closed), 0.0)
        for opened in (0.5, 1.0):
            self.assertEqual(_gripper_opening_m(opened), 0.04)


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
