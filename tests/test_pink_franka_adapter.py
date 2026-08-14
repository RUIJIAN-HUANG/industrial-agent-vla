from __future__ import annotations

import unittest

import numpy as np

from simulation.pink_franka_adapter import (
    _joint_indices,
    _wxyz_rotation_matrix,
)


class PinkFrankaAdapterMathTests(unittest.TestCase):
    def test_identity_quaternion_becomes_identity_matrix(self):
        np.testing.assert_allclose(
            _wxyz_rotation_matrix(np.asarray([1.0, 0.0, 0.0, 0.0])),
            np.eye(3),
        )

    def test_joint_indices_follow_articulation_order(self):
        names = [
            "panda_joint3",
            "panda_finger_joint1",
            "panda_joint1",
            "panda_joint2",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint2",
        ]
        self.assertEqual(_joint_indices(names), [2, 3, 0, 4, 5, 6, 7])

    def test_missing_arm_joint_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "panda_joint7"):
            _joint_indices([f"panda_joint{i}" for i in range(1, 7)])

    def test_compute_rolls_differential_steps_into_absolute_target(self):
        from simulation.pink_franka_adapter import PinkFrankaAdapter

        class Pin:
            @staticmethod
            def SE3(rotation, translation):
                return rotation, translation

        class FrameTask:
            def set_target(self, target):
                self.target = target

        class DifferentialController:
            def __init__(self):
                self.inputs = []

            def compute(self, current, dt_s):
                del dt_s
                current = np.asarray(current, dtype=float)
                self.inputs.append(current.copy())
                return current[:7] + (1.0 - current[:7]) * 0.5

        pink = object.__new__(PinkFrankaAdapter)
        controller = DifferentialController()
        pink._pin = Pin()
        pink._controllers = {"Arm_A": controller}
        pink._frame_tasks = {"Arm_A": FrameTask()}
        pink._controlled_indices = {"Arm_A": list(range(7))}
        pink._diagnostics = {"Arm_A": {}}

        targets = pink.compute(
            arm_id="Arm_A",
            current_joint_positions=np.zeros(9),
            target_position_base_m=np.zeros(3),
            target_orientation_base_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            dt_s=0.1,
        )

        self.assertGreater(len(controller.inputs), 1)
        np.testing.assert_allclose(controller.inputs[1][:7], 0.5)
        np.testing.assert_allclose(targets, 1.0, atol=2e-5)
        self.assertGreater(pink.diagnostics("Arm_A")["rollout_iterations"], 1)

    def test_compute_rejects_joint_state_that_does_not_cover_arm(self):
        from simulation.pink_franka_adapter import PinkFrankaAdapter

        pink = object.__new__(PinkFrankaAdapter)
        pink._controlled_indices = {"Arm_A": list(range(7))}
        with self.assertRaisesRegex(ValueError, "does not cover"):
            pink.compute(
                arm_id="Arm_A",
                current_joint_positions=np.zeros(6),
                target_position_base_m=np.zeros(3),
                target_orientation_base_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
                dt_s=0.1,
            )


if __name__ == "__main__":
    unittest.main()
