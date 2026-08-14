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


if __name__ == "__main__":
    unittest.main()
