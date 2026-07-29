from __future__ import annotations

from math import sqrt
import unittest

import numpy as np

from simulation.isaac_franka_controller import (
    _rotate_vector,
    _rotation_matrix_to_quaternion,
)


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

    def test_invalid_rotation_matrix_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "invalid"):
            _rotation_matrix_to_quaternion(np.zeros((2, 2)))


if __name__ == "__main__":
    unittest.main()
