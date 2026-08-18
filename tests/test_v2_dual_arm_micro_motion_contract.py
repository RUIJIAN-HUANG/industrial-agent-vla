from __future__ import annotations

import unittest

from simulation.run_v2_dual_arm_micro_motion_acceptance import (
    DELTA_Z_M,
    FINGER_TOLERANCE_M,
    MOTION_SETTLE_PHYSICS_STEPS,
    OTHER_ARM_JOINT_TOLERANCE_RAD,
    TCP_DELTA_TOLERANCE_M,
    TCP_RETURN_TOLERANCE_M,
    _bounded_return_delta_z_m,
    _micro_action_values,
    _settle_motion,
)


class V2DualArmMicroMotionContractTests(unittest.TestCase):
    def test_up_and_down_actions_are_five_mm_and_keep_gripper_open(self) -> None:
        self.assertEqual(_micro_action_values(DELTA_Z_M), [0, 0, 0.005, 0, 0, 0, 1])
        self.assertEqual(_micro_action_values(-DELTA_Z_M), [0, 0, -0.005, 0, 0, 0, 1])

    def test_larger_motion_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot exceed 5 mm"):
            _micro_action_values(0.0051)

    def test_return_delta_uses_the_measured_upward_motion(self) -> None:
        self.assertAlmostEqual(_bounded_return_delta_z_m([0.0, 0.0, 0.00363]), -0.00363)

    def test_return_delta_remains_capped_at_five_mm(self) -> None:
        self.assertEqual(_bounded_return_delta_z_m([0.0, 0.0, 0.006]), -0.005)
        self.assertEqual(_bounded_return_delta_z_m([0.0, 0.0, -0.006]), 0.005)

    def test_motion_settle_advances_the_configured_physics_steps(self) -> None:
        class FakeWorld:
            def __init__(self) -> None:
                self.render_values: list[bool] = []

            def step(self, *, render: bool) -> None:
                self.render_values.append(render)

        world = FakeWorld()
        _settle_motion(world)
        self.assertEqual(len(world.render_values), MOTION_SETTLE_PHYSICS_STEPS)
        self.assertTrue(all(world.render_values))

    def test_motion_settle_rejects_non_positive_steps(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be positive"):
            _settle_motion(object(), steps=0)

    def test_acceptance_tolerances_remain_tight(self) -> None:
        self.assertLessEqual(TCP_DELTA_TOLERANCE_M, 0.0015)
        self.assertLessEqual(TCP_RETURN_TOLERANCE_M, 0.0015)
        self.assertLessEqual(OTHER_ARM_JOINT_TOLERANCE_RAD, 0.002)
        self.assertLessEqual(FINGER_TOLERANCE_M, 0.001)


if __name__ == "__main__":
    unittest.main()
