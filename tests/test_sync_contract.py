from __future__ import annotations

import unittest

from industrial_agent.sync_contract import (
    FROZEN_MULTI_RATE,
    GRIPPER_CLOSE_COMMAND_MAX,
    GRIPPER_OPEN_COMMAND_MIN,
    STATE_7D_ORDER,
    canonical_observed_state_7d,
    canonical_state_7d,
    canonical_state_7d_from_opening,
    normalize_gripper_opening,
    resolve_gripper_command,
)


class FrozenStateContractTests(unittest.TestCase):
    def test_state_7d_uses_robot_base_rotation_vector(self) -> None:
        self.assertEqual(
            STATE_7D_ORDER,
            (
                "x_m",
                "y_m",
                "z_m",
                "ax_rad",
                "ay_rad",
                "az_rad",
                "gripper_norm",
            ),
        )
        self.assertEqual(
            canonical_state_7d([0.4, 0.1, 0.5, 0.01, -0.02, 0.03], True),
            [0.4, 0.1, 0.5, 0.01, -0.02, 0.03, 1.0],
        )

    def test_state_7d_rejects_wrong_pose_length_or_unknown_gripper(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 6"):
            canonical_state_7d([0.0] * 7, True)
        with self.assertRaisesRegex(TypeError, "controller-confirmed boolean"):
            canonical_state_7d([0.0] * 6, None)

    def test_v2_state_preserves_continuous_measured_opening(self) -> None:
        self.assertAlmostEqual(normalize_gripper_opening([0.015, 0.025]), 0.5)
        self.assertEqual(
            canonical_state_7d_from_opening(
                [0.4, 0.1, 0.5, 0.01, -0.02, 0.03],
                0.375,
            )[-1],
            0.375,
        )

    def test_v2_state_rejects_boolean_or_out_of_range_opening(self) -> None:
        with self.assertRaisesRegex(TypeError, "numeric measurement"):
            canonical_state_7d_from_opening([0.0] * 6, True)
        with self.assertRaisesRegex(ValueError, r"within \[0,1\]"):
            canonical_state_7d_from_opening([0.0] * 6, 1.01)

    def test_observed_state_preserves_continuous_gripper_measurement(self) -> None:
        pose = [0.4, 0.1, 0.5, 0.01, -0.02, 0.03]
        self.assertEqual(
            canonical_observed_state_7d(pose, [*pose, 0.375], False),
            [*pose, 0.375],
        )

    def test_observed_state_rejects_cross_field_pose_mismatch(self) -> None:
        pose = [0.4, 0.1, 0.5, 0.01, -0.02, 0.03]
        with self.assertRaisesRegex(ValueError, "does not match tcp_pose"):
            canonical_observed_state_7d(
                pose,
                [0.41, *pose[1:], 0.375],
                False,
            )

    def test_observed_state_rejects_cross_field_gripper_mismatch(self) -> None:
        pose = [0.4, 0.1, 0.5, 0.01, -0.02, 0.03]
        with self.assertRaisesRegex(ValueError, "does not match gripper_open"):
            canonical_observed_state_7d(pose, [*pose, 0.75], False)

    def test_gripper_command_hysteresis_switches_only_at_outer_thresholds(self) -> None:
        self.assertFalse(resolve_gripper_command(GRIPPER_CLOSE_COMMAND_MAX, True))
        self.assertTrue(resolve_gripper_command(GRIPPER_OPEN_COMMAND_MIN, False))
        self.assertTrue(resolve_gripper_command(0.5, True))
        self.assertFalse(resolve_gripper_command(0.5, False))

    def test_gripper_command_hysteresis_rejects_uninitialized_deadband(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires a previous command"):
            resolve_gripper_command(0.5, None)


class FrozenMultiRateContractTests(unittest.TestCase):
    def test_one_model_step_maps_to_all_integer_rate_domains(self) -> None:
        self.assertEqual(FROZEN_MULTI_RATE.model_step_duration_ms, 100)
        self.assertEqual(FROZEN_MULTI_RATE.control_ticks_per_model_step, 6)
        self.assertEqual(FROZEN_MULTI_RATE.physics_ticks_per_model_step, 12)
        self.assertEqual(FROZEN_MULTI_RATE.render_frames_per_model_step, 3)
        self.assertEqual(FROZEN_MULTI_RATE.control_ticks_for_duration_ms(100), 6)
        self.assertEqual(FROZEN_MULTI_RATE.physics_ticks_for_duration_ms(100), 12)

    def test_non_aligned_duration_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "physics grid"):
            FROZEN_MULTI_RATE.control_ticks_for_duration_ms(137)


if __name__ == "__main__":
    unittest.main()
