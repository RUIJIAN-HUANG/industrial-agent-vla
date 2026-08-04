from __future__ import annotations

import unittest

import numpy as np

from simulation.scripted_expert_plan import (
    P01ExpertTuning,
    bounded_world_delta,
    conservative_step_limit,
    first_bin_slot_local_center,
    frozen_success_vote,
)


class ScriptedExpertPlanTests(unittest.TestCase):
    def test_default_tuning_is_valid(self) -> None:
        P01ExpertTuning().validate()

    def test_world_delta_is_bounded_without_changing_direction(self) -> None:
        delta = bounded_world_delta(
            [0.0, 0.0, 0.0],
            [0.03, 0.04, 0.0],
            max_step_m=0.02,
        )
        np.testing.assert_allclose(delta, [0.012, 0.016, 0.0])

    def test_small_world_delta_is_not_stretched(self) -> None:
        delta = bounded_world_delta(
            [1.0, 2.0, 3.0],
            [1.001, 2.002, 3.0],
            max_step_m=0.02,
        )
        np.testing.assert_allclose(delta, [0.001, 0.002, 0.0])

    def test_first_slot_uses_frozen_two_by_three_geometry(self) -> None:
        center = first_bin_slot_local_center(
            size_m=[0.18, 0.12, 0.07],
            wall_thickness_m=0.006,
            bottom_thickness_m=0.006,
            part_height_m=0.044,
        )
        np.testing.assert_allclose(center, [-0.056, -0.027, -0.007])

    def test_success_requires_two_of_exactly_three_fresh_frames(self) -> None:
        self.assertTrue(frozen_success_vote([True, False, True]))
        self.assertFalse(frozen_success_vote([False, True, False]))
        with self.assertRaisesRegex(ValueError, "exactly three"):
            frozen_success_vote([True, True])

    def test_step_limit_has_tracking_margin(self) -> None:
        self.assertEqual(conservative_step_limit(0.10, 0.02), 28)


if __name__ == "__main__":
    unittest.main()
