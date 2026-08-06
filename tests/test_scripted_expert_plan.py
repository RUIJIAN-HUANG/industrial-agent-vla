from __future__ import annotations

import unittest

import numpy as np

from simulation.scripted_expert_plan import (
    P01ExpertTuning,
    bin_slot_local_centers,
    bounded_world_delta,
    conservative_step_limit,
    first_bin_slot_local_center,
    frozen_success_vote,
    grasp_follow_report,
    minimum_xy_radius_along_segment,
    motion_sample_violation,
    orthogonal_transfer_waypoints,
    select_safest_slot_index,
    top_down_tilt_error_rad,
    yaw_preserving_top_down_rotation,
)


class ScriptedExpertPlanTests(unittest.TestCase):
    def test_default_tuning_is_valid(self) -> None:
        P01ExpertTuning().validate()

    def test_grasp_uses_one_explicit_geometric_offset(self) -> None:
        tuning = P01ExpertTuning()
        self.assertEqual(tuning.grasp_tcp_center_offset_m, 0.060)
        self.assertEqual(tuning.max_rotation_steps, 24)

    def test_grasp_rejects_unsafe_offset(self) -> None:
        with self.assertRaisesRegex(ValueError, "grasp_tcp_center_offset_m"):
            P01ExpertTuning(grasp_tcp_center_offset_m=0.01).validate()

    def test_top_down_tilt_ignores_cylinder_irrelevant_yaw(self) -> None:
        for yaw_rad in (0.0, 0.4, -1.7, 3.0):
            cosine = np.cos(yaw_rad)
            sine = np.sin(yaw_rad)
            rotation = np.asarray(
                [
                    [cosine, sine, 0.0],
                    [sine, -cosine, 0.0],
                    [0.0, 0.0, -1.0],
                ]
            )
            self.assertAlmostEqual(top_down_tilt_error_rad(rotation), 0.0)

    def test_top_down_target_is_rigid_yaw_preserving_and_points_down(self) -> None:
        yaw_rad = 0.63
        tilt_rad = 0.28
        yaw = np.asarray(
            [
                [np.cos(yaw_rad), -np.sin(yaw_rad), 0.0],
                [np.sin(yaw_rad), np.cos(yaw_rad), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        tilt = np.asarray(
            [
                [1.0, 0.0, 0.0],
                [0.0, np.cos(tilt_rad), -np.sin(tilt_rad)],
                [0.0, np.sin(tilt_rad), np.cos(tilt_rad)],
            ]
        )
        current = yaw @ tilt
        target = yaw_preserving_top_down_rotation(current)
        np.testing.assert_allclose(target.T @ target, np.eye(3), atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(target)), 1.0)
        np.testing.assert_allclose(target[:, 2], [0.0, 0.0, -1.0])
        current_x_heading = current[:2, 0] / np.linalg.norm(current[:2, 0])
        np.testing.assert_allclose(target[:2, 0], current_x_heading)
        self.assertAlmostEqual(top_down_tilt_error_rad(target), 0.0)

    def test_probe_lift_rejects_empty_grasp(self) -> None:
        report = grasp_follow_report(
            tcp_before_world_m=[0.0, 0.0, 0.80],
            tcp_after_world_m=[0.0, 0.0, 0.84],
            part_before_world_m=[0.0, 0.0, 0.77],
            part_after_world_m=[0.0, 0.0, 0.77],
            minimum_follow_ratio=0.60,
            maximum_follow_error_m=0.015,
        )
        self.assertFalse(report["pass"])

    def test_probe_lift_accepts_part_following_tcp(self) -> None:
        report = grasp_follow_report(
            tcp_before_world_m=[0.0, 0.0, 0.80],
            tcp_after_world_m=[0.0, 0.0, 0.84],
            part_before_world_m=[0.0, 0.0, 0.77],
            part_after_world_m=[0.001, 0.0, 0.809],
            minimum_follow_ratio=0.60,
            maximum_follow_error_m=0.015,
        )
        self.assertTrue(report["pass"])

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

    def test_all_six_slot_centers_preserve_frozen_geometry(self) -> None:
        centers = bin_slot_local_centers(
            size_m=[0.18, 0.12, 0.07],
            wall_thickness_m=0.006,
            bottom_thickness_m=0.006,
            part_height_m=0.044,
        )
        self.assertEqual(len(centers), 6)
        self.assertEqual(
            {(round(float(p[0]), 3), round(float(p[1]), 3)) for p in centers},
            {
                (-0.056, -0.027),
                (0.0, -0.027),
                (0.056, -0.027),
                (-0.056, 0.027),
                (0.0, 0.027),
                (0.056, 0.027),
            },
        )

    def test_safest_slot_avoids_cramped_inner_arm_region(self) -> None:
        candidates = [
            [-0.406, -0.177, 0.933],
            [-0.350, -0.177, 0.933],
            [-0.294, -0.177, 0.933],
            [-0.406, -0.123, 0.933],
            [-0.350, -0.123, 0.933],
            [-0.294, -0.123, 0.933],
        ]
        index = select_safest_slot_index(
            candidates,
            arm_base_world_m=[-0.55, -0.3, 0.75],
            soft_work_radius_m=0.65,
            work_radius_margin_m=0.03,
        )
        self.assertEqual(index, 5)

    def test_transfer_route_uses_safer_orthogonal_corner(self) -> None:
        start = [-0.9, 0.2, 0.923]
        destination = [-0.294, -0.123, 0.933]
        base = [-0.55, -0.3, 0.75]
        waypoints = orthogonal_transfer_waypoints(
            start,
            destination,
            arm_base_world_m=base,
            transit_clearance_m=0.04,
        )
        np.testing.assert_allclose(waypoints[0], [-0.294, 0.2, 0.973])
        np.testing.assert_allclose(waypoints[1], [-0.294, -0.123, 0.973])
        np.testing.assert_allclose(waypoints[2], destination)
        self.assertGreater(
            minimum_xy_radius_along_segment(start, waypoints[0], arm_base_world_m=base),
            0.30,
        )

    def test_motion_sample_guard_rejects_jump_and_divergence(self) -> None:
        self.assertIsNone(
            motion_sample_violation(
                [0.0, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                max_actual_step_m=0.06,
                divergence_tolerance_m=0.004,
            )
        )
        self.assertIn(
            "jumped",
            motion_sample_violation(
                [0.0, 0.0, 0.0],
                [0.07, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                max_actual_step_m=0.06,
                divergence_tolerance_m=0.004,
            ),
        )
        self.assertIn(
            "away",
            motion_sample_violation(
                [0.0, 0.0, 0.0],
                [-0.01, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                max_actual_step_m=0.06,
                divergence_tolerance_m=0.004,
            ),
        )

    def test_success_requires_two_of_exactly_three_fresh_frames(self) -> None:
        self.assertTrue(frozen_success_vote([True, False, True]))
        self.assertFalse(frozen_success_vote([False, True, False]))
        with self.assertRaisesRegex(ValueError, "exactly three"):
            frozen_success_vote([True, True])

    def test_step_limit_has_tracking_margin(self) -> None:
        self.assertEqual(conservative_step_limit(0.10, 0.02), 28)


if __name__ == "__main__":
    unittest.main()
