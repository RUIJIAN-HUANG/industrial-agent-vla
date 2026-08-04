from __future__ import annotations

import math
import unittest
from copy import deepcopy

from simulation.scene_layout import (
    DEFAULT_CONFIG_PATH,
    load_config,
    reach_report,
    validate_scene_config,
)


class FrozenSceneLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(DEFAULT_CONFIG_PATH)

    def test_frozen_config_is_valid(self) -> None:
        self.assertEqual(validate_scene_config(self.config), [])

    def test_expected_planar_reach_distances(self) -> None:
        actual = {
            (item["robot_id"], item["target_id"]): item
            for item in reach_report(self.config)
        }
        expected = {
            ("Arm_A", "PACK_STATION"): 0.25,
            ("Arm_A", "P01"): math.sqrt(0.35**2 + 0.5**2),
            ("Arm_A", "P02"): math.sqrt(0.25**2 + 0.5**2),
            ("Arm_A", "P03"): math.sqrt(0.05**2 + 0.5**2),
            ("Arm_A", "P04"): math.sqrt(0.3**2 + 0.3**2),
            ("Arm_A", "HANDOFF_CENTER"): math.sqrt(0.55**2 + 0.3**2),
            ("Arm_B", "HANDOFF_CENTER"): math.sqrt(0.5**2 + 0.3**2),
            ("Arm_B", "FINISHED_01"): math.sqrt(0.2**2 + 0.4**2),
        }
        self.assertEqual(set(actual), set(expected))
        for endpoint, expected_distance in expected.items():
            with self.subTest(endpoint=endpoint):
                self.assertAlmostEqual(
                    actual[endpoint]["distance_m"],
                    expected_distance,
                    places=9,
                )
                self.assertTrue(actual[endpoint]["within_soft_limit"])

    def test_rejects_coordinate_drift_and_out_of_reach_target(self) -> None:
        config = deepcopy(self.config)
        config["stations"][1]["pose"]["position_m"] = [0.2, 0.2, 0.785]

        errors = validate_scene_config(config)

        self.assertTrue(
            any("HANDOFF_CENTER.pose.position_m" in error for error in errors)
        )
        self.assertTrue(any("Arm_A -> HANDOFF_CENTER" in error for error in errors))

    def test_rejects_missing_or_changed_explicit_home(self) -> None:
        mutations = {
            "missing home": lambda config: config["robots"][0].pop("home"),
            "wrong arm length": lambda config: config["robots"][0]["home"].update(
                {"arm_joint_positions_rad": [0.0] * 6}
            ),
            "startup-derived drift": lambda config: config["robots"][1]["home"][
                "arm_joint_positions_rad"
            ].__setitem__(0, 0.02),
            "closed home gripper": lambda config: config["robots"][1]["home"].update(
                {"finger_joint_positions_m": [0.0, 0.0]}
            ),
        }
        for label, mutation in mutations.items():
            config = deepcopy(self.config)
            mutation(config)
            with self.subTest(label=label):
                errors = validate_scene_config(config)
                self.assertTrue(
                    any(".home" in error for error in errors),
                    errors,
                )

    def test_rejects_inverted_part_pose_recipe_and_token_tampering(self) -> None:
        mutations = {
            "inverted pose": lambda config: config["parts"][1]["pose"].update(
                {"rpy_deg": [0.0, 0.0, 0.0]}
            ),
            "recipe": lambda config: config["bin"].update(
                {"recipe_part_ids": ["P01", "P02", "P03", "P03"]}
            ),
            "token order": lambda config: config["workflow"].update(
                {"token_sequence": ["A_ONLY", "B_ONLY"]}
            ),
        }
        expected_messages = {
            "inverted pose": "parts.P02.pose.rpy_deg",
            "recipe": "bin.recipe_part_ids",
            "token order": "workflow.token_sequence",
        }
        for label, mutation in mutations.items():
            config = deepcopy(self.config)
            mutation(config)
            with self.subTest(label=label):
                errors = validate_scene_config(config)
                self.assertTrue(
                    any(expected_messages[label] in error for error in errors),
                    errors,
                )

    def test_rejects_duplicate_object_id_and_wrong_zone_distribution(self) -> None:
        config = deepcopy(self.config)
        config["parts"][3]["id"] = "P03"
        config["parts"][2]["zone_id"] = "A"

        errors = validate_scene_config(config)

        self.assertTrue(any("duplicate id 'P03'" in error for error in errors))
        self.assertTrue(any("zone A contains 3 parts" in error for error in errors))
        self.assertTrue(any("zone B contains 0 parts" in error for error in errors))

    def test_rejects_missing_builder_required_bin_and_physics_fields(self) -> None:
        mutations = {
            "bin.mass_kg": lambda config: config["bin"].pop("mass_kg"),
            "bin.wall_thickness_m": lambda config: config["bin"].pop(
                "wall_thickness_m"
            ),
            "bin.divider_thickness_m": lambda config: config["bin"].pop(
                "divider_thickness_m"
            ),
            "bin.bottom_thickness_m": lambda config: config["bin"].pop(
                "bottom_thickness_m"
            ),
            "bin.handle": lambda config: config["bin"].pop("handle"),
            "physics.gravity_m_s2": lambda config: config["physics"].pop(
                "gravity_m_s2"
            ),
            "physics.physics_dt_s": lambda config: config["physics"].pop(
                "physics_dt_s"
            ),
            "physics.rendering_dt_s": lambda config: config["physics"].pop(
                "rendering_dt_s"
            ),
            "physics.control_frequency_hz": lambda config: config["physics"].pop(
                "control_frequency_hz"
            ),
            "physics.render_frequency_hz": lambda config: config["physics"].pop(
                "render_frequency_hz"
            ),
            "physics.model_inference_frequency_hz": lambda config: config[
                "physics"
            ].pop("model_inference_frequency_hz"),
        }

        for expected_label, mutation in mutations.items():
            config = deepcopy(self.config)
            mutation(config)
            with self.subTest(field=expected_label):
                errors = validate_scene_config(config)
                self.assertTrue(
                    any(expected_label in error for error in errors),
                    errors,
                )

    def test_rejects_agent_owned_policy_fields_and_legacy_event_name(self) -> None:
        config = deepcopy(self.config)
        config["safety"]["normal_workspace_limits"] = {
            "Arm_A": {"max_x_m": -0.16},
            "Arm_B": {"min_x_m": 0.16},
        }
        config["workflow"]["handoff_verify_stable_cycles"] = 3
        config["workflow"]["handoff_ready_event"] = "handoff_ready"

        errors = validate_scene_config(config)

        self.assertTrue(
            any("safety.normal_workspace_limits" in error for error in errors),
            errors,
        )
        self.assertTrue(
            any("workflow.handoff_verify_stable_cycles" in error for error in errors),
            errors,
        )
        self.assertTrue(
            any("workflow.handoff_ready_event" in error for error in errors),
            errors,
        )


if __name__ == "__main__":
    unittest.main()
