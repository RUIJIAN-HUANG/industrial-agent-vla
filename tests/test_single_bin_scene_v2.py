from __future__ import annotations

import unittest
from copy import deepcopy

from simulation.v2_scene_contract import (
    DEFAULT_CONFIG_PATH,
    EXPECTED_PARTS,
    EXPECTED_SLOTS,
    load_config,
    mass_budget,
    validate_config,
)


class V2SceneContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config(DEFAULT_CONFIG_PATH)

    def test_frozen_v2_config_is_valid(self) -> None:
        self.assertEqual(validate_config(self.config), [])

    def test_eight_parts_and_slots_have_exact_mapping(self) -> None:
        self.assertEqual(
            {part["id"] for part in self.config["parts"]}, set(EXPECTED_PARTS)
        )
        self.assertEqual(
            {slot["id"] for slot in self.config["bin"]["slots"]},
            set(EXPECTED_SLOTS),
        )
        self.assertEqual(
            {slot["part_id"] for slot in self.config["bin"]["slots"]},
            set(EXPECTED_PARTS),
        )

    def test_two_upright_and_two_inverted_shafts_are_frozen(self) -> None:
        states = {
            part["id"]: part["initial_orientation_state"]
            for part in self.config["parts"]
            if part["part_type"] == "shaft"
        }
        self.assertEqual(
            states,
            {
                "P01": "upright",
                "P02": "upright",
                "P03": "inverted",
                "P04": "inverted",
            },
        )

    def test_mass_and_center_of_mass_budget(self) -> None:
        budget = mass_budget(self.config)
        self.assertAlmostEqual(budget["planned_loaded_mass_kg"], 1.0)
        self.assertLessEqual(budget["planned_loaded_mass_kg"], 1.2)
        self.assertLessEqual(budget["carry_tcp_projection_error_m"], 0.010)

    def test_rejects_orientation_slot_and_tcp_drift(self) -> None:
        mutations = (
            (
                lambda config: config["parts"][2].update(
                    {"initial_orientation_state": "upright"}
                ),
                "parts.P03.initial_orientation_state",
            ),
            (
                lambda config: config["bin"]["slots"][0].update(
                    {"part_id": "P02"}
                ),
                "bin.slots.S11.part_id",
            ),
            (
                lambda config: config["bin"]["carry_handle"].update(
                    {"frame_id": "WRONG_TCP"}
                ),
                "BIN_CARRY_TCP",
            ),
        )
        for mutation, expected in mutations:
            config = deepcopy(self.config)
            mutation(config)
            with self.subTest(expected=expected):
                self.assertTrue(
                    any(expected in error for error in validate_config(config))
                )

    def test_rejects_camera_token_and_online_gt_drift(self) -> None:
        config = deepcopy(self.config)
        config["cameras"].pop()
        config["workflow"]["token_sequence"] = ["A_ONLY", "B_ONLY"]
        config["collection"]["online_gt_allowed"] = True
        errors = validate_config(config)
        self.assertTrue(any("three frozen IDs" in error for error in errors))
        self.assertTrue(any("token_sequence" in error for error in errors))
        self.assertTrue(any("online_gt_allowed" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
