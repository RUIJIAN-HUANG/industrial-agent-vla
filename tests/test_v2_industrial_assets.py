from __future__ import annotations

import unittest
from copy import deepcopy

from simulation.v2_industrial_assets import asset_summary, validate_part_spec
from simulation.v2_scene_contract import load_config


class V2IndustrialAssetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_config()

    def test_all_eight_asset_specs_are_program_generated_and_valid(self) -> None:
        summary = asset_summary(self.config["parts"])
        self.assertEqual(summary["errors"], [])
        self.assertEqual(summary["type_counts"], {"nut": 2, "shaft": 4, "wrench": 2})
        self.assertTrue(summary["all_program_generated"])
        self.assertEqual(summary["external_assets"], [])

    def test_shaft_flange_makes_orientation_visible(self) -> None:
        shaft = deepcopy(self.config["parts"][0])
        shaft["geometry"]["flange_radius_m"] = shaft["geometry"]["radius_m"]
        self.assertTrue(
            any("orientation visible" in item for item in validate_part_spec(shaft))
        )

    def test_nut_requires_a_real_visible_hole(self) -> None:
        nut = deepcopy(self.config["parts"][4])
        nut["geometry"]["hole_diameter_m"] = nut["geometry"]["across_flats_m"]
        self.assertTrue(
            any("wall thickness" in item for item in validate_part_spec(nut))
        )

    def test_wrench_requires_distinct_head_and_long_handle(self) -> None:
        wrench = deepcopy(self.config["parts"][6])
        wrench["geometry"]["head_width_m"] = wrench["geometry"]["handle_width_m"]
        self.assertTrue(
            any("visually distinct" in item for item in validate_part_spec(wrench))
        )


if __name__ == "__main__":
    unittest.main()
