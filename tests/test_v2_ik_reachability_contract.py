from __future__ import annotations

import unittest

from simulation.run_v2_ik_reachability_acceptance import _ik_targets
from simulation.v2_scene_contract import load_config


class V2IkReachabilityContractTests(unittest.TestCase):
    def test_eight_safe_position_only_targets_are_frozen(self) -> None:
        targets = _ik_targets(load_config())
        self.assertEqual(len(targets), 8)
        self.assertEqual(
            [item["target_id"] for item in targets],
            [
                "ARM_A_ZONE_A_SAFE_APPROACH",
                "ARM_A_ZONE_B_SAFE_APPROACH",
                "ARM_A_ZONE_C_SAFE_APPROACH",
                "ARM_A_ZONE_D_SAFE_APPROACH",
                "ARM_A_PACK_STATION_HANDLE_APPROACH",
                "ARM_A_HANDOFF_CENTER_HANDLE_APPROACH",
                "ARM_B_HANDOFF_CENTER_HANDLE_APPROACH",
                "ARM_B_FINISHED_01_HANDLE_APPROACH",
            ],
        )
        self.assertTrue(all(item["position_world_m"][2] >= 0.98 for item in targets))

    def test_position_only_probe_assigns_targets_to_expected_arms(self) -> None:
        targets = _ik_targets(load_config())
        self.assertEqual(
            [item["arm_id"] for item in targets],
            ["Arm_A"] * 6 + ["Arm_B"] * 2,
        )


if __name__ == "__main__":
    unittest.main()
