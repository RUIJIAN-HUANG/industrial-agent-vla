from __future__ import annotations

import unittest

from simulation.run_v2_ik_reachability_acceptance import _ik_targets
from simulation.v2_scene_contract import load_config


class V2IkReachabilityContractTests(unittest.TestCase):
    def test_eight_safe_position_only_targets_follow_the_frozen_relay(self) -> None:
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

    def test_arm_b_starts_at_handoff_and_never_probes_arm_a_pack_station(self) -> None:
        config = load_config()
        targets = {item["target_id"]: item for item in _ik_targets(config)}
        self.assertNotIn("ARM_B_PACK_STATION_HANDLE_APPROACH", targets)

        target = targets["ARM_B_HANDOFF_CENTER_HANDLE_APPROACH"]
        stations = {item["id"]: item for item in config["stations"]}
        initial_station = stations[config["bin"]["initial_station_id"]]
        bin_offset = [
            config["bin"]["pose"]["position_m"][index]
            - initial_station["pose"]["position_m"][index]
            for index in range(3)
        ]
        expected = [
            value + bin_offset[index]
            for index, value in enumerate(
                stations["HANDOFF_CENTER"]["pose"]["position_m"]
            )
        ]
        expected[2] += config["bin"]["carry_handle"]["position_local_m"][2]
        expected[2] += config["bin"]["carry_handle"]["approach_offset_m"][2]
        self.assertEqual(target["position_world_m"], expected)


if __name__ == "__main__":
    unittest.main()
