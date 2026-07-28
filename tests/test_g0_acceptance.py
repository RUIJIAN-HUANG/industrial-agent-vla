from __future__ import annotations

import unittest

from simulation.run_g0_acceptance import REQUIRED_PRIMS, _robot_state


class FakeArticulation:
    def __init__(self, *, dof_names=None, joint_names=None) -> None:
        if dof_names is not None:
            self.dof_names = dof_names
        if joint_names is not None:
            self.joint_names = joint_names

    def get_joint_positions(self):
        return [0.1, 0.2]

    def get_joint_velocities(self):
        return [0.0, 0.0]


class G0AcceptanceTests(unittest.TestCase):
    def test_frozen_station_prims_are_required(self) -> None:
        self.assertTrue(
            {
                "/World/Stations/PACK_STATION",
                "/World/Stations/HANDOFF_CENTER",
                "/World/Stations/FINISHED_01",
            }.issubset(REQUIRED_PRIMS)
        )

    def test_robot_state_uses_isaac_51_dof_names(self) -> None:
        arm = FakeArticulation(dof_names=["joint_a", "joint_b"])

        state = _robot_state(arm, "Arm_A")

        self.assertEqual(state["joint_names"], ["joint_a", "joint_b"])

    def test_robot_state_falls_back_to_legacy_joint_names(self) -> None:
        arm = FakeArticulation(joint_names=["joint_a", "joint_b"])

        state = _robot_state(arm, "Arm_A")

        self.assertEqual(state["joint_names"], ["joint_a", "joint_b"])

    def test_robot_state_rejects_mismatched_lengths(self) -> None:
        arm = FakeArticulation(dof_names=["joint_a"])

        with self.assertRaisesRegex(RuntimeError, "size mismatch"):
            _robot_state(arm, "Arm_A")


if __name__ == "__main__":
    unittest.main()
