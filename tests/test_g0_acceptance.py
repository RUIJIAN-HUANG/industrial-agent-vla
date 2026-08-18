from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from simulation.run_g0_acceptance import (
    REQUIRED_PRIMS,
    _explicit_home_target,
    _home_readback_errors,
    _robot_state,
    _write_explicit_home,
    _write_ppm,
)


class FakeArticulation:
    def __init__(self, *, dof_names=None, joint_names=None) -> None:
        if dof_names is not None:
            self.dof_names = dof_names
        if joint_names is not None:
            self.joint_names = joint_names
        self.positions = [0.1, 0.2]
        self.velocities = [0.0, 0.0]

    def get_joint_positions(self):
        return list(self.positions)

    def get_joint_velocities(self):
        return list(self.velocities)

    def set_joint_positions(self, values):
        self.positions = list(values)

    def set_joint_velocities(self, values):
        self.velocities = list(values)


class G0AcceptanceTests(unittest.TestCase):
    @staticmethod
    def _home_config() -> dict:
        home = {
            "arm_joint_positions_rad": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
            "finger_joint_positions_m": [0.04, 0.04],
        }
        return {"robots": [{"id": "Arm_A", "home": home}]}

    @staticmethod
    def _franka() -> FakeArticulation:
        return FakeArticulation(
            dof_names=[
                "panda_joint1",
                "panda_joint2",
                "panda_joint3",
                "panda_joint4",
                "panda_joint5",
                "panda_joint6",
                "panda_joint7",
                "panda_finger_joint1",
                "panda_finger_joint2",
            ]
        )

    def test_explicit_home_target_uses_config_and_articulation_order(self) -> None:
        arm = self._franka()
        target = _explicit_home_target(self._home_config(), arm, "Arm_A")
        self.assertEqual(target, [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.04, 0.04])

    def test_explicit_home_is_written_and_read_back(self) -> None:
        arm = self._franka()
        target = _write_explicit_home(self._home_config(), arm, "Arm_A")
        self.assertEqual(_home_readback_errors(arm, "Arm_A", target), [])
        self.assertEqual(arm.velocities, [0.0] * 9)

    def test_home_readback_rejects_position_drift(self) -> None:
        arm = self._franka()
        target = _write_explicit_home(self._home_config(), arm, "Arm_A")
        arm.positions[0] += 0.002
        self.assertTrue(_home_readback_errors(arm, "Arm_A", target))

    def test_frozen_station_prims_are_required(self) -> None:
        self.assertTrue(
            {
                "/World/Stations/PACK_STATION",
                "/World/Stations/HANDOFF_CENTER",
                "/World/Stations/FINISHED_01",
            }.issubset(REQUIRED_PRIMS)
        )
        self.assertEqual(len(REQUIRED_PRIMS), 13)

    def test_camera_evidence_requires_frozen_resolution(self) -> None:
        with TemporaryDirectory() as directory:
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "resolution mismatch"):
                _write_ppm(Path(directory) / "frame.ppm", frame)

    def test_camera_evidence_rejects_uniform_frame(self) -> None:
        with TemporaryDirectory() as directory:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            with self.assertRaisesRegex(ValueError, "spatially uniform"):
                _write_ppm(Path(directory) / "frame.ppm", frame)

    def test_camera_evidence_records_pixel_digest(self) -> None:
        with TemporaryDirectory() as directory:
            frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            frame[0, 0, :] = 255
            path = Path(directory) / "frame.ppm"

            stats = _write_ppm(path, frame)

            self.assertTrue(path.is_file())
            self.assertEqual(stats["actual_resolution_px"], [1280, 720])
            self.assertEqual(len(stats["pixel_sha256"]), 64)

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
