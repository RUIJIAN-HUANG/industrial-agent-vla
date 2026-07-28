from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from simulation.run_g0_acceptance import (
    REQUIRED_PRIMS,
    _robot_state,
    _write_ppm,
)


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
