from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from simulation.pink_franka_adapter import (
    PinkFrankaAdapter,
    _joint_indices,
    _resolve_franka_mesh_root,
    _wxyz_rotation_matrix,
)


class PinkFrankaAdapterMathTests(unittest.TestCase):
    def test_identity_quaternion_becomes_identity_matrix(self):
        np.testing.assert_allclose(
            _wxyz_rotation_matrix(np.asarray([1.0, 0.0, 0.0, 0.0])),
            np.eye(3),
        )

    def test_joint_indices_follow_articulation_order(self):
        names = [
            "panda_joint3",
            "panda_finger_joint1",
            "panda_joint1",
            "panda_joint2",
            "panda_joint4",
            "panda_joint5",
            "panda_joint6",
            "panda_joint7",
            "panda_finger_joint2",
        ]
        self.assertEqual(_joint_indices(names), [2, 3, 0, 4, 5, 6, 7])

    def test_missing_arm_joint_fails_closed(self):
        with self.assertRaisesRegex(RuntimeError, "panda_joint7"):
            _joint_indices([f"panda_joint{i}" for i in range(1, 7)])

    def test_mesh_root_is_resolved_from_isaac_sim_tree(self):
        with TemporaryDirectory() as root:
            root_path = Path(root)
            urdf = (
                root_path
                / "exts"
                / "isaacsim.robot_motion.motion_generation"
                / "motion_policy_configs"
                / "franka"
                / "lula_franka_gen.urdf"
            )
            mesh = (
                root_path
                / "exts"
                / "isaacsim.asset.importer.urdf"
                / "data"
                / "urdf"
                / "robots"
                / "franka_description"
                / "meshes"
                / "collision"
                / "link0.stl"
            )
            urdf.parent.mkdir(parents=True)
            mesh.parent.mkdir(parents=True)
            urdf.touch()
            mesh.touch()
            self.assertTrue(
                Path(_resolve_franka_mesh_root(str(urdf))).samefile(mesh.parents[3]),
            )

    def test_compute_rolls_differential_steps_into_absolute_target(self):
        class Pin:
            @staticmethod
            def SE3(rotation, translation):
                return rotation, translation

        class FrameTask:
            def set_target(self, target):
                self.target = target

        class DifferentialController:
            def __init__(self):
                self.inputs = []

            def compute(self, current, dt_s):
                del dt_s
                current = np.asarray(current, dtype=float)
                self.inputs.append(current.copy())
                return current[:7] + (1.0 - current[:7]) * 0.5

        pink = object.__new__(PinkFrankaAdapter)
        controller = DifferentialController()
        pink._pin = Pin()
        pink._controllers = {"Arm_A": controller}
        pink._frame_tasks = {"Arm_A": FrameTask()}
        pink._controlled_indices = {"Arm_A": list(range(7))}
        pink._diagnostics = {"Arm_A": {}}

        targets = pink.compute(
            arm_id="Arm_A",
            current_joint_positions=np.zeros(9),
            target_position_base_m=np.zeros(3),
            target_orientation_base_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
            dt_s=0.1,
        )

        self.assertGreater(len(controller.inputs), 1)
        np.testing.assert_allclose(controller.inputs[1][:7], 0.5)

        # Predictive rollout approaches 1.0 rad, but one keyboard action must
        # remain inside the cumulative joint-space safety envelope.
        np.testing.assert_allclose(targets, 0.12, atol=1e-12)
        diagnostics = pink.diagnostics("Arm_A")
        self.assertGreater(diagnostics["rollout_iterations"], 1)
        self.assertTrue(diagnostics["rollout_clamped"])
        self.assertGreater(diagnostics["rollout_cumulative_delta_rad"], 0.12)
        self.assertEqual(diagnostics["rollout_action_joint_limit_rad"], 0.12)
        self.assertAlmostEqual(diagnostics["rollout_total_joint_delta_rad"], 0.12)

    def test_compute_rejects_joint_state_that_does_not_cover_arm(self):
        from simulation.pink_franka_adapter import PinkFrankaAdapter

        pink = object.__new__(PinkFrankaAdapter)
        pink._controlled_indices = {"Arm_A": list(range(7))}
        with self.assertRaisesRegex(ValueError, "does not cover"):
            pink.compute(
                arm_id="Arm_A",
                current_joint_positions=np.zeros(6),
                target_position_base_m=np.zeros(3),
                target_orientation_base_wxyz=np.asarray([1.0, 0.0, 0.0, 0.0]),
                dt_s=0.1,
            )

    def test_virtual_fk_updates_only_pink_configuration(self):
        class Transform:
            translation = np.asarray([0.1, 0.2, 0.3])
            rotation = np.eye(3)

        class Configuration:
            def __init__(self):
                self.updated = None

            def update(self, values):
                self.updated = np.asarray(values, dtype=float).copy()

            def get_transform_frame_to_world(self, frame):
                self.frame = frame
                return Transform()

        class Controller:
            pink_configuration = Configuration()

        pink = object.__new__(PinkFrankaAdapter)
        pink._controllers = {"Arm_B": Controller()}
        pink._control_frame_names = {"Arm_B": "right_gripper"}
        pink._controlled_indices = {"Arm_B": list(range(7))}

        position, rotation = pink.control_frame_pose_in_base(
            arm_id="Arm_B",
            joint_positions=np.arange(9, dtype=float),
        )

        np.testing.assert_allclose(position, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(rotation, np.eye(3))
        np.testing.assert_allclose(
            pink._controllers["Arm_B"].pink_configuration.updated,
            np.arange(7, dtype=float),
        )


if __name__ == "__main__":
    unittest.main()
