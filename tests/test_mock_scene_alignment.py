from __future__ import annotations

import unittest

from industrial_agent.environment import execution_guard_digest
from industrial_agent.mock import FixedDualArmMockSimulator


class MockSceneAlignmentTests(unittest.TestCase):
    def test_mock_uses_three_frozen_cameras_at_scene_resolution(self) -> None:
        camera = FixedDualArmMockSimulator().observe()["camera"]

        self.assertEqual(
            {reference["camera_id"] for reference in camera.values()},
            {"CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP"},
        )
        self.assertNotIn("wrist_image", camera)
        for key, reference in camera.items():
            with self.subTest(image=key):
                self.assertEqual(
                    (reference["width"], reference["height"]),
                    (1280, 720),
                )

    def test_execution_guard_and_observation_share_the_same_state_builder(
        self,
    ) -> None:
        simulator = FixedDualArmMockSimulator()
        observation = simulator.observe()
        guarded_keys = ("robot", "safety", "task", "objects", "quality")
        observed_state = {key: observation[key] for key in guarded_keys}

        self.assertEqual(simulator._critical_observation_data(), observed_state)
        self.assertEqual(
            execution_guard_digest(simulator._critical_observation_data()),
            execution_guard_digest(observation),
        )


if __name__ == "__main__":
    unittest.main()
