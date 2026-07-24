from __future__ import annotations

import unittest

from industrial_agent.errors import FailureCode, ObservationError
from industrial_agent.observation import ObservationGateway


def raw_observation() -> dict[str, object]:
    return {
        "observation_version": "1.0",
        "observation_id": "obs-security-1",
        "timestamp_ms": 1,
        "camera": {},
        "objects": [],
        "robot": {"tcp_pose_m_rad": [0.5, 0.0, 0.5, 0.0, 0.0, 0.0]},
        "safety": {
            "emergency_stop": False,
            "protective_stop": False,
            "system_fault": None,
        },
        "task": {"status": "pending"},
        "quality": {"confidence": 1.0},
    }


class ObservationGroundTruthSecurityTests(unittest.TestCase):
    def assert_forbidden(self, metadata: object) -> None:
        raw = raw_observation()
        raw["camera"] = {"metadata": metadata}

        with self.assertRaises(ObservationError) as caught:
            ObservationGateway().ingest_online(raw)

        self.assertEqual(
            caught.exception.code,
            FailureCode.OBSERVATION_GT_FORBIDDEN,
        )

    def test_rejects_gt_and_privileged_keys_in_common_naming_styles(self) -> None:
        forbidden_keys = (
            "targetPose",
            "target-pose",
            "target_pose",
            "target pose",
            "desired_pose",
            "goalPosition",
            "graspCoordinates",
            "waypoint position",
            "groundTruthPose",
            "ground-truth-pose",
            "ground_truth_pose",
            "privileged_state",
            "true_pose",
            "objectCoordinates",
            "reference_pose",
            "target_x",
            "targetX",
            "grasp_y",
            "desiredYaw",
            "truth_pose",
            "targetLocation",
            "targetMatrix",
            "target_rx",
            "actualPose",
            "realEuler",
            "target1Pose",
            "target2_x",
            "targetsPose",
            "goalsCoordinates",
            "waypoint1Pose",
            "waypoints2",
        )

        for key in forbidden_keys:
            with self.subTest(key=key):
                self.assert_forbidden({key: [0.1, 0.2, 0.3]})

    def test_rejects_nested_target_grasp_and_waypoint_geometry(self) -> None:
        forbidden_payloads = (
            {"target": {"pose": [0.1, 0.2, 0.3]}},
            {"desired": {"coordinates": [0.1, 0.2, 0.3]}},
            {"goal": {"x": 0.1, "y": 0.2, "z": 0.3}},
            {"grasp": {"point": [0.1, 0.2, 0.3]}},
            {"route": {"waypoints": [[0.1, 0.2, 0.3]]}},
            {"target": [0.1, 0.2, 0.3]},
            {"target": {"value": [0.1, 0.2, 0.3]}},
            {"grasp": {"data": [0.1, 0.2, 0.3]}},
            {"desired": [[1.0, 0.0], [0.0, 1.0]]},
            {"actual": {"value": 0.1}},
        )

        for payload in forbidden_payloads:
            with self.subTest(payload=payload):
                self.assert_forbidden(payload)

    def test_allows_normal_task_object_and_robot_observation_fields(self) -> None:
        raw = raw_observation()
        raw["task"] = {
            "status": "pending",
            "target": {"object_id": "red-cylinder", "status": "visible"},
            "desired_status": "ready",
        }
        raw["objects"] = [
            {
                "object_id": "red-cylinder",
                "status": "visible",
                "pose": [0.1, 0.2, 0.3],
                "position": [0.1, 0.2, 0.3],
            }
        ]

        observation = ObservationGateway().ingest_online(raw)

        self.assertEqual(observation.data["task"]["status"], "pending")
        self.assertEqual(observation.data["objects"][0]["status"], "visible")


if __name__ == "__main__":
    unittest.main()
