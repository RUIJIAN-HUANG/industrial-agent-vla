from __future__ import annotations

import unittest

from industrial_agent.contracts import ActionStep, Postcondition, TaskSchema
from industrial_agent.errors import FailureCode, ObservationError
from industrial_agent.observation import ObservationGateway


def raw_observation() -> dict[str, object]:
    return {
        "observation_version": "1.0",
        "observation_id": "obs-1",
        "timestamp_ms": 1,
        "camera": {},
        "objects": [],
        "robot": {"tcp_pose_m_rad": [0.5, 0.0, 0.5, 0, 0, 0]},
        "safety": {
            "emergency_stop": False,
            "protective_stop": False,
            "system_fault": None,
        },
        "task": {"status": "pending"},
        "quality": {"confidence": 1.0},
    }


class ContractAndObservationTests(unittest.TestCase):
    def test_action_step_requires_exactly_seven_dimensions(self) -> None:
        with self.assertRaisesRegex(Exception, "7-D"):
            ActionStep.from_sequence([0.0] * 6)

    def test_ground_truth_is_rejected_at_any_depth(self) -> None:
        raw = raw_observation()
        raw["camera"] = {"metadata": {"ground_truth": {"pose": [1, 2, 3]}}}
        with self.assertRaises(ObservationError) as caught:
            ObservationGateway().ingest_online(raw)
        self.assertEqual(caught.exception.code, FailureCode.OBSERVATION_GT_FORBIDDEN)

    def test_compound_gt_and_privileged_pose_fields_are_rejected(self) -> None:
        for forbidden_key in ("ground_truth_pose", "sim_gt_mask", "target_pose"):
            raw = raw_observation()
            raw["camera"] = {"metadata": {forbidden_key: [1, 2, 3]}}
            with self.subTest(forbidden_key=forbidden_key):
                with self.assertRaises(ObservationError) as caught:
                    ObservationGateway().ingest_online(raw)
                self.assertEqual(
                    caught.exception.code, FailureCode.OBSERVATION_GT_FORBIDDEN
                )

    def test_required_fields_and_fresh_observation_ids(self) -> None:
        raw = raw_observation()
        gateway = ObservationGateway()
        gateway.ingest_online(raw)
        with self.assertRaises(ObservationError):
            gateway.ingest_online(raw)

        missing_safety = raw_observation()
        del missing_safety["safety"]
        with self.assertRaises(ObservationError) as caught:
            ObservationGateway().ingest_online(missing_safety)
        self.assertEqual(caught.exception.code, FailureCode.OBSERVATION_INVALID)

    def test_non_allowlisted_top_level_field_is_rejected(self) -> None:
        raw = raw_observation()
        raw["debug_dump"] = {}
        with self.assertRaises(ObservationError) as caught:
            ObservationGateway().ingest_online(raw)
        self.assertEqual(caught.exception.code, FailureCode.OBSERVATION_INVALID)

    def test_task_rejects_incompatible_major_version(self) -> None:
        task = TaskSchema(
            task_id="t",
            instruction="do task",
            task_type="mock_demo",
            schema_version="2.0",
            postconditions=(
                Postcondition(kind="field_equals", path="task.status", expected="done"),
            ),
        )
        with self.assertRaisesRegex(Exception, "incompatible"):
            task.validate()


if __name__ == "__main__":
    unittest.main()
