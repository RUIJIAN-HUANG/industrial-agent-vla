from __future__ import annotations

import unittest

from industrial_agent.contracts import Postcondition, TaskSchema
from industrial_agent.observation import ObservationGateway
from industrial_agent.verifier import PostconditionVerifier, Verdict

from tests.test_contracts_and_observation import raw_observation


def frame(status: str, confidence: float = 1.0, index: int = 1):
    raw = raw_observation()
    raw["observation_id"] = f"obs-{index}"
    raw["task"] = {"status": status}
    raw["quality"] = {"confidence": confidence}
    return ObservationGateway().ingest_online(raw)


class VerifierTests(unittest.TestCase):
    def setUp(self) -> None:
        self.task = TaskSchema(
            task_id="verify",
            instruction="verify task",
            task_type="mock_demo",
            postconditions=(
                Postcondition(
                    kind="field_equals",
                    path="task.status",
                    expected="done",
                    min_confidence=0.6,
                    required_votes=2,
                ),
            ),
        )

    def test_two_of_three_pass(self) -> None:
        result = PostconditionVerifier().verify(
            self.task,
            [frame("done", index=1), frame("pending", index=2), frame("done", index=3)],
        )
        self.assertEqual(result.verdict, Verdict.PASS)

    def test_low_confidence_frames_produce_uncertain(self) -> None:
        result = PostconditionVerifier().verify(
            self.task,
            [
                frame("done", 0.2, 1),
                frame("done", 0.2, 2),
                frame("done", 1.0, 3),
            ],
        )
        self.assertEqual(result.verdict, Verdict.UNCERTAIN)

    def test_invalid_frame_confidence_never_votes_pass(self) -> None:
        invalid_values = (
            "not-a-number",
            True,
            None,
            [],
            float("nan"),
            float("inf"),
            float("-inf"),
        )
        for index, value in enumerate(invalid_values, start=1):
            with self.subTest(value=value):
                result = PostconditionVerifier().verify(
                    self.task,
                    [frame("done", value, index)],  # type: ignore[arg-type]
                )
                self.assertEqual(result.conditions[0].pass_votes, 0)
                self.assertEqual(result.conditions[0].uncertain_votes, 1)

    def test_non_object_frame_quality_never_votes_pass(self) -> None:
        raw = raw_observation()
        raw["quality"] = None
        observation = ObservationGateway().ingest_online(raw)
        result = PostconditionVerifier().verify(self.task, [observation])
        self.assertEqual(result.conditions[0].pass_votes, 0)
        self.assertEqual(result.conditions[0].uncertain_votes, 1)

    def test_duplicate_observation_cannot_vote_twice(self) -> None:
        duplicate = frame("done", index=1)
        result = PostconditionVerifier().verify(
            self.task, [duplicate, duplicate, frame("pending", index=2)]
        )
        self.assertEqual(result.verdict, Verdict.UNCERTAIN)

    def test_numeric_range_rejects_non_finite_and_non_numeric_values(self) -> None:
        task = TaskSchema(
            task_id="numeric-verify",
            instruction="verify numeric task",
            task_type="mock_demo",
            postconditions=(
                Postcondition(
                    kind="numeric_range",
                    path="task.value",
                    minimum=0.0,
                    maximum=1.0,
                    required_votes=1,
                ),
            ),
        )
        invalid_values = (
            "not-a-number",
            True,
            float("nan"),
            float("inf"),
            -float("inf"),
        )
        for index, value in enumerate(invalid_values, start=1):
            with self.subTest(value=value):
                raw = raw_observation()
                raw["observation_id"] = f"numeric-{index}"
                raw["task"] = {"value": value}
                observation = ObservationGateway().ingest_online(raw)
                result = PostconditionVerifier().verify(task, [observation])
                self.assertEqual(result.verdict, Verdict.UNCERTAIN)
                self.assertEqual(result.conditions[0].uncertain_votes, 1)
                self.assertIn("finite numeric value", result.conditions[0].detail)

    def test_conflicting_vote_quorums_fail_closed(self) -> None:
        result = PostconditionVerifier().verify(
            self.task,
            [
                frame("done", index=1),
                frame("done", index=2),
                frame("pending", index=3),
                frame("pending", index=4),
            ],
        )
        self.assertEqual(result.verdict, Verdict.FAIL)
        self.assertIn("fail-closed", result.conditions[0].detail)

    def test_invalid_object_confidence_never_votes_pass(self) -> None:
        task = TaskSchema(
            task_id="object-verify",
            instruction="verify object",
            task_type="mock_demo",
            postconditions=(
                Postcondition(
                    kind="object_detected",
                    object_id="red-box",
                    required_votes=1,
                ),
            ),
        )
        invalid_values = (
            "not-a-number",
            True,
            float("nan"),
            float("inf"),
            float("-inf"),
        )
        for index, value in enumerate(invalid_values, start=1):
            with self.subTest(value=value):
                raw = raw_observation()
                raw["observation_id"] = f"object-{index}"
                raw["objects"] = [
                    {"object_id": "red-box", "confidence": value},
                ]
                observation = ObservationGateway().ingest_online(raw)
                result = PostconditionVerifier().verify(task, [observation])
                self.assertEqual(result.verdict, Verdict.UNCERTAIN)
                self.assertEqual(result.conditions[0].pass_votes, 0)
                self.assertEqual(result.conditions[0].uncertain_votes, 1)


if __name__ == "__main__":
    unittest.main()
