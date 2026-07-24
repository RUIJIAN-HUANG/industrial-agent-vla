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

    def test_duplicate_observation_cannot_vote_twice(self) -> None:
        duplicate = frame("done", index=1)
        result = PostconditionVerifier().verify(
            self.task, [duplicate, duplicate, frame("pending", index=2)]
        )
        self.assertEqual(result.verdict, Verdict.UNCERTAIN)


if __name__ == "__main__":
    unittest.main()
