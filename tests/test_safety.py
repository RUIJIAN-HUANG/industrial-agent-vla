from __future__ import annotations

import math
import unittest

from industrial_agent.contracts import ACTION_CONTRACT_VERSION, ActionChunk, ActionStep
from industrial_agent.errors import FailureCode
from industrial_agent.observation import ObservationGateway
from industrial_agent.safety import ActionSafetyValidator, safety_state_failure

from tests.test_contracts_and_observation import raw_observation


def chunk(values: list[float]) -> ActionChunk:
    return ActionChunk(
        contract_version=ACTION_CONTRACT_VERSION,
        chunk_id="c1",
        task_id="t1",
        executor="mock",
        steps=(ActionStep.from_sequence(values),),
    )


class SafetyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.observation = ObservationGateway().ingest_online(raw_observation())
        self.validator = ActionSafetyValidator()

    def test_axis_values_are_limited_before_execution(self) -> None:
        decision = self.validator.validate_and_limit(
            chunk([0.5, 0, 0, 1.0, 0, 0, 3.0]), self.observation
        )
        self.assertTrue(decision.accepted)
        assert decision.chunk is not None
        self.assertEqual(decision.chunk.steps[0].values[0], 0.05)
        self.assertEqual(decision.chunk.steps[0].values[3], 0.25)
        self.assertEqual(decision.chunk.steps[0].values[6], 1.0)
        self.assertEqual(set(decision.limited_axes), {"dx", "droll", "gripper"})

    def test_nan_is_rejected_not_limited(self) -> None:
        decision = self.validator.validate_and_limit(
            chunk([math.nan, 0, 0, 0, 0, 0, 0]), self.observation
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, FailureCode.ACTION_NON_FINITE)

    def test_projected_workspace_breach_is_rejected(self) -> None:
        raw = raw_observation()
        raw["robot"] = {"tcp_pose_m_rad": [0.99, 0.0, 0.5, 0, 0, 0]}
        observation = ObservationGateway().ingest_online(raw)
        decision = self.validator.validate_and_limit(
            chunk([0.2, 0, 0, 0, 0, 0, 0]), observation
        )
        self.assertFalse(decision.accepted)
        self.assertEqual(decision.code, FailureCode.ACTION_WORKSPACE_BREACH)

    def test_incomplete_safety_state_is_a_system_fault(self) -> None:
        raw = raw_observation()
        raw["safety"] = {"emergency_stop": False}
        observation = ObservationGateway().ingest_online(raw)
        failure = safety_state_failure(observation)
        self.assertIsNotNone(failure)
        assert failure is not None
        self.assertEqual(failure[0], FailureCode.SYSTEM_FAULT)


if __name__ == "__main__":
    unittest.main()
