from __future__ import annotations

from copy import deepcopy
import unittest

from industrial_agent.contracts import ActionStep
from industrial_agent.environment import SafeStopReceipt, execution_guard_digest
from industrial_agent.isaac_environment import IsaacExecutionEnvironment


class _ObservationSource:
    def __init__(self) -> None:
        self.counter = 0
        self.arm_a_retreated = True
        self.arm_b_retreated = True
        self.protective_stop = False

    def __call__(self):
        self.counter += 1
        return {
            "observation_version": "1.0",
            "observation_id": f"obs-{self.counter}",
            "timestamp_ms": self.counter,
            "camera": {},
            "objects": [],
            "robot": {
                "active_arm": "NONE",
                "arm_a": {"retreated": self.arm_a_retreated},
                "arm_b": {"retreated": self.arm_b_retreated},
            },
            "safety": {
                "emergency_stop": False,
                "protective_stop": self.protective_stop,
                "system_fault": None,
            },
            "task": {"status": "pending"},
            "quality": {"confidence": 1.0},
        }


class _Controller:
    def __init__(self) -> None:
        self.ready_calls = []
        self.actions = []
        self.stop_reasons = []
        self.fail_execution = False

    def validate_ready(self, arm_id):
        self.ready_calls.append(arm_id)

    def execute_action(self, action, *, arm_id):
        self.actions.append((arm_id, action))
        if self.fail_execution:
            raise RuntimeError("controller write failed")

    def safe_stop(self, reason):
        self.stop_reasons.append(reason)
        return SafeStopReceipt(True, True, True, True, "stop-1")


class IsaacExecutionEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = _ObservationSource()
        self.controller = _Controller()
        self.environment = IsaacExecutionEnvironment(
            observation_source=self.source,
            controller=self.controller,
        )
        self.action = ActionStep.from_sequence([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])

    def _execute(self, observation, **overrides):
        arguments = {
            "arm_id": "Arm_A",
            "control_token": "A_ONLY",
            "command_id": "command-1",
            "expected_observation_id": observation["observation_id"],
            "expected_state_digest": execution_guard_digest(observation),
        }
        arguments.update(overrides)
        return self.environment.step(self.action, **arguments)

    def test_valid_action_reaches_controller_and_returns_fresh_observation(self):
        observation = self.environment.observe()
        result = self._execute(observation)
        self.assertEqual(self.controller.ready_calls, ["Arm_A"])
        self.assertEqual(self.controller.actions, [("Arm_A", self.action)])
        self.assertNotEqual(result["observation_id"], observation["observation_id"])

    def test_wrong_token_is_rejected_before_controller_write(self):
        observation = self.environment.observe()
        with self.assertRaisesRegex(RuntimeError, "requires token A_ONLY"):
            self._execute(observation, control_token="B_ONLY")
        self.assertEqual(self.controller.actions, [])

    def test_stale_observation_and_digest_are_rejected(self):
        observation = self.environment.observe()
        with self.assertRaisesRegex(RuntimeError, "stale action rejected"):
            self._execute(observation, expected_observation_id="older")
        changed = deepcopy(observation)
        changed["task"]["status"] = "changed"
        with self.assertRaisesRegex(RuntimeError, "state digest changed"):
            self._execute(
                observation,
                expected_state_digest=execution_guard_digest(changed),
            )
        self.assertEqual(self.controller.actions, [])

    def test_duplicate_command_is_rejected_exactly_once(self):
        observation = self.environment.observe()
        next_observation = self._execute(observation)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self._execute(
                next_observation,
                expected_observation_id=next_observation["observation_id"],
                expected_state_digest=execution_guard_digest(next_observation),
            )
        self.assertEqual(len(self.controller.actions), 1)

    def test_safety_and_retreat_interlocks_fail_closed(self):
        self.source.protective_stop = True
        observation = self.environment.observe()
        with self.assertRaisesRegex(RuntimeError, "protective stop"):
            self._execute(observation)
        self.source.protective_stop = False
        self.source.arm_b_retreated = False
        observation = self.environment.observe()
        with self.assertRaisesRegex(RuntimeError, "Arm_B retreated"):
            self._execute(observation)
        self.assertEqual(self.controller.actions, [])

    def test_controller_failure_triggers_safe_stop_and_consumes_command_id(self):
        observation = self.environment.observe()
        self.controller.fail_execution = True
        with self.assertRaisesRegex(RuntimeError, "controller write failed"):
            self._execute(observation)
        self.assertEqual(len(self.controller.stop_reasons), 1)
        with self.assertRaisesRegex(RuntimeError, "duplicate"):
            self._execute(observation)


if __name__ == "__main__":
    unittest.main()
