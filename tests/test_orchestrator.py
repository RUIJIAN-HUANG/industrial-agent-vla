from __future__ import annotations

import unittest

from industrial_agent.contracts import (
    ACTION_CONTRACT_VERSION,
    ActionChunk,
    ActionStep,
    Observation,
    Postcondition,
    TaskSchema,
)
from industrial_agent.executor import ExecutionContext, ExecutorDescriptor
from industrial_agent.errors import FailureCode
from industrial_agent.fsm import AgentState
from industrial_agent.mock import MockExecutor, MockSimulator
from industrial_agent.orchestrator import IndustrialAgent


def demo_task(task_id: str) -> TaskSchema:
    return TaskSchema(
        task_id=task_id,
        instruction="把红色物体放入料箱",
        task_type="mock_demo",
        preferred_executor="openvla_oft",
        postconditions=(
            Postcondition(
                kind="field_equals",
                path="task.status",
                expected="done",
                required_votes=2,
            ),
        ),
    )


class OrchestratorTests(unittest.TestCase):
    def make_agent(
        self,
    ) -> tuple[IndustrialAgent, MockExecutor, MockExecutor]:
        openvla = MockExecutor("openvla_oft", 0.01)
        pi05 = MockExecutor("pi05", 0.02)
        return (
            IndustrialAgent([openvla, pi05], max_decisions_per_strategy_attempt=1),
            openvla,
            pi05,
        )

    def test_success_path(self) -> None:
        agent, openvla, pi05 = self.make_agent()
        result = agent.run(demo_task("success"), MockSimulator("success"))
        self.assertTrue(result.success)
        self.assertEqual(result.state, AgentState.SUCCEEDED)
        self.assertEqual(openvla.plan_calls, 1)
        self.assertEqual(pi05.plan_calls, 0)

    def test_one_replan_on_same_strategy(self) -> None:
        agent, openvla, pi05 = self.make_agent()
        result = agent.run(demo_task("recovery"), MockSimulator("recovery"))
        self.assertTrue(result.success)
        self.assertEqual(result.replan_counts["openvla_oft"], 1)
        self.assertEqual(result.switch_count, 0)
        self.assertEqual(openvla.plan_calls, 2)
        self.assertEqual(pi05.plan_calls, 0)

    def test_one_switch_and_no_switch_back(self) -> None:
        agent, openvla, pi05 = self.make_agent()
        result = agent.run(demo_task("switch"), MockSimulator("switch"))
        self.assertTrue(result.success)
        self.assertEqual(result.executor_history, ("openvla_oft", "pi05"))
        self.assertEqual(result.switch_count, 1)
        self.assertEqual(openvla.plan_calls, 2)
        self.assertEqual(pi05.plan_calls, 1)

    def test_system_fault_immediately_safe_stops(self) -> None:
        agent, _, _ = self.make_agent()
        simulator = MockSimulator("system_fault")
        result = agent.run(demo_task("fault"), simulator)
        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.SYSTEM_FAULT)
        self.assertTrue(simulator.safe_stop_called)
        self.assertNotIn(AgentState.VERIFYING, [x.current for x in result.transitions])

    def test_gt_leak_after_action_safe_stops_without_recovery(self) -> None:
        class LeakySimulator(MockSimulator):
            def step(self, action: ActionStep) -> dict[str, object]:
                observation = super().step(action)
                observation["camera"] = {"ground_truth_pose": [0, 0, 0]}
                return observation

        agent, openvla, pi05 = self.make_agent()
        simulator = LeakySimulator("success")
        result = agent.run(demo_task("gt-leak"), simulator)
        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.OBSERVATION_GT_FORBIDDEN)
        self.assertEqual(openvla.plan_calls, 1)
        self.assertEqual(pi05.plan_calls, 0)
        self.assertEqual(result.switch_count, 0)
        self.assertTrue(simulator.safe_stop_called)

    def test_gt_leak_in_verification_frame_safe_stops(self) -> None:
        class VerificationLeakySimulator(MockSimulator):
            def __init__(self) -> None:
                super().__init__("success")
                self.action_executed = False

            def step(self, action: ActionStep) -> dict[str, object]:
                observation = super().step(action)
                self.action_executed = True
                return observation

            def observe(self) -> dict[str, object]:
                observation = super().observe()
                if self.action_executed:
                    observation["task"] = {
                        "status": "done",
                        "groundTruthPose": [0, 0, 0],
                    }
                return observation

        agent, openvla, pi05 = self.make_agent()
        simulator = VerificationLeakySimulator()
        result = agent.run(demo_task("verification-gt-leak"), simulator)
        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.OBSERVATION_GT_FORBIDDEN)
        self.assertEqual(openvla.plan_calls, 1)
        self.assertEqual(pi05.plan_calls, 0)
        self.assertEqual(result.switch_count, 0)
        self.assertTrue(simulator.safe_stop_called)

    def test_multi_step_chunk_reinfers_before_second_action(self) -> None:
        class RecedingHorizonExecutor:
            def __init__(self) -> None:
                self.descriptor = ExecutorDescriptor(
                    name="openvla_oft",
                    task_types=frozenset({"mock_demo"}),
                    action_contract_version=ACTION_CONTRACT_VERSION,
                    checkpoint_sha=f"sha256:{'c' * 64}",
                    norm_stats_sha=f"sha256:{'d' * 64}",
                )
                self.observation_ids: list[str] = []

            def health(self) -> bool:
                return True

            def plan(
                self,
                task: TaskSchema,
                observation: Observation,
                context: ExecutionContext,
            ) -> ActionChunk:
                self.observation_ids.append(observation.observation_id)
                steps = [ActionStep.from_sequence([0.01, 0, 0, 0, 0, 0, 0])]
                if len(self.observation_ids) == 1:
                    # This second action is stale after the first changes phase.
                    steps.append(ActionStep.from_sequence([-0.01, 0, 0, 0, 0, 0, 0]))
                return ActionChunk(
                    contract_version=ACTION_CONTRACT_VERSION,
                    chunk_id=f"chunk-{len(self.observation_ids)}",
                    task_id=task.task_id,
                    executor=self.descriptor.name,
                    steps=tuple(steps),
                )

            def cancel(self, task_id: str, reason: str) -> None:
                return

        class PhaseEnvironment:
            def __init__(self) -> None:
                self.phase = 0
                self.counter = 0
                self.applied_dx: list[float] = []

            def observation(self) -> dict[str, object]:
                self.counter += 1
                return {
                    "observation_version": "1.0",
                    "observation_id": f"rh-{self.counter}",
                    "timestamp_ms": self.counter,
                    "camera": {},
                    "objects": [],
                    "robot": {
                        "tcp_pose_m_rad": [
                            0.5 + sum(self.applied_dx),
                            0,
                            0.5,
                            0,
                            0,
                            0,
                        ]
                    },
                    "safety": {
                        "emergency_stop": False,
                        "protective_stop": False,
                        "system_fault": None,
                    },
                    "task": {"status": "done" if self.phase == 2 else "pending"},
                    "quality": {"confidence": 1.0},
                }

            def observe(self) -> dict[str, object]:
                return self.observation()

            def step(self, action: ActionStep) -> dict[str, object]:
                dx = action.values[0]
                if self.phase == 1 and dx < 0:
                    raise AssertionError("stale second action was executed")
                self.applied_dx.append(dx)
                self.phase += 1
                return self.observation()

            def safe_stop(self, reason: str) -> None:
                raise AssertionError(f"unexpected safe stop: {reason}")

        executor = RecedingHorizonExecutor()
        environment = PhaseEnvironment()
        agent = IndustrialAgent([executor], max_decisions_per_strategy_attempt=4)
        result = agent.run(demo_task("receding-horizon"), environment)
        self.assertTrue(result.success)
        self.assertEqual(environment.applied_dx, [0.01, 0.01])
        self.assertEqual(len(executor.observation_ids), 2)
        self.assertNotEqual(executor.observation_ids[0], executor.observation_ids[1])
        accepted = [
            event
            for event in result.events
            if event.event_type == "action_chunk.accepted"
        ]
        self.assertEqual(accepted[0].payload["discarded_steps"], 1)


if __name__ == "__main__":
    unittest.main()
