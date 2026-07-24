from __future__ import annotations

import unittest
from dataclasses import replace
from time import time_ns

from industrial_agent.contracts import (
    ActionStep,
    Postcondition,
    Subtask,
    TaskPlan,
    TaskSchema,
)
from industrial_agent.mock import MockExecutor
from industrial_agent.orchestrator import IndustrialAgent
from industrial_agent.planner import SemanticTaskPlanner


class TwoSubtaskPlanner:
    def plan(self, task: TaskSchema, episode_id: str) -> TaskPlan:
        return TaskPlan(
            plan_id="two-subtask-plan",
            episode_id=episode_id,
            task_id=task.task_id,
            subtasks=[
                Subtask(
                    subtask_id="S01",
                    sequence=1,
                    instruction="完成第一阶段语义动作",
                    task_type="mock_demo",
                    preconditions=(),
                    postconditions=(
                        Postcondition(
                            kind="field_equals",
                            path="task.phase",
                            expected=1,
                            required_votes=2,
                        ),
                    ),
                ),
                Subtask(
                    subtask_id="S02",
                    sequence=2,
                    instruction="完成第二阶段语义动作",
                    task_type="mock_demo",
                    preconditions=(
                        Postcondition(
                            kind="field_equals",
                            path="task.phase",
                            expected=1,
                        ),
                    ),
                    postconditions=(
                        Postcondition(
                            kind="field_equals",
                            path="task.phase",
                            expected=2,
                            required_votes=2,
                        ),
                    ),
                    depends_on=("S01",),
                ),
            ],
        )


class PhaseEnvironment:
    def __init__(self) -> None:
        self.phase = 0
        self.step_calls = 0
        self.observation_counter = 0

    def _observation(self) -> dict[str, object]:
        self.observation_counter += 1
        return {
            "observation_version": "1.0",
            "observation_id": f"phase-{self.observation_counter}",
            "timestamp_ms": time_ns() // 1_000_000,
            "camera": {},
            "objects": [],
            "robot": {"tcp_pose_m_rad": [0.5, 0.0, 0.5, 0, 0, 0]},
            "safety": {
                "emergency_stop": False,
                "protective_stop": False,
                "system_fault": None,
            },
            "task": {"phase": self.phase},
            "quality": {"confidence": 1.0},
        }

    def observe(self) -> dict[str, object]:
        return self._observation()

    def step(self, action: object) -> dict[str, object]:
        self.step_calls += 1
        if self.step_calls == 1:
            self.phase = 1
        elif self.step_calls >= 3:
            self.phase = 2
        return self._observation()

    def safe_stop(self, reason: str) -> None:
        raise AssertionError(f"unexpected safe stop: {reason}")


class SwitchAcrossSubtasksEnvironment:
    def __init__(self) -> None:
        self.phase = 0
        self.observation_counter = 0

    def _observation(self) -> dict[str, object]:
        self.observation_counter += 1
        return {
            "observation_version": "1.0",
            "observation_id": f"switch-phase-{self.observation_counter}",
            "timestamp_ms": self.observation_counter,
            "camera": {},
            "objects": [],
            "robot": {"tcp_pose_m_rad": [0.5, 0.0, 0.5, 0, 0, 0]},
            "safety": {
                "emergency_stop": False,
                "protective_stop": False,
                "system_fault": None,
            },
            "task": {"phase": self.phase},
            "quality": {"confidence": 1.0},
        }

    def observe(self) -> dict[str, object]:
        return self._observation()

    def step(self, action: ActionStep) -> dict[str, object]:
        dx = action.values[0]
        if dx >= 0.019:
            self.phase = min(2, self.phase + 1)
        return self._observation()

    def safe_stop(self, reason: str) -> None:
        raise AssertionError(f"unexpected safe stop: {reason}")


class AlreadyFullEnvironment:
    def __init__(self) -> None:
        self.observation_counter = 0

    def observe(self) -> dict[str, object]:
        self.observation_counter += 1
        return {
            "observation_version": "1.0",
            "observation_id": f"full-{self.observation_counter}",
            "timestamp_ms": self.observation_counter,
            "camera": {},
            "objects": [],
            "robot": {"tcp_pose_m_rad": [0.5, 0.0, 0.5, 0, 0, 0]},
            "safety": {
                "emergency_stop": False,
                "protective_stop": False,
                "system_fault": None,
            },
            "task": {"container_full": True},
            "quality": {"confidence": 1.0},
        }

    def step(self, action: ActionStep) -> dict[str, object]:
        raise AssertionError("already-full workflow must not execute an action")

    def safe_stop(self, reason: str) -> None:
        raise AssertionError(f"unexpected safe stop: {reason}")


class TaskPlanTests(unittest.TestCase):
    def test_plan_rejects_self_dependency(self) -> None:
        plan = TaskPlan(
            plan_id="invalid",
            episode_id="episode",
            task_id="task",
            subtasks=[
                Subtask(
                    subtask_id="S01",
                    sequence=1,
                    instruction="执行第一步",
                    task_type="mock_demo",
                    preconditions=(),
                    postconditions=(
                        Postcondition(
                            kind="field_equals",
                            path="task.done",
                            expected=True,
                        ),
                    ),
                    depends_on=("S01",),
                )
            ],
        )
        with self.assertRaisesRegex(Exception, "non-prior dependencies"):
            plan.validate()

    def test_failure_retries_only_current_subtask(self) -> None:
        executor = MockExecutor("openvla_oft", 0.01)
        agent = IndustrialAgent(
            [executor],
            planner=TwoSubtaskPlanner(),  # type: ignore[arg-type]
            max_decisions_per_strategy_attempt=1,
        )
        task = TaskSchema(
            task_id="two-step",
            instruction="依次完成两阶段任务",
            task_type="mock_demo",
            preferred_executor="openvla_oft",
            postconditions=(
                Postcondition(
                    kind="field_equals",
                    path="task.phase",
                    expected=2,
                    required_votes=2,
                ),
            ),
        )
        result = agent.run(task, PhaseEnvironment())
        self.assertTrue(result.success)
        self.assertEqual(executor.plan_calls, 3)
        self.assertEqual(result.replan_counts["openvla_oft"], 1)
        verified = [
            event.payload["subtask_id"]
            for event in result.events
            if event.event_type == "subtask.verified"
        ]
        self.assertEqual(verified, ["S01", "S02"])
        accepted = [
            event.payload["subtask_id"]
            for event in result.events
            if event.event_type == "action_chunk.accepted"
        ]
        self.assertEqual(accepted, ["S01", "S02", "S02"])

    def test_switched_executor_is_excluded_for_later_subtasks(self) -> None:
        openvla = MockExecutor("openvla_oft", 0.01)
        pi05 = MockExecutor("pi05", 0.02)
        agent = IndustrialAgent(
            [openvla, pi05],
            planner=TwoSubtaskPlanner(),  # type: ignore[arg-type]
            max_decisions_per_strategy_attempt=1,
        )
        task = TaskSchema(
            task_id="no-switch-back",
            instruction="依次完成两阶段任务",
            task_type="mock_demo",
            preferred_executor="openvla_oft",
            postconditions=(
                Postcondition(
                    kind="field_equals",
                    path="task.phase",
                    expected=2,
                    required_votes=2,
                ),
            ),
        )
        result = agent.run(task, SwitchAcrossSubtasksEnvironment())
        self.assertTrue(result.success)
        self.assertEqual(result.switch_count, 1)
        self.assertEqual(result.executor_history, ("openvla_oft", "pi05", "pi05"))
        self.assertEqual(openvla.plan_calls, 2)
        self.assertEqual(pi05.plan_calls, 2)

    def test_builtin_plans_are_semantic_and_bounded(self) -> None:
        task = TaskSchema(
            task_id="pack",
            instruction="装箱直到装满",
            task_type="pick_place",
            constraints={
                "workflow": "pack_until_full",
                "container_id": "bin-A",
                "max_pack_iterations": 8,
            },
            postconditions=(
                Postcondition(
                    kind="field_equals",
                    path="task.container_full",
                    expected=True,
                ),
            ),
        )
        plan = SemanticTaskPlanner().plan(task, "episode-1")
        self.assertEqual(len(plan.subtasks), 1)
        self.assertTrue(plan.subtasks[0].repeat_until_postcondition)
        self.assertEqual(plan.subtasks[0].max_iterations, 8)
        serialized = plan.to_dict()
        self.assertNotIn("coordinates", str(serialized).lower())
        self.assertNotIn("trajectory", str(serialized).lower())
        self.assertNotIn("grasp_point", str(serialized).lower())

    def test_repeat_workflow_does_not_act_when_already_complete(self) -> None:
        executor = MockExecutor("openvla_oft", 0.01)
        task = TaskSchema(
            task_id="already-full",
            instruction="装箱直到装满",
            task_type="mock_demo",
            constraints={
                "workflow": "pack_until_full",
                "container_id": "bin-A",
                "max_pack_iterations": 8,
            },
            postconditions=(
                Postcondition(
                    kind="field_equals",
                    path="task.container_full",
                    expected=True,
                ),
            ),
        )
        # The built-in pack workflow routes as pick_place.
        executor.descriptor = replace(
            executor.descriptor,
            task_types=frozenset({"mock_demo", "pick_place"}),
        )
        result = IndustrialAgent([executor]).run(task, AlreadyFullEnvironment())
        self.assertTrue(result.success)
        self.assertEqual(executor.plan_calls, 0)


if __name__ == "__main__":
    unittest.main()
