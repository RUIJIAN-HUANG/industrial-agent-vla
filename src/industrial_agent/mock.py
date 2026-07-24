"""Dependency-free mock simulator and executors for contract demonstrations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from time import time_ns
from typing import Literal
from uuid import uuid4

from .contracts import (
    ACTION_CONTRACT_VERSION,
    ActionChunk,
    ActionStep,
    Observation,
    TaskSchema,
)
from .executor import ExecutionContext, ExecutorDescriptor

MockScenario = Literal["success", "recovery", "switch", "system_fault"]


class MockExecutor:
    def __init__(self, name: str, dx_m: float):
        checkpoint_sha = hashlib.sha256(f"{name}:checkpoint".encode()).hexdigest()
        norm_stats_sha = hashlib.sha256(f"{name}:norm-stats".encode()).hexdigest()
        self.descriptor = ExecutorDescriptor(
            name=name,
            task_types=frozenset({"mock_demo"}),
            action_contract_version=ACTION_CONTRACT_VERSION,
            checkpoint_sha=f"sha256:{checkpoint_sha}",
            norm_stats_sha=f"sha256:{norm_stats_sha}",
        )
        self.dx_m = dx_m
        self.plan_calls = 0
        self.cancel_calls: list[tuple[str, str]] = []

    def health(self) -> bool:
        return True

    def plan(
        self, task: TaskSchema, observation: Observation, context: ExecutionContext
    ) -> ActionChunk:
        self.plan_calls += 1
        return ActionChunk(
            contract_version=ACTION_CONTRACT_VERSION,
            chunk_id=f"{self.descriptor.name}-{self.plan_calls}-{uuid4()}",
            task_id=task.task_id,
            executor=self.descriptor.name,
            steps=(
                ActionStep.from_sequence([self.dx_m, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5]),
            ),
        )

    def cancel(self, task_id: str, reason: str) -> None:
        self.cancel_calls.append((task_id, reason))


@dataclass
class MockSimulator:
    """Sensor-visible outcome simulator; no GT is exposed in observations."""

    scenario: MockScenario
    status: str = "pending"
    applied_steps: int = 0
    safe_stop_called: bool = False
    _observation_counter: int = 0

    def _observation(self) -> dict[str, object]:
        self._observation_counter += 1
        fault: str | None = None
        if self.scenario == "system_fault" and self.applied_steps >= 1:
            fault = "MOCK_DRIVE_FAULT"
        return {
            "observation_version": "1.0",
            "observation_id": f"mock-observation-{self._observation_counter}",
            "timestamp_ms": time_ns() // 1_000_000,
            "camera": {
                "frame_id": f"frame-{self._observation_counter}",
                "full_image": "mock://full-image",
                "wrist_image": "mock://wrist-image",
            },
            "objects": [
                {
                    "object_id": "red_block",
                    "confidence": 0.99,
                    "zone_id": "box" if self.status == "done" else "table",
                }
            ],
            "robot": {
                "tcp_pose_m_rad": [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
                "state": [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
            },
            "safety": {
                "emergency_stop": False,
                "protective_stop": False,
                "system_fault": fault,
            },
            "task": {"status": self.status},
            "quality": {"confidence": 0.99},
        }

    def observe(self) -> dict[str, object]:
        return self._observation()

    def step(self, action: ActionStep) -> dict[str, object]:
        self.applied_steps += 1
        dx = action.values[0]
        if self.scenario == "success":
            self.status = "done"
        elif self.scenario == "recovery" and self.applied_steps >= 2:
            self.status = "done"
        elif self.scenario == "switch" and dx >= 0.019:
            self.status = "done"
        return self._observation()

    def safe_stop(self, reason: str) -> None:
        self.safe_stop_called = True
