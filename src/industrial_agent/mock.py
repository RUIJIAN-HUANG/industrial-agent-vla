"""Dependency-free mock simulator and executors for contract demonstrations."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from uuid import uuid4

from .contracts import (
    ACTION_CONTRACT_VERSION,
    ActionChunk,
    ActionStep,
    Observation,
    TaskSchema,
)
from .executor import ExecutionContext, ExecutorDescriptor
from .environment import SafeStopReceipt, execution_guard_digest


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
class FixedDualArmMockSimulator:
    """Deterministic two-arm environment for the frozen lifecycle.

    Ownership is bound only by the explicit ``arm_id`` and ``control_token``
    arguments at the environment adapter boundary.
    """

    arm_a_success_after: int = 1
    arm_b_success_after: int = 1
    arm_a_retreated_on_handoff: bool = True
    arm_a_gripper_open_on_handoff: bool = True
    arm_b_gripper_open_on_finish: bool = True
    bin_speed_on_handoff_m_s: float = 0.0
    bin_speed_on_finish_m_s: float = 0.0
    packed_part_count: int = 0
    bin_at_handoff: bool = False
    bin_speed_m_s: float = 0.0
    arm_a_retreated: bool = False
    arm_b_retreated: bool = True
    arm_a_gripper_open: bool = True
    arm_b_gripper_open: bool = True
    arm_a_stationary: bool = True
    arm_b_stationary: bool = True
    bin_at_finished: bool = False
    safe_stop_called: bool = False
    arm_a_steps: int = 0
    arm_b_steps: int = 0
    illegal_arm_b_attempts: int = 0
    step_owners: list[str] = field(default_factory=list)
    step_authorizations: list[tuple[str, str]] = field(default_factory=list)
    executed_command_ids: set[str] = field(default_factory=set)
    _observation_counter: int = 0
    _last_observation_id: str = ""

    def __post_init__(self) -> None:
        if self.arm_a_success_after < 1 or self.arm_b_success_after < 1:
            raise ValueError("success thresholds must be >= 1")

    def _critical_observation_data(self) -> dict[str, object]:
        if self.safe_stop_called:
            active_arm = "NONE"
        elif not self.arm_a_retreated and not self.bin_at_handoff:
            active_arm = "Arm_A"
        elif not self.arm_b_retreated:
            active_arm = "Arm_B"
        else:
            active_arm = "NONE"
        arm_a_state = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
        arm_b_state = [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
        return {
            "objects": [
                {
                    "object_id": "Bin_01",
                    "confidence": 0.99,
                    "zone_id": (
                        "FINISHED_01"
                        if self.bin_at_finished
                        else "HANDOFF_CENTER"
                        if self.bin_at_handoff
                        else "PACK_STATION"
                    ),
                }
            ],
            "robot": {
                "active_arm": active_arm,
                "arm_a": {
                    "tcp_pose_m_rad": arm_a_state[:6],
                    "state": arm_a_state,
                    "retreated": self.arm_a_retreated,
                    "gripper_open": self.arm_a_gripper_open,
                    "stationary": self.arm_a_stationary,
                },
                "arm_b": {
                    "tcp_pose_m_rad": arm_b_state[:6],
                    "state": arm_b_state,
                    "retreated": self.arm_b_retreated,
                    "gripper_open": self.arm_b_gripper_open,
                    "stationary": self.arm_b_stationary,
                },
            },
            "safety": {
                "emergency_stop": False,
                "protective_stop": False,
                "system_fault": None,
            },
            "task": {
                "packed_part_count": self.packed_part_count,
                "bin_at_handoff": self.bin_at_handoff,
                "arm_a_retreated": self.arm_a_retreated,
                "arm_b_retreated": self.arm_b_retreated,
                "bin_at_finished": self.bin_at_finished,
                "bin_speed_m_s": self.bin_speed_m_s,
                "status": "done" if self.bin_at_finished else "pending",
            },
            "quality": {"confidence": 0.99},
        }

    def _observation(self) -> dict[str, object]:
        self._observation_counter += 1
        frame_id = f"dual-frame-{self._observation_counter}"
        observation_id = f"dual-observation-{self._observation_counter}"
        self._last_observation_id = observation_id
        frame_sha = hashlib.sha256(frame_id.encode()).hexdigest()
        arm_a_sha = hashlib.sha256(f"{frame_id}:arm_a".encode()).hexdigest()
        arm_b_sha = hashlib.sha256(f"{frame_id}:arm_b".encode()).hexdigest()
        if self.safe_stop_called:
            active_arm = "NONE"
        elif not self.arm_a_retreated and not self.bin_at_handoff:
            active_arm = "Arm_A"
        elif not self.arm_b_retreated:
            active_arm = "Arm_B"
        else:
            active_arm = "NONE"
        arm_a_state = [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
        arm_b_state = [0.4, 0.0, 0.5, 0.0, 0.0, 0.0, 0.5]
        return {
            "observation_version": "1.0",
            "observation_id": observation_id,
            "timestamp_ms": self._observation_counter,
            "camera": {
                "full_image": {
                    "uri": f"cas://sha256/{frame_sha}",
                    "image_sha256": f"sha256:{frame_sha}",
                    "camera_id": "CAM_HANDOFF",
                    "width": 640,
                    "height": 480,
                },
                "arm_a_rgb": {
                    "uri": f"cas://sha256/{arm_a_sha}",
                    "image_sha256": f"sha256:{arm_a_sha}",
                    "camera_id": "CAM_A_TOP",
                    "width": 640,
                    "height": 480,
                },
                "handoff_rgb": {
                    "uri": f"cas://sha256/{frame_sha}",
                    "image_sha256": f"sha256:{frame_sha}",
                    "camera_id": "CAM_HANDOFF",
                    "width": 640,
                    "height": 480,
                },
                "arm_b_rgb": {
                    "uri": f"cas://sha256/{arm_b_sha}",
                    "image_sha256": f"sha256:{arm_b_sha}",
                    "camera_id": "CAM_B_TOP",
                    "width": 640,
                    "height": 480,
                },
            },
            "objects": [
                {
                    "object_id": "Bin_01",
                    "confidence": 0.99,
                    "zone_id": (
                        "FINISHED_01"
                        if self.bin_at_finished
                        else "HANDOFF_CENTER"
                        if self.bin_at_handoff
                        else "PACK_STATION"
                    ),
                }
            ],
            "robot": {
                "active_arm": active_arm,
                "arm_a": {
                    "tcp_pose_m_rad": arm_a_state[:6],
                    "state": arm_a_state,
                    "retreated": self.arm_a_retreated,
                    "gripper_open": self.arm_a_gripper_open,
                    "stationary": self.arm_a_stationary,
                },
                "arm_b": {
                    "tcp_pose_m_rad": arm_b_state[:6],
                    "state": arm_b_state,
                    "retreated": self.arm_b_retreated,
                    "gripper_open": self.arm_b_gripper_open,
                    "stationary": self.arm_b_stationary,
                },
            },
            "safety": {
                "emergency_stop": False,
                "protective_stop": False,
                "system_fault": None,
            },
            "task": {
                "packed_part_count": self.packed_part_count,
                "bin_at_handoff": self.bin_at_handoff,
                "arm_a_retreated": self.arm_a_retreated,
                "arm_b_retreated": self.arm_b_retreated,
                "bin_at_finished": self.bin_at_finished,
                "bin_speed_m_s": self.bin_speed_m_s,
                "status": "done" if self.bin_at_finished else "pending",
            },
            "quality": {"confidence": 0.99},
        }

    def observe(self) -> dict[str, object]:
        return self._observation()

    def step(
        self,
        action: ActionStep,
        *,
        arm_id: str,
        control_token: str,
        command_id: str,
        expected_observation_id: str,
        expected_state_digest: str,
    ) -> dict[str, object]:
        expected = {
            "Arm_A": ("A_ONLY", "pi05"),
            "Arm_B": ("B_ONLY", "openvla_oft"),
        }
        if arm_id not in expected:
            raise RuntimeError(f"fixed dual-arm adapter rejected arm_id={arm_id!r}")
        expected_token, owner = expected[arm_id]
        if control_token != expected_token:
            raise RuntimeError(
                f"{arm_id} requires token {expected_token}, got {control_token!r}"
            )
        if not command_id or command_id in self.executed_command_ids:
            raise RuntimeError(
                f"duplicate or empty controller command_id rejected: {command_id!r}"
            )
        if expected_observation_id != self._last_observation_id:
            raise RuntimeError(
                "stale action rejected: "
                f"expected observation {self._last_observation_id!r}, "
                f"got {expected_observation_id!r}"
            )
        actual_state_digest = execution_guard_digest(self._critical_observation_data())
        if expected_state_digest != actual_state_digest:
            raise RuntimeError("stale action rejected: execution state digest changed")
        if arm_id == "Arm_A" and not self.arm_b_retreated:
            raise RuntimeError("Arm_A boundary interlock requires Arm_B retreated")
        if arm_id == "Arm_B" and not self.arm_a_retreated:
            raise RuntimeError("Arm_B boundary interlock requires Arm_A retreated")
        self.executed_command_ids.add(command_id)
        self.step_authorizations.append((arm_id, control_token))
        self.step_owners.append(owner)
        if owner == "pi05":
            if self.bin_at_handoff and self.arm_a_retreated:
                raise RuntimeError("Arm A acted after handoff ownership changed")
            self.arm_a_steps += 1
            self.arm_a_stationary = False
            self.arm_a_gripper_open = False
            if self.arm_a_steps >= self.arm_a_success_after:
                self.packed_part_count = 4
                self.bin_at_handoff = True
                self.bin_at_finished = False
                self.bin_speed_m_s = self.bin_speed_on_handoff_m_s
                self.arm_a_retreated = self.arm_a_retreated_on_handoff
                self.arm_a_gripper_open = self.arm_a_gripper_open_on_handoff
                self.arm_a_stationary = True
        else:
            if not self.bin_at_handoff or not self.arm_a_retreated:
                self.illegal_arm_b_attempts += 1
                raise RuntimeError(
                    "Arm B acted before stable handoff and Arm A retreat"
                )
            self.arm_b_steps += 1
            self.arm_b_stationary = False
            self.arm_b_retreated = False
            self.arm_b_gripper_open = False
            if self.arm_b_steps >= self.arm_b_success_after:
                self.bin_at_handoff = False
                self.bin_at_finished = True
                self.bin_speed_m_s = self.bin_speed_on_finish_m_s
                self.arm_b_retreated = True
                self.arm_b_gripper_open = self.arm_b_gripper_open_on_finish
                self.arm_b_stationary = True
        if owner == "pi05":
            self.arm_a_stationary = True
        else:
            self.arm_b_stationary = True
        return self._observation()

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        self.safe_stop_called = True
        self.arm_a_stationary = True
        self.arm_b_stationary = True
        return SafeStopReceipt(
            controller_ack=True,
            buffers_cleared=True,
            arm_a_stopped=True,
            arm_b_stopped=True,
            stop_epoch=f"mock-stop-{self._observation_counter + 1}",
        )
