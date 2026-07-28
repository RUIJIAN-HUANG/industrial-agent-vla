"""Demonstrate the frozen π0.5 -> handoff -> OpenVLA pipeline.

This script is intentionally dependency-free. It validates orchestration order,
token ownership, candidate-precheck versus locked three-frame handoff voting,
canonical handoff event ordering, and same-role recovery. It does not load real
VLA weights, YOLO weights, or Isaac Sim, and it models rather than proves fsync
durability.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

_TASK_PROFILE = json.loads(
    (Path(__file__).resolve().parents[1] / "configs" / "agent.default.json").read_text(
        encoding="utf-8"
    )
)["lifecycle"]["task_profile"]
ARM_A_INSTRUCTION = _TASK_PROFILE["arm_a_instruction"]
ARM_B_INSTRUCTION = _TASK_PROFILE["arm_b_instruction"]
HANDOFF_CANDIDATE_EVENT_TYPE = "handoff.candidate_checked"
HANDOFF_EVENT_SEQUENCE = (
    "handoff.verified",
    "handoff.ready",
)


@dataclass
class MockVLA:
    """A role-locked VLA test double."""

    name: str
    arm_id: str
    required_token: str
    calls: list[dict[str, str]] = field(default_factory=list)

    def infer(
        self,
        *,
        instruction: str,
        observation_id: str,
        token: str,
    ) -> dict[str, object]:
        if token != self.required_token:
            raise RuntimeError(
                f"{self.name} requires {self.required_token}, got {token}"
            )
        call = {
            "executor": self.name,
            "arm_id": self.arm_id,
            "observation_id": observation_id,
            "instruction": instruction,
        }
        self.calls.append(call)
        return {
            "chunk_id": f"{self.name}-chunk-{len(self.calls)}",
            "executor": self.name,
            "arm_id": self.arm_id,
            "steps": 1,
        }


class FrozenPipelineDemo:
    """Small deterministic model of the frozen dual-arm lifecycle."""

    def __init__(self, scenario: str) -> None:
        self.scenario = scenario
        self.pi05 = MockVLA("pi05", "Arm_A", "A_ONLY")
        self.openvla = MockVLA("openvla_oft", "Arm_B", "B_ONLY")
        self.events: list[dict[str, object]] = []
        self.call_order: list[str] = []
        self.token = "NONE"
        self.token_history = ["NONE"]
        self.yolo_detect_calls = 0

    def emit(self, event_type: str, state: str, **payload: object) -> None:
        self.events.append(
            {
                "sequence": len(self.events) + 1,
                "event_type": event_type,
                "state": state,
                "token": self.token,
                "payload": payload,
            }
        )

    def set_token(self, token: str, state: str) -> None:
        self.token = token
        self.token_history.append(token)
        self.emit(f"token.{token.lower()}", state)

    def detect(self, observation_id: str, camera_id: str) -> None:
        self.yolo_detect_calls += 1
        self.emit(
            "perception.archived",
            "PERCEIVING",
            observation_id=observation_id,
            camera_id=camera_id,
            detections_saved=True,
        )

    def infer_pi05(self, observation_id: str) -> None:
        self.pi05.infer(
            instruction=ARM_A_INSTRUCTION,
            observation_id=observation_id,
            token=self.token,
        )
        self.call_order.append("pi05")
        self.emit(
            "vla.inferred",
            "EXECUTING",
            executor="pi05",
            arm_id="Arm_A",
            observation_id=observation_id,
        )

    def infer_openvla(self, observation_id: str) -> None:
        self.openvla.infer(
            instruction=ARM_B_INSTRUCTION,
            observation_id=observation_id,
            token=self.token,
        )
        self.call_order.append("openvla_oft")
        self.emit(
            "vla.inferred",
            "EXECUTING",
            executor="openvla_oft",
            arm_id="Arm_B",
            observation_id=observation_id,
        )

    def verify_handoff(self) -> list[bool]:
        votes: list[bool] = []
        for index in range(1, 4):
            observation_id = f"obs-handoff-{index}"
            self.detect(observation_id, "CAM_HANDOFF")
            vote = True
            votes.append(vote)
            self.emit(
                "handoff.vote",
                "VERIFYING",
                observation_id=observation_id,
                vote="PASS" if vote else "FAIL",
            )
        if sum(votes) < 2:
            raise RuntimeError("handoff verification did not reach two votes")
        return votes

    def run(self) -> dict[str, object]:
        self.emit("task.accepted", "VALIDATING_TASK")
        self.emit("scene.reset", "PLANNING", arm_a="HOME_A", arm_b="HOME_B")

        self.set_token("A_ONLY", "OBSERVING")
        self.detect("obs-a-1", "CAM_A_TOP")
        self.infer_pi05("obs-a-1")

        if self.scenario == "arm_a_recovery":
            self.emit("arm_a.action_failed", "REPLANNING")
            self.detect("obs-a-retry-1", "CAM_A_TOP")
            self.infer_pi05("obs-a-retry-1")

        self.emit("arm_a.pack_complete", "VERIFYING", parts=4)
        self.detect("obs-a-handoff", "CAM_A_TOP")
        self.infer_pi05("obs-a-handoff")
        self.emit(
            "arm_a.handoff_released",
            "ADVANCING_SUBTASK",
            station="HANDOFF_CENTER",
        )
        self.emit("arm_a.retreat_complete", "ADVANCING_SUBTASK", pose="HOME_A")

        candidate_observation_id = "obs-handoff-candidate"
        self.detect(candidate_observation_id, "CAM_HANDOFF")
        self.emit(
            HANDOFF_CANDIDATE_EVENT_TYPE,
            "VERIFYING",
            observation_id=candidate_observation_id,
            verdict="PASS",
            contributes_to_quorum=False,
            grants_b_only=False,
        )
        self.set_token("HANDOFF_VERIFY", "VERIFYING")
        handoff_votes = self.verify_handoff()
        self.emit(
            "handoff.verified",
            "VERIFYING",
            votes=sum(handoff_votes),
            frame_count=len(handoff_votes),
            frames_captured_after_lock=True,
            quorum_passed=True,
            grants_b_only=False,
            persistence="mock_ordering_only",
        )
        self.emit(
            "handoff.ready",
            "VERIFYING",
            verified_event_recorded=True,
            durable_ack=True,
            grants_b_only=True,
            persistence="mock_simulated_durable_ack",
        )

        self.set_token("B_ONLY", "OBSERVING")
        self.detect("obs-b-1", "CAM_B_TOP")
        self.infer_openvla("obs-b-1")

        if self.scenario == "arm_b_recovery":
            self.emit("arm_b.action_failed", "REPLANNING")
            self.detect("obs-b-retry-1", "CAM_B_TOP")
            self.infer_openvla("obs-b-retry-1")

        self.emit(
            "arm_b.transport_complete",
            "VERIFYING",
            station="FINISHED_01",
        )
        self.emit("arm_b.retreat_complete", "ADVANCING_SUBTASK", pose="HOME_B")
        self.set_token("NONE", "SUCCEEDED")
        self.emit("task.succeeded", "SUCCEEDED")

        expected_tokens = [
            "NONE",
            "A_ONLY",
            "HANDOFF_VERIFY",
            "B_ONLY",
            "NONE",
        ]
        first_openvla = self.call_order.index("openvla_oft")
        order_is_valid = all(name == "pi05" for name in self.call_order[:first_openvla])
        handoff_event_types = [
            event["event_type"]
            for event in self.events
            if event["event_type"] in HANDOFF_EVENT_SEQUENCE
        ]
        success = (
            self.token_history == expected_tokens
            and order_is_valid
            and handoff_event_types == list(HANDOFF_EVENT_SEQUENCE)
            and bool(self.pi05.calls)
            and bool(self.openvla.calls)
            and sum(handoff_votes) >= 2
        )
        return {
            "scenario": self.scenario,
            "success": success,
            "final_state": "SUCCEEDED" if success else "FAILED",
            "call_order": self.call_order,
            "pi05_plan_calls": len(self.pi05.calls),
            "openvla_plan_calls": len(self.openvla.calls),
            "handoff_votes": handoff_votes,
            "token_history": self.token_history,
            "yolo_detect_calls": self.yolo_detect_calls,
            "event_count": len(self.events),
        }


def main() -> int:
    scenarios = ("success", "arm_a_recovery", "arm_b_recovery")
    results = [FrozenPipelineDemo(name).run() for name in scenarios]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0 if all(item["success"] for item in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
