"""Run a dependency-free smoke of the three-agent architecture.

The Supervisor owns sequencing and safety, YOLO owns perception evidence, and
one π0.5 policy serves either arm through the explicit ``arm_id`` field.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field


@dataclass
class MockPi05:
    calls: list[dict[str, str]] = field(default_factory=list)

    def infer(self, *, arm_id: str, instruction: str, observation_id: str) -> None:
        if arm_id not in {"Arm_A", "Arm_B"}:
            raise ValueError(f"unsupported arm_id: {arm_id}")
        self.calls.append(
            {
                "executor": "pi05",
                "arm_id": arm_id,
                "instruction": instruction,
                "observation_id": observation_id,
            }
        )


@dataclass
class MockYolo:
    calls: list[dict[str, str]] = field(default_factory=list)

    def detect(self, *, camera_id: str, observation_id: str) -> None:
        self.calls.append(
            {
                "detector": "yolo",
                "camera_id": camera_id,
                "observation_id": observation_id,
            }
        )


def run() -> dict[str, object]:
    pi05 = MockPi05()
    yolo = MockYolo()
    events: list[dict[str, str]] = []

    def stage(
        arm_id: str, camera_id: str, instruction: str, observation_id: str
    ) -> None:
        yolo.detect(camera_id=camera_id, observation_id=observation_id)
        pi05.infer(
            arm_id=arm_id,
            instruction=instruction,
            observation_id=observation_id,
        )
        events.append(
            {"agent": "pi05", "arm_id": arm_id, "observation_id": observation_id}
        )

    stage("Arm_A", "CAM_A_TOP", "装箱并将料箱放到交接位", "obs-a-1")
    events.append({"agent": "supervisor", "event": "handoff.ready"})
    stage("Arm_B", "CAM_B_TOP", "将料箱搬到成品区", "obs-b-1")

    return {
        "agents": ["supervisor", "yolo", "pi05"],
        "pi05_calls": pi05.calls,
        "yolo_calls": yolo.calls,
        "events": events,
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
