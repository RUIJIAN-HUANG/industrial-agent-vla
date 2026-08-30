from __future__ import annotations

import json
import unittest
from pathlib import Path

from industrial_agent.contracts import SUPPORTED_TASK_TYPES
from industrial_agent.lifecycle import (
    FROZEN_HANDOFF_EVENT_SEQUENCE,
)
from scripts.run_mock_demo import run


ROOT = Path(__file__).resolve().parents[1]


class DocumentationContractTests(unittest.TestCase):
    def test_frozen_task_examples_use_the_machine_truth_source(self) -> None:
        profile = json.loads(
            (ROOT / "configs" / "v2-task-profile.json").read_text(encoding="utf-8")
        )
        interface_contract = (
            ROOT / "docs" / "architecture" / "interface-contracts.md"
        ).read_text(encoding="utf-8")
        architecture = (
            ROOT / "docs" / "architecture" / "agent-framework.md"
        ).read_text(encoding="utf-8")
        self.assertEqual(
            profile["formal_task_ids"],
            ["P01_TO_S11", "W01_TO_S14", "BIN01_TO_FINISHED01"],
        )
        self.assertEqual({task["active_arm"] for task in profile["tasks"]}, {"Arm_A"})
        self.assertIn("只有三个 Agent", architecture)
        self.assertIn("唯一 VLA", architecture)
        self.assertIn("arm_id", interface_contract)
        self.assertIn("visual_manipulation", SUPPORTED_TASK_TYPES)
        self.assertIn('task_type: "visual_manipulation"', interface_contract)

    def test_mock_uses_canonical_handoff_event_order(self) -> None:
        result = run()
        self.assertEqual(result["agents"], ["supervisor", "yolo", "pi05"])
        self.assertEqual(
            [event["arm_id"] for event in result["events"] if "arm_id" in event],
            ["Arm_A", "Arm_B"],
        )
        self.assertEqual(
            [call["arm_id"] for call in result["pi05_calls"]],
            ["Arm_A", "Arm_B"],
        )
        self.assertEqual(
            FROZEN_HANDOFF_EVENT_SEQUENCE,
            ("handoff.verified", "handoff.ready"),
        )

    def test_architecture_freezes_three_cameras_and_null_wrist_image(self) -> None:
        documents = [
            ROOT / "docs" / "architecture" / "agent-framework.md",
            ROOT / "docs" / "architecture" / "interface-contracts.md",
            ROOT / "docs" / "architecture" / "final-frozen-scene-and-flow.md",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in documents)

        for event_type in FROZEN_HANDOFF_EVENT_SEQUENCE:
            self.assertIn(event_type, combined)
        for camera_id in ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP"):
            self.assertIn(camera_id, combined)
        self.assertIn("三路 RGB", combined)
        self.assertIn("wrist_image", combined)
        self.assertIn('"wrist_image": null', combined)
        self.assertNotIn('"width": 640', combined)
        self.assertNotIn('"height": 480', combined)

    def test_dashboard_uses_ci_as_the_test_count_truth_source(self) -> None:
        dashboard = (ROOT / "docs" / "project-management" / "dashboard.md").read_text(
            encoding="utf-8"
        )

        self.assertIn("python -m pytest -q", dashboard)
        self.assertIn("以最近一次成功 Run 为准", dashboard)
        self.assertNotIn("66/66", dashboard)

    def test_scene_metadata_uses_the_runtime_handoff_ready_event(self) -> None:
        scene = json.loads(
            (ROOT / "simulation" / "configs" / "single_bin_scene_v1.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(scene["workflow"]["handoff_ready_event"], "handoff.ready")

    def test_v2_layout_asset_matches_the_scene_truth_source(self) -> None:
        scene = json.loads(
            (ROOT / "simulation" / "configs" / "single_bin_scene_v2.json").read_text(
                encoding="utf-8"
            )
        )
        layout = (
            ROOT
            / "docs"
            / "architecture"
            / "assets"
            / "isaac-sim-single-bin-static-handoff-layout-v2.svg"
        ).read_text(encoding="utf-8")

        self.assertIn("双臂八零件", layout)
        self.assertIn("2×4", layout)
        self.assertIn("READY=8/8", layout)
        self.assertNotIn("单箱四零件", layout)
        self.assertNotIn("2×3", layout)
        self.assertNotIn("READY=4/4", layout)
        for slot in scene["bin"]["slots"]:
            self.assertIn(f"{slot['id']}={slot['part_id']}", layout)
        for camera in scene["cameras"]:
            self.assertIn(camera["id"], layout)


if __name__ == "__main__":
    unittest.main()
