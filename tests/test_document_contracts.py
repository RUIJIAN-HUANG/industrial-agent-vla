from __future__ import annotations

import json
import unittest
from pathlib import Path

from industrial_agent.contracts import SUPPORTED_TASK_TYPES
from industrial_agent.lifecycle import (
    FROZEN_HANDOFF_EVENT_SEQUENCE,
    HANDOFF_READY_EVENT_TYPE,
)
from scripts.run_mock_demo import (
    ARM_A_INSTRUCTION,
    ARM_B_INSTRUCTION,
    HANDOFF_CANDIDATE_EVENT_TYPE,
    HANDOFF_EVENT_SEQUENCE,
    FrozenPipelineDemo,
)


ROOT = Path(__file__).resolve().parents[1]


class DocumentationContractTests(unittest.TestCase):
    def test_frozen_task_examples_use_the_machine_truth_source(self) -> None:
        config = json.loads(
            (ROOT / "configs" / "agent.v1.legacy.json").read_text(encoding="utf-8")
        )
        profile = config["lifecycle"]["task_profile"]
        interface_contract = (
            ROOT / "docs" / "architecture" / "interface-contracts.md"
        ).read_text(encoding="utf-8")
        frozen_flow = (
            ROOT / "docs" / "architecture" / "final-frozen-scene-and-flow.md"
        ).read_text(encoding="utf-8")

        self.assertEqual(ARM_A_INSTRUCTION, profile["arm_a_instruction"])
        self.assertEqual(ARM_B_INSTRUCTION, profile["arm_b_instruction"])
        self.assertIn(ARM_A_INSTRUCTION, interface_contract)
        self.assertIn(ARM_B_INSTRUCTION, interface_contract)
        self.assertIn(ARM_A_INSTRUCTION, frozen_flow)
        self.assertIn(ARM_B_INSTRUCTION, frozen_flow)
        self.assertIn("visual_manipulation", SUPPORTED_TASK_TYPES)
        self.assertIn('"task_type": "visual_manipulation"', interface_contract)
        self.assertNotIn("帮我把零件最多的区域装箱", interface_contract)

    def test_mock_uses_canonical_handoff_event_order(self) -> None:
        self.assertEqual(
            HANDOFF_EVENT_SEQUENCE,
            FROZEN_HANDOFF_EVENT_SEQUENCE,
        )
        demo = FrozenPipelineDemo("success")
        result = demo.run()
        self.assertTrue(result["success"])

        handoff_events = [
            event
            for event in demo.events
            if event["event_type"] in HANDOFF_EVENT_SEQUENCE
        ]
        self.assertEqual(
            [event["event_type"] for event in handoff_events],
            list(HANDOFF_EVENT_SEQUENCE),
        )
        self.assertEqual(
            [event["token"] for event in handoff_events],
            ["HANDOFF_VERIFY", "HANDOFF_VERIFY"],
        )
        candidate_events = [
            event
            for event in demo.events
            if event["event_type"] == HANDOFF_CANDIDATE_EVENT_TYPE
        ]
        self.assertGreaterEqual(len(candidate_events), 1)
        self.assertFalse(
            candidate_events[-1]["payload"]["contributes_to_quorum"],
        )
        self.assertFalse(candidate_events[-1]["payload"]["grants_b_only"])
        self.assertEqual(handoff_events[0]["payload"]["frame_count"], 3)
        self.assertFalse(handoff_events[0]["payload"]["grants_b_only"])
        self.assertTrue(handoff_events[1]["payload"]["durable_ack"])
        self.assertTrue(handoff_events[1]["payload"]["grants_b_only"])

    def test_architecture_freezes_three_cameras_and_null_wrist_image(self) -> None:
        documents = [
            ROOT / "docs" / "architecture" / "agent-framework.md",
            ROOT / "docs" / "architecture" / "interface-contracts.md",
            ROOT / "docs" / "architecture" / "final-frozen-scene-and-flow.md",
        ]
        combined = "\n".join(path.read_text(encoding="utf-8") for path in documents)

        for event_type in HANDOFF_EVENT_SEQUENCE:
            self.assertIn(event_type, combined)
        self.assertIn(HANDOFF_CANDIDATE_EVENT_TYPE, combined)
        for camera_id in ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP"):
            self.assertIn(camera_id, combined)
        self.assertIn("三台物理", combined)
        self.assertIn("wrist_image", combined)
        self.assertIn("`null`", combined)
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
        self.assertEqual(
            scene["workflow"]["handoff_ready_event"],
            HANDOFF_READY_EVENT_TYPE,
        )

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
