from __future__ import annotations

import json
import unittest
from pathlib import Path

from industrial_agent.contracts import ActionStep, Postcondition, TaskSchema
from industrial_agent.errors import FailureCode, ObservationError
from industrial_agent.observation import ObservationGateway


def image_reference(camera_id: str, digest_char: str) -> dict[str, object]:
    digest = digest_char * 64
    return {
        "uri": f"cas://sha256/{digest}",
        "image_sha256": f"sha256:{digest}",
        "camera_id": camera_id,
        "width": 1280,
        "height": 720,
    }


def raw_observation() -> dict[str, object]:
    return {
        "observation_version": "1.0",
        "observation_id": "obs-1",
        "timestamp_ms": 1,
        "camera": {
            "full_image": image_reference("CAM_HANDOFF", "a"),
            "arm_a_rgb": image_reference("CAM_A_TOP", "b"),
            "handoff_rgb": image_reference("CAM_HANDOFF", "c"),
            "arm_b_rgb": image_reference("CAM_B_TOP", "d"),
        },
        "objects": [],
        "robot": {"tcp_pose_m_rad": [0.5, 0.0, 0.5, 0, 0, 0]},
        "safety": {
            "emergency_stop": False,
            "protective_stop": False,
            "system_fault": None,
        },
        "task": {"status": "pending"},
        "quality": {"confidence": 1.0},
    }


class ContractAndObservationTests(unittest.TestCase):
    def test_action_step_requires_exactly_seven_dimensions(self) -> None:
        with self.assertRaisesRegex(Exception, "7-D"):
            ActionStep.from_sequence([0.0] * 6)

    def test_ground_truth_is_rejected_at_any_depth(self) -> None:
        raw = raw_observation()
        raw["camera"] = {"metadata": {"ground_truth": {"pose": [1, 2, 3]}}}
        with self.assertRaises(ObservationError) as caught:
            ObservationGateway().ingest_online(raw)
        self.assertEqual(caught.exception.code, FailureCode.OBSERVATION_GT_FORBIDDEN)

    def test_compound_gt_and_privileged_pose_fields_are_rejected(self) -> None:
        for forbidden_key in ("ground_truth_pose", "sim_gt_mask", "target_pose"):
            raw = raw_observation()
            raw["camera"] = {"metadata": {forbidden_key: [1, 2, 3]}}
            with self.subTest(forbidden_key=forbidden_key):
                with self.assertRaises(ObservationError) as caught:
                    ObservationGateway().ingest_online(raw)
                self.assertEqual(
                    caught.exception.code, FailureCode.OBSERVATION_GT_FORBIDDEN
                )

    def test_required_fields_and_fresh_observation_ids(self) -> None:
        raw = raw_observation()
        gateway = ObservationGateway()
        gateway.ingest_online(raw)
        with self.assertRaises(ObservationError):
            gateway.ingest_online(raw)

        missing_safety = raw_observation()
        del missing_safety["safety"]
        with self.assertRaises(ObservationError) as caught:
            ObservationGateway().ingest_online(missing_safety)
        self.assertEqual(caught.exception.code, FailureCode.OBSERVATION_INVALID)

    def test_frozen_camera_stream_rejects_noncanonical_resolution(self) -> None:
        raw = raw_observation()
        camera = raw["camera"]
        assert isinstance(camera, dict)
        camera["arm_a_rgb"]["width"] = 640
        camera["arm_a_rgb"]["height"] = 480

        with self.assertRaises(ObservationError) as caught:
            ObservationGateway().ingest_online(raw)

        self.assertEqual(caught.exception.code, FailureCode.OBSERVATION_INVALID)
        self.assertIn("1280x720", str(caught.exception))

    def test_online_schema_freezes_all_four_logical_stream_resolutions(self) -> None:
        root = Path(__file__).resolve().parents[1]
        schema = json.loads(
            (root / "schemas" / "online-observation.schema.json").read_text(
                encoding="utf-8"
            )
        )
        camera_properties = schema["properties"]["camera"]["properties"]

        for stream in ("full_image", "arm_a_rgb", "handoff_rgb", "arm_b_rgb"):
            with self.subTest(stream=stream):
                frozen = camera_properties[stream]["allOf"][1]["properties"]
                self.assertEqual(frozen["width"]["const"], 1280)
                self.assertEqual(frozen["height"]["const"], 720)

        self.assertIsNone(camera_properties["wrist_image"]["const"])

    def test_frozen_profile_rejects_non_null_wrist_image(self) -> None:
        raw = raw_observation()
        camera = raw["camera"]
        assert isinstance(camera, dict)
        camera["wrist_image"] = image_reference("CAM_WRIST", "e")

        with self.assertRaises(ObservationError) as caught:
            ObservationGateway().ingest_online(raw)

        self.assertEqual(caught.exception.code, FailureCode.OBSERVATION_INVALID)
        self.assertIn("wrist_image=null", str(caught.exception))

    def test_frozen_image_reference_rejects_unknown_fields(self) -> None:
        raw = raw_observation()
        camera = raw["camera"]
        assert isinstance(camera, dict)
        arm_a_rgb = camera["arm_a_rgb"]
        assert isinstance(arm_a_rgb, dict)
        arm_a_rgb["debug_path"] = "must-not-cross-online-boundary"

        with self.assertRaises(ObservationError) as caught:
            ObservationGateway().ingest_online(raw)

        self.assertEqual(caught.exception.code, FailureCode.OBSERVATION_INVALID)
        self.assertIn("contain exactly", str(caught.exception))

    def test_non_allowlisted_top_level_field_is_rejected(self) -> None:
        raw = raw_observation()
        raw["debug_dump"] = {}
        with self.assertRaises(ObservationError) as caught:
            ObservationGateway().ingest_online(raw)
        self.assertEqual(caught.exception.code, FailureCode.OBSERVATION_INVALID)

    def test_task_rejects_incompatible_major_version(self) -> None:
        task = TaskSchema(
            task_id="t",
            instruction="do task",
            task_type="mock_demo",
            schema_version="2.0",
            postconditions=(
                Postcondition(kind="field_equals", path="task.status", expected="done"),
            ),
        )
        with self.assertRaisesRegex(Exception, "incompatible"):
            task.validate()


if __name__ == "__main__":
    unittest.main()
