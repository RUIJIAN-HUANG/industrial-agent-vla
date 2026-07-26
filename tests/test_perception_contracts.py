from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

from industrial_agent.errors import ContractError, FailureCode
from industrial_agent.perception import (
    DETECTION_CONTRACT_VERSION,
    Detection,
    DetectionPacket,
    ImageReference,
    MockPerceptionAgent,
    PerceptionAgent,
    PerceptionContext,
    PerceptionError,
    PerceptionTiming,
    YoloHTTPAdapter,
    build_perception_from_config,
)

CHECKPOINT_SHA = f"sha256:{'1' * 64}"
CLASS_MAP_SHA = f"sha256:{'2' * 64}"
CONFIG_SHA = f"sha256:{'3' * 64}"
IMAGE_SHA = f"sha256:{'4' * 64}"
OTHER_IMAGE_SHA = f"sha256:{'5' * 64}"


def make_image(image_sha256: str = IMAGE_SHA) -> ImageReference:
    digest = image_sha256.removeprefix("sha256:")
    return ImageReference(
        uri=f"cas://sha256/{digest}",
        image_sha256=image_sha256,
        camera_id="overhead",
        width=640,
        height=480,
    )


def make_context(
    *,
    image_sha256: str = IMAGE_SHA,
    observation_id: str = "obs-1",
    allowed_class_names: tuple[str, ...] = (),
) -> PerceptionContext:
    return PerceptionContext(
        run_id="run-1",
        task_id="task-1:S01",
        subtask_id="S01",
        step_id=4,
        observation_id=observation_id,
        image=make_image(image_sha256),
        allowed_class_names=allowed_class_names,
    )


def make_detection(**overrides: Any) -> Detection:
    values: dict[str, Any] = {
        "detection_id": "det-1",
        "class_id": 3,
        "class_name": "red_part",
        "confidence": 0.91,
        "bbox_xyxy": (10, 20, 110, 220),
        "camera_id": "overhead",
        "image_width": 640,
        "image_height": 480,
        "track_id": "track-7",
        "zone_id": "input_bin",
        "attributes": {"orientation_state": "upright", "occluded": False},
    }
    values.update(overrides)
    return Detection(**values)


def make_packet(
    *,
    request_id: str = "req-1",
    observation_id: str = "obs-1",
    image_sha256: str = IMAGE_SHA,
) -> DetectionPacket:
    return DetectionPacket(
        packet_id="packet-1",
        request_id=request_id,
        trace_id="run-1",
        episode_id="run-1",
        task_id="task-1:S01",
        subtask_id="S01",
        step_id=4,
        observation_id=observation_id,
        image_sha256=image_sha256,
        camera_id="overhead",
        image_width=640,
        image_height=480,
        checkpoint_sha=CHECKPOINT_SHA,
        class_map_sha=CLASS_MAP_SHA,
        config_sha=CONFIG_SHA,
        detections=(make_detection(),),
        timing=PerceptionTiming(1.0, 8.0, 0.5, 10.0),
    )


class EchoYoloTransport:
    def __init__(
        self,
        *,
        response_overrides: Mapping[str, Any] | None = None,
        packet_overrides: Mapping[str, Any] | None = None,
        health_overrides: Mapping[str, Any] | None = None,
        timeout: bool = False,
    ):
        self.calls: list[tuple[str, Mapping[str, Any], int]] = []
        self.response_overrides = dict(response_overrides or {})
        self.packet_overrides = dict(packet_overrides or {})
        self.health_overrides = dict(health_overrides or {})
        self.timeout = timeout

    def request(
        self, route: str, payload: Mapping[str, Any], timeout_ms: int
    ) -> Mapping[str, Any]:
        self.calls.append((route, payload, timeout_ms))
        if self.timeout and route == "/v1/detect":
            raise TimeoutError("mock deadline")
        if route == "/health":
            response = {
                "schema_version": "1.0",
                "service": "yolo",
                "status": "ready",
                "checkpoint_sha": CHECKPOINT_SHA,
                "class_map_sha": CLASS_MAP_SHA,
                "config_sha": CONFIG_SHA,
                "supported_task_types": [
                    "pick_place",
                    "object_localization",
                    "visual_manipulation",
                    "instruction_interaction",
                    "mock_demo",
                ],
                "supported_detection_contracts": [DETECTION_CONTRACT_VERSION],
            }
            response.update(self.health_overrides)
            return response
        if route == "/v1/cancel":
            return {"status": "cancelled"}
        packet = {
            "schema_version": "1.0",
            "detection_contract_version": DETECTION_CONTRACT_VERSION,
            "packet_id": "packet-from-yolo",
            "request_id": payload["request_id"],
            "trace_id": payload["trace_id"],
            "episode_id": payload["episode_id"],
            "task_id": payload["task_id"],
            "subtask_id": payload["subtask_id"],
            "step_id": payload["step_id"],
            "observation_id": payload["observation_id"],
            "image_sha256": payload["image_sha256"],
            "camera_id": payload["image"]["camera_id"],
            "image_width": payload["image"]["width"],
            "image_height": payload["image"]["height"],
            "checkpoint_sha": payload["checkpoint_sha"],
            "class_map_sha": payload["class_map_sha"],
            "config_sha": payload["config_sha"],
            "detections": [make_detection().to_dict()],
            "timing": {
                "preprocess_ms": 1.0,
                "inference_ms": 8.0,
                "nms_ms": 0.5,
                "total_ms": 10.0,
            },
        }
        packet.update(self.packet_overrides)
        response = {
            key: payload[key]
            for key in (
                "schema_version",
                "request_id",
                "trace_id",
                "episode_id",
                "task_id",
                "subtask_id",
                "step_id",
                "observation_id",
                "image_sha256",
                "detector",
                "checkpoint_sha",
                "class_map_sha",
                "config_sha",
            )
        }
        response.update(
            {
                "status": "ok",
                "detection_packet": packet,
            }
        )
        response.update(self.response_overrides)
        return response


class PerceptionContractTests(unittest.TestCase):
    def test_detection_and_packet_round_trip(self) -> None:
        packet = make_packet()
        restored = DetectionPacket.from_dict(packet.to_dict())
        self.assertEqual(restored, packet)
        self.assertEqual(restored.detections[0].bbox_format, "xyxy_pixels")
        self.assertEqual(
            restored.detections[0].attributes["orientation_state"], "upright"
        )

    def test_bbox_is_finite_ordered_and_inside_image(self) -> None:
        invalid_boxes = (
            (-1, 20, 110, 220),
            (10, 20, 10, 220),
            (10, 220, 110, 20),
            (10, 20, 641, 220),
            (10, 20, 110, 481),
            (10, 20, float("nan"), 220),
            (True, 20, 110, 220),
        )
        for bbox in invalid_boxes:
            with self.subTest(bbox=bbox):
                with self.assertRaises(ContractError):
                    make_detection(bbox_xyxy=bbox)

    def test_artifact_and_image_identity_requires_full_sha256(self) -> None:
        for value in ("latest", "sha256:abc", "1" * 64):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    make_image(value)
                with self.assertRaises(ValueError):
                    YoloHTTPAdapter(
                        EchoYoloTransport(),
                        checkpoint_sha=value,
                        class_map_sha=CLASS_MAP_SHA,
                        config_sha=CONFIG_SHA,
                    )

    def test_packet_rejects_mixed_camera_or_dimensions(self) -> None:
        for overrides in (
            {"camera_id": "wrist"},
            {"image_width": 320},
            {"image_height": 240},
        ):
            with self.subTest(overrides=overrides):
                with self.assertRaises(ContractError):
                    DetectionPacket(
                        **{
                            **make_packet().__dict__,
                            "detections": (make_detection(**overrides),),
                        }
                    )

    def test_packet_detects_stale_or_wrong_image(self) -> None:
        adapter = YoloHTTPAdapter(
            EchoYoloTransport(packet_overrides={"image_sha256": OTHER_IMAGE_SHA}),
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
        )
        with self.assertRaises(PerceptionError) as caught:
            adapter.detect(make_context())
        self.assertEqual(caught.exception.code, FailureCode.PERCEPTION_BAD_RESPONSE)

    def test_adapter_distinguishes_deployment_revision_mismatch(self) -> None:
        adapter = YoloHTTPAdapter(
            EchoYoloTransport(response_overrides={"checkpoint_sha": OTHER_IMAGE_SHA}),
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
        )
        with self.assertRaises(PerceptionError) as caught:
            adapter.detect(make_context())
        self.assertEqual(
            caught.exception.code,
            FailureCode.PERCEPTION_REVISION_MISMATCH,
        )

    def test_unknown_or_privileged_online_fields_are_rejected(self) -> None:
        raw = make_detection().to_dict()
        raw["offline_annotation"] = {}
        with self.assertRaises(ContractError):
            Detection.from_dict(raw)
        with self.assertRaises(ContractError):
            make_detection(attributes={"ground_truth": {"box": [1, 2, 3, 4]}})
        with self.assertRaises(ContractError):
            make_detection(attributes={"target_pose": [1, 2, 3]})
        for forbidden in (
            "groundTruth",
            "annotation",
            "labels",
            "oraclePose",
            "ground-truth",
        ):
            with self.subTest(forbidden=forbidden):
                with self.assertRaises(ContractError):
                    make_detection(attributes={forbidden: [1, 2, 3, 4]})

    def test_http_adapter_emits_strict_request_and_returns_packet(self) -> None:
        transport = EchoYoloTransport()
        adapter = YoloHTTPAdapter(
            transport,
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
        )
        context = make_context(
            allowed_class_names=("red_part", "blue_part"),
        )
        packet = adapter.detect(context)
        self.assertEqual(packet.observation_id, "obs-1")
        self.assertEqual(packet.image_sha256, IMAGE_SHA)
        route, payload, timeout_ms = transport.calls[-1]
        self.assertEqual(route, "/v1/detect")
        self.assertEqual(timeout_ms, 5_000)
        self.assertEqual(payload["image_sha256"], payload["image"]["image_sha256"])
        self.assertEqual(
            payload["allowed_class_names"],
            ["red_part", "blue_part"],
        )
        self.assertNotIn("task_context", payload)
        serialized = json.dumps(payload).lower()
        self.assertNotIn("ground_truth", serialized)
        self.assertNotIn('"gt"', serialized)

    def test_adapter_health_and_timeout(self) -> None:
        healthy = YoloHTTPAdapter(
            EchoYoloTransport(),
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
        )
        self.assertTrue(healthy.health())
        unhealthy = YoloHTTPAdapter(
            EchoYoloTransport(health_overrides={"checkpoint_sha": OTHER_IMAGE_SHA}),
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
        )
        self.assertFalse(unhealthy.health())

        timing_out = YoloHTTPAdapter(
            EchoYoloTransport(timeout=True),
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
        )
        with self.assertRaises(PerceptionError) as caught:
            timing_out.detect(make_context())
        self.assertEqual(caught.exception.code, FailureCode.PERCEPTION_TIMEOUT)
        self.assertTrue(caught.exception.retryable)

    def test_mock_implements_protocol_and_preserves_frame_binding(self) -> None:
        mock = MockPerceptionAgent(
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
            detector=lambda context: (make_detection(),),
        )
        self.assertIsInstance(mock, PerceptionAgent)
        packet = mock.detect(make_context())
        self.assertEqual(packet.observation_id, "obs-1")
        self.assertEqual(packet.image_sha256, IMAGE_SHA)

    def test_json_schemas_validate_canonical_exchange(self) -> None:
        root = Path(__file__).resolve().parents[1]
        loaded = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((root / "schemas").glob("*.schema.json"))
        ]
        registry = Registry().with_resources(
            (schema["$id"], Resource.from_contents(schema)) for schema in loaded
        )
        detection_schema = next(
            item
            for item in loaded
            if item["$id"].endswith("/detection-packet.schema.json")
        )
        Draft202012Validator(detection_schema, registry=registry).validate(
            make_packet().to_dict()
        )

        detect_schema = next(
            item
            for item in loaded
            if item["$id"].endswith("/perception-detect.schema.json")
        )
        transport = EchoYoloTransport()
        adapter = YoloHTTPAdapter(
            transport,
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
        )
        adapter.detect(make_context())
        request = transport.calls[-1][1]
        request_validator = Draft202012Validator(
            detect_schema, registry=registry
        ).evolve(schema=detect_schema["$defs"]["request"])
        request_validator.validate(request)

        response = EchoYoloTransport().request("/v1/detect", request, 5_000)
        response_validator = Draft202012Validator(
            detect_schema, registry=registry
        ).evolve(schema=detect_schema["$defs"]["response"])
        response_validator.validate(response)
        stale = deepcopy(response)
        stale["detection_packet"]["detections"][0]["bbox_format"] = "xywh_pixels"
        with self.assertRaises(ValidationError):
            response_validator.validate(stale)

    def test_factory_consumes_yolo_url_and_pinned_identity(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        config["perception"]["checkpoint_sha"] = CHECKPOINT_SHA
        config["perception"]["class_map_sha"] = CLASS_MAP_SHA
        config["perception"]["config_sha"] = CONFIG_SHA
        calls: list[tuple[str, str]] = []

        def factory(name: str, base_url: str) -> EchoYoloTransport:
            calls.append((name, base_url))
            return EchoYoloTransport()

        adapter = build_perception_from_config(config, factory)
        self.assertEqual(
            calls,
            [("yolo", "http://127.0.0.1:8103")],
        )
        self.assertEqual(adapter.descriptor.checkpoint_sha, CHECKPOINT_SHA)
        self.assertEqual(adapter.descriptor.class_map_sha, CLASS_MAP_SHA)
        self.assertEqual(adapter.descriptor.config_sha, CONFIG_SHA)

    def test_factory_rejects_unpinned_yolo_artifacts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        with self.assertRaisesRegex(ValueError, "unsafe placeholder"):
            build_perception_from_config(
                config,
                lambda name, base_url: EchoYoloTransport(),
            )


if __name__ == "__main__":
    unittest.main()
