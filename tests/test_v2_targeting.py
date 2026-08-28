from __future__ import annotations

import json
from pathlib import Path

import pytest

from industrial_agent.perception import (
    Detection,
    DetectionPacket,
    ImageReference,
    PerceptionContext,
    PerceptionTiming,
)
from industrial_agent.v2_targeting import (
    V2_OBJECTS_BY_ID,
    V2_SLOTS_BY_ID,
    infer_v2_slot_id,
    infer_v2_zone_id,
    resolve_v2_target_for_task,
    resolve_v2_target_for_task_or_instruction,
    resolve_v2_target_instruction,
    select_target_detection,
    select_target_slot_detection,
    targeted_perception_context,
    v2_slot,
)


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = f"sha256:{'1' * 64}"
CLASS_MAP_SHA = f"sha256:{'2' * 64}"
CONFIG_SHA = f"sha256:{'3' * 64}"
IMAGE_SHA = f"sha256:{'4' * 64}"


def _image() -> ImageReference:
    digest = IMAGE_SHA.removeprefix("sha256:")
    return ImageReference(
        uri=f"cas://sha256/{digest}",
        image_sha256=IMAGE_SHA,
        camera_id="CAM_A_TOP",
        width=1280,
        height=720,
    )


def _context() -> PerceptionContext:
    return PerceptionContext(
        run_id="run-1",
        task_id="P01_TO_S11",
        subtask_id="P01_TO_S11",
        step_id=0,
        observation_id="obs-1",
        image=_image(),
        allowed_class_names=("hex_nut", "bin_box"),
    )


def _detection(
    detection_id: str,
    class_name: str,
    confidence: float,
    *,
    class_id: int = 0,
    camera_id: str = "CAM_A_TOP",
    zone_id: str | None = None,
    bbox_xyxy: tuple[float, float, float, float] = (10.0, 20.0, 110.0, 120.0),
) -> Detection:
    return Detection(
        detection_id=detection_id,
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox_xyxy=bbox_xyxy,
        camera_id=camera_id,
        image_width=1280,
        image_height=720,
        zone_id=zone_id,
    )


def _packet(*detections: Detection) -> DetectionPacket:
    return DetectionPacket(
        packet_id="packet-1",
        request_id="req-1",
        trace_id="run-1",
        episode_id="run-1",
        task_id="P01_TO_S11",
        subtask_id="P01_TO_S11",
        step_id=0,
        observation_id="obs-1",
        image_sha256=IMAGE_SHA,
        camera_id="CAM_A_TOP",
        image_width=1280,
        image_height=720,
        checkpoint_sha=CHECKPOINT_SHA,
        class_map_sha=CLASS_MAP_SHA,
        config_sha=CONFIG_SHA,
        detections=detections,
        timing=PerceptionTiming(1.0, 2.0, 0.5, 4.0),
    )


def test_p01_task_resolves_to_only_hex_nut_for_yolo() -> None:
    target = resolve_v2_target_for_task("P01_TO_S11")

    assert target.target_object.object_id == "P01"
    assert target.target_slot is not None
    assert target.target_slot.slot_id == "S11"
    assert target.allowed_class_names == ("hex_nut",)

    context = targeted_perception_context(_context(), target)
    assert context.allowed_class_names == ("hex_nut",)


def test_literal_instruction_resolves_object_slot_and_slot_alias() -> None:
    target = resolve_v2_target_instruction("请将螺母 P01 放到 s01")

    assert target.target_object.class_name == "hex_nut"
    assert target.target_slot is not None
    assert target.target_slot.slot_id == "S11"
    assert v2_slot("s01").slot_id == "S11"


def test_task_or_instruction_falls_back_to_literal_v2_ids() -> None:
    target = resolve_v2_target_for_task_or_instruction(
        "operator-free-text",
        "把 P01 放到 s01",
    )

    assert target.task_id == "P01_TO_S11"
    assert target.target_object.object_id == "P01"
    assert target.target_slot is not None
    assert target.target_slot.slot_id == "S11"
    assert target.allowed_class_names == ("hex_nut",)


def test_selector_ignores_other_classes_and_locks_target_zone() -> None:
    target = resolve_v2_target_for_task("P01_TO_S11")
    packet = _packet(
        _detection("det-wrench", "open_end_wrench", 0.99, class_id=3, zone_id="D"),
        _detection("det-n02", "hex_nut", 0.96, class_id=2, zone_id="C"),
        _detection("det-p01", "hex_nut", 0.91, class_id=2, zone_id="A"),
    )

    lock = select_target_detection(packet, target)

    assert lock.object_id == "P01"
    assert lock.slot_id == "S11"
    assert lock.class_name == "hex_nut"
    assert lock.detection.detection_id == "det-p01"
    assert lock.candidate_count == 2


def test_selector_infers_p01_zone_from_bbox_when_yolo_has_no_instance_id() -> None:
    target = resolve_v2_target_for_task("P01_TO_S11")
    packet = _packet(
        _detection(
            "det-n02",
            "hex_nut",
            0.96,
            class_id=2,
            bbox_xyxy=(120.0, 450.0, 220.0, 550.0),
        ),
        _detection(
            "det-p01",
            "hex_nut",
            0.91,
            class_id=2,
            bbox_xyxy=(120.0, 80.0, 220.0, 180.0),
        ),
    )

    lock = select_target_detection(packet, target)

    assert lock.object_id == "P01"
    assert lock.detection.detection_id == "det-p01"
    assert lock.detection.zone_id == "A"
    assert infer_v2_zone_id(lock.detection) == "A"


def test_slot_alias_s01_locks_to_s11_from_bin_slot_bbox() -> None:
    detection = _detection(
        "det-slot-s11",
        "bin_slot",
        0.88,
        class_id=5,
        bbox_xyxy=(340.0, 245.0, 440.0, 315.0),
    )
    packet = _packet(detection)

    lock = select_target_slot_detection(packet, "s01")

    assert lock.slot_id == "S11"
    assert lock.slot_index == 1
    assert lock.detection.detection_id == "det-slot-s11"
    assert infer_v2_slot_id(detection) == "S11"


def test_selector_fails_closed_when_target_class_is_missing() -> None:
    packet = _packet(_detection("det-wrench", "open_end_wrench", 0.99, class_id=3))

    with pytest.raises(ValueError, match="no detection matches P01"):
        select_target_detection(packet, "P01")


def test_object_catalog_json_matches_runtime_catalog() -> None:
    payload = json.loads(
        (ROOT / "configs" / "v2-object-catalog.json").read_text(encoding="utf-8")
    )

    assert payload["catalog_id"] == "single_bin_manual_industrial_v2"
    assert {item["object_id"] for item in payload["objects"]} == set(
        V2_OBJECTS_BY_ID
    )
    assert {item["slot_id"] for item in payload["slots"]} == set(V2_SLOTS_BY_ID)
    for item in payload["objects"]:
        runtime = V2_OBJECTS_BY_ID[item["object_id"]]
        assert item["class_name"] == runtime.class_name
        assert item["zone_id"] == runtime.zone_id
