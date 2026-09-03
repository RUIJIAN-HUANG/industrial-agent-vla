from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from industrial_agent.online_task_state import OnlineTaskStateProvider
from industrial_agent.perception import (
    Detection,
    DetectionPacket,
    PerceptionContext,
    PerceptionDescriptor,
    PerceptionTiming,
)
from industrial_agent.v2_task_profile import require_formal_v2_task


ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_SHA = "sha256:" + "a" * 64
CLASS_MAP_SHA = "sha256:" + "b" * 64
CONFIG_SHA = "sha256:" + "c" * 64


class _Yolo:
    def __init__(self, outputs: list[tuple[str, str | None, float] | None]) -> None:
        self.outputs = list(outputs)
        self.calls: list[PerceptionContext] = []
        self.descriptor = PerceptionDescriptor(
            name="yolo",
            task_types=frozenset({"object_localization", "visual_manipulation"}),
            detection_contract_version="1.0",
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
        )

    def health(self) -> bool:
        return True

    def detect(self, context: PerceptionContext) -> DetectionPacket:
        self.calls.append(context)
        output = self.outputs.pop(0)
        detections = ()
        if output is not None:
            class_name, zone_id, confidence = output
            detections = (
                Detection(
                    detection_id=f"detection-{len(self.calls)}",
                    class_id=0,
                    class_name=class_name,
                    confidence=confidence,
                    bbox_xyxy=(100.0, 100.0, 140.0, 140.0),
                    camera_id=context.image.camera_id,
                    image_width=context.image.width,
                    image_height=context.image.height,
                    zone_id=zone_id,
                ),
            )
        return DetectionPacket(
            packet_id=f"packet-{len(self.calls)}",
            request_id=f"request-{len(self.calls)}",
            trace_id=context.run_id,
            episode_id=context.run_id,
            task_id=context.task_id,
            subtask_id=context.subtask_id,
            step_id=context.step_id,
            observation_id=context.observation_id,
            image_sha256=context.image.image_sha256,
            camera_id=context.image.camera_id,
            image_width=context.image.width,
            image_height=context.image.height,
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
            detections=detections,
            timing=PerceptionTiming(1.0, 1.0, 1.0, 3.0),
        )

    def cancel(self, task_id: str, reason: str) -> None:
        return


def _scene() -> dict[str, Any]:
    return json.loads(
        (ROOT / "simulation" / "configs" / "single_bin_scene_v2.json").read_text(
            encoding="utf-8"
        )
    )


def _image(camera_id: str, digit: str) -> dict[str, Any]:
    return {
        "uri": f"cas://sha256/{digit * 64}",
        "image_sha256": f"sha256:{digit * 64}",
        "camera_id": camera_id,
        "width": 1280,
        "height": 720,
    }


def _camera() -> dict[str, Any]:
    return {
        "arm_a_rgb": _image("CAM_A_TOP", "1"),
        "handoff_rgb": _image("CAM_HANDOFF", "2"),
        "arm_b_rgb": _image("CAM_B_TOP", "3"),
    }


def _arm(*, open_: bool = True, stationary: bool = True, retreated: bool = True):
    return {
        "tcp_pose_m_rad": [0.0] * 6,
        "state": [0.0] * 7,
        "gripper_open": open_,
        "stationary": stationary,
        "retreated": retreated,
    }


def _robot(
    *,
    arm_a_open: bool = True,
    arm_a_stationary: bool = True,
    arm_a_retreated: bool = True,
    arm_b_open: bool = True,
) -> dict[str, Any]:
    return {
        "active_arm": "Arm_A",
        "arm_a": _arm(
            open_=arm_a_open,
            stationary=arm_a_stationary,
            retreated=arm_a_retreated,
        ),
        "arm_b": _arm(open_=arm_b_open),
    }


def _provider(task_id: str, outputs: list[tuple[str, str | None, float] | None]):
    yolo = _Yolo(outputs)
    provider = OnlineTaskStateProvider(
        task_spec=require_formal_v2_task(task_id),
        perception=yolo,
        scene_config=_scene(),
        run_id="run-online-state",
    )
    return provider, yolo


def _update(
    provider: OnlineTaskStateProvider,
    index: int,
    *,
    robot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return provider.update(
        observation_id=f"observation-{index}",
        timestamp_ms=1_000 + index,
        camera=_camera(),
        robot=robot or _robot(),
    )


@pytest.mark.parametrize(
    ("task_id", "class_name", "target_id", "camera_id"),
    [
        ("P01_TO_S11", "shaft_upright", "S11", "CAM_A_TOP"),
        ("W01_TO_S14", "open_end_wrench", "S14", "CAM_A_TOP"),
    ],
)
def test_part_provider_requires_three_fresh_frames_and_two_votes(
    task_id: str,
    class_name: str,
    target_id: str,
    camera_id: str,
) -> None:
    provider, yolo = _provider(
        task_id,
        [(class_name, target_id, 0.8), None, (class_name, target_id, 0.7)],
    )

    first = _update(provider, 1)
    second = _update(provider, 2)
    final = _update(provider, 3)

    assert first["terminal"] is False and first["verification_votes"] == 1
    assert second["terminal"] is False and second["verification_votes"] == 1
    assert final == {
        "task_id": task_id,
        "target_object_id": require_formal_v2_task(task_id).target_object,
        "target_slot_id": target_id,
        "status": "SUCCEEDED",
        "terminal": True,
        "terminal_confidence": 0.7,
        "verification_votes": 2,
    }
    assert provider.control_token() == "NONE"
    assert [call.image.camera_id for call in yolo.calls] == [camera_id] * 3


def test_part_provider_combines_yolo_with_measured_gripper_and_motion_state() -> None:
    provider, _ = _provider(
        "P01_TO_S11",
        [("shaft_upright", "S11", 0.9)] * 3,
    )

    _update(provider, 1, robot=_robot(arm_a_open=False))
    _update(provider, 2, robot=_robot(arm_a_stationary=False))
    state = _update(provider, 3)

    assert state["terminal"] is False
    assert state["verification_votes"] == 1


def test_provider_rejects_replayed_observation_id() -> None:
    provider, yolo = _provider("P01_TO_S11", [None])
    _update(provider, 1)

    with pytest.raises(ValueError, match="not fresh"):
        _update(provider, 1)

    assert len(yolo.calls) == 1


def test_bin_handoff_uses_one_yolo_agent_and_exact_token_sequence() -> None:
    provider, yolo = _provider(
        "BIN01_TO_FINISHED01",
        [
            None,
            ("bin_box", "HANDOFF_CENTER", 0.9),
            ("bin_box", "HANDOFF_CENTER", 0.8),
            ("bin_box", "HANDOFF_CENTER", 0.7),
            ("bin_box", "FINISHED_01", 0.9),
            ("bin_box", "FINISHED_01", 0.8),
            ("bin_box", "FINISHED_01", 0.7),
        ],
    )
    token_sequence = [provider.control_token()]

    _update(provider, 1)
    _update(provider, 2)
    token_sequence.append(provider.control_token())
    _update(provider, 3)
    handoff = _update(provider, 4)
    token_sequence.append(provider.control_token())
    first_final = _update(provider, 5)
    second_final = _update(provider, 6)
    final = _update(provider, 7)
    token_sequence.append(provider.control_token())

    assert token_sequence == ["A_ONLY", "HANDOFF_VERIFY", "B_ONLY", "NONE"]
    assert handoff["terminal"] is False
    assert first_final["terminal"] is False
    assert second_final["terminal"] is False
    assert final["terminal"] is True
    assert final["verification_votes"] == 3
    assert provider.active_arm() == "NONE"
    assert [call.image.camera_id for call in yolo.calls] == [
        "CAM_HANDOFF",
        "CAM_HANDOFF",
        "CAM_HANDOFF",
        "CAM_HANDOFF",
        "CAM_B_TOP",
        "CAM_B_TOP",
        "CAM_B_TOP",
    ]
    assert {id(yolo)} == {id(provider.perception)}
    assert [call.subtask_id for call in yolo.calls[-3:]] == ["S02_ARM_B_TRANSPORT"] * 3


def test_handoff_does_not_switch_until_both_arms_are_parked() -> None:
    provider, _ = _provider(
        "BIN01_TO_FINISHED01",
        [("bin_box", "HANDOFF_CENTER", 0.9)] * 2,
    )

    _update(provider, 1, robot=_robot(arm_a_retreated=False))
    assert provider.control_token() == "A_ONLY"
    _update(provider, 2)
    assert provider.control_token() == "HANDOFF_VERIFY"


def test_generic_yolo_bbox_can_vote_inside_projected_slot_calibration() -> None:
    provider, yolo = _provider("P01_TO_S11", [None])
    x_min, y_min, x_max, y_max = provider._projected_target_region(
        "S11", camera_id="CAM_A_TOP", width=1280, height=720
    )
    context_image = _camera()["arm_a_rgb"]
    yolo.outputs = []

    class _ProjectedYolo(_Yolo):
        def detect(self, context: PerceptionContext) -> DetectionPacket:
            self.calls.append(context)
            detection = Detection(
                detection_id="projected",
                class_id=0,
                class_name="shaft_upright",
                confidence=0.9,
                bbox_xyxy=(
                    (x_min + x_max) / 2 - 5,
                    (y_min + y_max) / 2 - 5,
                    (x_min + x_max) / 2 + 5,
                    (y_min + y_max) / 2 + 5,
                ),
                camera_id=context.image.camera_id,
                image_width=context.image.width,
                image_height=context.image.height,
            )
            return DetectionPacket(
                packet_id="packet-projected",
                request_id="request-projected",
                trace_id=context.run_id,
                episode_id=context.run_id,
                task_id=context.task_id,
                subtask_id=context.subtask_id,
                step_id=context.step_id,
                observation_id=context.observation_id,
                image_sha256=context.image.image_sha256,
                camera_id=context.image.camera_id,
                image_width=context.image.width,
                image_height=context.image.height,
                checkpoint_sha=CHECKPOINT_SHA,
                class_map_sha=CLASS_MAP_SHA,
                config_sha=CONFIG_SHA,
                detections=(detection,),
                timing=PerceptionTiming(1.0, 1.0, 1.0, 3.0),
            )

    projected = _ProjectedYolo([])
    provider = OnlineTaskStateProvider(
        task_spec=require_formal_v2_task("P01_TO_S11"),
        perception=projected,
        scene_config=_scene(),
        run_id="run-projected",
    )

    state = _update(provider, 1)

    assert state["verification_votes"] == 1
    assert projected.calls[0].image.to_dict() == context_image
