from __future__ import annotations

import ast
from dataclasses import replace
import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from industrial_agent.contracts import Observation
from industrial_agent.errors import FailureCode
from industrial_agent.perception import (
    DetectionPacket,
    PerceptionError,
    PerceptionContext,
    PerceptionDescriptor,
    PerceptionTiming,
)
from simulation.yolo_camera_probe import (
    discover_yolo_http_agent,
    probe_yolo_cameras,
)


CHECKPOINT_SHA = "sha256:" + "a" * 64
CLASS_MAP_SHA = "sha256:" + "b" * 64
CONFIG_SHA = "sha256:" + "c" * 64
CAMERAS = (
    ("arm_a_rgb", "CAM_A_TOP", "1"),
    ("handoff_rgb", "CAM_HANDOFF", "2"),
    ("arm_b_rgb", "CAM_B_TOP", "3"),
)


def _image_reference(camera_id: str, digest_character: str) -> dict[str, Any]:
    digest = digest_character * 64
    return {
        "uri": f"cas://sha256/{digest}",
        "image_sha256": f"sha256:{digest}",
        "camera_id": camera_id,
        "width": 1280,
        "height": 720,
    }


def _observation() -> Observation:
    return Observation(
        observation_id="observation-1",
        timestamp_ms=123,
        data={
            "camera": {
                stream: _image_reference(camera_id, digest_character)
                for stream, camera_id, digest_character in CAMERAS
            }
        },
    )


class RecordingYolo:
    def __init__(self) -> None:
        self.descriptor = PerceptionDescriptor(
            name="yolo",
            task_types=frozenset({"object_localization"}),
            detection_contract_version="1.0",
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
        )
        self.calls: list[PerceptionContext] = []

    def health(self) -> bool:
        return True

    def detect(self, context: PerceptionContext) -> DetectionPacket:
        self.calls.append(context)
        call_number = len(self.calls)
        return DetectionPacket(
            packet_id=f"packet-{call_number}",
            request_id=f"request-{call_number}",
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
            checkpoint_sha=self.descriptor.checkpoint_sha,
            class_map_sha=self.descriptor.class_map_sha,
            config_sha=self.descriptor.config_sha,
            detections=(),
            timing=PerceptionTiming(1.0, 2.0, 3.0, 6.0),
        )

    def cancel(self, task_id: str, reason: str) -> None:
        return


class HealthTransport:
    def __init__(self, *, mode: str = "real") -> None:
        self.mode = mode
        self.calls: list[str] = []

    def request(self, route: str, payload: Any, timeout_ms: int) -> dict[str, Any]:
        self.calls.append(route)
        assert payload == {}
        assert timeout_ms > 0
        return {
            "schema_version": "1.0",
            "service": "yolo",
            "service_version": "0.2.0",
            "status": "ready",
            "checkpoint_sha": CHECKPOINT_SHA,
            "class_map_sha": CLASS_MAP_SHA,
            "config_sha": CONFIG_SHA,
            "supported_task_types": ["object_localization"],
            "supported_detection_contracts": ["1.0"],
            "queue": {"max_concurrent_requests": 1},
            "device": {"mode": self.mode, "target": "cpu"},
        }


def _run_probe(
    perception: RecordingYolo,
    *,
    observation: Observation | None = None,
    evidence_jsonl_path: Path | None = None,
) -> dict[str, Any]:
    return probe_yolo_cameras(
        observation or _observation(),
        perception,
        run_id="run-1",
        task_id="task-1",
        subtask_id="S01_ARM_A_PACK_HANDOFF",
        step_id=7,
        timeout_ms=3210,
        allowed_class_names=("part", "bin"),
        confidence_threshold=0.3,
        iou_threshold=0.4,
        evidence_jsonl_path=evidence_jsonl_path,
    )


def test_probe_detects_all_three_cameras_in_frozen_order_on_same_observation() -> None:
    perception = RecordingYolo()

    summary = _run_probe(perception)

    assert [call.image.camera_id for call in perception.calls] == [
        "CAM_A_TOP",
        "CAM_HANDOFF",
        "CAM_B_TOP",
    ]
    assert {call.observation_id for call in perception.calls} == {"observation-1"}
    assert {call.step_id for call in perception.calls} == {7}
    assert all(call.timeout_ms == 3210 for call in perception.calls)
    assert all(call.allowed_class_names == ("part", "bin") for call in perception.calls)
    assert summary["camera_order"] == [
        "CAM_A_TOP",
        "CAM_HANDOFF",
        "CAM_B_TOP",
    ]
    assert [result["stream_name"] for result in summary["results"]] == [
        "arm_a_rgb",
        "handoff_rgb",
        "arm_b_rgb",
    ]
    assert [result["detection_count"] for result in summary["results"]] == [0, 0, 0]
    assert summary["status"] == "ok"
    assert summary["successful_camera_count"] == 3
    assert summary["failed_camera_count"] == 0
    assert json.loads(json.dumps(summary, allow_nan=False)) == summary


def test_probe_validates_every_reference_before_calling_detector() -> None:
    observation = _observation()
    camera = dict(observation.data["camera"])
    camera["arm_b_rgb"] = _image_reference("CAM_HANDOFF", "3")
    invalid = replace(observation, data={"camera": camera})
    perception = RecordingYolo()

    with pytest.raises(ValueError, match="arm_b_rgb.camera_id"):
        _run_probe(perception, observation=invalid)

    assert perception.calls == []


def test_probe_persists_partial_failure_and_continues_remaining_camera(
    tmp_path: Path,
) -> None:
    class WrongFrameYolo(RecordingYolo):
        def detect(self, context: PerceptionContext) -> DetectionPacket:
            packet = super().detect(context)
            if context.image.camera_id == "CAM_HANDOFF":
                return replace(packet, observation_id="stale-observation")
            return packet

    evidence_path = tmp_path / "probe.jsonl"
    perception = WrongFrameYolo()

    summary = _run_probe(perception, evidence_jsonl_path=evidence_path)

    assert [call.image.camera_id for call in perception.calls] == [
        "CAM_A_TOP",
        "CAM_HANDOFF",
        "CAM_B_TOP",
    ]
    assert summary["status"] == "partial_failure"
    assert [item["status"] for item in summary["results"]] == [
        "ok",
        "error",
        "ok",
    ]
    assert summary["results"][1]["error"]["code"] == "PERC_2203_BAD_RESPONSE"
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == summary


def test_probe_persists_all_timeout_failures(tmp_path: Path) -> None:
    class TimeoutYolo(RecordingYolo):
        def detect(self, context: PerceptionContext) -> DetectionPacket:
            self.calls.append(context)
            raise PerceptionError(
                FailureCode.PERCEPTION_TIMEOUT,
                "detector timed out",
                retryable=True,
            )

    evidence_path = tmp_path / "probe.jsonl"
    perception = TimeoutYolo()

    summary = _run_probe(perception, evidence_jsonl_path=evidence_path)

    assert len(perception.calls) == 3
    assert summary["status"] == "failed"
    assert summary["successful_camera_count"] == 0
    assert summary["failed_camera_count"] == 3
    assert {item["status"] for item in summary["results"]} == {"timeout"}
    assert all(item["error"]["retryable"] for item in summary["results"])
    assert json.loads(evidence_path.read_text(encoding="utf-8")) == summary


def test_probe_rejects_non_frozen_subtask_id() -> None:
    with pytest.raises(ValueError, match="subtask_id must be"):
        probe_yolo_cameras(
            _observation(),
            RecordingYolo(),
            run_id="run-1",
            task_id="task-1",
            subtask_id="three-camera-yolo-probe",
        )


@pytest.mark.parametrize("subtask_id", ["P01_TO_S11", "W01_TO_S14"])
def test_probe_accepts_formal_v2_curriculum_subtask_ids(subtask_id: str) -> None:
    perception = RecordingYolo()

    summary = probe_yolo_cameras(
        _observation(),
        perception,
        run_id="run-v2",
        task_id=subtask_id,
        subtask_id=subtask_id,
    )

    assert summary["status"] == "ok"
    assert {call.subtask_id for call in perception.calls} == {subtask_id}


def test_probe_durably_appends_one_json_line_after_complete_success(
    tmp_path: Path,
) -> None:
    evidence_path = tmp_path / "nested" / "probe.jsonl"
    perception = RecordingYolo()

    with patch("simulation.yolo_camera_probe.os.fsync") as fsync:
        first = _run_probe(perception, evidence_jsonl_path=evidence_path)
        second = _run_probe(RecordingYolo(), evidence_jsonl_path=evidence_path)

    # Linux additionally fsyncs the parent directory after creating the file.
    assert fsync.call_count >= 2
    persisted = [
        json.loads(line)
        for line in evidence_path.read_text(encoding="utf-8").splitlines()
    ]
    assert persisted == [first, second]


def test_failed_fsync_rolls_back_unacknowledged_json_line(tmp_path: Path) -> None:
    evidence_path = tmp_path / "probe.jsonl"

    with patch(
        "simulation.yolo_camera_probe.os.fsync",
        side_effect=OSError("disk unavailable"),
    ):
        with pytest.raises(RuntimeError, match="JSONL append failed"):
            _run_probe(RecordingYolo(), evidence_jsonl_path=evidence_path)

    assert evidence_path.read_bytes() == b""


def test_probe_module_has_no_isaac_or_omniverse_imports() -> None:
    source_path = (
        Path(__file__).resolve().parents[1] / "simulation" / "yolo_camera_probe.py"
    )
    syntax = ast.parse(source_path.read_text(encoding="utf-8"))
    roots: set[str] = set()
    for node in ast.walk(syntax):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".", 1)[0])

    assert roots.isdisjoint({"isaacsim", "omni"})


def test_discover_yolo_agent_pins_live_health_identity() -> None:
    transport = HealthTransport()

    adapter, health = discover_yolo_http_agent(
        "http://127.0.0.1:8103",
        transport=transport,
    )

    assert transport.calls == ["/health", "/health"]
    assert adapter.descriptor.checkpoint_sha == CHECKPOINT_SHA
    assert adapter.descriptor.class_map_sha == CLASS_MAP_SHA
    assert health["device"]["mode"] == "real"


def test_discover_yolo_agent_rejects_mock_by_default() -> None:
    with pytest.raises(RuntimeError, match="refuses mock mode"):
        discover_yolo_http_agent(
            "http://127.0.0.1:8103",
            transport=HealthTransport(mode="mock"),
        )
