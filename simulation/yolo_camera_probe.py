"""Run the YOLO perception contract against one synchronized camera triplet.

This module intentionally imports no Isaac Sim packages.  A simulation owner
must first capture and publish its RGB frames to the shared CAS, validate the
resulting online :class:`~industrial_agent.contracts.Observation`, and only then
pass that observation to :func:`probe_yolo_cameras`.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from threading import Lock
from typing import Any, Mapping, Sequence

from industrial_agent.contracts import Observation
from industrial_agent.errors import ContractError, FailureCode
from industrial_agent.http_transport import BoundedHTTPTransport
from industrial_agent.observation import FROZEN_IMAGE_HEIGHT, FROZEN_IMAGE_WIDTH
from industrial_agent.perception import (
    DetectionPacket,
    ImageReference,
    PerceptionAgent,
    PerceptionContext,
    PerceptionDescriptor,
    YoloHTTPAdapter,
)


PROBE_SCHEMA_VERSION = "1.0"
CAMERA_STREAMS: tuple[tuple[str, str], ...] = (
    ("arm_a_rgb", "CAM_A_TOP"),
    ("handoff_rgb", "CAM_HANDOFF"),
    ("arm_b_rgb", "CAM_B_TOP"),
)

_JSONL_LOCK = Lock()


def discover_yolo_http_agent(
    base_url: str,
    *,
    timeout_ms: int = 5_000,
    allow_mock: bool = False,
    transport: Any | None = None,
) -> tuple[YoloHTTPAdapter, dict[str, Any]]:
    """Discover a verified live YOLO identity and build its strict HTTP client."""

    client_transport = transport or BoundedHTTPTransport(base_url)
    health = client_transport.request("/health", {}, min(timeout_ms, 5_000))
    if not isinstance(health, Mapping):
        raise RuntimeError("YOLO /health must return an object")
    device = health.get("device")
    if not isinstance(device, Mapping) or device.get("mode") not in {"mock", "real"}:
        raise RuntimeError("YOLO /health device identity is invalid")
    if device["mode"] == "mock" and not allow_mock:
        raise RuntimeError(
            "YOLO camera smoke refuses mock mode; pass allow_mock only for software tests"
        )
    task_types = health.get("supported_task_types")
    if (
        not isinstance(task_types, Sequence)
        or isinstance(task_types, (str, bytes, bytearray))
        or not task_types
        or any(not isinstance(item, str) or not item for item in task_types)
    ):
        raise RuntimeError("YOLO /health supported_task_types is invalid")
    try:
        adapter = YoloHTTPAdapter(
            client_transport,
            checkpoint_sha=health["checkpoint_sha"],
            class_map_sha=health["class_map_sha"],
            config_sha=health["config_sha"],
            task_types=frozenset(task_types),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("YOLO /health deployment digests are invalid") from exc
    if not adapter.health():
        raise RuntimeError("YOLO health identity changed during camera smoke discovery")
    return adapter, dict(health)


def _camera_references(
    observation: Observation,
) -> tuple[tuple[str, ImageReference], ...]:
    if not isinstance(observation, Observation):
        raise TypeError("observation must be a validated Observation")
    if not isinstance(observation.data, Mapping):
        raise TypeError("observation.data must be an object")
    camera = observation.data.get("camera")
    if not isinstance(camera, Mapping):
        raise ValueError("observation.data.camera must be an object")

    references: list[tuple[str, ImageReference]] = []
    for stream_name, expected_camera_id in CAMERA_STREAMS:
        raw_reference = camera.get(stream_name)
        if not isinstance(raw_reference, Mapping):
            raise ValueError(
                f"observation.data.camera.{stream_name} must be an ImageReference"
            )
        reference = ImageReference.from_dict(raw_reference)
        if reference.camera_id != expected_camera_id:
            raise ValueError(
                f"observation.data.camera.{stream_name}.camera_id must be "
                f"{expected_camera_id!r}"
            )
        if (reference.width, reference.height) != (
            FROZEN_IMAGE_WIDTH,
            FROZEN_IMAGE_HEIGHT,
        ):
            raise ValueError(
                f"observation.data.camera.{stream_name} must use frozen "
                f"{FROZEN_IMAGE_WIDTH}x{FROZEN_IMAGE_HEIGHT} resolution"
            )
        references.append((stream_name, reference))
    return tuple(references)


def _require_yolo_descriptor(perception: PerceptionAgent) -> PerceptionDescriptor:
    descriptor = getattr(perception, "descriptor", None)
    if not isinstance(descriptor, PerceptionDescriptor):
        raise TypeError("perception.descriptor must be a PerceptionDescriptor")
    if descriptor.name != "yolo":
        raise ValueError("the camera probe requires the perception Agent named 'yolo'")
    return descriptor


def _validate_packet_correlation(
    packet: DetectionPacket,
    *,
    context: PerceptionContext,
    descriptor: PerceptionDescriptor,
) -> None:
    packet.validate_against(
        observation_id=context.observation_id,
        image=context.image,
        descriptor=descriptor,
    )
    expected = {
        "trace_id": context.run_id,
        "episode_id": context.run_id,
        "task_id": context.task_id,
        "subtask_id": context.subtask_id,
        "step_id": context.step_id,
    }
    mismatches = {
        key: {"expected": expected_value, "actual": getattr(packet, key)}
        for key, expected_value in expected.items()
        if getattr(packet, key) != expected_value
    }
    if mismatches:
        raise ContractError(
            FailureCode.PERCEPTION_BAD_RESPONSE,
            f"YOLO probe packet correlation mismatch: {mismatches}",
        )


def _result_summary(
    *,
    stream_name: str,
    packet: DetectionPacket,
) -> dict[str, Any]:
    return {
        "stream_name": stream_name,
        "camera_id": packet.camera_id,
        "image_sha256": packet.image_sha256,
        "image_width": packet.image_width,
        "image_height": packet.image_height,
        "packet_id": packet.packet_id,
        "request_id": packet.request_id,
        "detection_count": len(packet.detections),
        "detections": [detection.to_dict() for detection in packet.detections],
        "timing": packet.timing.to_dict(),
    }


def _encode_jsonl_record(record: Mapping[str, Any]) -> bytes:
    try:
        return (
            json.dumps(
                dict(record),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            + "\n"
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise RuntimeError("YOLO camera probe summary is not JSON-safe") from exc


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _append_durable_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    encoded = _encode_jsonl_record(record)
    with _JSONL_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_symlink() or (path.exists() and not path.is_file()):
            raise RuntimeError("YOLO camera probe JSONL path must be a regular file")
        file_existed = path.exists()
        try:
            with path.open("ab+") as handle:
                handle.seek(0, os.SEEK_END)
                previous_size = handle.tell()
                try:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                except BaseException:
                    try:
                        handle.seek(previous_size)
                        handle.truncate()
                        handle.flush()
                        os.fsync(handle.fileno())
                    except OSError:
                        pass
                    raise
            if not file_existed:
                _fsync_directory(path.parent)
        except OSError as exc:
            raise RuntimeError(
                f"YOLO camera probe JSONL append failed: {path}"
            ) from exc


def probe_yolo_cameras(
    observation: Observation,
    perception: PerceptionAgent,
    *,
    run_id: str,
    task_id: str,
    subtask_id: str = "three-camera-yolo-probe",
    step_id: int = 0,
    timeout_ms: int = 5_000,
    allowed_class_names: Sequence[str] = (),
    confidence_threshold: float = 0.25,
    iou_threshold: float = 0.45,
    evidence_jsonl_path: str | Path | None = None,
) -> dict[str, Any]:
    """Detect the three synchronized CAS frames and return a JSON-safe summary.

    All three image references are validated before the first detector call.
    Detection runs are deliberately sequential and share one ``observation_id``.
    A summary is durably appended only after all three packets pass frame,
    deployment, and correlation validation.
    """

    descriptor = _require_yolo_descriptor(perception)
    references = _camera_references(observation)
    if isinstance(allowed_class_names, (str, bytes, bytearray)):
        raise TypeError("allowed_class_names must be a sequence of class names")
    class_names = tuple(allowed_class_names)
    results: list[dict[str, Any]] = []

    for stream_name, image in references:
        context = PerceptionContext(
            run_id=run_id,
            task_id=task_id,
            subtask_id=subtask_id,
            step_id=step_id,
            observation_id=observation.observation_id,
            image=image,
            timeout_ms=timeout_ms,
            allowed_class_names=class_names,
            confidence_threshold=confidence_threshold,
            iou_threshold=iou_threshold,
        )
        packet = perception.detect(context)
        if not isinstance(packet, DetectionPacket):
            raise TypeError("perception.detect() must return a DetectionPacket")
        _validate_packet_correlation(
            packet,
            context=context,
            descriptor=descriptor,
        )
        results.append(_result_summary(stream_name=stream_name, packet=packet))

    summary: dict[str, Any] = {
        "probe_schema_version": PROBE_SCHEMA_VERSION,
        "record_type": "three_camera_yolo_probe",
        "run_id": run_id,
        "task_id": task_id,
        "subtask_id": subtask_id,
        "step_id": step_id,
        "observation_id": observation.observation_id,
        "detector": descriptor.name,
        "checkpoint_sha": descriptor.checkpoint_sha,
        "class_map_sha": descriptor.class_map_sha,
        "config_sha": descriptor.config_sha,
        "detection_contract_version": descriptor.detection_contract_version,
        "camera_order": [camera_id for _stream, camera_id in CAMERA_STREAMS],
        "results": results,
    }
    _encode_jsonl_record(summary)
    if evidence_jsonl_path is not None:
        _append_durable_jsonl(Path(evidence_jsonl_path), summary)
    return summary


__all__ = [
    "CAMERA_STREAMS",
    "PROBE_SCHEMA_VERSION",
    "discover_yolo_http_agent",
    "probe_yolo_cameras",
]
