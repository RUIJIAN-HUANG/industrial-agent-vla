"""Transport-neutral contracts for the independent YOLO perception Agent.

The supervisor owns orchestration.  A perception Agent only turns one immutable
camera frame into detections; it never selects or invokes a VLA executor.
Ground-truth annotations belong to offline evaluation and are deliberately
absent from every online type in this module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from pathlib import Path
import re
from threading import Lock
from types import MappingProxyType
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable
from uuid import uuid4

from .errors import AgentError, ContractError, FailureCode
from .observation import find_forbidden_online_path

PERCEPTION_SCHEMA_VERSION = "1.0"
DETECTION_CONTRACT_VERSION = "1.0"


class PerceptionMode(str, Enum):
    """How YOLO evidence may influence the robot control path."""

    SHADOW_SCORE = "SHADOW_SCORE"


ARTIFACT_DIGEST_PATTERN = re.compile(r"sha256:[0-9a-fA-F]{64}")
CAS_IMAGE_URI_PATTERN = re.compile(r"cas://sha256/([0-9a-fA-F]{64})")
VERSION_PATTERN = re.compile(r"1\.[0-9]+")


def _contract_failure(message: str) -> ContractError:
    return ContractError(FailureCode.OBSERVATION_INVALID, message)


def _require_non_blank(value: Any, field_name: str, maximum: int = 512) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise _contract_failure(
            f"{field_name} must contain 1..{maximum} non-blank characters"
        )
    return value


def _require_strict_int(
    value: Any, field_name: str, *, minimum: int = 0, maximum: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _contract_failure(f"{field_name} must be an integer >= {minimum}")
    if maximum is not None and value > maximum:
        raise _contract_failure(f"{field_name} must be <= {maximum}")
    return value


def _require_finite_number(
    value: Any,
    field_name: str,
    *,
    minimum: float = 0.0,
    maximum: float | None = None,
) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(float(value))
    ):
        raise _contract_failure(f"{field_name} must be a finite number")
    normalized = float(value)
    if normalized < minimum or (maximum is not None and normalized > maximum):
        interval = f"[{minimum}, {maximum}]" if maximum is not None else f">= {minimum}"
        raise _contract_failure(f"{field_name} must be {interval}")
    return normalized


def is_pinned_artifact_digest(value: Any) -> bool:
    """Return whether ``value`` is an immutable SHA-256 identifier."""

    return (
        isinstance(value, str) and ARTIFACT_DIGEST_PATTERN.fullmatch(value) is not None
    )


def _require_digest(value: Any, field_name: str) -> str:
    if not is_pinned_artifact_digest(value):
        raise ValueError(
            f"{field_name} must match 'sha256:<64 hexadecimal characters>'"
        )
    return value


def _require_version(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or VERSION_PATTERN.fullmatch(value) is None:
        raise _contract_failure(f"{field_name} must be compatible with version 1.x")
    return value


def _require_exact_fields(
    value: Mapping[str, Any],
    *,
    required: frozenset[str],
    optional: frozenset[str] = frozenset(),
    object_name: str,
) -> None:
    missing = required - set(value)
    unknown = set(value) - required - optional
    if missing or unknown:
        raise _contract_failure(
            f"{object_name} fields invalid; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )


def _reject_privileged_keys(value: Any, path: str = "attributes") -> None:
    forbidden_path = find_forbidden_online_path(value, path)
    if forbidden_path is not None:
        raise _contract_failure(
            f"privileged offline field is forbidden online at {forbidden_path}"
        )
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise _contract_failure(f"{path} keys must be strings")
            _reject_privileged_keys(nested, f"{path}.{key}")
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for index, nested in enumerate(value):
            _reject_privileged_keys(nested, f"{path}[{index}]")
    elif value is not None and not isinstance(value, (str, int, float, bool)):
        raise _contract_failure(f"{path} must contain only JSON-compatible values")
    elif isinstance(value, float) and not isfinite(value):
        raise _contract_failure(f"{path} numbers must be finite")


@dataclass(frozen=True)
class PerceptionDescriptor:
    """Immutable identity and capability declaration for one YOLO deployment."""

    name: str
    task_types: frozenset[str]
    detection_contract_version: str
    checkpoint_sha: str
    class_map_sha: str
    config_sha: str

    def __post_init__(self) -> None:
        if self.name != "yolo":
            raise ValueError("perception service name must be 'yolo'")
        if not self.task_types or any(
            not isinstance(item, str) or not item for item in self.task_types
        ):
            raise ValueError("task_types must contain non-empty strings")
        _require_version(self.detection_contract_version, "detection_contract_version")
        _require_digest(self.checkpoint_sha, "checkpoint_sha")
        _require_digest(self.class_map_sha, "class_map_sha")
        _require_digest(self.config_sha, "config_sha")


@dataclass(frozen=True)
class ImageReference:
    """One immutable online image supplied to both YOLO and the selected VLA."""

    uri: str
    image_sha256: str
    camera_id: str
    width: int
    height: int

    def __post_init__(self) -> None:
        _require_non_blank(self.uri, "image.uri", 4096)
        _require_digest(self.image_sha256, "image.image_sha256")
        uri_match = CAS_IMAGE_URI_PATTERN.fullmatch(self.uri)
        if uri_match is None:
            raise _contract_failure(
                "image.uri must be a content-addressed cas://sha256/<digest> URI"
            )
        if (
            uri_match.group(1).casefold()
            != self.image_sha256.split(":", 1)[1].casefold()
        ):
            raise _contract_failure("image.uri digest must match image.image_sha256")
        _require_non_blank(self.camera_id, "image.camera_id", 128)
        _require_strict_int(self.width, "image.width", minimum=1, maximum=100_000)
        _require_strict_int(self.height, "image.height", minimum=1, maximum=100_000)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ImageReference":
        if not isinstance(value, Mapping):
            raise _contract_failure("image must be an object")
        _require_exact_fields(
            value,
            required=frozenset({"uri", "image_sha256", "camera_id", "width", "height"}),
            object_name="image",
        )
        return cls(
            uri=value["uri"],
            image_sha256=value["image_sha256"],
            camera_id=value["camera_id"],
            width=value["width"],
            height=value["height"],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "image_sha256": self.image_sha256,
            "camera_id": self.camera_id,
            "width": self.width,
            "height": self.height,
        }


@dataclass(frozen=True)
class PerceptionContext:
    """Correlation, deadline, frame identity, and non-privileged detector hints."""

    run_id: str
    task_id: str
    subtask_id: str
    step_id: int
    observation_id: str
    image: ImageReference
    timeout_ms: int = 5_000
    allowed_class_names: tuple[str, ...] = ()
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45

    def __post_init__(self) -> None:
        _require_non_blank(self.run_id, "run_id", 256)
        _require_non_blank(self.task_id, "task_id", 256)
        _require_non_blank(self.subtask_id, "subtask_id", 256)
        _require_strict_int(self.step_id, "step_id")
        _require_non_blank(self.observation_id, "observation_id", 256)
        if not isinstance(self.image, ImageReference):
            raise _contract_failure("image must be an ImageReference")
        _require_strict_int(self.timeout_ms, "timeout_ms", minimum=1, maximum=120_000)
        names = tuple(self.allowed_class_names)
        if len(names) > 1_000:
            raise _contract_failure("allowed_class_names cannot exceed 1000 items")
        for name in names:
            _require_non_blank(name, "allowed_class_names item", 256)
        if len(set(names)) != len(names):
            raise _contract_failure("allowed_class_names must be unique")
        object.__setattr__(self, "allowed_class_names", names)
        object.__setattr__(
            self,
            "confidence_threshold",
            _require_finite_number(
                self.confidence_threshold,
                "confidence_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )
        object.__setattr__(
            self,
            "iou_threshold",
            _require_finite_number(
                self.iou_threshold,
                "iou_threshold",
                minimum=0.0,
                maximum=1.0,
            ),
        )


@dataclass(frozen=True)
class Detection:
    """One pixel-space ``xyxy`` detection bound to a specific source image."""

    detection_id: str
    class_id: int
    class_name: str
    confidence: float
    bbox_xyxy: tuple[float, float, float, float]
    camera_id: str
    image_width: int
    image_height: int
    track_id: str | None = None
    zone_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    bbox_format: str = "xyxy_pixels"

    def __post_init__(self) -> None:
        _require_non_blank(self.detection_id, "detection_id", 256)
        _require_strict_int(self.class_id, "class_id")
        _require_non_blank(self.class_name, "class_name", 256)
        object.__setattr__(
            self,
            "confidence",
            _require_finite_number(
                self.confidence, "confidence", minimum=0.0, maximum=1.0
            ),
        )
        _require_non_blank(self.camera_id, "camera_id", 128)
        if self.bbox_format != "xyxy_pixels":
            raise _contract_failure("bbox_format must be 'xyxy_pixels'")
        _require_strict_int(self.image_width, "image_width", minimum=1, maximum=100_000)
        _require_strict_int(
            self.image_height, "image_height", minimum=1, maximum=100_000
        )
        if self.track_id is not None:
            _require_non_blank(self.track_id, "track_id", 256)
        if self.zone_id is not None:
            _require_non_blank(self.zone_id, "zone_id", 256)
        if not isinstance(self.bbox_xyxy, Sequence) or isinstance(
            self.bbox_xyxy, (str, bytes, bytearray)
        ):
            raise _contract_failure("bbox_xyxy must be a four-number sequence")
        if len(self.bbox_xyxy) != 4:
            raise _contract_failure("bbox_xyxy must contain exactly four numbers")
        bbox = tuple(
            _require_finite_number(item, f"bbox_xyxy[{index}]")
            for index, item in enumerate(self.bbox_xyxy)
        )
        x_min, y_min, x_max, y_max = bbox
        if not (x_min < x_max <= self.image_width):
            raise _contract_failure(
                "bbox x coordinates must satisfy 0 <= x_min < x_max <= image_width"
            )
        if not (y_min < y_max <= self.image_height):
            raise _contract_failure(
                "bbox y coordinates must satisfy 0 <= y_min < y_max <= image_height"
            )
        object.__setattr__(self, "bbox_xyxy", bbox)
        if not isinstance(self.attributes, Mapping):
            raise _contract_failure("attributes must be an object")
        copied_attributes = dict(self.attributes)
        _reject_privileged_keys(copied_attributes)
        for key, value in copied_attributes.items():
            _require_non_blank(key, "attributes key", 256)
            if not isinstance(value, (str, bool, int, float, type(None))):
                raise _contract_failure(
                    "detection attributes are limited to flat scalar values"
                )
            if isinstance(value, float) and not isfinite(value):
                raise _contract_failure(f"detection attribute {key!r} must be finite")
        object.__setattr__(self, "attributes", MappingProxyType(copied_attributes))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Detection":
        if not isinstance(value, Mapping):
            raise _contract_failure("detection must be an object")
        required = frozenset(
            {
                "detection_id",
                "class_id",
                "class_name",
                "confidence",
                "bbox_xyxy",
                "camera_id",
                "image_width",
                "image_height",
                "bbox_format",
            }
        )
        optional = frozenset({"track_id", "zone_id", "attributes"})
        _require_exact_fields(
            value,
            required=required,
            optional=optional,
            object_name="detection",
        )
        raw_bbox = value["bbox_xyxy"]
        if not isinstance(raw_bbox, Sequence) or isinstance(
            raw_bbox, (str, bytes, bytearray)
        ):
            raise _contract_failure("bbox_xyxy must be a four-number sequence")
        return cls(
            detection_id=value["detection_id"],
            class_id=value["class_id"],
            class_name=value["class_name"],
            confidence=value["confidence"],
            bbox_xyxy=tuple(raw_bbox),  # type: ignore[arg-type]
            camera_id=value["camera_id"],
            image_width=value["image_width"],
            image_height=value["image_height"],
            track_id=value.get("track_id"),
            zone_id=value.get("zone_id"),
            attributes=value.get("attributes", {}),
            bbox_format=value["bbox_format"],
        )

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "detection_id": self.detection_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "confidence": self.confidence,
            "bbox_xyxy": list(self.bbox_xyxy),
            "bbox_format": "xyxy_pixels",
            "camera_id": self.camera_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "attributes": dict(self.attributes),
        }
        if self.track_id is not None:
            result["track_id"] = self.track_id
        if self.zone_id is not None:
            result["zone_id"] = self.zone_id
        return result


@dataclass(frozen=True)
class PerceptionTiming:
    preprocess_ms: float
    inference_ms: float
    nms_ms: float
    total_ms: float

    def __post_init__(self) -> None:
        values = {
            name: _require_finite_number(value, name)
            for name, value in (
                ("preprocess_ms", self.preprocess_ms),
                ("inference_ms", self.inference_ms),
                ("nms_ms", self.nms_ms),
                ("total_ms", self.total_ms),
            )
        }
        for name, value in values.items():
            object.__setattr__(self, name, value)
        if self.total_ms < max(self.preprocess_ms, self.inference_ms, self.nms_ms):
            raise _contract_failure(
                "total_ms cannot be less than an individual timing component"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PerceptionTiming":
        if not isinstance(value, Mapping):
            raise _contract_failure("timing must be an object")
        required = frozenset({"preprocess_ms", "inference_ms", "nms_ms", "total_ms"})
        _require_exact_fields(value, required=required, object_name="perception timing")
        return cls(
            preprocess_ms=value["preprocess_ms"],
            inference_ms=value["inference_ms"],
            nms_ms=value["nms_ms"],
            total_ms=value["total_ms"],
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "preprocess_ms": self.preprocess_ms,
            "inference_ms": self.inference_ms,
            "nms_ms": self.nms_ms,
            "total_ms": self.total_ms,
        }


@dataclass(frozen=True)
class DetectionPacket:
    """Canonical, auditable YOLO output for exactly one immutable frame."""

    packet_id: str
    request_id: str
    trace_id: str
    episode_id: str
    task_id: str
    subtask_id: str
    step_id: int
    observation_id: str
    image_sha256: str
    camera_id: str
    image_width: int
    image_height: int
    checkpoint_sha: str
    class_map_sha: str
    config_sha: str
    detections: tuple[Detection, ...]
    timing: PerceptionTiming
    schema_version: str = PERCEPTION_SCHEMA_VERSION
    detection_contract_version: str = DETECTION_CONTRACT_VERSION

    def __post_init__(self) -> None:
        _require_version(self.schema_version, "schema_version")
        _require_version(self.detection_contract_version, "detection_contract_version")
        for field_name in (
            "packet_id",
            "request_id",
            "trace_id",
            "episode_id",
            "task_id",
            "subtask_id",
            "observation_id",
            "camera_id",
        ):
            _require_non_blank(getattr(self, field_name), field_name, 512)
        _require_strict_int(self.step_id, "step_id")
        _require_digest(self.image_sha256, "image_sha256")
        _require_digest(self.checkpoint_sha, "checkpoint_sha")
        _require_digest(self.class_map_sha, "class_map_sha")
        _require_digest(self.config_sha, "config_sha")
        _require_strict_int(self.image_width, "image_width", minimum=1, maximum=100_000)
        _require_strict_int(
            self.image_height, "image_height", minimum=1, maximum=100_000
        )
        if not isinstance(self.timing, PerceptionTiming):
            raise _contract_failure("timing must be a PerceptionTiming")
        detections = tuple(self.detections)
        if len(detections) > 10_000:
            raise _contract_failure("detections cannot exceed 10000 items")
        identifiers: set[str] = set()
        for detection in detections:
            if not isinstance(detection, Detection):
                raise _contract_failure("detections must contain Detection values")
            if detection.detection_id in identifiers:
                raise _contract_failure(
                    f"duplicate detection_id: {detection.detection_id}"
                )
            identifiers.add(detection.detection_id)
            if (
                detection.camera_id != self.camera_id
                or detection.image_width != self.image_width
                or detection.image_height != self.image_height
            ):
                raise _contract_failure(
                    "each detection must match packet camera and image dimensions"
                )
        object.__setattr__(self, "detections", detections)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DetectionPacket":
        if not isinstance(value, Mapping):
            raise _contract_failure("detection_packet must be an object")
        required = frozenset(
            {
                "schema_version",
                "detection_contract_version",
                "packet_id",
                "request_id",
                "trace_id",
                "episode_id",
                "task_id",
                "subtask_id",
                "step_id",
                "observation_id",
                "image_sha256",
                "camera_id",
                "image_width",
                "image_height",
                "checkpoint_sha",
                "class_map_sha",
                "config_sha",
                "detections",
                "timing",
            }
        )
        _require_exact_fields(value, required=required, object_name="detection_packet")
        raw_detections = value["detections"]
        if not isinstance(raw_detections, Sequence) or isinstance(
            raw_detections, (str, bytes, bytearray)
        ):
            raise _contract_failure("detections must be an array")
        return cls(
            schema_version=value["schema_version"],
            detection_contract_version=value["detection_contract_version"],
            packet_id=value["packet_id"],
            request_id=value["request_id"],
            trace_id=value["trace_id"],
            episode_id=value["episode_id"],
            task_id=value["task_id"],
            subtask_id=value["subtask_id"],
            step_id=value["step_id"],
            observation_id=value["observation_id"],
            image_sha256=value["image_sha256"],
            camera_id=value["camera_id"],
            image_width=value["image_width"],
            image_height=value["image_height"],
            checkpoint_sha=value["checkpoint_sha"],
            class_map_sha=value["class_map_sha"],
            config_sha=value["config_sha"],
            detections=tuple(Detection.from_dict(item) for item in raw_detections),
            timing=PerceptionTiming.from_dict(value["timing"]),
        )

    def validate_against(
        self,
        *,
        observation_id: str,
        image: ImageReference,
        descriptor: PerceptionDescriptor,
    ) -> None:
        """Fail closed if a response belongs to another frame or deployment."""

        expected = {
            "observation_id": observation_id,
            "image_sha256": image.image_sha256,
            "camera_id": image.camera_id,
            "image_width": image.width,
            "image_height": image.height,
            "checkpoint_sha": descriptor.checkpoint_sha,
            "class_map_sha": descriptor.class_map_sha,
            "config_sha": descriptor.config_sha,
            "detection_contract_version": descriptor.detection_contract_version,
        }
        mismatches = {
            key: {"expected": expected_value, "actual": getattr(self, key)}
            for key, expected_value in expected.items()
            if getattr(self, key) != expected_value
        }
        if mismatches:
            raise _contract_failure(
                f"detection packet frame/deployment mismatch: {mismatches}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "detection_contract_version": self.detection_contract_version,
            "packet_id": self.packet_id,
            "request_id": self.request_id,
            "trace_id": self.trace_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "subtask_id": self.subtask_id,
            "step_id": self.step_id,
            "observation_id": self.observation_id,
            "image_sha256": self.image_sha256,
            "camera_id": self.camera_id,
            "image_width": self.image_width,
            "image_height": self.image_height,
            "checkpoint_sha": self.checkpoint_sha,
            "class_map_sha": self.class_map_sha,
            "config_sha": self.config_sha,
            "detections": [item.to_dict() for item in self.detections],
            "timing": self.timing.to_dict(),
        }


@dataclass(frozen=True)
class CocoExportManifest:
    """Frozen bridge from online identities to COCO dataset identities."""

    class_map_sha: str
    image_id_by_frame_key: Mapping[tuple[str, str, str], int]
    image_sha_by_frame_key: Mapping[tuple[str, str, str], str]
    category_id_by_class_name: Mapping[str, int]

    def __post_init__(self) -> None:
        _require_digest(self.class_map_sha, "manifest.class_map_sha")
        image_ids = dict(self.image_id_by_frame_key)
        image_shas = dict(self.image_sha_by_frame_key)
        category_ids = dict(self.category_id_by_class_name)
        if not image_ids or set(image_ids) != set(image_shas):
            raise _contract_failure(
                "manifest image-id and image-SHA frame keys must match"
            )
        if not category_ids:
            raise _contract_failure("manifest categories cannot be empty")
        for frame_key, image_id in image_ids.items():
            if not isinstance(frame_key, tuple) or len(frame_key) != 3:
                raise _contract_failure(
                    "manifest frame key must be (trace_id, observation_id, camera_id)"
                )
            trace_id, observation_id, camera_id = frame_key
            _require_non_blank(trace_id, "manifest.trace_id", 256)
            _require_non_blank(observation_id, "manifest.observation_id", 256)
            _require_non_blank(camera_id, "manifest.camera_id", 256)
            _require_strict_int(image_id, "manifest.image_id")
            _require_digest(
                image_shas[frame_key],
                f"manifest image {frame_key!r} SHA",
            )
        if len(set(image_ids.values())) != len(image_ids):
            raise _contract_failure("manifest image_id values must be unique")
        for class_name, category_id in category_ids.items():
            _require_non_blank(class_name, "manifest.class_name", 256)
            _require_strict_int(category_id, "manifest.category_id")
        if len(set(category_ids.values())) != len(category_ids):
            raise _contract_failure("manifest category_id values must be unique")
        object.__setattr__(
            self,
            "image_id_by_frame_key",
            MappingProxyType(image_ids),
        )
        object.__setattr__(
            self,
            "image_sha_by_frame_key",
            MappingProxyType(image_shas),
        )
        object.__setattr__(
            self,
            "category_id_by_class_name",
            MappingProxyType(category_ids),
        )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CocoExportManifest":
        if not isinstance(value, Mapping):
            raise _contract_failure("COCO export manifest must be an object")
        _require_exact_fields(
            value,
            required=frozenset(
                {"schema_version", "class_map_sha", "images", "categories"}
            ),
            object_name="COCO export manifest",
        )
        _require_version(value["schema_version"], "manifest.schema_version")
        raw_images = value["images"]
        raw_categories = value["categories"]
        if not isinstance(raw_images, Sequence) or isinstance(
            raw_images, (str, bytes, bytearray)
        ):
            raise _contract_failure("manifest.images must be an array")
        if not isinstance(raw_categories, Sequence) or isinstance(
            raw_categories, (str, bytes, bytearray)
        ):
            raise _contract_failure("manifest.categories must be an array")
        image_ids: dict[tuple[str, str, str], int] = {}
        image_shas: dict[tuple[str, str, str], str] = {}
        for index, raw_image in enumerate(raw_images):
            if not isinstance(raw_image, Mapping):
                raise _contract_failure(f"manifest.images[{index}] must be an object")
            _require_exact_fields(
                raw_image,
                required=frozenset(
                    {
                        "trace_id",
                        "observation_id",
                        "camera_id",
                        "image_sha256",
                        "image_id",
                    }
                ),
                object_name=f"manifest.images[{index}]",
            )
            trace_id = _require_non_blank(
                raw_image["trace_id"],
                f"manifest.images[{index}].trace_id",
                256,
            )
            observation_id = _require_non_blank(
                raw_image["observation_id"],
                f"manifest.images[{index}].observation_id",
                256,
            )
            camera_id = _require_non_blank(
                raw_image["camera_id"],
                f"manifest.images[{index}].camera_id",
                256,
            )
            frame_key = (trace_id, observation_id, camera_id)
            if frame_key in image_ids:
                raise _contract_failure(
                    f"duplicate manifest frame identity: {frame_key!r}"
                )
            image_ids[frame_key] = _require_strict_int(
                raw_image["image_id"],
                f"manifest.images[{index}].image_id",
            )
            image_shas[frame_key] = _require_digest(
                raw_image["image_sha256"],
                f"manifest.images[{index}].image_sha256",
            )
        category_ids: dict[str, int] = {}
        for index, raw_category in enumerate(raw_categories):
            if not isinstance(raw_category, Mapping):
                raise _contract_failure(
                    f"manifest.categories[{index}] must be an object"
                )
            _require_exact_fields(
                raw_category,
                required=frozenset({"class_name", "category_id"}),
                object_name=f"manifest.categories[{index}]",
            )
            class_name = _require_non_blank(
                raw_category["class_name"],
                f"manifest.categories[{index}].class_name",
                256,
            )
            if class_name in category_ids:
                raise _contract_failure(
                    f"duplicate manifest class_name: {class_name!r}"
                )
            category_ids[class_name] = _require_strict_int(
                raw_category["category_id"],
                f"manifest.categories[{index}].category_id",
            )
        return cls(
            class_map_sha=value["class_map_sha"],
            image_id_by_frame_key=image_ids,
            image_sha_by_frame_key=image_shas,
            category_id_by_class_name=category_ids,
        )


class DetectionEvidenceSink:
    """Append-only online YOLO evidence with an offline COCO export.

    Records contain only online predictions, correlation identifiers, timing,
    and immutable model/frame identities. Ground truth is intentionally absent
    and is supplied only to the separate offline evaluator.
    """

    EVIDENCE_SCHEMA_VERSION = "1.0"

    def __init__(self, jsonl_path: str | Path | None = None):
        self.records: list[dict[str, Any]] = []
        self._path = Path(jsonl_path) if jsonl_path is not None else None
        self._lock = Lock()

    def _append(self, record: Mapping[str, Any]) -> dict[str, Any]:
        copied = dict(record)
        with self._lock:
            self.records.append(copied)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            copied,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
        return copied

    def record_packet(
        self,
        packet: DetectionPacket,
        *,
        mode: PerceptionMode = PerceptionMode.SHADOW_SCORE,
    ) -> dict[str, Any]:
        """Persist a validated packet, including a valid zero-detection frame."""

        if not isinstance(packet, DetectionPacket):
            raise TypeError("packet must be a DetectionPacket")
        return self._append(
            {
                "evidence_schema_version": self.EVIDENCE_SCHEMA_VERSION,
                "record_type": "detection_packet",
                "perception_mode": mode.value,
                **packet.to_dict(),
            }
        )

    def record_failure(
        self,
        *,
        trace_id: str,
        task_id: str,
        subtask_id: str,
        step_id: int,
        observation_id: str,
        image: ImageReference | None,
        descriptor: PerceptionDescriptor,
        failure_code: FailureCode,
        message: str,
        mode: PerceptionMode = PerceptionMode.SHADOW_SCORE,
    ) -> dict[str, Any]:
        """Persist a failed score-sidecar attempt without changing control state."""

        return self._append(
            {
                "evidence_schema_version": self.EVIDENCE_SCHEMA_VERSION,
                "record_type": "detection_failure",
                "perception_mode": mode.value,
                "trace_id": trace_id,
                "episode_id": trace_id,
                "task_id": task_id,
                "subtask_id": subtask_id,
                "step_id": step_id,
                "observation_id": observation_id,
                "image_sha256": (image.image_sha256 if image is not None else None),
                "camera_id": image.camera_id if image is not None else None,
                "image_width": image.width if image is not None else None,
                "image_height": image.height if image is not None else None,
                "checkpoint_sha": descriptor.checkpoint_sha,
                "class_map_sha": descriptor.class_map_sha,
                "config_sha": descriptor.config_sha,
                "detections": [],
                "timing": None,
                "failure_code": failure_code.value,
                "message": message,
            }
        )

    @staticmethod
    def _coco_envelope(
        records: Sequence[Mapping[str, Any]],
        manifest: CocoExportManifest,
    ) -> dict[str, Any]:
        predictions: list[dict[str, Any]] = []
        frame_latencies: list[dict[str, Any]] = []
        frames: list[dict[str, Any]] = []
        failed_frames: list[dict[str, Any]] = []
        for record in records:
            record_type = record.get("record_type")
            if record_type == "detection_failure":
                failed_frames.append(dict(record))
                continue
            if record_type != "detection_packet":
                continue
            frame = dict(record)
            trace_id = frame.get("trace_id")
            observation_id = frame.get("observation_id")
            camera_id = frame.get("camera_id")
            if not all(
                isinstance(value, str) and value
                for value in (trace_id, observation_id, camera_id)
            ):
                raise _contract_failure(
                    "detection evidence lacks a complete "
                    "(trace_id, observation_id, camera_id) identity"
                )
            frame_key = (trace_id, observation_id, camera_id)
            if frame_key not in manifest.image_id_by_frame_key:
                raise _contract_failure(
                    f"capture manifest has no image_id for {frame_key!r}"
                )
            if frame.get("image_sha256") != manifest.image_sha_by_frame_key[frame_key]:
                raise _contract_failure(
                    f"capture manifest image SHA mismatch for {frame_key!r}"
                )
            if frame.get("class_map_sha") != manifest.class_map_sha:
                raise _contract_failure(
                    f"class-map SHA mismatch for {observation_id!r}"
                )
            image_id = manifest.image_id_by_frame_key[frame_key]
            frames.append(frame)
            timing = frame.get("timing")
            total_ms = timing.get("total_ms") if isinstance(timing, Mapping) else None
            frame_latencies.append(
                {
                    "trace_id": frame.get("trace_id"),
                    "observation_id": frame.get("observation_id"),
                    "image_id": image_id,
                    "latency_ms": total_ms,
                }
            )
            raw_detections = frame.get("detections", ())
            if not isinstance(raw_detections, Sequence) or isinstance(
                raw_detections, (str, bytes, bytearray)
            ):
                continue
            for detection in raw_detections:
                if not isinstance(detection, Mapping):
                    continue
                bbox = detection.get("bbox_xyxy")
                if (
                    not isinstance(bbox, Sequence)
                    or isinstance(bbox, (str, bytes, bytearray))
                    or len(bbox) != 4
                ):
                    continue
                x_min, y_min, x_max, y_max = bbox
                class_name = detection.get("class_name")
                if (
                    not isinstance(class_name, str)
                    or class_name not in manifest.category_id_by_class_name
                ):
                    raise _contract_failure(
                        f"class map has no COCO category for {class_name!r}"
                    )
                predictions.append(
                    {
                        "image_id": image_id,
                        "category_id": manifest.category_id_by_class_name[class_name],
                        "bbox": [
                            x_min,
                            y_min,
                            x_max - x_min,
                            y_max - y_min,
                        ],
                        "score": detection.get("confidence"),
                        "trace_id": frame.get("trace_id"),
                        "observation_id": frame.get("observation_id"),
                        "image_sha256": frame.get("image_sha256"),
                        "latency_ms": total_ms,
                        "packet_id": frame.get("packet_id"),
                        "detection_id": detection.get("detection_id"),
                        "class_name": detection.get("class_name"),
                        "checkpoint_sha": frame.get("checkpoint_sha"),
                        "class_map_sha": frame.get("class_map_sha"),
                        "config_sha": frame.get("config_sha"),
                    }
                )
        return {
            "evidence_schema_version": DetectionEvidenceSink.EVIDENCE_SCHEMA_VERSION,
            "predictions": predictions,
            "frame_latencies": frame_latencies,
            "frames": frames,
            "failed_frames": failed_frames,
        }

    def export_coco_predictions(
        self,
        output_path: str | Path,
        *,
        manifest: CocoExportManifest | Mapping[str, Any],
    ) -> Path:
        """Export an evaluator-ready envelope while preserving raw frame packets."""

        destination = Path(output_path)
        resolved_manifest = (
            manifest
            if isinstance(manifest, CocoExportManifest)
            else CocoExportManifest.from_dict(manifest)
        )
        with self._lock:
            envelope = self._coco_envelope(tuple(self.records), resolved_manifest)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(
            json.dumps(envelope, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return destination


class PerceptionError(AgentError):
    """Typed transport/service failure for the independent perception Agent."""

    def __init__(
        self,
        code: FailureCode,
        message: str,
        *,
        retryable: bool = False,
        retry_after_ms: int | None = None,
    ):
        super().__init__(code, message)
        self.retryable = retryable
        self.retry_after_ms = retry_after_ms


@runtime_checkable
class PerceptionAgent(Protocol):
    descriptor: PerceptionDescriptor

    def health(self) -> bool:
        """Return readiness without loading YOLO in the supervisor process."""

    def detect(
        self,
        context: PerceptionContext,
    ) -> DetectionPacket:
        """Detect objects using only the immutable image and trace context."""

    def cancel(self, task_id: str, reason: str) -> None:
        """Best-effort cancellation of outstanding detector work."""


@runtime_checkable
class PerceptionTransport(Protocol):
    def request(
        self, route: str, payload: Mapping[str, Any], timeout_ms: int
    ) -> Mapping[str, Any]: ...


def _response_correlation(
    response: Mapping[str, Any],
    *,
    payload: Mapping[str, Any],
    descriptor: PerceptionDescriptor,
) -> None:
    expected = {
        key: payload[key]
        for key in (
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
    expected["detector"] = descriptor.name
    mismatches = {
        key: {"expected": value, "actual": response.get(key)}
        for key, value in expected.items()
        if response.get(key) != value
    }
    if mismatches:
        code = (
            FailureCode.PERCEPTION_REVISION_MISMATCH
            if {"checkpoint_sha", "class_map_sha", "config_sha"} & set(mismatches)
            else FailureCode.PERCEPTION_BAD_RESPONSE
        )
        raise PerceptionError(
            code,
            f"YOLO response correlation mismatch: {mismatches}",
        )


class YoloHTTPAdapter:
    """Strict client for a separately deployed YOLO Agent service."""

    def __init__(
        self,
        transport: PerceptionTransport,
        *,
        checkpoint_sha: str,
        class_map_sha: str,
        config_sha: str,
        task_types: frozenset[str] | None = None,
    ):
        self.transport = transport
        self.descriptor = PerceptionDescriptor(
            name="yolo",
            task_types=task_types
            or frozenset(
                {
                    "pick_place",
                    "object_localization",
                    "visual_manipulation",
                    "instruction_interaction",
                    "mock_demo",
                }
            ),
            detection_contract_version=DETECTION_CONTRACT_VERSION,
            checkpoint_sha=checkpoint_sha,
            class_map_sha=class_map_sha,
            config_sha=config_sha,
        )
        self._cancel_context_by_task: dict[str, tuple[str, str]] = {}

    def health(self) -> bool:
        try:
            response = self.transport.request("/health", {}, 1_000)
            if not isinstance(response, Mapping):
                return False
            expected = {
                "schema_version": PERCEPTION_SCHEMA_VERSION,
                "service": self.descriptor.name,
                "status": "ready",
                "checkpoint_sha": self.descriptor.checkpoint_sha,
                "class_map_sha": self.descriptor.class_map_sha,
                "config_sha": self.descriptor.config_sha,
            }
            if any(response.get(key) != value for key, value in expected.items()):
                return False
            contracts = response.get("supported_detection_contracts")
            task_types = response.get("supported_task_types")
            return (
                isinstance(contracts, Sequence)
                and not isinstance(contracts, (str, bytes, bytearray))
                and self.descriptor.detection_contract_version in contracts
                and isinstance(task_types, Sequence)
                and not isinstance(task_types, (str, bytes, bytearray))
                and self.descriptor.task_types.issubset(set(task_types))
            )
        except Exception:
            return False

    def detect(
        self,
        context: PerceptionContext,
    ) -> DetectionPacket:
        request_id = str(uuid4())
        self._cancel_context_by_task[context.task_id] = (
            context.run_id,
            context.subtask_id,
        )
        payload: dict[str, Any] = {
            "schema_version": PERCEPTION_SCHEMA_VERSION,
            "request_id": request_id,
            "trace_id": context.run_id,
            "episode_id": context.run_id,
            "task_id": context.task_id,
            "subtask_id": context.subtask_id,
            "step_id": context.step_id,
            "observation_id": context.observation_id,
            "image_sha256": context.image.image_sha256,
            "deadline_ms": context.timeout_ms,
            "detector": self.descriptor.name,
            "checkpoint_sha": self.descriptor.checkpoint_sha,
            "class_map_sha": self.descriptor.class_map_sha,
            "config_sha": self.descriptor.config_sha,
            "expected_detection_contract": (self.descriptor.detection_contract_version),
            "image": context.image.to_dict(),
            "allowed_class_names": list(context.allowed_class_names),
            "thresholds": {
                "confidence": context.confidence_threshold,
                "iou": context.iou_threshold,
            },
        }
        try:
            response = self.transport.request("/v1/detect", payload, context.timeout_ms)
        except TimeoutError as exc:
            raise PerceptionError(
                FailureCode.PERCEPTION_TIMEOUT,
                "YOLO detection timed out",
                retryable=True,
            ) from exc
        except PerceptionError:
            raise
        except Exception as exc:
            raise PerceptionError(
                FailureCode.PERCEPTION_UNAVAILABLE,
                f"YOLO transport failed: {exc}",
                retryable=True,
            ) from exc
        if not isinstance(response, Mapping):
            raise PerceptionError(
                FailureCode.PERCEPTION_BAD_RESPONSE,
                "YOLO response must be an object",
            )
        required = frozenset(
            {
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
                "status",
            }
        )
        optional = frozenset({"detection_packet", "error"})
        missing = required - set(response)
        unknown = set(response) - required - optional
        if missing or unknown:
            raise PerceptionError(
                FailureCode.PERCEPTION_BAD_RESPONSE,
                f"YOLO response fields invalid; "
                f"missing={sorted(missing)}, unknown={sorted(unknown)}",
            )
        if response.get("schema_version") != PERCEPTION_SCHEMA_VERSION:
            raise PerceptionError(
                FailureCode.PERCEPTION_BAD_RESPONSE,
                "YOLO response schema_version is incompatible",
            )
        _response_correlation(response, payload=payload, descriptor=self.descriptor)
        status = response.get("status")
        if status != "ok":
            raw_error = response.get("error")
            if status not in {"error", "cancelled"} or not isinstance(
                raw_error, Mapping
            ):
                raise PerceptionError(
                    FailureCode.PERCEPTION_BAD_RESPONSE,
                    "non-ok YOLO response requires a structured error",
                )
            try:
                code = FailureCode(str(raw_error.get("code")))
            except ValueError as exc:
                raise PerceptionError(
                    FailureCode.PERCEPTION_BAD_RESPONSE,
                    "YOLO response contains an unknown failure code",
                ) from exc
            retryable = raw_error.get("retryable", False)
            retry_after_ms = raw_error.get("retry_after_ms")
            if not isinstance(retryable, bool) or (
                retry_after_ms is not None
                and (
                    isinstance(retry_after_ms, bool)
                    or not isinstance(retry_after_ms, int)
                    or retry_after_ms < 0
                )
            ):
                raise PerceptionError(
                    FailureCode.PERCEPTION_BAD_RESPONSE,
                    "YOLO error retry metadata is invalid",
                )
            raise PerceptionError(
                code,
                str(raw_error.get("message", "YOLO service error")),
                retryable=retryable,
                retry_after_ms=retry_after_ms,
            )
        if "error" in response or not isinstance(
            response.get("detection_packet"), Mapping
        ):
            raise PerceptionError(
                FailureCode.PERCEPTION_BAD_RESPONSE,
                "ok YOLO response requires only detection_packet",
            )
        try:
            packet = DetectionPacket.from_dict(response["detection_packet"])
            revision_mismatches = {
                key: {
                    "expected": getattr(self.descriptor, key),
                    "actual": getattr(packet, key),
                }
                for key in (
                    "checkpoint_sha",
                    "class_map_sha",
                    "config_sha",
                    "detection_contract_version",
                )
                if getattr(packet, key) != getattr(self.descriptor, key)
            }
            if revision_mismatches:
                raise PerceptionError(
                    FailureCode.PERCEPTION_REVISION_MISMATCH,
                    f"YOLO packet deployment mismatch: {revision_mismatches}",
                )
            packet.validate_against(
                observation_id=context.observation_id,
                image=context.image,
                descriptor=self.descriptor,
            )
        except PerceptionError:
            raise
        except (ContractError, ValueError) as exc:
            raise PerceptionError(
                FailureCode.PERCEPTION_BAD_RESPONSE,
                f"invalid YOLO detection packet: {exc}",
            ) from exc
        packet_correlation = {
            "request_id": request_id,
            "trace_id": context.run_id,
            "episode_id": context.run_id,
            "task_id": context.task_id,
            "subtask_id": context.subtask_id,
            "step_id": context.step_id,
        }
        if any(
            getattr(packet, key) != value for key, value in packet_correlation.items()
        ):
            raise PerceptionError(
                FailureCode.PERCEPTION_BAD_RESPONSE,
                "detection packet request/task correlation mismatch",
            )
        return packet

    def cancel(self, task_id: str, reason: str) -> None:
        trace_id, subtask_id = self._cancel_context_by_task.get(
            task_id, (task_id, task_id)
        )
        try:
            self.transport.request(
                "/v1/cancel",
                {
                    "schema_version": PERCEPTION_SCHEMA_VERSION,
                    "request_id": str(uuid4()),
                    "trace_id": trace_id,
                    "episode_id": trace_id,
                    "task_id": task_id,
                    "subtask_id": subtask_id,
                    "reason": reason,
                },
                1_000,
            )
        except Exception:
            return


class MockPerceptionAgent:
    """Deterministic in-process test double; production uses ``YoloHTTPAdapter``."""

    def __init__(
        self,
        *,
        checkpoint_sha: str,
        class_map_sha: str,
        config_sha: str,
        detector: Callable[[PerceptionContext], Sequence[Detection]] | None = None,
    ):
        self.descriptor = PerceptionDescriptor(
            name="yolo",
            task_types=frozenset(
                {
                    "pick_place",
                    "object_localization",
                    "visual_manipulation",
                    "instruction_interaction",
                    "mock_demo",
                }
            ),
            detection_contract_version=DETECTION_CONTRACT_VERSION,
            checkpoint_sha=checkpoint_sha,
            class_map_sha=class_map_sha,
            config_sha=config_sha,
        )
        self._detector = detector or (lambda context: ())
        self.calls: list[tuple[str, str]] = []

    def health(self) -> bool:
        return True

    def detect(
        self,
        context: PerceptionContext,
    ) -> DetectionPacket:
        self.calls.append((context.task_id, context.observation_id))
        detections = tuple(self._detector(context))
        packet = DetectionPacket(
            packet_id=str(uuid4()),
            request_id=str(uuid4()),
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
            detections=detections,
            timing=PerceptionTiming(0.0, 0.0, 0.0, 0.0),
        )
        packet.validate_against(
            observation_id=context.observation_id,
            image=context.image,
            descriptor=self.descriptor,
        )
        return packet

    def cancel(self, task_id: str, reason: str) -> None:
        return


def build_perception_from_config(
    config: Mapping[str, Any],
    transport_factory: Callable[[str, str], PerceptionTransport],
) -> YoloHTTPAdapter:
    """Build the mandatory YOLO service adapter from the versioned Agent config.

    The transport factory receives ``("yolo", base_url)``. Artifact placeholders
    are accepted by the JSON template but are rejected here before a run starts.
    """

    raw = config.get("perception")
    if not isinstance(raw, Mapping):
        raise ValueError("config.perception must be an object")
    if raw.get("required") is not True:
        raise ValueError("config.perception.required must remain true")
    base_url = raw.get("base_url")
    if not isinstance(base_url, str) or not base_url.startswith(
        ("http://", "https://")
    ):
        raise ValueError("config.perception.base_url must be an HTTP(S) URL")

    artifact_values: dict[str, str] = {}
    for field_name in ("checkpoint_sha", "class_map_sha", "config_sha"):
        value = raw.get(field_name)
        if value == "REPLACE_WITH_PINNED_SHA":
            raise ValueError(
                f"config.perception.{field_name} is still an unsafe placeholder"
            )
        if not is_pinned_artifact_digest(value):
            raise ValueError(
                f"config.perception.{field_name} must match "
                "'sha256:<64 hexadecimal characters>'"
            )
        artifact_values[field_name] = value

    return YoloHTTPAdapter(
        transport_factory("yolo", base_url),
        checkpoint_sha=artifact_values["checkpoint_sha"],
        class_map_sha=artifact_values["class_map_sha"],
        config_sha=artifact_values["config_sha"],
    )
