"""Resolve V2 named targets into YOLO class filters and locked detections."""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from types import MappingProxyType
from typing import Mapping

from .perception import Detection, DetectionPacket, PerceptionContext
from .v2_task_profile import V2TaskSpec, v2_task, v2_task_for_instruction


V2_TARGET_CATALOG_ID = "single_bin_manual_industrial_v2"
_OBJECT_ID_PATTERN = re.compile(r"\b(?:P|N|W)\d{2}\b|\bBin_01\b", re.IGNORECASE)
_SLOT_ID_PATTERN = re.compile(r"\bS\d{2}\b", re.IGNORECASE)


@dataclass(frozen=True)
class V2ObjectSpec:
    object_id: str
    display_name: str
    part_type: str
    orientation_state: str
    class_name: str
    zone_id: str
    preferred_camera: str

    @property
    def allowed_class_names(self) -> tuple[str, ...]:
        return (self.class_name,)


@dataclass(frozen=True)
class V2SlotSpec:
    slot_id: str
    part_id: str
    profile: str
    slot_index: int
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True)
class V2ResolvedTarget:
    task_id: str
    target_object: V2ObjectSpec
    target_slot: V2SlotSpec | None
    instruction: str | None = None

    @property
    def allowed_class_names(self) -> tuple[str, ...]:
        return self.target_object.allowed_class_names


@dataclass(frozen=True)
class V2TargetLock:
    object_id: str
    slot_id: str | None
    slot_index: int | None
    class_name: str
    detection: Detection
    candidate_count: int
    selection_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "object_id": self.object_id,
            "slot_id": self.slot_id,
            "slot_index": self.slot_index,
            "class_name": self.class_name,
            "detection": self.detection.to_dict(),
            "candidate_count": self.candidate_count,
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class V2SlotLock:
    slot_id: str
    slot_index: int
    class_name: str
    detection: Detection
    selection_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "slot_id": self.slot_id,
            "slot_index": self.slot_index,
            "class_name": self.class_name,
            "detection": self.detection.to_dict(),
            "selection_reason": self.selection_reason,
        }


@dataclass(frozen=True)
class V2ImageRegion:
    region_id: str
    camera_id: str
    normalized_xyxy: tuple[float, float, float, float]

    def contains_detection_center(self, detection: Detection) -> bool:
        if detection.camera_id != self.camera_id:
            return False
        x1, y1, x2, y2 = detection.bbox_xyxy
        center_x = ((x1 + x2) / 2.0) / detection.image_width
        center_y = ((y1 + y2) / 2.0) / detection.image_height
        left, top, right, bottom = self.normalized_xyxy
        return left <= center_x < right and top <= center_y < bottom


_OBJECTS: tuple[V2ObjectSpec, ...] = (
    V2ObjectSpec("P01", "螺母", "nut", "flat", "hex_nut", "A", "CAM_A_TOP"),
    V2ObjectSpec(
        "P02", "正放轴件", "shaft", "upright", "shaft_upright", "A", "CAM_A_TOP"
    ),
    V2ObjectSpec(
        "P03", "倒放轴件", "shaft", "inverted", "shaft_inverted", "B", "CAM_A_TOP"
    ),
    V2ObjectSpec(
        "P04", "倒放轴件", "shaft", "inverted", "shaft_inverted", "B", "CAM_A_TOP"
    ),
    V2ObjectSpec(
        "N01", "正放轴件", "shaft", "upright", "shaft_upright", "C", "CAM_A_TOP"
    ),
    V2ObjectSpec("N02", "螺母", "nut", "flat", "hex_nut", "C", "CAM_A_TOP"),
    V2ObjectSpec(
        "W01", "开口扳手", "wrench", "flat_y", "open_end_wrench", "D", "CAM_A_TOP"
    ),
    V2ObjectSpec(
        "W02", "开口扳手", "wrench", "flat_y", "open_end_wrench", "D", "CAM_A_TOP"
    ),
    V2ObjectSpec(
        "Bin_01", "料箱", "bin", "upright", "bin_box", "PACK_STATION", "CAM_HANDOFF"
    ),
)

_SLOTS: tuple[V2SlotSpec, ...] = (
    V2SlotSpec("S11", "P01", "nut", 1, ("S01",)),
    V2SlotSpec("S12", "P03", "shaft", 2, ("S02",)),
    V2SlotSpec("S13", "N01", "shaft", 3, ("S03",)),
    V2SlotSpec("S14", "W01", "wrench_y", 4, ("S04",)),
    V2SlotSpec("S21", "P02", "shaft", 5, ("S05",)),
    V2SlotSpec("S22", "P04", "shaft", 6, ("S06",)),
    V2SlotSpec("S23", "N02", "nut", 7, ("S07",)),
    V2SlotSpec("S24", "W02", "wrench_y", 8, ("S08",)),
)

_OBJECT_ZONE_REGIONS: tuple[V2ImageRegion, ...] = (
    V2ImageRegion("A", "CAM_A_TOP", (0.00, 0.00, 0.50, 0.50)),
    V2ImageRegion("B", "CAM_A_TOP", (0.50, 0.00, 1.00, 0.50)),
    V2ImageRegion("C", "CAM_A_TOP", (0.00, 0.50, 0.50, 1.00)),
    V2ImageRegion("D", "CAM_A_TOP", (0.50, 0.50, 1.00, 1.00)),
    V2ImageRegion("PACK_STATION", "CAM_HANDOFF", (0.25, 0.25, 0.75, 0.75)),
)

_SLOT_REGIONS: tuple[V2ImageRegion, ...] = tuple(
    V2ImageRegion(
        slot.slot_id,
        "CAM_A_TOP",
        (
            0.25 + ((slot.slot_index - 1) % 4) * 0.125,
            0.30 + ((slot.slot_index - 1) // 4) * 0.20,
            0.25 + (((slot.slot_index - 1) % 4) + 1) * 0.125,
            0.30 + (((slot.slot_index - 1) // 4) + 1) * 0.20,
        ),
    )
    for slot in _SLOTS
)

V2_OBJECTS_BY_ID: Mapping[str, V2ObjectSpec] = MappingProxyType(
    {item.object_id: item for item in _OBJECTS}
)
V2_SLOTS_BY_ID: Mapping[str, V2SlotSpec] = MappingProxyType(
    {item.slot_id: item for item in _SLOTS}
)
V2_SLOT_ALIASES: Mapping[str, str] = MappingProxyType(
    {
        alias.casefold(): slot.slot_id
        for slot in _SLOTS
        for alias in (slot.slot_id, *slot.aliases)
    }
)
V2_OBJECT_ZONE_REGIONS: Mapping[str, V2ImageRegion] = MappingProxyType(
    {item.region_id: item for item in _OBJECT_ZONE_REGIONS}
)
V2_SLOT_REGIONS: Mapping[str, V2ImageRegion] = MappingProxyType(
    {item.region_id: item for item in _SLOT_REGIONS}
)


def v2_object(object_id: str) -> V2ObjectSpec:
    normalized = _normalize_object_id(object_id)
    try:
        return V2_OBJECTS_BY_ID[normalized]
    except KeyError as exc:
        raise ValueError(f"unknown V2 object id: {object_id!r}") from exc


def v2_slot(slot_id: str) -> V2SlotSpec:
    normalized = V2_SLOT_ALIASES.get(str(slot_id).casefold())
    if normalized is None:
        raise ValueError(f"unknown V2 slot id: {slot_id!r}")
    return V2_SLOTS_BY_ID[normalized]


def resolve_v2_target_for_task(task: str | V2TaskSpec) -> V2ResolvedTarget:
    spec = v2_task(task) if isinstance(task, str) else task
    target_slot = v2_slot(spec.target_slot) if spec.target_slot is not None else None
    return V2ResolvedTarget(
        task_id=spec.task_id,
        target_object=v2_object(spec.target_object),
        target_slot=target_slot,
        instruction=spec.instruction,
    )


def resolve_v2_target_instruction(instruction: str) -> V2ResolvedTarget:
    """Resolve a frozen or literal V2 instruction into a YOLO target filter."""

    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError("instruction must be a non-empty string")
    try:
        return resolve_v2_target_for_task(v2_task_for_instruction(instruction))
    except ValueError:
        pass

    object_match = _OBJECT_ID_PATTERN.search(instruction)
    slot_match = _SLOT_ID_PATTERN.search(instruction)
    if object_match is None:
        raise ValueError("instruction must mention a known V2 object id")
    target_object = v2_object(object_match.group(0))
    target_slot = v2_slot(slot_match.group(0)) if slot_match else None
    task_id = (
        f"{target_object.object_id}_TO_{target_slot.slot_id}"
        if target_slot is not None
        else target_object.object_id
    )
    return V2ResolvedTarget(
        task_id=task_id,
        target_object=target_object,
        target_slot=target_slot,
        instruction=instruction,
    )


def targeted_perception_context(
    context: PerceptionContext,
    target: V2ResolvedTarget | V2ObjectSpec | str,
) -> PerceptionContext:
    """Return a copy of ``context`` that asks YOLO for only the target class."""

    target_object = _target_object(target)
    return PerceptionContext(
        run_id=context.run_id,
        task_id=context.task_id,
        subtask_id=context.subtask_id,
        step_id=context.step_id,
        observation_id=context.observation_id,
        image=context.image,
        timeout_ms=context.timeout_ms,
        allowed_class_names=target_object.allowed_class_names,
        confidence_threshold=context.confidence_threshold,
        iou_threshold=context.iou_threshold,
    )


def select_target_detection(
    packet: DetectionPacket,
    target: V2ResolvedTarget | V2ObjectSpec | str,
    *,
    min_confidence: float = 0.0,
) -> V2TargetLock:
    """Lock one detection for the named V2 object."""

    target_object = _target_object(target)
    target_slot = target.target_slot if isinstance(target, V2ResolvedTarget) else None
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    candidates = tuple(
        _with_inferred_zone(detection)
        for detection in packet.detections
        if detection.class_name == target_object.class_name
        and detection.confidence >= min_confidence
    )
    if not candidates:
        raise ValueError(
            f"no detection matches {target_object.object_id} "
            f"as class {target_object.class_name!r}"
        )

    def score(detection: Detection) -> tuple[int, int, float, str]:
        zone_match = int(_effective_zone_id(detection) == target_object.zone_id)
        camera_match = int(detection.camera_id == target_object.preferred_camera)
        return (zone_match, camera_match, detection.confidence, detection.detection_id)

    selected = max(candidates, key=score)
    reason = (
        "matched target class"
        f" {target_object.class_name!r}, zone {target_object.zone_id!r},"
        f" camera {target_object.preferred_camera!r}"
    )
    return V2TargetLock(
        object_id=target_object.object_id,
        slot_id=target_slot.slot_id if target_slot is not None else None,
        slot_index=target_slot.slot_index if target_slot is not None else None,
        class_name=target_object.class_name,
        detection=selected,
        candidate_count=len(candidates),
        selection_reason=reason,
    )


def infer_v2_zone_id(detection: Detection) -> str | None:
    """Infer the fixed V2 object zone from a detection center."""

    for region in _OBJECT_ZONE_REGIONS:
        if region.contains_detection_center(detection):
            return region.region_id
    return None


def infer_v2_slot_id(detection: Detection) -> str | None:
    """Infer the fixed V2 bin slot from a slot detection center."""

    for region in _SLOT_REGIONS:
        if region.contains_detection_center(detection):
            return region.region_id
    return None


def select_target_slot_detection(
    packet: DetectionPacket,
    target: V2ResolvedTarget | V2SlotSpec | str,
    *,
    min_confidence: float = 0.0,
) -> V2SlotLock:
    """Lock one ``bin_slot`` detection to the requested fixed V2 slot."""

    target_slot = _target_slot(target)
    if not 0.0 <= min_confidence <= 1.0:
        raise ValueError("min_confidence must be between 0 and 1")
    candidates = tuple(
        detection
        for detection in packet.detections
        if detection.class_name == "bin_slot"
        and detection.confidence >= min_confidence
        and infer_v2_slot_id(detection) == target_slot.slot_id
    )
    if not candidates:
        raise ValueError(f"no detection matches V2 slot {target_slot.slot_id}")
    selected = max(candidates, key=lambda item: (item.confidence, item.detection_id))
    return V2SlotLock(
        slot_id=target_slot.slot_id,
        slot_index=target_slot.slot_index,
        class_name="bin_slot",
        detection=selected,
        selection_reason=(
            f"matched bin_slot center inside fixed slot {target_slot.slot_id}"
        ),
    )


def _target_object(target: V2ResolvedTarget | V2ObjectSpec | str) -> V2ObjectSpec:
    if isinstance(target, V2ResolvedTarget):
        return target.target_object
    if isinstance(target, V2ObjectSpec):
        return target
    return v2_object(target)


def _target_slot(target: V2ResolvedTarget | V2SlotSpec | str) -> V2SlotSpec:
    if isinstance(target, V2ResolvedTarget):
        if target.target_slot is None:
            raise ValueError(f"target {target.task_id!r} has no slot")
        return target.target_slot
    if isinstance(target, V2SlotSpec):
        return target
    return v2_slot(target)


def _effective_zone_id(detection: Detection) -> str | None:
    return detection.zone_id or infer_v2_zone_id(detection)


def _with_inferred_zone(detection: Detection) -> Detection:
    if detection.zone_id is not None:
        return detection
    zone_id = infer_v2_zone_id(detection)
    if zone_id is None:
        return detection
    return replace(detection, zone_id=zone_id)


def _normalize_object_id(object_id: str) -> str:
    raw = str(object_id)
    if raw.casefold() == "bin_01":
        return "Bin_01"
    return raw.upper()


__all__ = [
    "V2_OBJECTS_BY_ID",
    "V2_SLOT_ALIASES",
    "V2_SLOTS_BY_ID",
    "V2_TARGET_CATALOG_ID",
    "V2ImageRegion",
    "V2ObjectSpec",
    "V2ResolvedTarget",
    "V2SlotSpec",
    "V2SlotLock",
    "V2TargetLock",
    "infer_v2_slot_id",
    "infer_v2_zone_id",
    "resolve_v2_target_for_task",
    "resolve_v2_target_instruction",
    "select_target_detection",
    "select_target_slot_detection",
    "targeted_perception_context",
    "v2_object",
    "v2_slot",
]
