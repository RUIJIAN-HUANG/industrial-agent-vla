"""Sensor-only online task-state provider for the formal V2 tasks.

The provider consumes immutable camera references, correlated YOLO packets and
measured robot state.  It never reads Isaac stage/object poses at runtime.  The
scene geometry used below is fixed calibration data used only to project target
regions into camera pixels.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import sqrt, tan, pi
from threading import Lock
from typing import Any

from .perception import (
    Detection,
    DetectionPacket,
    ImageReference,
    PerceptionAgent,
    PerceptionContext,
)
from .v2_task_profile import V2TaskSpec, require_formal_v2_task


BIN_HANDOFF_TASK_ID = "BIN01_TO_FINISHED01"
CONTROL_TOKENS = frozenset({"A_ONLY", "HANDOFF_VERIFY", "B_ONLY", "NONE"})
_TASK_CLASS_NAMES = {
    "P01_TO_S11": ("shaft_upright",),
    "W01_TO_S14": ("open_end_wrench",),
    BIN_HANDOFF_TASK_ID: ("bin_box",),
}
_CAMERA_STREAM_BY_STAGE = {
    "S11": "arm_a_rgb",
    "S14": "arm_a_rgb",
    "HANDOFF_CENTER": "handoff_rgb",
    "FINISHED_01": "arm_b_rgb",
}


@dataclass(frozen=True)
class _Vote:
    passed: bool
    confidence: float


class OnlineTaskStateProvider:
    """Produce the exact seven-field V2 task state from fresh sensor frames."""

    def __init__(
        self,
        *,
        task_spec: V2TaskSpec,
        perception: PerceptionAgent,
        scene_config: Mapping[str, Any],
        run_id: str,
        verification_frames: int = 3,
        required_votes: int = 2,
        min_confidence: float = 0.6,
        timeout_ms: int = 5_000,
    ) -> None:
        expected = require_formal_v2_task(task_spec.task_id)
        if task_spec != expected:
            raise ValueError("task_spec does not match the frozen V2 catalog")
        if verification_frames != 3 or required_votes != 2:
            raise ValueError("formal online verification requires 3 frames and 2 votes")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError("min_confidence must be in [0, 1]")
        if timeout_ms < 1:
            raise ValueError("timeout_ms must be positive")
        descriptor = getattr(perception, "descriptor", None)
        if getattr(descriptor, "name", None) != "yolo":
            raise ValueError("online task state requires the YOLO perception Agent")
        if not run_id:
            raise ValueError("run_id is required")

        self.task_spec = task_spec
        self.perception = perception
        self.scene_config = dict(scene_config)
        self.run_id = run_id
        self.verification_frames = verification_frames
        self.required_votes = required_votes
        self.min_confidence = float(min_confidence)
        self.timeout_ms = int(timeout_ms)
        self._lock = Lock()
        self._seen_observation_ids: set[str] = set()
        self._last_timestamp_ms: int | None = None
        self._step_id = 0
        self._votes: deque[_Vote] = deque(maxlen=verification_frames)
        self._status = "ACTIVE"
        self._terminal = False
        self._terminal_confidence = 0.0
        self._verification_votes = 0
        self._token = "A_ONLY"
        self._last_error: str | None = None
        self._last_detection_packet: DetectionPacket | None = None

    def __call__(self) -> Mapping[str, Any]:
        return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return self._snapshot_unlocked()

    def control_token(self) -> str:
        with self._lock:
            return self._token

    def active_arm(self) -> str:
        token = self.control_token()
        if token == "A_ONLY":
            return "Arm_A"
        if token == "B_ONLY":
            return "Arm_B"
        return "NONE"

    @property
    def last_error(self) -> str | None:
        with self._lock:
            return self._last_error

    def update(
        self,
        *,
        observation_id: str,
        timestamp_ms: int,
        camera: Mapping[str, Any],
        robot: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Consume exactly one fresh synchronized observation and update state."""

        with self._lock:
            self._require_fresh(observation_id, timestamp_ms)
            if self._terminal:
                return self._snapshot_unlocked()

            target_id = self._current_target_id()
            self._last_detection_packet = None
            try:
                confidence = self._visual_confidence(
                    observation_id=observation_id,
                    camera=camera,
                    target_id=target_id,
                )
                self._last_error = None
            except Exception as exc:  # Fail closed: a detector failure is a no vote.
                confidence = 0.0
                self._last_error = f"{type(exc).__name__}: {exc}"

            passed = confidence >= self.min_confidence and self._robot_ready(robot)
            vote = _Vote(passed=passed, confidence=confidence if passed else 0.0)

            if self.task_spec.task_id == BIN_HANDOFF_TASK_ID:
                self._update_handoff(vote)
            else:
                self._votes.append(vote)
                self._publish_window(terminal_stage=True)
            return self._snapshot_unlocked()

    def latest_detection_view(self) -> dict[str, Any] | None:
        """Return the latest validated YOLO result for UI presentation.

        This is deliberately separate from :meth:`snapshot`, which is the
        frozen seven-field task-state contract consumed by the Supervisor.
        """

        with self._lock:
            packet = self._last_detection_packet
            if packet is None:
                return None
            return {
                "observation_id": packet.observation_id,
                "camera_id": packet.camera_id,
                "image_width": packet.image_width,
                "image_height": packet.image_height,
                "detections": [item.to_dict() for item in packet.detections],
            }

    def _require_fresh(self, observation_id: str, timestamp_ms: int) -> None:
        if not isinstance(observation_id, str) or not observation_id:
            raise ValueError("observation_id is required")
        if observation_id in self._seen_observation_ids:
            raise ValueError(
                f"online task-state observation is not fresh: {observation_id}"
            )
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms < 0
        ):
            raise ValueError("timestamp_ms must be a non-negative integer")
        if (
            self._last_timestamp_ms is not None
            and timestamp_ms < self._last_timestamp_ms
        ):
            raise ValueError("online task-state timestamp moved backwards")
        self._seen_observation_ids.add(observation_id)
        self._last_timestamp_ms = timestamp_ms

    def _current_target_id(self) -> str:
        if self.task_spec.task_id != BIN_HANDOFF_TASK_ID:
            assert self.task_spec.target_slot is not None
            return self.task_spec.target_slot
        return "FINISHED_01" if self._token == "B_ONLY" else "HANDOFF_CENTER"

    def _subtask_id(self) -> str:
        if self.task_spec.task_id != BIN_HANDOFF_TASK_ID:
            return self.task_spec.task_id
        return (
            "S02_ARM_B_TRANSPORT"
            if self._token == "B_ONLY"
            else "S01_ARM_A_PACK_HANDOFF"
        )

    def _visual_confidence(
        self,
        *,
        observation_id: str,
        camera: Mapping[str, Any],
        target_id: str,
    ) -> float:
        stream_name = _CAMERA_STREAM_BY_STAGE[target_id]
        raw_image = camera.get(stream_name)
        if not isinstance(raw_image, Mapping):
            raise ValueError(f"camera.{stream_name} must be an image reference")
        image = ImageReference.from_dict(raw_image)
        class_names = _TASK_CLASS_NAMES[self.task_spec.task_id]
        context = PerceptionContext(
            run_id=self.run_id,
            task_id=self.task_spec.task_id,
            subtask_id=self._subtask_id(),
            step_id=self._step_id,
            observation_id=observation_id,
            image=image,
            timeout_ms=self.timeout_ms,
            allowed_class_names=class_names,
            confidence_threshold=self.min_confidence,
            iou_threshold=0.45,
        )
        self._step_id += 1
        packet = self.perception.detect(context)
        if not isinstance(packet, DetectionPacket):
            raise TypeError("perception.detect() must return a DetectionPacket")
        packet.validate_against(
            observation_id=observation_id,
            image=image,
            descriptor=self.perception.descriptor,
        )
        self._last_detection_packet = packet
        candidates = [
            item for item in packet.detections if item.class_name in class_names
        ]
        matching = [
            item
            for item in candidates
            if self._detection_is_in_target(item, target_id=target_id)
        ]
        return max((float(item.confidence) for item in matching), default=0.0)

    def _detection_is_in_target(self, detection: Detection, *, target_id: str) -> bool:
        if detection.zone_id == target_id:
            return True
        attributes = detection.attributes
        for key in ("slot_id", "station_id", "target_id", "zone_id"):
            if attributes.get(key) == target_id:
                return True
        if attributes.get("object_id") not in {None, self.task_spec.target_object}:
            return False
        x_min, y_min, x_max, y_max = self._projected_target_region(
            target_id,
            camera_id=detection.camera_id,
            width=detection.image_width,
            height=detection.image_height,
        )
        box_x_min, box_y_min, box_x_max, box_y_max = detection.bbox_xyxy
        center_x = (box_x_min + box_x_max) / 2.0
        center_y = (box_y_min + box_y_max) / 2.0
        return x_min <= center_x <= x_max and y_min <= center_y <= y_max

    def _projected_target_region(
        self,
        target_id: str,
        *,
        camera_id: str,
        width: int,
        height: int,
    ) -> tuple[float, float, float, float]:
        center, half_size = self._target_world_region(target_id)
        z = center[2]
        corners = (
            (center[0] - half_size[0], center[1] - half_size[1], z),
            (center[0] - half_size[0], center[1] + half_size[1], z),
            (center[0] + half_size[0], center[1] - half_size[1], z),
            (center[0] + half_size[0], center[1] + half_size[1], z),
        )
        pixels = [self._project(point, camera_id, width, height) for point in corners]
        xs = [item[0] for item in pixels]
        ys = [item[1] for item in pixels]
        return min(xs), min(ys), max(xs), max(ys)

    def _target_world_region(
        self, target_id: str
    ) -> tuple[tuple[float, float, float], tuple[float, float]]:
        if target_id.startswith("S"):
            bin_spec = self.scene_config.get("bin")
            if not isinstance(bin_spec, Mapping):
                raise ValueError("scene config has no bin calibration")
            bin_pose = _position(bin_spec.get("pose"), "bin.pose")
            slots = bin_spec.get("slots")
            if not isinstance(slots, Sequence):
                raise ValueError("scene config has no slot calibration")
            slot = next(
                (
                    item
                    for item in slots
                    if isinstance(item, Mapping) and item.get("id") == target_id
                ),
                None,
            )
            if not isinstance(slot, Mapping):
                raise ValueError(f"scene config has no slot {target_id}")
            local = slot.get("center_local_m")
            if not isinstance(local, Sequence) or len(local) != 3:
                raise ValueError(f"slot {target_id} has invalid center calibration")
            size = bin_spec.get("size_m")
            grid = bin_spec.get("grid")
            if (
                not isinstance(size, Sequence)
                or len(size) < 2
                or not isinstance(grid, Mapping)
            ):
                raise ValueError("scene bin grid calibration is invalid")
            columns = int(grid.get("columns", 0))
            rows = int(grid.get("rows", 0))
            if columns < 1 or rows < 1:
                raise ValueError("scene bin grid dimensions are invalid")
            center = (
                bin_pose[0] + float(local[0]),
                bin_pose[1] + float(local[1]),
                bin_pose[2] + float(local[2]),
            )
            return center, (float(size[0]) / columns / 2.0, float(size[1]) / rows / 2.0)

        stations = self.scene_config.get("stations")
        if not isinstance(stations, Sequence):
            raise ValueError("scene config has no station calibration")
        station = next(
            (
                item
                for item in stations
                if isinstance(item, Mapping) and item.get("id") == target_id
            ),
            None,
        )
        if not isinstance(station, Mapping):
            raise ValueError(f"scene config has no station {target_id}")
        center = _position(station.get("pose"), f"station {target_id}.pose")
        footprint = station.get("footprint_m")
        if not isinstance(footprint, Sequence) or len(footprint) != 2:
            raise ValueError(f"station {target_id} has invalid footprint calibration")
        return center, (float(footprint[0]) / 2.0, float(footprint[1]) / 2.0)

    def _project(
        self,
        point: tuple[float, float, float],
        camera_id: str,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        cameras = self.scene_config.get("cameras")
        if not isinstance(cameras, Sequence):
            raise ValueError("scene config has no camera calibration")
        camera = next(
            (
                item
                for item in cameras
                if isinstance(item, Mapping) and item.get("id") == camera_id
            ),
            None,
        )
        if not isinstance(camera, Mapping):
            raise ValueError(f"scene config has no camera {camera_id}")
        position = _position(camera.get("pose"), f"camera {camera_id}.pose")
        look_at_raw = camera.get("look_at_m")
        if not isinstance(look_at_raw, Sequence) or len(look_at_raw) != 3:
            raise ValueError(f"camera {camera_id} has invalid look_at calibration")
        look_at = tuple(float(item) for item in look_at_raw)
        forward = _normalize(_subtract(look_at, position))
        right = _normalize(_cross(forward, (0.0, 0.0, 1.0)))
        up = _cross(right, forward)
        relative = _subtract(point, position)
        depth = _dot(relative, forward)
        if depth <= 0.0:
            raise ValueError(f"target is behind camera {camera_id}")
        focal = width / (
            2.0 * tan(float(camera.get("horizontal_fov_deg", 82.0)) * pi / 360.0)
        )
        return (
            width / 2.0 + focal * _dot(relative, right) / depth,
            height / 2.0 - focal * _dot(relative, up) / depth,
        )

    def _robot_ready(self, robot: Mapping[str, Any]) -> bool:
        arm_a = robot.get("arm_a")
        arm_b = robot.get("arm_b")
        if not isinstance(arm_a, Mapping) or not isinstance(arm_b, Mapping):
            return False
        if self.task_spec.task_id != BIN_HANDOFF_TASK_ID:
            return arm_a.get("gripper_open") is True and arm_a.get("stationary") is True
        parked_a = arm_a.get("retreated") is True and arm_a.get("stationary") is True
        parked_b = arm_b.get("retreated") is True and arm_b.get("stationary") is True
        if self._token in {"A_ONLY", "HANDOFF_VERIFY"}:
            return parked_a and parked_b and arm_a.get("gripper_open") is True
        return parked_a and parked_b and arm_b.get("gripper_open") is True

    def _update_handoff(self, vote: _Vote) -> None:
        if self._token == "A_ONLY":
            if not vote.passed:
                self._verification_votes = 0
                self._terminal_confidence = 0.0
                return
            self._token = "HANDOFF_VERIFY"
            self._votes.clear()
            self._votes.append(vote)
            self._publish_window(terminal_stage=False)
            return
        self._votes.append(vote)
        if self._token == "HANDOFF_VERIFY":
            verified = self._publish_window(terminal_stage=False)
            if verified:
                self._token = "B_ONLY"
                self._votes.clear()
                self._verification_votes = 0
                self._terminal_confidence = 0.0
            return
        if self._token == "B_ONLY":
            self._publish_window(terminal_stage=True)

    def _publish_window(self, *, terminal_stage: bool) -> bool:
        passed = [vote for vote in self._votes if vote.passed]
        self._verification_votes = len(passed)
        self._terminal_confidence = min(
            (vote.confidence for vote in passed),
            default=0.0,
        )
        verified = (
            len(self._votes) == self.verification_frames
            and self._verification_votes >= self.required_votes
            and self._terminal_confidence >= self.min_confidence
        )
        if verified and terminal_stage:
            self._status = "SUCCEEDED"
            self._terminal = True
            self._token = "NONE"
        return verified

    def _snapshot_unlocked(self) -> dict[str, Any]:
        return {
            "task_id": self.task_spec.task_id,
            "target_object_id": self.task_spec.target_object,
            "target_slot_id": self.task_spec.target_slot,
            "status": self._status,
            "terminal": self._terminal,
            "terminal_confidence": self._terminal_confidence,
            "verification_votes": self._verification_votes,
        }


def _position(value: Any, label: str) -> tuple[float, float, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    raw = value.get("position_m")
    if not isinstance(raw, Sequence) or len(raw) != 3:
        raise ValueError(f"{label}.position_m must contain three numbers")
    return tuple(float(item) for item in raw)  # type: ignore[return-value]


def _subtract(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return tuple(a - b for a, b in zip(left, right))  # type: ignore[return-value]


def _dot(left: tuple[float, float, float], right: tuple[float, float, float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _cross(
    left: tuple[float, float, float], right: tuple[float, float, float]
) -> tuple[float, float, float]:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _normalize(value: tuple[float, float, float]) -> tuple[float, float, float]:
    length = sqrt(_dot(value, value))
    if length <= 0.0:
        raise ValueError("calibration vector has zero length")
    return tuple(item / length for item in value)  # type: ignore[return-value]


__all__ = ["BIN_HANDOFF_TASK_ID", "CONTROL_TOKENS", "OnlineTaskStateProvider"]
