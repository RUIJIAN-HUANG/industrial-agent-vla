"""Atomic multi-rate HDF5 recorder for the frozen industrial workcell.

The recorder consumes verified CAS image references and keeps independent
30 Hz camera, 60 Hz robot-state, and 10 Hz model-action timelines. It never
modifies the frozen online Observation or ActionChunk contracts.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import re
import shutil
from threading import RLock
from typing import Any, Mapping, Sequence
from uuid import uuid4

import h5py
import numpy as np

from industrial_agent.errors import ImageCasError
from industrial_agent.image_cas import ImageCas
from industrial_agent.perception import ImageReference
from industrial_agent.sync_contract import FROZEN_MULTI_RATE

from .padding import PaddingPolicy, PaddingResult, pad_actions


CANONICAL_EPISODE_VERSION = "1.0"
FROZEN_SCENE_ID = "single_bin_static_handoff_v1"
V2_MANUAL_SCENE_ID = "single_bin_manual_industrial_v2"
ALLOWED_SCENE_IDS = frozenset({FROZEN_SCENE_ID, V2_MANUAL_SCENE_ID})
CAMERA_IDS = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
ARM_IDS = ("Arm_A", "Arm_B")
EXECUTOR_BY_ARM = {"Arm_A": "pi05", "Arm_B": "openvla_oft"}
IMAGE_WIDTH = 1280
IMAGE_HEIGHT = 720
STATE_DIM = 7
ACTION_DIM = 7
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-fA-F]{64}$")
_FINAL_OUTCOMES = frozenset({"SUCCEEDED", "FAILED", "SAFE_STOPPED", "SAFE_STOP_FAILED"})
_STRING_DTYPE = h5py.string_dtype(encoding="utf-8")
_UINT64_MAX = int(np.iinfo(np.uint64).max)


def _non_blank(value: Any, field_name: str, *, maximum: int = 4096) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    normalized = value.strip()
    if len(normalized) > maximum:
        raise ValueError(f"{field_name} exceeds {maximum} characters")
    return normalized


@dataclass(frozen=True)
class EpisodeMetadata:
    """Immutable identity required before recording any stream."""

    episode_id: str
    task_id: str
    instruction: str
    scene_seed: int
    git_sha: str
    scene_config_sha256: str
    scene_id: str = FROZEN_SCENE_ID

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_id, str)
            or _SAFE_EPISODE_ID.fullmatch(self.episode_id) is None
        ):
            raise ValueError(
                "episode_id must match ^[A-Za-z0-9._-]{1,128}$ for safe storage"
            )
        object.__setattr__(self, "task_id", _non_blank(self.task_id, "task_id"))
        object.__setattr__(
            self,
            "instruction",
            _non_blank(self.instruction, "instruction", maximum=16_384),
        )
        if (
            isinstance(self.scene_seed, bool)
            or not isinstance(self.scene_seed, int)
            or self.scene_seed < 0
        ):
            raise ValueError("scene_seed must be a non-negative integer")
        if self.scene_id not in ALLOWED_SCENE_IDS:
            allowed = ", ".join(sorted(ALLOWED_SCENE_IDS))
            raise ValueError(f"scene_id must be one of the audited values: {allowed}")
        if (
            not isinstance(self.git_sha, str)
            or _GIT_SHA.fullmatch(self.git_sha) is None
        ):
            raise ValueError("git_sha must contain exactly 40 hexadecimal characters")
        if (
            not isinstance(self.scene_config_sha256, str)
            or _SHA256.fullmatch(self.scene_config_sha256) is None
        ):
            raise ValueError(
                "scene_config_sha256 must be sha256:<64 hexadecimal characters>"
            )
        object.__setattr__(self, "git_sha", self.git_sha.lower())
        object.__setattr__(
            self, "scene_config_sha256", self.scene_config_sha256.lower()
        )


@dataclass
class _StreamTracker:
    stride: int
    expected_sequence_id: int = 0
    last_timestamp_ns: int = -1
    last_physics_tick: int = -1
    timestamps_ns: list[int] = field(default_factory=list)
    physics_ticks: list[int] = field(default_factory=list)

    def preview(
        self,
        *,
        sequence_id: int,
        timestamp_ns: int,
        physics_tick: int,
    ) -> int:
        for value, field_name in (
            (sequence_id, "sequence_id"),
            (timestamp_ns, "timestamp_ns"),
            (physics_tick, "physics_tick"),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{field_name} must be a non-negative integer")
            if value > _UINT64_MAX:
                raise ValueError(f"{field_name} exceeds uint64 storage range")
        if sequence_id != self.expected_sequence_id:
            raise ValueError(
                f"sequence_id must be contiguous: expected "
                f"{self.expected_sequence_id}, got {sequence_id}"
            )
        if timestamp_ns <= self.last_timestamp_ns:
            raise ValueError("timestamp_ns must be strictly increasing per stream")
        if physics_tick % self.stride:
            raise ValueError(
                f"physics_tick {physics_tick} is not aligned to stride {self.stride}"
            )
        if physics_tick <= self.last_physics_tick:
            raise ValueError("physics_tick must be strictly increasing per stream")
        if self.last_physics_tick < 0:
            return 0
        tick_gap = physics_tick - self.last_physics_tick
        if tick_gap % self.stride:
            raise ValueError("physics_tick gap is not aligned to the stream rate")
        return max(0, tick_gap // self.stride - 1)

    def commit(self, *, timestamp_ns: int, physics_tick: int) -> None:
        self.expected_sequence_id += 1
        self.last_timestamp_ns = timestamp_ns
        self.last_physics_tick = physics_tick
        self.timestamps_ns.append(timestamp_ns)
        self.physics_ticks.append(physics_tick)

    @property
    def count(self) -> int:
        return self.expected_sequence_id


def _numeric_vector(
    value: Sequence[float] | np.ndarray,
    *,
    field_name: str,
    expected_dim: int,
    gripper_min: float,
    gripper_max: float,
) -> np.ndarray:
    if isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{field_name} must be a numeric vector")
    try:
        vector = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must contain only numeric values") from exc
    if vector.shape != (expected_dim,):
        raise ValueError(f"{field_name} must have shape [{expected_dim}]")
    if not all(isfinite(float(item)) for item in vector):
        raise ValueError(f"{field_name} must contain only finite values")
    gripper = float(vector[6])
    if gripper < gripper_min or gripper > gripper_max:
        raise ValueError(
            f"{field_name}[6] must be within [{gripper_min}, {gripper_max}]"
        )
    return np.ascontiguousarray(vector)


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return f"sha256:{digest.hexdigest()}"


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class CanonicalRecorder:
    """Incremental recorder that atomically publishes one episode directory."""

    def __init__(
        self,
        output_root: str | Path,
        metadata: EpisodeMetadata,
        *,
        image_cas: ImageCas,
        padding_policy: PaddingPolicy | None = None,
    ) -> None:
        if not isinstance(metadata, EpisodeMetadata):
            raise TypeError("metadata must be EpisodeMetadata")
        if not isinstance(image_cas, ImageCas):
            raise TypeError("image_cas must be ImageCas")
        if padding_policy is not None and not isinstance(padding_policy, PaddingPolicy):
            raise TypeError("padding_policy must be PaddingPolicy or None")

        root = Path(output_root).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        if root.is_symlink() or not root.is_dir():
            raise ValueError("output_root must be a real directory, not a symlink")
        self.output_root = root.resolve()
        self.metadata = metadata
        self.image_cas = image_cas
        self.padding_policy = padding_policy or PaddingPolicy()
        self._lock = RLock()
        self._state = "OPEN"
        self._final_path = self.output_root / metadata.episode_id
        if self._final_path.exists():
            raise FileExistsError(f"episode already exists: {self._final_path}")

        self._temporary_path = self.output_root / (
            f".{metadata.episode_id}.{uuid4().hex}.tmp"
        )
        self._temporary_path.mkdir()
        self._hdf5_path = self._temporary_path / "episode.h5"
        self._h5 = h5py.File(self._hdf5_path, "w")
        self._camera_trackers = {
            camera_id: _StreamTracker(FROZEN_MULTI_RATE.physics_ticks_per_render)
            for camera_id in CAMERA_IDS
        }
        self._state_trackers = {
            arm_id: _StreamTracker(FROZEN_MULTI_RATE.physics_ticks_per_control)
            for arm_id in ARM_IDS
        }
        self._action_tracker = _StreamTracker(
            FROZEN_MULTI_RATE.physics_ticks_per_model_step
        )
        try:
            self._initialize_hdf5()
        except (OSError, RuntimeError, TypeError, ValueError):
            self._h5.close()
            shutil.rmtree(self._temporary_path, ignore_errors=True)
            raise

    def _initialize_hdf5(self) -> None:
        self._h5.attrs.update(
            {
                "schema_version": CANONICAL_EPISODE_VERSION,
                "episode_id": self.metadata.episode_id,
                "scene_id": self.metadata.scene_id,
                "task_id": self.metadata.task_id,
                "instruction": self.metadata.instruction,
                "scene_seed": self.metadata.scene_seed,
                "git_sha": self.metadata.git_sha,
                "scene_config_sha256": self.metadata.scene_config_sha256,
                "wrist_image": "null",
                "offline_gt_included": False,
                "physics_hz": FROZEN_MULTI_RATE.physics_hz,
                "control_hz": FROZEN_MULTI_RATE.control_hz,
                "render_hz": FROZEN_MULTI_RATE.render_hz,
                "model_inference_hz": FROZEN_MULTI_RATE.model_inference_hz,
                "padding_strategy": self.padding_policy.strategy.value,
                "padding_target_length": (
                    -1
                    if self.padding_policy.target_length is None
                    else self.padding_policy.target_length
                ),
            }
        )
        cameras = self._h5.create_group("cameras")
        for camera_id in CAMERA_IDS:
            group = cameras.create_group(camera_id)
            self._create_common_timeline(group)
            group.create_dataset(
                "rgb",
                shape=(0, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                maxshape=(None, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                dtype=np.uint8,
                compression="gzip",
                compression_opts=1,
                shuffle=True,
            )
            group.create_dataset(
                "cas_uri", shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE
            )
            group.create_dataset(
                "image_sha256", shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE
            )
            group.create_dataset(
                "is_fallback", shape=(0,), maxshape=(None,), dtype=np.bool_
            )
            group.create_dataset(
                "dropped_before", shape=(0,), maxshape=(None,), dtype=np.uint32
            )

        states = self._h5.create_group("robot_state")
        for arm_id in ARM_IDS:
            group = states.create_group(arm_id)
            self._create_common_timeline(group)
            group.create_dataset(
                "state_7d",
                shape=(0, STATE_DIM),
                maxshape=(None, STATE_DIM),
                chunks=(256, STATE_DIM),
                dtype=np.float32,
            )

        actions = self._h5.create_group("actions")
        self._create_common_timeline(actions)
        actions.create_dataset(
            "action_7d",
            shape=(0, ACTION_DIM),
            maxshape=(None, ACTION_DIM),
            chunks=(256, ACTION_DIM),
            dtype=np.float32,
        )
        actions.create_dataset(
            "duration_ms", shape=(0,), maxshape=(None,), dtype=np.uint16
        )
        actions.create_dataset(
            "valid_mask", shape=(0,), maxshape=(None,), dtype=np.bool_
        )
        actions.create_dataset(
            "arm_id", shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE
        )
        actions.create_dataset(
            "executor", shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE
        )
        actions.create_dataset(
            "subtask_id", shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE
        )
        actions.create_dataset(
            "chunk_id", shape=(0,), maxshape=(None,), dtype=_STRING_DTYPE
        )
        actions.create_dataset(
            "chunk_position", shape=(0,), maxshape=(None,), dtype=np.uint16
        )
        self._h5.flush()

    @staticmethod
    def _create_common_timeline(group: h5py.Group) -> None:
        group.create_dataset(
            "timestamp_ns", shape=(0,), maxshape=(None,), dtype=np.uint64
        )
        group.create_dataset(
            "physics_tick", shape=(0,), maxshape=(None,), dtype=np.uint64
        )
        group.create_dataset(
            "sequence_id", shape=(0,), maxshape=(None,), dtype=np.uint64
        )

    def _require_open(self) -> None:
        if self._state != "OPEN":
            raise RuntimeError(f"recorder is not open (state={self._state})")

    def _append_record(self, group: h5py.Group, values: Mapping[str, Any]) -> None:
        if set(values) != set(group.keys()):
            missing = sorted(set(group.keys()) - set(values))
            extra = sorted(set(values) - set(group.keys()))
            raise RuntimeError(
                f"internal stream record mismatch: missing={missing}, extra={extra}"
            )
        old_size = next(iter(group.values())).shape[0]
        resized: list[h5py.Dataset] = []
        try:
            for name, dataset in group.items():
                if dataset.shape[0] != old_size:
                    raise RuntimeError("HDF5 stream datasets have inconsistent lengths")
                dataset.resize(old_size + 1, axis=0)
                resized.append(dataset)
                dataset[old_size] = values[name]
        except (OSError, RuntimeError, TypeError, ValueError) as exc:
            rollback_error: BaseException | None = None
            for dataset in resized:
                try:
                    dataset.resize(old_size, axis=0)
                except (OSError, RuntimeError, TypeError, ValueError) as rollback_exc:
                    rollback_error = rollback_exc
            self._state = "FAILED"
            if rollback_error is not None:
                raise RuntimeError("HDF5 append rollback failed") from rollback_error
            raise RuntimeError("HDF5 stream append failed and was rolled back") from exc

    def add_frame(
        self,
        *,
        camera_id: str,
        timestamp_ns: int,
        physics_tick: int,
        sequence_id: int,
        image_reference: ImageReference | Mapping[str, Any],
        is_fallback: bool = False,
    ) -> None:
        """Append one verified 1280x720 RGB frame on the 30 Hz timeline."""

        if camera_id not in CAMERA_IDS:
            raise ValueError(f"camera_id must be one of {CAMERA_IDS}")
        if not isinstance(is_fallback, bool):
            raise TypeError("is_fallback must be bool")
        try:
            reference = (
                image_reference
                if isinstance(image_reference, ImageReference)
                else ImageReference.from_dict(image_reference)
            )
            resolved = self.image_cas.resolve_rgb(
                reference,
                expected_camera_id=camera_id,
                expected_size=(IMAGE_WIDTH, IMAGE_HEIGHT),
            )
        except (ImageCasError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid CAS-backed RGB frame for {camera_id}: {exc}"
            ) from exc
        frame = np.asarray(resolved.rgb)
        if frame.dtype != np.uint8 or frame.shape != (
            IMAGE_HEIGHT,
            IMAGE_WIDTH,
            3,
        ):
            raise ValueError(
                f"{camera_id} RGB must be uint8 [{IMAGE_HEIGHT},{IMAGE_WIDTH},3]"
            )

        with self._lock:
            self._require_open()
            tracker = self._camera_trackers[camera_id]
            dropped_before = tracker.preview(
                sequence_id=sequence_id,
                timestamp_ns=timestamp_ns,
                physics_tick=physics_tick,
            )
            self._append_record(
                self._h5[f"cameras/{camera_id}"],
                {
                    "timestamp_ns": timestamp_ns,
                    "physics_tick": physics_tick,
                    "sequence_id": sequence_id,
                    "rgb": frame,
                    "cas_uri": reference.uri,
                    "image_sha256": reference.image_sha256.lower(),
                    "is_fallback": is_fallback,
                    "dropped_before": dropped_before,
                },
            )
            tracker.commit(timestamp_ns=timestamp_ns, physics_tick=physics_tick)

    def add_state(
        self,
        *,
        arm_id: str,
        timestamp_ns: int,
        physics_tick: int,
        sequence_id: int,
        state_7d: Sequence[float] | np.ndarray,
    ) -> None:
        """Append one robot-base rotation-vector state on the 60 Hz timeline."""

        if arm_id not in ARM_IDS:
            raise ValueError(f"arm_id must be one of {ARM_IDS}")
        state = _numeric_vector(
            state_7d,
            field_name="state_7d",
            expected_dim=STATE_DIM,
            gripper_min=0.0,
            gripper_max=1.0,
        )
        with self._lock:
            self._require_open()
            tracker = self._state_trackers[arm_id]
            tracker.preview(
                sequence_id=sequence_id,
                timestamp_ns=timestamp_ns,
                physics_tick=physics_tick,
            )
            self._append_record(
                self._h5[f"robot_state/{arm_id}"],
                {
                    "timestamp_ns": timestamp_ns,
                    "physics_tick": physics_tick,
                    "sequence_id": sequence_id,
                    "state_7d": state,
                },
            )
            tracker.commit(timestamp_ns=timestamp_ns, physics_tick=physics_tick)

    def add_action(
        self,
        *,
        arm_id: str,
        executor: str,
        subtask_id: str,
        chunk_id: str,
        timestamp_ns: int,
        physics_tick: int,
        sequence_id: int,
        action_7d: Sequence[float] | np.ndarray,
        duration_ms: int = 100,
    ) -> None:
        """Append one real 10 Hz action; this method cannot create padding."""

        action = _numeric_vector(
            action_7d,
            field_name="action_7d",
            expected_dim=ACTION_DIM,
            gripper_min=-1.0,
            gripper_max=1.0,
        )
        self._append_action(
            arm_id=arm_id,
            executor=executor,
            subtask_id=subtask_id,
            chunk_id=chunk_id,
            chunk_position=0,
            timestamp_ns=timestamp_ns,
            physics_tick=physics_tick,
            sequence_id=sequence_id,
            action_7d=action,
            duration_ms=duration_ms,
            valid=True,
        )

    def add_action_chunk(
        self,
        *,
        arm_id: str,
        executor: str,
        subtask_id: str,
        chunk_id: str,
        start_timestamp_ns: int,
        start_physics_tick: int,
        start_sequence_id: int,
        actions: Sequence[Sequence[float]] | np.ndarray,
        duration_ms: int = 100,
    ) -> PaddingResult:
        """Append one training chunk and persist an explicit valid mask."""

        result = pad_actions(actions, self.padding_policy)
        self._validate_action_metadata(
            arm_id=arm_id,
            executor=executor,
            subtask_id=subtask_id,
            chunk_id=chunk_id,
            duration_ms=duration_ms,
        )
        if result.target_length > 65_536:
            raise ValueError("padded action chunk exceeds uint16 chunk_position range")
        step_ns = 1_000_000_000 // FROZEN_MULTI_RATE.model_inference_hz
        tick_stride = FROZEN_MULTI_RATE.physics_ticks_per_model_step
        final_position = result.target_length - 1
        for value, field_name in (
            (start_timestamp_ns + final_position * step_ns, "final timestamp_ns"),
            (start_physics_tick + final_position * tick_stride, "final physics_tick"),
            (start_sequence_id + final_position, "final sequence_id"),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value > _UINT64_MAX
            ):
                raise ValueError(f"{field_name} exceeds uint64 storage range")
        with self._lock:
            self._require_open()
            for position, action in enumerate(result.values):
                self._append_action(
                    arm_id=arm_id,
                    executor=executor,
                    subtask_id=subtask_id,
                    chunk_id=chunk_id,
                    chunk_position=position,
                    timestamp_ns=start_timestamp_ns + position * step_ns,
                    physics_tick=start_physics_tick + position * tick_stride,
                    sequence_id=start_sequence_id + position,
                    action_7d=action,
                    duration_ms=duration_ms,
                    valid=bool(result.valid_mask[position]),
                )
        return result

    @staticmethod
    def _validate_action_metadata(
        *,
        arm_id: str,
        executor: str,
        subtask_id: str,
        chunk_id: str,
        duration_ms: int,
    ) -> tuple[str, str]:
        if arm_id not in ARM_IDS:
            raise ValueError(f"arm_id must be one of {ARM_IDS}")
        expected_executor = EXECUTOR_BY_ARM[arm_id]
        if executor != expected_executor:
            raise ValueError(f"{arm_id} requires executor {expected_executor!r}")
        subtask = _non_blank(subtask_id, "subtask_id", maximum=256)
        chunk = _non_blank(chunk_id, "chunk_id", maximum=256)
        if duration_ms != FROZEN_MULTI_RATE.model_step_duration_ms:
            raise ValueError(
                f"duration_ms must be the frozen 10 Hz value "
                f"{FROZEN_MULTI_RATE.model_step_duration_ms}"
            )
        return subtask, chunk

    def _append_action(
        self,
        *,
        arm_id: str,
        executor: str,
        subtask_id: str,
        chunk_id: str,
        chunk_position: int,
        timestamp_ns: int,
        physics_tick: int,
        sequence_id: int,
        action_7d: Sequence[float] | np.ndarray,
        duration_ms: int,
        valid: bool,
    ) -> None:
        subtask, chunk = self._validate_action_metadata(
            arm_id=arm_id,
            executor=executor,
            subtask_id=subtask_id,
            chunk_id=chunk_id,
            duration_ms=duration_ms,
        )
        if (
            isinstance(chunk_position, bool)
            or not isinstance(chunk_position, int)
            or chunk_position < 0
            or chunk_position > 65_535
        ):
            raise ValueError("chunk_position must be an integer in [0, 65535]")
        if not isinstance(valid, bool):
            raise TypeError("valid must be bool")
        action = _numeric_vector(
            action_7d,
            field_name="action_7d",
            expected_dim=ACTION_DIM,
            gripper_min=-1.0,
            gripper_max=1.0,
        )
        with self._lock:
            self._require_open()
            self._action_tracker.preview(
                sequence_id=sequence_id,
                timestamp_ns=timestamp_ns,
                physics_tick=physics_tick,
            )
            self._append_record(
                self._h5["actions"],
                {
                    "timestamp_ns": timestamp_ns,
                    "physics_tick": physics_tick,
                    "sequence_id": sequence_id,
                    "action_7d": action,
                    "duration_ms": duration_ms,
                    "valid_mask": valid,
                    "arm_id": arm_id,
                    "executor": executor,
                    "subtask_id": subtask,
                    "chunk_id": chunk,
                    "chunk_position": chunk_position,
                },
            )
            self._action_tracker.commit(
                timestamp_ns=timestamp_ns, physics_tick=physics_tick
            )

    def _validate_complete(self) -> None:
        camera_counts = {tracker.count for tracker in self._camera_trackers.values()}
        if camera_counts == {0}:
            raise ValueError("episode has no RGB frames")
        if len(camera_counts) != 1 or 0 in camera_counts:
            raise ValueError("three camera streams must have equal non-zero counts")
        camera_ticks = [
            tracker.physics_ticks for tracker in self._camera_trackers.values()
        ]
        camera_timestamps = [
            tracker.timestamps_ns for tracker in self._camera_trackers.values()
        ]
        if any(ticks != camera_ticks[0] for ticks in camera_ticks[1:]):
            raise ValueError("three camera streams must share the same render ticks")
        if any(values != camera_timestamps[0] for values in camera_timestamps[1:]):
            raise ValueError("three camera streams must share synchronized timestamps")

        state_counts = {tracker.count for tracker in self._state_trackers.values()}
        if len(state_counts) != 1 or 0 in state_counts:
            raise ValueError("Arm_A and Arm_B state streams must be equal and non-zero")
        state_ticks = [
            tracker.physics_ticks for tracker in self._state_trackers.values()
        ]
        if state_ticks[0] != state_ticks[1]:
            raise ValueError("Arm_A and Arm_B state streams must share control ticks")
        if self._action_tracker.count < 1:
            raise ValueError("episode has no action records")
        valid_mask = np.asarray(self._h5["actions/valid_mask"][:], dtype=np.bool_)
        if not np.any(valid_mask):
            raise ValueError("episode must contain at least one valid action")

    @staticmethod
    def _dataset_descriptor(dataset: h5py.Dataset) -> dict[str, Any]:
        string_info = h5py.check_string_dtype(dataset.dtype)
        dtype = "utf-8" if string_info is not None else str(dataset.dtype)
        return {
            "path": dataset.name,
            "dtype": dtype,
            "shape": [int(value) for value in dataset.shape],
        }

    def _stream_summary(
        self,
        group: h5py.Group,
        *,
        frequency_hz: int,
        fallback_count: int | None = None,
        valid_count: int | None = None,
    ) -> dict[str, Any]:
        datasets = {
            name: self._dataset_descriptor(dataset)
            for name, dataset in sorted(group.items())
        }
        count = int(next(iter(group.values())).shape[0])
        result: dict[str, Any] = {
            "count": count,
            "frequency_hz": frequency_hz,
            "datasets": datasets,
        }
        if fallback_count is not None:
            result["fallback_count"] = fallback_count
        if valid_count is not None:
            result["valid_count"] = valid_count
        return result

    def _manifest(
        self,
        *,
        outcome: str,
        failure_code: str | None,
        storage_sha256: str,
    ) -> dict[str, Any]:
        cameras: dict[str, Any] = {}
        for camera_id in CAMERA_IDS:
            group = self._h5[f"cameras/{camera_id}"]
            cameras[camera_id] = self._stream_summary(
                group,
                frequency_hz=FROZEN_MULTI_RATE.render_hz,
                fallback_count=int(np.count_nonzero(group["is_fallback"][:])),
            )
        states = {
            arm_id: self._stream_summary(
                self._h5[f"robot_state/{arm_id}"],
                frequency_hz=FROZEN_MULTI_RATE.control_hz,
            )
            for arm_id in ARM_IDS
        }
        actions_group = self._h5["actions"]
        actions = self._stream_summary(
            actions_group,
            frequency_hz=FROZEN_MULTI_RATE.model_inference_hz,
            valid_count=int(np.count_nonzero(actions_group["valid_mask"][:])),
        )
        return {
            "schema_version": CANONICAL_EPISODE_VERSION,
            "metadata": {
                "episode_id": self.metadata.episode_id,
                "scene_id": self.metadata.scene_id,
                "task_id": self.metadata.task_id,
                "instruction": self.metadata.instruction,
                "scene_seed": self.metadata.scene_seed,
                "git_sha": self.metadata.git_sha,
                "scene_config_sha256": self.metadata.scene_config_sha256,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "outcome": outcome,
                "failure_code": failure_code,
                "wrist_image": None,
                "offline_gt_included": False,
                "frequency_contract": {
                    "physics_hz": FROZEN_MULTI_RATE.physics_hz,
                    "control_hz": FROZEN_MULTI_RATE.control_hz,
                    "render_hz": FROZEN_MULTI_RATE.render_hz,
                    "model_inference_hz": FROZEN_MULTI_RATE.model_inference_hz,
                },
                "padding_policy": self.padding_policy.to_dict(),
            },
            "storage": {
                "format": "hdf5",
                "episode_file": "episode.h5",
                "structure_file": "structure.json",
                "sha256": storage_sha256,
            },
            "streams": {
                "cameras": cameras,
                "robot_state": states,
                "actions": actions,
            },
        }

    def save_episode(
        self,
        *,
        outcome: str,
        failure_code: str | None = None,
    ) -> Path:
        """Validate, fsync, and atomically publish the episode directory."""

        if outcome not in _FINAL_OUTCOMES:
            raise ValueError(f"outcome must be one of {sorted(_FINAL_OUTCOMES)}")
        if outcome == "SUCCEEDED":
            if failure_code is not None:
                raise ValueError("SUCCEEDED episode must have failure_code=None")
        else:
            failure_code = _non_blank(failure_code, "failure_code", maximum=256)

        with self._lock:
            self._require_open()
            self._validate_complete()
            self._h5.attrs["outcome"] = outcome
            self._h5.attrs["failure_code"] = failure_code or ""
            self._h5.flush()
            self._h5.close()
            try:
                with self._hdf5_path.open("rb+") as stream:
                    os.fsync(stream.fileno())
                storage_sha256 = _file_sha256(self._hdf5_path)
                self._h5 = h5py.File(self._hdf5_path, "r")
                manifest = self._manifest(
                    outcome=outcome,
                    failure_code=failure_code,
                    storage_sha256=storage_sha256,
                )
                self._h5.close()
                structure_path = self._temporary_path / "structure.json"
                encoded = json.dumps(
                    manifest,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                    allow_nan=False,
                ).encode("utf-8")
                with structure_path.open("xb") as stream:
                    stream.write(encoded)
                    stream.flush()
                    os.fsync(stream.fileno())
                _fsync_directory(self._temporary_path)
                if self._final_path.exists():
                    raise FileExistsError(f"episode already exists: {self._final_path}")
                os.replace(self._temporary_path, self._final_path)
                try:
                    _fsync_directory(self.output_root)
                except OSError as exc:
                    self._state = "PUBLISHED_UNCONFIRMED"
                    raise RuntimeError(
                        "episode directory is complete but parent-directory "
                        "durability could not be confirmed"
                    ) from exc
            except (OSError, RuntimeError, TypeError, ValueError) as exc:
                if self._state != "PUBLISHED_UNCONFIRMED":
                    self._state = "FAILED"
                raise RuntimeError(
                    "episode publication did not reach a confirmed durable state"
                ) from exc
            self._state = "SAVED"
            return self._final_path

    def abort(self) -> None:
        """Close and remove an unpublished temporary episode."""

        with self._lock:
            if self._state in {"SAVED", "PUBLISHED_UNCONFIRMED"}:
                raise RuntimeError("a published episode cannot be aborted")
            if getattr(self, "_h5", None) is not None and self._h5.id.valid:
                self._h5.close()
            try:
                shutil.rmtree(self._temporary_path)
            except FileNotFoundError:
                self._state = "ABORTED"
                return
            except OSError as exc:
                raise RuntimeError("failed to remove temporary episode") from exc
            self._state = "ABORTED"

    def __enter__(self) -> CanonicalRecorder:
        self._require_open()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        if self._state in {"OPEN", "FAILED"}:
            self.abort()
