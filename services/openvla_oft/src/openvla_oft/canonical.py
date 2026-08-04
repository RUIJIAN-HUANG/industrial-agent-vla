"""Canonical HDF5 loader for OpenVLA-OFT Arm_B offline inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from industrial_agent.data import CanonicalEpisodeReader, SplitRegistry
from industrial_agent.lifecycle import (
    ARM_B_TRANSPORT_SUBTASK_ID,
    FixedTaskProfile,
)

from .dataset import ARM_B_CAMERA_ID, ARM_B_ROLE
from .exceptions import ServiceError

ARM_B_ID = "Arm_B"
OPENVLA_EXECUTOR = "openvla_oft"
IMAGE_SHAPE = (720, 1280, 3)
STATE_DIM = 7
ACTION_DIM = 7
EXPECTED_INSTRUCTION = FixedTaskProfile().arm_b_instruction


@dataclass(frozen=True)
class CanonicalSource:
    """Traceability fields that let an exported step point back to HDF5."""

    episode_id: str
    task_id: str
    action_sequence_id: int
    action_physics_tick: int
    action_timestamp_ns: int
    camera_id: str
    camera_sequence_id: int
    camera_physics_tick: int
    camera_timestamp_ns: int
    camera_image_sha256: str
    state_arm_id: str
    state_sequence_id: int
    state_physics_tick: int
    state_timestamp_ns: int
    split: str
    split_registry_sha256: str


@dataclass(frozen=True)
class OpenVLACanonicalStep:
    """One verified Arm_B OpenVLA training/inference step.

    ``image`` is RGB ``uint8`` with shape ``[720, 1280, 3]``.
    ``state_7d`` and ``action_7d`` use the frozen robot-base 7-D contract.
    """

    episode_id: str
    task_id: str
    language_instruction: str
    step_index: int
    image: np.ndarray
    state_7d: tuple[float, float, float, float, float, float, float]
    action_7d: tuple[float, float, float, float, float, float, float]
    is_first: bool
    is_last: bool
    is_terminal: bool
    source: CanonicalSource

    def to_training_sample(self) -> dict[str, Any]:
        """Return an in-memory OpenVLA sample with loaded image pixels."""

        return {
            "task_id": self.task_id,
            "episode_id": self.episode_id,
            "step_id": self.step_index,
            "robot_role": ARM_B_ROLE,
            "model_input": {
                "task_description": self.language_instruction,
                "full_image": self.image.copy(),
                "wrist_image": None,
                "state": list(self.state_7d),
            },
            "action": [list(self.action_7d)],
            "source": self.source.__dict__.copy(),
        }


def load_openvla_arm_b_steps(
    episode_path: str | Path,
    *,
    split_registry: SplitRegistry,
) -> tuple[OpenVLACanonicalStep, ...]:
    """Load verified Train-split Arm_B steps from one successful Episode."""

    if not isinstance(split_registry, SplitRegistry):
        raise TypeError("split_registry must be a verified SplitRegistry")
    try:
        reader = CanonicalEpisodeReader(
            episode_path,
            split_registry=split_registry,
            is_training=True,
        )
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise _bad_episode(
            f"authoritative Canonical reader rejected Episode: {exc}"
        ) from exc

    try:
        manifest = reader.manifest
        metadata = _metadata(manifest)
        episode_id = _required_text(metadata, "episode_id")
        task_id = _required_text(metadata, "task_id")
        instruction = _required_text(metadata, "instruction")
        if instruction != EXPECTED_INSTRUCTION:
            raise _bad_episode(
                "metadata.instruction does not match the frozen Arm_B task profile"
            )
        if metadata.get("outcome") != "SUCCEEDED":
            raise _bad_episode(
                "only SUCCEEDED Episodes are eligible for training export"
            )
        assignment = reader.split_assignment
        if assignment is None or assignment.split.value != "train":
            raise _bad_episode("training export requires a verified Train assignment")

        frames = _camera_frames(reader)
        camera_sequence_ids = _stream_ints(
            reader, f"cameras/{ARM_B_CAMERA_ID}", "sequence_id"
        )
        camera_ticks = _stream_ints(
            reader, f"cameras/{ARM_B_CAMERA_ID}", "physics_tick"
        )
        camera_indices = _unique_tick_index(camera_ticks, ARM_B_CAMERA_ID)
        camera_timestamps = _stream_ints(
            reader, f"cameras/{ARM_B_CAMERA_ID}", "timestamp_ns"
        )
        camera_hashes = _stream_texts(
            reader, f"cameras/{ARM_B_CAMERA_ID}", "image_sha256"
        )
        camera_fallback = _stream_bools(
            reader, f"cameras/{ARM_B_CAMERA_ID}", "is_fallback"
        )
        state_stream = reader.state_stream(ARM_B_ID)
        state_values = np.asarray(state_stream["state_7d"], dtype=np.float32)
        state_sequence_ids = _array_ints(
            state_stream["sequence_id"], "Arm_B sequence_id"
        )
        state_ticks = _array_ints(state_stream["physics_tick"], "Arm_B physics_tick")
        state_indices = _unique_tick_index(state_ticks, ARM_B_ID)
        state_timestamps = _array_ints(
            state_stream["timestamp_ns"], "Arm_B timestamp_ns"
        )

        raw_steps = []
        for action in reader.iter_valid_actions():
            if action.arm_id != ARM_B_ID or action.executor != OPENVLA_EXECUTOR:
                continue
            if action.subtask_id != ARM_B_TRANSPORT_SUBTASK_ID:
                raise _bad_episode("Arm_B OpenVLA action must use S02_ARM_B_TRANSPORT")
            camera_index = camera_indices.get(action.physics_tick)
            if camera_index is None:
                raise _bad_episode(
                    f"{ARM_B_CAMERA_ID} has no sample at action physics_tick "
                    f"{action.physics_tick}"
                )
            state_index = state_indices.get(action.physics_tick)
            if state_index is None:
                raise _bad_episode(
                    f"{ARM_B_ID} has no sample at action physics_tick "
                    f"{action.physics_tick}"
                )
            if camera_fallback[camera_index]:
                raise _bad_episode("CAM_B_TOP fallback frames are not trainable")
            image = np.asarray(frames[camera_index], dtype=np.uint8)
            _validate_image(image)
            state_7d = _vector7(state_values[state_index], "Arm_B state_7d")
            action_7d = _vector7(action.action_7d, "Arm_B action_7d")
            raw_steps.append(
                {
                    "image": image.copy(),
                    "state_7d": state_7d,
                    "action_7d": action_7d,
                    "source": CanonicalSource(
                        episode_id=episode_id,
                        task_id=task_id,
                        action_sequence_id=action.sequence_id,
                        action_physics_tick=action.physics_tick,
                        action_timestamp_ns=action.timestamp_ns,
                        camera_id=ARM_B_CAMERA_ID,
                        camera_sequence_id=camera_sequence_ids[camera_index],
                        camera_physics_tick=camera_ticks[camera_index],
                        camera_timestamp_ns=camera_timestamps[camera_index],
                        camera_image_sha256=camera_hashes[camera_index],
                        state_arm_id=ARM_B_ID,
                        state_sequence_id=state_sequence_ids[state_index],
                        state_physics_tick=state_ticks[state_index],
                        state_timestamp_ns=state_timestamps[state_index],
                        split=assignment.split.value,
                        split_registry_sha256=split_registry.registry_sha256,
                    ),
                }
            )
        if not raw_steps:
            raise _bad_episode(
                "canonical episode contains no valid Arm_B/openvla_oft actions"
            )
        final_index = len(raw_steps) - 1
        return tuple(
            OpenVLACanonicalStep(
                episode_id=episode_id,
                task_id=task_id,
                language_instruction=instruction,
                step_index=index,
                image=item["image"],
                state_7d=item["state_7d"],
                action_7d=item["action_7d"],
                is_first=index == 0,
                is_last=index == final_index,
                is_terminal=index == final_index,
                source=item["source"],
            )
            for index, item in enumerate(raw_steps)
        )
    finally:
        reader.close()


def _metadata(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    value = manifest.get("metadata")
    if not isinstance(value, Mapping):
        raise _bad_episode("canonical manifest metadata must be an object")
    if value.get("wrist_image") is not None:
        raise _bad_episode("OpenVLA-OFT requires wrist_image=null")
    if value.get("offline_gt_included") is not False:
        raise _bad_episode("online Canonical episode must not include GT")
    return value


def _camera_frames(reader: CanonicalEpisodeReader) -> np.ndarray:
    frames = np.asarray(reader.camera_frames(ARM_B_CAMERA_ID), dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[1:] != IMAGE_SHAPE:
        raise _bad_episode("CAM_B_TOP RGB must be uint8 [T,720,1280,3]")
    if frames.shape[0] < 1:
        raise _bad_episode("CAM_B_TOP stream is empty")
    return frames


def _stream_ints(
    reader: CanonicalEpisodeReader,
    group_path: str,
    dataset_name: str,
) -> tuple[int, ...]:
    h5 = getattr(reader, "_h5", None)
    if h5 is None:
        raise _bad_episode("canonical reader does not expose HDF5 streams")
    try:
        values = np.asarray(h5[f"{group_path}/{dataset_name}"][:])
    except (KeyError, TypeError, ValueError) as exc:
        raise _bad_episode(
            f"missing canonical stream {group_path}/{dataset_name}"
        ) from exc
    return _array_ints(values, f"{group_path}/{dataset_name}")


def _stream_texts(
    reader: CanonicalEpisodeReader,
    group_path: str,
    dataset_name: str,
) -> tuple[str, ...]:
    h5 = getattr(reader, "_h5", None)
    if h5 is None:
        raise _bad_episode("canonical reader does not expose HDF5 streams")
    try:
        values = h5[f"{group_path}/{dataset_name}"].asstr()[:].tolist()
    except (KeyError, TypeError, ValueError) as exc:
        raise _bad_episode(
            f"missing canonical stream {group_path}/{dataset_name}"
        ) from exc
    return tuple(str(item) for item in values)


def _stream_bools(
    reader: CanonicalEpisodeReader,
    group_path: str,
    dataset_name: str,
) -> tuple[bool, ...]:
    h5 = getattr(reader, "_h5", None)
    if h5 is None:
        raise _bad_episode("canonical reader does not expose HDF5 streams")
    try:
        values = np.asarray(h5[f"{group_path}/{dataset_name}"][:], dtype=np.bool_)
    except (KeyError, TypeError, ValueError) as exc:
        raise _bad_episode(
            f"missing canonical stream {group_path}/{dataset_name}"
        ) from exc
    if values.ndim != 1:
        raise _bad_episode(f"{group_path}/{dataset_name} must be one-dimensional")
    return tuple(bool(item) for item in values.tolist())


def _array_ints(values: Any, field_name: str) -> tuple[int, ...]:
    array = np.asarray(values)
    if array.ndim != 1:
        raise _bad_episode(f"{field_name} must be a one-dimensional stream")
    return tuple(int(item) for item in array.tolist())


def _unique_tick_index(ticks: tuple[int, ...], stream_name: str) -> dict[int, int]:
    result: dict[int, int] = {}
    for index, tick in enumerate(ticks):
        if tick in result:
            raise _bad_episode(f"{stream_name} contains duplicate physics_tick {tick}")
        result[tick] = index
    return result


def _validate_image(image: np.ndarray) -> None:
    if image.dtype != np.uint8 or image.shape != IMAGE_SHAPE:
        raise _bad_episode("CAM_B_TOP image must be uint8 [720,1280,3]")


def _vector7(
    value: Any, field_name: str
) -> tuple[float, float, float, float, float, float, float]:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (STATE_DIM,) or not np.all(np.isfinite(array)):
        raise _bad_episode(f"{field_name} must be finite 7-D")
    values = tuple(float(item) for item in array.tolist())
    if len(values) != ACTION_DIM:
        raise _bad_episode(f"{field_name} must be finite 7-D")
    return values  # type: ignore[return-value]


def _required_text(mapping: Mapping[str, Any], field_name: str) -> str:
    value = mapping.get(field_name)
    if not isinstance(value, str) or not value:
        raise _bad_episode(
            f"canonical metadata.{field_name} must be a non-empty string"
        )
    return value


def _bad_episode(message: str) -> ServiceError:
    return ServiceError("DATA_3002_CANONICAL_EPISODE_INVALID", message, retryable=False)
