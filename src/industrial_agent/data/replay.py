"""Verified reader and offline-only replay for Canonical HDF5 episodes."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import h5py
import numpy as np

from industrial_agent.sync_contract import FROZEN_MULTI_RATE

from .recorder import (
    ACTION_DIM,
    ARM_IDS,
    CAMERA_IDS,
    CANONICAL_EPISODE_VERSION,
    EXECUTOR_BY_ARM,
    IMAGE_HEIGHT,
    IMAGE_WIDTH,
    STATE_DIM,
    _FINAL_OUTCOMES,
    _file_sha256,
)
from .padding import PaddingPolicy
from .split_registry import DataLeakageError, SplitAssignment, SplitRegistry


def _decoded_strings(dataset: h5py.Dataset) -> list[str]:
    values = dataset.asstr()[:]
    return [str(value) for value in values.tolist()]


@dataclass(frozen=True)
class OfflineReplayAction:
    """One verified, non-padding action for offline analysis only."""

    sequence_id: int
    timestamp_ns: int
    physics_tick: int
    arm_id: str
    executor: str
    subtask_id: str
    chunk_id: str
    chunk_position: int
    action_7d: tuple[float, float, float, float, float, float, float]
    duration_ms: int


class CanonicalEpisodeReader:
    """Fail-closed reader that verifies manifest, hash, and HDF5 layout."""

    def __init__(
        self,
        episode_path: str | Path,
        *,
        split_registry: SplitRegistry | None = None,
        is_training: bool = False,
    ) -> None:
        if not isinstance(is_training, bool):
            raise TypeError("is_training must be a bool")
        if split_registry is not None and not isinstance(split_registry, SplitRegistry):
            raise TypeError("split_registry must be SplitRegistry or None")
        if is_training and split_registry is None:
            raise DataLeakageError("training access requires a verified split registry")
        root = Path(episode_path).expanduser()
        if root.is_symlink() or not root.is_dir():
            raise ValueError("episode_path must be a real episode directory")
        self.episode_path = root.resolve()
        structure_path = self.episode_path / "structure.json"
        if structure_path.is_symlink() or not structure_path.is_file():
            raise ValueError("episode structure.json is missing or unsafe")
        try:
            raw_manifest = json.loads(structure_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("episode structure.json is unreadable") from exc
        if not isinstance(raw_manifest, dict):
            raise ValueError("episode structure.json must contain an object")
        self.manifest: dict[str, Any] = raw_manifest
        self._validate_manifest_envelope()
        self.split_assignment: SplitAssignment | None = None
        if split_registry is not None:
            self.split_assignment = split_registry.assert_episode_allowed(
                self.manifest["metadata"]["episode_id"],
                is_training=is_training,
            )

        hdf5_path = self.episode_path / "episode.h5"
        if hdf5_path.is_symlink() or not hdf5_path.is_file():
            raise ValueError("episode.h5 is missing or unsafe")
        expected_digest = self.manifest["storage"]["sha256"]
        actual_digest = _file_sha256(hdf5_path)
        if actual_digest != expected_digest:
            raise ValueError(
                f"episode.h5 SHA-256 mismatch: expected {expected_digest}, "
                f"got {actual_digest}"
            )
        try:
            self._h5 = h5py.File(hdf5_path, "r")
            self._validate_hdf5()
        except (OSError, RuntimeError, TypeError, ValueError):
            if getattr(self, "_h5", None) is not None and self._h5.id.valid:
                self._h5.close()
            raise

    def _validate_manifest_envelope(self) -> None:
        if set(self.manifest) != {"schema_version", "metadata", "storage", "streams"}:
            raise ValueError("canonical manifest has invalid top-level fields")
        if self.manifest["schema_version"] != CANONICAL_EPISODE_VERSION:
            raise ValueError("unsupported canonical episode schema version")
        metadata = self.manifest.get("metadata")
        storage = self.manifest.get("storage")
        streams = self.manifest.get("streams")
        if not all(isinstance(item, dict) for item in (metadata, storage, streams)):
            raise ValueError("canonical manifest sections must be objects")
        if (
            storage.get("format") != "hdf5"
            or storage.get("episode_file") != "episode.h5"
        ):
            raise ValueError("canonical storage contract must use episode.h5")
        if storage.get("structure_file") != "structure.json":
            raise ValueError("canonical structure filename is invalid")
        if metadata.get("wrist_image") is not None:
            raise ValueError("frozen episode contract requires wrist_image=null")
        if metadata.get("offline_gt_included") is not False:
            raise ValueError("online/canonical episode must not include offline GT")
        expected_rates = {
            "physics_hz": FROZEN_MULTI_RATE.physics_hz,
            "control_hz": FROZEN_MULTI_RATE.control_hz,
            "render_hz": FROZEN_MULTI_RATE.render_hz,
            "model_inference_hz": FROZEN_MULTI_RATE.model_inference_hz,
        }
        if metadata.get("frequency_contract") != expected_rates:
            raise ValueError(
                "episode frequency contract does not match 120/60/30/10 Hz"
            )
        outcome = metadata.get("outcome")
        failure_code = metadata.get("failure_code")
        if outcome not in _FINAL_OUTCOMES:
            raise ValueError("episode outcome is not a frozen terminal state")
        if outcome == "SUCCEEDED" and failure_code is not None:
            raise ValueError("SUCCEEDED episode must not contain a failure code")
        if outcome != "SUCCEEDED" and (
            not isinstance(failure_code, str) or not failure_code
        ):
            raise ValueError("non-success episode must contain a failure code")
        padding_policy = metadata.get("padding_policy")
        try:
            PaddingPolicy.from_mapping(padding_policy)
        except (TypeError, ValueError) as exc:
            raise ValueError("episode padding policy is invalid") from exc

    def _validate_hdf5(self) -> None:
        if set(self._h5.keys()) != {"cameras", "robot_state", "actions"}:
            raise ValueError("episode.h5 contains missing or forbidden root groups")
        if self._h5.attrs.get("wrist_image") != "null":
            raise ValueError("HDF5 wrist_image marker must be null")
        if bool(self._h5.attrs.get("offline_gt_included")):
            raise ValueError("episode.h5 must not contain offline GT")
        metadata = self.manifest["metadata"]
        attribute_pairs = {
            "schema_version": self.manifest["schema_version"],
            "episode_id": metadata.get("episode_id"),
            "scene_id": metadata.get("scene_id"),
            "task_id": metadata.get("task_id"),
            "instruction": metadata.get("instruction"),
            "scene_seed": metadata.get("scene_seed"),
            "git_sha": metadata.get("git_sha"),
            "scene_config_sha256": metadata.get("scene_config_sha256"),
            "outcome": metadata.get("outcome"),
            "failure_code": metadata.get("failure_code") or "",
        }
        for name, expected in attribute_pairs.items():
            if self._h5.attrs.get(name) != expected:
                raise ValueError(f"HDF5 attribute {name} does not match manifest")
        expected_rate_attrs = {
            "physics_hz": FROZEN_MULTI_RATE.physics_hz,
            "control_hz": FROZEN_MULTI_RATE.control_hz,
            "render_hz": FROZEN_MULTI_RATE.render_hz,
            "model_inference_hz": FROZEN_MULTI_RATE.model_inference_hz,
        }
        for name, expected in expected_rate_attrs.items():
            if int(self._h5.attrs.get(name, -1)) != expected:
                raise ValueError(
                    f"HDF5 attribute {name} violates the frequency contract"
                )
        if set(self._h5["cameras"].keys()) != set(CAMERA_IDS):
            raise ValueError("episode.h5 must contain exactly the three frozen cameras")
        if set(self._h5["robot_state"].keys()) != set(ARM_IDS):
            raise ValueError("episode.h5 must contain exactly Arm_A and Arm_B state")

        stream_manifest = self.manifest["streams"]
        self._validate_streams(
            stream_manifest["cameras"],
            {camera_id: self._h5[f"cameras/{camera_id}"] for camera_id in CAMERA_IDS},
            expected_frequency_hz=FROZEN_MULTI_RATE.render_hz,
        )
        self._validate_streams(
            stream_manifest["robot_state"],
            {arm_id: self._h5[f"robot_state/{arm_id}"] for arm_id in ARM_IDS},
            expected_frequency_hz=FROZEN_MULTI_RATE.control_hz,
        )
        self._validate_one_stream(
            stream_manifest["actions"],
            self._h5["actions"],
            expected_frequency_hz=FROZEN_MULTI_RATE.model_inference_hz,
        )

        camera_ticks = [
            np.asarray(self._h5[f"cameras/{camera_id}/physics_tick"][:])
            for camera_id in CAMERA_IDS
        ]
        camera_times = [
            np.asarray(self._h5[f"cameras/{camera_id}/timestamp_ns"][:])
            for camera_id in CAMERA_IDS
        ]
        if any(
            not np.array_equal(value, camera_ticks[0]) for value in camera_ticks[1:]
        ):
            raise ValueError("camera render ticks are not synchronized")
        if any(
            not np.array_equal(value, camera_times[0]) for value in camera_times[1:]
        ):
            raise ValueError("camera timestamps are not synchronized")
        arm_ticks = [
            np.asarray(self._h5[f"robot_state/{arm_id}/physics_tick"][:])
            for arm_id in ARM_IDS
        ]
        if not np.array_equal(arm_ticks[0], arm_ticks[1]):
            raise ValueError("Arm_A and Arm_B state ticks are not synchronized")

        for camera_id in CAMERA_IDS:
            group = self._h5[f"cameras/{camera_id}"]
            self._validate_timeline(
                group,
                stride=FROZEN_MULTI_RATE.physics_ticks_per_render,
            )
            rgb = group["rgb"]
            if rgb.dtype != np.uint8 or rgb.shape[1:] != (
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                3,
            ):
                raise ValueError(
                    f"{camera_id} RGB dataset has an invalid shape or dtype"
                )
            uris = _decoded_strings(group["cas_uri"])
            digests = _decoded_strings(group["image_sha256"])
            for uri, digest in zip(uris, digests):
                digest_value = digest.split(":", 1)[-1]
                if (
                    not digest.startswith("sha256:")
                    or len(digest) != 71
                    or any(
                        character not in "0123456789abcdef"
                        for character in digest_value
                    )
                    or uri != f"cas://sha256/{digest_value}"
                ):
                    raise ValueError(f"{camera_id} contains invalid CAS lineage")
            fallback_count = int(np.count_nonzero(group["is_fallback"][:]))
            if (
                stream_manifest["cameras"][camera_id].get("fallback_count")
                != fallback_count
            ):
                raise ValueError(f"{camera_id} fallback count does not match manifest")
        for arm_id in ARM_IDS:
            group = self._h5[f"robot_state/{arm_id}"]
            self._validate_timeline(
                group,
                stride=FROZEN_MULTI_RATE.physics_ticks_per_control,
            )
            state = group["state_7d"]
            if state.dtype != np.float32 or state.shape[1:] != (STATE_DIM,):
                raise ValueError(f"{arm_id} state_7d dataset is invalid")
            state_values = np.asarray(state[:], dtype=np.float32)
            if not np.all(np.isfinite(state_values)):
                raise ValueError(f"{arm_id} state_7d contains non-finite values")
            if np.any(state_values[:, 6] < 0.0) or np.any(state_values[:, 6] > 1.0):
                raise ValueError(f"{arm_id} state gripper is outside [0, 1]")

        action_group = self._h5["actions"]
        self._validate_timeline(
            action_group,
            stride=FROZEN_MULTI_RATE.physics_ticks_per_model_step,
        )
        actions = action_group["action_7d"]
        if actions.dtype != np.float32 or actions.shape[1:] != (ACTION_DIM,):
            raise ValueError("action_7d dataset is invalid")
        action_values = np.asarray(actions[:], dtype=np.float32)
        if not np.all(np.isfinite(action_values)):
            raise ValueError("action_7d contains non-finite values")
        if np.any(action_values[:, 6] < -1.0) or np.any(action_values[:, 6] > 1.0):
            raise ValueError("action gripper is outside [-1, 1]")
        durations = np.asarray(action_group["duration_ms"][:])
        if not np.all(durations == FROZEN_MULTI_RATE.model_step_duration_ms):
            raise ValueError("action duration does not match the frozen 10 Hz contract")
        valid_mask = np.asarray(action_group["valid_mask"][:], dtype=np.bool_)
        if not np.any(valid_mask):
            raise ValueError("episode contains no valid action")
        valid_count = int(np.count_nonzero(valid_mask))
        if stream_manifest["actions"].get("valid_count") != valid_count:
            raise ValueError("action valid_count does not match manifest")
        arm_ids = _decoded_strings(action_group["arm_id"])
        executors = _decoded_strings(action_group["executor"])
        for arm_id, executor in zip(arm_ids, executors):
            if arm_id not in EXECUTOR_BY_ARM or EXECUTOR_BY_ARM[arm_id] != executor:
                raise ValueError(
                    "stored action violates the frozen arm/executor mapping"
                )

    def _validate_streams(
        self,
        manifests: Mapping[str, Any],
        groups: Mapping[str, h5py.Group],
        *,
        expected_frequency_hz: int,
    ) -> None:
        if not isinstance(manifests, Mapping) or set(manifests) != set(groups):
            raise ValueError("stream manifest names do not match HDF5 groups")
        for name, group in groups.items():
            self._validate_one_stream(
                manifests[name],
                group,
                expected_frequency_hz=expected_frequency_hz,
            )

    @staticmethod
    def _validate_one_stream(
        manifest: Mapping[str, Any],
        group: h5py.Group,
        *,
        expected_frequency_hz: int,
    ) -> None:
        if not isinstance(manifest, Mapping):
            raise ValueError("stream manifest must be an object")
        count = manifest.get("count")
        datasets = manifest.get("datasets")
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("stream count must be a positive integer")
        if manifest.get("frequency_hz") != expected_frequency_hz:
            raise ValueError("stream frequency does not match the frozen contract")
        if not isinstance(datasets, Mapping) or set(datasets) != set(group.keys()):
            raise ValueError("stream dataset manifest does not match HDF5")
        for name, descriptor in datasets.items():
            if not isinstance(descriptor, Mapping):
                raise ValueError("dataset descriptor must be an object")
            dataset = group[name]
            if descriptor.get("path") != dataset.name:
                raise ValueError(f"dataset path mismatch for {dataset.name}")
            if descriptor.get("shape") != [int(value) for value in dataset.shape]:
                raise ValueError(f"dataset shape mismatch for {dataset.name}")
            if dataset.shape[0] != count:
                raise ValueError(f"stream count mismatch for {dataset.name}")
            string_info = h5py.check_string_dtype(dataset.dtype)
            actual_dtype = "utf-8" if string_info is not None else str(dataset.dtype)
            if descriptor.get("dtype") != actual_dtype:
                raise ValueError(f"dataset dtype mismatch for {dataset.name}")

    @staticmethod
    def _validate_timeline(group: h5py.Group, *, stride: int) -> None:
        sequence_ids = np.asarray(group["sequence_id"][:], dtype=np.uint64)
        timestamps = np.asarray(group["timestamp_ns"][:], dtype=np.uint64)
        physics_ticks = np.asarray(group["physics_tick"][:], dtype=np.uint64)
        expected_sequence = np.arange(len(sequence_ids), dtype=np.uint64)
        if not np.array_equal(sequence_ids, expected_sequence):
            raise ValueError(f"{group.name} sequence_id is not contiguous")
        timestamp_values = [int(value) for value in timestamps]
        if any(
            current <= previous
            for previous, current in zip(timestamp_values, timestamp_values[1:])
        ):
            raise ValueError(f"{group.name} timestamps are not strictly increasing")
        if np.any(physics_ticks % stride):
            raise ValueError(f"{group.name} physics ticks are off-grid")
        if len(physics_ticks) > 1:
            tick_values = [int(value) for value in physics_ticks]
            tick_deltas = [
                current - previous
                for previous, current in zip(tick_values, tick_values[1:])
            ]
            if any(delta <= 0 or delta % stride for delta in tick_deltas):
                raise ValueError(f"{group.name} physics tick gaps are off-grid")

    def camera_frames(self, camera_id: str) -> np.ndarray:
        if camera_id not in CAMERA_IDS:
            raise ValueError(f"camera_id must be one of {CAMERA_IDS}")
        return np.asarray(self._h5[f"cameras/{camera_id}/rgb"][:]).copy()

    def state_stream(self, arm_id: str) -> dict[str, np.ndarray]:
        if arm_id not in ARM_IDS:
            raise ValueError(f"arm_id must be one of {ARM_IDS}")
        group = self._h5[f"robot_state/{arm_id}"]
        return {name: np.asarray(dataset[:]).copy() for name, dataset in group.items()}

    def iter_valid_actions(self) -> Iterator[OfflineReplayAction]:
        group = self._h5["actions"]
        arm_ids = _decoded_strings(group["arm_id"])
        executors = _decoded_strings(group["executor"])
        subtasks = _decoded_strings(group["subtask_id"])
        chunk_ids = _decoded_strings(group["chunk_id"])
        valid_mask = np.asarray(group["valid_mask"][:], dtype=np.bool_)
        for index in np.flatnonzero(valid_mask):
            values = tuple(float(item) for item in group["action_7d"][index])
            if len(values) != ACTION_DIM:
                raise ValueError("stored replay action is not 7-D")
            yield OfflineReplayAction(
                sequence_id=int(group["sequence_id"][index]),
                timestamp_ns=int(group["timestamp_ns"][index]),
                physics_tick=int(group["physics_tick"][index]),
                arm_id=arm_ids[index],
                executor=executors[index],
                subtask_id=subtasks[index],
                chunk_id=chunk_ids[index],
                chunk_position=int(group["chunk_position"][index]),
                action_7d=values,  # type: ignore[arg-type]
                duration_ms=int(group["duration_ms"][index]),
            )

    def close(self) -> None:
        if getattr(self, "_h5", None) is not None and self._h5.id.valid:
            self._h5.close()

    def __enter__(self) -> CanonicalEpisodeReader:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


class OfflineEpisodeReplay:
    """Offline iterator with no controller or environment dependency."""

    def __init__(self, reader: CanonicalEpisodeReader) -> None:
        if not isinstance(reader, CanonicalEpisodeReader):
            raise TypeError("reader must be CanonicalEpisodeReader")
        self.reader = reader

    def actions(self) -> tuple[OfflineReplayAction, ...]:
        """Return only valid rows; masked padding is never replayed."""

        return tuple(self.reader.iter_valid_actions())
