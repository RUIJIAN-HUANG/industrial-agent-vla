"""Fail-closed reader for the frozen Canonical Episode V2 contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterator, Mapping

import h5py
import numpy as np
from jsonschema import Draft202012Validator, FormatChecker


CANONICAL_SCHEMA_VERSION = "2.0"
EXPECTED_SCENE_ID = "single_bin_manual_industrial_v2"
EXPECTED_TASK_ID = "P01_TO_S11"
EXPECTED_INSTRUCTION = "把P01放到S11中"
EXPECTED_TASK_INSTRUCTIONS = {
    EXPECTED_TASK_ID: EXPECTED_INSTRUCTION,
    "W01_TO_S14": "把W01放到S14中",
    "BIN01_TO_FINISHED01": "把Bin_01搬到FINISHED_01",
}
EXPECTED_ARM_ID = "Arm_A"
EXPECTED_EXECUTOR = "pi05"
EXPECTED_TASK_ACTION_IDENTITIES = {
    "P01_TO_S11": ("Arm_A", "pi05"),
    "W01_TO_S14": ("Arm_A", "pi05"),
    "BIN01_TO_FINISHED01": ("Arm_B", "openvla_oft"),
}
EXPECTED_TASK_CAMERA_IDS = {
    "P01_TO_S11": "CAM_A_TOP",
    "W01_TO_S14": "CAM_A_TOP",
    "BIN01_TO_FINISHED01": "CAM_B_TOP",
}
EXPECTED_CAMERA_IDS = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
EXPECTED_ARM_IDS = ("Arm_A", "Arm_B")
STATE_DIM = 7
ACTION_DIM = 7
PHYSICS_HZ = 120
CONTROL_HZ = 60
RENDER_HZ = 30
MODEL_INFERENCE_HZ = 10
ACTION_DURATION_MS = 100
SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "schemas" / "canonical-episode-v2.schema.json"
)


class CanonicalV2Error(ValueError):
    """A stable validation failure with Episode and field context."""

    def __init__(self, message: str, *, episode_id: str, field: str) -> None:
        self.episode_id = episode_id
        self.field = field
        super().__init__(f"episode_id={episode_id!r} field={field!r}: {message}")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _decoded_strings(dataset: h5py.Dataset) -> list[str]:
    return [str(value) for value in dataset.asstr()[:].tolist()]


def _dtype_name(dataset: h5py.Dataset) -> str:
    return "utf-8" if h5py.check_string_dtype(dataset.dtype) else str(dataset.dtype)


class CanonicalV2Reader:
    """Validate one ``structure.json + episode.h5`` V2 Episode.

    JSON Schema validates the manifest envelope. This reader additionally checks
    the referenced HDF5 file and the value-level invariants JSON Schema cannot
    express, including finite 7-D vectors and the no-padding policy.
    """

    def __init__(
        self,
        episode_dir: str | Path,
        *,
        schema_path: str | Path = SCHEMA_PATH,
    ) -> None:
        root = Path(episode_dir)
        provisional_id = root.name or "<unknown>"
        if root.is_symlink() or not root.is_dir():
            raise CanonicalV2Error(
                "Episode directory is missing or is a symlink",
                episode_id=provisional_id,
                field="episode_dir",
            )
        self.episode_path = root.resolve()
        self._h5: h5py.File | None = None

        structure_path = self.episode_path / "structure.json"
        hdf5_path = self.episode_path / "episode.h5"
        if structure_path.is_symlink() or not structure_path.is_file():
            raise CanonicalV2Error(
                "structure.json is missing or unsafe",
                episode_id=provisional_id,
                field="structure.json",
            )
        if hdf5_path.is_symlink() or not hdf5_path.is_file():
            raise CanonicalV2Error(
                "episode.h5 is missing or unsafe",
                episode_id=provisional_id,
                field="episode.h5",
            )

        try:
            schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
            manifest = json.loads(structure_path.read_text(encoding="utf-8"))
            Draft202012Validator.check_schema(schema)
            Draft202012Validator(
                schema,
                format_checker=FormatChecker(),
            ).validate(manifest)
        except Exception as exc:
            raise CanonicalV2Error(
                f"V2 manifest validation failed: {exc}",
                episode_id=provisional_id,
                field="structure.json",
            ) from exc
        if not isinstance(manifest, dict):
            raise CanonicalV2Error(
                "manifest must be an object",
                episode_id=provisional_id,
                field="structure.json",
            )
        self.manifest: dict[str, Any] = manifest
        self.episode_id = str(manifest["metadata"]["episode_id"])

        expected_sha = str(manifest["storage"]["sha256"])
        actual_sha = _sha256_file(hdf5_path)
        if actual_sha != expected_sha:
            raise CanonicalV2Error(
                f"HDF5 SHA-256 mismatch: expected={expected_sha} actual={actual_sha}",
                episode_id=self.episode_id,
                field="storage.sha256",
            )

        try:
            self._h5 = h5py.File(hdf5_path, "r")
            self._validate_hdf5()
        except CanonicalV2Error:
            self.close()
            raise
        except Exception as exc:
            self.close()
            raise CanonicalV2Error(
                f"HDF5 validation failed: {exc}",
                episode_id=self.episode_id,
                field="episode.h5",
            ) from exc

    @property
    def h5(self) -> h5py.File:
        if self._h5 is None or not self._h5.id.valid:
            raise RuntimeError("Canonical V2 reader is closed")
        return self._h5

    def _fail(self, message: str, field: str) -> None:
        raise CanonicalV2Error(message, episode_id=self.episode_id, field=field)

    def _validate_hdf5(self) -> None:
        if set(self.h5.keys()) != {"cameras", "robot_state", "actions"}:
            self._fail(
                "root groups must be exactly cameras, robot_state, actions",
                "episode.h5.groups",
            )
        metadata = self.manifest["metadata"]
        task_id = str(metadata["task_id"])
        instruction = str(metadata["instruction"])
        if EXPECTED_TASK_INSTRUCTIONS.get(task_id) != instruction:
            self._fail("unsupported task/instruction pair", "metadata.task_id")
        expected_attributes = {
            "canonical_schema_version": CANONICAL_SCHEMA_VERSION,
            "episode_id": self.episode_id,
            "scene_id": EXPECTED_SCENE_ID,
            "task_id": task_id,
            "instruction": instruction,
            "scene_seed": metadata["scene_seed"],
            "git_sha": metadata["git_sha"],
            "scene_config_sha256": metadata["scene_config_sha256"],
            "outcome": metadata["outcome"],
            "failure_code": metadata["failure_code"] or "",
            "wrist_image": "null",
            "offline_gt_included": False,
            "physics_hz": PHYSICS_HZ,
            "control_hz": CONTROL_HZ,
            "render_hz": RENDER_HZ,
            "model_inference_hz": MODEL_INFERENCE_HZ,
            "padding_strategy": "none",
            "padding_target_length": -1,
        }
        if set(self.h5.attrs.keys()) != set(expected_attributes):
            self._fail(
                "HDF5 attributes contain missing or forbidden names",
                "episode.h5.attrs",
            )
        for name, expected in expected_attributes.items():
            actual = self.h5.attrs.get(name)
            if actual != expected:
                self._fail(
                    f"attribute mismatch: expected={expected!r} actual={actual!r}",
                    f"episode.h5.attrs.{name}",
                )

        if set(self.h5["cameras"].keys()) != set(EXPECTED_CAMERA_IDS):
            self._fail("camera IDs do not match V2", "cameras")
        if set(self.h5["robot_state"].keys()) != set(EXPECTED_ARM_IDS):
            self._fail("robot state arm IDs do not match V2", "robot_state")

        stream_manifest = self.manifest["streams"]
        for camera_id in EXPECTED_CAMERA_IDS:
            self._validate_stream_descriptor(
                stream_manifest["cameras"][camera_id],
                self.h5[f"cameras/{camera_id}"],
            )
            self._validate_timeline(
                self.h5[f"cameras/{camera_id}"],
                stride=PHYSICS_HZ // RENDER_HZ,
            )
        for arm_id in EXPECTED_ARM_IDS:
            self._validate_stream_descriptor(
                stream_manifest["robot_state"][arm_id],
                self.h5[f"robot_state/{arm_id}"],
            )
            self._validate_timeline(
                self.h5[f"robot_state/{arm_id}"],
                stride=PHYSICS_HZ // CONTROL_HZ,
            )
        self._validate_stream_descriptor(
            stream_manifest["actions"],
            self.h5["actions"],
        )
        self._validate_timeline(
            self.h5["actions"],
            stride=PHYSICS_HZ // MODEL_INFERENCE_HZ,
        )
        self._validate_synchronization()
        self._validate_cameras()
        self._validate_states()
        self._validate_actions()

    def _validate_stream_descriptor(
        self,
        manifest: Mapping[str, Any],
        group: h5py.Group,
    ) -> None:
        if set(manifest["datasets"]) != set(group.keys()):
            self._fail("manifest dataset names do not match HDF5", group.name)
        count = int(manifest["count"])
        for name, descriptor in manifest["datasets"].items():
            dataset = group[name]
            if dataset.shape[0] != count:
                self._fail("dataset length does not match stream count", dataset.name)
            if descriptor["path"] != dataset.name:
                self._fail("dataset path does not match HDF5", dataset.name)
            if descriptor["shape"] != [int(value) for value in dataset.shape]:
                self._fail("dataset shape does not match HDF5", dataset.name)
            if descriptor["dtype"] != _dtype_name(dataset):
                self._fail("dataset dtype does not match HDF5", dataset.name)

    def _validate_timeline(self, group: h5py.Group, *, stride: int) -> None:
        sequence_ids = np.asarray(group["sequence_id"][:], dtype=np.uint64)
        timestamps = np.asarray(group["timestamp_ns"][:], dtype=np.uint64)
        ticks = np.asarray(group["physics_tick"][:], dtype=np.uint64)
        if not np.array_equal(
            sequence_ids,
            np.arange(len(sequence_ids), dtype=np.uint64),
        ):
            self._fail("sequence_id must be contiguous from zero", group.name)
        timestamp_values = [int(value) for value in timestamps]
        if any(
            current <= previous
            for previous, current in zip(timestamp_values, timestamp_values[1:])
        ):
            self._fail("timestamps must be strictly increasing", group.name)
        if np.any(ticks % stride):
            self._fail("physics ticks are off-grid", group.name)
        if len(ticks) > 1:
            tick_values = [int(value) for value in ticks]
            deltas = [
                current - previous
                for previous, current in zip(tick_values, tick_values[1:])
            ]
            if any(delta <= 0 or delta % stride for delta in deltas):
                self._fail("physics tick gaps are invalid", group.name)

    def _validate_synchronization(self) -> None:
        camera_ticks = [
            np.asarray(self.h5[f"cameras/{camera_id}/physics_tick"][:])
            for camera_id in EXPECTED_CAMERA_IDS
        ]
        camera_times = [
            np.asarray(self.h5[f"cameras/{camera_id}/timestamp_ns"][:])
            for camera_id in EXPECTED_CAMERA_IDS
        ]
        if any(not np.array_equal(item, camera_ticks[0]) for item in camera_ticks[1:]):
            self._fail("camera physics ticks are not synchronized", "cameras")
        if any(not np.array_equal(item, camera_times[0]) for item in camera_times[1:]):
            self._fail("camera timestamps are not synchronized", "cameras")
        arm_ticks = [
            np.asarray(self.h5[f"robot_state/{arm_id}/physics_tick"][:])
            for arm_id in EXPECTED_ARM_IDS
        ]
        if not np.array_equal(arm_ticks[0], arm_ticks[1]):
            self._fail(
                "Arm_A and Arm_B state ticks are not synchronized", "robot_state"
            )

    def _validate_cameras(self) -> None:
        for camera_id in EXPECTED_CAMERA_IDS:
            group = self.h5[f"cameras/{camera_id}"]
            rgb = group["rgb"]
            if rgb.dtype != np.uint8 or rgb.shape[1:] != (720, 1280, 3):
                self._fail("RGB must be uint8[N,720,1280,3]", rgb.name)
            uris = _decoded_strings(group["cas_uri"])
            digests = _decoded_strings(group["image_sha256"])
            for uri, digest in zip(uris, digests, strict=True):
                if not digest.startswith("sha256:") or uri != (
                    f"cas://sha256/{digest.split(':', 1)[-1]}"
                ):
                    self._fail("CAS URI and image digest do not match", group.name)
            fallback_count = int(np.count_nonzero(group["is_fallback"][:]))
            expected = int(
                self.manifest["streams"]["cameras"][camera_id]["fallback_count"]
            )
            if fallback_count != expected:
                self._fail("fallback_count does not match HDF5", group.name)
        if np.any(self.h5["cameras/CAM_A_TOP/is_fallback"][:]):
            self._fail("CAM_A_TOP fallback frames are forbidden", "cameras.CAM_A_TOP")

    def _validate_states(self) -> None:
        for arm_id in EXPECTED_ARM_IDS:
            dataset = self.h5[f"robot_state/{arm_id}/state_7d"]
            values = np.asarray(dataset[:])
            if dataset.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 7:
                self._fail("state must be float32[N,7]", dataset.name)
            if not np.all(np.isfinite(values)):
                self._fail("state contains NaN or Infinity", dataset.name)
            if np.any(values[:, 6] < 0.0) or np.any(values[:, 6] > 1.0):
                self._fail("state gripper must be in [0,1]", dataset.name)

    def _validate_actions(self) -> None:
        group = self.h5["actions"]
        dataset = group["action_7d"]
        values = np.asarray(dataset[:])
        if dataset.dtype != np.float32 or values.ndim != 2 or values.shape[1] != 7:
            self._fail("action must be float32[N,7]", dataset.name)
        if not np.all(np.isfinite(values)):
            self._fail("action contains NaN or Infinity", dataset.name)
        if not np.all(np.isin(values[:, 6], np.asarray([0.0, 1.0], dtype=np.float32))):
            self._fail(
                "action gripper must be exactly 0.0 or 1.0",
                dataset.name,
            )
        if not np.all(np.asarray(group["duration_ms"][:]) == ACTION_DURATION_MS):
            self._fail("every action duration must be 100 ms", "actions.duration_ms")
        valid_mask = np.asarray(group["valid_mask"][:], dtype=np.bool_)
        if not np.all(valid_mask):
            self._fail("padding/masked action rows are forbidden", "actions.valid_mask")
        count = int(self.manifest["streams"]["actions"]["count"])
        valid_count = int(self.manifest["streams"]["actions"]["valid_count"])
        if valid_count != count or valid_count != int(np.count_nonzero(valid_mask)):
            self._fail("valid_count must equal count", "actions.valid_count")
        task_id = str(self.manifest["metadata"]["task_id"])
        expected_arm_id, expected_executor = EXPECTED_TASK_ACTION_IDENTITIES[task_id]
        identities = {
            "arm_id": expected_arm_id,
            "executor": expected_executor,
            "subtask_id": task_id,
        }
        for field, expected in identities.items():
            values_text = _decoded_strings(group[field])
            if any(value != expected for value in values_text):
                self._fail(
                    f"every value must equal {expected!r}",
                    f"actions.{field}",
                )

    def iter_action_7d(self) -> Iterator[np.ndarray]:
        """Yield defensive copies of validated V2 action rows."""

        for row in self.h5["actions/action_7d"]:
            yield np.ascontiguousarray(row, dtype=np.float32)

    def state_7d(self, arm_id: str) -> np.ndarray:
        """Return a defensive copy of one validated state stream."""

        if arm_id not in EXPECTED_ARM_IDS:
            raise ValueError(f"arm_id must be one of {EXPECTED_ARM_IDS}")
        return np.asarray(
            self.h5[f"robot_state/{arm_id}/state_7d"][:],
            dtype=np.float32,
        ).copy()

    def close(self) -> None:
        if self._h5 is not None and self._h5.id.valid:
            self._h5.close()

    def __enter__(self) -> CanonicalV2Reader:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        del exc_type, exc, traceback
        self.close()


def read_canonical_v2_episode(episode_dir: str | Path) -> CanonicalV2Reader:
    """Open and validate one V2 Episode; caller must close the returned reader."""

    return CanonicalV2Reader(episode_dir)


__all__ = [
    "ACTION_DIM",
    "CANONICAL_SCHEMA_VERSION",
    "CanonicalV2Error",
    "CanonicalV2Reader",
    "EXPECTED_INSTRUCTION",
    "EXPECTED_SCENE_ID",
    "EXPECTED_TASK_ID",
    "EXPECTED_TASK_ACTION_IDENTITIES",
    "EXPECTED_TASK_CAMERA_IDS",
    "EXPECTED_TASK_INSTRUCTIONS",
    "STATE_DIM",
    "read_canonical_v2_episode",
]
