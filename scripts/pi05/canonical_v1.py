"""PI05 adapter over the authoritative Canonical HDF5 reader.

The framework-owned :class:`industrial_agent.data.CanonicalEpisodeReader`
remains the only parser and validator for ``episode.h5 + structure.json``.
This module only applies the role-E projection: verified Train/Val/Test split,
Arm_A + pi05 action filtering, and exact physics-tick joins to CAM_A_TOP and
Arm_A state.  It never falls back to another camera or a neighbouring sample.
"""

from __future__ import annotations

import hashlib
import importlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np

from industrial_agent.data import CanonicalEpisodeReader, SplitRegistry


CANONICAL_SCHEMA_VERSION = "1.0"
EXPECTED_ROBOT_ROLE = "arm_a_pi05"
EXPECTED_ARM_ID = "Arm_A"
EXPECTED_EXECUTOR = "pi05"
EXPECTED_CAMERA_ID = "CAM_A_TOP"
EXPECTED_IMAGE_SIZE = (1280, 720)
VALID_SPLITS = frozenset({"train", "val", "test"})
CANONICAL_TCP_POSE_ORDER = ("x", "y", "z", "qx", "qy", "qz", "qw")
CANONICAL_QUATERNION_ORDER_XYZW = ("qx", "qy", "qz", "qw")
LIBRARY_QUATERNION_ORDER_WXYZ = ("qw", "qx", "qy", "qz")
STATE_7D_ORDER = (
    "x_m",
    "y_m",
    "z_m",
    "ax_rad",
    "ay_rad",
    "az_rad",
    "gripper_norm",
)


class CanonicalV1Error(ValueError):
    """Fail-closed PI05 projection error with stable source context."""

    def __init__(
        self,
        message: str,
        *,
        episode_id: str,
        field: str,
        step_index: int | None = None,
    ) -> None:
        self.episode_id = episode_id
        self.step_index = step_index
        self.field = field
        step_text = "episode" if step_index is None else str(step_index)
        super().__init__(
            f"episode_id={episode_id!r} step_index={step_text} "
            f"field={field!r}: {message}"
        )


@dataclass(frozen=True)
class CanonicalStep:
    """One valid Arm_A/pi05 action aligned to exact source samples.

    ``state_7d`` and ``action_7d`` are finite ``float32[7]`` arrays in
    ``robot_base``.  Translation uses metres, rotation uses radians as a
    rotation-vector, and gripper values follow the frozen normalized contract.
    ``cam_a_top_rgb`` is raw ``uint8[720,1280,3]`` RGB.
    """

    step_index: int
    timestamp_ns: int
    observation_id: str
    state_7d: np.ndarray
    action_7d: np.ndarray
    action_duration_s: float
    valid_for_training: bool
    cam_a_top_rgb: np.ndarray
    cam_a_top_relative_path: str
    cam_a_top_sha256: str
    physics_tick: int
    action_sequence_id: int
    camera_sequence_id: int
    camera_timestamp_ns: int
    state_sequence_id: int
    state_timestamp_ns: int
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class CanonicalEpisode:
    """Role-E view of one framework-validated Canonical Episode."""

    root: Path
    episode_id: str
    split: str
    instruction: str
    robot_role: str
    eligible_for_imitation: bool
    meta: Mapping[str, Any]
    steps: tuple[CanonicalStep, ...]
    recorder_git_sha: str
    structure_sha256: str
    hdf5_sha256: str
    split_registry_sha256: str

    @property
    def training_steps(self) -> tuple[CanonicalStep, ...]:
        """Return valid rows; masked padding never enters this projection."""

        return self.steps if self.eligible_for_imitation else ()

    @property
    def imitation_steps(self) -> tuple[CanonicalStep, ...]:
        """Return the complete role-E Episode or fail closed."""

        if not self.eligible_for_imitation or not self.steps:
            raise CanonicalV1Error(
                "Episode contains no eligible Arm_A/pi05 actions",
                episode_id=self.episode_id,
                field="actions",
            )
        return self.steps


@runtime_checkable
class StateMapper(Protocol):
    """Explicit mapping from aligned Canonical state to model state."""

    name: str
    state_dim: int
    approved_for_production: bool

    def map_state(self, episode: CanonicalEpisode, step: CanonicalStep) -> np.ndarray:
        """Return finite ``float32[state_dim]`` for one aligned action."""


def quaternion_xyzw_to_rotation_vector(
    quaternion_xyzw: np.ndarray,
    *,
    episode_id: str,
    step_index: int,
) -> np.ndarray:
    """Compatibility utility for historical fixtures; production does not use it."""

    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise CanonicalV1Error(
            "quaternion must be finite float[4] in xyzw order",
            episode_id=episode_id,
            step_index=step_index,
            field="tcp_pose.quaternion_xyzw",
        )
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= np.finfo(np.float64).eps:
        raise CanonicalV1Error(
            "quaternion norm must be non-zero",
            episode_id=episode_id,
            step_index=step_index,
            field="tcp_pose.quaternion_xyzw",
        )
    quaternion /= norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    vector = quaternion[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = min(max(2.0 * math.atan2(vector_norm, float(quaternion[3])), 0.0), math.pi)
    return np.ascontiguousarray(vector * (angle / vector_norm), dtype=np.float32)


class CanonicalPi05StateMapper:
    """Approved identity mapper for the framework-frozen state_7d stream."""

    name = "canonical_pi05_state_7d_hdf5_v1"
    version = "1.0"
    state_dim = 7
    approved_for_production = True

    def map_state(self, episode: CanonicalEpisode, step: CanonicalStep) -> np.ndarray:
        del episode
        return step.state_7d.copy()


def require_state_mapper(
    mapper: StateMapper | None,
    *,
    production: bool,
) -> StateMapper:
    """Validate an explicitly injected mapper and its approval status."""

    if mapper is None:
        raise RuntimeError("PI05 StateMapper is required; no implicit default exists")
    for attribute in ("name", "state_dim", "approved_for_production", "map_state"):
        if not hasattr(mapper, attribute):
            raise TypeError(f"StateMapper is missing required attribute {attribute!r}")
    if not isinstance(mapper.name, str) or not mapper.name.strip():
        raise TypeError("StateMapper.name must be a non-empty string")
    if (
        isinstance(mapper.state_dim, bool)
        or not isinstance(mapper.state_dim, int)
        or mapper.state_dim < 1
    ):
        raise TypeError("StateMapper.state_dim must be a positive integer")
    if not isinstance(mapper.approved_for_production, bool):
        raise TypeError("StateMapper.approved_for_production must be a boolean")
    if not callable(mapper.map_state):
        raise TypeError("StateMapper.map_state must be callable")
    if production and mapper.approved_for_production is not True:
        raise RuntimeError(
            f"StateMapper {mapper.name!r} is not approved for production"
        )
    return mapper


def load_state_mapper(spec: str, *, production: bool = True) -> StateMapper:
    """Load an explicit ``module:attribute`` StateMapper specification."""

    if ":" not in spec:
        raise ValueError("--state-mapper must use module:attribute syntax")
    module_name, attribute_name = spec.split(":", 1)
    if not module_name or not attribute_name:
        raise ValueError("--state-mapper must use module:attribute syntax")
    module = importlib.import_module(module_name)
    candidate = getattr(module, attribute_name)
    mapper = candidate() if isinstance(candidate, type) else candidate
    return require_state_mapper(mapper, production=production)


def map_state(
    mapper: StateMapper,
    episode: CanonicalEpisode,
    step: CanonicalStep,
) -> np.ndarray:
    """Apply one mapper and validate its finite float32 output."""

    try:
        state = np.asarray(mapper.map_state(episode, step), dtype=np.float32)
    except CanonicalV1Error:
        raise
    except Exception as exc:
        raise CanonicalV1Error(
            f"StateMapper {mapper.name!r} failed: {exc}",
            episode_id=episode.episode_id,
            step_index=step.step_index,
            field="state",
        ) from exc
    if state.shape != (int(mapper.state_dim),) or not np.all(np.isfinite(state)):
        raise CanonicalV1Error(
            f"StateMapper {mapper.name!r} returned invalid shape/values {state.shape}",
            episode_id=episode.episode_id,
            step_index=step.step_index,
            field="state",
        )
    return np.ascontiguousarray(state)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_split_registry(path: str | Path) -> SplitRegistry:
    """Load and verify the authoritative external Split Registry."""

    return SplitRegistry.load(Path(path))


def _unique_tick_index(
    ticks: np.ndarray,
    *,
    episode_id: str,
    field: str,
) -> dict[int, int]:
    result: dict[int, int] = {}
    for index, raw_tick in enumerate(ticks):
        tick = int(raw_tick)
        if tick in result:
            raise CanonicalV1Error(
                f"duplicate physics_tick {tick}",
                episode_id=episode_id,
                field=field,
            )
        result[tick] = index
    return result


def _decoded_strings(dataset: Any) -> list[str]:
    return [str(value) for value in dataset.asstr()[:].tolist()]


def read_canonical_episode(
    episode_dir: str | Path,
    *,
    split_registry: SplitRegistry | None = None,
) -> CanonicalEpisode:
    """Project one authoritative HDF5 Episode into exact role-E samples."""

    root = Path(episode_dir)
    provisional_id = root.name or "<unknown>"
    if split_registry is None:
        raise CanonicalV1Error(
            "a verified external SplitRegistry is required",
            episode_id=provisional_id,
            field="split_registry",
        )
    try:
        reader = CanonicalEpisodeReader(
            root,
            split_registry=split_registry,
            is_training=False,
        )
    except Exception as exc:
        raise CanonicalV1Error(
            f"authoritative Canonical reader rejected Episode: {exc}",
            episode_id=provisional_id,
            field="canonical_reader",
        ) from exc

    try:
        metadata = reader.manifest["metadata"]
        episode_id = str(metadata["episode_id"])
        assignment = reader.split_assignment
        if assignment is None or assignment.split.value not in VALID_SPLITS:
            raise CanonicalV1Error(
                "Episode has no verified split assignment",
                episode_id=episode_id,
                field="split_registry",
            )
        h5 = getattr(reader, "_h5", None)
        if h5 is None:
            raise RuntimeError("authoritative reader did not expose verified HDF5")

        camera_group = h5[f"cameras/{EXPECTED_CAMERA_ID}"]
        camera_ticks = np.asarray(camera_group["physics_tick"][:], dtype=np.uint64)
        camera_indices = _unique_tick_index(
            camera_ticks,
            episode_id=episode_id,
            field="cameras.CAM_A_TOP.physics_tick",
        )
        camera_sequence_ids = np.asarray(
            camera_group["sequence_id"][:], dtype=np.uint64
        )
        camera_timestamps = np.asarray(camera_group["timestamp_ns"][:], dtype=np.uint64)
        camera_fallback = np.asarray(camera_group["is_fallback"][:], dtype=np.bool_)
        camera_hashes = _decoded_strings(camera_group["image_sha256"])
        camera_frames = reader.camera_frames(EXPECTED_CAMERA_ID)

        state_stream = reader.state_stream(EXPECTED_ARM_ID)
        state_ticks = np.asarray(state_stream["physics_tick"], dtype=np.uint64)
        state_indices = _unique_tick_index(
            state_ticks,
            episode_id=episode_id,
            field="robot_state.Arm_A.physics_tick",
        )
        state_sequence_ids = np.asarray(state_stream["sequence_id"], dtype=np.uint64)
        state_timestamps = np.asarray(state_stream["timestamp_ns"], dtype=np.uint64)
        state_values = np.asarray(state_stream["state_7d"], dtype=np.float32)

        actions = tuple(reader.iter_valid_actions())
        if not actions:
            raise CanonicalV1Error(
                "Episode contains no valid action",
                episode_id=episode_id,
                field="actions.valid_mask",
            )
        steps: list[CanonicalStep] = []
        for action in actions:
            if action.arm_id != EXPECTED_ARM_ID or action.executor != EXPECTED_EXECUTOR:
                raise CanonicalV1Error(
                    "role-E Episode contains a valid non-Arm_A/pi05 action",
                    episode_id=episode_id,
                    step_index=action.sequence_id,
                    field="actions.arm_id/executor",
                )
            tick = int(action.physics_tick)
            camera_index = camera_indices.get(tick)
            state_index = state_indices.get(tick)
            if camera_index is None:
                raise CanonicalV1Error(
                    f"no CAM_A_TOP sample at action physics_tick {tick}",
                    episode_id=episode_id,
                    step_index=action.sequence_id,
                    field="cameras.CAM_A_TOP.physics_tick",
                )
            if state_index is None:
                raise CanonicalV1Error(
                    f"no Arm_A state at action physics_tick {tick}",
                    episode_id=episode_id,
                    step_index=action.sequence_id,
                    field="robot_state.Arm_A.physics_tick",
                )
            if bool(camera_fallback[camera_index]):
                raise CanonicalV1Error(
                    "CAM_A_TOP fallback frames are forbidden",
                    episode_id=episode_id,
                    step_index=action.sequence_id,
                    field="cameras.CAM_A_TOP.is_fallback",
                )
            state = np.ascontiguousarray(state_values[state_index], dtype=np.float32)
            action_values = np.ascontiguousarray(action.action_7d, dtype=np.float32)
            if state.shape != (7,) or action_values.shape != (7,):
                raise CanonicalV1Error(
                    "state/action must both be 7-D",
                    episode_id=episode_id,
                    step_index=action.sequence_id,
                    field="state_7d/action_7d",
                )
            if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action_values)):
                raise CanonicalV1Error(
                    "state/action contains NaN or Infinity",
                    episode_id=episode_id,
                    step_index=action.sequence_id,
                    field="state_7d/action_7d",
                )
            image = np.ascontiguousarray(camera_frames[camera_index], dtype=np.uint8)
            if image.shape != (720, 1280, 3):
                raise CanonicalV1Error(
                    f"CAM_A_TOP has invalid shape {image.shape}",
                    episode_id=episode_id,
                    step_index=action.sequence_id,
                    field="cameras.CAM_A_TOP.rgb",
                )
            steps.append(
                CanonicalStep(
                    step_index=action.sequence_id,
                    timestamp_ns=action.timestamp_ns,
                    observation_id=f"{episode_id}:physics_tick:{tick}",
                    state_7d=state,
                    action_7d=action_values,
                    action_duration_s=action.duration_ms / 1000.0,
                    valid_for_training=True,
                    cam_a_top_rgb=image,
                    cam_a_top_relative_path=(
                        f"/cameras/{EXPECTED_CAMERA_ID}/rgb[{camera_index}]"
                    ),
                    cam_a_top_sha256=camera_hashes[camera_index].split(":", 1)[-1],
                    physics_tick=tick,
                    action_sequence_id=action.sequence_id,
                    camera_sequence_id=int(camera_sequence_ids[camera_index]),
                    camera_timestamp_ns=int(camera_timestamps[camera_index]),
                    state_sequence_id=int(state_sequence_ids[state_index]),
                    state_timestamp_ns=int(state_timestamps[state_index]),
                    raw={
                        "subtask_id": action.subtask_id,
                        "chunk_id": action.chunk_id,
                        "chunk_position": action.chunk_position,
                    },
                )
            )
        if not steps:
            raise CanonicalV1Error(
                "Episode contains no aligned role-E samples",
                episode_id=episode_id,
                field="actions",
            )
        return CanonicalEpisode(
            root=reader.episode_path,
            episode_id=episode_id,
            split=assignment.split.value,
            instruction=str(metadata["instruction"]),
            robot_role=EXPECTED_ROBOT_ROLE,
            eligible_for_imitation=True,
            meta=metadata,
            steps=tuple(steps),
            recorder_git_sha=str(metadata["git_sha"]),
            structure_sha256=_sha256_file(reader.episode_path / "structure.json"),
            hdf5_sha256=str(reader.manifest["storage"]["sha256"]).split(":", 1)[-1],
            split_registry_sha256=split_registry.registry_sha256.split(":", 1)[-1],
        )
    finally:
        reader.close()


def load_rgb_image(step: CanonicalStep, *, episode_id: str) -> np.ndarray:
    """Return a defensive copy of the verified raw CAM_A_TOP frame."""

    image = np.asarray(step.cam_a_top_rgb)
    if image.dtype != np.uint8 or image.shape != (720, 1280, 3):
        raise CanonicalV1Error(
            "CAM_A_TOP must be uint8[720,1280,3]",
            episode_id=episode_id,
            step_index=step.step_index,
            field="cameras.CAM_A_TOP.rgb",
        )
    return np.ascontiguousarray(image.copy())


def find_episode_dirs(data_dir: str | Path) -> list[Path]:
    """Enumerate HDF5 Canonical Episode directories without format guessing."""

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Canonical dataset directory does not exist: {root}")
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    episodes = [
        path
        for path in candidates
        if (path / "structure.json").is_file() and (path / "episode.h5").is_file()
    ]
    if not episodes:
        raise CanonicalV1Error(
            "no Canonical v1 Episodes found; expected episode.h5 + structure.json",
            episode_id=root.name,
            field="dataset",
        )
    return episodes


def read_canonical_dataset(
    data_dir: str | Path,
    *,
    split_registry: SplitRegistry | None = None,
) -> tuple[CanonicalEpisode, ...]:
    """Read every Episode through the authoritative framework reader."""

    if split_registry is None:
        raise CanonicalV1Error(
            "a verified external SplitRegistry is required",
            episode_id=Path(data_dir).name,
            field="split_registry",
        )
    return tuple(
        read_canonical_episode(path, split_registry=split_registry)
        for path in find_episode_dirs(data_dir)
    )
