"""Strict PI05 reader for the frozen Canonical Episode v1 contract.

This module owns Canonical parsing for both conversion and normalization.  It
does not invent the OpenPI state vector: callers must explicitly inject an
approved :class:`StateMapper` before producing state-bearing artifacts.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

import numpy as np
from PIL import Image, UnidentifiedImageError

CANONICAL_SCHEMA_VERSION = "1.0"
EXPECTED_ROBOT_ROLE = "arm_a_pi05"
EXPECTED_CAMERA_ID = "CAM_A_TOP"
EXPECTED_IMAGE_SIZE = (1280, 720)  # width, height
VALID_SPLITS = frozenset({"train", "val", "test"})
VALID_OUTCOMES = frozenset({"success", "failure", "recovery_success"})
LEGACY_MARKERS = ("steps.parquet", "steps.hdf5", "front_rgb")
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
EXPECTED_CONTROL_HZ = 60
EXPECTED_RENDER_HZ = 30
_SHA256_LINE = re.compile(r"^(?:sha256:)?([0-9a-fA-F]{64})\s+\*?(.+?)\s*$")
_SHA256_HEX = re.compile(r"^[0-9a-fA-F]{64}$")
_GIT_REVISION = re.compile(r"^(?:[0-9a-fA-F]{40}|[0-9a-fA-F]{64})$")


class CanonicalV1Error(ValueError):
    """A fail-closed Canonical v1 validation error with stable context."""

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
    """One validated Canonical v1 step without an inferred model state.

    Arrays preserve the Canonical physical fields.  ``tcp_pose`` is
    float64[7] ``[x,y,z,qx,qy,qz,qw]`` in robot_base.  ``action_7d`` is
    float32[7] ``[dx,dy,dz,dax,day,daz,gripper]`` in robot_base, metres,
    radians and normalized gripper units.
    """

    step_index: int
    timestamp_ns: int
    observation_id: str
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    tcp_pose: np.ndarray
    gripper_state: float
    arm_a_gripper_open: bool
    action_7d: np.ndarray
    action_duration_s: float
    valid_for_training: bool
    cam_a_top_path: Path
    cam_a_top_relative_path: str
    cam_a_top_sha256: str
    raw: Mapping[str, Any]


@dataclass(frozen=True)
class CanonicalEpisode:
    """A fully validated Canonical v1 episode."""

    root: Path
    episode_id: str
    split: str
    instruction: str
    robot_role: str
    eligible_for_imitation: bool
    meta: Mapping[str, Any]
    steps: tuple[CanonicalStep, ...]

    @property
    def training_steps(self) -> tuple[CanonicalStep, ...]:
        """Return only structurally valid behavior targets."""

        if not self.eligible_for_imitation:
            return ()
        return tuple(step for step in self.steps if step.valid_for_training)

    @property
    def imitation_steps(self) -> tuple[CanonicalStep, ...]:
        """Return a whole Episode only when every imitation Gate is valid."""

        if not self.eligible_for_imitation:
            raise CanonicalV1Error(
                "Episode is not eligible for imitation",
                episode_id=self.episode_id,
                field="eligible_for_imitation",
            )
        for step in self.steps:
            if not step.valid_for_training:
                raise CanonicalV1Error(
                    "one invalid training Step rejects the entire Episode",
                    episode_id=self.episode_id,
                    step_index=step.step_index,
                    field="valid_for_training",
                )
        return self.steps


@runtime_checkable
class StateMapper(Protocol):
    """Explicit mapping from preserved Canonical fields to model state.

    Production mappers expose ``approved_for_production=True`` only for the
    frozen semantic order, frame and units implemented by that mapper.
    """

    name: str
    state_dim: int
    approved_for_production: bool

    def map_state(
        self, episode: CanonicalEpisode, step: CanonicalStep
    ) -> np.ndarray:
        """Return finite float32[state_dim] for one step."""


def quaternion_xyzw_to_rotation_vector(
    quaternion_xyzw: np.ndarray,
    *,
    episode_id: str,
    step_index: int,
) -> np.ndarray:
    """Convert finite float64[4] ``[qx,qy,qz,qw]`` to float32[3] rotvec.

    The input order is a compile-time Canonical contract and is deliberately
    not configurable.  The result is the shortest rotation vector in
    ``robot_base``, in radians, with angle in ``[0, pi]``.
    """

    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64)
    if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
        raise CanonicalV1Error(
            "quaternion must be finite float[4] in compile-time xyzw order",
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
    quaternion = quaternion / norm
    if quaternion[3] < 0.0:
        quaternion = -quaternion
    vector = quaternion[:3]
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return np.zeros(3, dtype=np.float32)
    angle = 2.0 * math.atan2(vector_norm, float(quaternion[3]))
    angle = min(max(angle, 0.0), math.pi)
    result = vector * (angle / vector_norm)
    return np.ascontiguousarray(result, dtype=np.float32)


class CanonicalPi05StateMapper:
    """Approved explicit mapper for the frozen PI05 state_7d contract."""

    name = "canonical_pi05_state_7d_xyzw_v1"
    version = "1.0"
    state_dim = 7
    approved_for_production = True

    def map_state(
        self, episode: CanonicalEpisode, step: CanonicalStep
    ) -> np.ndarray:
        """Return float32[7] in robot_base, metres/radians/normalized bool."""

        rotation_vector = quaternion_xyzw_to_rotation_vector(
            step.tcp_pose[3:7],
            episode_id=episode.episode_id,
            step_index=step.step_index,
        )
        state = np.empty(7, dtype=np.float32)
        state[:3] = step.tcp_pose[:3]
        state[3:6] = rotation_vector
        state[6] = 1.0 if step.arm_a_gripper_open else 0.0
        return state


def require_state_mapper(
    mapper: StateMapper | None,
    *,
    production: bool,
) -> StateMapper:
    """Validate an explicitly injected mapper and its approval status."""

    if mapper is None:
        raise RuntimeError(
            "PI05 StateMapper is required; production mapping must be explicitly "
            "injected and no implicit default is permitted"
        )
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
            f"StateMapper {mapper.name!r} is not approved for production; "
            "use the explicitly approved production mapper"
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
    """Apply one mapper and strictly validate float32 shape and finiteness."""

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
    if state.shape != (int(mapper.state_dim),):
        raise CanonicalV1Error(
            f"StateMapper {mapper.name!r} returned shape {state.shape}; "
            f"expected ({int(mapper.state_dim)},)",
            episode_id=episode.episode_id,
            step_index=step.step_index,
            field="state",
        )
    if not np.all(np.isfinite(state)):
        raise CanonicalV1Error(
            f"StateMapper {mapper.name!r} returned NaN or Infinity",
            episode_id=episode.episode_id,
            step_index=step.step_index,
            field="state",
        )
    return np.ascontiguousarray(state)


def _error(
    message: str,
    *,
    episode_id: str,
    field: str,
    step_index: int | None = None,
) -> CanonicalV1Error:
    return CanonicalV1Error(
        message,
        episode_id=episode_id,
        field=field,
        step_index=step_index,
    )


def _read_json_object(path: Path, *, episode_id: str, field: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise _error(
            f"cannot read valid JSON object from {path.name}: {exc}",
            episode_id=episode_id,
            field=field,
        ) from exc
    if not isinstance(value, dict):
        raise _error(
            f"{path.name} must contain a JSON object",
            episode_id=episode_id,
            field=field,
        )
    return value


def _required_string(
    value: Mapping[str, Any],
    field: str,
    *,
    episode_id: str,
    step_index: int | None = None,
) -> str:
    item = value.get(field)
    if not isinstance(item, str) or not item.strip():
        raise _error(
            "must be a non-empty string",
            episode_id=episode_id,
            field=field,
            step_index=step_index,
        )
    return item


def _required_integer(
    value: Mapping[str, Any],
    field: str,
    *,
    episode_id: str,
    minimum: int | None = None,
) -> int:
    item = value.get(field)
    if isinstance(item, bool) or not isinstance(item, int):
        raise _error("must be an integer", episode_id=episode_id, field=field)
    if minimum is not None and item < minimum:
        raise _error(
            f"must be >= {minimum}", episode_id=episode_id, field=field
        )
    return item


def _required_hash(
    value: Mapping[str, Any],
    field: str,
    *,
    episode_id: str,
    pattern: re.Pattern[str],
    description: str,
) -> str:
    item = _required_string(value, field, episode_id=episode_id)
    if pattern.fullmatch(item) is None:
        raise _error(description, episode_id=episode_id, field=field)
    return item.lower()


def _parse_iso8601(value: Any, *, episode_id: str, field: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise _error(
            "must be a non-empty ISO-8601 string",
            episode_id=episode_id,
            field=field,
        )
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise _error(
            "must be a valid ISO-8601 timestamp",
            episode_id=episode_id,
            field=field,
        ) from exc
    if parsed.tzinfo is None:
        raise _error(
            "must include an explicit timezone",
            episode_id=episode_id,
            field=field,
        )
    return parsed


def _validate_meta(meta: Mapping[str, Any], *, episode_id: str) -> None:
    """Validate the complete Canonical v1 Episode metadata contract."""

    _required_string(meta, "scenario_group_id", episode_id=episode_id)
    _required_integer(meta, "scene_seed", episode_id=episode_id, minimum=0)
    _required_string(meta, "asset_variant", episode_id=episode_id)
    _required_string(meta, "task_id", episode_id=episode_id)
    _required_hash(
        meta,
        "scene_config_sha256",
        episode_id=episode_id,
        pattern=_SHA256_HEX,
        description="must be a 64-hex SHA-256",
    )
    for field in ("controller_version", "recorder_version"):
        _required_hash(
            meta,
            field,
            episode_id=episode_id,
            pattern=_GIT_REVISION,
            description="must be a 40- or 64-hex Git revision",
        )
    camera_ids = meta.get("camera_ids")
    if (
        not isinstance(camera_ids, list)
        or not camera_ids
        or not all(isinstance(camera_id, str) and camera_id for camera_id in camera_ids)
    ):
        raise _error(
            "must be a non-empty array of non-empty Camera IDs",
            episode_id=episode_id,
            field="camera_ids",
        )
    if len(set(camera_ids)) != len(camera_ids):
        raise _error(
            "must not contain duplicate Camera IDs",
            episode_id=episode_id,
            field="camera_ids",
        )
    if EXPECTED_CAMERA_ID not in camera_ids:
        raise _error(
            f"must include {EXPECTED_CAMERA_ID}",
            episode_id=episode_id,
            field="camera_ids",
        )
    control_hz = _required_integer(
        meta, "control_hz", episode_id=episode_id, minimum=1
    )
    render_hz = _required_integer(meta, "render_hz", episode_id=episode_id, minimum=1)
    if control_hz != EXPECTED_CONTROL_HZ:
        raise _error(
            f"must equal frozen controller rate {EXPECTED_CONTROL_HZ}",
            episode_id=episode_id,
            field="control_hz",
        )
    if render_hz != EXPECTED_RENDER_HZ:
        raise _error(
            f"must equal frozen render rate {EXPECTED_RENDER_HZ}",
            episode_id=episode_id,
            field="render_hz",
        )
    started_at = _parse_iso8601(
        meta.get("started_at"), episode_id=episode_id, field="started_at"
    )
    ended_at = _parse_iso8601(
        meta.get("ended_at"), episode_id=episode_id, field="ended_at"
    )
    if ended_at < started_at:
        raise _error(
            "must not be earlier than started_at",
            episode_id=episode_id,
            field="ended_at",
        )
    outcome = _required_string(meta, "outcome", episode_id=episode_id)
    if outcome not in VALID_OUTCOMES:
        raise _error(
            f"must be one of {sorted(VALID_OUTCOMES)}, got {outcome!r}",
            episode_id=episode_id,
            field="outcome",
        )
    for field in ("dataset_failure_label", "parent_episode_id"):
        if field not in meta:
            raise _error(
                "is required and may be JSON null",
                episode_id=episode_id,
                field=field,
            )
        item = meta.get(field)
        if item is not None and (not isinstance(item, str) or not item.strip()):
            raise _error(
                "must be null or a non-empty string",
                episode_id=episode_id,
                field=field,
            )


def _finite_vector(
    value: Any,
    *,
    episode_id: str,
    step_index: int,
    field: str,
    expected_length: int | None = None,
    dtype: Any = np.float64,
) -> np.ndarray:
    if not isinstance(value, list):
        raise _error(
            "must be a JSON array",
            episode_id=episode_id,
            step_index=step_index,
            field=field,
        )
    try:
        array = np.asarray(value, dtype=dtype)
    except (TypeError, ValueError) as exc:
        raise _error(
            f"must contain only numeric values: {exc}",
            episode_id=episode_id,
            step_index=step_index,
            field=field,
        ) from exc
    if array.ndim != 1:
        raise _error(
            f"must be one-dimensional, got shape {array.shape}",
            episode_id=episode_id,
            step_index=step_index,
            field=field,
        )
    if expected_length is not None and array.shape != (expected_length,):
        raise _error(
            f"must have length {expected_length}, got shape {array.shape}",
            episode_id=episode_id,
            step_index=step_index,
            field=field,
        )
    if not np.all(np.isfinite(array)):
        raise _error(
            "contains NaN or Infinity",
            episode_id=episode_id,
            step_index=step_index,
            field=field,
        )
    return np.ascontiguousarray(array)


def _finite_number(
    value: Any,
    *,
    episode_id: str,
    step_index: int,
    field: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _error(
            "must be a finite number",
            episode_id=episode_id,
            step_index=step_index,
            field=field,
        )
    result = float(value)
    if not math.isfinite(result):
        raise _error(
            "must be finite",
            episode_id=episode_id,
            step_index=step_index,
            field=field,
        )
    return result


def _contains_forbidden_key(value: Any, *, allow_robot_retreat: bool = False) -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if "wrist" in lowered or "offline_gt" in lowered:
                return str(key)
            if "arm_b" in lowered:
                if not (
                    allow_robot_retreat
                    and lowered == "arm_b"
                    and isinstance(nested, Mapping)
                    and set(nested) == {"retreated"}
                ):
                    return str(key)
            found = _contains_forbidden_key(
                nested,
                allow_robot_retreat=allow_robot_retreat or lowered == "robot",
            )
            if found is not None:
                return found
    elif isinstance(value, list):
        for nested in value:
            found = _contains_forbidden_key(nested, allow_robot_retreat=False)
            if found is not None:
                return found
    elif isinstance(value, str) and "offline_gt" in value.lower().replace("\\", "/"):
        return "offline_gt"
    return None


def _safe_episode_path(
    root: Path,
    relative: Any,
    *,
    episode_id: str,
    step_index: int,
) -> tuple[Path, str]:
    if not isinstance(relative, str) or not relative.strip():
        raise _error(
            "CAM_A_TOP path must be a non-empty relative string",
            episode_id=episode_id,
            step_index=step_index,
            field="rgb.CAM_A_TOP",
        )
    normalized = relative.replace("\\", "/")
    relative_path = Path(normalized)
    if relative_path.is_absolute() or ".." in relative_path.parts:
        raise _error(
            f"path escapes the Episode: {relative!r}",
            episode_id=episode_id,
            step_index=step_index,
            field="rgb.CAM_A_TOP",
        )
    if tuple(relative_path.parts[:2]) != ("rgb", EXPECTED_CAMERA_ID):
        raise _error(
            f"must point under rgb/{EXPECTED_CAMERA_ID}/, got {relative!r}",
            episode_id=episode_id,
            step_index=step_index,
            field="rgb.CAM_A_TOP",
        )
    resolved_root = root.resolve()
    resolved = (root / relative_path).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise _error(
            f"path escapes the Episode: {relative!r}",
            episode_id=episode_id,
            step_index=step_index,
            field="rgb.CAM_A_TOP",
        )
    return resolved, relative_path.as_posix()


def _parse_checksums(path: Path, *, episode_id: str, root: Path) -> dict[str, str]:
    if not path.is_file():
        raise _error(
            "checksums.sha256 is required",
            episode_id=episode_id,
            field="checksums.sha256",
        )
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _SHA256_LINE.fullmatch(line)
        if match is None:
            raise _error(
                f"invalid checksum line {line_number}: {raw_line!r}",
                episode_id=episode_id,
                field="checksums.sha256",
            )
        digest, relative = match.groups()
        normalized = relative.replace("\\", "/").removeprefix("./")
        relative_path = Path(normalized)
        resolved = (root / relative_path).resolve()
        if relative_path.is_absolute() or not resolved.is_relative_to(root.resolve()):
            raise _error(
                f"checksum path escapes the Episode: {relative!r}",
                episode_id=episode_id,
                field="checksums.sha256",
            )
        if normalized in checksums:
            raise _error(
                f"duplicate checksum entry for {normalized!r}",
                episode_id=episode_id,
                field="checksums.sha256",
            )
        checksums[normalized] = digest.lower()
    if not checksums:
        raise _error(
            "checksums.sha256 contains no entries",
            episode_id=episode_id,
            field="checksums.sha256",
        )
    return checksums


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_checksum(
    root: Path,
    relative: str,
    checksums: Mapping[str, str],
    *,
    episode_id: str,
    field: str,
    step_index: int | None = None,
) -> str:
    expected = checksums.get(relative)
    if expected is None:
        raise _error(
            f"missing checksum entry for {relative!r}",
            episode_id=episode_id,
            step_index=step_index,
            field=field,
        )
    target = root / Path(relative)
    if not target.is_file():
        raise _error(
            f"referenced file does not exist: {relative!r}",
            episode_id=episode_id,
            step_index=step_index,
            field=field,
        )
    actual = _sha256_file(target)
    if actual != expected:
        raise _error(
            f"SHA-256 mismatch for {relative!r}: expected={expected} actual={actual}",
            episode_id=episode_id,
            step_index=step_index,
            field=field,
        )
    return actual


def load_rgb_image(step: CanonicalStep, *, episode_id: str) -> np.ndarray:
    """Decode and re-check one immutable raw 1280x720 RGB image."""

    actual = _sha256_file(step.cam_a_top_path) if step.cam_a_top_path.is_file() else ""
    if actual != step.cam_a_top_sha256:
        raise _error(
            "image changed after Canonical validation",
            episode_id=episode_id,
            step_index=step.step_index,
            field="rgb.CAM_A_TOP.checksum",
        )
    try:
        with Image.open(step.cam_a_top_path) as image:
            image.load()
            if image.mode != "RGB":
                raise _error(
                    f"image mode must be RGB, got {image.mode!r}",
                    episode_id=episode_id,
                    step_index=step.step_index,
                    field="rgb.CAM_A_TOP",
                )
            if image.size != EXPECTED_IMAGE_SIZE:
                raise _error(
                    f"image size must be 1280x720, got {image.size[0]}x{image.size[1]}",
                    episode_id=episode_id,
                    step_index=step.step_index,
                    field="rgb.CAM_A_TOP",
                )
            array = np.asarray(image, dtype=np.uint8)
    except CanonicalV1Error:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise _error(
            f"image cannot be decoded as RGB: {exc}",
            episode_id=episode_id,
            step_index=step.step_index,
            field="rgb.CAM_A_TOP",
        ) from exc
    if array.shape != (720, 1280, 3):
        raise _error(
            f"decoded image must be uint8[720,1280,3], got {array.shape}",
            episode_id=episode_id,
            step_index=step.step_index,
            field="rgb.CAM_A_TOP",
        )
    return np.ascontiguousarray(array)


def _validate_legacy_markers(root: Path, *, episode_id: str) -> None:
    found = [name for name in LEGACY_MARKERS if (root / name).exists()]
    if found:
        raise _error(
            "legacy Canonical input is not supported; found "
            f"{found}. Current input must use meta.json + steps.jsonl + "
            "rgb/CAM_A_TOP",
            episode_id=episode_id,
            field="input_format",
        )


def read_canonical_episode(episode_dir: str | Path) -> CanonicalEpisode:
    """Read and fully validate one PI05 Canonical v1 Episode."""

    root = Path(episode_dir)
    provisional_id = root.name or "<unknown>"
    if not root.is_dir():
        raise _error(
            f"Episode directory does not exist: {root}",
            episode_id=provisional_id,
            field="episode_dir",
        )
    _validate_legacy_markers(root, episode_id=provisional_id)

    meta_path = root / "meta.json"
    steps_path = root / "steps.jsonl"
    if not meta_path.is_file():
        raise _error(
            "meta.json is required",
            episode_id=provisional_id,
            field="meta.json",
        )
    if not steps_path.is_file():
        raise _error(
            "steps.jsonl is required; current input must use meta.json + "
            "steps.jsonl + rgb/CAM_A_TOP",
            episode_id=provisional_id,
            field="steps.jsonl",
        )

    meta = _read_json_object(meta_path, episode_id=provisional_id, field="meta.json")
    episode_id = _required_string(meta, "episode_id", episode_id=provisional_id)
    if episode_id != root.name:
        raise _error(
            f"must match Episode directory name {root.name!r}",
            episode_id=episode_id,
            field="episode_id",
        )
    if meta.get("schema_version") != CANONICAL_SCHEMA_VERSION:
        raise _error(
            f"must equal {CANONICAL_SCHEMA_VERSION!r}",
            episode_id=episode_id,
            field="schema_version",
        )
    robot_role = _required_string(meta, "robot_role", episode_id=episode_id)
    if robot_role != EXPECTED_ROBOT_ROLE:
        raise _error(
            f"must equal {EXPECTED_ROBOT_ROLE!r}, got {robot_role!r}",
            episode_id=episode_id,
            field="robot_role",
        )
    split = _required_string(meta, "split", episode_id=episode_id)
    if split not in VALID_SPLITS:
        raise _error(
            f"must be one of {sorted(VALID_SPLITS)}, got {split!r}",
            episode_id=episode_id,
            field="split",
        )
    instruction = _required_string(meta, "instruction", episode_id=episode_id)
    eligible = meta.get("eligible_for_imitation")
    if not isinstance(eligible, bool):
        raise _error(
            "must be a boolean",
            episode_id=episode_id,
            field="eligible_for_imitation",
        )
    _validate_meta(meta, episode_id=episode_id)
    forbidden_meta = _contains_forbidden_key(meta)
    if forbidden_meta is not None:
        raise _error(
            f"forbidden PI05 field or path {forbidden_meta!r}",
            episode_id=episode_id,
            field=forbidden_meta,
        )

    checksums = _parse_checksums(
        root / "checksums.sha256", episode_id=episode_id, root=root
    )
    _verify_checksum(
        root,
        "meta.json",
        checksums,
        episode_id=episode_id,
        field="meta.json.checksum",
    )
    _verify_checksum(
        root,
        "steps.jsonl",
        checksums,
        episode_id=episode_id,
        field="steps.jsonl.checksum",
    )

    parsed_steps: list[CanonicalStep] = []
    seen_observations: set[str] = set()
    previous_timestamp: int | None = None
    lines = steps_path.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise _error(
            "must contain at least one Step",
            episode_id=episode_id,
            field="steps.jsonl",
        )

    for expected_index, raw_line in enumerate(lines):
        try:
            raw = json.loads(raw_line)
        except Exception as exc:
            raise _error(
                f"line {expected_index + 1} is not valid JSON: {exc}",
                episode_id=episode_id,
                step_index=expected_index,
                field="steps.jsonl",
            ) from exc
        if not isinstance(raw, dict):
            raise _error(
                "Step must be a JSON object",
                episode_id=episode_id,
                step_index=expected_index,
                field="step",
            )
        step_index = raw.get("step_index")
        if isinstance(step_index, bool) or step_index != expected_index:
            raise _error(
                f"must be contiguous from 0; expected {expected_index}, got {step_index!r}",
                episode_id=episode_id,
                step_index=expected_index,
                field="step_index",
            )
        timestamp_ns = raw.get("timestamp_ns")
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int):
            raise _error(
                "must be an integer",
                episode_id=episode_id,
                step_index=step_index,
                field="timestamp_ns",
            )
        if previous_timestamp is not None and timestamp_ns <= previous_timestamp:
            raise _error(
                f"must be strictly monotonic; previous={previous_timestamp} "
                f"current={timestamp_ns}",
                episode_id=episode_id,
                step_index=step_index,
                field="timestamp_ns",
            )
        previous_timestamp = timestamp_ns
        observation_id = _required_string(
            raw,
            "observation_id",
            episode_id=episode_id,
            step_index=step_index,
        )
        if observation_id in seen_observations:
            raise _error(
                f"duplicate observation_id {observation_id!r}",
                episode_id=episode_id,
                step_index=step_index,
                field="observation_id",
            )
        seen_observations.add(observation_id)

        valid_for_training = raw.get("valid_for_training")
        if not isinstance(valid_for_training, bool):
            raise _error(
                "must be a boolean",
                episode_id=episode_id,
                step_index=step_index,
                field="valid_for_training",
            )
        if "wrist_image" not in raw or raw["wrist_image"] is not None:
            raise _error(
                "must be present and JSON null; wrist camera data is forbidden",
                episode_id=episode_id,
                step_index=step_index,
                field="wrist_image",
            )
        forbidden_step = _contains_forbidden_key(
            {key: value for key, value in raw.items() if key != "wrist_image"},
            allow_robot_retreat=True,
        )
        if forbidden_step is not None:
            raise _error(
                f"forbidden PI05 field or path {forbidden_step!r}",
                episode_id=episode_id,
                step_index=step_index,
                field=forbidden_step,
            )

        rgb = raw.get("rgb")
        if not isinstance(rgb, dict):
            raise _error(
                "must be a Camera ID to relative path object",
                episode_id=episode_id,
                step_index=step_index,
                field="rgb",
            )
        for camera_id in rgb:
            if "WRIST" in str(camera_id).upper():
                raise _error(
                    f"wrist camera {camera_id!r} is forbidden",
                    episode_id=episode_id,
                    step_index=step_index,
                    field="rgb",
                )
        image_path, image_relative = _safe_episode_path(
            root,
            rgb.get(EXPECTED_CAMERA_ID),
            episode_id=episode_id,
            step_index=step_index,
        )
        image_sha = _verify_checksum(
            root,
            image_relative,
            checksums,
            episode_id=episode_id,
            step_index=step_index,
            field="rgb.CAM_A_TOP.checksum",
        )

        joint_position = _finite_vector(
            raw.get("joint_position"),
            episode_id=episode_id,
            step_index=step_index,
            field="joint_position",
        )
        joint_velocity = _finite_vector(
            raw.get("joint_velocity"),
            episode_id=episode_id,
            step_index=step_index,
            field="joint_velocity",
        )
        if joint_position.size == 0 or joint_velocity.shape != joint_position.shape:
            raise _error(
                "must be non-empty and match joint_position shape",
                episode_id=episode_id,
                step_index=step_index,
                field="joint_velocity",
            )
        tcp_pose = _finite_vector(
            raw.get("tcp_pose"),
            episode_id=episode_id,
            step_index=step_index,
            field="tcp_pose",
            expected_length=7,
        )
        gripper_state = _finite_number(
            raw.get("gripper_state"),
            episode_id=episode_id,
            step_index=step_index,
            field="gripper_state",
        )
        action = _finite_vector(
            raw.get("action_7d"),
            episode_id=episode_id,
            step_index=step_index,
            field="action_7d",
            expected_length=7,
            dtype=np.float32,
        )
        duration = _finite_number(
            raw.get("action_duration_s"),
            episode_id=episode_id,
            step_index=step_index,
            field="action_duration_s",
        )
        if duration <= 0:
            raise _error(
                "must be greater than zero",
                episode_id=episode_id,
                step_index=step_index,
                field="action_duration_s",
            )
        for field in ("agent_state", "operation_phase", "handoff_token"):
            _required_string(
                raw,
                field,
                episode_id=episode_id,
                step_index=step_index,
            )
        safety_flags = raw.get("safety_flags")
        if not isinstance(safety_flags, list) or not all(
            isinstance(flag, str) for flag in safety_flags
        ):
            raise _error(
                "must be an array of strings",
                episode_id=episode_id,
                step_index=step_index,
                field="safety_flags",
            )
        robot = raw.get("robot")
        if not isinstance(robot, dict):
            raise _error(
                "must be an object",
                episode_id=episode_id,
                step_index=step_index,
                field="robot",
            )
        for arm in ("arm_a", "arm_b"):
            arm_value = robot.get(arm)
            if (
                not isinstance(arm_value, dict)
                or not isinstance(arm_value.get("retreated"), bool)
            ):
                raise _error(
                    f"robot.{arm}.retreated must be boolean",
                    episode_id=episode_id,
                    step_index=step_index,
                    field=f"robot.{arm}.retreated",
                )
        arm_a_gripper_open = robot["arm_a"].get("gripper_open")
        if not isinstance(arm_a_gripper_open, bool):
            raise _error(
                "must be a controller-confirmed boolean",
                episode_id=episode_id,
                step_index=step_index,
                field="robot.arm_a.gripper_open",
            )

        step = CanonicalStep(
            step_index=step_index,
            timestamp_ns=timestamp_ns,
            observation_id=observation_id,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            tcp_pose=tcp_pose,
            gripper_state=gripper_state,
            arm_a_gripper_open=arm_a_gripper_open,
            action_7d=action,
            action_duration_s=duration,
            valid_for_training=valid_for_training,
            cam_a_top_path=image_path,
            cam_a_top_relative_path=image_relative,
            cam_a_top_sha256=image_sha,
            raw=raw,
        )
        load_rgb_image(step, episode_id=episode_id)
        parsed_steps.append(step)

    return CanonicalEpisode(
        root=root.resolve(),
        episode_id=episode_id,
        split=split,
        instruction=instruction,
        robot_role=robot_role,
        eligible_for_imitation=eligible,
        meta=meta,
        steps=tuple(parsed_steps),
    )


def find_episode_dirs(data_dir: str | Path) -> list[Path]:
    """Enumerate candidate Episode directories without silently guessing format."""

    root = Path(data_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Canonical dataset directory does not exist: {root}")
    candidates = sorted(path for path in root.iterdir() if path.is_dir())
    legacy_roots = [path for path in candidates if any((path / item).exists() for item in LEGACY_MARKERS)]
    if legacy_roots:
        names = [path.name for path in legacy_roots]
        raise CanonicalV1Error(
            "legacy Canonical input is not supported in Episodes "
            f"{names}; current input must use meta.json + steps.jsonl + "
            "rgb/CAM_A_TOP",
            episode_id=names[0],
            field="input_format",
        )
    episodes = [path for path in candidates if (path / "meta.json").is_file()]
    if not episodes:
        raise CanonicalV1Error(
            "no Canonical v1 Episodes found; expected child directories with "
            "meta.json + steps.jsonl + rgb/CAM_A_TOP",
            episode_id=root.name,
            field="dataset",
        )
    return episodes


def read_canonical_dataset(data_dir: str | Path) -> tuple[CanonicalEpisode, ...]:
    """Read every Episode and fail the whole dataset on the first invalid item."""

    return tuple(read_canonical_episode(path) for path in find_episode_dirs(data_dir))
