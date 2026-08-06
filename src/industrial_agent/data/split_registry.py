"""Immutable episode split assignments for offline training data.

The registry is deliberately separate from the frozen online Observation and
ActionChunk contracts.  It enforces the dataset grouping and seed isolation
rules before a canonical episode can be consumed by a training reader.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
import hmac
import json
import os
from pathlib import Path
import re
from threading import RLock
from typing import Any, Iterable, Mapping
from uuid import uuid4


SPLIT_REGISTRY_VERSION = "1.0"
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_SHA256 = re.compile(r"^sha256:[0-9a-f]{64}$")
_ASSIGNMENT_FIELDS = frozenset(
    {
        "split",
        "scenario_group_id",
        "scene_seed",
        "asset_variant",
        "camera_seed",
        "lighting_seed",
        "parent_episode_id",
    }
)


class DatasetSplit(str, Enum):
    """The three split names frozen by the data contract."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class SplitRegistryError(ValueError):
    """Base error for malformed or inconsistent split registries."""


class SplitRegistryIntegrityError(SplitRegistryError):
    """Raised when a persisted registry fails its SHA-256 check."""


class SplitAssignmentError(SplitRegistryError):
    """Raised when an existing episode assignment would be changed."""


class DataLeakageError(RuntimeError):
    """Fatal error raised when non-training data reaches a training reader."""


def _non_blank(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise SplitRegistryError(f"{field_name} must be a non-empty string")
    return value.strip()


def _non_negative_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise SplitRegistryError(f"{field_name} must be a non-negative integer")
    return value


def _episode_id(value: Any, field_name: str = "episode_id") -> str:
    if not isinstance(value, str) or _SAFE_EPISODE_ID.fullmatch(value) is None:
        raise SplitRegistryError(f"{field_name} must match ^[A-Za-z0-9._-]{{1,128}}$")
    return value


def _split(value: DatasetSplit | str) -> DatasetSplit:
    if isinstance(value, DatasetSplit):
        return value
    if not isinstance(value, str):
        raise SplitRegistryError("split must be one of train, val, test")
    try:
        return DatasetSplit(value)
    except ValueError as exc:
        raise SplitRegistryError("split must be one of train, val, test") from exc


@dataclass(frozen=True)
class SplitAssignment:
    """One immutable episode-to-split assignment and its frozen group key."""

    episode_id: str
    split: DatasetSplit
    scenario_group_id: str
    scene_seed: int
    asset_variant: str
    camera_seed: int
    lighting_seed: int
    parent_episode_id: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "episode_id", _episode_id(self.episode_id))
        object.__setattr__(self, "split", _split(self.split))
        object.__setattr__(
            self,
            "scenario_group_id",
            _non_blank(self.scenario_group_id, "scenario_group_id"),
        )
        object.__setattr__(
            self,
            "scene_seed",
            _non_negative_int(self.scene_seed, "scene_seed"),
        )
        object.__setattr__(
            self,
            "asset_variant",
            _non_blank(self.asset_variant, "asset_variant"),
        )
        object.__setattr__(
            self,
            "camera_seed",
            _non_negative_int(self.camera_seed, "camera_seed"),
        )
        object.__setattr__(
            self,
            "lighting_seed",
            _non_negative_int(self.lighting_seed, "lighting_seed"),
        )
        if self.parent_episode_id is not None:
            object.__setattr__(
                self,
                "parent_episode_id",
                _episode_id(self.parent_episode_id, "parent_episode_id"),
            )
            if self.parent_episode_id == self.episode_id:
                raise SplitRegistryError("an episode cannot be its own parent")

    @property
    def group_key(self) -> tuple[str, int, str, int, int]:
        """Return the indivisible grouping key frozen by the data contract."""

        return (
            self.scenario_group_id,
            self.scene_seed,
            self.asset_variant,
            self.camera_seed,
            self.lighting_seed,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "split": self.split.value,
            "scenario_group_id": self.scenario_group_id,
            "scene_seed": self.scene_seed,
            "asset_variant": self.asset_variant,
            "camera_seed": self.camera_seed,
            "lighting_seed": self.lighting_seed,
            "parent_episode_id": self.parent_episode_id,
        }

    @classmethod
    def from_payload(
        cls,
        episode_id: str,
        payload: Mapping[str, Any],
    ) -> SplitAssignment:
        if not isinstance(payload, Mapping) or set(payload) != _ASSIGNMENT_FIELDS:
            raise SplitRegistryError(
                f"assignment {episode_id!r} has missing or unexpected fields"
            )
        return cls(
            episode_id=episode_id,
            split=payload["split"],
            scenario_group_id=payload["scenario_group_id"],
            scene_seed=payload["scene_seed"],
            asset_variant=payload["asset_variant"],
            camera_seed=payload["camera_seed"],
            lighting_seed=payload["lighting_seed"],
            parent_episode_id=payload["parent_episode_id"],
        )


def _canonical_bytes(payload: Mapping[str, Any]) -> bytes:
    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _payload_digest(payload: Mapping[str, Any]) -> str:
    return f"sha256:{sha256(_canonical_bytes(payload)).hexdigest()}"


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SplitRegistryError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


class SplitRegistry:
    """Mutable-by-addition registry with immutable existing assignments."""

    def __init__(self, assignments: Iterable[SplitAssignment] = ()) -> None:
        self._lock = RLock()
        entries: dict[str, SplitAssignment] = {}
        for assignment in assignments:
            if not isinstance(assignment, SplitAssignment):
                raise TypeError("assignments must contain SplitAssignment values")
            if assignment.episode_id in entries:
                raise SplitAssignmentError(
                    f"duplicate episode assignment: {assignment.episode_id}"
                )
            entries[assignment.episode_id] = assignment
        self._validate_assignments(entries)
        self._assignments = entries

    def __len__(self) -> int:
        with self._lock:
            return len(self._assignments)

    @property
    def assignments(self) -> tuple[SplitAssignment, ...]:
        with self._lock:
            return tuple(
                self._assignments[episode_id]
                for episode_id in sorted(self._assignments)
            )

    @property
    def registry_sha256(self) -> str:
        with self._lock:
            return _payload_digest(self._content_payload())

    def assign_episode(
        self,
        episode_id: str,
        split: DatasetSplit | str,
        *,
        scenario_group_id: str,
        scene_seed: int,
        asset_variant: str,
        camera_seed: int,
        lighting_seed: int,
        parent_episode_id: str | None = None,
    ) -> SplitAssignment:
        """Add an assignment or return the exact existing assignment.

        Any attempt to change an already assigned episode fails closed.
        """

        candidate = SplitAssignment(
            episode_id=episode_id,
            split=_split(split),
            scenario_group_id=scenario_group_id,
            scene_seed=scene_seed,
            asset_variant=asset_variant,
            camera_seed=camera_seed,
            lighting_seed=lighting_seed,
            parent_episode_id=parent_episode_id,
        )
        with self._lock:
            existing = self._assignments.get(candidate.episode_id)
            if existing is not None:
                if existing == candidate:
                    return existing
                raise SplitAssignmentError(
                    f"episode {candidate.episode_id!r} is already assigned to "
                    f"{existing.split.value!r}; reassignment is forbidden"
                )
            proposed = dict(self._assignments)
            proposed[candidate.episode_id] = candidate
            self._validate_assignments(proposed)
            self._assignments[candidate.episode_id] = candidate
            return candidate

    def get_assignment(self, episode_id: str) -> SplitAssignment:
        normalized = _episode_id(episode_id)
        with self._lock:
            assignment = self._assignments.get(normalized)
            if assignment is None:
                raise SplitRegistryError(
                    f"episode {normalized!r} is not registered in any split"
                )
            return assignment

    def get_split(self, episode_id: str) -> DatasetSplit:
        return self.get_assignment(episode_id).split

    def assert_episode_allowed(
        self,
        episode_id: str,
        *,
        is_training: bool,
    ) -> SplitAssignment:
        """Fail closed when a training reader requests Val/Test data."""

        if not isinstance(is_training, bool):
            raise TypeError("is_training must be a bool")
        try:
            assignment = self.get_assignment(episode_id)
        except SplitRegistryError as exc:
            if is_training:
                raise DataLeakageError(
                    f"training access blocked for unregistered episode {episode_id!r}"
                ) from exc
            raise
        if is_training and assignment.split is not DatasetSplit.TRAIN:
            raise DataLeakageError(
                f"training access blocked: episode {episode_id!r} belongs to "
                f"split {assignment.split.value!r}"
            )
        return assignment

    def _content_payload(self) -> dict[str, Any]:
        return {
            "schema_version": SPLIT_REGISTRY_VERSION,
            "assignments": {
                episode_id: self._assignments[episode_id].to_payload()
                for episode_id in sorted(self._assignments)
            },
        }

    def to_document(self) -> dict[str, Any]:
        with self._lock:
            payload = self._content_payload()
            return {**payload, "registry_sha256": _payload_digest(payload)}

    def save(self, path: str | Path) -> Path:
        """Atomically persist the registry without removing old assignments."""

        target = Path(path).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise SplitRegistryError("registry path must not be a symlink")
        target = target.resolve()
        with self._lock:
            if target.exists():
                existing = self.load(target)
                for assignment in existing.assignments:
                    current = self._assignments.get(assignment.episode_id)
                    if current is None:
                        raise SplitAssignmentError(
                            f"save would remove episode {assignment.episode_id!r}"
                        )
                    if current != assignment:
                        raise SplitAssignmentError(
                            f"save would overwrite episode {assignment.episode_id!r}"
                        )

            document = self.to_document()
            serialized = (
                json.dumps(
                    document,
                    ensure_ascii=False,
                    indent=2,
                    sort_keys=True,
                )
                + "\n"
            )
            temporary = target.parent / f".{target.name}.{uuid4().hex}.tmp"
            try:
                with temporary.open("x", encoding="utf-8", newline="\n") as stream:
                    stream.write(serialized)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, target)
            finally:
                if temporary.exists():
                    temporary.unlink()
        return target

    @classmethod
    def load(cls, path: str | Path) -> SplitRegistry:
        source = Path(path).expanduser()
        if source.is_symlink() or not source.is_file():
            raise SplitRegistryError("registry path must be a real file")
        try:
            document = json.loads(
                source.read_text(encoding="utf-8"),
                object_pairs_hook=_reject_duplicate_keys,
            )
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise SplitRegistryError("split registry is unreadable") from exc
        if not isinstance(document, dict) or set(document) != {
            "schema_version",
            "assignments",
            "registry_sha256",
        }:
            raise SplitRegistryError("split registry has invalid top-level fields")
        if document["schema_version"] != SPLIT_REGISTRY_VERSION:
            raise SplitRegistryError("unsupported split registry schema version")
        assignments = document["assignments"]
        if not isinstance(assignments, dict):
            raise SplitRegistryError("split registry assignments must be an object")
        expected_digest = document["registry_sha256"]
        if (
            not isinstance(expected_digest, str)
            or _SHA256.fullmatch(expected_digest) is None
        ):
            raise SplitRegistryIntegrityError("split registry SHA-256 is malformed")
        payload = {
            "schema_version": document["schema_version"],
            "assignments": assignments,
        }
        actual_digest = _payload_digest(payload)
        if not hmac.compare_digest(expected_digest, actual_digest):
            raise SplitRegistryIntegrityError(
                f"split registry SHA-256 mismatch: expected {expected_digest}, "
                f"got {actual_digest}"
            )
        parsed = [
            SplitAssignment.from_payload(episode_id, assignment)
            for episode_id, assignment in assignments.items()
        ]
        return cls(parsed)

    @staticmethod
    def _validate_assignments(
        assignments: Mapping[str, SplitAssignment],
    ) -> None:
        group_splits: dict[tuple[str, int, str, int, int], DatasetSplit] = {}
        scene_seed_splits: dict[int, DatasetSplit] = {}
        for assignment in assignments.values():
            group_split = group_splits.setdefault(
                assignment.group_key,
                assignment.split,
            )
            if group_split is not assignment.split:
                raise SplitAssignmentError(
                    "episodes with the same frozen group key must share one split"
                )
            seed_split = scene_seed_splits.setdefault(
                assignment.scene_seed,
                assignment.split,
            )
            if seed_split is not assignment.split:
                raise SplitAssignmentError(
                    f"scene_seed {assignment.scene_seed} appears in multiple splits"
                )
        for assignment in assignments.values():
            if assignment.parent_episode_id is None:
                continue
            parent = assignments.get(assignment.parent_episode_id)
            if parent is None:
                raise SplitAssignmentError(
                    f"parent episode {assignment.parent_episode_id!r} is not registered"
                )
            if parent.split is not assignment.split:
                raise SplitAssignmentError(
                    "recovery episode and parent episode must share one split"
                )
