"""Online observation gateway with allowlist and ground-truth isolation."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, Mapping

from .contracts import OBSERVATION_VERSION, Observation
from .errors import FailureCode, ObservationError

ONLINE_TOP_LEVEL_ALLOWLIST = frozenset(
    {
        "observation_version",
        "observation_id",
        "timestamp_ms",
        "camera",
        "objects",
        "robot",
        "safety",
        "task",
        "quality",
    }
)

GT_TOKENS = frozenset(
    {
        "gt",
        "groundtruth",
        "label",
        "labels",
        "annotation",
        "annotations",
        "oracle",
        "privileged",
        "privileged_state",
        "truth",
    }
)

PRIVILEGED_REFERENCE_TOKENS = frozenset(
    {
        "actual",
        "actuals",
        "desired",
        "desireds",
        "destination",
        "destinations",
        "goal",
        "goals",
        "grasp",
        "grasps",
        "object",
        "real",
        "reals",
        "reference",
        "references",
        "target",
        "targets",
        "true",
        "waypoint",
        "waypoints",
    }
)

PRIVILEGED_GEOMETRY_TOKENS = frozenset(
    {
        "coord",
        "coordinate",
        "coordinates",
        "coords",
        "euler",
        "eulers",
        "matrix",
        "matrices",
        "orientation",
        "orientations",
        "location",
        "locations",
        "point",
        "points",
        "pos",
        "pose",
        "poses",
        "position",
        "positions",
        "quat",
        "quaternion",
        "quaternions",
        "rotation",
        "rotations",
        "rpy",
        "se3",
        "tf",
        "transform",
        "transforms",
        "translation",
        "translations",
        "xyz",
        "xyzw",
    }
)

PRIVILEGED_AXIS_TOKENS = frozenset(
    {
        "pitch",
        "qw",
        "qx",
        "qy",
        "qz",
        "rx",
        "ry",
        "rz",
        "roll",
        "theta",
        "x",
        "y",
        "yaw",
        "z",
    }
)

ALWAYS_PRIVILEGED_TOKENS = frozenset(
    {
        "trajectory",
        "trajectories",
        "waypoint",
        "waypoints",
    }
)

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "observation_version",
        "observation_id",
        "timestamp_ms",
        "camera",
        "robot",
        "safety",
        "task",
        "quality",
    }
)


def _normalized_key_tokens(key: object) -> tuple[str, ...]:
    """Normalize common field-name styles into lower-case semantic tokens."""

    raw = str(key).strip()
    raw = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", raw)
    raw = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw)
    raw = re.sub(r"[\W_]+", " ", raw, flags=re.UNICODE)
    tokens: list[str] = []
    for part in raw.split():
        normalized = re.sub(r"\d+$", "", part.casefold())
        if normalized:
            tokens.append(normalized)
    return tuple(tokens)


def _contains_gt_marker(tokens: tuple[str, ...]) -> bool:
    token_set = frozenset(tokens)
    compact = "".join(tokens)
    has_ground_truth_pair = any(
        first == "ground" and second == "truth"
        for first, second in zip(tokens, tokens[1:])
    )
    return (
        bool(token_set.intersection(GT_TOKENS))
        or has_ground_truth_pair
        or "groundtruth" in compact
    )


def _contains_numeric_payload(value: Any) -> bool:
    """Detect scalars, vectors, or matrices that can encode privileged geometry."""

    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return True
    if isinstance(value, Mapping):
        return any(_contains_numeric_payload(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_numeric_payload(child) for child in value)
    return False


def _scan_for_gt(
    value: Any,
    path: str = "$",
    *,
    privileged_reference_context: bool = False,
) -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            child_path = f"{path}.{key}"
            tokens = _normalized_key_tokens(key)
            token_set = frozenset(tokens)
            has_reference = bool(token_set.intersection(PRIVILEGED_REFERENCE_TOKENS))
            has_geometry = bool(token_set.intersection(PRIVILEGED_GEOMETRY_TOKENS))
            has_axis = bool(token_set.intersection(PRIVILEGED_AXIS_TOKENS))

            if _contains_gt_marker(tokens):
                return child_path
            if token_set.intersection(ALWAYS_PRIVILEGED_TOKENS):
                return child_path
            if has_reference and (has_geometry or has_axis):
                return child_path
            if privileged_reference_context and (has_geometry or has_axis):
                return child_path
            if (
                has_reference or privileged_reference_context
            ) and _contains_numeric_payload(child):
                return child_path

            found = _scan_for_gt(
                child,
                child_path,
                privileged_reference_context=(
                    privileged_reference_context or has_reference
                ),
            )
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _scan_for_gt(
                child,
                f"{path}[{index}]",
                privileged_reference_context=privileged_reference_context,
            )
            if found:
                return found
    return None


def find_forbidden_online_path(value: Any, path: str = "$") -> str | None:
    """Return the first GT/oracle-like nested path using the canonical scanner."""

    return _scan_for_gt(value, path)


class ObservationGateway:
    """Single ingress for data visible to the online decision loop."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        """Start a new run-local freshness window."""

        self._seen_observation_ids: set[str] = set()
        self._last_timestamp_ms: int | None = None

    def ingest_online(self, raw: Mapping[str, Any]) -> Observation:
        if not isinstance(raw, Mapping):
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID,
                "online observation must be an object",
            )
        forbidden_path = find_forbidden_online_path(raw)
        if forbidden_path:
            raise ObservationError(
                FailureCode.OBSERVATION_GT_FORBIDDEN,
                f"ground-truth-like field is forbidden online: {forbidden_path}",
            )
        unknown = set(raw) - ONLINE_TOP_LEVEL_ALLOWLIST
        if unknown:
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID,
                f"online observation contains non-allowlisted fields: {sorted(unknown)}",
            )
        missing = REQUIRED_TOP_LEVEL_FIELDS - set(raw)
        if missing:
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID,
                f"online observation is missing required fields: {sorted(missing)}",
            )
        version = raw.get("observation_version")
        if (
            not isinstance(version, str)
            or version.split(".", 1)[0] != OBSERVATION_VERSION.split(".", 1)[0]
        ):
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID,
                f"incompatible observation version: {version}",
            )
        observation_id = raw.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID, "observation_id is required"
            )
        if observation_id in self._seen_observation_ids:
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID,
                f"observation_id must be fresh within a run: {observation_id}",
            )
        timestamp = raw.get("timestamp_ms")
        if (
            isinstance(timestamp, bool)
            or not isinstance(timestamp, int)
            or timestamp < 0
        ):
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID,
                "timestamp_ms must be a non-negative integer",
            )
        if self._last_timestamp_ms is not None and timestamp < self._last_timestamp_ms:
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID,
                "timestamp_ms moved backwards within a run",
            )
        data = {
            key: deepcopy(value)
            for key, value in raw.items()
            if key not in {"observation_version", "observation_id", "timestamp_ms"}
        }
        self._seen_observation_ids.add(observation_id)
        self._last_timestamp_ms = timestamp
        return Observation(
            observation_version=version,
            observation_id=observation_id,
            timestamp_ms=timestamp,
            data=data,
        )
