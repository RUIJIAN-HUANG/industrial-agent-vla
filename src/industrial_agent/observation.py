"""Online observation gateway with allowlist and ground-truth isolation."""

from __future__ import annotations

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

# Compared case-insensitively and after replacing '-' with '_'.
GT_TOKENS = frozenset(
    {
        "gt",
        "ground_truth",
        "groundtruth",
        "label",
        "labels",
        "annotation",
        "annotations",
        "oracle",
        "privileged_state",
    }
)

PRIVILEGED_GEOMETRY_TOKENS = frozenset(
    {
        "target_coordinate",
        "target_coordinates",
        "target_pose",
        "target_position",
        "grasp_point",
        "grasp_pose",
        "trajectory",
        "waypoint",
        "waypoints",
    }
)

REQUIRED_TOP_LEVEL_FIELDS = frozenset(
    {
        "observation_version",
        "observation_id",
        "timestamp_ms",
        "robot",
        "safety",
    }
)


def _scan_for_gt(value: Any, path: str = "$") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized = str(key).lower().replace("-", "_")
            parts = frozenset(part for part in normalized.split("_") if part)
            if (
                normalized in GT_TOKENS
                or "ground_truth" in normalized
                or "groundtruth" in normalized
                or normalized == "gt"
                or normalized.startswith("gt_")
                or normalized.endswith("_gt")
                or "_gt_" in normalized
                or bool(
                    parts.intersection(
                        {"label", "labels", "annotation", "annotations", "oracle"}
                    )
                )
                or normalized in PRIVILEGED_GEOMETRY_TOKENS
            ):
                return f"{path}.{key}"
            found = _scan_for_gt(child, f"{path}.{key}")
            if found:
                return found
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            found = _scan_for_gt(child, f"{path}[{index}]")
            if found:
                return found
    return None


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
        forbidden_path = _scan_for_gt(raw)
        if forbidden_path:
            raise ObservationError(
                FailureCode.OBSERVATION_GT_FORBIDDEN,
                f"ground-truth-like field is forbidden online: {forbidden_path}",
            )
        unknown = set(raw) - ONLINE_TOP_LEVEL_ALLOWLIST
        if unknown:
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID,
                f"online observation contains non-allowlisted fields: "
                f"{sorted(unknown)}",
            )
        missing = REQUIRED_TOP_LEVEL_FIELDS - set(raw)
        if missing:
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID,
                f"online observation is missing required fields: {sorted(missing)}",
            )
        version = str(raw.get("observation_version", OBSERVATION_VERSION))
        if version.split(".", 1)[0] != OBSERVATION_VERSION.split(".", 1)[0]:
            raise ObservationError(
                FailureCode.OBSERVATION_INVALID,
                f"incompatible observation version: {version}",
            )
        observation_id = str(raw.get("observation_id", ""))
        if not observation_id:
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
