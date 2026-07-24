"""Versioned, transport-neutral domain contracts.

Only Python standard-library types are used so this module can be imported by
the supervisor, simulator, test tools, and executor service clients.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence

from .errors import ContractError, FailureCode

TASK_SCHEMA_VERSION = "1.0"
OBSERVATION_VERSION = "1.0"
ACTION_CONTRACT_VERSION = "1.0"
TASK_SCHEMA_VERSION_PATTERN = re.compile(r"^1\.[0-9]+$")

SUPPORTED_TASK_TYPES = frozenset(
    {
        "pick_place",
        "object_localization",
        "visual_manipulation",
        "instruction_interaction",
        "mock_demo",
    }
)


class SubtaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    VERIFIED = "VERIFIED"
    FAILED = "FAILED"


def _major(version: str) -> str:
    return version.split(".", 1)[0]


@dataclass(frozen=True)
class Postcondition:
    """Observable completion rule; never refers to ground truth."""

    kind: str
    path: str = ""
    expected: Any = None
    minimum: float | None = None
    maximum: float | None = None
    object_id: str | None = None
    zone_id: str | None = None
    min_confidence: float = 0.60
    required_votes: int = 2

    def validate(self) -> None:
        if self.kind not in {
            "field_equals",
            "numeric_range",
            "object_detected",
            "object_in_zone",
        }:
            raise ContractError(
                FailureCode.INVALID_TASK, f"unsupported postcondition kind: {self.kind}"
            )
        if (
            isinstance(self.required_votes, bool)
            or not isinstance(self.required_votes, int)
            or self.required_votes < 1
            or self.required_votes > 9
        ):
            raise ContractError(
                FailureCode.INVALID_TASK,
                "required_votes must be an integer in [1, 9]",
            )
        if (
            isinstance(self.min_confidence, bool)
            or not isinstance(self.min_confidence, (int, float))
            or not isfinite(float(self.min_confidence))
            or not 0.0 <= self.min_confidence <= 1.0
        ):
            raise ContractError(
                FailureCode.INVALID_TASK,
                "min_confidence must be a finite number in [0, 1]",
            )
        if self.kind in {"field_equals", "numeric_range"} and not self.path:
            raise ContractError(FailureCode.INVALID_TASK, f"{self.kind} requires path")
        if self.kind == "numeric_range":
            if self.minimum is None and self.maximum is None:
                raise ContractError(
                    FailureCode.INVALID_TASK,
                    "numeric_range requires minimum and/or maximum",
                )
            for name, value in (
                ("minimum", self.minimum),
                ("maximum", self.maximum),
            ):
                if value is not None and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not isfinite(float(value))
                ):
                    raise ContractError(
                        FailureCode.INVALID_TASK,
                        f"numeric_range {name} must be a finite number",
                    )
            if (
                self.minimum is not None
                and self.maximum is not None
                and self.minimum > self.maximum
            ):
                raise ContractError(
                    FailureCode.INVALID_TASK, "minimum cannot exceed maximum"
                )
        if self.kind in {"object_detected", "object_in_zone"} and not self.object_id:
            raise ContractError(
                FailureCode.INVALID_TASK, f"{self.kind} requires object_id"
            )
        if self.kind == "object_in_zone" and not self.zone_id:
            raise ContractError(
                FailureCode.INVALID_TASK, "object_in_zone requires zone_id"
            )

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "Postcondition":
        condition = cls(
            kind=str(value.get("kind", "")),
            path=str(value.get("path", "")),
            expected=value.get("expected"),
            minimum=value.get("minimum"),
            maximum=value.get("maximum"),
            object_id=value.get("object_id"),
            zone_id=value.get("zone_id"),
            min_confidence=value.get("min_confidence", 0.60),
            required_votes=value.get("required_votes", 2),
        )
        condition.validate()
        return condition

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "kind": self.kind,
            "min_confidence": self.min_confidence,
            "required_votes": self.required_votes,
        }
        for key in (
            "path",
            "expected",
            "minimum",
            "maximum",
            "object_id",
            "zone_id",
        ):
            value = getattr(self, key)
            if value not in (None, ""):
                result[key] = value
        return result


@dataclass(frozen=True)
class TaskSchema:
    task_id: str
    instruction: str
    task_type: str
    postconditions: tuple[Postcondition, ...]
    schema_version: str = TASK_SCHEMA_VERSION
    target_object: str | None = None
    target_location: str | None = None
    preferred_executor: str | None = None
    constraints: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if (
            not isinstance(self.schema_version, str)
            or TASK_SCHEMA_VERSION_PATTERN.fullmatch(self.schema_version) is None
        ):
            raise ContractError(
                FailureCode.UNSUPPORTED_TASK_VERSION,
                f"task schema {self.schema_version!r} is incompatible with {TASK_SCHEMA_VERSION!r}",
            )
        if not self.task_id or len(self.task_id) > 128:
            raise ContractError(
                FailureCode.INVALID_TASK, "task_id must contain 1..128 characters"
            )
        if not self.instruction.strip() or len(self.instruction) > 2000:
            raise ContractError(
                FailureCode.INVALID_TASK,
                "instruction must contain 1..2000 non-blank characters",
            )
        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ContractError(
                FailureCode.INVALID_TASK,
                f"unsupported task_type: {self.task_type}",
            )
        if self.preferred_executor not in {None, "openvla_oft", "pi05"}:
            raise ContractError(
                FailureCode.INVALID_TASK,
                f"unsupported preferred_executor: {self.preferred_executor}",
            )
        if not self.postconditions:
            raise ContractError(
                FailureCode.INVALID_TASK, "at least one postcondition is required"
            )
        for condition in self.postconditions:
            condition.validate()

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TaskSchema":
        task = cls(
            schema_version=str(value.get("schema_version", TASK_SCHEMA_VERSION)),
            task_id=str(value.get("task_id", "")),
            instruction=str(value.get("instruction", "")),
            task_type=str(value.get("task_type", "")),
            target_object=value.get("target_object"),
            target_location=value.get("target_location"),
            preferred_executor=value.get("preferred_executor"),
            constraints=dict(value.get("constraints", {})),
            metadata=dict(value.get("metadata", {})),
            postconditions=tuple(
                Postcondition.from_dict(item)
                for item in value.get("postconditions", ())
            ),
        )
        task.validate()
        return task

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "schema_version": self.schema_version,
            "task_id": self.task_id,
            "instruction": self.instruction,
            "task_type": self.task_type,
            "postconditions": [item.to_dict() for item in self.postconditions],
            "constraints": dict(self.constraints),
            "metadata": dict(self.metadata),
        }
        for key in ("target_object", "target_location", "preferred_executor"):
            value = getattr(self, key)
            if value is not None:
                result[key] = value
        return result


@dataclass
class Subtask:
    """Ordered semantic instruction with no low-level geometry fields."""

    subtask_id: str
    sequence: int
    instruction: str
    task_type: str
    preconditions: tuple[Postcondition, ...]
    postconditions: tuple[Postcondition, ...]
    depends_on: tuple[str, ...] = ()
    assigned_executor: str | None = None
    repeat_until_postcondition: bool = False
    max_iterations: int = 1
    status: SubtaskStatus = SubtaskStatus.PENDING

    def validate(self) -> None:
        if not self.subtask_id or self.sequence < 1:
            raise ContractError(
                FailureCode.INVALID_TASK,
                "subtask_id is required and sequence must be >= 1",
            )
        if not self.instruction.strip():
            raise ContractError(
                FailureCode.INVALID_TASK, "subtask instruction is required"
            )
        if self.task_type not in SUPPORTED_TASK_TYPES:
            raise ContractError(
                FailureCode.INVALID_TASK,
                f"unsupported subtask task_type: {self.task_type}",
            )
        if not self.postconditions:
            raise ContractError(
                FailureCode.INVALID_TASK,
                f"subtask {self.subtask_id} requires postconditions",
            )
        if self.max_iterations < 1 or self.max_iterations > 100:
            raise ContractError(
                FailureCode.INVALID_TASK, "max_iterations must be in [1, 100]"
            )
        if not self.repeat_until_postcondition and self.max_iterations != 1:
            raise ContractError(
                FailureCode.INVALID_TASK,
                "non-repeating subtask must have max_iterations=1",
            )
        for condition in (*self.preconditions, *self.postconditions):
            condition.validate()

    def as_task(self, parent: TaskSchema) -> TaskSchema:
        task = TaskSchema(
            task_id=f"{parent.task_id}:{self.subtask_id}",
            instruction=self.instruction,
            task_type=self.task_type,
            postconditions=self.postconditions,
            schema_version=parent.schema_version,
            target_object=parent.target_object,
            target_location=parent.target_location,
            preferred_executor=self.assigned_executor or parent.preferred_executor,
            constraints=parent.constraints,
            metadata={
                **dict(parent.metadata),
                "parent_task_id": parent.task_id,
                "subtask_id": self.subtask_id,
                "subtask_sequence": self.sequence,
            },
        )
        task.validate()
        return task

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "subtask_id": self.subtask_id,
            "sequence": self.sequence,
            "instruction": self.instruction,
            "task_type": self.task_type,
            "preconditions": [item.to_dict() for item in self.preconditions],
            "postconditions": [item.to_dict() for item in self.postconditions],
            "depends_on": list(self.depends_on),
            "repeat_until_postcondition": self.repeat_until_postcondition,
            "max_iterations": self.max_iterations,
            "status": self.status.value,
        }
        if self.assigned_executor is not None:
            result["assigned_executor"] = self.assigned_executor
        return result


@dataclass
class TaskPlan:
    plan_id: str
    episode_id: str
    task_id: str
    subtasks: list[Subtask]
    plan_version: str = "1.0"

    def validate(self) -> None:
        if _major(self.plan_version) != "1":
            raise ContractError(
                FailureCode.INVALID_TASK,
                f"unsupported TaskPlan version: {self.plan_version}",
            )
        if not self.plan_id or not self.episode_id or not self.task_id:
            raise ContractError(
                FailureCode.INVALID_TASK,
                "plan_id, episode_id and task_id are required",
            )
        if not self.subtasks:
            raise ContractError(
                FailureCode.INVALID_TASK, "TaskPlan requires at least one subtask"
            )
        ids: set[str] = set()
        sequences: list[int] = []
        for subtask in self.subtasks:
            subtask.validate()
            if subtask.subtask_id in ids:
                raise ContractError(
                    FailureCode.INVALID_TASK,
                    f"duplicate subtask_id: {subtask.subtask_id}",
                )
            unknown_dependencies = set(subtask.depends_on) - ids
            if unknown_dependencies:
                raise ContractError(
                    FailureCode.INVALID_TASK,
                    f"subtask {subtask.subtask_id} has non-prior dependencies: "
                    f"{sorted(unknown_dependencies)}",
                )
            ids.add(subtask.subtask_id)
            sequences.append(subtask.sequence)
        if sequences != list(range(1, len(self.subtasks) + 1)):
            raise ContractError(
                FailureCode.INVALID_TASK,
                "subtask sequences must be contiguous and ordered from 1",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "plan_version": self.plan_version,
            "plan_id": self.plan_id,
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "subtasks": [item.to_dict() for item in self.subtasks],
        }


@dataclass(frozen=True)
class Observation:
    """Sanitized online observation.

    `data` is a recursively copied view produced by ObservationGateway. Callers
    cannot construct a valid online observation containing GT fields through
    the gateway.
    """

    observation_id: str
    timestamp_ms: int
    data: Mapping[str, Any]
    observation_version: str = OBSERVATION_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_version": self.observation_version,
            "observation_id": self.observation_id,
            "timestamp_ms": self.timestamp_ms,
            **dict(self.data),
        }


@dataclass(frozen=True)
class ActionStep:
    """One 7-D physical command.

    Values are `[dx_m, dy_m, dz_m, droll_rad, dpitch_rad, dyaw_rad,
    gripper_norm]`. `duration_ms` is transport timing metadata, not an eighth
    model output dimension.
    """

    values: tuple[float, float, float, float, float, float, float]
    duration_ms: int = 100

    @classmethod
    def from_sequence(
        cls, values: Sequence[float], duration_ms: int = 100
    ) -> "ActionStep":
        if len(values) != 7:
            raise ContractError(
                FailureCode.ACTION_CONTRACT_INVALID,
                f"7-D action required, received {len(values)} dimensions",
            )
        if any(isinstance(item, bool) for item in values):
            raise ContractError(
                FailureCode.ACTION_CONTRACT_INVALID,
                "boolean values are not valid physical action numbers",
            )
        try:
            normalized = tuple(float(item) for item in values)
        except (TypeError, ValueError) as exc:
            raise ContractError(
                FailureCode.ACTION_CONTRACT_INVALID,
                "action values must be numeric",
            ) from exc
        if duration_ms < 1 or duration_ms > 10_000:
            raise ContractError(
                FailureCode.ACTION_CONTRACT_INVALID,
                "duration_ms must be in [1, 10000]",
            )
        return cls(values=normalized, duration_ms=duration_ms)  # type: ignore[arg-type]

    def has_non_finite(self) -> bool:
        return not all(isfinite(value) for value in self.values)

    def to_dict(self) -> dict[str, Any]:
        return {"values": list(self.values), "duration_ms": self.duration_ms}


@dataclass(frozen=True)
class ActionChunk:
    contract_version: str
    chunk_id: str
    task_id: str
    executor: str
    steps: tuple[ActionStep, ...]
    action_space: str = "ee_delta_pose_gripper"
    frame: str = "robot_base"
    translation_unit: str = "m"
    rotation_unit: str = "rad"
    gripper_unit: str = "normalized"

    def validate_contract(self) -> None:
        if _major(self.contract_version) != _major(ACTION_CONTRACT_VERSION):
            raise ContractError(
                FailureCode.ACTION_CONTRACT_INVALID,
                f"action contract {self.contract_version!r} is incompatible with "
                f"{ACTION_CONTRACT_VERSION!r}",
            )
        if self.action_space != "ee_delta_pose_gripper":
            raise ContractError(
                FailureCode.ACTION_CONTRACT_INVALID,
                f"unsupported action_space: {self.action_space}",
            )
        if (
            self.frame,
            self.translation_unit,
            self.rotation_unit,
            self.gripper_unit,
        ) != (
            "robot_base",
            "m",
            "rad",
            "normalized",
        ):
            raise ContractError(
                FailureCode.ACTION_CONTRACT_INVALID,
                "frame/unit tuple must be robot_base/m/rad/normalized",
            )
        if not self.chunk_id or not self.task_id or not self.executor:
            raise ContractError(
                FailureCode.ACTION_CONTRACT_INVALID,
                "chunk_id, task_id and executor are required",
            )
        if not self.steps:
            raise ContractError(
                FailureCode.ACTION_CONTRACT_INVALID,
                "action chunk must contain at least one step",
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "contract_version": self.contract_version,
            "chunk_id": self.chunk_id,
            "task_id": self.task_id,
            "executor": self.executor,
            "action_space": self.action_space,
            "frame": self.frame,
            "translation_unit": self.translation_unit,
            "rotation_unit": self.rotation_unit,
            "gripper_unit": self.gripper_unit,
            "steps": [step.to_dict() for step in self.steps],
        }
