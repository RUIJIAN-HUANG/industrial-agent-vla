"""Postcondition verification using multiple online observation frames."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isfinite
from typing import Any, Mapping, Sequence

from .contracts import Observation, Postcondition, TaskSchema
from .errors import FailureCode


class Verdict(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    UNCERTAIN = "UNCERTAIN"


@dataclass(frozen=True)
class ConditionResult:
    kind: str
    verdict: Verdict
    pass_votes: int
    fail_votes: int
    uncertain_votes: int
    required_votes: int
    detail: str


@dataclass(frozen=True)
class VerificationResult:
    verdict: Verdict
    code: FailureCode
    conditions: tuple[ConditionResult, ...]


_MISSING = object()


def _resolve_path(data: Mapping[str, Any], path: str) -> Any:
    current: Any = data
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return _MISSING
        current = current[part]
    return current


def _frame_verdict(
    condition: Postcondition, observation: Observation
) -> tuple[Verdict, str]:
    quality = observation.data.get("quality", {})
    if not isinstance(quality, Mapping):
        return Verdict.UNCERTAIN, "frame quality is not an object"
    confidence = quality.get("confidence", 1.0)
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not isfinite(float(confidence))
    ):
        return Verdict.UNCERTAIN, "frame confidence is not a finite number"
    if float(confidence) < condition.min_confidence:
        return Verdict.UNCERTAIN, "frame confidence below threshold"
    if condition.kind == "field_equals":
        value = _resolve_path(observation.data, condition.path)
        if value is _MISSING:
            return Verdict.UNCERTAIN, f"path {condition.path!r} missing"
        return (
            (Verdict.PASS, "field equals expected")
            if value == condition.expected
            else (Verdict.FAIL, f"observed {value!r}, expected {condition.expected!r}")
        )
    if condition.kind == "numeric_range":
        value = _resolve_path(observation.data, condition.path)
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(float(value))
        ):
            return (
                Verdict.UNCERTAIN,
                f"path {condition.path!r} is not a finite numeric value",
            )
        if condition.minimum is not None and value < condition.minimum:
            return Verdict.FAIL, f"{value} is below {condition.minimum}"
        if condition.maximum is not None and value > condition.maximum:
            return Verdict.FAIL, f"{value} is above {condition.maximum}"
        return Verdict.PASS, "numeric value is within range"
    objects = observation.data.get("objects", ())
    if not isinstance(objects, (list, tuple)):
        return Verdict.UNCERTAIN, "objects is not an array"
    matched = None
    for item in objects:
        if isinstance(item, Mapping) and item.get("object_id") == condition.object_id:
            matched = item
            break
    if matched is None:
        return Verdict.FAIL, f"object {condition.object_id!r} not detected"
    detection_confidence = matched.get("confidence", 0.0)
    if (
        not isinstance(detection_confidence, (int, float))
        or isinstance(detection_confidence, bool)
        or not isfinite(float(detection_confidence))
        or float(detection_confidence) < condition.min_confidence
    ):
        return (
            Verdict.UNCERTAIN,
            "object detection confidence is invalid or below threshold",
        )
    if condition.kind == "object_detected":
        return Verdict.PASS, "object detected"
    observed_zone = matched.get("zone_id")
    return (
        (Verdict.PASS, "object is in expected zone")
        if observed_zone == condition.zone_id
        else (
            Verdict.FAIL,
            f"object zone is {observed_zone!r}, expected {condition.zone_id!r}",
        )
    )


class PostconditionVerifier:
    def verify(
        self, task: TaskSchema, frames: Sequence[Observation]
    ) -> VerificationResult:
        if not frames:
            return VerificationResult(
                Verdict.UNCERTAIN, FailureCode.VERIFICATION_UNCERTAIN, ()
            )
        observation_ids = [frame.observation_id for frame in frames]
        timestamps = [frame.timestamp_ms for frame in frames]
        if len(observation_ids) != len(set(observation_ids)):
            return VerificationResult(
                Verdict.UNCERTAIN, FailureCode.OBSERVATION_INVALID, ()
            )
        if timestamps != sorted(timestamps):
            return VerificationResult(
                Verdict.UNCERTAIN, FailureCode.OBSERVATION_INVALID, ()
            )
        results: list[ConditionResult] = []
        for condition in task.postconditions:
            votes = [_frame_verdict(condition, frame) for frame in frames]
            pass_votes = sum(item[0] is Verdict.PASS for item in votes)
            fail_votes = sum(item[0] is Verdict.FAIL for item in votes)
            uncertain_votes = len(votes) - pass_votes - fail_votes
            conflicting_quorum = (
                pass_votes >= condition.required_votes
                and fail_votes >= condition.required_votes
            )
            if conflicting_quorum:
                # Sufficient contradictory evidence must never assert task
                # completion. FAIL is deliberately stricter than UNCERTAIN so
                # the supervisor enters its bounded recovery path.
                verdict = Verdict.FAIL
            elif pass_votes >= condition.required_votes:
                verdict = Verdict.PASS
            elif fail_votes >= condition.required_votes:
                verdict = Verdict.FAIL
            else:
                verdict = Verdict.UNCERTAIN
            detail = "; ".join(detail for _, detail in votes)
            if conflicting_quorum:
                detail = f"conflicting pass/fail quorums; fail-closed; {detail}"
            results.append(
                ConditionResult(
                    kind=condition.kind,
                    verdict=verdict,
                    pass_votes=pass_votes,
                    fail_votes=fail_votes,
                    uncertain_votes=uncertain_votes,
                    required_votes=condition.required_votes,
                    detail=detail,
                )
            )
        if all(item.verdict is Verdict.PASS for item in results):
            return VerificationResult(Verdict.PASS, FailureCode.NONE, tuple(results))
        if any(item.verdict is Verdict.FAIL for item in results):
            return VerificationResult(
                Verdict.FAIL, FailureCode.POSTCONDITION_FAILED, tuple(results)
            )
        return VerificationResult(
            Verdict.UNCERTAIN, FailureCode.VERIFICATION_UNCERTAIN, tuple(results)
        )
