"""Frozen offline success gates for the V2 P01-to-S11 task.

This module deliberately has no Isaac Sim imports.  It accepts measurements
from :mod:`simulation.offline_gt` and returns auditable, deterministic gate
results.  Detailed measurements remain in an ``offline_gt`` sidecar; the
Canonical episode receives only the final outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence


P01_MAX_VERTICAL_ERROR_RAD = math.radians(15.0)
P01_GT_VOTE_COUNT = 3
P01_GT_MIN_PASSES = 2
P01_HOLD_DURATION_S = 1.0
P01_MAX_HOLD_DRIFT_M = 0.001


def _finite_vector(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    try:
        vector = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(item) for item in vector):
        raise ValueError(f"{name} must contain finite values")
    norm = math.sqrt(sum(item * item for item in vector))
    if norm <= 0.0:
        raise ValueError(f"{name} must not be the zero vector")
    return vector


def _finite_point(value: Sequence[float], *, name: str) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError(f"{name} must contain exactly three values")
    try:
        point = tuple(float(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric values") from exc
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{name} must contain finite values")
    return point


def vertical_error_rad(
    p01_axis_world: Sequence[float],
    bin_vertical_world: Sequence[float],
) -> float:
    """Return the directed angle between P01's axis and bin vertical.

    The direction is intentionally not folded with ``abs(dot)``: an inverted
    P01 is not equivalent to an upright P01 for the frozen task.
    """

    part = _finite_vector(p01_axis_world, name="p01_axis_world")
    vertical = _finite_vector(bin_vertical_world, name="bin_vertical_world")
    part_norm = math.sqrt(sum(item * item for item in part))
    vertical_norm = math.sqrt(sum(item * item for item in vertical))
    cosine = sum(a * b for a, b in zip(part, vertical)) / (part_norm * vertical_norm)
    return math.acos(max(-1.0, min(1.0, cosine)))


def _timestamp_seconds(report: Mapping[str, Any]) -> float:
    if "timestamp_s" in report:
        value = report["timestamp_s"]
    elif "timestamp_ns" in report:
        value = float(report["timestamp_ns"]) / 1_000_000_000.0
    else:
        raise ValueError("GT report requires timestamp_s or timestamp_ns")
    try:
        timestamp = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("GT timestamp must be numeric") from exc
    if not math.isfinite(timestamp):
        raise ValueError("GT timestamp must be finite")
    return timestamp


def evaluate_fresh_gt_votes(
    reports: Sequence[Mapping[str, Any]],
    *,
    expected_count: int = P01_GT_VOTE_COUNT,
    minimum_passes: int = P01_GT_MIN_PASSES,
    failure_prefix: str = "P01",
) -> dict[str, Any]:
    """Require exactly three unique, chronological fresh GT observations."""

    if len(reports) != expected_count:
        return {
            "pass": False,
            "count": len(reports),
            "passed_count": 0,
            "failure_codes": [f"{failure_prefix}_GT_NOT_FRESH"],
        }

    ids: list[str] = []
    timestamps: list[float] = []
    for report in reports:
        observation_id = report.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            return {
                "pass": False,
                "count": len(reports),
                "passed_count": 0,
                "failure_codes": [f"{failure_prefix}_GT_NOT_FRESH"],
            }
        ids.append(observation_id)
        try:
            timestamps.append(_timestamp_seconds(report))
        except ValueError:
            return {
                "pass": False,
                "count": len(reports),
                "passed_count": 0,
                "failure_codes": [f"{failure_prefix}_GT_NOT_FRESH"],
            }

    if len(set(ids)) != expected_count or any(
        current <= previous for previous, current in zip(timestamps, timestamps[1:])
    ):
        return {
            "pass": False,
            "count": len(reports),
            "passed_count": 0,
            "failure_codes": [f"{failure_prefix}_GT_NOT_FRESH"],
        }

    passed_count = sum(report.get("pass") is True for report in reports)
    passed = passed_count >= minimum_passes
    return {
        "pass": passed,
        "count": len(reports),
        "passed_count": passed_count,
        "failure_codes": [] if passed else [f"{failure_prefix}_GT_VOTE_INSUFFICIENT"],
    }


def evaluate_terminal_hold_drift(
    positions_world: Sequence[Sequence[float]],
    timestamps_s: Sequence[float],
    *,
    minimum_duration_s: float = P01_HOLD_DURATION_S,
    max_drift_m: float = P01_MAX_HOLD_DRIFT_M,
    failure_prefix: str = "P01",
) -> dict[str, Any]:
    """Check a real one-second hold against the initial P01 position."""

    if len(positions_world) < 2 or len(positions_world) != len(timestamps_s):
        return {
            "pass": False,
            "duration_s": 0.0,
            "max_drift_m": None,
            "failure_codes": [f"{failure_prefix}_TERMINAL_HOLD_TOO_SHORT"],
        }
    try:
        points = [
            _finite_point(point, name="position_world") for point in positions_world
        ]
        times = [float(value) for value in timestamps_s]
    except (TypeError, ValueError) as exc:
        raise ValueError("terminal hold positions and timestamps are invalid") from exc
    if not all(math.isfinite(value) for value in times):
        raise ValueError("terminal hold timestamps must be finite")
    if any(current <= previous for previous, current in zip(times, times[1:])):
        raise ValueError("terminal hold timestamps must be strictly increasing")
    duration_s = times[-1] - times[0]
    max_drift = max(
        math.sqrt(sum((current[i] - points[0][i]) ** 2 for i in range(3)))
        for current in points
    )
    passed = duration_s >= minimum_duration_s and max_drift <= max_drift_m
    failure_codes: list[str] = []
    if duration_s < minimum_duration_s:
        failure_codes.append(f"{failure_prefix}_TERMINAL_HOLD_TOO_SHORT")
    if max_drift > max_drift_m:
        failure_codes.append(f"{failure_prefix}_TERMINAL_DRIFT_EXCEEDED")
    return {
        "pass": passed,
        "duration_s": duration_s,
        "max_drift_m": max_drift,
        "failure_codes": failure_codes,
    }


@dataclass(frozen=True)
class P01TerminalSuccess:
    """Complete, serializable result for the three frozen gates."""

    passed: bool
    orientation_pass: bool
    orientation_error_rad: float
    fresh_vote_pass: bool
    fresh_vote: Mapping[str, Any]
    hold_drift_pass: bool
    hold_drift: Mapping[str, Any]
    failure_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "orientation_pass": self.orientation_pass,
            "orientation_error_rad": self.orientation_error_rad,
            "orientation_threshold_rad": P01_MAX_VERTICAL_ERROR_RAD,
            "fresh_vote_pass": self.fresh_vote_pass,
            "fresh_vote": dict(self.fresh_vote),
            "hold_drift_pass": self.hold_drift_pass,
            "hold_drift": dict(self.hold_drift),
            "hold_duration_threshold_s": P01_HOLD_DURATION_S,
            "hold_drift_threshold_m": P01_MAX_HOLD_DRIFT_M,
            "failure_codes": list(self.failure_codes),
            "canonical_included": False,
        }


def evaluate_p01_terminal_success(
    *,
    orientation_error_rad: float,
    vote_reports: Sequence[Mapping[str, Any]],
    positions_world: Sequence[Sequence[float]],
    timestamps_s: Sequence[float],
) -> P01TerminalSuccess:
    """Evaluate orientation, three fresh votes, and the one-second hold."""

    try:
        angle = float(orientation_error_rad)
    except (TypeError, ValueError) as exc:
        raise ValueError("orientation_error_rad must be numeric") from exc
    if not math.isfinite(angle) or angle < 0.0:
        raise ValueError("orientation_error_rad must be finite and non-negative")
    orientation_pass = angle <= P01_MAX_VERTICAL_ERROR_RAD
    orientation_codes = [] if orientation_pass else ["P01_ORIENTATION_EXCEEDED"]
    vote = evaluate_fresh_gt_votes(vote_reports)
    drift = evaluate_terminal_hold_drift(positions_world, timestamps_s)
    failure_codes = tuple(
        dict.fromkeys(
            orientation_codes
            + list(vote["failure_codes"])
            + list(drift["failure_codes"])
        )
    )
    return P01TerminalSuccess(
        passed=orientation_pass and vote["pass"] and drift["pass"],
        orientation_pass=orientation_pass,
        orientation_error_rad=angle,
        fresh_vote_pass=bool(vote["pass"]),
        fresh_vote=vote,
        hold_drift_pass=bool(drift["pass"]),
        hold_drift=drift,
        failure_codes=failure_codes,
    )


@dataclass(frozen=True)
class W01TerminalSuccess:
    passed: bool
    orientation_pass: bool
    flat_error_rad: float
    heading_error_rad: float
    fresh_vote_pass: bool
    fresh_vote: Mapping[str, Any]
    hold_drift_pass: bool
    hold_drift: Mapping[str, Any]
    failure_codes: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "orientation_pass": self.orientation_pass,
            "flat_error_rad": self.flat_error_rad,
            "heading_error_rad": self.heading_error_rad,
            "orientation_threshold_rad": P01_MAX_VERTICAL_ERROR_RAD,
            "fresh_vote_pass": self.fresh_vote_pass,
            "fresh_vote": dict(self.fresh_vote),
            "hold_drift_pass": self.hold_drift_pass,
            "hold_drift": dict(self.hold_drift),
            "hold_duration_threshold_s": P01_HOLD_DURATION_S,
            "hold_drift_threshold_m": P01_MAX_HOLD_DRIFT_M,
            "failure_codes": list(self.failure_codes),
            "canonical_included": False,
        }


def evaluate_w01_terminal_success(
    *,
    flat_error_rad: float,
    heading_error_rad: float,
    vote_reports: Sequence[Mapping[str, Any]],
    positions_world: Sequence[Sequence[float]],
    timestamps_s: Sequence[float],
) -> W01TerminalSuccess:
    angles = (float(flat_error_rad), float(heading_error_rad))
    if any(not math.isfinite(value) or value < 0.0 for value in angles):
        raise ValueError("W01 orientation errors must be finite and non-negative")
    orientation_pass = all(value <= P01_MAX_VERTICAL_ERROR_RAD for value in angles)
    orientation_codes = [] if orientation_pass else ["W01_ORIENTATION_EXCEEDED"]
    vote = evaluate_fresh_gt_votes(vote_reports, failure_prefix="W01")
    drift = evaluate_terminal_hold_drift(
        positions_world, timestamps_s, failure_prefix="W01"
    )
    failure_codes = tuple(dict.fromkeys(
        orientation_codes + list(vote["failure_codes"]) + list(drift["failure_codes"])
    ))
    return W01TerminalSuccess(
        passed=orientation_pass and vote["pass"] and drift["pass"],
        orientation_pass=orientation_pass,
        flat_error_rad=angles[0],
        heading_error_rad=angles[1],
        fresh_vote_pass=bool(vote["pass"]),
        fresh_vote=vote,
        hold_drift_pass=bool(drift["pass"]),
        hold_drift=drift,
        failure_codes=failure_codes,
    )
