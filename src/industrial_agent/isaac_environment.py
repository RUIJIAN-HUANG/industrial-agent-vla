"""Fail-closed compare-and-execute boundary for Isaac Sim.

The Supervisor invokes environment methods from deadline worker threads.  All
Isaac-backed observation, validation, action and stop operations are therefore
marshalled through an injected owner-thread gate.  The module itself has no
Isaac/Omniverse imports and remains unit-testable on ordinary Python.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from math import isfinite
import os
from pathlib import Path
from threading import Event, Lock
from typing import Any, Callable, Mapping, Protocol

from .contracts import ActionStep
from .environment import (
    PreWriteStateStaleError,
    SafeStopReceipt,
    execution_guard_digest,
)
from .isaac_runtime import IsaacGateTimeoutError
from .sync_contract import FROZEN_MULTI_RATE


class IsaacFrankaController(Protocol):
    """Controller operations backed by the live Isaac owner thread."""

    def validate_ready(self, arm_id: str) -> None:
        """Fail unless the arm can move and the opposite arm is stopped."""

    def execute_action(self, action: ActionStep, *, arm_id: str) -> None:
        """Send one validated action to the selected Franka controller."""

    def request_stop(self, reason: str) -> str:
        """Thread-safely revoke motion without touching an Isaac API."""

    def confirm_safe_stop(
        self,
        reason: str,
        *,
        stop_epoch: str,
    ) -> SafeStopReceipt:
        """On the owner thread, hold/pause both arms and read back the stop."""


class IsaacRuntimeGate(Protocol):
    """Small protocol implemented by :class:`IsaacMainThreadGate`."""

    def call(
        self,
        operation: Callable[[], Any],
        *,
        timeout_s: float,
        label: str,
        on_started_timeout: Callable[[], None] | None = None,
    ) -> Any:
        """Run normal work on the controlled runtime thread."""

    def call_stop(
        self,
        *,
        signal_stop: Callable[[], None],
        apply_stop: Callable[[], Any],
        timeout_s: float,
        label: str = "Isaac safe-stop",
    ) -> Any:
        """Signal stop immediately and run confirmation with urgent priority."""


@dataclass(frozen=True)
class _LedgerEntry:
    state: str
    request_digest: str
    result: dict[str, Any] | None = None


class DurableCommandIdLedger:
    """Append-only, fsync-backed command state journal.

    State transitions are:

    ``CLAIMED -> ABORTED`` when no hardware write was attempted, or
    ``CLAIMED -> APPLIED -> ACKED`` after a controller write and durable result.

    A restart with ``CLAIMED`` or ``APPLIED`` is quarantined because the
    physical outcome is unknown.  An ``ACKED`` duplicate with the same request
    digest returns the original observation and never moves a robot again.
    """

    _VERSION = "2.0"
    _TERMINAL_STATES = frozenset({"ABORTED", "ACKED"})
    _UNRESOLVED_STATES = frozenset({"CLAIMED", "APPLIED"})

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser()
        self._lock = Lock()
        self._entries = self._load()

    @staticmethod
    def _record_fields(state: str) -> set[str]:
        common = {
            "ledger_version",
            "command_id",
            "state",
            "request_digest",
        }
        if state == "ACKED":
            return common | {"result"}
        if state == "ABORTED":
            return common | {"reason"}
        return common

    def _load(self) -> dict[str, _LedgerEntry]:
        if not self.path.exists():
            return {}
        if self.path.is_symlink() or not self.path.is_file():
            raise RuntimeError(f"command ledger must be a regular file: {self.path}")

        entries: dict[str, _LedgerEntry] = {}
        try:
            with self.path.open("r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        raise RuntimeError(
                            f"command ledger contains a blank line at {line_number}"
                        )
                    record = json.loads(line)
                    if not isinstance(record, dict):
                        raise RuntimeError(
                            f"command ledger line {line_number} is not an object"
                        )
                    state = record.get("state")
                    if (
                        record.get("ledger_version") != self._VERSION
                        or state
                        not in {
                            "CLAIMED",
                            "ABORTED",
                            "APPLIED",
                            "ACKED",
                        }
                        or set(record) != self._record_fields(str(state))
                    ):
                        raise RuntimeError(
                            f"command ledger line {line_number} has an invalid shape"
                        )
                    command_id = record.get("command_id")
                    request_digest = record.get("request_digest")
                    if (
                        not isinstance(command_id, str)
                        or not command_id
                        or not isinstance(request_digest, str)
                        or not request_digest.startswith("sha256:")
                    ):
                        raise RuntimeError(
                            f"command ledger line {line_number} has invalid identity"
                        )

                    previous = entries.get(command_id)
                    previous_state = previous.state if previous is not None else None
                    allowed_previous = {
                        "CLAIMED": {None},
                        "ABORTED": {"CLAIMED"},
                        "APPLIED": {"CLAIMED"},
                        "ACKED": {"APPLIED"},
                    }[str(state)]
                    if previous_state not in allowed_previous:
                        raise RuntimeError(
                            "command ledger contains an illegal transition for "
                            f"{command_id!r}: {previous_state!r}->{state!r}"
                        )
                    if (
                        previous is not None
                        and previous.request_digest != request_digest
                    ):
                        raise RuntimeError(
                            f"command ledger request digest changed for {command_id!r}"
                        )

                    result: dict[str, Any] | None = None
                    if state == "ACKED":
                        raw_result = record.get("result")
                        if not isinstance(raw_result, dict):
                            raise RuntimeError(
                                f"ACKED command {command_id!r} has no result object"
                            )
                        result = deepcopy(raw_result)
                    entries[command_id] = _LedgerEntry(
                        state=str(state),
                        request_digest=request_digest,
                        result=result,
                    )
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(
                f"command ledger cannot be loaded safely: {self.path}"
            ) from exc
        return entries

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def _append(self, record: Mapping[str, Any]) -> None:
        try:
            encoded = (
                json.dumps(
                    dict(record),
                    ensure_ascii=False,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                )
                + "\n"
            ).encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise RuntimeError("command ledger record is not JSON-safe") from exc

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists() and (self.path.is_symlink() or not self.path.is_file()):
            raise RuntimeError("command ledger must remain a regular file")
        file_existed = self.path.exists()
        try:
            with self.path.open("ab+") as handle:
                handle.seek(0, os.SEEK_END)
                previous_size = handle.tell()
                try:
                    handle.write(encoded)
                    handle.flush()
                    os.fsync(handle.fileno())
                except BaseException:
                    try:
                        handle.seek(previous_size)
                        handle.truncate()
                        handle.flush()
                        os.fsync(handle.fileno())
                    except OSError:
                        pass
                    raise
            if not file_existed:
                self._fsync_directory(self.path.parent)
        except OSError as exc:
            raise RuntimeError(f"command ledger append failed: {self.path}") from exc

    def acknowledged_result(
        self,
        command_id: str,
        request_digest: str,
    ) -> dict[str, Any] | None:
        """Return an idempotent ACK, or fail on ID reuse/unknown outcome."""

        with self._lock:
            entry = self._entries.get(command_id)
            if entry is None:
                return None
            if entry.request_digest != request_digest:
                raise RuntimeError(
                    f"command_id {command_id!r} was reused for a different request"
                )
            if entry.state == "ACKED":
                assert entry.result is not None
                return deepcopy(entry.result)
            if entry.state == "ABORTED":
                raise RuntimeError(
                    f"command_id {command_id!r} was aborted; use a new command_id"
                )
            raise RuntimeError(
                f"command_id {command_id!r} has unknown physical outcome "
                f"({entry.state})"
            )

    def claim(self, command_id: str, request_digest: str) -> None:
        with self._lock:
            if command_id in self._entries:
                raise RuntimeError(f"duplicate command_id rejected: {command_id!r}")
            record = {
                "ledger_version": self._VERSION,
                "command_id": command_id,
                "state": "CLAIMED",
                "request_digest": request_digest,
            }
            self._append(record)
            self._entries[command_id] = _LedgerEntry(
                state="CLAIMED",
                request_digest=request_digest,
            )

    def _transition(
        self,
        command_id: str,
        request_digest: str,
        *,
        expected_state: str,
        target_state: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        with self._lock:
            entry = self._entries.get(command_id)
            if (
                entry is None
                or entry.request_digest != request_digest
                or entry.state != expected_state
            ):
                raise RuntimeError(
                    f"illegal durable command transition for {command_id!r}: "
                    f"expected {expected_state}"
                )
            record = {
                "ledger_version": self._VERSION,
                "command_id": command_id,
                "state": target_state,
                "request_digest": request_digest,
                **dict(extra or {}),
            }
            self._append(record)
            result = deepcopy(record["result"]) if target_state == "ACKED" else None
            self._entries[command_id] = _LedgerEntry(
                state=target_state,
                request_digest=request_digest,
                result=result,
            )

    def abort(
        self,
        command_id: str,
        request_digest: str,
        *,
        reason: str,
    ) -> None:
        self._transition(
            command_id,
            request_digest,
            expected_state="CLAIMED",
            target_state="ABORTED",
            extra={"reason": reason[:512]},
        )

    def mark_applied(self, command_id: str, request_digest: str) -> None:
        self._transition(
            command_id,
            request_digest,
            expected_state="CLAIMED",
            target_state="APPLIED",
        )

    def acknowledge(
        self,
        command_id: str,
        request_digest: str,
        result: Mapping[str, Any],
    ) -> None:
        self._transition(
            command_id,
            request_digest,
            expected_state="APPLIED",
            target_state="ACKED",
            extra={"result": deepcopy(dict(result))},
        )

    def is_unresolved(self, command_id: str) -> bool:
        with self._lock:
            entry = self._entries.get(command_id)
            return entry is not None and entry.state in self._UNRESOLVED_STATES

    @property
    def unresolved_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(
                sorted(
                    command_id
                    for command_id, entry in self._entries.items()
                    if entry.state in self._UNRESOLVED_STATES
                )
            )


class _UnsafeGuardChange(RuntimeError):
    """A live change that requires immediate safe-stop."""


class _StaleGuardChange(RuntimeError):
    """A live task/object change that requires a fresh VLA plan."""


def _numeric_sequence_close(
    expected: Any,
    current: Any,
    *,
    tolerance: float,
) -> bool:
    if (
        not isinstance(expected, (list, tuple))
        or not isinstance(current, (list, tuple))
        or len(expected) != len(current)
    ):
        return False
    for left, right in zip(expected, current):
        if (
            isinstance(left, bool)
            or isinstance(right, bool)
            or not isinstance(left, (int, float))
            or not isinstance(right, (int, float))
            or not isfinite(float(left))
            or not isfinite(float(right))
            or abs(float(left) - float(right)) > tolerance
        ):
            return False
    return True


def _numeric_scalar_close(expected: Any, current: Any, *, tolerance: float) -> bool:
    if (
        isinstance(expected, bool)
        or isinstance(current, bool)
        or not isinstance(expected, (int, float))
        or not isinstance(current, (int, float))
    ):
        return False
    left = float(expected)
    right = float(current)
    return isfinite(left) and isfinite(right) and abs(left - right) <= tolerance


class IsaacExecutionEnvironment:
    """Atomic, durable execution boundary for the frozen dual-arm cell."""

    _AUTHORIZATION = {
        "Arm_A": "A_ONLY",
        "Arm_B": "B_ONLY",
    }

    def __init__(
        self,
        *,
        observation_source: Callable[[], Mapping[str, Any]],
        state_guard_source: Callable[[], Mapping[str, Any]],
        control_lease_source: Callable[[], str],
        controller: IsaacFrankaController,
        runtime_gate: IsaacRuntimeGate,
        command_ledger_path: str | Path,
        runtime_observe_timeout_s: float = 1.0,
        runtime_action_timeout_s: float = 10.0,
        runtime_stop_timeout_s: float = 1.0,
        tcp_tolerance_m_rad: float = 1e-3,
        robot_state_tolerance: float = 1e-3,
        confidence_tolerance: float = 0.02,
        bin_speed_tolerance_m_s: float = 0.005,
    ) -> None:
        positive_values = {
            "runtime_observe_timeout_s": runtime_observe_timeout_s,
            "runtime_action_timeout_s": runtime_action_timeout_s,
            "runtime_stop_timeout_s": runtime_stop_timeout_s,
            "tcp_tolerance_m_rad": tcp_tolerance_m_rad,
            "robot_state_tolerance": robot_state_tolerance,
            "confidence_tolerance": confidence_tolerance,
            "bin_speed_tolerance_m_s": bin_speed_tolerance_m_s,
        }
        for name, value in positive_values.items():
            if not isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be positive and finite")

        self._observation_source = observation_source
        self._state_guard_source = state_guard_source
        self._control_lease_source = control_lease_source
        self._controller = controller
        self._runtime_gate = runtime_gate
        self._runtime_observe_timeout_s = float(runtime_observe_timeout_s)
        self._runtime_action_timeout_s = float(runtime_action_timeout_s)
        self._runtime_stop_timeout_s = float(runtime_stop_timeout_s)
        self._tcp_tolerance = float(tcp_tolerance_m_rad)
        self._robot_state_tolerance = float(robot_state_tolerance)
        self._confidence_tolerance = float(confidence_tolerance)
        self._bin_speed_tolerance = float(bin_speed_tolerance_m_s)
        self._state_lock = Lock()
        self._step_lock = Lock()
        self._stop_requested = Event()
        self._stop_epoch = 0
        self._last_observation: dict[str, Any] | None = None
        self._observation_generation = 0
        self._seen_observation_ids: set[str] = set()
        self._last_timestamp_ms: int | None = None
        self._command_ledger = DurableCommandIdLedger(command_ledger_path)
        self._startup_stop_receipt: SafeStopReceipt | None = None

        unresolved = self._command_ledger.unresolved_ids
        if unresolved:
            self._startup_stop_receipt = self.safe_stop(
                "unresolved durable Isaac commands at startup: " + ", ".join(unresolved)
            )

    @property
    def startup_stop_receipt(self) -> SafeStopReceipt | None:
        return self._startup_stop_receipt

    @staticmethod
    def _copy_observation(
        value: Mapping[str, Any],
        *,
        context: str,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise RuntimeError(f"{context} did not return an observation object")
        # The source may reuse and mutate nested telemetry buffers after this
        # call. Publish an owned snapshot so the execution digest cannot drift
        # without a new observation generation.
        observation = deepcopy(dict(value))
        observation_id = observation.get("observation_id")
        if not isinstance(observation_id, str) or not observation_id:
            raise RuntimeError(f"{context} returned no observation_id")
        timestamp_ms = observation.get("timestamp_ms")
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms < 0
        ):
            raise RuntimeError(
                f"{context} returned an invalid non-negative timestamp_ms"
            )
        return observation

    def _publish_observation(
        self,
        observation: dict[str, Any],
        *,
        expected_previous_id: str | None = None,
        expected_generation: int | None = None,
        required_stop_epoch: int | None = None,
    ) -> dict[str, Any]:
        observation_id = str(observation["observation_id"])
        timestamp_ms = int(observation["timestamp_ms"])
        with self._state_lock:
            if required_stop_epoch is not None and (
                self._stop_requested.is_set() or required_stop_epoch != self._stop_epoch
            ):
                raise RuntimeError(
                    "late observation discarded after control lease revocation"
                )
            if (
                expected_generation is not None
                and expected_generation != self._observation_generation
            ):
                raise RuntimeError(
                    "post-action observation rejected because a newer observation "
                    "was already published"
                )
            if (
                expected_previous_id is not None
                and observation_id == expected_previous_id
            ):
                raise RuntimeError(
                    "post-action observation_id must differ from the "
                    "pre-action observation_id"
                )
            if observation_id in self._seen_observation_ids:
                raise RuntimeError(
                    f"Isaac observation_id is not fresh: {observation_id!r}"
                )
            if (
                self._last_timestamp_ms is not None
                and timestamp_ms < self._last_timestamp_ms
            ):
                raise RuntimeError("Isaac observation timestamp_ms moved backwards")
            self._seen_observation_ids.add(observation_id)
            self._last_timestamp_ms = timestamp_ms
            self._last_observation = observation
            self._observation_generation += 1
        return observation

    def _signal_stop_only(self, reason: str) -> str:
        with self._state_lock:
            if not self._stop_requested.is_set():
                self._stop_epoch += 1
                self._stop_requested.set()
        return self._controller.request_stop(reason)

    def observe(self) -> Mapping[str, Any]:
        def capture_and_publish() -> dict[str, Any]:
            observation = self._copy_observation(
                self._observation_source(),
                context="Isaac observation source",
            )
            return self._publish_observation(observation)

        return self._runtime_gate.call(
            capture_and_publish,
            timeout_s=self._runtime_observe_timeout_s,
            label="Isaac observe",
            on_started_timeout=lambda: self._signal_stop_only(
                "Isaac observation gate timeout"
            ),
        )

    @staticmethod
    def _require_safe_observation(observation: Mapping[str, Any]) -> None:
        safety = observation.get("safety")
        if not isinstance(safety, Mapping):
            raise _UnsafeGuardChange("Isaac observation contains no safety state")
        if safety.get("emergency_stop") is True:
            raise _UnsafeGuardChange(
                "controller boundary rejected active emergency stop"
            )
        if safety.get("protective_stop") is True:
            raise _UnsafeGuardChange(
                "controller boundary rejected active protective stop"
            )
        if safety.get("system_fault") not in (None, "", False, "NONE", "none"):
            raise _UnsafeGuardChange(
                "controller boundary rejected system fault: "
                f"{safety.get('system_fault')!r}"
            )

    @staticmethod
    def _require_retreat_interlock(
        observation: Mapping[str, Any],
        arm_id: str,
    ) -> None:
        robot = observation.get("robot")
        if not isinstance(robot, Mapping):
            raise _UnsafeGuardChange("Isaac observation contains no robot state")
        other_key = "arm_b" if arm_id == "Arm_A" else "arm_a"
        other = robot.get(other_key)
        if not isinstance(other, Mapping) or other.get("retreated") is not True:
            other_arm = "Arm_B" if arm_id == "Arm_A" else "Arm_A"
            raise _UnsafeGuardChange(
                f"{arm_id} boundary interlock requires {other_arm} retreated"
            )

    def _require_current_lease(
        self,
        observation: Mapping[str, Any],
        *,
        arm_id: str,
        requested_token: str,
    ) -> None:
        current_token = self._control_lease_source()
        if current_token != requested_token:
            raise _UnsafeGuardChange(
                "controller boundary rejected stale control lease: "
                f"current={current_token!r}, requested={requested_token!r}"
            )
        robot = observation.get("robot")
        active_arm = robot.get("active_arm") if isinstance(robot, Mapping) else None
        if active_arm not in {arm_id, "NONE"}:
            raise _UnsafeGuardChange(
                f"{arm_id} command conflicts with active_arm={active_arm!r}"
            )

    def _require_live_guard_compatible(
        self,
        expected: Mapping[str, Any],
        current: Mapping[str, Any],
        *,
        arm_id: str,
        requested_token: str,
    ) -> None:
        self._require_safe_observation(current)
        self._require_retreat_interlock(current, arm_id)
        self._require_current_lease(
            current,
            arm_id=arm_id,
            requested_token=requested_token,
        )

        if expected.get("safety") != current.get("safety"):
            raise _UnsafeGuardChange("live safety state changed before execution")

        expected_robot = expected.get("robot")
        current_robot = current.get("robot")
        if not isinstance(expected_robot, Mapping) or not isinstance(
            current_robot, Mapping
        ):
            raise _UnsafeGuardChange("live robot state is incomplete")
        if expected_robot.get("active_arm") != current_robot.get("active_arm"):
            raise _UnsafeGuardChange("live active_arm changed before execution")
        for arm_key in ("arm_a", "arm_b"):
            expected_arm = expected_robot.get(arm_key)
            current_arm = current_robot.get(arm_key)
            if not isinstance(expected_arm, Mapping) or not isinstance(
                current_arm, Mapping
            ):
                raise _UnsafeGuardChange(f"live robot.{arm_key} is incomplete")
            for field in ("retreated", "gripper_open", "stationary"):
                if expected_arm.get(field) != current_arm.get(field):
                    raise _UnsafeGuardChange(
                        f"live robot.{arm_key}.{field} changed before execution"
                    )
            if not _numeric_sequence_close(
                expected_arm.get("tcp_pose_m_rad"),
                current_arm.get("tcp_pose_m_rad"),
                tolerance=self._tcp_tolerance,
            ):
                raise _UnsafeGuardChange(
                    f"live robot.{arm_key}.tcp_pose_m_rad exceeded tolerance"
                )
            if not _numeric_sequence_close(
                expected_arm.get("state"),
                current_arm.get("state"),
                tolerance=self._robot_state_tolerance,
            ):
                raise _UnsafeGuardChange(
                    f"live robot.{arm_key}.state exceeded tolerance"
                )

        expected_task = expected.get("task")
        current_task = current.get("task")
        if not isinstance(expected_task, Mapping) or not isinstance(
            current_task, Mapping
        ):
            raise _StaleGuardChange("live task facts are incomplete")
        if set(expected_task) != set(current_task):
            raise _StaleGuardChange("live task fact fields changed")
        for field in expected_task:
            if field == "bin_speed_m_s":
                if not _numeric_scalar_close(
                    expected_task[field],
                    current_task[field],
                    tolerance=self._bin_speed_tolerance,
                ):
                    raise _StaleGuardChange(
                        "live bin_speed_m_s is invalid or exceeded tolerance"
                    )
            elif expected_task[field] != current_task[field]:
                raise _StaleGuardChange(
                    f"live task fact {field!r} changed before execution"
                )

        if not self._objects_compatible(
            expected.get("objects"),
            current.get("objects"),
        ):
            raise _StaleGuardChange("live object facts changed before execution")

        expected_quality = expected.get("quality")
        current_quality = current.get("quality")
        if not isinstance(expected_quality, Mapping) or not isinstance(
            current_quality, Mapping
        ):
            raise _StaleGuardChange("live quality state is incomplete")
        if set(expected_quality) != set(current_quality):
            raise _StaleGuardChange("live quality fields changed")
        for field in expected_quality:
            if field == "confidence":
                if not _numeric_scalar_close(
                    expected_quality[field],
                    current_quality[field],
                    tolerance=self._confidence_tolerance,
                ):
                    raise _StaleGuardChange(
                        "live quality confidence is invalid or exceeded tolerance"
                    )
            elif expected_quality[field] != current_quality[field]:
                raise _StaleGuardChange(f"live quality field {field!r} changed")

    def _objects_compatible(self, expected: Any, current: Any) -> bool:
        if not isinstance(expected, list) or not isinstance(current, list):
            return False
        if len(expected) != len(current):
            return False
        try:
            expected_by_id = {
                item["object_id"]: item
                for item in expected
                if isinstance(item, Mapping) and isinstance(item.get("object_id"), str)
            }
            current_by_id = {
                item["object_id"]: item
                for item in current
                if isinstance(item, Mapping) and isinstance(item.get("object_id"), str)
            }
        except (KeyError, TypeError):
            return False
        if (
            len(expected_by_id) != len(expected)
            or len(current_by_id) != len(current)
            or set(expected_by_id) != set(current_by_id)
        ):
            return False
        for object_id, expected_item in expected_by_id.items():
            current_item = current_by_id[object_id]
            if set(expected_item) != set(current_item):
                return False
            for field in expected_item:
                if field == "confidence":
                    if not _numeric_scalar_close(
                        expected_item[field],
                        current_item[field],
                        tolerance=self._confidence_tolerance,
                    ):
                        return False
                elif expected_item[field] != current_item[field]:
                    return False
        return True

    @staticmethod
    def _request_digest(
        action: ActionStep,
        *,
        arm_id: str,
        control_token: str,
        expected_observation_id: str,
        expected_state_digest: str,
    ) -> str:
        payload = json.dumps(
            {
                "action": action.to_dict(),
                "arm_id": arm_id,
                "control_token": control_token,
                "expected_observation_id": expected_observation_id,
                "expected_state_digest": expected_state_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return f"sha256:{hashlib.sha256(payload).hexdigest()}"

    def _require_current_snapshot(
        self,
        *,
        expected_observation_id: str,
        expected_generation: int,
        expected_stop_epoch: int,
    ) -> None:
        with self._state_lock:
            actual_id = (
                self._last_observation.get("observation_id")
                if self._last_observation is not None
                else None
            )
            if self._stop_requested.is_set() or self._stop_epoch != expected_stop_epoch:
                raise _UnsafeGuardChange(
                    "controller command rejected after control lease revocation"
                )
            if (
                self._observation_generation != expected_generation
                or actual_id != expected_observation_id
            ):
                raise _StaleGuardChange(
                    "controller command rejected because the expected "
                    "observation is no longer latest"
                )

    def step(
        self,
        action: ActionStep,
        *,
        arm_id: str,
        control_token: str,
        command_id: str,
        expected_observation_id: str,
        expected_state_digest: str,
    ) -> Mapping[str, Any]:
        required_token = self._AUTHORIZATION.get(arm_id)
        if required_token is None:
            raise RuntimeError(f"Isaac controller boundary rejected arm_id={arm_id!r}")
        if control_token != required_token:
            raise RuntimeError(
                f"{arm_id} requires token {required_token}, got {control_token!r}"
            )
        if not command_id:
            raise RuntimeError("empty controller command_id rejected")
        if action.has_non_finite():
            raise RuntimeError("Isaac controller rejected non-finite action")
        try:
            FROZEN_MULTI_RATE.control_ticks_for_duration_ms(action.duration_ms)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Isaac controller rejected unaligned action timing: {exc}"
            ) from exc

        request_digest = self._request_digest(
            action,
            arm_id=arm_id,
            control_token=control_token,
            expected_observation_id=expected_observation_id,
            expected_state_digest=expected_state_digest,
        )
        acknowledged = self._command_ledger.acknowledged_result(
            command_id,
            request_digest,
        )
        if acknowledged is not None:
            return acknowledged

        if not self._step_lock.acquire(blocking=False):
            raise RuntimeError(
                "Isaac controller boundary rejected overlapping step request"
            )
        try:
            with self._state_lock:
                if self._stop_requested.is_set():
                    raise RuntimeError(
                        "Isaac controller boundary is quarantined after safe-stop"
                    )
                if self._last_observation is None:
                    raise RuntimeError("observe() is required before Isaac execution")
                cached_observation = deepcopy(self._last_observation)
                generation = self._observation_generation
                stop_epoch = self._stop_epoch

            actual_observation_id = cached_observation.get("observation_id")
            if expected_observation_id != actual_observation_id:
                raise RuntimeError(
                    "stale action rejected: "
                    f"expected observation {actual_observation_id!r}, "
                    f"got {expected_observation_id!r}"
                )
            if expected_state_digest != execution_guard_digest(cached_observation):
                raise RuntimeError(
                    "stale action rejected: execution state digest changed"
                )
            try:
                self._require_safe_observation(cached_observation)
                self._require_retreat_interlock(cached_observation, arm_id)
            except _UnsafeGuardChange:
                self.safe_stop(
                    f"unsafe cached observation rejected command {command_id}"
                )
                raise

            def compare_and_execute() -> dict[str, Any]:
                self._require_current_snapshot(
                    expected_observation_id=expected_observation_id,
                    expected_generation=generation,
                    expected_stop_epoch=stop_epoch,
                )
                first_guard = self._state_guard_source()
                if not isinstance(first_guard, Mapping):
                    raise RuntimeError("Isaac live state guard must be an object")
                try:
                    self._require_live_guard_compatible(
                        cached_observation,
                        first_guard,
                        arm_id=arm_id,
                        requested_token=control_token,
                    )
                except _StaleGuardChange as exc:
                    raise PreWriteStateStaleError(str(exc)) from exc
                self._controller.validate_ready(arm_id)

                self._command_ledger.claim(command_id, request_digest)
                try:
                    # Fsync may take long enough for safety, lease or telemetry
                    # to change. Re-read every guarded source after the claim.
                    self._require_current_snapshot(
                        expected_observation_id=expected_observation_id,
                        expected_generation=generation,
                        expected_stop_epoch=stop_epoch,
                    )
                    final_guard = self._state_guard_source()
                    if not isinstance(final_guard, Mapping):
                        raise RuntimeError(
                            "Isaac final live state guard must be an object"
                        )
                    try:
                        self._require_live_guard_compatible(
                            cached_observation,
                            final_guard,
                            arm_id=arm_id,
                            requested_token=control_token,
                        )
                    except _StaleGuardChange as exc:
                        self._command_ledger.abort(
                            command_id,
                            request_digest,
                            reason=f"pre-write state changed: {exc}",
                        )
                        raise PreWriteStateStaleError(str(exc)) from exc
                    self._controller.validate_ready(arm_id)
                    self._require_current_snapshot(
                        expected_observation_id=expected_observation_id,
                        expected_generation=generation,
                        expected_stop_epoch=stop_epoch,
                    )
                except PreWriteStateStaleError:
                    raise
                except BaseException as exc:
                    self._command_ledger.abort(
                        command_id,
                        request_digest,
                        reason=f"pre-write validation failed: {exc}",
                    )
                    raise

                self._controller.execute_action(action, arm_id=arm_id)
                self._command_ledger.mark_applied(command_id, request_digest)
                self._require_current_snapshot(
                    expected_observation_id=expected_observation_id,
                    expected_generation=generation,
                    expected_stop_epoch=stop_epoch,
                )
                observation = self._copy_observation(
                    self._observation_source(),
                    context="Isaac post-action observation source",
                )
                published = self._publish_observation(
                    observation,
                    expected_previous_id=expected_observation_id,
                    expected_generation=generation,
                    required_stop_epoch=stop_epoch,
                )
                self._command_ledger.acknowledge(
                    command_id,
                    request_digest,
                    published,
                )
                return published

            try:
                return self._runtime_gate.call(
                    compare_and_execute,
                    timeout_s=self._runtime_action_timeout_s,
                    label=f"Isaac compare-and-execute {command_id}",
                    on_started_timeout=lambda: self._signal_stop_only(
                        f"Isaac command gate timeout for {command_id}"
                    ),
                )
            except BaseException as exc:
                must_stop = isinstance(
                    exc,
                    (_UnsafeGuardChange, IsaacGateTimeoutError),
                ) or self._command_ledger.is_unresolved(command_id)
                if must_stop and not self._stop_requested.is_set():
                    self.safe_stop(
                        f"Isaac command failed or became unsafe: {command_id}"
                    )
                raise
        finally:
            self._step_lock.release()

    @staticmethod
    def _unconfirmed_receipt(stop_epoch: str) -> SafeStopReceipt:
        return SafeStopReceipt(
            controller_ack=False,
            buffers_cleared=False,
            arm_a_stopped=False,
            arm_b_stopped=False,
            stop_epoch=stop_epoch,
        )

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        with self._state_lock:
            if not self._stop_requested.is_set():
                self._stop_epoch += 1
                self._stop_requested.set()
            local_epoch = self._stop_epoch

        controller_epoch: dict[str, str] = {}

        def signal_stop() -> None:
            controller_epoch["value"] = self._controller.request_stop(reason)

        def apply_stop() -> SafeStopReceipt:
            stop_epoch = controller_epoch.get("value")
            if not stop_epoch:
                raise RuntimeError("controller request_stop returned no stop epoch")
            return self._controller.confirm_safe_stop(
                reason,
                stop_epoch=stop_epoch,
            )

        try:
            receipt = self._runtime_gate.call_stop(
                signal_stop=signal_stop,
                apply_stop=apply_stop,
                timeout_s=self._runtime_stop_timeout_s,
                label="Isaac safe-stop confirmation",
            )
        except BaseException:
            return self._unconfirmed_receipt(
                controller_epoch.get("value", f"adapter-stop-{local_epoch}")
            )
        if not isinstance(receipt, SafeStopReceipt) or not receipt.stop_epoch:
            return self._unconfirmed_receipt(
                controller_epoch.get("value", f"adapter-stop-{local_epoch}")
            )
        return receipt
