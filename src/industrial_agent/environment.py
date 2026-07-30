"""Robot/simulator boundary."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import ActionStep
from .errors import AgentError, FailureCode


class PreWriteStateStaleError(AgentError):
    """A controller command was durably rejected before any hardware write.

    This is the only execution-boundary failure that authorizes the Supervisor
    to discard the old chunk and ask the same VLA for one bounded replan. Any
    timeout, write attempt, unknown journal state, robot/safety change, or
    generic adapter exception remains outcome-unknown and must fail closed.
    """

    retryable = True
    hardware_write_attempted = False

    def __init__(self, message: str) -> None:
        super().__init__(FailureCode.OBSERVATION_INVALID, message)


def execution_guard_digest(observation_data: Mapping[str, Any]) -> str:
    """Hash all state that can invalidate a VLA action before compare-and-execute."""

    guarded = {
        key: observation_data.get(key)
        for key in ("robot", "safety", "task", "objects", "quality")
    }
    payload = json.dumps(
        guarded,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


@dataclass(frozen=True)
class SafeStopReceipt:
    """Controller acknowledgement required before claiming physical stop."""

    controller_ack: bool
    buffers_cleared: bool
    arm_a_stopped: bool
    arm_b_stopped: bool
    stop_epoch: str

    @property
    def confirmed(self) -> bool:
        return (
            self.controller_ack
            and self.buffers_cleared
            and self.arm_a_stopped
            and self.arm_b_stopped
            and bool(self.stop_epoch)
        )


@runtime_checkable
class ExecutionEnvironment(Protocol):
    def observe(self) -> Mapping[str, Any]:
        """Return a raw online observation for allowlist ingestion."""

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
        """Atomically execute one action for the authorized arm/token.

        The real dual-arm adapter must reject a mismatched arm/token, a stale
        observation id/state digest, a duplicate command id, or a violated
        authoritative current control lease, opposite-arm retreat interlock,
        or stop epoch at the controller boundary before writing any command.
        The adapter must durably journal command states and preserve
        exactly-once acknowledgement across process restarts.

        Raise :class:`PreWriteStateStaleError` only when task/object/quality
        facts changed and the adapter has durably established that no hardware
        write was attempted. All other failures are outcome-unknown.
        """

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        """Stop both arms and return a controller-backed confirmation receipt."""
