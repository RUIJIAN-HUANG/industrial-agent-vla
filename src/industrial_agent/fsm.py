"""Explicit finite-state machine for the supervisor."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from time import time_ns


class AgentState(str, Enum):
    IDLE = "IDLE"
    VALIDATING_TASK = "VALIDATING_TASK"
    PLANNING = "PLANNING"
    OBSERVING = "OBSERVING"
    PERCEIVING = "PERCEIVING"
    ASSIGNING_ROLE = "ASSIGNING_ROLE"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    ADVANCING_SUBTASK = "ADVANCING_SUBTASK"
    REPLANNING = "REPLANNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SAFE_STOPPED = "SAFE_STOPPED"
    SAFE_STOP_FAILED = "SAFE_STOP_FAILED"


TERMINAL_STATES = frozenset(
    {
        AgentState.SUCCEEDED,
        AgentState.FAILED,
        AgentState.SAFE_STOPPED,
        AgentState.SAFE_STOP_FAILED,
    }
)

ALLOWED_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.IDLE: frozenset({AgentState.VALIDATING_TASK}),
    AgentState.VALIDATING_TASK: frozenset({AgentState.PLANNING, AgentState.FAILED}),
    AgentState.PLANNING: frozenset({AgentState.OBSERVING, AgentState.FAILED}),
    AgentState.OBSERVING: frozenset(
        {
            AgentState.PERCEIVING,
            AgentState.ASSIGNING_ROLE,
            AgentState.ADVANCING_SUBTASK,
            AgentState.SUCCEEDED,
            AgentState.SAFE_STOPPED,
            AgentState.FAILED,
        }
    ),
    AgentState.PERCEIVING: frozenset(
        {
            AgentState.OBSERVING,
            AgentState.ASSIGNING_ROLE,
            AgentState.FAILED,
            AgentState.SAFE_STOPPED,
        }
    ),
    AgentState.ASSIGNING_ROLE: frozenset(
        {
            AgentState.EXECUTING,
            AgentState.FAILED,
            AgentState.SAFE_STOPPED,
        }
    ),
    AgentState.EXECUTING: frozenset(
        {
            AgentState.VERIFYING,
            AgentState.REPLANNING,
            AgentState.FAILED,
            AgentState.SAFE_STOPPED,
        }
    ),
    AgentState.VERIFYING: frozenset(
        {
            AgentState.OBSERVING,
            AgentState.SUCCEEDED,
            AgentState.ADVANCING_SUBTASK,
            AgentState.REPLANNING,
            AgentState.FAILED,
            AgentState.SAFE_STOPPED,
        }
    ),
    AgentState.ADVANCING_SUBTASK: frozenset(
        {AgentState.OBSERVING, AgentState.FAILED, AgentState.SAFE_STOPPED}
    ),
    AgentState.REPLANNING: frozenset(
        {AgentState.OBSERVING, AgentState.SAFE_STOPPED, AgentState.FAILED}
    ),
    AgentState.SUCCEEDED: frozenset(),
    AgentState.FAILED: frozenset(),
    AgentState.SAFE_STOPPED: frozenset(),
    AgentState.SAFE_STOP_FAILED: frozenset(),
}


@dataclass(frozen=True)
class StateTransition:
    previous: AgentState
    current: AgentState
    reason: str
    timestamp_ms: int


@dataclass
class AgentFSM:
    state: AgentState = AgentState.IDLE
    history: list[StateTransition] = field(default_factory=list)

    def transition(self, target: AgentState, reason: str) -> StateTransition:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(
                f"illegal FSM transition {self.state.value} -> {target.value}"
            )
        record = StateTransition(
            previous=self.state,
            current=target,
            reason=reason,
            timestamp_ms=time_ns() // 1_000_000,
        )
        self.state = target
        self.history.append(record)
        return record

    def force_safety_terminal(
        self,
        target: AgentState,
        reason: str,
    ) -> StateTransition:
        """Record a fail-closed emergency terminal from any current state."""

        if target not in {AgentState.SAFE_STOPPED, AgentState.SAFE_STOP_FAILED}:
            raise ValueError("force_safety_terminal only accepts safety terminals")
        record = StateTransition(
            previous=self.state,
            current=target,
            reason=reason,
            timestamp_ms=time_ns() // 1_000_000,
        )
        self.state = target
        self.history.append(record)
        return record
