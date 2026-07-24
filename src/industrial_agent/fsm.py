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
    SELECTING_EXECUTOR = "SELECTING_EXECUTOR"
    EXECUTING = "EXECUTING"
    VERIFYING = "VERIFYING"
    ADVANCING_SUBTASK = "ADVANCING_SUBTASK"
    REPLANNING = "REPLANNING"
    SWITCHING = "SWITCHING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    SAFE_STOPPED = "SAFE_STOPPED"


TERMINAL_STATES = frozenset(
    {AgentState.SUCCEEDED, AgentState.FAILED, AgentState.SAFE_STOPPED}
)

ALLOWED_TRANSITIONS: dict[AgentState, frozenset[AgentState]] = {
    AgentState.IDLE: frozenset({AgentState.VALIDATING_TASK}),
    AgentState.VALIDATING_TASK: frozenset({AgentState.PLANNING, AgentState.FAILED}),
    AgentState.PLANNING: frozenset({AgentState.OBSERVING, AgentState.FAILED}),
    AgentState.OBSERVING: frozenset(
        {
            AgentState.SELECTING_EXECUTOR,
            AgentState.ADVANCING_SUBTASK,
            AgentState.SUCCEEDED,
            AgentState.SAFE_STOPPED,
            AgentState.FAILED,
        }
    ),
    AgentState.SELECTING_EXECUTOR: frozenset(
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
            AgentState.SWITCHING,
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
            AgentState.SWITCHING,
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
    AgentState.SWITCHING: frozenset(
        {AgentState.OBSERVING, AgentState.SAFE_STOPPED, AgentState.FAILED}
    ),
    AgentState.SUCCEEDED: frozenset(),
    AgentState.FAILED: frozenset(),
    AgentState.SAFE_STOPPED: frozenset(),
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
