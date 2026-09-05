"""Immutable result contract returned by the three-agent Supervisor runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .errors import FailureCode
from .fsm import AgentState, StateTransition
from .telemetry import EventRecord
from .verifier import VerificationResult


@dataclass(frozen=True)
class RunResult:
    """Auditable final report for one Supervisor-controlled task run."""

    run_id: str
    task_id: str
    state: AgentState
    success: bool
    failure_code: FailureCode
    message: str
    executor_history: tuple[str, ...]
    control_token_history: tuple[str, ...]
    replan_counts: dict[str, int]
    transitions: tuple[StateTransition, ...]
    verification: VerificationResult | None
    task_plan: dict[str, Any]
    events: tuple[EventRecord, ...]
