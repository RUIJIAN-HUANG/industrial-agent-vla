"""Structured event log and compact run memory."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from threading import Lock
from time import time_ns
from typing import Any, Mapping
from uuid import uuid4

from .errors import FailureCode
from .fsm import AgentState

EVENT_SCHEMA_VERSION = "1.0"


@dataclass(frozen=True)
class EventRecord:
    event_id: str
    sequence: int
    timestamp_ms: int
    run_id: str
    task_id: str
    event_type: str
    state: str
    payload: Mapping[str, Any]
    schema_version: str = EVENT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EventSink:
    """Structured sink with optional fsync-backed JSONL durability."""

    def __init__(self, jsonl_path: str | Path | None = None):
        self.events: list[EventRecord] = []
        self._path = Path(jsonl_path) if jsonl_path is not None else None
        self._sequence_by_run: dict[str, int] = {}
        self._lock = Lock()

    @property
    def durable(self) -> bool:
        """Whether a successful emit is acknowledged only after fsync."""

        return self._path is not None

    def emit(
        self,
        *,
        run_id: str,
        task_id: str,
        event_type: str,
        state: AgentState,
        payload: Mapping[str, Any] | None = None,
    ) -> EventRecord:
        with self._lock:
            sequence = self._sequence_by_run.get(run_id, 0) + 1
            self._sequence_by_run[run_id] = sequence
            event = EventRecord(
                event_id=str(uuid4()),
                sequence=sequence,
                timestamp_ms=time_ns() // 1_000_000,
                run_id=run_id,
                task_id=task_id,
                event_type=event_type,
                state=state.value,
                payload=dict(payload or {}),
            )
            self.events.append(event)
            if self._path is not None:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                with self._path.open("a", encoding="utf-8") as handle:
                    handle.write(
                        json.dumps(
                            event.to_dict(),
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        + "\n"
                    )
                    handle.flush()
                    os.fsync(handle.fileno())
        return event

    def events_for_run(self, run_id: str) -> tuple[EventRecord, ...]:
        """Return an atomic snapshot containing only one run's evidence."""

        with self._lock:
            return tuple(event for event in self.events if event.run_id == run_id)


@dataclass
class RunMemory:
    """Small deterministic memory; no free-form hidden reasoning is stored."""

    run_id: str
    task_id: str
    plan_id: str | None = None
    active_subtask_id: str | None = None
    active_executor: str | None = None
    executor_history: list[str] = field(default_factory=list)
    replan_counts: dict[str, int] = field(default_factory=dict)
    last_failure_code: str = FailureCode.NONE.value
    last_observation_id: str | None = None
    completed_chunk_ids: list[str] = field(default_factory=list)
    control_token: str | None = None
    control_token_history: list[str] = field(default_factory=list)

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "task_id": self.task_id,
            "plan_id": self.plan_id,
            "active_subtask_id": self.active_subtask_id,
            "active_executor": self.active_executor,
            "executor_history": list(self.executor_history),
            "replan_counts": dict(self.replan_counts),
            "last_failure_code": self.last_failure_code,
            "last_observation_id": self.last_observation_id,
            "completed_chunk_ids": list(self.completed_chunk_ids),
            "control_token": self.control_token,
            "control_token_history": list(self.control_token_history),
        }


class MemoryStore:
    def __init__(self):
        self._runs: dict[str, RunMemory] = {}

    def create(self, run_id: str, task_id: str) -> RunMemory:
        memory = RunMemory(run_id=run_id, task_id=task_id)
        self._runs[run_id] = memory
        return memory

    def get(self, run_id: str) -> RunMemory:
        return self._runs[run_id]
