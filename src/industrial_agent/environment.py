"""Robot/simulator boundary."""

from __future__ import annotations

from typing import Any, Mapping, Protocol, runtime_checkable

from .contracts import ActionStep


@runtime_checkable
class ExecutionEnvironment(Protocol):
    def observe(self) -> Mapping[str, Any]:
        """Return a raw online observation for allowlist ingestion."""

    def step(self, action: ActionStep) -> Mapping[str, Any]:
        """Execute exactly one already-safety-checked physical action."""

    def safe_stop(self, reason: str) -> None:
        """Stop motion, clear controller buffers, and hold a safe state."""
