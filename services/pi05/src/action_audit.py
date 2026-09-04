"""Opt-in, read-only π0.5 inference-chain audit logging.

The audit is intentionally outside the action contract.  It records enough
information to correlate an observation with the model output and the
post-processing output without storing image pixels or changing any array.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger("pi05_action_audit")


def _enabled_from_env() -> bool:
    return os.environ.get("PI05_ACTION_AUDIT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def array_sha256(value: Any) -> str:
    """Return a digest of an array's contiguous bytes (never its pixel values)."""

    array = np.ascontiguousarray(np.asarray(value))
    return "sha256:" + hashlib.sha256(array.tobytes()).hexdigest()


def array_payload(value: Any) -> dict[str, Any]:
    """Serialize an action/state array with shape, dtype and finite extrema."""

    array = np.asarray(value)
    payload: dict[str, Any] = {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "sha256": array_sha256(array),
        "values": array.tolist(),
    }
    if array.size:
        numeric = np.asarray(array, dtype=np.float64)
        payload["min"] = float(np.min(numeric))
        payload["max"] = float(np.max(numeric))
        payload["finite"] = bool(np.all(np.isfinite(numeric)))
    else:
        payload["finite"] = True
    return payload


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


class ActionAudit:
    """Process-local JSONL writer, disabled unless explicitly requested."""

    _lock = threading.Lock()

    def __init__(self) -> None:
        self.enabled = _enabled_from_env()
        configured = os.environ.get(
            "PI05_ACTION_AUDIT_PATH", "/tmp/pi05-action-audit.jsonl"
        )
        self.path = Path(configured).expanduser()
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        stage: str,
        *,
        context: Mapping[str, Any] | None = None,
        **payload: Any,
    ) -> None:
        if not self.enabled:
            return
        record: dict[str, Any] = {
            "schema_version": "pi05-action-audit-v1",
            "timestamp_ns": time.time_ns(),
            "pid": os.getpid(),
            "stage": stage,
        }
        if context:
            record.update(_json_safe(dict(context)))
        record.update(_json_safe(payload))
        try:
            line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
            with self._lock:
                with self.path.open("a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except Exception as exc:  # diagnostics must never alter inference
            logger.warning("audit write failed path=%s error=%s", self.path, exc)


def observation_context(obs: Any) -> dict[str, Any]:
    """Extract correlation IDs from an ObsPacket without changing its schema."""

    flags = getattr(obs, "runtime_flags", {}) or {}
    return {
        "request_id": flags.get("request_id", ""),
        "trace_id": flags.get("trace_id", ""),
        "episode_id": getattr(obs, "episode_id", ""),
        "task_id": flags.get("task_id", ""),
        "subtask_id": flags.get("subtask_id", ""),
        "step_id": int(getattr(obs, "step_id", 0)),
        "observation_id": flags.get("observation_id", ""),
        "arm_id": flags.get("arm_id", ""),
    }


__all__ = ["ActionAudit", "array_payload", "array_sha256", "observation_context"]
