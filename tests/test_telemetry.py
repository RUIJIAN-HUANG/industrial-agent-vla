from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from industrial_agent.fsm import AgentState
from industrial_agent.telemetry import EventSink


class DurableEventSinkTests(unittest.TestCase):
    def test_failed_fsync_leaves_no_event_and_does_not_consume_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            jsonl_path = Path(temporary_directory) / "events.jsonl"
            sink = EventSink(jsonl_path)

            with patch.object(os, "fsync", side_effect=OSError("disk unavailable")):
                with self.assertRaises(OSError):
                    sink.emit(
                        run_id="run-1",
                        task_id="task-1",
                        event_type="handoff.ready",
                        state=AgentState.VERIFYING,
                        payload={"grants_b_only": True},
                    )

            self.assertEqual(sink.events, [])
            self.assertEqual(sink.events_for_run("run-1"), ())
            self.assertEqual(jsonl_path.read_bytes(), b"")

            event = sink.emit(
                run_id="run-1",
                task_id="task-1",
                event_type="run.safe_stopped",
                state=AgentState.SAFE_STOPPED,
                payload={"grants_b_only": False},
            )
            self.assertEqual(event.sequence, 1)
            persisted = json.loads(jsonl_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["sequence"], 1)
            self.assertEqual(persisted["event_type"], "run.safe_stopped")


if __name__ == "__main__":
    unittest.main()
