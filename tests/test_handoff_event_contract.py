from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError


ROOT = Path(__file__).resolve().parents[1]


class HandoffEventSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        schema = json.loads(
            (ROOT / "schemas" / "event.schema.json").read_text(encoding="utf-8")
        )
        cls.validator = Draft202012Validator(schema)

    @staticmethod
    def event(event_type: str, payload: dict[str, object]) -> dict[str, object]:
        return {
            "schema_version": "1.0",
            "event_id": f"event:{event_type}",
            "sequence": 1,
            "timestamp_ms": 1,
            "run_id": "run-1",
            "task_id": "task-1",
            "event_type": event_type,
            "state": "VERIFYING",
            "payload": payload,
        }

    def test_canonical_handoff_events_enforce_grant_semantics(self) -> None:
        candidate = self.event(
            "handoff.candidate_checked",
            {"grants_b_only": False, "control_token": "A_ONLY"},
        )
        verified = self.event(
            "handoff.verified",
            {
                "quorum_passed": True,
                "grants_b_only": False,
                "control_token": "HANDOFF_VERIFY",
            },
        )
        ready = self.event(
            "handoff.ready",
            {
                "grants_b_only": True,
                "durable_ack": True,
                "control_token": "HANDOFF_VERIFY",
            },
        )

        # This schema validates individual candidate records, not run-level
        # existence. Runtime requires at least one after handoff is reached,
        # and retries may add more; every record remains non-authorizing.
        self.validator.validate(candidate)
        repeated_candidate = deepcopy(candidate)
        repeated_candidate["event_id"] = "event:candidate:2"
        repeated_candidate["sequence"] = 2
        self.validator.validate(repeated_candidate)
        self.validator.validate(verified)
        self.validator.validate(ready)

        invalid_verified = deepcopy(verified)
        invalid_verified["payload"]["grants_b_only"] = True
        with self.assertRaises(ValidationError):
            self.validator.validate(invalid_verified)

        for invalid_ready_payload in (
            {"grants_b_only": True},
            {"grants_b_only": True, "durable_ack": False},
        ):
            with self.subTest(payload=invalid_ready_payload):
                with self.assertRaises(ValidationError):
                    self.validator.validate(
                        self.event("handoff.ready", invalid_ready_payload)
                    )

        invalid_non_ready = self.event(
            "verification.completed",
            {"grants_b_only": True},
        )
        with self.assertRaises(ValidationError):
            self.validator.validate(invalid_non_ready)

    def test_underscore_alias_and_payload_readiness_are_rejected(self) -> None:
        alias = self.event("handoff_ready", {"grants_b_only": True})
        with self.assertRaises(ValidationError):
            self.validator.validate(alias)

        leaked_readiness = self.event(
            "handoff.verified",
            {
                "handoff_ready": True,
                "quorum_passed": True,
                "grants_b_only": False,
            },
        )
        with self.assertRaises(ValidationError):
            self.validator.validate(leaked_readiness)


if __name__ == "__main__":
    unittest.main()
