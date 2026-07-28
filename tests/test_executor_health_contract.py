from __future__ import annotations

import unittest

from industrial_agent.errors import ExecutorError, FailureCode
from industrial_agent.executor import (
    PI05_TASK_TYPES,
    ExecutorDescriptor,
    _validate_health_response,
)


CHECKPOINT_SHA = f"sha256:{'1' * 64}"
NORM_STATS_SHA = f"sha256:{'2' * 64}"


class ExecutorHealthContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.descriptor = ExecutorDescriptor(
            name="pi05",
            task_types=PI05_TASK_TYPES,
            action_contract_version="1.0",
            checkpoint_sha=CHECKPOINT_SHA,
            norm_stats_sha=NORM_STATS_SHA,
        )

    @classmethod
    def response(cls, **overrides: object) -> dict[str, object]:
        result: dict[str, object] = {
            "schema_version": "1.0",
            "service": "pi05",
            "status": "ready",
            "checkpoint_sha": CHECKPOINT_SHA,
            "norm_stats_sha": NORM_STATS_SHA,
            "supported_task_types": sorted(PI05_TASK_TYPES),
            "supported_action_contracts": ["1.0"],
        }
        result.update(overrides)
        return result

    def assert_bad_response(self, field: str, value: object) -> None:
        with self.subTest(field=field, value=value):
            with self.assertRaises(ExecutorError) as caught:
                _validate_health_response(
                    self.response(**{field: value}),
                    self.descriptor,
                )
            self.assertEqual(
                caught.exception.code,
                FailureCode.EXECUTOR_BAD_RESPONSE,
            )

    def test_valid_optional_health_fields_match_json_schema(self) -> None:
        _validate_health_response(
            self.response(
                service_version="2026.7.28",
                uptime_ms=0,
                queue={"depth": 0},
                device={"kind": "cuda", "index": 0},
                time_ms=1,
            ),
            self.descriptor,
        )

    def test_optional_health_field_types_and_ranges_fail_closed(self) -> None:
        for value in (None, 1, False, {}):
            self.assert_bad_response("service_version", value)
        for field in ("uptime_ms", "time_ms"):
            for value in (-1, 1.5, True, "1", None):
                self.assert_bad_response(field, value)
        for field in ("queue", "device"):
            for value in (None, [], "cuda", 1, False):
                self.assert_bad_response(field, value)


if __name__ == "__main__":
    unittest.main()
