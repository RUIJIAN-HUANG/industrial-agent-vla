from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from industrial_agent.contracts import (
    ACTION_CONTRACT_VERSION,
    MAX_ACTION_CHUNK_STEPS,
    PI05_EXECUTOR_NAME,
    ActionChunk,
    ActionStep,
    Postcondition,
    TaskSchema,
)
from industrial_agent.errors import ContractError, FailureCode
from industrial_agent.executor import EXECUTOR_HEALTH_RESPONSE_FIELDS


class ContractAlignmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]

    @staticmethod
    def _condition() -> Postcondition:
        return Postcondition(
            kind="field_equals",
            path="task.done",
            expected=True,
        )

    def _task(self, **overrides: object) -> TaskSchema:
        values: dict[str, object] = {
            "task_id": "task-1",
            "instruction": "move the red box",
            "task_type": "pick_place",
            "postconditions": (self._condition(),),
        }
        values.update(overrides)
        return TaskSchema(**values)  # type: ignore[arg-type]

    def test_health_schema_requires_supported_task_types(self) -> None:
        schema = json.loads(
            (self.root / "schemas" / "executor-health.schema.json").read_text(
                encoding="utf-8"
            )
        )
        response = {
            "schema_version": "1.0",
            "service": "pi05",
            "status": "ready",
            "checkpoint_sha": f"sha256:{'a' * 64}",
            "norm_stats_sha": f"sha256:{'b' * 64}",
            "supported_task_types": ["pick_place"],
            "supported_action_contracts": ["1.0"],
        }
        validator = Draft202012Validator(schema)
        validator.validate(response)
        self.assertEqual(
            frozenset(schema["properties"]),
            EXECUTOR_HEALTH_RESPONSE_FIELDS,
        )

        del response["supported_task_types"]
        with self.assertRaises(ValidationError):
            validator.validate(response)

    def test_task_schema_version_matches_published_pattern(self) -> None:
        for version in ("1.0", "1.17"):
            with self.subTest(version=version):
                self._task(schema_version=version).validate()

        for version in ("1", "1.bad", "1.0.0", "2.0", "", 1, None):
            with self.subTest(version=version):
                with self.assertRaises(ContractError) as caught:
                    self._task(schema_version=version).validate()
                self.assertEqual(
                    caught.exception.code,
                    FailureCode.UNSUPPORTED_TASK_VERSION,
                )

    def test_obsolete_executor_preference_is_rejected_at_contract_boundary(
        self,
    ) -> None:
        raw = self._task().to_dict()
        raw["preferred_executor"] = "retired_executor"
        with self.assertRaises(ContractError) as caught:
            TaskSchema.from_dict(raw)
        self.assertEqual(caught.exception.code, FailureCode.INVALID_TASK)

    def test_numeric_range_bounds_must_be_finite_numbers(self) -> None:
        Postcondition(kind="numeric_range", path="robot.load", minimum=0).validate()
        Postcondition(kind="numeric_range", path="robot.load", maximum=1.5).validate()

        for bound in ("0", True, float("nan"), float("inf"), float("-inf")):
            with self.subTest(bound=bound):
                with self.assertRaises(ContractError) as caught:
                    Postcondition(
                        kind="numeric_range",
                        path="robot.load",
                        minimum=bound,  # type: ignore[arg-type]
                    ).validate()
                self.assertEqual(caught.exception.code, FailureCode.INVALID_TASK)

    def test_vote_and_confidence_types_are_strict(self) -> None:
        base = {
            "kind": "field_equals",
            "path": "task.done",
            "expected": True,
        }
        for field, value in (
            ("required_votes", True),
            ("required_votes", "2"),
            ("required_votes", 1.5),
            ("min_confidence", True),
            ("min_confidence", "0.6"),
            ("min_confidence", float("nan")),
            ("min_confidence", float("inf")),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaises(ContractError) as caught:
                    Postcondition.from_dict({**base, field: value})
                self.assertEqual(caught.exception.code, FailureCode.INVALID_TASK)

    def test_action_chunk_python_contract_matches_frozen_schema_limits(self) -> None:
        schema = json.loads(
            (self.root / "schemas" / "action-chunk.schema.json").read_text(
                encoding="utf-8"
            )
        )
        properties = schema["properties"]
        self.assertEqual(properties["executor"]["const"], PI05_EXECUTOR_NAME)
        self.assertEqual(
            properties["steps"]["maxItems"],
            MAX_ACTION_CHUNK_STEPS,
        )

        step = ActionStep.from_sequence([0, 0, 0, 0, 0, 0, 0])
        valid = ActionChunk(
            contract_version=ACTION_CONTRACT_VERSION,
            chunk_id="chunk-32",
            task_id="task-1",
            executor=PI05_EXECUTOR_NAME,
            steps=(step,) * MAX_ACTION_CHUNK_STEPS,
        )
        valid.validate_contract()

        invalid_executor = ActionChunk(
            contract_version=ACTION_CONTRACT_VERSION,
            chunk_id="chunk-third-vla",
            task_id="task-1",
            executor="rt2",
            steps=(step,),
        )
        with self.assertRaises(ContractError) as caught:
            invalid_executor.validate_contract()
        self.assertEqual(
            caught.exception.code,
            FailureCode.ACTION_CONTRACT_INVALID,
        )

        oversized = ActionChunk(
            contract_version=ACTION_CONTRACT_VERSION,
            chunk_id="chunk-33",
            task_id="task-1",
            executor=PI05_EXECUTOR_NAME,
            steps=(step,) * (MAX_ACTION_CHUNK_STEPS + 1),
        )
        with self.assertRaises(ContractError) as caught:
            oversized.validate_contract()
        self.assertEqual(
            caught.exception.code,
            FailureCode.ACTION_CONTRACT_INVALID,
        )


if __name__ == "__main__":
    unittest.main()
