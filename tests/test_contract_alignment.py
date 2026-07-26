from __future__ import annotations

import json
import unittest
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError

from industrial_agent.contracts import Postcondition, TaskSchema
from industrial_agent.errors import ContractError, FailureCode


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
            "service": "openvla_oft",
            "status": "ready",
            "checkpoint_sha": f"sha256:{'a' * 64}",
            "norm_stats_sha": f"sha256:{'b' * 64}",
            "supported_task_types": ["pick_place"],
            "supported_action_contracts": ["1.0"],
        }
        validator = Draft202012Validator(schema)
        validator.validate(response)

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
        raw["preferred_executor"] = "openvla_oft"
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


if __name__ == "__main__":
    unittest.main()
