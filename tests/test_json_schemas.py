from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource


class JsonSchemaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = Path(__file__).resolve().parents[1]

    def test_all_schemas_are_valid_draft_2020_12(self) -> None:
        schema_paths = sorted((self.root / "schemas").glob("*.schema.json"))
        self.assertGreater(len(schema_paths), 0)
        for path in schema_paths:
            with self.subTest(schema=path.name):
                schema = json.loads(path.read_text(encoding="utf-8"))
                Draft202012Validator.check_schema(schema)

    def test_default_config_matches_agent_config_schema(self) -> None:
        schema = json.loads(
            (self.root / "schemas" / "agent-config.schema.json").read_text(
                encoding="utf-8"
            )
        )
        config = json.loads(
            (self.root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        Draft202012Validator(schema).validate(config)

    def test_executor_response_resolves_action_chunk_reference(self) -> None:
        loaded = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted((self.root / "schemas").glob("*.schema.json"))
        ]
        registry = Registry().with_resources(
            (schema["$id"], Resource.from_contents(schema)) for schema in loaded
        )
        infer_schema = next(
            schema
            for schema in loaded
            if schema["$id"].endswith("/executor-infer.schema.json")
        )
        validator = Draft202012Validator(infer_schema, registry=registry).evolve(
            schema=infer_schema["$defs"]["response"]
        )
        response = {
            "schema_version": "1.0",
            "request_id": "req-1",
            "trace_id": "run-1",
            "episode_id": "run-1",
            "task_id": "task:S01",
            "subtask_id": "S01",
            "step_id": 0,
            "observation_id": "obs-1",
            "executor": "openvla_oft",
            "checkpoint_sha": "sha256:checkpoint",
            "norm_stats_sha": "sha256:norm",
            "status": "ok",
            "action_chunk": {
                "contract_version": "1.0",
                "chunk_id": "chunk-1",
                "task_id": "task:S01",
                "executor": "openvla_oft",
                "action_space": "ee_delta_pose_gripper",
                "frame": "robot_base",
                "translation_unit": "m",
                "rotation_unit": "rad",
                "gripper_unit": "normalized",
                "steps": [
                    {
                        "values": [0.01, 0, 0, 0, 0, 0, 0.5],
                        "duration_ms": 100,
                    }
                ],
            },
        }
        validator.validate(response)
        missing_chunk = deepcopy(response)
        del missing_chunk["action_chunk"]
        with self.assertRaises(ValidationError):
            validator.validate(missing_chunk)
        invalid_gripper = deepcopy(response)
        invalid_gripper["action_chunk"]["steps"][0]["values"][6] = 2.0
        with self.assertRaises(ValidationError):
            validator.validate(invalid_gripper)


if __name__ == "__main__":
    unittest.main()
