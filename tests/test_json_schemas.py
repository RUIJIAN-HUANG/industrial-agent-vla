from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
from referencing import Registry, Resource

CHECKPOINT_SHA = f"sha256:{'a' * 64}"
NORM_STATS_SHA = f"sha256:{'b' * 64}"


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
            (self.root / "schemas" / "agent-config-v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        config = json.loads(
            (self.root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        Draft202012Validator(schema).validate(config)

    def test_explicit_v2_config_alias_matches_formal_default(self) -> None:
        default = json.loads(
            (self.root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        explicit = json.loads(
            (self.root / "configs" / "agent.v2.default.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(explicit, default)

    def test_agent_config_rejects_invalid_axis_limits(self) -> None:
        schema = json.loads(
            (self.root / "schemas" / "agent-config-v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        config = json.loads(
            (self.root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        invalid_limits = (
            ["bad", False, None, {}, [], -999, "also-bad"],
            [0.05, 0.05, 0.05, 0.25, 0.25, 0.25, 1.01],
            [0.05, 0.05, 0.05, 0.25, 0.0, 0.25, 1.0],
        )
        validator = Draft202012Validator(schema)
        for limits in invalid_limits:
            invalid = deepcopy(config)
            invalid["safety"]["axis_abs_limits"] = limits
            with self.subTest(limits=limits):
                with self.assertRaises(ValidationError):
                    validator.validate(invalid)

    def test_agent_config_rejects_mutable_artifact_aliases(self) -> None:
        schema = json.loads(
            (self.root / "schemas" / "agent-config-v2.schema.json").read_text(
                encoding="utf-8"
            )
        )
        config = json.loads(
            (self.root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        validator = Draft202012Validator(schema)

        valid = deepcopy(config)
        valid["executors"]["pi05"]["checkpoint_sha"] = CHECKPOINT_SHA
        valid["executors"]["pi05"]["norm_stats_sha"] = NORM_STATS_SHA
        validator.validate(valid)

        for alias in ("latest00", "version1", "sha256:abc", "a" * 40):
            invalid = deepcopy(valid)
            invalid["executors"]["pi05"]["checkpoint_sha"] = alias
            with self.subTest(alias=alias):
                with self.assertRaises(ValidationError):
                    validator.validate(invalid)

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
            "executor": "pi05",
            "checkpoint_sha": CHECKPOINT_SHA,
            "norm_stats_sha": NORM_STATS_SHA,
            "status": "ok",
            "action_chunk": {
                "contract_version": "1.0",
                "chunk_id": "chunk-1",
                "task_id": "task:S01",
                "executor": "pi05",
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
        mutable_alias = deepcopy(response)
        mutable_alias["checkpoint_sha"] = "latest00"
        with self.assertRaises(ValidationError):
            validator.validate(mutable_alias)

    def test_pi05_request_schema_requires_null_wrist_images(self) -> None:
        schema = json.loads(
            (self.root / "schemas" / "executor-infer.schema.json").read_text(
                encoding="utf-8"
            )
        )
        definitions = schema["$defs"]
        self.assertIsNone(
            definitions["pi05ModelInput"]["properties"]["observation"]["properties"][
                "camera"
            ]["properties"]["wrist_image"]["const"]
        )
        pi05_image = definitions["pi05ModelInput"]["properties"]["observation"][
            "properties"
        ]["camera"]["properties"]["full_image"]["allOf"][1]["properties"]
        self.assertEqual(pi05_image["width"]["const"], 1280)
        self.assertEqual(pi05_image["height"]["const"], 720)

        pi05_robot = definitions["pi05ModelInput"]["properties"]["observation"][
            "properties"
        ]["robot"]["properties"]
        state_schema = pi05_robot["state"]
        self.assertEqual(state_schema["minItems"], 7)
        self.assertEqual(state_schema["maxItems"], 7)
        self.assertIn("rotation vector", state_schema["description"])
        self.assertEqual(pi05_robot["tcp_pose_m_rad"]["minItems"], 6)
        self.assertEqual(pi05_robot["tcp_pose_m_rad"]["maxItems"], 6)


if __name__ == "__main__":
    unittest.main()
