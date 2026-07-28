from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from industrial_agent.contracts import Postcondition, SubtaskStatus, TaskSchema
from industrial_agent.lifecycle import (
    FROZEN_SUBTASK_EXECUTOR_ASSIGNMENTS,
    FROZEN_TOKEN_SEQUENCE,
    FixedDualVLAPlanner,
    FixedTaskProfile,
)
from industrial_agent.mock import MockExecutor
from industrial_agent.orchestrator import IndustrialAgent
from industrial_agent.perception import MockPerceptionAgent
from industrial_agent.telemetry import EventSink


PERCEPTION_CHECKPOINT_SHA = f"sha256:{'c' * 64}"
CLASS_MAP_SHA = f"sha256:{'d' * 64}"
PERCEPTION_CONFIG_SHA = f"sha256:{'e' * 64}"


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        root = Path(__file__).resolve().parents[1]
        cls.config = json.loads(
            (root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        cls.schema = json.loads(
            (root / "schemas" / "agent-config.schema.json").read_text(encoding="utf-8")
        )

    @staticmethod
    def _executors() -> tuple[MockExecutor, MockExecutor]:
        return (
            MockExecutor("openvla_oft", 0.01),
            MockExecutor("pi05", 0.02),
        )

    def _config_for(
        self,
        executors: tuple[MockExecutor, MockExecutor],
    ) -> dict:
        config = deepcopy(self.config)
        for executor in executors:
            raw = config["executors"][executor.descriptor.name]
            raw["enabled"] = True
            raw["checkpoint_sha"] = executor.descriptor.checkpoint_sha
            raw["norm_stats_sha"] = executor.descriptor.norm_stats_sha
        config["perception"]["checkpoint_sha"] = PERCEPTION_CHECKPOINT_SHA
        config["perception"]["class_map_sha"] = CLASS_MAP_SHA
        config["perception"]["config_sha"] = PERCEPTION_CONFIG_SHA
        return config

    @staticmethod
    def _perception() -> MockPerceptionAgent:
        return MockPerceptionAgent(
            checkpoint_sha=PERCEPTION_CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=PERCEPTION_CONFIG_SHA,
        )

    def test_default_config_matches_json_schema(self) -> None:
        try:
            import jsonschema
        except ImportError:  # pragma: no cover - optional local dependency
            self.skipTest("jsonschema is not installed")
        jsonschema.Draft202012Validator(self.schema).validate(self.config)

    def test_default_config_freezes_shared_image_cas(self) -> None:
        image_cas = self.config["image_cas"]
        self.assertEqual(image_cas["layout"], "sha256-v1")
        self.assertEqual(image_cas["encoding"], "png")
        self.assertEqual(image_cas["digest_scope"], "encoded_bytes")
        self.assertGreaterEqual(image_cas["max_pixels"], 1280 * 720)
        self.assertEqual(image_cas["missing_retry_count"], 1)

    def test_planner_and_config_share_frozen_lifecycle_constants(self) -> None:
        profile = FixedTaskProfile()
        expected_token_sequence = [token.value for token in FROZEN_TOKEN_SEQUENCE]
        self.assertEqual(
            self.config["lifecycle"]["token_sequence"],
            expected_token_sequence,
        )

        task = TaskSchema(
            task_id="task-frozen-plan",
            instruction=profile.arm_a_instruction,
            task_type="pick_place",
            postconditions=(
                Postcondition(
                    kind="field_equals",
                    path="task.bin_at_finished",
                    expected=True,
                ),
            ),
        )
        plan = FixedDualVLAPlanner(profile).plan(task, "episode-frozen-plan")
        self.assertEqual(
            tuple(
                (subtask.subtask_id, subtask.assigned_executor)
                for subtask in plan.subtasks
            ),
            FROZEN_SUBTASK_EXECUTOR_ASSIGNMENTS,
        )
        self.assertTrue(
            all(subtask.status is SubtaskStatus.PENDING for subtask in plan.subtasks)
        )

    def test_frozen_role_instruction_truth_sources_are_aligned(self) -> None:
        root = Path(__file__).resolve().parents[1]
        profile = FixedTaskProfile()
        task_profile = self.config["lifecycle"]["task_profile"]
        schema_profile = self.schema["$defs"]["fixedTaskProfile"]["properties"]
        interface_contract = (
            root / "docs" / "architecture" / "interface-contracts.md"
        ).read_text(encoding="utf-8")
        frozen_flow = (
            root / "docs" / "architecture" / "final-frozen-scene-and-flow.md"
        ).read_text(encoding="utf-8")

        for field_name in ("arm_a_instruction", "arm_b_instruction"):
            expected = getattr(profile, field_name)
            self.assertEqual(task_profile[field_name], expected)
            self.assertEqual(schema_profile[field_name]["const"], expected)
            self.assertIn(expected, interface_contract)
            self.assertIn(expected, frozen_flow)

        self.assertNotIn("帮我把零件最多的区域装箱", interface_contract)
        self.assertNotIn(
            "把中央交接区的同一料箱搬到完成区并摆正",
            interface_contract,
        )

    def test_default_config_builds_fixed_dual_vla_core(self) -> None:
        executors = self._executors()
        agent = IndustrialAgent.from_config(
            executors,
            self._config_for(executors),
            perception=self._perception(),
        )
        self.assertEqual(agent.topology_mode, "FIXED_DUAL_VLA_SERIAL")
        self.assertEqual(agent.verification_frames, 3)
        self.assertEqual(agent.max_decisions_per_strategy_attempt, 32)
        self.assertEqual(agent.safety.policy.max_chunk_steps, 32)
        self.assertEqual(
            tuple(agent.executors._executors),
            ("openvla_oft", "pi05"),
        )
        self.assertEqual(agent.task_profile.primary_executor, "pi05")
        self.assertEqual(
            agent.task_profile.collaborative_executor,
            "openvla_oft",
        )
        self.assertEqual(agent.task_profile.handoff_verification_frames, 3)
        self.assertEqual(agent.task_profile.handoff_required_votes, 2)
        self.assertIn("HOME_A", agent.task_profile.arm_a_instruction)
        self.assertIn("FINISHED_01", agent.task_profile.arm_b_instruction)

    def test_invalid_perception_mode_is_reported_before_mode_policy(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported perception_mode"):
            IndustrialAgent(
                self._executors(),
                perception=self._perception(),
                perception_mode="NOT_A_MODE",
                max_perception_attempts=2,
            )

    def test_fixed_runtime_rejects_non_durable_handoff_sink(self) -> None:
        with self.assertRaisesRegex(ValueError, "fsync-backed EventSink"):
            IndustrialAgent(
                self._executors(),
                perception=self._perception(),
                events=EventSink(),
            )
        with self.assertRaisesRegex(ValueError, "cannot disable"):
            IndustrialAgent(
                self._executors(),
                perception=self._perception(),
                events=EventSink(),
                require_durable_handoff=False,
            )

    def test_legacy_routing_and_switch_fields_are_rejected(self) -> None:
        executors = self._executors()
        config = self._config_for(executors)
        config["routing"] = {"default_executor": "pi05"}
        with self.assertRaisesRegex(ValueError, "routing is obsolete"):
            IndustrialAgent.from_config(
                executors,
                config,
                perception=self._perception(),
            )

        config = self._config_for(executors)
        config["recovery"]["allow_switch_back"] = False
        with self.assertRaisesRegex(ValueError, "frozen"):
            IndustrialAgent.from_config(
                executors,
                config,
                perception=self._perception(),
            )

    def test_config_requires_exact_frozen_lifecycle(self) -> None:
        executors = self._executors()
        mutations = (
            (
                lambda config: config["lifecycle"].update({"supervisor_nlp": True}),
                "supervisor_nlp",
            ),
            (
                lambda config: config["lifecycle"].update(
                    {"token_sequence": ["A_ONLY", "B_ONLY"]}
                ),
                "token_sequence",
            ),
            (
                lambda config: config["lifecycle"]["task_profile"].update(
                    {"primary_executor": "openvla_oft"}
                ),
                "cannot be changed",
            ),
            (
                lambda config: config["lifecycle"]["task_profile"].update(
                    {"handoff_required_votes": 3}
                ),
                "cannot be changed",
            ),
            (
                lambda config: config["lifecycle"]["task_profile"].update(
                    {"arm_a_instruction": "帮我把零件最多的区域装箱"}
                ),
                "cannot be changed",
            ),
            (
                lambda config: config["lifecycle"]["task_profile"].update(
                    {"arm_b_instruction": "临时生成的指令"}
                ),
                "cannot be changed",
            ),
            (
                lambda config: config.update({"verification_frames": 2}),
                "exactly 3 verification frames",
            ),
            (
                lambda config: config["recovery"].update({"max_switches_per_run": 1}),
                "frozen",
            ),
        )
        for mutation, message in mutations:
            config = self._config_for(executors)
            mutation(config)
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    IndustrialAgent.from_config(
                        executors,
                        config,
                        perception=self._perception(),
                    )

    def test_config_requires_both_executors_enabled_and_injected(self) -> None:
        executors = self._executors()
        config = self._config_for(executors)
        config["executors"]["openvla_oft"]["enabled"] = False
        with self.assertRaisesRegex(ValueError, "both executors enabled"):
            IndustrialAgent.from_config(
                (executors[1],),
                config,
                perception=self._perception(),
            )

        with self.assertRaisesRegex(ValueError, "enabled executor set mismatch"):
            IndustrialAgent.from_config(
                (executors[0],),
                self._config_for(executors),
                perception=self._perception(),
            )

    def test_config_accepts_exact_executor_artifact_identity(self) -> None:
        executors = self._executors()
        agent = IndustrialAgent.from_config(
            executors,
            self._config_for(executors),
            perception=self._perception(),
        )
        self.assertTrue(agent.perception_required)
        self.assertEqual(agent.perception_timeout_ms, 5000)
        self.assertEqual(agent.max_perception_attempts, 1)

    def test_config_rejects_executor_artifact_mismatch(self) -> None:
        executors = self._executors()
        for name in ("openvla_oft", "pi05"):
            for field in ("checkpoint_sha", "norm_stats_sha"):
                config = self._config_for(executors)
                config["executors"][name][field] = f"sha256:{'f' * 64}"
                with self.subTest(executor=name, field=field):
                    with self.assertRaisesRegex(ValueError, field):
                        IndustrialAgent.from_config(
                            executors,
                            config,
                            perception=self._perception(),
                        )

    def test_config_rejects_action_contract_mismatch(self) -> None:
        executors = list(self._executors())
        executors[0].descriptor = replace(
            executors[0].descriptor,
            action_contract_version="2.0",
        )
        executor_tuple = (executors[0], executors[1])
        with self.assertRaisesRegex(
            ValueError,
            "action_contract_version mismatch",
        ):
            IndustrialAgent.from_config(
                executor_tuple,
                self._config_for(executor_tuple),
                perception=self._perception(),
            )

    def test_config_requires_matching_yolo_agent_identity(self) -> None:
        executors = self._executors()
        config = self._config_for(executors)
        with self.assertRaisesRegex(ValueError, "requires an injected YOLO"):
            IndustrialAgent.from_config(executors, config)

        mismatched = MockPerceptionAgent(
            checkpoint_sha=f"sha256:{'f' * 64}",
            class_map_sha=CLASS_MAP_SHA,
            config_sha=PERCEPTION_CONFIG_SHA,
        )
        with self.assertRaisesRegex(ValueError, "checkpoint_sha mismatch"):
            IndustrialAgent.from_config(
                executors,
                config,
                perception=mismatched,
            )

    def test_placeholder_cannot_bypass_artifact_pinning(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsafe placeholder"):
            IndustrialAgent.from_config(
                self._executors(),
                self.config,
                perception=self._perception(),
            )


if __name__ == "__main__":
    unittest.main()
