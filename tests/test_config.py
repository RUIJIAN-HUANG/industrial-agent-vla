from __future__ import annotations

import json
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

from industrial_agent.mock import MockExecutor
from industrial_agent.orchestrator import IndustrialAgent


class ConfigTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        config_path = (
            Path(__file__).resolve().parents[1] / "configs" / "agent.default.json"
        )
        cls.config = json.loads(config_path.read_text(encoding="utf-8"))

    def _config_for(self, executor: MockExecutor) -> dict:
        config = deepcopy(self.config)
        raw = config["executors"][executor.descriptor.name]
        raw["checkpoint_sha"] = executor.descriptor.checkpoint_sha
        raw["norm_stats_sha"] = executor.descriptor.norm_stats_sha
        return config

    def test_default_config_builds_core(self) -> None:
        executor = MockExecutor("openvla_oft", 0.01)
        agent = IndustrialAgent.from_config([executor], self._config_for(executor))
        self.assertEqual(agent.verification_frames, 3)
        self.assertEqual(agent.max_decisions_per_strategy_attempt, 8)
        self.assertEqual(agent.safety.policy.max_chunk_steps, 32)

    def test_config_cannot_weaken_frozen_recovery_invariants(self) -> None:
        config = deepcopy(self.config)
        config["recovery"]["allow_switch_back"] = True
        with self.assertRaisesRegex(ValueError, "frozen"):
            IndustrialAgent.from_config([MockExecutor("openvla_oft", 0.01)], config)

    def test_config_cannot_widen_canonical_action_bounds(self) -> None:
        for key, value in (
            ("max_chunk_steps", 33),
            ("axis_abs_limits", [0.05] * 6 + [5.0]),
        ):
            config = deepcopy(self.config)
            config["safety"][key] = value
            with self.subTest(key=key):
                with self.assertRaises(ValueError):
                    IndustrialAgent.from_config(
                        [MockExecutor("openvla_oft", 0.01)], config
                    )

    def test_config_accepts_exact_executor_artifact_identity(self) -> None:
        executor = MockExecutor("openvla_oft", 0.01)
        config = self._config_for(executor)
        agent = IndustrialAgent.from_config([executor], config)
        self.assertEqual(tuple(agent.router._executors), ("openvla_oft",))

    def test_config_rejects_executor_artifact_mismatch(self) -> None:
        executor = MockExecutor("openvla_oft", 0.01)
        for field in ("checkpoint_sha", "norm_stats_sha"):
            config = self._config_for(executor)
            config["executors"]["openvla_oft"][field] = f"sha256:{'f' * 64}"
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, field):
                    IndustrialAgent.from_config([executor], config)

    def test_config_rejects_undeclared_executor(self) -> None:
        with self.assertRaisesRegex(ValueError, "not declared"):
            IndustrialAgent.from_config([MockExecutor("other", 0.01)], self.config)

    def test_placeholder_cannot_be_bypassed_by_mock_task_type(self) -> None:
        executor = MockExecutor("openvla_oft", 0.01)
        with self.assertRaisesRegex(ValueError, "unsafe placeholder"):
            IndustrialAgent.from_config([executor], self.config)

    def test_config_rejects_executor_action_contract_mismatch(self) -> None:
        executor = MockExecutor("openvla_oft", 0.01)
        executor.descriptor = replace(
            executor.descriptor,
            action_contract_version="2.0",
        )
        with self.assertRaisesRegex(ValueError, "action_contract_version mismatch"):
            IndustrialAgent.from_config([executor], self.config)


if __name__ == "__main__":
    unittest.main()
