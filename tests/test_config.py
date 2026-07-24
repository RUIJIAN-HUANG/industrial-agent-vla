from __future__ import annotations

import json
import unittest
from copy import deepcopy
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

    def test_default_config_builds_core(self) -> None:
        agent = IndustrialAgent.from_config(
            [MockExecutor("openvla_oft", 0.01)], self.config
        )
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


if __name__ == "__main__":
    unittest.main()
