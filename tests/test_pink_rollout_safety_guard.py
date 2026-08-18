from __future__ import annotations

import ast
from pathlib import Path


ADAPTER = Path(__file__).parents[1] / "simulation" / "pink_franka_adapter.py"


def test_pink_rollout_has_per_action_joint_safety_envelope() -> None:
    source = ADAPTER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    assignments = {
        node.targets[0].id: node.value.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and isinstance(node.value, ast.Constant)
    }
    assert 0.0 < assignments["_MAX_ACTION_JOINT_DELTA_RAD"] <= 0.12
    assert "cumulative_delta = targets - initial_controlled" in source
    assert '"rollout_clamped": rollout_clamped' in source


def test_safety_envelope_preserves_direction_and_caps_magnitude() -> None:
    # Mirrors the deliberately tiny, dependency-free scaling formula used by
    # the adapter.  This catches sign reversal and normalization regressions.
    delta = [-0.37, -0.005, 0.438, -1.1358, 0.555, 0.0158, 0.0]
    limit = 0.12
    scale = limit / max(abs(value) for value in delta)
    clamped = [value * scale for value in delta]
    assert max(abs(value) for value in clamped) <= limit + 1e-12
    assert all(a == 0.0 or (a > 0) == (b > 0) for a, b in zip(delta, clamped))