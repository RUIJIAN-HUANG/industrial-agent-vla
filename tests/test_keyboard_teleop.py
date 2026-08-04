from math import radians

import pytest

from simulation.keyboard_teleop import KeyboardTeleopMapper
from simulation.run_keyboard_teleop_smoke import (
    _require_action_evidence,
    _result_identity,
)


def test_motion_keys_follow_frozen_7d_rotation_vector_order() -> None:
    mapper = KeyboardTeleopMapper()

    assert mapper.parse("w").action.values == (0.005, 0, 0, 0, 0, 0, 1)
    assert mapper.parse("a").action.values == (0, 0.005, 0, 0, 0, 0, 1)
    assert mapper.parse("q").action.values == (0, 0, 0.005, 0, 0, 0, 1)
    assert mapper.parse("i").action.values == (
        0,
        0,
        0,
        radians(2),
        0,
        0,
        1,
    )
    assert mapper.parse("j").action.values[4] == radians(2)
    assert mapper.parse("u").action.values[5] == radians(2)
    assert mapper.parse("w").action.duration_ms == 100


def test_gripper_toggle_is_endpoint_not_delta() -> None:
    mapper = KeyboardTeleopMapper(gripper_open=True)

    assert mapper.parse("g").action.values == (0, 0, 0, 0, 0, 0, 0)
    assert mapper.parse("w").action.values[-1] == 0
    assert mapper.parse("g").action.values[-1] == 1


def test_control_and_unknown_keys() -> None:
    mapper = KeyboardTeleopMapper()

    assert mapper.parse("r").kind == "reset"
    assert mapper.parse("space").kind == "checkpoint"
    assert mapper.parse("x").kind == "quit"
    with pytest.raises(ValueError, match="unknown teleop key"):
        mapper.parse("not-a-key")


def test_smoke_requires_at_least_one_successful_action() -> None:
    with pytest.raises(RuntimeError, match="at least one successful action"):
        _require_action_evidence(0)

    _require_action_evidence(1)


def test_smoke_result_identity_marks_every_outcome_noncanonical() -> None:
    assert _result_identity(
        session_id="smoke-1",
        arm_id="Arm_B",
        input_mode="gui",
    ) == {
        "smoke_only": True,
        "not_canonical_episode": True,
        "session_id": "smoke-1",
        "arm_id": "Arm_B",
        "input_mode": "gui",
    }
