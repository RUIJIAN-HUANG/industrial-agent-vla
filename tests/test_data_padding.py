from __future__ import annotations

import numpy as np
import pytest

from industrial_agent.data import (
    PaddingPolicy,
    PaddingStrategy,
    pad_actions,
)


def _actions(length: int = 2) -> np.ndarray:
    values = np.zeros((length, 7), dtype=np.float32)
    values[:, 0] = np.arange(length, dtype=np.float32)
    values[:, 6] = 1.0
    return values


def test_default_policy_preserves_variable_length_actions() -> None:
    result = pad_actions(_actions(2))

    assert result.values.shape == (2, 7)
    assert result.valid_mask.tolist() == [True, True]
    assert result.valid_length == 2
    assert result.strategy is PaddingStrategy.NONE


def test_zero_padding_is_masked_and_never_executable() -> None:
    result = pad_actions(
        _actions(2),
        PaddingPolicy(PaddingStrategy.ZERO_MASKED, target_length=4),
    )

    assert result.values.shape == (4, 7)
    assert result.valid_mask.tolist() == [True, True, False, False]
    assert np.array_equal(result.values[2:], np.zeros((2, 7), dtype=np.float32))
    assert np.array_equal(result.executable_values(), _actions(2))


def test_repeat_last_padding_remains_masked() -> None:
    result = pad_actions(
        _actions(2),
        PaddingPolicy("repeat_last_masked", target_length=3),
    )

    assert result.valid_mask.tolist() == [True, True, False]
    assert np.array_equal(result.values[2], result.values[1])


@pytest.mark.parametrize(
    "actions",
    [
        np.zeros((2, 6), dtype=np.float32),
        np.zeros((0, 7), dtype=np.float32),
        np.full((1, 7), np.nan, dtype=np.float32),
    ],
)
def test_invalid_action_shape_or_values_fail_closed(actions: np.ndarray) -> None:
    with pytest.raises(ValueError):
        pad_actions(actions)


def test_gripper_range_and_truncation_are_rejected() -> None:
    invalid_gripper = _actions(1)
    invalid_gripper[0, 6] = 1.1
    with pytest.raises(ValueError, match="gripper"):
        pad_actions(invalid_gripper)

    with pytest.raises(ValueError, match="truncation is forbidden"):
        pad_actions(
            _actions(2),
            PaddingPolicy(PaddingStrategy.ZERO_MASKED, target_length=1),
        )


def test_policy_mapping_is_strict_and_non_padding_requires_exact_length() -> None:
    policy = PaddingPolicy.from_mapping({"strategy": "none", "target_length": 3})
    with pytest.raises(ValueError, match="requires the action length"):
        pad_actions(_actions(2), policy)

    with pytest.raises(ValueError, match="unknown padding policy fields"):
        PaddingPolicy.from_mapping({"strategy": "none", "horizon": 10})

    with pytest.raises(ValueError, match="requires target_length"):
        PaddingPolicy(PaddingStrategy.ZERO_MASKED)
