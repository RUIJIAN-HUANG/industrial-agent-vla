from __future__ import annotations

import numpy as np
import pytest

from configs.pi05.train_config import (
    CANONICAL_ACTION_DIM,
    MODEL_ACTION_DIM,
    PI05_INDUSTRIAL_CONFIG,
    project_policy_actions,
)


def test_pi05_model_head_stays_base_checkpoint_compatible() -> None:
    assert PI05_INDUSTRIAL_CONFIG.model.action_dim == MODEL_ACTION_DIM == 32
    assert PI05_INDUSTRIAL_CONFIG.model.pi05 is True
    assert PI05_INDUSTRIAL_CONFIG.model.discrete_state_input is True


def test_pi05_policy_projection_returns_canonical_n_by_7() -> None:
    actions = np.arange(3 * MODEL_ACTION_DIM, dtype=np.float32).reshape(
        3, MODEL_ACTION_DIM
    )

    projected = project_policy_actions(actions)

    assert projected.shape == (3, CANONICAL_ACTION_DIM)
    np.testing.assert_array_equal(projected, actions[:, :CANONICAL_ACTION_DIM])


@pytest.mark.parametrize("invalid_dim", [6, 7, 31, 33])
def test_pi05_policy_projection_rejects_unexpected_model_head(
    invalid_dim: int,
) -> None:
    with pytest.raises(ValueError, match="32 action dimensions"):
        project_policy_actions(np.zeros((2, invalid_dim), dtype=np.float32))
