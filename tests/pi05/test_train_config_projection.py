from __future__ import annotations

from pathlib import Path

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


def test_pi05_docker_uses_uv_managed_python_instead_of_ubuntu_apt() -> None:
    dockerfile = (
        Path(__file__).resolve().parents[2] / "services" / "pi05" / "Dockerfile"
    ).read_text(encoding="utf-8")

    assert "FROM ghcr.io/astral-sh/uv:0.7.19 AS uv" in dockerfile
    assert "COPY --from=uv /uv /uvx /bin/" in dockerfile
    assert "uv python install 3.11" in dockerfile
    assert "uv sync --frozen --python 3.11" in dockerfile
    assert "apt-get install" in dockerfile
    assert "python3.11 \\" not in dockerfile
