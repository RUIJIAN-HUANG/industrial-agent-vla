from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.pi05_base.contracts import (
    CANONICAL_BASE_CHECKPOINT,
    ExperimentConfig,
    ExperimentObservation,
    PolicyOutput,
    is_pi05_base_checkpoint,
)
from experiments.pi05_base.base_client import OpenPiBaseClient
from experiments.pi05_base.mock_client import MockBaseClient


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "base_probe.json"


def _observation(*, wrist: bool = False) -> ExperimentObservation:
    return ExperimentObservation(
        observation_id="obs-1",
        timestamp_ns=1,
        front_rgb=np.zeros((32, 48, 3), dtype=np.uint8),
        wrist_rgb=(np.zeros((16, 16, 3), dtype=np.uint8) if wrist else None),
        joint_position=np.zeros(7, dtype=np.float32),
        gripper_position=1.0,
        prompt="Pick up the red part.",
    )


def test_default_config_is_pinned_to_base_checkpoint() -> None:
    config = ExperimentConfig.from_path(CONFIG_PATH)
    assert config.checkpoint_uri == CANONICAL_BASE_CHECKPOINT
    assert config.input_profile == "droid_joint_gripper"


@pytest.mark.parametrize(
    "reference, expected",
    [
        (CANONICAL_BASE_CHECKPOINT, True),
        (r"D:\models\pi05_base", True),
        ("gs://openpi-assets/checkpoints/pi05_droid", False),
        ("gs://openpi-assets/checkpoints/pi05_libero", False),
        ("./checkpoints/pi05_industrial", False),
    ],
)
def test_checkpoint_identity(reference: str, expected: bool) -> None:
    assert is_pi05_base_checkpoint(reference) is expected


def test_observation_owns_read_only_validated_arrays() -> None:
    source = np.zeros((32, 48, 3), dtype=np.uint8)
    observation = ExperimentObservation(
        observation_id="obs-owned",
        timestamp_ns=1,
        front_rgb=source,
        joint_position=np.zeros(7, dtype=np.float32),
        gripper_position=1.0,
        prompt="Pick up the red part.",
    )
    source[:] = 255
    assert observation.front_rgb.dtype == np.uint8
    assert observation.front_rgb.flags.writeable is False
    assert np.count_nonzero(observation.front_rgb) == 0


@pytest.mark.parametrize(
    "field, value, message",
    [
        ("front_rgb", np.zeros((32, 48), dtype=np.uint8), "shape"),
        ("front_rgb", np.zeros((32, 48, 3), dtype=np.float32), "uint8"),
        ("joint_position", np.zeros(6, dtype=np.float32), "seven"),
        (
            "joint_position",
            np.asarray([0, 0, 0, 0, 0, 0, np.nan], dtype=np.float32),
            "NaN",
        ),
        ("gripper_position", 2.0, "[0,1]"),
        ("prompt", "  ", "non-empty"),
    ],
)
def test_observation_rejects_invalid_inputs(field, value, message) -> None:
    kwargs = {
        "observation_id": "obs-1",
        "timestamp_ns": 1,
        "front_rgb": np.zeros((32, 48, 3), dtype=np.uint8),
        "joint_position": np.zeros(7, dtype=np.float32),
        "gripper_position": 1.0,
        "prompt": "Pick.",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=message):
        ExperimentObservation(**kwargs)


def test_droid_example_requires_real_wrist_image() -> None:
    with pytest.raises(ValueError, match="real wrist_rgb"):
        _observation().to_droid_example()
    example = _observation(wrist=True).to_droid_example()
    assert example["observation/joint_position"].shape == (7,)
    assert example["observation/gripper_position"].shape == (1,)


def test_real_client_rejects_missing_wrist_before_openpi_import() -> None:
    config = ExperimentConfig.from_path(CONFIG_PATH)
    client = OpenPiBaseClient(config)
    with pytest.raises(ValueError, match="real wrist_rgb"):
        client.infer(_observation())


def test_mock_is_deterministic_and_never_evidence() -> None:
    config = ExperimentConfig.from_path(CONFIG_PATH)
    client = MockBaseClient(config)
    first = client.infer(_observation())
    second = client.infer(_observation())
    assert isinstance(first, PolicyOutput)
    np.testing.assert_array_equal(first.actions, second.actions)
    assert first.actions.shape == (15, 32)
    assert first.policy_mode == "mock"
    assert first.metadata["valid_experiment_evidence"] is False
