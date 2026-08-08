from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.pi05_base.contracts import (
    ExperimentConfig,
    ExperimentObservation,
    PolicyOutput,
)
from experiments.pi05_base.probe_base_inference import build_report, main
from experiments.pi05_base.validate_actions import (
    inspect_actions,
    require_finite_actions,
)


CONFIG_PATH = Path(__file__).parents[1] / "configs" / "base_probe.json"


def _observation() -> ExperimentObservation:
    return ExperimentObservation(
        observation_id="obs-report",
        timestamp_ns=10,
        front_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        joint_position=np.zeros(7, dtype=np.float32),
        gripper_position=0.0,
        prompt="Pick up the part.",
    )


def test_action_inspection_preserves_native_shape() -> None:
    actions = np.arange(96, dtype=np.float32).reshape(3, 32)
    report = inspect_actions(actions)
    assert report.shape == (3, 32)
    assert report.action_dim == 32
    assert report.all_finite is True


def test_non_finite_output_is_reported_and_rejected_for_execution() -> None:
    actions = np.zeros((2, 32), dtype=np.float32)
    actions[0, 3] = np.nan
    report = inspect_actions(actions)
    assert report.all_finite is False
    assert report.non_finite_count == 1
    with pytest.raises(ValueError, match="NaN/Inf"):
        require_finite_actions(actions)


def test_mock_report_cannot_claim_real_or_closed_loop_evidence() -> None:
    config = ExperimentConfig.from_path(CONFIG_PATH)
    observation = _observation()
    output = PolicyOutput(
        actions=np.zeros((15, 32), dtype=np.float32),
        policy_mode="mock",
        checkpoint_reference=config.checkpoint_uri,
        latency_ms=1.0,
    )
    report = build_report(
        config=config,
        observation=observation,
        output=output,
    )
    assert report["input"]["ground_truth_included"] is False
    assert report["evidence"]["valid_model_inference_evidence"] is False
    assert report["evidence"]["valid_closed_loop_evidence"] is False
    assert report["evidence"]["action_semantics_confirmed"] is False


def test_windows_mock_cli_writes_json_report(tmp_path: Path) -> None:
    output_path = tmp_path / "probe.json"
    assert (
        main(
            [
                "--mode",
                "mock",
                "--config",
                str(CONFIG_PATH),
                "--report",
                str(output_path),
            ]
        )
        == 0
    )
    report = json.loads(output_path.read_text(encoding="utf-8"))
    assert report["policy"]["mode"] == "mock"
    assert report["output"]["shape"] == [15, 32]
    assert report["evidence"]["valid_model_inference_evidence"] is False

