from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
from industrial_agent.executor import OPENVLA_OFT_TASK_TYPES, OpenVLAOFTAdapter

from openvla_oft.config import load_config

SERVICE_ROOT = Path(__file__).resolve().parents[1]


def test_health_matches_executor_contract(service):
    status, response = service.health()

    assert status == 200
    assert response["service"] == "openvla_oft"
    assert response["status"] == "ready"
    assert response["schema_version"].startswith("1.")
    assert response["checkpoint_sha"].startswith("sha256:")
    assert response["norm_stats_sha"].startswith("sha256:")
    assert response["supported_action_contracts"] == ["1.0"]
    assert response["supported_task_types"] == sorted(OPENVLA_OFT_TASK_TYPES)


def test_health_is_accepted_by_supervisor_adapter(service, config):
    class InProcessTransport:
        def request(self, route, payload, timeout_ms):
            del payload, timeout_ms
            assert route == "/health"
            return service.health()[1]

    adapter = OpenVLAOFTAdapter(
        InProcessTransport(),
        checkpoint_sha=config["artifacts"]["checkpoint_sha"],
        norm_stats_sha=config["artifacts"]["norm_stats_sha"],
    )

    assert adapter.health() is True


def test_real_mode_fails_closed_when_checkpoint_is_missing(config):
    from openvla_oft.model import OpenVLAOFTModel

    real_config = deepcopy(config)
    real_config["mock_mode"] = False
    real_config["runtime"]["unnorm_key"] = "industrial_arm_b"

    with pytest.raises(RuntimeError, match="checkpoint_dir does not exist"):
        OpenVLAOFTModel(real_config)


def test_default_real_mode_rejects_all_zero_artifact_digests(monkeypatch):
    monkeypatch.delenv("OPENVLA_OFT_USE_MOCK", raising=False)
    monkeypatch.delenv("OPENVLA_OFT_CHECKPOINT_SHA", raising=False)
    monkeypatch.delenv("OPENVLA_OFT_NORM_STATS_SHA", raising=False)

    with pytest.raises(ValueError, match="all-zero mock digest"):
        load_config(SERVICE_ROOT / "configs")


def test_real_mode_accepts_explicit_nonzero_artifact_digests(monkeypatch):
    monkeypatch.setenv("OPENVLA_OFT_USE_MOCK", "0")
    monkeypatch.setenv("OPENVLA_OFT_CHECKPOINT_SHA", f"sha256:{'a' * 64}")
    monkeypatch.setenv("OPENVLA_OFT_NORM_STATS_SHA", f"sha256:{'b' * 64}")
    monkeypatch.setenv("OPENVLA_OFT_UNNORM_KEY", "industrial_arm_b")

    config = load_config(SERVICE_ROOT / "configs")

    assert config["mock_mode"] is False
    assert config["runtime"]["unnorm_key"] == "industrial_arm_b"
