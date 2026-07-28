from __future__ import annotations

from copy import deepcopy


def test_health_matches_executor_contract(service):
    status, response = service.health()

    assert status == 200
    assert response["service"] == "openvla_oft"
    assert response["status"] == "ready"
    assert response["schema_version"].startswith("1.")
    assert response["checkpoint_sha"].startswith("sha256:")
    assert response["norm_stats_sha"].startswith("sha256:")
    assert response["supported_action_contracts"] == ["1.0"]


def test_health_reports_degraded_in_real_mode(config):
    from openvla_oft.routes import OpenVLAOFTService

    real_config = deepcopy(config)
    real_config["mock_mode"] = False
    service = OpenVLAOFTService(real_config)

    status, response = service.health()

    assert status == 200
    assert response["status"] == "degraded"
    assert response["device"]["mode"] == "real"
