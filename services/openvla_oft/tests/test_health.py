from __future__ import annotations


def test_health_matches_executor_schema(service, schema_validator):
    status, response = service.health()

    assert status == 200
    schema_validator("executor-health.schema.json").validate(response)
    assert response["service"] == "openvla_oft"
    assert response["status"] == "ready"
    assert response["supported_action_contracts"] == ["1.0"]
