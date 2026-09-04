"""Opt-in π0.5 action-chain audit tests.

Auditing is diagnostic only: the normal action contract and clipping behavior
must remain unchanged when it is enabled or disabled.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import numpy as np

from services.pi05.src.action_audit import ActionAudit, array_payload
from services.pi05.src.observation import ObsPacket
from services.pi05.src.pi05 import Pi05Executor
from services.pi05.src.pi05_client import LocalOpenPiPolicyClient


def test_action_audit_is_disabled_by_default(monkeypatch, tmp_path):
    monkeypatch.delenv("PI05_ACTION_AUDIT", raising=False)
    monkeypatch.setenv("PI05_ACTION_AUDIT_PATH", str(tmp_path / "audit.jsonl"))

    audit = ActionAudit()
    audit.emit("should_not_exist", value=1)

    assert audit.enabled is False
    assert not (tmp_path / "audit.jsonl").exists()


def test_action_audit_writes_jsonl_without_pixel_values(monkeypatch, tmp_path):
    path = tmp_path / "audit" / "chain.jsonl"
    monkeypatch.setenv("PI05_ACTION_AUDIT", "1")
    monkeypatch.setenv("PI05_ACTION_AUDIT_PATH", str(path))

    audit = ActionAudit()
    audit.emit(
        "decoded_observation",
        context={"request_id": "req-1", "step_id": 3},
        image={"sha256": "sha256:abc"},
        state=array_payload(np.zeros(7, dtype=np.float32)),
    )

    record = json.loads(audit.path.read_text(encoding="utf-8").splitlines()[0])
    assert record["stage"] == "decoded_observation"
    assert record["request_id"] == "req-1"
    assert record["step_id"] == 3
    assert "pixels" not in record
    assert "values" not in record["image"]


def test_executor_audit_does_not_change_published_action(monkeypatch, tmp_path):
    monkeypatch.setenv("PI05_MODE", "dummy")
    monkeypatch.setenv("PI05_ACTION_AUDIT", "1")
    monkeypatch.setenv("PI05_ACTION_AUDIT_PATH", str(tmp_path / "audit.jsonl"))

    executor = Pi05Executor()
    raw = np.zeros((10, 7), dtype=np.float32)
    raw[:, 1] = 0.06
    raw[:, 6] = 0.75
    executor._infer_mock = lambda _obs: raw.copy()  # type: ignore[method-assign]

    obs = ObsPacket(
        episode_id="ep-1",
        step_id=2,
        timestamp_ns=1,
        rgb_front=np.zeros((4, 4, 3), dtype=np.uint8),
        rgb_wrist=None,
        robot_state=np.zeros(7, dtype=np.float32),
        instruction="把P01放到S11中",
        runtime_flags={"request_id": "req-1", "arm_id": "Arm_A"},
    )
    chunk = executor.infer(obs)

    assert chunk.actions.shape == (1, 7)
    np.testing.assert_allclose(chunk.actions[0], [0, 0.05, 0, 0, 0, 0, 0.75])
    records = [
        json.loads(line)
        for line in executor._audit.path.read_text(encoding="utf-8").splitlines()
    ]
    stages = [record["stage"] for record in records]
    assert stages == [
        "policy_return_physical",
        "clip_actions",
        "published_first_action",
    ]
    assert all(record["request_id"] == "req-1" for record in records)
    clip = next(record for record in records if record["stage"] == "clip_actions")
    assert clip["clipped_dimensions"] == ["dy"]


def test_action_audit_initialization_is_fail_open(monkeypatch, tmp_path):
    parent = tmp_path / "not-a-directory"
    parent.write_text("occupied", encoding="utf-8")
    monkeypatch.setenv("PI05_ACTION_AUDIT", "1")
    monkeypatch.setenv("PI05_ACTION_AUDIT_PATH", str(parent / "audit.jsonl"))

    audit = ActionAudit()

    assert audit.enabled is False


def test_action_audit_uses_a_distinct_path_per_process(monkeypatch, tmp_path):
    configured = tmp_path / "audit.jsonl"
    monkeypatch.setenv("PI05_ACTION_AUDIT", "1")
    monkeypatch.setenv("PI05_ACTION_AUDIT_PATH", str(configured))

    with monkeypatch.context() as process_one:
        process_one.setattr("services.pi05.src.action_audit.os.getpid", lambda: 101)
        first = ActionAudit()
    with monkeypatch.context() as process_two:
        process_two.setattr("services.pi05.src.action_audit.os.getpid", lambda: 202)
        second = ActionAudit()

    assert first.path != second.path
    assert first.path.name == "audit.pid-101.jsonl"
    assert second.path.name == "audit.pid-202.jsonl"


def test_action_audit_write_failure_disables_audit(monkeypatch, tmp_path):
    monkeypatch.setenv("PI05_ACTION_AUDIT", "1")
    monkeypatch.setenv("PI05_ACTION_AUDIT_PATH", str(tmp_path / "audit.jsonl"))
    audit = ActionAudit()
    broken_target = tmp_path / "directory-target"
    broken_target.mkdir()
    audit.path = broken_target

    audit.emit("write_failure", value=1)

    assert audit.enabled is False


def test_missing_output_transform_emits_audit_unavailable():
    """An OpenPI compatibility break must be visible in the audit stream."""

    client = object.__new__(LocalOpenPiPolicyClient)
    client._policy = object()
    client._audit = MagicMock(enabled=True)
    client._audit_degraded = False

    client._install_audit_transform()

    assert client._audit_degraded is True
    assert client.audit_degraded is True
    client._audit.emit.assert_called_once_with(
        "audit_unavailable",
        component="policy_output_transform",
        reason="callable _output_transform not found",
    )
