from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping
from unittest.mock import patch

import pytest

from industrial_agent.contracts import ActionStep, Postcondition, TaskSchema
from industrial_agent.environment import SafeStopReceipt
from industrial_agent.errors import FailureCode
from industrial_agent.fsm import AgentState
from industrial_agent.orchestrator import RunResult
from industrial_agent.supervisor_main import (
    DirectEnvironmentHost,
    build_supervisor,
    load_agent_config,
    load_task,
    main,
    resolve_environment_host,
    run_result_to_dict,
)
from industrial_agent.v2_supervisor import V2Supervisor
from industrial_agent.v2_task_profile import require_formal_v2_task


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPOSITORY_ROOT / "configs" / "agent.default.json"
TASK_EXAMPLE_PATH = REPOSITORY_ROOT / "configs" / "task.v2.p01-to-s11.example.json"
PINNED_SHA = f"sha256:{'a' * 64}"


def _production_config() -> dict[str, Any]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    for executor in config["executors"].values():
        executor["checkpoint_sha"] = PINNED_SHA
        executor["norm_stats_sha"] = PINNED_SHA
    return config


class _RecordingTransport:
    def request(
        self,
        route: str,
        payload: Mapping[str, Any],
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        raise AssertionError(
            f"unexpected request during composition: {route}/{payload}/{timeout_ms}"
        )


class _Environment:
    def __init__(self, *, stop_confirmed: bool = True) -> None:
        self.stop_confirmed = stop_confirmed
        self.stop_reasons: list[str] = []

    def observe(self) -> Mapping[str, Any]:
        return {}

    def step(
        self,
        action: ActionStep,
        *,
        arm_id: str,
        control_token: str,
        command_id: str,
        expected_observation_id: str,
        expected_state_digest: str,
    ) -> Mapping[str, Any]:
        del (
            action,
            arm_id,
            control_token,
            command_id,
            expected_observation_id,
            expected_state_digest,
        )
        return {}

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        assert reason
        self.stop_reasons.append(reason)
        return SafeStopReceipt(
            controller_ack=self.stop_confirmed,
            buffers_cleared=self.stop_confirmed,
            arm_a_stopped=self.stop_confirmed,
            arm_b_stopped=self.stop_confirmed,
            stop_epoch="test-stop-1",
        )


class _Host:
    def __init__(self, environment: _Environment) -> None:
        self.environment = environment
        self.close_reasons: list[str] = []

    def run(self, operation: Any) -> Any:
        return operation()

    def close(self, reason: str) -> None:
        self.close_reasons.append(reason)


class _Supervisor:
    def __init__(self, result: RunResult) -> None:
        self.result = result

    def run(self, task: TaskSchema, environment: _Environment) -> RunResult:
        assert task.task_id
        assert environment is not None
        return self.result


def test_build_supervisor_wires_formal_v2_pi05_and_shadow_yolo() -> None:
    config = _production_config()
    calls: list[tuple[str, str]] = []

    def factory(service_name: str, base_url: str) -> _RecordingTransport:
        calls.append((service_name, base_url))
        return _RecordingTransport()

    supervisor = build_supervisor(config, transport_factory=factory)
    assert isinstance(supervisor, V2Supervisor)
    assert calls == [
        ("pi05", "http://127.0.0.1:8101"),
        ("yolo", "http://127.0.0.1:8103"),
    ]


def test_build_supervisor_rejects_artifact_placeholders() -> None:
    with pytest.raises(ValueError, match="sha256"):
        build_supervisor(
            json.loads(CONFIG_PATH.read_text(encoding="utf-8")),
            transport_factory=lambda _name, _url: _RecordingTransport(),
        )


def test_resolve_environment_host_wraps_direct_environment() -> None:
    module = SimpleNamespace(create=lambda _config: _Environment())
    with patch(
        "industrial_agent.supervisor_main.importlib.import_module",
        return_value=module,
    ):
        host = resolve_environment_host("platform.runtime:create", {})
    assert isinstance(host, DirectEnvironmentHost)
    assert isinstance(host.environment, _Environment)


def test_resolve_environment_host_rejects_none() -> None:
    module = SimpleNamespace(create=lambda _config: None)
    with patch(
        "industrial_agent.supervisor_main.importlib.import_module",
        return_value=module,
    ):
        with pytest.raises(TypeError, match="returned None"):
            resolve_environment_host("platform.runtime:create", {})


def test_loaders_require_bounded_json_objects(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps(_production_config(), ensure_ascii=False),
        encoding="utf-8",
    )
    assert load_agent_config(config_path)["config_version"] == "2.0"

    task_path = tmp_path / "task.json"
    task = TaskSchema(
        task_id="runtime-task",
        instruction=(
            "将工作区中的四个红色零件依次装入料箱；倒放零件先调整为正向。"
            "装箱完成后，将料箱放到中央交接位并返回 HOME_A。"
            "失败时重新观察后继续。"
        ),
        task_type="pick_place",
        postconditions=(
            Postcondition(
                kind="field_equals",
                path="task.bin_at_finished",
                expected=True,
                required_votes=2,
            ),
        ),
    )
    task_path.write_text(
        json.dumps(task.to_dict(), ensure_ascii=False),
        encoding="utf-8",
    )
    assert load_task(task_path) == task

    scalar_path = tmp_path / "scalar.json"
    scalar_path.write_text("null", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON object"):
        load_agent_config(scalar_path)


def test_repository_task_example_matches_frozen_v2_profile() -> None:
    task = load_task(TASK_EXAMPLE_PATH)
    profile = require_formal_v2_task("P01_TO_S11")
    assert task.instruction == profile.instruction
    assert task.target_object == profile.target_object
    assert task.target_location == profile.target_slot
    assert task.postconditions[0].required_votes == 2


def test_run_result_serialization_converts_enums() -> None:
    result = _run_result(success=False)
    converted = run_result_to_dict(result)
    assert converted["state"] == "FAILED"
    assert converted["failure_code"] == "TASK_1001_INVALID"


def _run_result(*, success: bool) -> RunResult:
    return RunResult(
        run_id="run-1",
        task_id="task-1",
        state=AgentState.SUCCEEDED if success else AgentState.FAILED,
        success=success,
        failure_code=FailureCode.NONE if success else FailureCode.INVALID_TASK,
        message="complete" if success else "invalid",
        executor_history=(),
        control_token_history=("NONE",),
        replan_counts={},
        transitions=(),
        verification=None,
        task_plan={},
        events=(),
    )


def test_cli_refuses_missing_environment_factory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("INDUSTRIAL_AGENT_ENVIRONMENT_FACTORY", raising=False)
    assert (
        main(
            [
                "--config",
                str(tmp_path / "unused-config.json"),
                "--task",
                str(tmp_path / "unused-task.json"),
            ]
        )
        == 2
    )


@pytest.mark.parametrize(
    ("stop_confirmed", "expected_exit"),
    [(True, 0), (False, 3)],
)
def test_cli_requires_confirmed_shutdown_stop(
    stop_confirmed: bool,
    expected_exit: int,
) -> None:
    environment = _Environment(stop_confirmed=stop_confirmed)
    host = _Host(environment)
    supervisor = _Supervisor(_run_result(success=True))
    task = TaskSchema(
        task_id="task-1",
        instruction="fixed instruction",
        task_type="pick_place",
        postconditions=(
            Postcondition(
                kind="field_equals",
                path="task.bin_at_finished",
                expected=True,
            ),
        ),
    )
    with (
        patch(
            "industrial_agent.supervisor_main.load_agent_config",
            return_value=_production_config(),
        ),
        patch("industrial_agent.supervisor_main.load_task", return_value=task),
        patch(
            "industrial_agent.supervisor_main.build_supervisor",
            return_value=supervisor,
        ),
        patch(
            "industrial_agent.supervisor_main.resolve_environment_host",
            return_value=host,
        ),
        patch(
            "industrial_agent.supervisor_main._install_signal_handlers",
            return_value={},
        ),
    ):
        exit_code = main(
            [
                "--config",
                "agent.json",
                "--task",
                "task.json",
                "--environment-factory",
                "platform.runtime:create",
            ]
        )
    assert exit_code == expected_exit
    assert environment.stop_reasons == [
        "Supervisor task completed; revoke motion before process exit"
    ]
    assert host.close_reasons == ["Supervisor process is exiting"]
