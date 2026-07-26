from __future__ import annotations

import json
import unittest
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from industrial_agent.contracts import Postcondition, TaskSchema
from industrial_agent.errors import ExecutorError, FailureCode
from industrial_agent.executor import (
    ExecutionContext,
    OpenVLAOFTAdapter,
    Pi05Adapter,
    build_executors_from_config,
)
from industrial_agent.observation import ObservationGateway

from tests.test_contracts_and_observation import raw_observation

CHECKPOINT_SHA = f"sha256:{'1' * 64}"
NORM_STATS_SHA = f"sha256:{'2' * 64}"
OTHER_CHECKPOINT_SHA = f"sha256:{'3' * 64}"
OTHER_NORM_STATS_SHA = f"sha256:{'4' * 64}"


class EchoTransport:
    def __init__(
        self,
        *,
        service: str = "openvla_oft",
        checkpoint_sha: str = CHECKPOINT_SHA,
        norm_stats_sha: str = NORM_STATS_SHA,
        corrupt_trace: bool = False,
        health_overrides: Mapping[str, Any] | None = None,
        chunk_overrides: Mapping[str, Any] | None = None,
        response_status: str = "ok",
        error_code: str | None = None,
    ):
        self.calls: list[tuple[str, Mapping[str, Any], int]] = []
        self.service = service
        self.checkpoint_sha = checkpoint_sha
        self.norm_stats_sha = norm_stats_sha
        self.corrupt_trace = corrupt_trace
        self.health_overrides = dict(health_overrides or {})
        self.chunk_overrides = dict(chunk_overrides or {})
        self.response_status = response_status
        self.error_code = error_code

    def request(
        self, route: str, payload: Mapping[str, Any], timeout_ms: int
    ) -> Mapping[str, Any]:
        self.calls.append((route, payload, timeout_ms))
        if route == "/health":
            health = {
                "schema_version": "1.0",
                "service": self.service,
                "status": "ready",
                "checkpoint_sha": self.checkpoint_sha,
                "norm_stats_sha": self.norm_stats_sha,
                "supported_task_types": (
                    [
                        "pick_place",
                        "object_localization",
                        "visual_manipulation",
                    ]
                    if self.service == "openvla_oft"
                    else [
                        "pick_place",
                        "visual_manipulation",
                        "instruction_interaction",
                    ]
                ),
                "supported_action_contracts": ["1.0"],
            }
            health.update(self.health_overrides)
            return health
        if route == "/v1/cancel":
            return {"status": "cancelled"}
        response = {
            key: payload[key]
            for key in (
                "schema_version",
                "request_id",
                "trace_id",
                "episode_id",
                "task_id",
                "subtask_id",
                "step_id",
                "observation_id",
                "executor",
                "checkpoint_sha",
                "norm_stats_sha",
            )
        }
        response["status"] = self.response_status
        action_chunk = {
            "contract_version": "1.0",
            "chunk_id": "canonical-chunk",
            "task_id": payload["task_id"],
            "executor": payload["executor"],
            "action_space": "ee_delta_pose_gripper",
            "frame": "robot_base",
            "translation_unit": "m",
            "rotation_unit": "rad",
            "gripper_unit": "normalized",
            "steps": [
                {
                    "values": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                    "duration_ms": 137,
                }
            ],
        }
        action_chunk.update(self.chunk_overrides)
        if self.response_status == "ok":
            response["action_chunk"] = action_chunk
            response["timing"] = {
                "queue_ms": 1,
                "inference_ms": 2,
                "total_ms": 3,
            }
        else:
            response["error"] = {
                "code": self.error_code
                or (
                    FailureCode.EXECUTOR_CANCELLED.value
                    if self.response_status == "cancelled"
                    else FailureCode.EXECUTOR_RUNTIME.value
                ),
                "message": f"mock {self.response_status}",
                "retryable": self.response_status == "error",
            }
        if self.corrupt_trace:
            response["trace_id"] = "wrong"
        return response


def task() -> TaskSchema:
    return TaskSchema(
        task_id="parent:S01",
        instruction="execute semantic action",
        task_type="pick_place",
        metadata={"subtask_id": "S01"},
        postconditions=(
            Postcondition(kind="field_equals", path="task.status", expected="done"),
        ),
    )


class ExecutorAdapterTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = raw_observation()

        def image(camera_id: str, digest_char: str) -> dict[str, object]:
            digest = digest_char * 64
            return {
                "uri": f"cas://sha256/{digest}",
                "image_sha256": f"sha256:{digest}",
                "camera_id": camera_id,
                "width": 640,
                "height": 480,
            }

        raw["camera"] = {
            "full_image": image("CAM_HANDOFF", "a"),
            "arm_a_rgb": image("CAM_A_TOP", "b"),
            "arm_b_rgb": image("CAM_B_TOP", "c"),
            "wrist_image": image("CAM_WRIST", "d"),
        }
        robot = raw["robot"]
        assert isinstance(robot, dict)
        base_pose = list(robot["tcp_pose_m_rad"])
        base_state = [*base_pose, 0.5]
        robot["arm_a"] = {
            "tcp_pose_m_rad": base_pose,
            "state": base_state,
            "retreated": False,
        }
        robot["arm_b"] = {
            "tcp_pose_m_rad": [0.4, *base_pose[1:]],
            "state": [0.4, *base_state[1:]],
            "retreated": True,
        }
        self.observation = ObservationGateway().ingest_online(raw)
        self.context = ExecutionContext(
            run_id="episode-1",
            strategy_attempt=1,
            replan_index=0,
            step_id=4,
        )

    def test_openvla_request_and_canonical_response(self) -> None:
        transport = EchoTransport()
        adapter = OpenVLAOFTAdapter(
            transport, checkpoint_sha=CHECKPOINT_SHA, norm_stats_sha=NORM_STATS_SHA
        )
        result = adapter.plan(task(), self.observation, self.context)
        self.assertEqual(result.chunk_id, "canonical-chunk")
        self.assertEqual(result.steps[0].duration_ms, 137)
        route, payload, _ = transport.calls[-1]
        self.assertEqual(route, "/v1/infer")
        self.assertEqual(payload["observation_id"], self.observation.observation_id)
        model_input = payload["model_input"]
        assert isinstance(model_input, Mapping)
        self.assertEqual(
            model_input["full_image"],
            self.observation.data["camera"]["arm_b_rgb"],
        )
        self.assertIn("task_description", model_input)

    def test_cancel_reuses_run_and_subtask_correlation(self) -> None:
        transport = EchoTransport()
        adapter = OpenVLAOFTAdapter(
            transport, checkpoint_sha=CHECKPOINT_SHA, norm_stats_sha=NORM_STATS_SHA
        )
        active_task = task()
        adapter.plan(active_task, self.observation, self.context)
        adapter.cancel(active_task.task_id, "replan")
        route, payload, _ = transport.calls[-1]
        self.assertEqual(route, "/v1/cancel")
        self.assertEqual(payload["trace_id"], self.context.run_id)
        self.assertEqual(payload["episode_id"], self.context.run_id)
        self.assertEqual(payload["subtask_id"], "S01")

    def test_pi05_request_contains_prompt_and_observation(self) -> None:
        transport = EchoTransport(
            service="pi05",
            checkpoint_sha=OTHER_CHECKPOINT_SHA,
            norm_stats_sha=OTHER_NORM_STATS_SHA,
        )
        adapter = Pi05Adapter(
            transport,
            checkpoint_sha=OTHER_CHECKPOINT_SHA,
            norm_stats_sha=OTHER_NORM_STATS_SHA,
        )
        result = adapter.plan(task(), self.observation, self.context)
        self.assertEqual(len(result.steps), 1)
        self.assertEqual(result.steps[0].duration_ms, 137)
        model_input = transport.calls[-1][1]["model_input"]
        assert isinstance(model_input, Mapping)
        self.assertEqual(model_input["prompt"], "execute semantic action")
        self.assertIn("observation", model_input)

    def test_response_correlation_mismatch_is_rejected(self) -> None:
        adapter = OpenVLAOFTAdapter(
            EchoTransport(corrupt_trace=True),
            checkpoint_sha=CHECKPOINT_SHA,
            norm_stats_sha=NORM_STATS_SHA,
        )
        with self.assertRaises(ExecutorError) as caught:
            adapter.plan(task(), self.observation, self.context)
        self.assertEqual(caught.exception.code, FailureCode.EXECUTOR_BAD_RESPONSE)

    def test_health_requires_pinned_identity_and_action_contract(self) -> None:
        adapter_cases = (
            (OpenVLAOFTAdapter, "openvla_oft"),
            (Pi05Adapter, "pi05"),
        )
        bad_health = (
            ("schema_version", "2.0"),
            ("service", "other"),
            ("checkpoint_sha", "wrong-checkpoint"),
            ("norm_stats_sha", "wrong-norm"),
            ("supported_action_contracts", ["2.0"]),
        )
        for adapter_type, service in adapter_cases:
            with self.subTest(adapter=service, field="valid"):
                adapter = adapter_type(
                    EchoTransport(service=service),
                    checkpoint_sha=CHECKPOINT_SHA,
                    norm_stats_sha=NORM_STATS_SHA,
                )
                self.assertTrue(adapter.health())
            for field, value in bad_health:
                with self.subTest(adapter=service, field=field):
                    adapter = adapter_type(
                        EchoTransport(
                            service=service,
                            health_overrides={field: value},
                        ),
                        checkpoint_sha=CHECKPOINT_SHA,
                        norm_stats_sha=NORM_STATS_SHA,
                    )
                    self.assertFalse(adapter.health())

    def test_both_adapters_reject_tampered_canonical_chunk_metadata(self) -> None:
        adapter_cases = (
            (OpenVLAOFTAdapter, "openvla_oft"),
            (Pi05Adapter, "pi05"),
        )
        corruptions = (
            ("contract_version", "2.0"),
            ("task_id", "wrong-task"),
            ("executor", "wrong-executor"),
            ("action_space", "joint_position"),
            ("frame", "camera"),
            ("translation_unit", "cm"),
            ("rotation_unit", "deg"),
            ("gripper_unit", "raw"),
        )
        for adapter_type, service in adapter_cases:
            for field, value in corruptions:
                with self.subTest(adapter=service, field=field):
                    adapter = adapter_type(
                        EchoTransport(
                            service=service,
                            chunk_overrides={field: value},
                        ),
                        checkpoint_sha=CHECKPOINT_SHA,
                        norm_stats_sha=NORM_STATS_SHA,
                    )
                    with self.assertRaises(ExecutorError) as caught:
                        adapter.plan(task(), self.observation, self.context)
                    self.assertEqual(
                        caught.exception.code, FailureCode.EXECUTOR_BAD_RESPONSE
                    )

    def test_both_adapters_preserve_stable_error_and_cancel_codes(self) -> None:
        adapter_cases = (
            (OpenVLAOFTAdapter, "openvla_oft"),
            (Pi05Adapter, "pi05"),
        )
        statuses = (
            ("error", FailureCode.EXECUTOR_TIMEOUT),
            ("cancelled", FailureCode.EXECUTOR_CANCELLED),
        )
        for adapter_type, service in adapter_cases:
            for status, code in statuses:
                with self.subTest(adapter=service, status=status):
                    adapter = adapter_type(
                        EchoTransport(
                            service=service,
                            response_status=status,
                            error_code=code.value,
                        ),
                        checkpoint_sha=CHECKPOINT_SHA,
                        norm_stats_sha=NORM_STATS_SHA,
                    )
                    with self.assertRaises(ExecutorError) as caught:
                        adapter.plan(task(), self.observation, self.context)
                    self.assertEqual(caught.exception.code, code)
                    self.assertEqual(caught.exception.retryable, status == "error")

    def test_invalid_step_duration_is_rejected_not_defaulted(self) -> None:
        adapter = OpenVLAOFTAdapter(
            EchoTransport(
                chunk_overrides={
                    "steps": [
                        {
                            "values": [0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                            "duration_ms": True,
                        }
                    ]
                }
            ),
            checkpoint_sha=CHECKPOINT_SHA,
            norm_stats_sha=NORM_STATS_SHA,
        )
        with self.assertRaises(ExecutorError) as caught:
            adapter.plan(task(), self.observation, self.context)
        self.assertEqual(caught.exception.code, FailureCode.EXECUTOR_BAD_RESPONSE)

    def test_executor_factory_consumes_configured_urls_and_artifact_ids(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        digest_pairs = {
            "openvla_oft": (CHECKPOINT_SHA, NORM_STATS_SHA),
            "pi05": (OTHER_CHECKPOINT_SHA, OTHER_NORM_STATS_SHA),
        }
        for name, raw in config["executors"].items():
            raw["checkpoint_sha"], raw["norm_stats_sha"] = digest_pairs[name]

        calls: list[tuple[str, str]] = []

        def factory(name: str, base_url: str) -> EchoTransport:
            calls.append((name, base_url))
            raw = config["executors"][name]
            return EchoTransport(
                service=name,
                checkpoint_sha=raw["checkpoint_sha"],
                norm_stats_sha=raw["norm_stats_sha"],
            )

        executors = build_executors_from_config(config, factory)
        self.assertEqual(
            calls,
            [
                ("openvla_oft", "http://127.0.0.1:8101"),
                ("pi05", "http://127.0.0.1:8102"),
            ],
        )
        self.assertEqual(
            [item.descriptor.name for item in executors],
            ["openvla_oft", "pi05"],
        )

    def test_executor_factory_rejects_unpinned_artifacts(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        config = deepcopy(config)
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            build_executors_from_config(
                config,
                lambda name, base_url: EchoTransport(service=name),
            )

    def test_executor_factory_builds_only_explicitly_enabled_services(self) -> None:
        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        config["executors"]["openvla_oft"]["checkpoint_sha"] = CHECKPOINT_SHA
        config["executors"]["openvla_oft"]["norm_stats_sha"] = NORM_STATS_SHA
        config["executors"]["pi05"]["enabled"] = False
        calls: list[tuple[str, str]] = []

        def factory(name: str, base_url: str) -> EchoTransport:
            calls.append((name, base_url))
            return EchoTransport(service=name)

        executors = build_executors_from_config(config, factory)
        self.assertEqual([item.descriptor.name for item in executors], ["openvla_oft"])
        self.assertEqual(calls, [("openvla_oft", "http://127.0.0.1:8101")])

    def test_adapters_and_factory_reject_mutable_artifact_aliases(self) -> None:
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            OpenVLAOFTAdapter(
                EchoTransport(),
                checkpoint_sha="latest00",
                norm_stats_sha=NORM_STATS_SHA,
            )
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            OpenVLAOFTAdapter(
                EchoTransport(),
                checkpoint_sha="REPLACE_WITH_PINNED_SHA",
                norm_stats_sha=NORM_STATS_SHA,
                task_types=frozenset({"mock_demo"}),
            )

        root = Path(__file__).resolve().parents[1]
        config = json.loads(
            (root / "configs" / "agent.default.json").read_text(encoding="utf-8")
        )
        config["executors"]["openvla_oft"]["checkpoint_sha"] = "latest00"
        config["executors"]["openvla_oft"]["norm_stats_sha"] = "version1"
        with self.assertRaisesRegex(ValueError, "64 hexadecimal"):
            build_executors_from_config(
                config,
                lambda name, base_url: EchoTransport(service=name),
            )


if __name__ == "__main__":
    unittest.main()
