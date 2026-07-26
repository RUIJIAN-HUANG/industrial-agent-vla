from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from typing import Any, Mapping

from industrial_agent.contracts import (
    ACTION_CONTRACT_VERSION,
    ActionChunk,
    ActionStep,
    Observation,
    Postcondition,
    TaskSchema,
)
from industrial_agent.errors import AgentError, ExecutorError, FailureCode
from industrial_agent.executor import (
    ExecutionContext,
    OpenVLAOFTAdapter,
    Pi05Adapter,
)
from industrial_agent.fsm import AgentState
from industrial_agent.lifecycle import FixedTaskProfile
from industrial_agent.mock import (
    FixedDualArmMockSimulator,
    MockExecutor,
)
from industrial_agent.observation import ObservationGateway
from industrial_agent.orchestrator import IndustrialAgent
from industrial_agent.perception import (
    CocoExportManifest,
    Detection,
    DetectionEvidenceSink,
    DetectionPacket,
    MockPerceptionAgent,
    PerceptionContext,
    PerceptionError,
)
from industrial_agent.telemetry import EventSink


CHECKPOINT_SHA = f"sha256:{'1' * 64}"
CLASS_MAP_SHA = f"sha256:{'2' * 64}"
CONFIG_SHA = f"sha256:{'3' * 64}"
NORM_STATS_SHA = f"sha256:{'4' * 64}"


def four_agent_task(task_id: str) -> TaskSchema:
    return TaskSchema(
        task_id=task_id,
        instruction=FixedTaskProfile().arm_a_instruction,
        task_type="mock_demo",
        target_object="industrial_part",
        target_location="FINISHED_01",
        postconditions=(
            Postcondition(
                kind="field_equals",
                path="task.status",
                expected="done",
                required_votes=2,
            ),
        ),
    )


def target_detection(
    context: PerceptionContext,
) -> tuple[Detection, ...]:
    return (
        Detection(
            detection_id=f"detection-{context.image.image_sha256[-8:]}",
            class_id=0,
            class_name="industrial_part",
            confidence=0.97,
            bbox_xyxy=(100.0, 80.0, 240.0, 220.0),
            camera_id=context.image.camera_id,
            image_width=context.image.width,
            image_height=context.image.height,
            track_id="industrial-part-track",
            zone_id="workcell",
        ),
    )


def make_perception() -> MockPerceptionAgent:
    return MockPerceptionAgent(
        checkpoint_sha=CHECKPOINT_SHA,
        class_map_sha=CLASS_MAP_SHA,
        config_sha=CONFIG_SHA,
        detector=target_detection,
    )


class RecordingExecutor(MockExecutor):
    def __init__(self, name: str, dx_m: float):
        super().__init__(name, dx_m)
        self.observation_ids: list[str] = []
        self.original_instructions: list[str | None] = []
        self.subtask_ids: list[str | None] = []

    def plan(
        self,
        task: TaskSchema,
        observation: Observation,
        context: ExecutionContext,
    ) -> ActionChunk:
        self.observation_ids.append(observation.observation_id)
        self.original_instructions.append(context.original_instruction)
        self.subtask_ids.append(task.metadata.get("subtask_id"))
        return super().plan(task, observation, context)


class FlakyPerception:
    def __init__(self, failure_pattern: list[bool]):
        self.delegate = make_perception()
        self.descriptor = self.delegate.descriptor
        self.failure_pattern = list(failure_pattern)
        self.observation_ids: list[str] = []

    def health(self) -> bool:
        return True

    def detect(
        self,
        context: PerceptionContext,
    ) -> DetectionPacket:
        self.observation_ids.append(context.observation_id)
        should_fail = self.failure_pattern.pop(0) if self.failure_pattern else False
        if should_fail:
            raise PerceptionError(
                FailureCode.PERCEPTION_TIMEOUT,
                "transient YOLO timeout",
                retryable=True,
            )
        return self.delegate.detect(context)

    def cancel(self, task_id: str, reason: str) -> None:
        return


class FourAgentOrchestrationTests(unittest.TestCase):
    def make_agent(
        self,
        perception: Any,
        *,
        evidence: DetectionEvidenceSink | None = None,
        max_decisions: int = 4,
        pi05: RecordingExecutor | None = None,
        openvla: RecordingExecutor | None = None,
        events: EventSink | None = None,
    ) -> tuple[IndustrialAgent, RecordingExecutor, RecordingExecutor]:
        openvla = openvla or RecordingExecutor("openvla_oft", 0.01)
        primary = pi05 or RecordingExecutor("pi05", 0.02)
        agent = IndustrialAgent(
            [openvla, primary],
            perception=perception,
            perception_evidence=evidence,
            events=events,
            require_perception=True,
            verification_frames=3,
            max_decisions_per_strategy_attempt=max_decisions,
            max_perception_attempts=1,
        )
        return agent, openvla, primary

    def test_both_vla_services_must_be_healthy_before_any_motion(self) -> None:
        class UnhealthyOpenVLA(RecordingExecutor):
            def health(self) -> bool:
                return False

        unhealthy_openvla = UnhealthyOpenVLA("openvla_oft", 0.01)
        agent, openvla, pi05 = self.make_agent(
            make_perception(),
            openvla=unhealthy_openvla,
        )
        simulator = FixedDualArmMockSimulator()

        result = agent.run(four_agent_task("preflight-health"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, FailureCode.EXECUTOR_UNAVAILABLE)
        self.assertEqual(simulator.arm_a_steps, 0)
        self.assertEqual(simulator.arm_b_steps, 0)
        self.assertEqual(pi05.plan_calls, 0)
        self.assertEqual(openvla.plan_calls, 0)

    def test_complete_run_calls_both_vlas_in_fixed_order(self) -> None:
        perception = make_perception()
        agent, openvla, pi05 = self.make_agent(perception)
        simulator = FixedDualArmMockSimulator()

        result = agent.run(four_agent_task("fixed-order"), simulator)

        self.assertTrue(result.success)
        self.assertEqual(result.state, AgentState.SUCCEEDED)
        self.assertEqual(result.executor_history, ("pi05", "openvla_oft"))
        self.assertEqual(pi05.plan_calls, 1)
        self.assertEqual(openvla.plan_calls, 1)
        self.assertEqual(simulator.step_owners, ["pi05", "openvla_oft"])
        self.assertEqual(
            simulator.step_authorizations,
            [("Arm_A", "A_ONLY"), ("Arm_B", "B_ONLY")],
        )
        self.assertEqual(
            result.control_token_history,
            ("A_ONLY", "HANDOFF_VERIFY", "B_ONLY", "NONE"),
        )
        self.assertFalse(
            any(event.event_type == "recovery.switch" for event in result.events)
        )

        selected = [
            (event.payload["executor"], event.payload["arm_id"])
            for event in result.events
            if event.event_type == "executor.selected"
        ]
        self.assertEqual(
            selected,
            [("pi05", "Arm_A"), ("openvla_oft", "Arm_B")],
        )
        handoff_index = next(
            index
            for index, event in enumerate(result.events)
            if event.event_type == "handoff_ready"
        )
        openvla_index = next(
            index
            for index, event in enumerate(result.events)
            if event.event_type == "executor.selected"
            and event.payload["executor"] == "openvla_oft"
        )
        self.assertLess(handoff_index, openvla_index)
        handoff_start_index = next(
            index
            for index, event in enumerate(result.events)
            if event.event_type == "handoff.verification_started"
        )
        handoff_verification_index = next(
            index
            for index, event in enumerate(result.events)
            if event.event_type == "verification.completed"
            and event.payload["subtask_id"] == "S01_ARM_A_PACK_HANDOFF"
        )
        self.assertLess(handoff_start_index, handoff_verification_index)
        self.assertEqual(
            result.events[handoff_verification_index].payload["control_token"],
            "HANDOFF_VERIFY",
        )

    def test_handoff_requires_three_frames_and_arm_a_retreat(self) -> None:
        agent, openvla, pi05 = self.make_agent(make_perception())
        result = agent.run(
            four_agent_task("three-frame-handoff"),
            FixedDualArmMockSimulator(),
        )

        self.assertTrue(result.success)
        handoff = next(
            event for event in result.events if event.event_type == "handoff.verified"
        )
        self.assertEqual(handoff.payload["stable_frames"], 3)
        self.assertEqual(handoff.payload["required_frames"], 3)
        self.assertEqual(handoff.payload["required_votes"], 2)
        self.assertTrue(handoff.payload["arm_a_retreated"])
        self.assertFalse(handoff.payload["oracle_coordinates_used"])
        handoff_conditions = result.task_plan["subtasks"][0]["postconditions"]
        self.assertEqual(
            {item["required_votes"] for item in handoff_conditions},
            {2},
        )
        precondition = next(
            event
            for event in result.events
            if event.event_type == "subtask.preconditions_checked"
            and event.payload["subtask_id"] == "S02_ARM_B_TRANSPORT"
        )
        self.assertEqual(precondition.payload["frames"], 3)
        self.assertEqual(pi05.plan_calls, 1)
        self.assertEqual(openvla.plan_calls, 1)

    def test_arm_b_is_never_called_before_arm_a_retreat(self) -> None:
        agent, openvla, pi05 = self.make_agent(
            make_perception(),
            max_decisions=1,
        )
        simulator = FixedDualArmMockSimulator(arm_a_retreated_on_handoff=False)

        result = agent.run(four_agent_task("a-not-retreated"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.RECOVERY_EXHAUSTED)
        self.assertTrue(simulator.safe_stop_called)
        self.assertEqual(pi05.plan_calls, 2)
        self.assertEqual(openvla.plan_calls, 0)
        self.assertEqual(simulator.illegal_arm_b_attempts, 0)
        self.assertNotIn("B_ONLY", result.control_token_history)
        self.assertFalse(
            any(event.event_type == "handoff_ready" for event in result.events)
        )

    def test_contradictory_arm_retreat_evidence_safe_stops_before_arm_b(self) -> None:
        class ContradictoryRetreatSimulator(FixedDualArmMockSimulator):
            def _observation(self) -> dict[str, object]:
                observation = super()._observation()
                if self.bin_at_handoff and not self.safe_stop_called:
                    task_state = observation["task"]
                    robot_state = observation["robot"]
                    assert isinstance(task_state, dict)
                    assert isinstance(robot_state, dict)
                    arm_a_state = robot_state["arm_a"]
                    assert isinstance(arm_a_state, dict)
                    task_state["arm_a_retreated"] = True
                    robot_state["active_arm"] = "Arm_A"
                    arm_a_state["retreated"] = False
                return observation

        agent, openvla, _ = self.make_agent(make_perception())
        simulator = ContradictoryRetreatSimulator()

        result = agent.run(
            four_agent_task("contradictory-retreat"),
            simulator,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.SAFETY_REJECTED)
        self.assertTrue(simulator.safe_stop_called)
        self.assertEqual(openvla.plan_calls, 0)
        self.assertNotIn("B_ONLY", result.control_token_history)

    def test_integer_lifecycle_flags_cannot_vote_as_booleans(self) -> None:
        class IntegerLifecycleFlagSimulator(FixedDualArmMockSimulator):
            def _observation(self) -> dict[str, object]:
                observation = super()._observation()
                if self.bin_at_handoff:
                    task_state = observation["task"]
                    assert isinstance(task_state, dict)
                    task_state["bin_at_handoff"] = 1
                return observation

        agent, openvla, _ = self.make_agent(make_perception())
        simulator = IntegerLifecycleFlagSimulator()

        result = agent.run(four_agent_task("integer-lifecycle-flag"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.SAFETY_REJECTED)
        self.assertTrue(simulator.safe_stop_called)
        self.assertEqual(result.control_token_history, ("A_ONLY", "NONE"))
        self.assertEqual(openvla.plan_calls, 0)

    def test_handoff_lockout_rejects_moving_arm_b(self) -> None:
        class MovingDuringHandoffSimulator(FixedDualArmMockSimulator):
            handoff_observations = 0

            def _observation(self) -> dict[str, object]:
                observation = super()._observation()
                if (
                    self.bin_at_handoff
                    and not self.bin_at_finished
                    and not self.safe_stop_called
                ):
                    self.handoff_observations += 1
                    if self.handoff_observations >= 2:
                        robot_state = observation["robot"]
                        task_state = observation["task"]
                        assert isinstance(robot_state, dict)
                        assert isinstance(task_state, dict)
                        arm_b_state = robot_state["arm_b"]
                        assert isinstance(arm_b_state, dict)
                        robot_state["active_arm"] = "Arm_B"
                        arm_b_state["retreated"] = False
                        task_state["arm_b_retreated"] = False
                return observation

        agent, openvla, _ = self.make_agent(make_perception())
        simulator = MovingDuringHandoffSimulator()

        result = agent.run(four_agent_task("moving-during-handoff"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.SAFETY_REJECTED)
        self.assertTrue(simulator.safe_stop_called)
        self.assertNotIn("B_ONLY", result.control_token_history)
        self.assertEqual(openvla.plan_calls, 0)

    def test_handoff_requires_composite_two_of_three_frames(self) -> None:
        class CrossShiftedHandoffSimulator(FixedDualArmMockSimulator):
            handoff_observations = 0

            def _observation(self) -> dict[str, object]:
                observation = super()._observation()
                if self.bin_at_handoff and not self.bin_at_finished:
                    self.handoff_observations += 1
                    task_state = observation["task"]
                    assert isinstance(task_state, dict)
                    if self.handoff_observations == 2:
                        task_state["bin_at_handoff"] = False
                    elif self.handoff_observations == 4:
                        task_state["packed_part_count"] = 0
                return observation

        agent, openvla, _ = self.make_agent(make_perception())
        simulator = CrossShiftedHandoffSimulator()

        result = agent.run(four_agent_task("cross-shifted-quorum"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertTrue(simulator.safe_stop_called)
        self.assertNotIn("B_ONLY", result.control_token_history)
        self.assertEqual(openvla.plan_calls, 0)

    def test_one_local_retry_does_not_switch_executor_role(self) -> None:
        agent, openvla, pi05 = self.make_agent(
            make_perception(),
            max_decisions=1,
        )
        simulator = FixedDualArmMockSimulator(arm_a_success_after=2)

        result = agent.run(four_agent_task("bounded-retry"), simulator)

        self.assertTrue(result.success)
        self.assertEqual(pi05.plan_calls, 2)
        self.assertEqual(openvla.plan_calls, 1)
        self.assertEqual(result.replan_counts["pi05"], 1)
        self.assertEqual(result.executor_history, ("pi05", "openvla_oft"))

    def test_retry_exhaustion_never_substitutes_openvla_for_pi05(self) -> None:
        agent, openvla, pi05 = self.make_agent(
            make_perception(),
            max_decisions=1,
        )
        result = agent.run(
            four_agent_task("retry-exhausted"),
            FixedDualArmMockSimulator(arm_a_success_after=99),
        )

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.RECOVERY_EXHAUSTED)
        self.assertEqual(result.control_token_history[-1], "NONE")
        self.assertTrue(result.events)
        self.assertTrue(
            any(event.event_type == "run.safe_stopped" for event in result.events)
        )
        self.assertEqual(pi05.plan_calls, 2)
        self.assertEqual(openvla.plan_calls, 0)
        exhausted = next(
            event
            for event in result.events
            if event.event_type == "recovery.phase_exhausted"
        )
        self.assertFalse(exhausted.payload["switch_allowed"])

    def test_illegal_action_contract_triggers_immediate_safe_stop(self) -> None:
        class InvalidPi05(RecordingExecutor):
            def plan(
                self,
                task: TaskSchema,
                observation: Observation,
                context: ExecutionContext,
            ) -> ActionChunk:
                self.plan_calls += 1
                return ActionChunk(
                    contract_version=ACTION_CONTRACT_VERSION,
                    chunk_id="invalid-owner",
                    task_id="wrong-task-id",
                    executor=self.descriptor.name,
                    steps=(ActionStep.from_sequence([0.02, 0, 0, 0, 0, 0, 0.5]),),
                )

        invalid_pi05 = InvalidPi05("pi05", 0.02)
        agent, openvla, _ = self.make_agent(
            make_perception(),
            pi05=invalid_pi05,
        )
        simulator = FixedDualArmMockSimulator()

        result = agent.run(four_agent_task("unsafe-action"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(
            result.failure_code,
            FailureCode.ACTION_CONTRACT_INVALID,
        )
        self.assertTrue(simulator.safe_stop_called)
        self.assertEqual(openvla.plan_calls, 0)
        self.assertEqual(
            result.control_token_history,
            ("A_ONLY", "NONE"),
        )

    def test_environment_token_rejection_immediately_safe_stops(self) -> None:
        class RejectingEnvironment(FixedDualArmMockSimulator):
            def step(
                self,
                action: ActionStep,
                *,
                arm_id: str,
                control_token: str,
                command_id: str,
                expected_observation_id: str,
                expected_state_digest: str,
            ) -> dict[str, object]:
                del (
                    action,
                    arm_id,
                    control_token,
                    command_id,
                    expected_observation_id,
                    expected_state_digest,
                )
                raise AgentError(
                    FailureCode.SAFETY_REJECTED,
                    "controller rejected arm/token authorization",
                )

        agent, openvla, pi05 = self.make_agent(make_perception())
        simulator = RejectingEnvironment()

        result = agent.run(four_agent_task("controller-rejection"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.SYSTEM_FAULT)
        self.assertIn("execution outcome is unknown", result.message)
        self.assertTrue(simulator.safe_stop_called)
        self.assertEqual(result.control_token_history, ("A_ONLY", "NONE"))
        self.assertEqual(pi05.plan_calls, 1)
        self.assertEqual(openvla.plan_calls, 0)

    def test_safe_stop_transport_failure_is_not_reported_as_stopped(self) -> None:
        class UnstoppableEnvironment(FixedDualArmMockSimulator):
            def step(
                self,
                action: ActionStep,
                *,
                arm_id: str,
                control_token: str,
                command_id: str,
                expected_observation_id: str,
                expected_state_digest: str,
            ) -> dict[str, object]:
                del (
                    action,
                    arm_id,
                    control_token,
                    command_id,
                    expected_observation_id,
                    expected_state_digest,
                )
                raise AgentError(
                    FailureCode.SAFETY_REJECTED,
                    "controller authorization rejected",
                )

            def safe_stop(self, reason: str) -> None:
                del reason
                raise OSError("e-stop channel unavailable")

        agent, _, _ = self.make_agent(make_perception())
        simulator = UnstoppableEnvironment()

        result = agent.run(four_agent_task("safe-stop-failed"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOP_FAILED)
        self.assertEqual(result.failure_code, FailureCode.SYSTEM_FAULT)
        self.assertIn("physical safe-stop confirmation failed", result.message)
        self.assertEqual(result.control_token_history, ("A_ONLY", "NONE"))

    def test_handoff_event_must_persist_before_b_only_is_granted(self) -> None:
        class FailingHandoffEventSink(EventSink):
            def emit(self, **kwargs: Any):
                if kwargs.get("event_type") == "handoff_ready":
                    raise OSError("event store unavailable")
                return super().emit(**kwargs)

        agent, openvla, _ = self.make_agent(
            make_perception(),
            events=FailingHandoffEventSink(),
        )
        simulator = FixedDualArmMockSimulator()

        result = agent.run(four_agent_task("handoff-persistence-failure"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.SYSTEM_FAULT)
        self.assertTrue(simulator.safe_stop_called)
        self.assertNotIn("B_ONLY", result.control_token_history)
        self.assertEqual(result.control_token_history[-1], "NONE")
        self.assertEqual(openvla.plan_calls, 0)

    def test_cancel_failure_after_motion_fails_closed(self) -> None:
        class CancelFailurePi05(RecordingExecutor):
            def cancel(self, task_id: str, reason: str) -> None:
                del task_id, reason
                raise OSError("cancel transport unavailable")

        pi05 = CancelFailurePi05("pi05", 0.02)
        agent, openvla, _ = self.make_agent(
            make_perception(),
            max_decisions=1,
            pi05=pi05,
        )
        simulator = FixedDualArmMockSimulator(arm_a_success_after=99)

        result = agent.run(four_agent_task("cancel-failure"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.SYSTEM_FAULT)
        self.assertTrue(simulator.safe_stop_called)
        self.assertEqual(simulator.arm_a_steps, 1)
        self.assertEqual(result.control_token_history[-1], "NONE")
        self.assertEqual(openvla.plan_calls, 0)

    def test_yolo_uses_same_frames_but_never_enters_vla_input(self) -> None:
        perception = make_perception()
        agent, openvla, pi05 = self.make_agent(perception)
        result = agent.run(
            four_agent_task("same-frame"),
            FixedDualArmMockSimulator(),
        )

        self.assertTrue(result.success)
        self.assertEqual(len(perception.calls), 2)
        detected_observation_ids = [item[1] for item in perception.calls]
        self.assertEqual(
            detected_observation_ids,
            pi05.observation_ids + openvla.observation_ids,
        )
        completed = [
            event
            for event in result.events
            if event.event_type == "perception.completed"
        ]
        self.assertEqual(len(completed), 2)
        self.assertTrue(
            all(event.payload["control_path_impact"] == "none" for event in completed)
        )
        self.assertFalse(
            any(event.event_type == "vla.perception_bound" for event in result.events)
        )

    def test_context_only_yolo_sidecar_cannot_change_vla_observation(self) -> None:
        class ContextOnlyPerception(MockPerceptionAgent):
            seen_context_fields: list[set[str]] = []

            def detect(
                self,
                context: PerceptionContext,
            ) -> DetectionPacket:
                fields = set(context.__dict__)
                assert "task" not in fields
                assert "observation" not in fields
                self.seen_context_fields.append(fields)
                return super().detect(context)

        class InspectingPi05(RecordingExecutor):
            seen_state_zero: list[float] = []
            seen_image_keys: list[set[str]] = []

            def plan(
                self,
                task: TaskSchema,
                observation: Observation,
                context: ExecutionContext,
            ) -> ActionChunk:
                robot = observation.data["robot"]
                camera = observation.data["camera"]
                assert isinstance(robot, Mapping)
                assert isinstance(camera, Mapping)
                arm_a = robot["arm_a"]
                full_image = camera["arm_a_rgb"]
                assert isinstance(arm_a, Mapping)
                assert isinstance(full_image, Mapping)
                self.seen_state_zero.append(float(arm_a["state"][0]))
                self.seen_image_keys.append(set(full_image))
                return super().plan(task, observation, context)

        perception = ContextOnlyPerception(
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
            detector=target_detection,
        )
        pi05 = InspectingPi05("pi05", 0.02)
        agent, _, _ = self.make_agent(perception, pi05=pi05)

        result = agent.run(
            four_agent_task("sidecar-isolation"),
            FixedDualArmMockSimulator(),
        )

        self.assertTrue(result.success)
        self.assertEqual(len(perception.seen_context_fields), 2)
        self.assertTrue(
            all("image" in fields for fields in perception.seen_context_fields)
        )
        self.assertEqual(pi05.seen_state_zero, [0.5])
        self.assertTrue(all("detections" not in keys for keys in pi05.seen_image_keys))

    def test_nested_detection_payload_is_rejected_before_vla(self) -> None:
        class PollutedImageSimulator(FixedDualArmMockSimulator):
            def _observation(self) -> dict[str, object]:
                observation = super()._observation()
                camera = observation["camera"]
                assert isinstance(camera, dict)
                image = camera["full_image"]
                assert isinstance(image, dict)
                image["detections"] = [{"bbox": [1, 2, 3, 4]}]
                return observation

        agent, openvla, pi05 = self.make_agent(make_perception())
        simulator = PollutedImageSimulator()

        result = agent.run(four_agent_task("polluted-image"), simulator)

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertTrue(simulator.safe_stop_called)
        self.assertEqual(pi05.plan_calls, 0)
        self.assertEqual(openvla.plan_calls, 0)

    def test_yolo_uses_phase_camera_and_full_class_map(self) -> None:
        captured: list[tuple[str, str, str, tuple[str, ...]]] = []

        def capture_phase(
            context: PerceptionContext,
        ) -> tuple[Detection, ...]:
            captured.append(
                (
                    context.subtask_id,
                    context.image.camera_id,
                    context.image.image_sha256,
                    context.allowed_class_names,
                )
            )
            return target_detection(context)

        class DistinctPhaseCameraSimulator(FixedDualArmMockSimulator):
            def _observation(self) -> dict[str, object]:
                observation = super()._observation()
                camera = observation["camera"]
                assert isinstance(camera, dict)
                for key, camera_id in (
                    ("arm_a_rgb", "CAM_A_TOP"),
                    ("handoff_rgb", "CAM_HANDOFF"),
                    ("arm_b_rgb", "CAM_B_TOP"),
                ):
                    digest = hashlib.sha256(
                        f"{key}:{self._observation_counter}".encode()
                    ).hexdigest()
                    camera[key] = {
                        "uri": f"cas://sha256/{digest}",
                        "image_sha256": f"sha256:{digest}",
                        "camera_id": camera_id,
                        "width": 640,
                        "height": 480,
                    }
                return observation

        perception = MockPerceptionAgent(
            checkpoint_sha=CHECKPOINT_SHA,
            class_map_sha=CLASS_MAP_SHA,
            config_sha=CONFIG_SHA,
            detector=capture_phase,
        )
        agent, _, _ = self.make_agent(perception)

        result = agent.run(
            four_agent_task("phase-camera"),
            DistinctPhaseCameraSimulator(),
        )

        self.assertTrue(result.success)
        self.assertEqual(
            [(item[0], item[1]) for item in captured],
            [
                ("S01_ARM_A_PACK_HANDOFF", "CAM_A_TOP"),
                ("S02_ARM_B_TRANSPORT", "CAM_B_TOP"),
            ],
        )
        self.assertTrue(all(item[3] == () for item in captured))

    def test_repeated_handoff_image_sha_cannot_form_quorum(self) -> None:
        class RepeatedHandoffImageSimulator(FixedDualArmMockSimulator):
            def _observation(self) -> dict[str, object]:
                observation = super()._observation()
                camera = observation["camera"]
                assert isinstance(camera, dict)
                camera["handoff_rgb"] = {
                    "uri": f"cas://sha256/{'f' * 64}",
                    "image_sha256": f"sha256:{'f' * 64}",
                    "camera_id": "CAM_HANDOFF",
                    "width": 640,
                    "height": 480,
                }
                return observation

        agent, openvla, _ = self.make_agent(make_perception())
        simulator = RepeatedHandoffImageSimulator()

        result = agent.run(
            four_agent_task("repeated-handoff-frame"),
            simulator,
        )

        self.assertFalse(result.success)
        self.assertEqual(result.state, AgentState.SAFE_STOPPED)
        self.assertEqual(result.failure_code, FailureCode.OBSERVATION_INVALID)
        self.assertTrue(simulator.safe_stop_called)
        self.assertEqual(openvla.plan_calls, 0)

    def test_pi05_gets_original_language_openvla_gets_fixed_handoff_text(
        self,
    ) -> None:
        task = four_agent_task("language-owner")
        agent, openvla, pi05 = self.make_agent(make_perception())
        result = agent.run(task, FixedDualArmMockSimulator())

        self.assertTrue(result.success)
        self.assertEqual(pi05.original_instructions, [task.instruction])
        self.assertEqual(len(openvla.original_instructions), 1)
        self.assertEqual(
            openvla.original_instructions,
            [FixedTaskProfile().arm_b_instruction],
        )
        self.assertNotEqual(
            openvla.original_instructions[0],
            task.instruction,
        )

    def test_fixed_task_rejects_non_two_vote_postcondition(self) -> None:
        task = four_agent_task("invalid-final-quorum")
        task = TaskSchema(
            task_id=task.task_id,
            instruction=task.instruction,
            task_type=task.task_type,
            target_object=task.target_object,
            target_location=task.target_location,
            postconditions=(
                Postcondition(
                    kind="field_equals",
                    path="task.status",
                    expected="done",
                    required_votes=1,
                ),
            ),
        )
        agent, openvla, pi05 = self.make_agent(make_perception())

        result = agent.run(task, FixedDualArmMockSimulator())

        self.assertFalse(result.success)
        self.assertEqual(result.failure_code, FailureCode.INVALID_TASK)
        self.assertEqual(openvla.plan_calls, 0)
        self.assertEqual(pi05.plan_calls, 0)

    def test_yolo_timeout_is_logged_and_does_not_change_control_path(
        self,
    ) -> None:
        perception = FlakyPerception([True, False])
        evidence = DetectionEvidenceSink()
        agent, openvla, pi05 = self.make_agent(
            perception,
            evidence=evidence,
        )
        result = agent.run(
            four_agent_task("yolo-timeout"),
            FixedDualArmMockSimulator(),
        )

        self.assertTrue(result.success)
        self.assertEqual(pi05.plan_calls, 1)
        self.assertEqual(openvla.plan_calls, 1)
        self.assertEqual(evidence.records[0]["record_type"], "detection_failure")
        failed = next(
            event for event in result.events if event.event_type == "perception.failed"
        )
        self.assertTrue(failed.payload["vla_recovery_untouched"])

    def test_detection_evidence_exports_map_input(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            jsonl_path = f"{temporary_directory}/yolo-evidence.jsonl"
            export_path = f"{temporary_directory}/raw-predictions.json"
            evidence = DetectionEvidenceSink(jsonl_path)
            agent, _, _ = self.make_agent(
                make_perception(),
                evidence=evidence,
            )
            result = agent.run(
                four_agent_task("map-evidence"),
                FixedDualArmMockSimulator(),
            )
            self.assertTrue(result.success)

            with open(jsonl_path, encoding="utf-8") as handle:
                persisted = [json.loads(line) for line in handle if line.strip()]
            self.assertEqual(len(persisted), 2)
            self.assertTrue(
                all(item["perception_mode"] == "SHADOW_SCORE" for item in persisted)
            )
            self.assertEqual(
                persisted[0]["detections"][0]["bbox_format"],
                "xyxy_pixels",
            )
            manifest = CocoExportManifest(
                class_map_sha=CLASS_MAP_SHA,
                image_id_by_frame_key={
                    (
                        item["trace_id"],
                        item["observation_id"],
                        item["camera_id"],
                    ): index
                    for index, item in enumerate(persisted, start=1)
                },
                image_sha_by_frame_key={
                    (
                        item["trace_id"],
                        item["observation_id"],
                        item["camera_id"],
                    ): item["image_sha256"]
                    for item in persisted
                },
                category_id_by_class_name={"industrial_part": 7},
            )
            evidence.export_coco_predictions(
                export_path,
                manifest=manifest,
            )
            with open(export_path, encoding="utf-8") as handle:
                exported = json.load(handle)
            self.assertEqual(len(exported["predictions"]), 2)
            self.assertEqual(
                exported["predictions"][0]["bbox"],
                [100.0, 80.0, 140.0, 140.0],
            )
            self.assertEqual(exported["predictions"][0]["category_id"], 7)
            self.assertEqual(
                exported["predictions"][0]["trace_id"],
                persisted[0]["trace_id"],
            )
            self.assertEqual(
                exported["predictions"][0]["observation_id"],
                persisted[0]["observation_id"],
            )
            self.assertEqual(
                exported["predictions"][0]["image_sha256"],
                persisted[0]["image_sha256"],
            )


class CapturingFailTransport:
    def __init__(self) -> None:
        self.calls: list[tuple[str, Mapping[str, Any], int]] = []

    def request(
        self,
        route: str,
        payload: Mapping[str, Any],
        timeout_ms: int,
    ) -> Mapping[str, Any]:
        self.calls.append((route, payload, timeout_ms))
        raise ConnectionError("capture request only")


class VLAAdapterInputTests(unittest.TestCase):
    def test_both_vla_adapters_receive_language_image_and_state_not_yolo(
        self,
    ) -> None:
        raw_observation = FixedDualArmMockSimulator().observe()
        camera = raw_observation["camera"]
        assert isinstance(camera, dict)
        camera["arm_a_rgb"] = {
            "uri": f"cas://sha256/{'a' * 64}",
            "image_sha256": f"sha256:{'a' * 64}",
            "camera_id": "CAM_A_TOP",
            "width": 640,
            "height": 480,
        }
        camera["arm_b_rgb"] = {
            "uri": f"cas://sha256/{'b' * 64}",
            "image_sha256": f"sha256:{'b' * 64}",
            "camera_id": "CAM_B_TOP",
            "width": 640,
            "height": 480,
        }
        observation = ObservationGateway().ingest_online(raw_observation)
        context = ExecutionContext(
            run_id="run-1",
            strategy_attempt=1,
            replan_index=0,
            original_instruction="stage-owned instruction",
        )
        task = TaskSchema(
            task_id="adapter-input",
            instruction="planner subtask text",
            task_type="pick_place",
            postconditions=(
                Postcondition(
                    kind="field_equals",
                    path="task.status",
                    expected="done",
                    required_votes=1,
                ),
            ),
        )
        adapter_cases = (
            OpenVLAOFTAdapter,
            Pi05Adapter,
        )
        for adapter_type in adapter_cases:
            with self.subTest(adapter=adapter_type.__name__):
                transport = CapturingFailTransport()
                adapter = adapter_type(
                    transport,
                    checkpoint_sha=CHECKPOINT_SHA,
                    norm_stats_sha=NORM_STATS_SHA,
                )
                with self.assertRaises(ExecutorError):
                    adapter.plan(task, observation, context)
                route, payload, _ = transport.calls[-1]
                self.assertEqual(route, "/v1/infer")
                model_input = payload["model_input"]
                assert isinstance(model_input, Mapping)
                self.assertNotIn("perception_context", model_input)
                self.assertNotIn("detections", model_input)
                if adapter_type is OpenVLAOFTAdapter:
                    self.assertEqual(
                        model_input["task_description"],
                        "stage-owned instruction",
                    )
                    self.assertEqual(
                        model_input["full_image"],
                        observation.data["camera"]["arm_b_rgb"],
                    )
                    self.assertEqual(
                        model_input["state"],
                        observation.data["robot"]["arm_b"]["state"],
                    )
                else:
                    self.assertEqual(
                        model_input["prompt"],
                        "stage-owned instruction",
                    )
                    pi_observation = model_input["observation"]
                    assert isinstance(pi_observation, Mapping)
                    self.assertEqual(
                        set(pi_observation),
                        {"camera", "robot"},
                    )
                    self.assertTrue(
                        {
                            "objects",
                            "task",
                            "quality",
                            "safety",
                        }.isdisjoint(pi_observation)
                    )
                    self.assertEqual(
                        pi_observation["camera"]["full_image"],
                        observation.data["camera"]["arm_a_rgb"],
                    )
                    self.assertEqual(
                        pi_observation["robot"]["state"],
                        observation.data["robot"]["arm_a"]["state"],
                    )


if __name__ == "__main__":
    unittest.main()
