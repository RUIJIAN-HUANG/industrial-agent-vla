from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread
import unittest

from industrial_agent.contracts import ActionStep
from industrial_agent.environment import (
    PreWriteStateStaleError,
    SafeStopReceipt,
    execution_guard_digest,
)
from industrial_agent.isaac_environment import (
    DurableCommandIdLedger,
    IsaacExecutionEnvironment,
)
from industrial_agent.observation import ObservationGateway


def _image_reference(camera_id: str, digest_character: str) -> dict:
    digest = digest_character * 64
    return {
        "uri": f"cas://sha256/{digest}",
        "image_sha256": f"sha256:{digest}",
        "camera_id": camera_id,
        "width": 1280,
        "height": 720,
    }


class _ObservationSource:
    def __init__(self) -> None:
        self.counter = 0
        self.active_arm = "NONE"
        self.arm_a_retreated = True
        self.arm_b_retreated = True
        self.protective_stop = False
        self.tcp_offset = 0.0
        self.confidence = 1.0
        self.packed_part_count = 0
        self.next_observation_id: str | None = None
        self.next_timestamp_ms: int | None = None

    def _guarded_state(self) -> dict:
        arm_state = {
            "tcp_pose_m_rad": [
                0.4 + self.tcp_offset,
                0.0,
                0.5,
                0.0,
                0.0,
                0.0,
            ],
            "state": [self.tcp_offset] + [0.0] * 8,
            "gripper_open": True,
            "stationary": True,
        }
        return {
            "objects": [],
            "robot": {
                "active_arm": self.active_arm,
                "arm_a": {**arm_state, "retreated": self.arm_a_retreated},
                "arm_b": {**arm_state, "retreated": self.arm_b_retreated},
            },
            "safety": {
                "emergency_stop": False,
                "protective_stop": self.protective_stop,
                "system_fault": None,
            },
            "task": {
                "packed_part_count": self.packed_part_count,
                "bin_at_handoff": False,
                "bin_at_finished": False,
                "bin_speed_m_s": 0.0,
            },
            "quality": {"confidence": self.confidence},
        }

    def guard(self) -> dict:
        return deepcopy(self._guarded_state())

    def __call__(self) -> dict:
        self.counter += 1
        observation_id = self.next_observation_id or f"obs-{self.counter}"
        timestamp_ms = (
            self.next_timestamp_ms
            if self.next_timestamp_ms is not None
            else self.counter
        )
        self.next_observation_id = None
        self.next_timestamp_ms = None
        return {
            "observation_version": "1.0",
            "observation_id": observation_id,
            "timestamp_ms": timestamp_ms,
            "camera": {
                "full_image": _image_reference("CAM_A_TOP", "a"),
                "arm_a_rgb": _image_reference("CAM_A_TOP", "a"),
                "handoff_rgb": _image_reference("CAM_HANDOFF", "b"),
                "arm_b_rgb": _image_reference("CAM_B_TOP", "c"),
                "wrist_image": None,
            },
            **self._guarded_state(),
        }


class _InlineGate:
    """Test-only gate for environment logic that does not touch Isaac."""

    @staticmethod
    def call(
        operation,
        *,
        timeout_s,
        label,
        on_started_timeout=None,
    ):
        del timeout_s, label, on_started_timeout
        return operation()

    @staticmethod
    def call_stop(
        *,
        signal_stop,
        apply_stop,
        timeout_s,
        label="Isaac safe-stop",
    ):
        del timeout_s, label
        signal_stop()
        return apply_stop()


class _Controller:
    def __init__(self) -> None:
        self.ready_calls = []
        self.actions = []
        self.stop_reasons = []
        self.fail_execution = False
        self.block_execution = False
        self.execution_entered = Event()
        self.release_execution = Event()
        self._stop_epoch = "stop-0"

    def validate_ready(self, arm_id):
        self.ready_calls.append(arm_id)

    def execute_action(self, action, *, arm_id):
        self.actions.append((arm_id, action))
        self.execution_entered.set()
        if self.fail_execution:
            raise RuntimeError("controller write failed")
        if self.block_execution and not self.release_execution.wait(timeout=2.0):
            raise RuntimeError("test controller remained blocked")

    def request_stop(self, reason):
        if self._stop_epoch == "stop-0":
            self._stop_epoch = "stop-1"
            self.stop_reasons.append(reason)
        return self._stop_epoch

    def confirm_safe_stop(self, reason, *, stop_epoch):
        del reason
        return SafeStopReceipt(True, True, True, True, stop_epoch)


class IsaacExecutionEnvironmentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.ledger_path = Path(self.temporary_directory.name) / "commands.jsonl"
        self.source = _ObservationSource()
        self.controller = _Controller()
        self.control_token = "A_ONLY"
        self.environment = self._environment(self.source, self.controller)
        self.action = ActionStep.from_sequence([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5])

    def _environment(self, source, controller, *, ledger_path=None):
        return IsaacExecutionEnvironment(
            observation_source=source,
            state_guard_source=source.guard,
            control_lease_source=lambda: self.control_token,
            controller=controller,
            runtime_gate=_InlineGate(),
            command_ledger_path=ledger_path or self.ledger_path,
        )

    def _arguments(self, observation, **overrides):
        arguments = {
            "arm_id": "Arm_A",
            "control_token": "A_ONLY",
            "command_id": "command-1",
            "expected_observation_id": observation["observation_id"],
            "expected_state_digest": execution_guard_digest(observation),
        }
        arguments.update(overrides)
        return arguments

    def _execute(self, observation, **overrides):
        return self.environment.step(
            self.action,
            **self._arguments(observation, **overrides),
        )

    def test_fixture_is_a_complete_frozen_online_observation(self):
        observation = self.source()
        ingested = ObservationGateway().ingest_online(observation)
        self.assertEqual(ingested.observation_id, "obs-1")

    def test_observe_owns_nested_snapshot_instead_of_aliasing_source_buffers(self):
        source_value = self.source()
        self.environment._observation_source = lambda: source_value

        observation = self.environment.observe()
        source_value["robot"]["arm_a"]["tcp_pose_m_rad"][0] = 99.0

        self.assertNotEqual(
            observation["robot"]["arm_a"]["tcp_pose_m_rad"][0],
            99.0,
        )

    def test_valid_action_is_atomically_checked_and_acknowledged(self):
        observation = self.environment.observe()
        result = self._execute(observation)
        self.assertEqual(self.controller.ready_calls, ["Arm_A", "Arm_A"])
        self.assertEqual(self.controller.actions, [("Arm_A", self.action)])
        self.assertNotEqual(result["observation_id"], observation["observation_id"])

        journal = self.ledger_path.read_text(encoding="utf-8")
        self.assertIn('"state":"CLAIMED"', journal)
        self.assertIn('"state":"APPLIED"', journal)
        self.assertIn('"state":"ACKED"', journal)

    def test_wrong_or_stale_token_is_rejected_before_controller_write(self):
        observation = self.environment.observe()
        with self.assertRaisesRegex(RuntimeError, "requires token A_ONLY"):
            self._execute(observation, control_token="B_ONLY")
        self.control_token = "B_ONLY"
        with self.assertRaisesRegex(RuntimeError, "stale control lease"):
            self._execute(observation)
        self.assertEqual(self.controller.actions, [])

    def test_active_arm_conflict_rejects_premature_arm_b(self):
        self.source.active_arm = "Arm_A"
        self.control_token = "B_ONLY"
        observation = self.environment.observe()
        with self.assertRaisesRegex(RuntimeError, "active_arm"):
            self.environment.step(
                self.action,
                **self._arguments(
                    observation,
                    arm_id="Arm_B",
                    control_token="B_ONLY",
                ),
            )
        self.assertEqual(self.controller.actions, [])
        self.assertTrue(self.controller.stop_reasons)

    def test_stale_observation_and_digest_are_rejected(self):
        observation = self.environment.observe()
        with self.assertRaisesRegex(RuntimeError, "stale action rejected"):
            self._execute(observation, expected_observation_id="older")
        changed = deepcopy(observation)
        changed["task"]["packed_part_count"] = 1
        with self.assertRaisesRegex(RuntimeError, "state digest changed"):
            self._execute(
                observation,
                expected_state_digest=execution_guard_digest(changed),
            )
        self.assertEqual(self.controller.actions, [])

    def test_live_safety_change_stops_before_controller_write(self):
        observation = self.environment.observe()
        self.source.protective_stop = True
        with self.assertRaisesRegex(RuntimeError, "protective stop"):
            self._execute(observation)
        self.assertEqual(self.controller.actions, [])
        self.assertTrue(self.controller.stop_reasons)

    def test_live_task_change_is_typed_prewrite_stale_without_safe_stop(self):
        observation = self.environment.observe()
        self.source.packed_part_count = 1

        with self.assertRaises(PreWriteStateStaleError):
            self._execute(observation)

        self.assertEqual(self.controller.actions, [])
        self.assertEqual(self.controller.stop_reasons, [])
        self.assertFalse(self.ledger_path.exists())

    def test_small_telemetry_noise_is_tolerated_but_large_motion_stops(self):
        observation = self.environment.observe()
        self.source.tcp_offset = 0.0005
        self.source.confidence = 0.99
        self._execute(observation)

        second_source = _ObservationSource()
        second_controller = _Controller()
        second_path = Path(self.temporary_directory.name) / "second.jsonl"
        second_environment = self._environment(
            second_source,
            second_controller,
            ledger_path=second_path,
        )
        second_observation = second_environment.observe()
        second_source.tcp_offset = 0.01
        with self.assertRaisesRegex(RuntimeError, "exceeded tolerance"):
            second_environment.step(
                self.action,
                **self._arguments(
                    second_observation,
                    command_id="large-motion",
                ),
            )
        self.assertEqual(second_controller.actions, [])
        self.assertTrue(second_controller.stop_reasons)

    def test_acked_duplicate_returns_original_result_without_reexecution(self):
        observation = self.environment.observe()
        result = self._execute(observation)
        repeated = self._execute(observation)
        self.assertEqual(repeated, result)
        self.assertEqual(len(self.controller.actions), 1)

        with self.assertRaisesRegex(RuntimeError, "reused for a different request"):
            self._execute(
                result,
                expected_observation_id=result["observation_id"],
                expected_state_digest=execution_guard_digest(result),
            )

    def test_ack_survives_restart_and_is_idempotently_recoverable(self):
        observation = self.environment.observe()
        result = self._execute(observation, command_id="persistent-command")

        restarted_source = _ObservationSource()
        restarted_controller = _Controller()
        restarted = self._environment(restarted_source, restarted_controller)
        recovered = restarted.step(
            self.action,
            **self._arguments(
                observation,
                command_id="persistent-command",
            ),
        )
        self.assertEqual(recovered, result)
        self.assertEqual(restarted_controller.actions, [])

    def test_unresolved_claim_on_restart_forces_stop_and_quarantine(self):
        ledger = DurableCommandIdLedger(self.ledger_path)
        ledger.claim("unknown-command", "sha256:" + "1" * 64)

        restarted_source = _ObservationSource()
        restarted_controller = _Controller()
        restarted = self._environment(restarted_source, restarted_controller)
        self.assertIsNotNone(restarted.startup_stop_receipt)
        self.assertTrue(restarted_controller.stop_reasons)
        observation = restarted.observe()
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            restarted.step(
                self.action,
                **self._arguments(
                    observation,
                    command_id="new-command",
                ),
            )

    def test_corrupt_command_ledger_fails_closed(self):
        corrupt_path = Path(self.temporary_directory.name) / "corrupt.jsonl"
        corrupt_path.write_text("not-json\n", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "cannot be loaded safely"):
            DurableCommandIdLedger(corrupt_path)

    def test_retreat_interlock_fails_closed(self):
        self.source.arm_b_retreated = False
        observation = self.environment.observe()
        with self.assertRaisesRegex(RuntimeError, "Arm_B retreated"):
            self._execute(observation)
        self.assertEqual(self.controller.actions, [])

    def test_controller_failure_stops_and_leaves_unknown_command_quarantined(self):
        observation = self.environment.observe()
        self.controller.fail_execution = True
        with self.assertRaisesRegex(RuntimeError, "controller write failed"):
            self._execute(observation)
        self.assertEqual(len(self.controller.stop_reasons), 1)
        with self.assertRaisesRegex(RuntimeError, "unknown physical outcome"):
            self._execute(observation)

    def test_same_post_action_observation_id_triggers_safe_stop(self):
        observation = self.environment.observe()
        self.source.next_observation_id = observation["observation_id"]
        with self.assertRaisesRegex(RuntimeError, "must differ"):
            self._execute(observation)
        self.assertEqual(len(self.controller.actions), 1)
        self.assertEqual(len(self.controller.stop_reasons), 1)

    def test_backwards_post_action_timestamp_triggers_safe_stop(self):
        observation = self.environment.observe()
        self.source.next_timestamp_ms = observation["timestamp_ms"] - 1
        with self.assertRaisesRegex(RuntimeError, "moved backwards"):
            self._execute(observation)
        self.assertEqual(len(self.controller.stop_reasons), 1)

    def test_hung_step_does_not_block_stop_and_late_completion_is_discarded(self):
        observation = self.environment.observe()
        self.controller.block_execution = True
        outcome: dict[str, BaseException] = {}

        def execute() -> None:
            try:
                self._execute(observation)
            except BaseException as exc:
                outcome["error"] = exc

        worker = Thread(target=execute)
        worker.start()
        self.assertTrue(self.controller.execution_entered.wait(timeout=0.5))

        receipt = self.environment.safe_stop("watchdog timeout")
        self.assertTrue(receipt.confirmed)
        self.assertEqual(self.controller.stop_reasons, ["watchdog timeout"])

        self.controller.release_execution.set()
        worker.join(timeout=0.5)
        self.assertFalse(worker.is_alive())
        self.assertRegex(str(outcome["error"]), "control lease revocation")
        self.assertEqual(self.source.counter, 1)

    def test_safe_stop_quarantines_actions_but_allows_post_stop_observation(self):
        observation = self.environment.observe()
        self.environment.safe_stop("operator stop")
        stopped_observation = self.environment.observe()
        self.assertNotEqual(
            stopped_observation["observation_id"],
            observation["observation_id"],
        )
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            self._execute(observation)
        self.assertEqual(self.controller.actions, [])

    def test_overlapping_step_is_rejected_without_waiting(self):
        observation = self.environment.observe()
        self.controller.block_execution = True
        first_done = Event()

        def first_step() -> None:
            try:
                self._execute(observation)
            except RuntimeError:
                pass
            finally:
                first_done.set()

        worker = Thread(target=first_step)
        worker.start()
        self.assertTrue(self.controller.execution_entered.wait(timeout=0.5))
        with self.assertRaisesRegex(RuntimeError, "overlapping step"):
            self._execute(observation, command_id="command-2")

        self.environment.safe_stop("test cleanup")
        self.controller.release_execution.set()
        self.assertTrue(first_done.wait(timeout=0.5))
        worker.join(timeout=0.5)

    def test_new_observation_during_claim_prevents_old_action_write(self):
        observation = self.environment.observe()
        claim_entered = Event()
        release_claim = Event()
        original_claim = self.environment._command_ledger.claim

        def blocking_claim(command_id, request_digest):
            original_claim(command_id, request_digest)
            claim_entered.set()
            release_claim.wait(timeout=1.0)

        self.environment._command_ledger.claim = blocking_claim
        outcome: dict[str, BaseException] = {}

        def execute() -> None:
            try:
                self._execute(observation)
            except BaseException as exc:
                outcome["error"] = exc

        worker = Thread(target=execute)
        worker.start()
        self.assertTrue(claim_entered.wait(timeout=0.5))
        self.environment.observe()
        release_claim.set()
        worker.join(timeout=0.5)
        self.assertFalse(worker.is_alive())
        self.assertRegex(str(outcome["error"]), "no longer latest")
        self.assertEqual(self.controller.actions, [])

    def test_task_change_after_claim_is_durably_aborted_and_retryable(self):
        observation = self.environment.observe()
        claim_entered = Event()
        release_claim = Event()
        original_claim = self.environment._command_ledger.claim

        def blocking_claim(command_id, request_digest):
            original_claim(command_id, request_digest)
            claim_entered.set()
            release_claim.wait(timeout=1.0)

        self.environment._command_ledger.claim = blocking_claim
        outcome: dict[str, BaseException] = {}

        def execute() -> None:
            try:
                self._execute(observation)
            except BaseException as exc:
                outcome["error"] = exc

        worker = Thread(target=execute)
        worker.start()
        self.assertTrue(claim_entered.wait(timeout=0.5))
        self.source.packed_part_count = 1
        release_claim.set()
        worker.join(timeout=0.5)

        self.assertFalse(worker.is_alive())
        self.assertIsInstance(outcome["error"], PreWriteStateStaleError)
        self.assertEqual(self.controller.actions, [])
        self.assertEqual(self.controller.stop_reasons, [])
        journal = self.ledger_path.read_text(encoding="utf-8")
        self.assertIn('"state":"CLAIMED"', journal)
        self.assertIn('"state":"ABORTED"', journal)
        self.assertNotIn('"state":"APPLIED"', journal)

    def test_stale_generation_after_controller_write_is_never_retryable(self):
        observation = self.environment.observe()
        self.controller.block_execution = True
        outcome: dict[str, BaseException] = {}

        def execute() -> None:
            try:
                self._execute(observation)
            except BaseException as exc:
                outcome["error"] = exc

        worker = Thread(target=execute)
        worker.start()
        self.assertTrue(self.controller.execution_entered.wait(timeout=0.5))
        self.environment.observe()
        self.controller.release_execution.set()
        worker.join(timeout=0.5)

        self.assertFalse(worker.is_alive())
        self.assertNotIsInstance(outcome["error"], PreWriteStateStaleError)
        self.assertEqual(len(self.controller.actions), 1)
        self.assertTrue(self.controller.stop_reasons)
        journal = self.ledger_path.read_text(encoding="utf-8")
        self.assertIn('"state":"APPLIED"', journal)
        self.assertNotIn('"state":"ACKED"', journal)

    def test_safety_change_during_claim_is_rechecked_before_write(self):
        observation = self.environment.observe()
        claim_entered = Event()
        release_claim = Event()
        original_claim = self.environment._command_ledger.claim

        def blocking_claim(command_id, request_digest):
            original_claim(command_id, request_digest)
            claim_entered.set()
            release_claim.wait(timeout=1.0)

        self.environment._command_ledger.claim = blocking_claim
        outcome: dict[str, BaseException] = {}

        def execute() -> None:
            try:
                self._execute(observation)
            except BaseException as exc:
                outcome["error"] = exc

        worker = Thread(target=execute)
        worker.start()
        self.assertTrue(claim_entered.wait(timeout=0.5))
        self.source.protective_stop = True
        release_claim.set()
        worker.join(timeout=0.5)
        self.assertFalse(worker.is_alive())
        self.assertRegex(str(outcome["error"]), "protective stop")
        self.assertEqual(self.controller.actions, [])


if __name__ == "__main__":
    unittest.main()
