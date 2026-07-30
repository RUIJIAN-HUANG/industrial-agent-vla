from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event, Thread, get_ident
from time import monotonic, sleep
import unittest

from industrial_agent.isaac_runtime import (
    IsaacGateClosedError,
    IsaacGateError,
    IsaacGateTimeoutError,
    IsaacMainThreadGate,
)
from industrial_agent.contracts import ActionStep
from industrial_agent.environment import SafeStopReceipt, execution_guard_digest
from industrial_agent.isaac_environment import IsaacExecutionEnvironment


def _start_call(target):
    result = {}

    def worker():
        try:
            result["value"] = target()
        except BaseException as exc:
            result["error"] = exc

    thread = Thread(target=worker)
    thread.start()
    return thread, result


class IsaacMainThreadGateTests(unittest.TestCase):
    def test_owner_thread_normal_call_is_rejected_instead_of_bypassing_deadline(self):
        gate = IsaacMainThreadGate()

        with self.assertRaisesRegex(IsaacGateError, "worker thread"):
            gate.call(lambda: "unsafe-inline", timeout_s=0.01, label="inline")

    def test_worker_operation_runs_only_on_owner_thread(self):
        gate = IsaacMainThreadGate()
        owner = get_ident()
        thread, result = _start_call(
            lambda: gate.call(
                get_ident,
                timeout_s=0.5,
                label="owner-check",
            )
        )
        while thread.is_alive():
            gate.pump()
        thread.join()
        self.assertEqual(result["value"], owner)

    def test_missing_pump_times_out_and_late_pump_does_not_execute(self):
        gate = IsaacMainThreadGate()
        executed = Event()
        thread, result = _start_call(
            lambda: gate.call(
                executed.set,
                timeout_s=0.03,
                label="unserviced",
            )
        )
        thread.join(0.2)
        self.assertIsInstance(result["error"], IsaacGateTimeoutError)
        self.assertFalse(result["error"].started)
        gate.pump()
        self.assertFalse(executed.is_set())

    def test_stop_signal_is_immediate_and_urgent_apply_precedes_queued_normal(self):
        gate = IsaacMainThreadGate(max_pending_normal=1)
        signal = Event()
        order = []
        normal_thread, normal_result = _start_call(
            lambda: gate.call(
                lambda: order.append("normal"),
                timeout_s=0.5,
                label="normal",
            )
        )
        stop_thread, stop_result = _start_call(
            lambda: gate.call_stop(
                signal_stop=signal.set,
                apply_stop=lambda: order.append("stop"),
                timeout_s=0.5,
            )
        )
        self.assertTrue(signal.wait(0.1))
        gate.pump()
        normal_thread.join()
        stop_thread.join()
        self.assertEqual(order, ["stop"])
        self.assertIsInstance(normal_result["error"], IsaacGateError)
        self.assertNotIn("error", stop_result)

    def test_active_action_observes_signal_then_urgent_stop_is_applied(self):
        gate = IsaacMainThreadGate()
        cancel = Event()
        action_started = Event()
        order = []

        def action():
            action_started.set()
            deadline = monotonic() + 0.5
            while not cancel.is_set() and monotonic() < deadline:
                sleep(0.001)
            if cancel.is_set():
                order.append("action-cancelled")

        action_thread, action_result = _start_call(
            lambda: gate.call(action, timeout_s=0.8, label="action")
        )

        def stop_call():
            action_started.wait(0.5)
            return gate.call_stop(
                signal_stop=cancel.set,
                apply_stop=lambda: order.append("stop-applied"),
                timeout_s=0.5,
            )

        stop_thread, stop_result = _start_call(stop_call)
        while action_thread.is_alive() or stop_thread.is_alive():
            gate.pump()
            sleep(0.001)
        action_thread.join()
        stop_thread.join()
        self.assertNotIn("error", action_result)
        self.assertNotIn("error", stop_result)
        self.assertEqual(order, ["action-cancelled", "stop-applied"])

    def test_started_timeout_invokes_thread_safe_cancellation_callback(self):
        gate = IsaacMainThreadGate()
        cancel = Event()
        started = Event()

        def operation():
            started.set()
            while not cancel.is_set():
                sleep(0.001)

        thread, result = _start_call(
            lambda: gate.call(
                operation,
                timeout_s=0.03,
                label="started-timeout",
                on_started_timeout=cancel.set,
            )
        )
        gate.pump()
        thread.join(0.2)
        self.assertTrue(started.is_set())
        self.assertTrue(cancel.is_set())
        self.assertIsInstance(result["error"], IsaacGateTimeoutError)
        self.assertTrue(result["error"].started)

    def test_post_stop_observation_requests_remain_allowed(self):
        gate = IsaacMainThreadGate()
        gate.call_stop(
            signal_stop=lambda: None,
            apply_stop=lambda: None,
            timeout_s=0.1,
        )
        thread, result = _start_call(
            lambda: gate.call(
                lambda: "observation",
                timeout_s=0.1,
                label="observe",
            )
        )
        while thread.is_alive():
            gate.pump()
        thread.join()
        self.assertEqual(result["value"], "observation")

    def test_stop_ack_timeout_keeps_late_owner_hold_queued(self):
        gate = IsaacMainThreadGate()
        applied = Event()
        thread, result = _start_call(
            lambda: gate.call_stop(
                signal_stop=lambda: None,
                apply_stop=applied.set,
                timeout_s=0.02,
            )
        )
        thread.join(0.2)
        self.assertIsInstance(result["error"], IsaacGateTimeoutError)
        self.assertFalse(applied.is_set())
        gate.pump(max_normal=0)
        self.assertTrue(applied.is_set())

    def test_close_wakes_queued_waiter(self):
        gate = IsaacMainThreadGate()
        thread, result = _start_call(
            lambda: gate.call(lambda: None, timeout_s=1.0, label="queued")
        )
        sleep(0.01)
        gate.close("shutdown")
        thread.join(0.2)
        self.assertIsInstance(result["error"], IsaacGateClosedError)

    def test_operation_exception_is_returned_to_worker(self):
        gate = IsaacMainThreadGate()

        def fail():
            raise ValueError("backend failure")

        thread, result = _start_call(
            lambda: gate.call(fail, timeout_s=0.5, label="failure")
        )
        while thread.is_alive():
            gate.pump()
        thread.join()
        self.assertIsInstance(result["error"], ValueError)

    def test_run_worker_pumps_nested_owner_requests(self):
        gate = IsaacMainThreadGate()
        owner = get_ident()
        result = gate.run_worker_until_complete(
            lambda: gate.call(
                get_ident,
                timeout_s=0.5,
                label="nested",
            )
        )
        self.assertEqual(result, owner)

    def test_run_worker_drains_late_urgent_stop_before_returning(self):
        gate = IsaacMainThreadGate()
        stop_signalled = Event()
        hold_applied = Event()
        owner_idle_started = Event()

        def workflow():
            self.assertTrue(owner_idle_started.wait(0.2))
            with self.assertRaises(IsaacGateTimeoutError):
                gate.call_stop(
                    signal_stop=stop_signalled.set,
                    apply_stop=lambda: hold_applied.set(),
                    timeout_s=0.01,
                )
            return "worker-finished"

        # Keep the owner out of pump() until the urgent caller has timed out.
        result = gate.run_worker_until_complete(
            workflow,
            poll_interval_s=0.001,
            idle_callback=lambda: (owner_idle_started.set(), sleep(0.03)),
        )

        self.assertEqual(result, "worker-finished")
        self.assertTrue(stop_signalled.is_set())
        self.assertTrue(hold_applied.is_set())

    def test_environment_marshals_observe_guard_action_and_stop_to_owner(self):
        gate = IsaacMainThreadGate()
        owner = get_ident()
        source_threads = []
        controller_threads = []
        signal_threads = []
        counter = 0

        def guarded_state():
            source_threads.append(("guard", get_ident()))
            arm = {
                "tcp_pose_m_rad": [0.4, 0.0, 0.5, 0.0, 0.0, 0.0],
                "state": [0.0] * 9,
                "retreated": True,
                "gripper_open": True,
                "stationary": True,
            }
            return {
                "objects": [],
                "robot": {
                    "active_arm": "NONE",
                    "arm_a": deepcopy(arm),
                    "arm_b": deepcopy(arm),
                },
                "safety": {
                    "emergency_stop": False,
                    "protective_stop": False,
                    "system_fault": None,
                },
                "task": {
                    "packed_part_count": 0,
                    "bin_at_handoff": False,
                    "bin_at_finished": False,
                    "bin_speed_m_s": 0.0,
                },
                "quality": {"confidence": 1.0},
            }

        def observation():
            nonlocal counter
            source_threads.append(("observation", get_ident()))
            counter += 1
            return {
                "observation_id": f"obs-{counter}",
                "timestamp_ms": counter,
                **guarded_state(),
            }

        class Controller:
            def validate_ready(self, arm_id):
                controller_threads.append(("validate", get_ident()))

            def execute_action(self, action, *, arm_id):
                controller_threads.append(("execute", get_ident()))

            def request_stop(self, reason):
                signal_threads.append(get_ident())
                return "stop-1"

            def confirm_safe_stop(self, reason, *, stop_epoch):
                controller_threads.append(("stop", get_ident()))
                return SafeStopReceipt(True, True, True, True, stop_epoch)

        with TemporaryDirectory() as directory:
            environment = IsaacExecutionEnvironment(
                observation_source=observation,
                state_guard_source=guarded_state,
                control_lease_source=lambda: "A_ONLY",
                controller=Controller(),
                runtime_gate=gate,
                command_ledger_path=Path(directory) / "commands.jsonl",
                runtime_observe_timeout_s=0.5,
                runtime_action_timeout_s=0.5,
                runtime_stop_timeout_s=0.5,
            )
            action = ActionStep.from_sequence([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

            def workflow():
                before = environment.observe()
                after = environment.step(
                    action,
                    arm_id="Arm_A",
                    control_token="A_ONLY",
                    command_id="command-1",
                    expected_observation_id=before["observation_id"],
                    expected_state_digest=execution_guard_digest(before),
                )
                receipt = environment.safe_stop("done")
                return after, receipt

            after, receipt = gate.run_worker_until_complete(workflow)

        self.assertEqual(after["observation_id"], "obs-2")
        self.assertTrue(receipt.confirmed)
        self.assertTrue(source_threads)
        self.assertTrue(controller_threads)
        self.assertTrue(all(thread_id == owner for _, thread_id in source_threads))
        self.assertTrue(all(thread_id == owner for _, thread_id in controller_threads))
        self.assertTrue(signal_threads)
        self.assertTrue(any(thread_id != owner for thread_id in signal_threads))


if __name__ == "__main__":
    unittest.main()
