"""Fail-closed execution adapter between the supervisor and Isaac Sim.

The supervisor deliberately depends on the small :class:`ExecutionEnvironment`
protocol.  This module supplies the production implementation while keeping
Isaac/Omniverse imports out of the supervisor process.  A concrete controller
backend is responsible for the actual Franka API calls.
"""

from __future__ import annotations

from threading import RLock
from typing import Any, Callable, Mapping, Protocol

from .contracts import ActionStep
from .environment import SafeStopReceipt, execution_guard_digest


class IsaacFrankaController(Protocol):
    """Controller operations that must be backed by the live Isaac runtime."""

    def validate_ready(self, arm_id: str) -> None:
        """Fail unless the requested arm can move and the other arm is stopped."""

    def execute_action(self, action: ActionStep, *, arm_id: str) -> None:
        """Send one validated action to the selected Franka controller."""

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        """Stop both articulations, clear commands, and report readback state."""


class IsaacExecutionEnvironment:
    """Production compare-and-execute boundary for the frozen dual-arm cell.

    ``observation_source`` must capture a complete raw online observation using
    real camera/CAS and robot telemetry.  It is intentionally injected so the
    adapter never manufactures image or task evidence.
    """

    _AUTHORIZATION = {
        "Arm_A": "A_ONLY",
        "Arm_B": "B_ONLY",
    }

    def __init__(
        self,
        *,
        observation_source: Callable[[], Mapping[str, Any]],
        controller: IsaacFrankaController,
    ) -> None:
        self._observation_source = observation_source
        self._controller = controller
        self._lock = RLock()
        self._last_observation: dict[str, Any] | None = None
        self._executed_command_ids: set[str] = set()

    def observe(self) -> Mapping[str, Any]:
        with self._lock:
            observation = dict(self._observation_source())
            observation_id = observation.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                raise RuntimeError(
                    "Isaac observation source returned no observation_id"
                )
            self._last_observation = observation
            return observation

    @staticmethod
    def _require_safe_observation(observation: Mapping[str, Any]) -> None:
        safety = observation.get("safety")
        if not isinstance(safety, Mapping):
            raise RuntimeError("Isaac observation contains no safety state")
        if safety.get("emergency_stop") is True:
            raise RuntimeError("controller boundary rejected active emergency stop")
        if safety.get("protective_stop") is True:
            raise RuntimeError("controller boundary rejected active protective stop")
        if safety.get("system_fault") not in (None, "", False):
            raise RuntimeError(
                "controller boundary rejected system fault: "
                f"{safety.get('system_fault')!r}"
            )

    @staticmethod
    def _require_retreat_interlock(observation: Mapping[str, Any], arm_id: str) -> None:
        robot = observation.get("robot")
        if not isinstance(robot, Mapping):
            raise RuntimeError("Isaac observation contains no robot state")
        other_key = "arm_b" if arm_id == "Arm_A" else "arm_a"
        other = robot.get(other_key)
        if not isinstance(other, Mapping) or other.get("retreated") is not True:
            other_arm = "Arm_B" if arm_id == "Arm_A" else "Arm_A"
            raise RuntimeError(
                f"{arm_id} boundary interlock requires {other_arm} retreated"
            )

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
        with self._lock:
            required_token = self._AUTHORIZATION.get(arm_id)
            if required_token is None:
                raise RuntimeError(
                    f"Isaac controller boundary rejected arm_id={arm_id!r}"
                )
            if control_token != required_token:
                raise RuntimeError(
                    f"{arm_id} requires token {required_token}, got {control_token!r}"
                )
            if not command_id or command_id in self._executed_command_ids:
                raise RuntimeError(
                    f"duplicate or empty controller command_id rejected: {command_id!r}"
                )
            if self._last_observation is None:
                raise RuntimeError("observe() is required before Isaac execution")
            actual_observation_id = self._last_observation.get("observation_id")
            if expected_observation_id != actual_observation_id:
                raise RuntimeError(
                    "stale action rejected: "
                    f"expected observation {actual_observation_id!r}, "
                    f"got {expected_observation_id!r}"
                )
            actual_digest = execution_guard_digest(self._last_observation)
            if expected_state_digest != actual_digest:
                raise RuntimeError(
                    "stale action rejected: execution state digest changed"
                )
            if action.has_non_finite():
                raise RuntimeError("Isaac controller rejected non-finite action")

            self._require_safe_observation(self._last_observation)
            self._require_retreat_interlock(self._last_observation, arm_id)
            self._controller.validate_ready(arm_id)

            # Claim before the hardware write.  If the backend times out after
            # accepting the command, reusing the id must never move the arm twice.
            self._executed_command_ids.add(command_id)
            try:
                self._controller.execute_action(action, arm_id=arm_id)
            except BaseException:
                self._controller.safe_stop(
                    f"controller execution failed for command {command_id}"
                )
                raise

            observation = dict(self._observation_source())
            observation_id = observation.get("observation_id")
            if not isinstance(observation_id, str) or not observation_id:
                self._controller.safe_stop(
                    f"post-action observation failed for command {command_id}"
                )
                raise RuntimeError(
                    "Isaac observation source returned no post-action observation_id"
                )
            self._last_observation = observation
            return observation

    def safe_stop(self, reason: str) -> SafeStopReceipt:
        with self._lock:
            receipt = self._controller.safe_stop(reason)
            if not isinstance(receipt, SafeStopReceipt):
                raise RuntimeError(
                    "Isaac controller returned an invalid safe-stop receipt"
                )
            return receipt
