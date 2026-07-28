"""OpenVLA-OFT policy boundary and action conversion."""

from __future__ import annotations

from threading import Event
from typing import Any, Mapping, Protocol

from .exceptions import ServiceError
from .utils import validate_action_matrix


class PolicyRunner(Protocol):
    ready: bool

    def predict(
        self,
        request: Mapping[str, Any],
        cancel_event: Event | None = None,
    ) -> list[list[float]]:
        """Return model-native ``N x 7`` end-effector delta actions."""


class ActionConverter:
    """Convert between the canonical ``N x 7`` contract and model-native actions.

    The frozen service role uses the same dimensional order on both sides:
    ``dx_m, dy_m, dz_m, droll_rad, dpitch_rad, dyaw_rad, gripper_norm``.
    Keeping the conversion explicit prevents future model integrations from
    silently changing units or axis order.
    """

    def __init__(self, *, max_steps: int) -> None:
        self.max_steps = max_steps

    def canonical_to_native(self, actions: Any) -> list[list[float]]:
        return validate_action_matrix(actions, max_steps=self.max_steps)

    def native_to_canonical(self, actions: Any) -> list[list[float]]:
        return validate_action_matrix(actions, max_steps=self.max_steps)


class MockOpenVLAPolicy:
    """Deterministic policy for contract and orchestration smoke tests only."""

    ready = True

    def __init__(self, *, steps: int) -> None:
        self.steps = steps

    def predict(
        self,
        request: Mapping[str, Any],
        cancel_event: Event | None = None,
    ) -> list[list[float]]:
        del request
        template = [
            [0.0, 0.015, 0.0, 0.0, 0.0, 0.0, -0.75],
            [0.0, 0.015, 0.0, 0.0, 0.0, 0.0, -0.75],
            [0.0, 0.010, 0.010, 0.0, 0.0, 0.0, -0.25],
            [0.0, 0.000, 0.000, 0.0, 0.0, 0.0, 0.75],
        ]
        output: list[list[float]] = []
        for row in template[: self.steps]:
            if cancel_event is not None and cancel_event.is_set():
                raise ServiceError(
                    "EXEC_2107_CANCELLED",
                    "mock OpenVLA-OFT inference was cancelled",
                    retryable=False,
                )
            output.append(row)
        return output


class RealOpenVLAPolicy:
    """Placeholder for the pinned OpenVLA-OFT implementation.

    This class deliberately fails closed until the upstream commit, tuned
    checkpoint, norm stats, and industrial fine-tuning evidence have been
    pinned. It should be replaced by a thin adapter around the selected
    OpenVLA-OFT repository, not by supervisor-side model loading.
    """

    ready = False

    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config

    def predict(
        self,
        request: Mapping[str, Any],
        cancel_event: Event | None = None,
    ) -> list[list[float]]:
        del request
        del cancel_event
        raise ServiceError(
            "EXEC_2101_UNAVAILABLE",
            "real OpenVLA-OFT inference is not integrated; "
            "use mock mode only for smoke tests",
            retryable=False,
        )


class OpenVLAOFTModel:
    def __init__(self, config: Mapping[str, Any]) -> None:
        self.config = config
        runtime = config.get("api", {})
        max_steps = (
            int(runtime.get("max_chunk_steps", 32))
            if isinstance(runtime, Mapping)
            else 32
        )
        self.converter = ActionConverter(max_steps=max_steps)
        mock_mode = bool(config.get("mock_mode", True))
        if mock_mode:
            self.runner: PolicyRunner = MockOpenVLAPolicy(steps=min(4, max_steps))
        else:
            self.runner = RealOpenVLAPolicy(config)

    @property
    def ready(self) -> bool:
        return bool(self.runner.ready)

    def predict(
        self,
        request: Mapping[str, Any],
        cancel_event: Event | None = None,
    ) -> list[list[float]]:
        native_actions = self.runner.predict(request, cancel_event=cancel_event)
        return self.converter.native_to_canonical(native_actions)
