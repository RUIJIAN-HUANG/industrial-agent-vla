"""Synchronous Isaac-to-Canonical Recorder bridge.

The bridge owns no Isaac objects.  It receives one state callback and one RGB
pipeline, then samples them only on the frozen 120/60/30/10 Hz integer grids.
Object ground truth is intentionally absent from this module and therefore
cannot leak into Canonical training fields.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import time
from typing import Any

from industrial_agent.contracts import ActionStep
from industrial_agent.data.recorder import (
    ARM_IDS,
    CAMERA_IDS,
    EXECUTOR_BY_ARM,
    CanonicalRecorder,
)
from industrial_agent.sync_contract import FROZEN_MULTI_RATE


StateSource = Callable[[], Mapping[str, Sequence[float]]]


class CanonicalRecorderBridge:
    """Record live Isaac ticks into exactly one Canonical episode."""

    def __init__(
        self,
        *,
        recorder: CanonicalRecorder,
        rgb_pipeline: Any,
        state_source: StateSource,
        timestamp_origin_ns: int | None = None,
    ) -> None:
        if not isinstance(recorder, CanonicalRecorder):
            raise TypeError("recorder must be CanonicalRecorder")
        if not callable(getattr(rgb_pipeline, "capture_references", None)):
            raise TypeError("rgb_pipeline must provide capture_references()")
        if not callable(state_source):
            raise TypeError("state_source must be callable")
        origin = time.time_ns() if timestamp_origin_ns is None else timestamp_origin_ns
        if isinstance(origin, bool) or not isinstance(origin, int) or origin < 0:
            raise ValueError("timestamp_origin_ns must be a non-negative integer")

        self._recorder = recorder
        self._rgb_pipeline = rgb_pipeline
        self._state_source = state_source
        self._timestamp_origin_ns = origin
        self._camera_sequence = 0
        self._state_sequence = 0
        self._action_sequence = 0
        self._last_tick = -1
        self._initial_recorded = False
        self._closed = False

    @staticmethod
    def _require_grid(tick: int, stride: int, name: str) -> None:
        if isinstance(tick, bool) or not isinstance(tick, int) or tick < 0:
            raise ValueError("physics_tick must be a non-negative integer")
        if tick % stride:
            raise ValueError(f"physics_tick {tick} is not on the {name} grid")

    def _timestamp(self, tick: int) -> int:
        return self._timestamp_origin_ns + (
            tick * 1_000_000_000 // FROZEN_MULTI_RATE.physics_hz
        )

    def _record_states(self, tick: int) -> None:
        states = self._state_source()
        if set(states) != set(ARM_IDS):
            raise ValueError("state_source must return exactly Arm_A and Arm_B")
        timestamp = self._timestamp(tick)
        for arm_id in ARM_IDS:
            self._recorder.add_state(
                arm_id=arm_id,
                timestamp_ns=timestamp,
                physics_tick=tick,
                sequence_id=self._state_sequence,
                state_7d=states[arm_id],
            )
        self._state_sequence += 1

    def _record_frames(self, tick: int) -> None:
        references = self._rgb_pipeline.capture_references()
        if set(references) != set(CAMERA_IDS):
            raise ValueError("RGB pipeline must return exactly the three cameras")
        timestamp = self._timestamp(tick)
        for camera_id in CAMERA_IDS:
            self._recorder.add_frame(
                camera_id=camera_id,
                timestamp_ns=timestamp,
                physics_tick=tick,
                sequence_id=self._camera_sequence,
                image_reference=references[camera_id],
            )
        self._camera_sequence += 1

    def record_initial(self, *, physics_tick: int = 0) -> None:
        """Write the synchronized tick-zero state and camera set once."""

        if self._closed:
            raise RuntimeError("bridge is closed")
        if self._initial_recorded:
            raise RuntimeError("initial sample was already recorded")
        self._require_grid(
            physics_tick,
            FROZEN_MULTI_RATE.physics_ticks_per_render,
            "render",
        )
        self._record_states(physics_tick)
        self._record_frames(physics_tick)
        self._last_tick = physics_tick
        self._initial_recorded = True

    def record_action(
        self,
        action: ActionStep,
        *,
        arm_id: str,
        subtask_id: str,
        chunk_id: str,
        physics_tick: int,
    ) -> None:
        """Record the 10 Hz command immediately before controller execution."""

        if self._closed or not self._initial_recorded:
            raise RuntimeError("record_initial() must run before actions")
        if not isinstance(action, ActionStep):
            raise TypeError("action must be ActionStep")
        if arm_id not in EXECUTOR_BY_ARM:
            raise ValueError(f"unknown arm_id: {arm_id!r}")
        self._require_grid(
            physics_tick,
            FROZEN_MULTI_RATE.physics_ticks_per_model_step,
            "model",
        )
        if physics_tick != self._last_tick:
            raise ValueError("action tick must equal the latest observed physics tick")
        self._recorder.add_action(
            arm_id=arm_id,
            executor=EXECUTOR_BY_ARM[arm_id],
            subtask_id=subtask_id,
            chunk_id=chunk_id,
            timestamp_ns=self._timestamp(physics_tick),
            physics_tick=physics_tick,
            sequence_id=self._action_sequence,
            action_7d=action.values,
            duration_ms=action.duration_ms,
        )
        self._action_sequence += 1

    def observe_physics_tick(self, physics_tick: int, render_due: bool) -> None:
        """Controller callback executed after one live 120 Hz world step."""

        if self._closed or not self._initial_recorded:
            raise RuntimeError("record_initial() must run before observing ticks")
        if physics_tick != self._last_tick + 1:
            raise ValueError(
                f"physics ticks must be contiguous: expected {self._last_tick + 1}, "
                f"got {physics_tick}"
            )
        if not isinstance(render_due, bool):
            raise TypeError("render_due must be bool")
        expected_render = physics_tick % FROZEN_MULTI_RATE.physics_ticks_per_render == 0
        if render_due != expected_render:
            raise ValueError("controller render flag drifted from the frozen grid")
        if physics_tick % FROZEN_MULTI_RATE.physics_ticks_per_control == 0:
            self._record_states(physics_tick)
        if render_due:
            self._record_frames(physics_tick)
        self._last_tick = physics_tick

    def save(self, *, outcome: str, failure_code: str | None = None):
        if self._closed:
            raise RuntimeError("bridge is closed")
        self._closed = True
        return self._recorder.save_episode(
            outcome=outcome,
            failure_code=failure_code,
        )

    def abort(self) -> None:
        if not self._closed:
            self._closed = True
            self._recorder.abort()

    @property
    def latest_physics_tick(self) -> int:
        return self._last_tick
