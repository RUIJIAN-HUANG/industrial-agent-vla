"""Safe Canonical V2 recording boundary for a future Isaac GUI collector."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from industrial_agent.data import (
    CanonicalV2EpisodeMetadata,
    CanonicalV2Recorder,
)
from industrial_agent.data.recorder_v2 import (
    V2_INSTRUCTION,
    V2_TASK_ID,
    V2_TASK_INSTRUCTIONS,
)
from industrial_agent.image_cas import ImageCas
from industrial_agent.perception import ImageReference


CAMERA_IDS = ("CAM_A_TOP", "CAM_HANDOFF", "CAM_B_TOP")
ARM_IDS = ("Arm_A", "Arm_B")


@dataclass(frozen=True)
class V2CollectionIdentity:
    """Per-Episode values supplied by the V2 collection preflight."""

    episode_id: str
    scene_seed: int
    git_sha: str
    scene_config_sha256: str
    task_id: str = V2_TASK_ID
    instruction: str = V2_INSTRUCTION

    def to_metadata(self) -> CanonicalV2EpisodeMetadata:
        return CanonicalV2EpisodeMetadata(
            episode_id=self.episode_id,
            task_id=self.task_id,
            instruction=self.instruction,
            scene_seed=self.scene_seed,
            git_sha=self.git_sha,
            scene_config_sha256=self.scene_config_sha256,
        )


class V2CollectionRecorder:
    """Join synchronized collection events to ``CanonicalV2Recorder``.

    This class is deliberately independent of Isaac imports. The future GUI
    entry point should feed verified CAS references and robot state here. An
    action is rejected unless all three cameras and both arms were recorded at
    the same physics tick.
    """

    def __init__(
        self,
        output_root: str | Path,
        identity: V2CollectionIdentity,
        *,
        image_cas: ImageCas,
    ) -> None:
        if not isinstance(identity, V2CollectionIdentity):
            raise TypeError("identity must be V2CollectionIdentity")
        if not isinstance(image_cas, ImageCas):
            raise TypeError("image_cas must be ImageCas")
        if identity.task_id not in V2_TASK_INSTRUCTIONS:
            raise ValueError(f"unsupported V2 task_id: {identity.task_id!r}")
        expected_instruction = V2_TASK_INSTRUCTIONS[identity.task_id]
        if identity.instruction != expected_instruction:
            raise ValueError(
                f"instruction for {identity.task_id} must equal "
                f"{expected_instruction!r}"
            )
        self.identity = identity
        self.image_cas = image_cas
        self.recorder = CanonicalV2Recorder(
            output_root,
            identity.to_metadata(),
            image_cas=image_cas,
        )
        self._camera_ticks: set[int] = set()
        self._state_ticks: set[int] = set()

    @staticmethod
    def _require_exact_keys(
        values: Mapping[str, Any],
        expected: tuple[str, ...],
        *,
        field: str,
    ) -> None:
        if not isinstance(values, Mapping) or set(values) != set(expected):
            raise ValueError(f"{field} must contain exactly {expected}")

    def record_camera_bundle(
        self,
        *,
        timestamp_ns: int,
        physics_tick: int,
        sequence_id: int,
        images: Mapping[str, ImageReference | Mapping[str, Any]],
    ) -> None:
        """Record one synchronized three-camera render bundle."""

        self._require_exact_keys(images, CAMERA_IDS, field="images")
        references: dict[str, ImageReference] = {}
        for camera_id in CAMERA_IDS:
            raw_reference = images[camera_id]
            reference = (
                raw_reference
                if isinstance(raw_reference, ImageReference)
                else ImageReference.from_dict(raw_reference)
            )
            self.image_cas.resolve_rgb(
                reference,
                expected_camera_id=camera_id,
                expected_size=(1280, 720),
            )
            references[camera_id] = reference
        for camera_id in CAMERA_IDS:
            self.recorder.add_frame(
                camera_id=camera_id,
                timestamp_ns=timestamp_ns,
                physics_tick=physics_tick,
                sequence_id=sequence_id,
                image_reference=references[camera_id],
                is_fallback=False,
            )
        self._camera_ticks.add(physics_tick)

    def record_state_bundle(
        self,
        *,
        timestamp_ns: int,
        physics_tick: int,
        sequence_id: int,
        states: Mapping[str, Sequence[float] | np.ndarray],
    ) -> None:
        """Record synchronized Arm_A and Arm_B finite state vectors."""

        self._require_exact_keys(states, ARM_IDS, field="states")
        validated: dict[str, np.ndarray] = {}
        for arm_id in ARM_IDS:
            state = np.asarray(states[arm_id], dtype=np.float32)
            if state.shape != (7,) or not np.all(np.isfinite(state)):
                raise ValueError(f"{arm_id} state must be finite float32[7]")
            if float(state[6]) < 0.0 or float(state[6]) > 1.0:
                raise ValueError(f"{arm_id} state gripper must be in [0,1]")
            validated[arm_id] = np.ascontiguousarray(state)
        for arm_id in ARM_IDS:
            self.recorder.add_state(
                arm_id=arm_id,
                timestamp_ns=timestamp_ns,
                physics_tick=physics_tick,
                sequence_id=sequence_id,
                state_7d=validated[arm_id],
            )
        self._state_ticks.add(physics_tick)

    def record_action(
        self,
        *,
        timestamp_ns: int,
        physics_tick: int,
        sequence_id: int,
        chunk_id: str,
        action_7d: Sequence[float] | np.ndarray,
    ) -> None:
        """Record one task action after exact-tick observation is present."""

        if physics_tick not in self._camera_ticks:
            raise ValueError(
                f"no synchronized three-camera bundle at action tick {physics_tick}"
            )
        if physics_tick not in self._state_ticks:
            raise ValueError(
                f"no synchronized dual-arm state bundle at action tick {physics_tick}"
            )
        values = np.asarray(action_7d, dtype=np.float32)
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ValueError("V2 action must be finite float32[7]")
        if float(values[6]) not in (0.0, 1.0):
            raise ValueError("V2 action gripper must be exactly 0.0 or 1.0")
        self.recorder.add_action(
            arm_id="Arm_A",
            executor="pi05",
            subtask_id=self.identity.task_id,
            chunk_id=chunk_id,
            timestamp_ns=timestamp_ns,
            physics_tick=physics_tick,
            sequence_id=sequence_id,
            action_7d=action_7d,
            duration_ms=100,
        )

    def finalize(
        self,
        *,
        outcome: str,
        failure_code: str | None = None,
    ) -> Path:
        return self.recorder.save_episode(
            outcome=outcome,
            failure_code=failure_code,
        )

    def __enter__(self) -> V2CollectionRecorder:
        self.recorder.__enter__()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.recorder.__exit__(exc_type, exc, traceback)


__all__ = [
    "ARM_IDS",
    "CAMERA_IDS",
    "V2CollectionIdentity",
    "V2CollectionRecorder",
]
