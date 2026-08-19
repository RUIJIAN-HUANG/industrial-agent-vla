"""Canonical V2 recorder profile built on the shared atomic HDF5 writer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Sequence

import numpy as np

from industrial_agent.image_cas import ImageCas

from .padding import PaddingPolicy, PaddingResult, PaddingStrategy
from .recorder import CanonicalRecorder, EpisodeMetadata


CANONICAL_V2_VERSION = "2.0"
V2_SCENE_ID = "single_bin_manual_industrial_v2"
V2_TASK_ID = "P01_TO_S11"
V2_INSTRUCTION = "把P01放到S11中"
V2_ARM_ID = "Arm_A"
V2_EXECUTOR = "pi05"
_SAFE_EPISODE_ID = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_GIT_SHA = re.compile(r"^[0-9a-fA-F]{40}$")
_SHA256 = re.compile(r"^sha256:[0-9a-fA-F]{64}$")


@dataclass(frozen=True)
class CanonicalV2EpisodeMetadata(EpisodeMetadata):
    """Identity fixed to the V2 P01-to-S11 manual industrial MVP."""

    scene_id: str = V2_SCENE_ID

    def __post_init__(self) -> None:
        if (
            not isinstance(self.episode_id, str)
            or _SAFE_EPISODE_ID.fullmatch(self.episode_id) is None
        ):
            raise ValueError(
                "episode_id must match ^[A-Za-z0-9._-]{1,128}$ for safe storage"
            )
        if self.scene_id != V2_SCENE_ID:
            raise ValueError(f"scene_id must equal {V2_SCENE_ID!r}")
        if self.task_id != V2_TASK_ID:
            raise ValueError(f"task_id must equal {V2_TASK_ID!r}")
        if self.instruction != V2_INSTRUCTION:
            raise ValueError(f"instruction must equal {V2_INSTRUCTION!r}")
        if (
            isinstance(self.scene_seed, bool)
            or not isinstance(self.scene_seed, int)
            or self.scene_seed < 0
        ):
            raise ValueError("scene_seed must be a non-negative integer")
        if not isinstance(self.git_sha, str) or _GIT_SHA.fullmatch(self.git_sha) is None:
            raise ValueError("git_sha must contain exactly 40 hexadecimal characters")
        if (
            not isinstance(self.scene_config_sha256, str)
            or _SHA256.fullmatch(self.scene_config_sha256) is None
        ):
            raise ValueError(
                "scene_config_sha256 must be sha256:<64 hexadecimal characters>"
            )
        object.__setattr__(self, "git_sha", self.git_sha.lower())
        object.__setattr__(
            self,
            "scene_config_sha256",
            self.scene_config_sha256.lower(),
        )


class CanonicalV2Recorder(CanonicalRecorder):
    """Write V2 Episodes without changing the V1 recorder defaults."""

    def __init__(
        self,
        output_root: str | Path,
        metadata: CanonicalV2EpisodeMetadata,
        *,
        image_cas: ImageCas,
        padding_policy: PaddingPolicy | None = None,
    ) -> None:
        if not isinstance(metadata, CanonicalV2EpisodeMetadata):
            raise TypeError("metadata must be CanonicalV2EpisodeMetadata")
        resolved_padding = padding_policy or PaddingPolicy()
        if (
            resolved_padding.strategy is not PaddingStrategy.NONE
            or resolved_padding.target_length is not None
        ):
            raise ValueError("Canonical V2 forbids padding and target_length")
        super().__init__(
            output_root,
            metadata,
            image_cas=image_cas,
            padding_policy=resolved_padding,
        )

    def _initialize_hdf5(self) -> None:
        super()._initialize_hdf5()
        del self._h5.attrs["schema_version"]
        self._h5.attrs["canonical_schema_version"] = CANONICAL_V2_VERSION
        self._h5.flush()

    @staticmethod
    def _validate_binary_action_gripper(action_7d: Sequence[float] | np.ndarray) -> None:
        """V2 action commands are hardware endpoints, never continuous values."""

        values = np.asarray(action_7d, dtype=np.float32)
        if values.shape != (7,):
            raise ValueError("V2 action_7d must have shape [7]")
        if not np.all(np.isfinite(values)):
            raise ValueError("V2 action_7d must contain finite values")
        if float(values[6]) not in (0.0, 1.0):
            raise ValueError(
                "V2 action gripper must be exactly 0.0 (closed) or 1.0 (open)"
            )

    @staticmethod
    def _validate_action_metadata(
        *,
        arm_id: str,
        executor: str,
        subtask_id: str,
        chunk_id: str,
        duration_ms: int,
    ) -> tuple[str, str]:
        if arm_id != V2_ARM_ID:
            raise ValueError(f"Canonical V2 actions require arm_id={V2_ARM_ID!r}")
        if executor != V2_EXECUTOR:
            raise ValueError(
                f"Canonical V2 actions require executor={V2_EXECUTOR!r}"
            )
        if subtask_id != V2_TASK_ID:
            raise ValueError(
                f"Canonical V2 actions require subtask_id={V2_TASK_ID!r}"
            )
        return CanonicalRecorder._validate_action_metadata(
            arm_id=arm_id,
            executor=executor,
            subtask_id=subtask_id,
            chunk_id=chunk_id,
            duration_ms=duration_ms,
        )

    def _append_action(
        self,
        *,
        arm_id: str,
        executor: str,
        subtask_id: str,
        chunk_id: str,
        chunk_position: int,
        timestamp_ns: int,
        physics_tick: int,
        sequence_id: int,
        action_7d: Sequence[float] | np.ndarray,
        duration_ms: int,
        valid: bool,
    ) -> None:
        if valid is not True:
            raise ValueError("Canonical V2 forbids padded or masked action rows")
        self._validate_binary_action_gripper(action_7d)
        super()._append_action(
            arm_id=arm_id,
            executor=executor,
            subtask_id=subtask_id,
            chunk_id=chunk_id,
            chunk_position=chunk_position,
            timestamp_ns=timestamp_ns,
            physics_tick=physics_tick,
            sequence_id=sequence_id,
            action_7d=action_7d,
            duration_ms=duration_ms,
            valid=valid,
        )

    def add_action_chunk(
        self,
        *,
        arm_id: str,
        executor: str,
        subtask_id: str,
        chunk_id: str,
        start_timestamp_ns: int,
        start_physics_tick: int,
        start_sequence_id: int,
        actions: Sequence[Sequence[float]] | np.ndarray,
        duration_ms: int = 100,
    ) -> PaddingResult:
        """Append a real variable-length chunk; every stored row remains valid."""

        result = super().add_action_chunk(
            arm_id=arm_id,
            executor=executor,
            subtask_id=subtask_id,
            chunk_id=chunk_id,
            start_timestamp_ns=start_timestamp_ns,
            start_physics_tick=start_physics_tick,
            start_sequence_id=start_sequence_id,
            actions=actions,
            duration_ms=duration_ms,
        )
        if not np.all(result.valid_mask):
            raise AssertionError("Canonical V2 recorder produced a masked row")
        return result

    def _validate_complete(self) -> None:
        super()._validate_complete()
        valid_mask = np.asarray(self._h5["actions/valid_mask"][:], dtype=np.bool_)
        if not np.all(valid_mask):
            raise ValueError("Canonical V2 valid_mask must contain only true values")

    def _manifest(
        self,
        *,
        outcome: str,
        failure_code: str | None,
        storage_sha256: str,
    ) -> dict[str, Any]:
        manifest = super()._manifest(
            outcome=outcome,
            failure_code=failure_code,
            storage_sha256=storage_sha256,
        )
        del manifest["schema_version"]
        manifest["canonical_schema_version"] = CANONICAL_V2_VERSION
        return manifest


__all__ = [
    "CANONICAL_V2_VERSION",
    "CanonicalV2EpisodeMetadata",
    "CanonicalV2Recorder",
    "V2_INSTRUCTION",
    "V2_SCENE_ID",
    "V2_TASK_ID",
]
