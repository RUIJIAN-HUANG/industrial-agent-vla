"""Data contracts for the isolated π0.5 base-checkpoint experiment."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np


CANONICAL_BASE_CHECKPOINT = "gs://openpi-assets/checkpoints/pi05_base"
SUPPORTED_INPUT_PROFILES = frozenset({"droid_joint_gripper"})


def is_pi05_base_checkpoint(reference: str) -> bool:
    """Return whether a reference identifies the unmodified π0.5 base checkpoint."""

    normalized = reference.strip().replace("\\", "/").rstrip("/")
    return normalized == CANONICAL_BASE_CHECKPOINT or normalized.endswith(
        "/pi05_base"
    )


def _owned_read_only_array(value: Any, *, dtype: np.dtype[Any]) -> np.ndarray:
    array = np.array(value, dtype=dtype, copy=True)
    array.setflags(write=False)
    return array


@dataclass(frozen=True)
class ExperimentObservation:
    """The complete online input allowed to cross the base-policy boundary."""

    observation_id: str
    timestamp_ns: int
    front_rgb: np.ndarray
    joint_position: np.ndarray
    gripper_position: float
    prompt: str
    wrist_rgb: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observation_id, str) or not self.observation_id.strip():
            raise ValueError("observation_id must be a non-empty string")
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, int):
            raise ValueError("timestamp_ns must be an integer")
        if self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative")

        front = _owned_read_only_array(self.front_rgb, dtype=np.dtype(np.uint8))
        if front.ndim != 3 or front.shape[2] != 3 or min(front.shape[:2]) < 1:
            raise ValueError("front_rgb must have shape [height,width,3]")
        if np.asarray(self.front_rgb).dtype != np.uint8:
            raise ValueError("front_rgb must be uint8 RGB without implicit scaling")

        wrist: np.ndarray | None = None
        if self.wrist_rgb is not None:
            wrist = _owned_read_only_array(
                self.wrist_rgb,
                dtype=np.dtype(np.uint8),
            )
            if wrist.ndim != 3 or wrist.shape[2] != 3 or min(wrist.shape[:2]) < 1:
                raise ValueError("wrist_rgb must have shape [height,width,3]")
            if np.asarray(self.wrist_rgb).dtype != np.uint8:
                raise ValueError("wrist_rgb must be uint8 RGB without implicit scaling")

        joints = _owned_read_only_array(
            self.joint_position,
            dtype=np.dtype(np.float32),
        )
        if joints.shape != (7,):
            raise ValueError("joint_position must contain exactly seven Franka joints")
        if not np.all(np.isfinite(joints)):
            raise ValueError("joint_position contains NaN or Infinity")

        gripper = float(self.gripper_position)
        if not np.isfinite(gripper):
            raise ValueError("gripper_position contains NaN or Infinity")
        if not 0.0 <= gripper <= 1.0:
            raise ValueError("gripper_position must be normalized to [0,1]")
        if not isinstance(self.prompt, str) or not self.prompt.strip():
            raise ValueError("prompt must be a non-empty string")

        object.__setattr__(self, "front_rgb", front)
        object.__setattr__(self, "wrist_rgb", wrist)
        object.__setattr__(self, "joint_position", joints)
        object.__setattr__(self, "gripper_position", gripper)
        object.__setattr__(self, "prompt", self.prompt.strip())

    def to_droid_example(self) -> dict[str, Any]:
        """Build the official DROID-style inference envelope.

        The DROID profile is only a compatibility hypothesis for a Franka-like
        embodiment.  It requires a real wrist image; silently fabricating one
        would contaminate the zero-shot experiment.
        """

        if self.wrist_rgb is None:
            raise ValueError(
                "droid_joint_gripper input requires a real wrist_rgb image; "
                "a black placeholder is not accepted in real mode"
            )
        return {
            "observation/exterior_image_1_left": self.front_rgb,
            "observation/wrist_image_left": self.wrist_rgb,
            "observation/joint_position": self.joint_position,
            "observation/gripper_position": np.asarray(
                [self.gripper_position],
                dtype=np.float32,
            ),
            "prompt": self.prompt,
        }


@dataclass(frozen=True)
class PolicyOutput:
    """Unmodified action output returned by one policy invocation."""

    actions: np.ndarray
    policy_mode: str
    checkpoint_reference: str
    latency_ms: float
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        actions = _owned_read_only_array(
            self.actions,
            dtype=np.dtype(np.float32),
        )
        if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
            raise ValueError("policy actions must have shape [horizon,action_dim]")
        if self.policy_mode not in {"mock", "real"}:
            raise ValueError("policy_mode must be 'mock' or 'real'")
        if not isinstance(self.latency_ms, (int, float)) or self.latency_ms < 0:
            raise ValueError("latency_ms must be non-negative")
        object.__setattr__(self, "actions", actions)
        object.__setattr__(self, "latency_ms", float(self.latency_ms))
        object.__setattr__(self, "metadata", dict(self.metadata))


@dataclass(frozen=True)
class ExperimentConfig:
    """Configuration that freezes model identity without touching production config."""

    schema_version: str
    experiment_id: str
    checkpoint_uri: str
    openpi_config_name: str
    input_profile: str
    prompt: str
    mock_action_horizon: int = 15
    mock_action_dim: int = 32

    def __post_init__(self) -> None:
        if self.schema_version != "1.0":
            raise ValueError("unsupported experiment schema_version")
        if not self.experiment_id.strip():
            raise ValueError("experiment_id must be non-empty")
        if not is_pi05_base_checkpoint(self.checkpoint_uri):
            raise ValueError(
                "checkpoint_uri must identify the unmodified pi05_base checkpoint"
            )
        if not self.openpi_config_name.strip():
            raise ValueError("openpi_config_name must be non-empty")
        if self.input_profile not in SUPPORTED_INPUT_PROFILES:
            raise ValueError(f"unsupported input_profile: {self.input_profile!r}")
        if not self.prompt.strip():
            raise ValueError("prompt must be non-empty")
        if self.mock_action_horizon < 1 or self.mock_action_dim < 1:
            raise ValueError("mock action shape must be positive")

    @classmethod
    def from_path(cls, path: Path) -> "ExperimentConfig":
        if not path.is_file():
            raise FileNotFoundError(f"experiment config not found: {path}")
        if path.stat().st_size > 64 * 1024:
            raise ValueError("experiment config exceeds 64 KiB")
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("experiment config must be a JSON object")
        return cls(**raw)
