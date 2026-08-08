"""Deterministic Windows-only client for testing the experiment plumbing."""

from __future__ import annotations

import hashlib
from time import perf_counter

import numpy as np

from .contracts import ExperimentConfig, ExperimentObservation, PolicyOutput


class MockBaseClient:
    """Return deterministic fake native actions, never experimental evidence."""

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config

    def infer(self, observation: ExperimentObservation) -> PolicyOutput:
        started = perf_counter()
        digest = hashlib.sha256()
        digest.update(observation.front_rgb.tobytes())
        digest.update(observation.joint_position.tobytes())
        digest.update(observation.prompt.encode("utf-8"))
        seed = int.from_bytes(digest.digest()[:4], byteorder="big", signed=False)
        rng = np.random.default_rng(seed)
        actions = rng.uniform(
            -0.01,
            0.01,
            size=(
                self._config.mock_action_horizon,
                self._config.mock_action_dim,
            ),
        ).astype(np.float32)
        latency_ms = (perf_counter() - started) * 1000.0
        return PolicyOutput(
            actions=actions,
            policy_mode="mock",
            checkpoint_reference=self._config.checkpoint_uri,
            latency_ms=latency_ms,
            metadata={
                "generator": "deterministic_mock",
                "valid_experiment_evidence": False,
                "weights_loaded": False,
                "training_invoked": False,
                "agent_invoked": False,
            },
        )

