"""Lazy real-policy client for the isolated π0.5 base experiment."""

from __future__ import annotations

from collections.abc import Mapping
from time import perf_counter
from typing import Any, Protocol, runtime_checkable

import numpy as np

from .contracts import ExperimentConfig, ExperimentObservation, PolicyOutput


@runtime_checkable
class BasePolicyClient(Protocol):
    """Small boundary shared by Windows mock tests and Linux real inference."""

    def infer(self, observation: ExperimentObservation) -> PolicyOutput:
        ...


class OpenPiBaseClient:
    """Load the official base checkpoint only when real inference is requested.

    Importing this module on Windows does not import JAX or OpenPI.  The OpenPI
    training *configuration* module is used solely to construct an inference
    policy; no optimizer, dataset, norm-stat computation, or training entrypoint
    is invoked here.
    """

    def __init__(self, config: ExperimentConfig) -> None:
        self._config = config
        self._policy: Any | None = None

    def _load_policy(self) -> Any:
        if self._policy is not None:
            return self._policy
        try:
            from openpi.policies import policy_config
            from openpi.shared import download
            from openpi.training import config as openpi_config
        except ImportError as exc:
            raise RuntimeError(
                "real π0.5 inference requires the official OpenPI environment; "
                "run mock mode on Windows or execute real mode in WSL/Ubuntu"
            ) from exc

        train_config = openpi_config.get_config(self._config.openpi_config_name)
        checkpoint_dir = download.maybe_download(self._config.checkpoint_uri)
        self._policy = policy_config.create_trained_policy(
            train_config,
            checkpoint_dir,
        )
        return self._policy

    def infer(self, observation: ExperimentObservation) -> PolicyOutput:
        if self._config.input_profile != "droid_joint_gripper":
            raise RuntimeError(
                f"unsupported real input profile: {self._config.input_profile}"
            )
        # Validate the honest online inputs before importing the heavy model stack.
        example = observation.to_droid_example()
        policy = self._load_policy()
        started = perf_counter()
        result = policy.infer(example)
        latency_ms = (perf_counter() - started) * 1000.0
        if not isinstance(result, Mapping) or "actions" not in result:
            raise RuntimeError("OpenPI policy did not return an 'actions' field")
        actions = np.asarray(result["actions"], dtype=np.float32)
        return PolicyOutput(
            actions=actions,
            policy_mode="real",
            checkpoint_reference=self._config.checkpoint_uri,
            latency_ms=latency_ms,
            metadata={
                "openpi_config_name": self._config.openpi_config_name,
                "input_profile": self._config.input_profile,
                "weights_modified": False,
                "training_invoked": False,
                "agent_invoked": False,
            },
        )

