"""OpenVLA-OFT policy boundary and action conversion."""

from __future__ import annotations

import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from typing import Any, Mapping, Protocol

import numpy as np

from .exceptions import ServiceError
from .utils import validate_action_matrix

FROZEN_ACTION_ORDER = [
    "dx_m",
    "dy_m",
    "dz_m",
    "droll_rad",
    "dpitch_rad",
    "dyaw_rad",
    "gripper_norm",
]
FROZEN_PROPRIO_ORDER = [
    "x_m",
    "y_m",
    "z_m",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
    "gripper_norm",
]


def _artifact_digest(value: str) -> str:
    return value.removeprefix("sha256:").casefold()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_inside(root: Path, relative_path: str) -> Path:
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(
            f"checkpoint manifest path escapes checkpoint_dir: {relative_path!r}"
        ) from exc
    return candidate


def verify_checkpoint_artifacts(config: Mapping[str, Any]) -> None:
    """Verify the immutable checkpoint manifest and every referenced file."""

    checkpoint_dir = Path(str(config["checkpoint_dir"])).expanduser().resolve()
    if not checkpoint_dir.is_dir():
        raise RuntimeError(f"checkpoint_dir does not exist: {checkpoint_dir}")

    artifacts = config["artifacts"]
    manifest_name = str(artifacts["checkpoint_manifest"])
    manifest_path = _resolve_inside(checkpoint_dir, manifest_name)
    if not manifest_path.is_file():
        raise RuntimeError(f"checkpoint manifest does not exist: {manifest_path}")
    actual_manifest_sha = _sha256_file(manifest_path)
    expected_manifest_sha = _artifact_digest(str(artifacts["checkpoint_sha"]))
    if actual_manifest_sha != expected_manifest_sha:
        raise RuntimeError(
            "checkpoint manifest SHA256 mismatch: "
            f"expected {expected_manifest_sha}, got {actual_manifest_sha}"
        )

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, Mapping) or manifest.get("schema_version") != "1.0":
        raise RuntimeError("checkpoint manifest must use schema_version 1.0")
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise RuntimeError("checkpoint manifest files must be a non-empty list")

    seen: set[str] = set()
    for index, entry in enumerate(files):
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"checkpoint manifest files[{index}] must be an object")
        relative_path = entry.get("path")
        expected_sha = entry.get("sha256")
        if (
            not isinstance(relative_path, str)
            or not relative_path
            or not isinstance(expected_sha, str)
            or len(expected_sha) != 64
            or any(char not in "0123456789abcdefABCDEF" for char in expected_sha)
        ):
            raise RuntimeError(f"checkpoint manifest files[{index}] is invalid")
        if relative_path in seen:
            raise RuntimeError(
                f"checkpoint manifest contains duplicate path: {relative_path}"
            )
        seen.add(relative_path)
        artifact_path = _resolve_inside(checkpoint_dir, relative_path)
        if not artifact_path.is_file():
            raise RuntimeError(f"checkpoint artifact is missing: {artifact_path}")
        actual_sha = _sha256_file(artifact_path)
        if actual_sha != expected_sha.casefold():
            raise RuntimeError(
                f"checkpoint artifact SHA256 mismatch for {relative_path}: "
                f"expected {expected_sha.casefold()}, got {actual_sha}"
            )

    norm_stats_path = _resolve_inside(
        checkpoint_dir,
        str(artifacts["norm_stats_file"]),
    )
    if not norm_stats_path.is_file():
        raise RuntimeError(f"norm stats file does not exist: {norm_stats_path}")
    actual_norm_sha = _sha256_file(norm_stats_path)
    expected_norm_sha = _artifact_digest(str(artifacts["norm_stats_sha"]))
    if actual_norm_sha != expected_norm_sha:
        raise RuntimeError(
            "norm stats SHA256 mismatch: "
            f"expected {expected_norm_sha}, got {actual_norm_sha}"
        )

    action_contract_path = _resolve_inside(
        checkpoint_dir,
        str(artifacts["action_contract_file"]),
    )
    if not action_contract_path.is_file():
        raise RuntimeError(
            f"checkpoint action contract does not exist: {action_contract_path}"
        )
    with action_contract_path.open("r", encoding="utf-8") as stream:
        action_contract = json.load(stream)
    expected_contract = {
        "schema_version": "1.0",
        "frame": "robot_base",
        "translation_unit": "m",
        "rotation_unit": "rad",
        "gripper_unit": "normalized",
        "action_order": FROZEN_ACTION_ORDER,
        "proprio_order": FROZEN_PROPRIO_ORDER,
    }
    if action_contract != expected_contract:
        raise RuntimeError(
            "checkpoint action_contract.json does not exactly match the frozen "
            "Arm_B state/action semantics"
        )


def _verify_upstream_checkout(upstream_dir: Path, expected_commit: str) -> None:
    if not upstream_dir.is_dir():
        raise RuntimeError(f"OpenVLA-OFT upstream checkout is missing: {upstream_dir}")
    try:
        result = subprocess.run(
            ["git", "-C", str(upstream_dir), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        raise RuntimeError(
            "cannot verify the OpenVLA-OFT upstream checkout commit"
        ) from exc
    actual_commit = result.stdout.strip().casefold()
    if actual_commit != expected_commit.casefold():
        raise RuntimeError(
            "OpenVLA-OFT upstream commit mismatch: "
            f"expected {expected_commit}, got {actual_commit}"
        )


@dataclass(frozen=True)
class OfficialBindings:
    """Loaded objects from the pinned official OpenVLA-OFT repository."""

    cfg: Any
    vla: Any
    processor: Any
    action_head: Any
    proprio_projector: Any
    get_vla_action: Any

    @classmethod
    def load(cls, config: Mapping[str, Any]) -> "OfficialBindings":
        upstream = config["upstream"]
        upstream_dir = Path(str(config["upstream_dir"])).expanduser().resolve()
        expected_commit = str(upstream["commit_sha"])
        _verify_upstream_checkout(upstream_dir, expected_commit)
        if str(upstream_dir) not in sys.path:
            sys.path.insert(0, str(upstream_dir))

        try:
            utilities = importlib.import_module("experiments.robot.openvla_utils")
        except ImportError as exc:
            raise RuntimeError(
                "cannot import the pinned official OpenVLA-OFT inference utilities"
            ) from exc

        runtime = config["runtime"]
        model_config = config["model"]
        cfg = SimpleNamespace(
            pretrained_checkpoint=str(config["checkpoint_dir"]),
            use_l1_regression=bool(runtime["use_l1_regression"]),
            use_diffusion=bool(runtime["use_diffusion"]),
            use_film=bool(runtime["use_film"]),
            num_images_in_input=int(runtime["num_images_in_input"]),
            use_proprio=bool(runtime["use_proprio"]),
            center_crop=bool(runtime["center_crop"]),
            num_open_loop_steps=int(runtime["num_open_loop_steps"]),
            lora_rank=int(runtime["lora_rank"]),
            unnorm_key=str(runtime["unnorm_key"]),
            load_in_8bit=bool(runtime["load_in_8bit"]),
            load_in_4bit=bool(runtime["load_in_4bit"]),
            num_diffusion_steps_train=int(runtime["num_diffusion_steps_train"]),
            num_diffusion_steps_inference=int(runtime["num_diffusion_steps_inference"]),
        )
        vla = utilities.get_vla(cfg)
        processor = utilities.get_processor(cfg)
        action_head = utilities.get_action_head(cfg, llm_dim=vla.llm_dim)
        proprio_projector = utilities.get_proprio_projector(
            cfg,
            llm_dim=vla.llm_dim,
            proprio_dim=int(model_config["proprio_dim"]),
        )
        norm_stats = getattr(vla, "norm_stats", None)
        if not isinstance(norm_stats, Mapping) or cfg.unnorm_key not in norm_stats:
            raise RuntimeError(
                f"checkpoint norm stats do not contain unnorm_key={cfg.unnorm_key!r}"
            )
        return cls(
            cfg=cfg,
            vla=vla,
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            get_vla_action=utilities.get_vla_action,
        )

    def predict(
        self,
        *,
        image: np.ndarray,
        state: np.ndarray,
        instruction: str,
    ) -> Any:
        observation = {
            "full_image": image,
            "state": state.copy(),
            "task_description": instruction,
        }
        return self.get_vla_action(
            self.cfg,
            self.vla,
            self.processor,
            observation,
            instruction,
            self.action_head,
            self.proprio_projector,
            use_film=bool(self.cfg.use_film),
        )


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
    """Thin, fail-closed adapter around the pinned official implementation."""

    ready = True

    def __init__(
        self,
        config: Mapping[str, Any],
        *,
        bindings: OfficialBindings | Any | None = None,
    ) -> None:
        self.config = config
        verify_checkpoint_artifacts(config)
        self.bindings = bindings or OfficialBindings.load(config)

    def predict(
        self,
        request: Mapping[str, Any],
        cancel_event: Event | None = None,
    ) -> list[list[float]]:
        if cancel_event is not None and cancel_event.is_set():
            raise ServiceError(
                "EXEC_2107_CANCELLED",
                "OpenVLA-OFT inference was cancelled before model execution",
                retryable=False,
            )
        model_input = request["model_input"]
        image = np.asarray(model_input["full_image"])
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ServiceError(
                "EXEC_2103_BAD_RESPONSE",
                "resolved full_image must be uint8 HxWx3, got "
                f"{image.shape}/{image.dtype}",
                retryable=False,
            )
        state = np.asarray(model_input["state"], dtype=np.float32)
        try:
            actions = self.bindings.predict(
                image=image,
                state=state,
                instruction=str(model_input["task_description"]),
            )
        except ServiceError:
            raise
        except Exception as exc:
            raise ServiceError(
                "EXEC_2104_RUNTIME",
                "official OpenVLA-OFT inference failed",
                retryable=False,
            ) from exc
        if cancel_event is not None and cancel_event.is_set():
            raise ServiceError(
                "EXEC_2107_CANCELLED",
                "OpenVLA-OFT inference was cancelled before action publication",
                retryable=False,
            )
        matrix = np.asarray(actions, dtype=np.float64)
        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)
        return matrix.tolist()


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
