"""Configuration loading for the standalone OpenVLA-OFT service."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from industrial_agent.executor import OPENVLA_OFT_TASK_TYPES

SHA256_PREFIX = "sha256:"
ZERO_SHA256 = f"{SHA256_PREFIX}{'0' * 64}"
IMAGE_CAS_DEFAULT_ROOT = "artifacts/cas"
PINNED_UPSTREAM_COMMIT = "e4287e94541f459edc4feabc4e181f537cd569a8"


def service_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _merge(base: dict[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), dict):
            result[key] = _merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return data


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    normalized = raw.casefold()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be an explicit boolean value")


def looks_like_sha256(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith(SHA256_PREFIX):
        return False
    suffix = value.removeprefix(SHA256_PREFIX)
    return len(suffix) == 64 and all(
        char in "0123456789abcdefABCDEF" for char in suffix
    )


def load_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    """Load public, reproducible service configuration.

    Environment variables are limited to immutable artifact identifiers and
    local model/CAS locations so checked-in configuration never contains
    personal machine paths.
    """

    base_dir = (
        Path(config_dir) if config_dir is not None else service_root() / "configs"
    )
    agent_config = _read_json(base_dir / "agent.default.json")
    model_config = _read_json(base_dir / "openvla.default.json")
    config = _merge(agent_config, model_config)

    artifacts = config.setdefault("artifacts", {})
    artifacts["checkpoint_sha"] = os.getenv(
        "OPENVLA_OFT_CHECKPOINT_SHA",
        artifacts.get("checkpoint_sha", ZERO_SHA256),
    )
    artifacts["norm_stats_sha"] = os.getenv(
        "OPENVLA_OFT_NORM_STATS_SHA",
        artifacts.get("norm_stats_sha", ZERO_SHA256),
    )
    image_cas = config.setdefault("image_cas", {})
    image_cas["root"] = os.getenv(
        "INDUSTRIAL_AGENT_CAS_ROOT",
        image_cas.get("root", IMAGE_CAS_DEFAULT_ROOT),
    )
    config["mock_mode"] = _env_bool(
        "OPENVLA_OFT_USE_MOCK",
        bool(config.get("mock_mode", True)),
    )
    config["checkpoint_dir"] = os.getenv(
        "OPENVLA_OFT_CHECKPOINT_DIR",
        str(config.get("checkpoint_dir", "")),
    )
    config["upstream_dir"] = os.getenv(
        "OPENVLA_OFT_UPSTREAM_DIR",
        str(config.get("upstream_dir", "")),
    )
    artifacts["checkpoint_manifest"] = os.getenv(
        "OPENVLA_OFT_CHECKPOINT_MANIFEST",
        str(artifacts.get("checkpoint_manifest", "checkpoint.manifest.json")),
    )
    artifacts["norm_stats_file"] = os.getenv(
        "OPENVLA_OFT_NORM_STATS_FILE",
        str(artifacts.get("norm_stats_file", "dataset_statistics.json")),
    )
    artifacts["action_contract_file"] = os.getenv(
        "OPENVLA_OFT_ACTION_CONTRACT_FILE",
        str(artifacts.get("action_contract_file", "action_contract.json")),
    )
    runtime = config.setdefault("runtime", {})
    runtime["unnorm_key"] = os.getenv(
        "OPENVLA_OFT_UNNORM_KEY",
        str(runtime.get("unnorm_key", "")),
    )
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "service",
        "arm_id",
        "required_subtask_id",
        "instruction",
        "action_contract_version",
        "camera_order",
        "image_size",
        "api",
        "artifacts",
        "image_cas",
        "model",
        "runtime",
        "upstream",
        "language_field",
        "task_id_field",
        "action_conversion",
    }
    missing = required - set(config)
    if missing:
        raise ValueError(f"OpenVLA-OFT config missing fields: {sorted(missing)}")
    if config["service"] != "openvla_oft":
        raise ValueError("service must be 'openvla_oft'")
    if config["arm_id"] != "Arm_B":
        raise ValueError("OpenVLA-OFT service is frozen to arm_id='Arm_B'")
    if config["required_subtask_id"] != "S02_ARM_B_TRANSPORT":
        raise ValueError("OpenVLA-OFT service only accepts S02_ARM_B_TRANSPORT")
    if config.get("supported_task_types") != sorted(OPENVLA_OFT_TASK_TYPES):
        raise ValueError(
            "supported_task_types must exactly match the Supervisor "
            "OpenVLA-OFT descriptor"
        )
    if config["action_contract_version"] != "1.0":
        raise ValueError("action_contract_version must be '1.0'")
    if config["camera_order"] != ["CAM_B_TOP"]:
        raise ValueError("camera_order must be ['CAM_B_TOP']")
    image_size = config["image_size"]
    if (
        not isinstance(image_size, list)
        or len(image_size) != 2
        or any(
            isinstance(item, bool) or not isinstance(item, int) or item < 1
            for item in image_size
        )
    ):
        raise ValueError("image_size must contain two positive integers")
    if image_size != [1280, 720]:
        raise ValueError("image_size must be [1280, 720]")
    if config["language_field"] != "model_input.task_description":
        raise ValueError("language_field must be model_input.task_description")
    if config["task_id_field"] != "task_id":
        raise ValueError("task_id_field must be task_id")
    action_conversion = config["action_conversion"]
    if not isinstance(action_conversion, Mapping):
        raise ValueError("action_conversion must be an object")
    frozen_order = [
        "dx_m",
        "dy_m",
        "dz_m",
        "dax_rad",
        "day_rad",
        "daz_rad",
        "gripper_norm",
    ]
    if action_conversion.get("canonical_order") != frozen_order:
        raise ValueError(
            "action_conversion.canonical_order must be the frozen N x 7 order"
        )
    if action_conversion.get("native_order") != frozen_order:
        raise ValueError(
            "action_conversion.native_order must match the frozen N x 7 order"
        )
    if action_conversion.get("rotation_representation") != "rotation_vector_axis_angle":
        raise ValueError(
            "action_conversion.rotation_representation must be "
            "'rotation_vector_axis_angle'"
        )
    if action_conversion.get("gripper_open_threshold") != 0.5:
        raise ValueError("action_conversion.gripper_open_threshold must be 0.5")
    model = config["model"]
    if not isinstance(model, Mapping):
        raise ValueError("model must be an object")
    if model.get("architecture") != "openvla-oft":
        raise ValueError("model.architecture must be openvla-oft")
    if model.get("input_image_size") != [1280, 720]:
        raise ValueError("model.input_image_size must be [1280, 720]")
    if model.get("uses_wrist_camera") is not False:
        raise ValueError("model.uses_wrist_camera must be false")
    if model.get("proprio_dim") != 7:
        raise ValueError("model.proprio_dim must be 7")
    frozen_proprio_order = [
        "x_m",
        "y_m",
        "z_m",
        "roll_rad",
        "pitch_rad",
        "yaw_rad",
        "gripper_norm",
    ]
    if model.get("proprio_order") != frozen_proprio_order:
        raise ValueError("model.proprio_order must match the frozen Arm_B state order")
    if model.get("action_dim") != 7:
        raise ValueError("model.action_dim must be 7")
    upstream = config["upstream"]
    if not isinstance(upstream, Mapping):
        raise ValueError("upstream must be an object")
    if upstream.get("repo") != "https://github.com/moojink/openvla-oft":
        raise ValueError("upstream.repo must use the official OpenVLA-OFT repository")
    if upstream.get("commit_sha") != PINNED_UPSTREAM_COMMIT:
        raise ValueError(
            f"upstream.commit_sha must be pinned to {PINNED_UPSTREAM_COMMIT}"
        )
    runtime = config["runtime"]
    if not isinstance(runtime, Mapping):
        raise ValueError("runtime must be an object")
    expected_runtime = {
        "use_l1_regression": True,
        "use_diffusion": False,
        "use_film": False,
        "num_images_in_input": 1,
        "use_proprio": True,
    }
    for field, expected in expected_runtime.items():
        if runtime.get(field) != expected:
            raise ValueError(f"runtime.{field} must be {expected!r}")
    for field in (
        "center_crop",
        "load_in_8bit",
        "load_in_4bit",
    ):
        if not isinstance(runtime.get(field), bool):
            raise ValueError(f"runtime.{field} must be a boolean")
    for field in (
        "num_open_loop_steps",
        "lora_rank",
        "num_diffusion_steps_train",
        "num_diffusion_steps_inference",
    ):
        value = runtime.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"runtime.{field} must be a positive integer")
    artifacts = config["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be an object")
    for field in ("checkpoint_sha", "norm_stats_sha"):
        if not looks_like_sha256(artifacts.get(field)):
            raise ValueError(f"artifacts.{field} must be sha256:<64 hex characters>")
        if (
            not bool(config.get("mock_mode"))
            and artifacts[field].casefold() == ZERO_SHA256
        ):
            raise ValueError(
                f"artifacts.{field} cannot use the all-zero mock digest in real mode"
            )
    for field in (
        "checkpoint_manifest",
        "norm_stats_file",
        "action_contract_file",
    ):
        if not isinstance(artifacts.get(field), str) or not artifacts[field].strip():
            raise ValueError(f"artifacts.{field} must be a non-empty relative path")
        if Path(artifacts[field]).is_absolute() or ".." in Path(artifacts[field]).parts:
            raise ValueError(f"artifacts.{field} must stay inside checkpoint_dir")
    if not bool(config.get("mock_mode")):
        for field in ("checkpoint_dir", "upstream_dir"):
            if not isinstance(config.get(field), str) or not config[field].strip():
                raise ValueError(f"{field} is required in real mode")
        if (
            not isinstance(runtime.get("unnorm_key"), str)
            or not runtime["unnorm_key"].strip()
        ):
            raise ValueError("runtime.unnorm_key is required in real mode")
    image_cas = config["image_cas"]
    if not isinstance(image_cas, Mapping):
        raise ValueError("image_cas must be an object")
    if not isinstance(image_cas.get("root"), str) or not image_cas["root"].strip():
        raise ValueError("image_cas.root must be a non-empty string")
    if image_cas.get("layout") != "sha256-v1":
        raise ValueError("image_cas.layout must be sha256-v1")
    if image_cas.get("encoding") != "png":
        raise ValueError("image_cas.encoding must be png")
    if image_cas.get("digest_scope") != "encoded_bytes":
        raise ValueError("image_cas.digest_scope must be encoded_bytes")
    for field in (
        "max_blob_bytes",
        "max_pixels",
        "cache_max_bytes",
        "missing_retry_count",
        "missing_retry_delay_ms",
    ):
        value = image_cas.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"image_cas.{field} must be a non-negative integer")
    api = config["api"]
    if not isinstance(api, Mapping):
        raise ValueError("api must be an object")
    for field in (
        "default_deadline_ms",
        "max_deadline_ms",
        "max_concurrent_requests",
        "completed_cache_max_entries",
    ):
        value = api.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"api.{field} must be a positive integer")
