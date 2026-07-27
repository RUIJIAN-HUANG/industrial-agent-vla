"""Configuration loading for the standalone OpenVLA-OFT service."""

from __future__ import annotations

import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

SHA256_PREFIX = "sha256:"
ZERO_SHA256 = f"{SHA256_PREFIX}{'0' * 64}"


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
    return raw.casefold() in {"1", "true", "yes", "on"}


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
    config["mock_mode"] = _env_bool(
        "OPENVLA_OFT_USE_MOCK",
        bool(config.get("mock_mode", True)),
    )
    config["checkpoint_dir"] = os.getenv("OPENVLA_OFT_CHECKPOINT_DIR", "")
    config["cas_root"] = os.getenv("CAS_ROOT", "")
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
        "model",
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
    if config["action_contract_version"] != "1.0":
        raise ValueError("action_contract_version must be '1.0'")
    if config["camera_order"] != ["CAM_B_TOP", "CAM_B_WRIST"]:
        raise ValueError("camera_order must be ['CAM_B_TOP', 'CAM_B_WRIST']")
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
    artifacts = config["artifacts"]
    if not isinstance(artifacts, Mapping):
        raise ValueError("artifacts must be an object")
    for field in ("checkpoint_sha", "norm_stats_sha"):
        if not looks_like_sha256(artifacts.get(field)):
            raise ValueError(f"artifacts.{field} must be sha256:<64 hex characters>")
    api = config["api"]
    if not isinstance(api, Mapping):
        raise ValueError("api must be an object")
    for field in (
        "default_deadline_ms",
        "max_deadline_ms",
        "max_concurrent_requests",
    ):
        value = api.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"api.{field} must be a positive integer")
