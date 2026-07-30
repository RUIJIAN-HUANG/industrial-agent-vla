"""Configuration loading for the YOLO service."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

ZERO_SHA256 = f"sha256:{'0' * 64}"


def service_root() -> Path:
    """Return the services/yolo directory."""

    return Path(__file__).resolve().parents[2]


def load_config() -> dict[str, Any]:
    """Load the checked-in configuration and local environment overrides."""

    config_path = service_root() / "configs" / "yolo.default.json"
    raw = config_path.read_bytes()
    config = json.loads(raw.decode("utf-8"))

    config["config_sha"] = f"sha256:{hashlib.sha256(raw).hexdigest()}"
    config["checkpoint_sha"] = os.getenv(
        "YOLO_CHECKPOINT_SHA",
        ZERO_SHA256,
    )
    config["class_map_sha"] = os.getenv(
        "YOLO_CLASS_MAP_SHA",
        ZERO_SHA256,
    )
    config["image_cas"]["root"] = os.getenv(
        "INDUSTRIAL_AGENT_CAS_ROOT",
        config["image_cas"]["root"],
    )
    config["mock_mode"] = _env_bool(
        "YOLO_USE_MOCK",
        bool(config.get("mock_mode", True)),
    )
    config["checkpoint_path"] = os.getenv("YOLO_CHECKPOINT_PATH", "")

    validate_config(config)
    return config


def validate_config(config: dict[str, Any]) -> None:
    """Reject configuration that violates the frozen service contract."""

    if config.get("schema_version") != "1.0":
        raise ValueError("schema_version must be '1.0'")
    if config.get("service") != "yolo":
        raise ValueError("service must be 'yolo'")
    if config.get("supported_detection_contracts") != ["1.0"]:
        raise ValueError("supported_detection_contracts must be ['1.0']")

    port = config.get("port")
    if isinstance(port, bool) or not isinstance(port, int):
        raise ValueError("port must be an integer")
    if not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    for field in ("checkpoint_sha", "class_map_sha", "config_sha"):
        value = config.get(field)
        if not _looks_like_sha256(value):
            raise ValueError(f"{field} must be sha256:<64 hex characters>")
    if not config["mock_mode"] and not config["checkpoint_path"]:
        raise ValueError("YOLO_CHECKPOINT_PATH is required in real mode")
    if not config["mock_mode"]:
        for field in ("checkpoint_sha", "class_map_sha"):
            if config[field].casefold() == ZERO_SHA256:
                raise ValueError(f"{field} must be pinned in real mode")


def _looks_like_sha256(value: object) -> bool:
    if not isinstance(value, str) or not value.startswith("sha256:"):
        return False

    digest = value.removeprefix("sha256:")
    return len(digest) == 64 and all(
        character in "0123456789abcdefABCDEF" for character in digest
    )


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
