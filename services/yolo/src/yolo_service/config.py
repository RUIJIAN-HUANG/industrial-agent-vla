"""Fail-closed configuration and deployment identity for the YOLO service."""

from __future__ import annotations

import hashlib
import json
import os
from importlib import resources
from pathlib import Path
from typing import Any, Mapping

ZERO_SHA256 = f"sha256:{'0' * 64}"
_RESOURCE_PACKAGE = "yolo_service.resources"
_DEFAULT_CONFIG_NAME = "yolo.default.json"
_DEFAULT_CLASS_MAP_NAME = "class_map.single_bin_v2.json"


def _resource_bytes(name: str) -> bytes:
    return resources.files(_RESOURCE_PACKAGE).joinpath(name).read_bytes()


def _sha256_bytes(raw: bytes) -> str:
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _optional_expected_digest(environment_name: str) -> str | None:
    value = os.getenv(environment_name)
    if value is None:
        return None
    if not _looks_like_sha256(value):
        raise ValueError(f"{environment_name} must match sha256:<64 hex characters>")
    return value.casefold()


def _verify_expected_digest(
    *,
    actual: str,
    expected: str | None,
    label: str,
) -> None:
    if expected is not None and actual.casefold() != expected.casefold():
        raise ValueError(
            f"{label} SHA256 mismatch: expected {expected}, actual {actual}"
        )


def _load_class_map(raw: bytes) -> tuple[str, ...]:
    try:
        document = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("YOLO class map must be valid UTF-8 JSON") from exc
    if not isinstance(document, Mapping) or set(document) != {
        "schema_version",
        "class_map_id",
        "classes",
    }:
        raise ValueError("YOLO class map fields are invalid")
    if document["schema_version"] != "1.0":
        raise ValueError("YOLO class map schema_version must be '1.0'")
    if (
        not isinstance(document["class_map_id"], str)
        or not document["class_map_id"].strip()
    ):
        raise ValueError("YOLO class_map_id must be a non-empty string")
    classes = document["classes"]
    if not isinstance(classes, list) or not classes:
        raise ValueError("YOLO class map classes must be a non-empty list")
    names: list[str] = []
    for expected_id, item in enumerate(classes):
        if not isinstance(item, Mapping) or set(item) != {"id", "name"}:
            raise ValueError(f"YOLO class map classes[{expected_id}] is invalid")
        if item["id"] != expected_id:
            raise ValueError("YOLO class IDs must be contiguous and start at zero")
        name = item["name"]
        if not isinstance(name, str) or not name.strip() or len(name) > 128:
            raise ValueError(f"YOLO class map classes[{expected_id}].name is invalid")
        names.append(name)
    if len(set(names)) != len(names):
        raise ValueError("YOLO class names must be unique")
    return tuple(names)


def _effective_config_sha(config: Mapping[str, Any]) -> str:
    image_cas = dict(config["image_cas"])
    image_cas.pop("root", None)
    identity = {
        "schema_version": config["schema_version"],
        "service": config["service"],
        "service_version": config["service_version"],
        "mock_mode": config["mock_mode"],
        "supported_task_types": config["supported_task_types"],
        "supported_detection_contracts": config["supported_detection_contracts"],
        "api": config["api"],
        "model": config["model"],
        "image_cas": image_cas,
    }
    encoded = json.dumps(
        identity,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def load_config() -> dict[str, Any]:
    """Load packaged defaults and verify every externally supplied identity."""

    raw = _resource_bytes(_DEFAULT_CONFIG_NAME)
    config = json.loads(raw.decode("utf-8"))
    config["image_cas"]["root"] = os.getenv(
        "INDUSTRIAL_AGENT_CAS_ROOT",
        config["image_cas"]["root"],
    )
    config["mock_mode"] = _env_bool(
        "YOLO_USE_MOCK",
        bool(config.get("mock_mode", True)),
    )
    config["model"]["device"] = os.getenv(
        "YOLO_DEVICE",
        str(config["model"]["device"]),
    )

    class_map_raw = _resource_bytes(_DEFAULT_CLASS_MAP_NAME)
    class_map_sha = _sha256_bytes(class_map_raw)
    _verify_expected_digest(
        actual=class_map_sha,
        expected=_optional_expected_digest("YOLO_CLASS_MAP_SHA"),
        label="YOLO class map",
    )
    config["class_map_sha"] = class_map_sha
    config["class_names"] = _load_class_map(class_map_raw)
    config["class_map_resource"] = (
        f"package:{_RESOURCE_PACKAGE}/{_DEFAULT_CLASS_MAP_NAME}"
    )

    checkpoint_path_value = os.getenv("YOLO_CHECKPOINT_PATH", "")
    checkpoint_expected = _optional_expected_digest("YOLO_CHECKPOINT_SHA")
    if config["mock_mode"]:
        config["checkpoint_path"] = checkpoint_path_value
        config["checkpoint_sha"] = checkpoint_expected or ZERO_SHA256
    else:
        if not checkpoint_path_value:
            raise ValueError("YOLO_CHECKPOINT_PATH is required in real mode")
        checkpoint_path = Path(checkpoint_path_value).expanduser().resolve()
        if not checkpoint_path.is_file():
            raise ValueError(
                f"YOLO checkpoint must be a readable file: {checkpoint_path}"
            )
        checkpoint_sha = _sha256_file(checkpoint_path)
        _verify_expected_digest(
            actual=checkpoint_sha,
            expected=checkpoint_expected,
            label="YOLO checkpoint",
        )
        config["checkpoint_path"] = str(checkpoint_path)
        config["checkpoint_sha"] = checkpoint_sha

    config["config_sha"] = _effective_config_sha(config)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    """Reject effective configuration that violates the frozen service contract."""

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

    api = config.get("api")
    if not isinstance(api, Mapping):
        raise ValueError("api must be an object")
    for field in (
        "default_deadline_ms",
        "max_deadline_ms",
        "max_concurrent_requests",
        "max_request_bytes",
    ):
        value = api.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"api.{field} must be a positive integer")
    if api["default_deadline_ms"] > api["max_deadline_ms"]:
        raise ValueError("api.default_deadline_ms cannot exceed max_deadline_ms")

    model = config.get("model")
    if not isinstance(model, Mapping) or model.get("architecture") != "yolo":
        raise ValueError("model.architecture must be 'yolo'")
    if not isinstance(model.get("device"), str) or not model["device"].strip():
        raise ValueError("model.device must be a non-empty string")

    for field in ("checkpoint_sha", "class_map_sha", "config_sha"):
        if not _looks_like_sha256(config.get(field)):
            raise ValueError(f"{field} must be sha256:<64 hex characters>")
    if config["class_map_sha"].casefold() == ZERO_SHA256:
        raise ValueError("class_map_sha cannot be the zero digest")
    if (
        not config.get("mock_mode")
        and config["checkpoint_sha"].casefold() == ZERO_SHA256
    ):
        raise ValueError("checkpoint_sha cannot be the zero digest in real mode")

    class_names = config.get("class_names")
    if not isinstance(class_names, (tuple, list)) or not class_names:
        raise ValueError("class_names must be a non-empty sequence")


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
