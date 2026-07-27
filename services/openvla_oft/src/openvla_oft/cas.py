"""Content-addressed image reference validation."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping

from .exceptions import ServiceError

CAS_URI_PATTERN = re.compile(r"^cas://sha256/([0-9a-fA-F]{64})$")
SHA256_PATTERN = re.compile(r"^sha256:([0-9a-fA-F]{64})$")


def validate_image_reference(
    value: Any,
    *,
    field_name: str,
    expected_camera_id: str,
    expected_size: tuple[int, int],
) -> dict[str, Any]:
    required = {"uri", "image_sha256", "camera_id", "width", "height"}
    if not isinstance(value, Mapping) or set(value) != required:
        raise ServiceError(
            "OBS_1101_INVALID",
            f"{field_name} must contain exactly {sorted(required)}",
            retryable=True,
        )
    uri = value["uri"]
    digest = value["image_sha256"]
    uri_match = CAS_URI_PATTERN.fullmatch(uri) if isinstance(uri, str) else None
    digest_match = SHA256_PATTERN.fullmatch(digest) if isinstance(digest, str) else None
    if uri_match is None or digest_match is None:
        raise ServiceError(
            "OBS_1101_INVALID",
            f"{field_name} must use cas://sha256/<digest> and sha256:<digest>",
            retryable=True,
        )
    if uri_match.group(1).casefold() != digest_match.group(1).casefold():
        raise ServiceError(
            "OBS_1101_INVALID",
            f"{field_name}.uri digest must match image_sha256",
            retryable=True,
        )
    if value["camera_id"] != expected_camera_id:
        raise ServiceError(
            "OBS_1101_INVALID",
            f"{field_name}.camera_id must be {expected_camera_id}",
            retryable=True,
        )
    width, height = value["width"], value["height"]
    if (width, height) != expected_size:
        raise ServiceError(
            "OBS_1101_INVALID",
            f"{field_name}.width/height must be {expected_size[0]}x{expected_size[1]}",
            retryable=True,
        )
    return {
        "uri": uri,
        "image_sha256": digest,
        "camera_id": expected_camera_id,
        "width": width,
        "height": height,
    }


def resolve_cas_path(cas_root: str | Path, image_sha256: str) -> Path:
    """Return the canonical CAS path without reading arbitrary user paths."""

    digest_match = SHA256_PATTERN.fullmatch(image_sha256)
    if digest_match is None:
        raise ValueError("image_sha256 must be sha256:<64 hex characters>")
    digest = digest_match.group(1).lower()
    root = Path(cas_root)
    return root / digest[:2] / digest
