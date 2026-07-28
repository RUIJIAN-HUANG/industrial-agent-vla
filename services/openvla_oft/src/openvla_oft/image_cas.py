"""Standalone content-addressed RGB image storage for OpenVLA-OFT."""

from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from time import sleep
from typing import Any, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

from .exceptions import ServiceError

CAS_URI_PATTERN = re.compile(r"^cas://sha256/([0-9a-fA-F]{64})$")
SHA256_PATTERN = re.compile(r"^sha256:([0-9a-fA-F]{64})$")
IMAGE_CAS_ENV = "INDUSTRIAL_AGENT_CAS_ROOT"


@dataclass(frozen=True)
class ImageCasConfig:
    """Frozen local CAS layout and resource limits."""

    root: Path
    max_blob_bytes: int = 16 * 1024 * 1024
    max_pixels: int = 4_194_304
    cache_max_bytes: int = 64 * 1024 * 1024
    missing_retry_count: int = 1
    missing_retry_delay_ms: int = 25

    @classmethod
    def from_mapping(
        cls,
        value: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ImageCasConfig:
        if not isinstance(value, Mapping):
            raise ValueError("image_cas must be an object")
        expected_fields = {
            "root",
            "layout",
            "encoding",
            "digest_scope",
            "max_blob_bytes",
            "max_pixels",
            "cache_max_bytes",
            "missing_retry_count",
            "missing_retry_delay_ms",
        }
        if set(value) != expected_fields:
            raise ValueError(
                f"image_cas must contain exactly {sorted(expected_fields)}"
            )
        environment = os.environ if environ is None else environ
        configured_root = environment.get(IMAGE_CAS_ENV, value.get("root"))
        if not isinstance(configured_root, str) or not configured_root.strip():
            raise ValueError(
                f"image_cas.root or {IMAGE_CAS_ENV} must be a non-empty path"
            )
        if value.get("layout") != "sha256-v1":
            raise ValueError("image_cas.layout must be sha256-v1")
        if value.get("encoding") != "png":
            raise ValueError("image_cas.encoding must be png")
        if value.get("digest_scope") != "encoded_bytes":
            raise ValueError("image_cas.digest_scope must be encoded_bytes")
        integer_fields = {
            "max_blob_bytes": (1, 128 * 1024 * 1024),
            "max_pixels": (1, 100_000_000),
            "cache_max_bytes": (0, 1024 * 1024 * 1024),
            "missing_retry_count": (0, 2),
            "missing_retry_delay_ms": (0, 1000),
        }
        parsed: dict[str, int] = {}
        for field, (minimum, maximum) in integer_fields.items():
            raw = value.get(field)
            if (
                isinstance(raw, bool)
                or not isinstance(raw, int)
                or raw < minimum
                or raw > maximum
            ):
                raise ValueError(
                    f"image_cas.{field} must be an integer in [{minimum}, {maximum}]"
                )
            parsed[field] = raw
        return cls(root=Path(configured_root).expanduser(), **parsed)


@dataclass(frozen=True)
class ResolvedRgbFrame:
    """A verified immutable RGB frame returned to the model service."""

    rgb: np.ndarray
    image_sha256: str
    camera_id: str
    width: int
    height: int
    encoded_byte_length: int


class ImageCas:
    """Read and write canonical PNG frames in a local shared CAS volume."""

    def __init__(self, config: ImageCasConfig):
        self.config = config

    @classmethod
    def from_mapping(
        cls,
        config: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ImageCas:
        return cls(ImageCasConfig.from_mapping(config, environ=environ))

    def assert_ready(self, *, writable: bool) -> None:
        root = self.config.root
        if writable:
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ServiceError(
                    "EXEC_2101_UNAVAILABLE",
                    f"CAS root cannot be created: {root}",
                    retryable=True,
                ) from exc
        if not root.is_dir():
            raise ServiceError(
                "EXEC_2101_UNAVAILABLE",
                f"CAS root is not a directory: {root}",
                retryable=True,
            )
        if not os.access(root, os.R_OK):
            raise ServiceError(
                "EXEC_2101_UNAVAILABLE",
                f"CAS root is not readable: {root}",
                retryable=True,
            )
        if writable and not os.access(root, os.W_OK):
            raise ServiceError(
                "EXEC_2101_UNAVAILABLE",
                f"CAS root is not writable: {root}",
                retryable=True,
            )

    def write_rgb(self, rgb: np.ndarray, *, camera_id: str) -> dict[str, Any]:
        self.assert_ready(writable=True)
        frame = _canonical_rgb_array(rgb, max_pixels=self.config.max_pixels)
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ServiceError(
                "EXEC_2103_BAD_RESPONSE",
                "camera_id must be a non-empty string",
            )
        buffer = BytesIO()
        Image.fromarray(frame, mode="RGB").save(
            buffer,
            format="PNG",
            optimize=False,
            compress_level=1,
        )
        encoded = buffer.getvalue()
        if len(encoded) > self.config.max_blob_bytes:
            raise ServiceError(
                "EXEC_2101_UNAVAILABLE",
                "encoded PNG exceeds image_cas.max_blob_bytes",
                retryable=True,
            )
        digest = sha256(encoded).hexdigest()
        target = self._path_for_digest(digest)
        self._atomic_write(target, encoded)
        height, width = frame.shape[:2]
        return {
            "uri": f"cas://sha256/{digest}",
            "image_sha256": f"sha256:{digest}",
            "camera_id": camera_id,
            "width": int(width),
            "height": int(height),
        }

    def resolve_rgb(
        self,
        image: Mapping[str, Any],
        *,
        expected_camera_id: str | None = None,
        expected_size: tuple[int, int] | None = None,
    ) -> ResolvedRgbFrame:
        reference = _validate_reference(image)
        digest = _reference_digest(reference)
        if (
            expected_camera_id is not None
            and reference["camera_id"] != expected_camera_id
        ):
            raise ServiceError(
                "OBS_1101_INVALID",
                f"expected camera {expected_camera_id}, got {reference['camera_id']}",
                retryable=True,
            )
        if (
            expected_size is not None
            and (
                reference["width"],
                reference["height"],
            )
            != expected_size
        ):
            raise ServiceError(
                "OBS_1101_INVALID",
                "ImageReference dimensions do not match the expected camera size",
                retryable=True,
            )
        self.assert_ready(writable=False)
        encoded = self._read_verified_blob(digest)
        rgb = self._decode_png(encoded)
        height, width = rgb.shape[:2]
        if (width, height) != (reference["width"], reference["height"]):
            raise ServiceError(
                "OBS_1101_INVALID",
                "decoded image dimensions do not match ImageReference",
                retryable=True,
            )
        return ResolvedRgbFrame(
            rgb=rgb,
            image_sha256=reference["image_sha256"].lower(),
            camera_id=reference["camera_id"],
            width=width,
            height=height,
            encoded_byte_length=len(encoded),
        )

    def _decode_png(self, encoded: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(encoded)) as image:
                if image.format != "PNG":
                    raise ServiceError(
                        "OBS_1101_INVALID",
                        "CAS image is not a PNG",
                        retryable=True,
                    )
                width, height = image.size
                if width < 1 or height < 1 or width * height > self.config.max_pixels:
                    raise ServiceError(
                        "EXEC_2101_UNAVAILABLE",
                        "decoded image exceeds image_cas.max_pixels",
                        retryable=True,
                    )
                if image.mode != "RGB":
                    raise ServiceError(
                        "OBS_1101_INVALID",
                        f"canonical CAS PNG must use RGB mode, got {image.mode}",
                        retryable=True,
                    )
                image.load()
                rgb = np.asarray(image, dtype=np.uint8).copy()
        except ServiceError:
            raise
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise ServiceError(
                "OBS_1101_INVALID",
                "CAS image could not be decoded as a safe RGB PNG",
                retryable=True,
            ) from exc
        return _freeze_array(rgb)

    def _read_verified_blob(self, digest: str) -> bytes:
        target = self._path_for_digest(digest)
        attempts = self.config.missing_retry_count + 1
        for attempt in range(attempts):
            try:
                if target.is_symlink():
                    raise ServiceError(
                        "EXEC_2101_UNAVAILABLE",
                        "CAS blobs must not be symbolic links",
                        retryable=True,
                    )
                encoded = _read_limited(target, self.config.max_blob_bytes)
                break
            except FileNotFoundError as exc:
                if attempt + 1 >= attempts:
                    raise ServiceError(
                        "OBS_1101_INVALID",
                        f"CAS image does not exist: sha256:{digest}",
                        retryable=True,
                    ) from exc
                sleep(self.config.missing_retry_delay_ms / 1000)
            except OSError as exc:
                raise ServiceError(
                    "EXEC_2101_UNAVAILABLE",
                    "CAS image could not be read",
                    retryable=True,
                ) from exc
        else:  # pragma: no cover - loop always returns or raises
            raise AssertionError("unreachable CAS read state")
        actual = sha256(encoded).hexdigest()
        if actual != digest:
            raise ServiceError(
                "OBS_1101_INVALID",
                f"CAS byte digest mismatch: expected {digest}, got {actual}",
                retryable=True,
            )
        return encoded

    def _path_for_digest(self, digest: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ServiceError(
                "OBS_1101_INVALID",
                "CAS digest must contain 64 lowercase hexadecimal characters",
                retryable=True,
            )
        return self.config.root / "sha256" / digest[:2] / digest

    def _atomic_write(self, target: Path, encoded: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ServiceError(
                "EXEC_2101_UNAVAILABLE",
                "CAS blobs must not be symbolic links",
                retryable=True,
            )
        if target.exists():
            existing = _read_limited(target, self.config.max_blob_bytes)
            if existing != encoded:
                raise ServiceError(
                    "OBS_1101_INVALID",
                    "existing CAS blob does not match its content address",
                    retryable=True,
                )
            return
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                prefix=".cas-",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as temporary:
                temporary.write(encoded)
                temporary.flush()
                os.fsync(temporary.fileno())
                temporary_path = Path(temporary.name)
            os.replace(temporary_path, target)
        except OSError as exc:
            raise ServiceError(
                "EXEC_2101_UNAVAILABLE",
                "CAS image could not be written atomically",
                retryable=True,
            ) from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass


def validate_image_reference(
    value: Any,
    *,
    field_name: str,
    expected_camera_id: str,
    expected_size: tuple[int, int],
) -> dict[str, Any]:
    reference = _validate_reference(value)
    if reference["camera_id"] != expected_camera_id:
        raise ServiceError(
            "OBS_1101_INVALID",
            f"{field_name}.camera_id must be {expected_camera_id}",
            retryable=True,
        )
    if (reference["width"], reference["height"]) != expected_size:
        raise ServiceError(
            "OBS_1101_INVALID",
            f"{field_name}.width/height must be {expected_size[0]}x{expected_size[1]}",
            retryable=True,
        )
    return reference


def _validate_reference(image: Mapping[str, Any]) -> dict[str, Any]:
    required = {"uri", "image_sha256", "camera_id", "width", "height"}
    if not isinstance(image, Mapping) or set(image) != required:
        raise ServiceError(
            "OBS_1101_INVALID",
            f"image reference must contain exactly {sorted(required)}",
            retryable=True,
        )
    uri = image["uri"]
    digest = image["image_sha256"]
    uri_match = CAS_URI_PATTERN.fullmatch(uri) if isinstance(uri, str) else None
    digest_match = SHA256_PATTERN.fullmatch(digest) if isinstance(digest, str) else None
    if uri_match is None or digest_match is None:
        raise ServiceError(
            "OBS_1101_INVALID",
            "image reference must use cas://sha256/<digest> and sha256:<digest>",
            retryable=True,
        )
    camera_id = image["camera_id"]
    width = image["width"]
    height = image["height"]
    if not isinstance(camera_id, str) or not camera_id.strip():
        raise ServiceError(
            "OBS_1101_INVALID",
            "image.camera_id must be a non-empty string",
            retryable=True,
        )
    if isinstance(width, bool) or not isinstance(width, int) or width < 1:
        raise ServiceError(
            "OBS_1101_INVALID",
            "image.width must be a positive integer",
            retryable=True,
        )
    if isinstance(height, bool) or not isinstance(height, int) or height < 1:
        raise ServiceError(
            "OBS_1101_INVALID",
            "image.height must be a positive integer",
            retryable=True,
        )
    if uri_match.group(1).casefold() != digest_match.group(1).casefold():
        raise ServiceError(
            "OBS_1101_INVALID",
            "image.uri digest must match image_sha256",
            retryable=True,
        )
    return {
        "uri": uri,
        "image_sha256": digest,
        "camera_id": camera_id,
        "width": width,
        "height": height,
    }


def _reference_digest(reference: Mapping[str, Any]) -> str:
    return reference["image_sha256"].removeprefix("sha256:").lower()


def _canonical_rgb_array(rgb: np.ndarray, *, max_pixels: int) -> np.ndarray:
    if not isinstance(rgb, np.ndarray):
        raise ServiceError(
            "OBS_1101_INVALID",
            "RGB frame must be a numpy.ndarray",
            retryable=True,
        )
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ServiceError(
            "OBS_1101_INVALID",
            "RGB frame must have dtype uint8 and shape HxWx3",
            retryable=True,
        )
    height, width = rgb.shape[:2]
    if height < 1 or width < 1 or height * width > max_pixels:
        raise ServiceError(
            "EXEC_2101_UNAVAILABLE",
            "RGB frame exceeds image_cas.max_pixels",
            retryable=True,
        )
    return np.ascontiguousarray(rgb)


def _freeze_array(rgb: np.ndarray) -> np.ndarray:
    rgb.setflags(write=False)
    return rgb


def _read_limited(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ServiceError(
            "EXEC_2101_UNAVAILABLE",
            "CAS blob exceeds image_cas.max_blob_bytes",
            retryable=True,
        )
    return content
