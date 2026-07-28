"""Shared content-addressed RGB image storage.

The Supervisor transports immutable :class:`ImageReference` values.  Camera
producers and image-consuming services share this implementation so that
π0.5, OpenVLA-OFT, and YOLO resolve exactly the same verified frame.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from io import BytesIO
import os
from pathlib import Path
import re
import tempfile
from threading import RLock
from time import sleep
from typing import Any, ClassVar, Mapping

import numpy as np
from PIL import Image, UnidentifiedImageError

from .errors import AgentError, FailureCode, ImageCasError
from .perception import ImageReference


CAS_URI_PATTERN = re.compile(r"^cas://sha256/([0-9a-fA-F]{64})$")
SHA256_PATTERN = re.compile(r"^sha256:([0-9a-fA-F]{64})$")
CAS_ROOT_ENV = "INDUSTRIAL_AGENT_CAS_ROOT"


@dataclass(frozen=True)
class ImageCasConfig:
    """Frozen local CAS layout and resource limits.

    Layout, encoding, and digest scope are protocol constants rather than
    deployment-tunable dataclass fields. ``from_mapping`` still validates their
    serialized config values fail-closed.
    """

    LAYOUT: ClassVar[str] = "sha256-v1"
    ENCODING: ClassVar[str] = "png"
    DIGEST_SCOPE: ClassVar[str] = "encoded_bytes"

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
        """Build configuration, allowing only the CAS root to be overridden."""

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
        configured_root = environment.get(CAS_ROOT_ENV, value.get("root"))
        if not isinstance(configured_root, str) or not configured_root.strip():
            raise ValueError(
                f"image_cas.root or {CAS_ROOT_ENV} must be a non-empty path"
            )
        if value.get("layout") != cls.LAYOUT:
            raise ValueError(f"image_cas.layout must be {cls.LAYOUT}")
        if value.get("encoding") != cls.ENCODING:
            raise ValueError(f"image_cas.encoding must be {cls.ENCODING}")
        if value.get("digest_scope") != cls.DIGEST_SCOPE:
            raise ValueError(f"image_cas.digest_scope must be {cls.DIGEST_SCOPE}")
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
    """A verified immutable RGB frame returned to a model service."""

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
        self._cache: OrderedDict[str, tuple[np.ndarray, int]] = OrderedDict()
        self._cache_bytes = 0
        self._cache_lock = RLock()

    @classmethod
    def from_agent_config(
        cls,
        config: Mapping[str, Any],
        *,
        environ: Mapping[str, str] | None = None,
    ) -> ImageCas:
        """Construct from the top-level ``image_cas`` agent configuration."""

        raw = config.get("image_cas")
        return cls(ImageCasConfig.from_mapping(raw, environ=environ))

    def assert_ready(self, *, writable: bool) -> None:
        """Fail closed when the configured volume cannot serve its role."""

        root = self.config.root
        if writable:
            try:
                root.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                raise ImageCasError(
                    FailureCode.CAS_UNAVAILABLE,
                    f"CAS root cannot be created: {root}",
                    retryable=True,
                ) from exc
        if not root.is_dir():
            raise ImageCasError(
                FailureCode.CAS_UNAVAILABLE,
                f"CAS root is not a directory: {root}",
                retryable=True,
            )
        if not os.access(root, os.R_OK):
            raise ImageCasError(
                FailureCode.CAS_UNAVAILABLE,
                f"CAS root is not readable: {root}",
                retryable=True,
            )
        if writable and not os.access(root, os.W_OK):
            raise ImageCasError(
                FailureCode.CAS_UNAVAILABLE,
                f"CAS root is not writable: {root}",
                retryable=True,
            )

    def write_rgb(self, rgb: np.ndarray, *, camera_id: str) -> ImageReference:
        """Encode one RGB array as PNG, write atomically, and return its reference."""

        self.assert_ready(writable=True)
        frame = _canonical_rgb_array(rgb, max_pixels=self.config.max_pixels)
        if not isinstance(camera_id, str) or not camera_id.strip():
            raise ImageCasError(
                FailureCode.CAS_METADATA_MISMATCH,
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
            raise ImageCasError(
                FailureCode.CAS_LIMIT_EXCEEDED,
                "encoded PNG exceeds image_cas.max_blob_bytes",
            )
        digest = sha256(encoded).hexdigest()
        target = self._path_for_digest(digest)
        self._atomic_write(target, encoded)
        height, width = frame.shape[:2]
        return ImageReference(
            uri=f"cas://sha256/{digest}",
            image_sha256=f"sha256:{digest}",
            camera_id=camera_id,
            width=int(width),
            height=int(height),
        )

    def resolve_rgb(
        self,
        image: ImageReference | Mapping[str, Any],
        *,
        expected_camera_id: str | None = None,
        expected_size: tuple[int, int] | None = None,
    ) -> ResolvedRgbFrame:
        """Resolve and verify one reference into an immutable RGB array.

        This method never returns a placeholder.  Missing, corrupt, oversized,
        or metadata-inconsistent frames fail closed with a stable CAS error.
        """

        try:
            reference = (
                image
                if isinstance(image, ImageReference)
                else ImageReference.from_dict(image)
            )
        except AgentError as exc:
            raise ImageCasError(
                FailureCode.CAS_METADATA_MISMATCH,
                f"invalid ImageReference: {exc}",
            ) from exc
        digest = _reference_digest(reference)
        if expected_camera_id is not None and reference.camera_id != expected_camera_id:
            raise ImageCasError(
                FailureCode.CAS_METADATA_MISMATCH,
                f"expected camera {expected_camera_id}, got {reference.camera_id}",
            )
        if (
            expected_size is not None
            and (
                reference.width,
                reference.height,
            )
            != expected_size
        ):
            raise ImageCasError(
                FailureCode.CAS_METADATA_MISMATCH,
                "ImageReference dimensions do not match the expected camera size",
            )

        cached = self._get_cached(digest)
        if cached is None:
            encoded = self._read_verified_blob(digest)
            rgb = self._decode_png(encoded)
            self._put_cached(digest, rgb, len(encoded))
        else:
            rgb, encoded_length = cached
            return self._resolved(reference, rgb, encoded_length)
        return self._resolved(reference, rgb, len(encoded))

    def _resolved(
        self,
        reference: ImageReference,
        rgb: np.ndarray,
        encoded_length: int,
    ) -> ResolvedRgbFrame:
        height, width = rgb.shape[:2]
        if (width, height) != (reference.width, reference.height):
            raise ImageCasError(
                FailureCode.CAS_METADATA_MISMATCH,
                "decoded image dimensions do not match ImageReference",
            )
        return ResolvedRgbFrame(
            rgb=rgb,
            image_sha256=reference.image_sha256.lower(),
            camera_id=reference.camera_id,
            width=width,
            height=height,
            encoded_byte_length=encoded_length,
        )

    def _decode_png(self, encoded: bytes) -> np.ndarray:
        try:
            with Image.open(BytesIO(encoded)) as image:
                if image.format != "PNG":
                    raise ImageCasError(
                        FailureCode.CAS_DECODE_FAILED,
                        "CAS image is not a PNG",
                    )
                width, height = image.size
                if width < 1 or height < 1 or width * height > self.config.max_pixels:
                    raise ImageCasError(
                        FailureCode.CAS_LIMIT_EXCEEDED,
                        "decoded image exceeds image_cas.max_pixels",
                    )
                if image.mode != "RGB":
                    raise ImageCasError(
                        FailureCode.CAS_DECODE_FAILED,
                        f"canonical CAS PNG must use RGB mode, got {image.mode}",
                    )
                image.load()
                rgb = np.asarray(image, dtype=np.uint8).copy()
        except ImageCasError:
            raise
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError) as exc:
            raise ImageCasError(
                FailureCode.CAS_DECODE_FAILED,
                "CAS image could not be decoded as a safe RGB PNG",
            ) from exc
        return _freeze_array(rgb)

    def _read_verified_blob(self, digest: str) -> bytes:
        target = self._path_for_digest(digest)
        attempts = self.config.missing_retry_count + 1
        for attempt in range(attempts):
            try:
                if target.is_symlink():
                    raise ImageCasError(
                        FailureCode.CAS_UNAVAILABLE,
                        "CAS blobs must not be symbolic links",
                    )
                if target.exists():
                    _require_inside_root(target, self.config.root)
                encoded = _read_limited(target, self.config.max_blob_bytes)
                break
            except FileNotFoundError as exc:
                if attempt + 1 >= attempts:
                    raise ImageCasError(
                        FailureCode.CAS_NOT_FOUND,
                        f"CAS image does not exist: sha256:{digest}",
                        retryable=True,
                    ) from exc
                sleep(self.config.missing_retry_delay_ms / 1000)
            except OSError as exc:
                raise ImageCasError(
                    FailureCode.CAS_UNAVAILABLE,
                    "CAS image could not be read",
                    retryable=True,
                ) from exc
        else:  # pragma: no cover - loop always returns or raises
            raise AssertionError("unreachable CAS read state")
        actual = sha256(encoded).hexdigest()
        if actual != digest:
            raise ImageCasError(
                FailureCode.CAS_DIGEST_MISMATCH,
                f"CAS byte digest mismatch: expected {digest}, got {actual}",
            )
        return encoded

    def _path_for_digest(self, digest: str) -> Path:
        if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
            raise ImageCasError(
                FailureCode.CAS_METADATA_MISMATCH,
                "CAS digest must contain 64 lowercase hexadecimal characters",
            )
        return self.config.root / "sha256" / digest[:2] / digest

    def _atomic_write(self, target: Path, encoded: bytes) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        _require_inside_root(target.parent, self.config.root)
        if target.is_symlink():
            raise ImageCasError(
                FailureCode.CAS_UNAVAILABLE,
                "CAS blobs must not be symbolic links",
            )
        if target.exists():
            existing = _read_limited(target, self.config.max_blob_bytes)
            if existing != encoded:
                raise ImageCasError(
                    FailureCode.CAS_DIGEST_MISMATCH,
                    "existing CAS blob does not match its content address",
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
            raise ImageCasError(
                FailureCode.CAS_UNAVAILABLE,
                "CAS image could not be written atomically",
                retryable=True,
            ) from exc
        finally:
            if temporary_path is not None and temporary_path.exists():
                try:
                    temporary_path.unlink()
                except OSError:
                    pass

    def _get_cached(self, digest: str) -> tuple[np.ndarray, int] | None:
        with self._cache_lock:
            value = self._cache.get(digest)
            if value is not None:
                self._cache.move_to_end(digest)
            return value

    def _put_cached(self, digest: str, rgb: np.ndarray, encoded_length: int) -> None:
        limit = self.config.cache_max_bytes
        size = int(rgb.nbytes)
        if limit == 0 or size > limit:
            return
        with self._cache_lock:
            previous = self._cache.pop(digest, None)
            if previous is not None:
                self._cache_bytes -= int(previous[0].nbytes)
            self._cache[digest] = (rgb, encoded_length)
            self._cache_bytes += size
            while self._cache_bytes > limit:
                _, (removed, _) = self._cache.popitem(last=False)
                self._cache_bytes -= int(removed.nbytes)


def _reference_digest(reference: ImageReference) -> str:
    uri_match = CAS_URI_PATTERN.fullmatch(reference.uri)
    sha_match = SHA256_PATTERN.fullmatch(reference.image_sha256)
    if uri_match is None or sha_match is None:
        raise ImageCasError(
            FailureCode.CAS_METADATA_MISMATCH,
            "image reference must use matching SHA-256 CAS identifiers",
        )
    uri_digest = uri_match.group(1).lower()
    sha_digest = sha_match.group(1).lower()
    if uri_digest != sha_digest:
        raise ImageCasError(
            FailureCode.CAS_METADATA_MISMATCH,
            "CAS URI digest does not match image_sha256",
        )
    return uri_digest


def _canonical_rgb_array(rgb: np.ndarray, *, max_pixels: int) -> np.ndarray:
    if not isinstance(rgb, np.ndarray):
        raise ImageCasError(
            FailureCode.CAS_METADATA_MISMATCH,
            "RGB frame must be a numpy.ndarray",
        )
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ImageCasError(
            FailureCode.CAS_METADATA_MISMATCH,
            "RGB frame must have dtype uint8 and shape HxWx3",
        )
    height, width = rgb.shape[:2]
    if height < 1 or width < 1 or height * width > max_pixels:
        raise ImageCasError(
            FailureCode.CAS_LIMIT_EXCEEDED,
            "RGB frame exceeds image_cas.max_pixels",
        )
    return np.ascontiguousarray(rgb)


def _freeze_array(rgb: np.ndarray) -> np.ndarray:
    rgb.setflags(write=False)
    return rgb


def _read_limited(path: Path, limit: int) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise ImageCasError(
            FailureCode.CAS_LIMIT_EXCEEDED,
            "CAS blob exceeds image_cas.max_blob_bytes",
        )
    return content


def _require_inside_root(path: Path, root: Path) -> None:
    try:
        path.resolve(strict=True).relative_to(root.resolve(strict=True))
    except (FileNotFoundError, OSError, ValueError) as exc:
        raise ImageCasError(
            FailureCode.CAS_UNAVAILABLE,
            "CAS path escapes or cannot be resolved beneath the configured root",
        ) from exc
