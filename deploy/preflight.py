"""Fail-closed production preflight for the three model-service containers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


_DIGEST_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")
_ZERO_DIGEST = "sha256:" + "0" * 64
_REQUIRED_DIRECTORIES = (
    "SHARED_CAS_DIR",
    "PI05_CACHE_DIR_HOST",
    "YOLO_CACHE_DIR_HOST",
    "PI05_CHECKPOINT_DIR_HOST",
)
_WRITABLE_DIRECTORIES = (
    "PI05_CACHE_DIR_HOST",
    "YOLO_CACHE_DIR_HOST",
)
_REQUIRED_FILES = (
    "PI05_NORM_STATS_FILE_HOST",
    "YOLO_MODEL_FILE_HOST",
)
_IMAGE_IDENTITIES = (
    ("PI05_IMAGE_REPOSITORY", "PI05_IMAGE_DIGEST"),
    ("YOLO_IMAGE_REPOSITORY", "YOLO_IMAGE_DIGEST"),
)
_ARTIFACT_DIGESTS = (
    "PI05_CHECKPOINT_SHA",
    "PI05_NORM_STATS_SHA",
    "YOLO_CHECKPOINT_SHA",
    "YOLO_CLASS_MAP_SHA",
    "YOLO_CONFIG_SHA",
)


class PreflightError(RuntimeError):
    """A deployment input is unsafe or cannot reproduce the frozen release."""


def load_env_file(path: str | Path) -> dict[str, str]:
    """Load the bounded KEY=VALUE subset used by Docker Compose env files."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise PreflightError(f"env file does not exist: {source}")
    values: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        source.read_text(encoding="utf-8-sig").splitlines(),
        start=1,
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line.removeprefix("export ").strip()
        if "=" not in line:
            raise PreflightError(f"env line {line_number} must use KEY=VALUE")
        key, raw_value = line.split("=", 1)
        key = key.strip()
        value = raw_value.strip()
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", key):
            raise PreflightError(f"env line {line_number} has an invalid key")
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key in values:
            raise PreflightError(f"env line {line_number} repeats {key}")
        values[key] = value
    return values


def _require(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name, "").strip()
    if not value:
        raise PreflightError(f"{name} is required")
    if "REPLACE_WITH" in value.upper() or "<" in value or ">" in value:
        raise PreflightError(f"{name} is still a placeholder")
    return value


def _require_digest(environment: Mapping[str, str], name: str) -> str:
    value = _require(environment, name).casefold()
    if not _DIGEST_PATTERN.fullmatch(value):
        raise PreflightError(f"{name} must be sha256:<64 lowercase hex characters>")
    if value == _ZERO_DIGEST:
        raise PreflightError(f"{name} cannot be the zero digest")
    return value


def _require_port(environment: Mapping[str, str], name: str, default: int) -> int:
    raw = environment.get(name, str(default)).strip()
    try:
        port = int(raw)
    except ValueError as exc:
        raise PreflightError(f"{name} must be an integer") from exc
    if not 1 <= port <= 65535:
        raise PreflightError(f"{name} must be between 1 and 65535")
    return port


def _require_path(
    environment: Mapping[str, str],
    name: str,
    *,
    kind: str,
) -> Path:
    path = Path(_require(environment, name)).expanduser()
    if not path.is_absolute():
        raise PreflightError(f"{name} must be an absolute host path")
    resolved = path.resolve()
    if kind == "directory" and not resolved.is_dir():
        raise PreflightError(f"{name} directory does not exist: {resolved}")
    if kind == "file" and not resolved.is_file():
        raise PreflightError(f"{name} file does not exist: {resolved}")
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _pi05_directory_digest(root: Path) -> str:
    files = sorted(
        (item for item in root.rglob("*") if item.is_file()),
        key=lambda item: item.relative_to(root).as_posix(),
    )
    if not files:
        raise PreflightError(f"PI05 checkpoint directory is empty: {root}")
    manifest = hashlib.sha256(b"industrial-agent-checkpoint-manifest-v1\0")
    for item in files:
        relative = item.relative_to(root).as_posix().encode("utf-8")
        content_digest = bytes.fromhex(_sha256_file(item).removeprefix("sha256:"))
        manifest.update(len(relative).to_bytes(8, "big"))
        manifest.update(relative)
        manifest.update(item.stat().st_size.to_bytes(8, "big"))
        manifest.update(content_digest)
    return "sha256:" + manifest.hexdigest()


def _record_check(
    results: list[dict[str, str]],
    name: str,
    operation: Callable[[], str | None],
) -> None:
    try:
        detail = operation() or "validated"
    except (OSError, ValueError, PreflightError) as exc:
        results.append({"name": name, "status": "FAIL", "detail": str(exc)})
    else:
        results.append({"name": name, "status": "PASS", "detail": detail})


def validate_environment(
    environment: Mapping[str, str],
    *,
    check_files: bool = True,
) -> dict[str, Any]:
    """Validate immutable identities, host mounts and local release assets."""

    results: list[dict[str, str]] = []
    warnings: list[str] = []

    for repository_name, digest_name in _IMAGE_IDENTITIES:

        def check_image(
            repository_name: str = repository_name,
            digest_name: str = digest_name,
        ) -> str:
            repository = _require(environment, repository_name)
            if any(char.isspace() for char in repository):
                raise PreflightError(f"{repository_name} cannot contain whitespace")
            if repository.casefold().endswith(":latest"):
                raise PreflightError(f"{repository_name} cannot use the latest tag")
            digest = _require_digest(environment, digest_name)
            return f"{repository}@{digest}"

        _record_check(results, f"image:{repository_name}", check_image)

    for name in _ARTIFACT_DIGESTS:
        _record_check(
            results,
            f"digest:{name}",
            lambda name=name: _require_digest(environment, name),
        )

    for name, default in (
        ("PI05_PORT", 8101),
        ("YOLO_PORT", 8103),
    ):
        _record_check(
            results,
            f"port:{name}",
            lambda name=name, default=default: str(
                _require_port(environment, name, default)
            ),
        )

    for name in ("PI05_GPU_ID", "YOLO_GPU_ID"):
        _record_check(
            results,
            f"gpu:{name}",
            lambda name=name: _require(environment, name),
        )
    gpu_ids = [
        environment.get(name, "").strip()
        for name in (
            "PI05_GPU_ID",
            "YOLO_GPU_ID",
        )
    ]
    if all(gpu_ids) and len(set(gpu_ids)) < len(gpu_ids):
        warnings.append(
            "multiple services share a GPU; validate peak memory and concurrent "
            "YOLO/VLA inference before production"
        )

    directories: dict[str, Path] = {}
    files: dict[str, Path] = {}
    if check_files:
        for name in _REQUIRED_DIRECTORIES:

            def check_directory(name: str = name) -> str:
                path = _require_path(environment, name, kind="directory")
                directories[name] = path
                if name in _WRITABLE_DIRECTORIES and not os.access(path, os.W_OK):
                    raise PreflightError(
                        f"{name} must be writable by the container user"
                    )
                return str(path)

            _record_check(results, f"mount:{name}", check_directory)
        for name in _REQUIRED_FILES:

            def check_file(name: str = name) -> str:
                path = _require_path(environment, name, kind="file")
                files[name] = path
                return str(path)

            _record_check(results, f"mount:{name}", check_file)

        def check_pi05_checkpoint() -> str:
            root = directories.get("PI05_CHECKPOINT_DIR_HOST")
            if root is None:
                raise PreflightError("PI05 checkpoint mount did not validate")
            actual = _pi05_directory_digest(root)
            expected = _require_digest(environment, "PI05_CHECKPOINT_SHA")
            if actual != expected:
                raise PreflightError(
                    f"PI05 checkpoint digest mismatch: expected={expected}, actual={actual}"
                )
            return actual

        def check_pi05_norm() -> str:
            path = files.get("PI05_NORM_STATS_FILE_HOST")
            if path is None:
                raise PreflightError("PI05 norm-stats mount did not validate")
            actual = _sha256_file(path)
            expected = _require_digest(environment, "PI05_NORM_STATS_SHA")
            if actual != expected:
                raise PreflightError(
                    f"PI05 norm-stats digest mismatch: expected={expected}, actual={actual}"
                )
            return actual

        def check_yolo_checkpoint() -> str:
            path = files.get("YOLO_MODEL_FILE_HOST")
            if path is None:
                raise PreflightError("YOLO model mount did not validate")
            actual = _sha256_file(path)
            expected = _require_digest(environment, "YOLO_CHECKPOINT_SHA")
            if actual != expected:
                raise PreflightError(
                    f"YOLO checkpoint digest mismatch: expected={expected}, actual={actual}"
                )
            return actual

        _record_check(results, "asset:pi05-checkpoint", check_pi05_checkpoint)
        _record_check(results, "asset:pi05-norm-stats", check_pi05_norm)
        _record_check(results, "asset:yolo-checkpoint", check_yolo_checkpoint)

    status = "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL"
    return {"status": status, "checks": results, "warnings": warnings}


def _health_payload(url: str, timeout_s: float) -> dict[str, Any]:
    try:
        with urlopen(url, timeout=timeout_s) as response:  # noqa: S310
            raw = response.read(1 << 20)
    except HTTPError as exc:
        body = exc.read(4096).decode("utf-8", errors="replace")
        raise PreflightError(f"health returned HTTP {exc.code}: {body}") from exc
    except URLError as exc:
        raise PreflightError(f"health request failed: {exc.reason}") from exc
    try:
        payload = json.loads(raw)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise PreflightError("health response is not valid JSON") from exc
    if not isinstance(payload, dict):
        raise PreflightError("health response must be a JSON object")
    return payload


def validate_services(
    environment: Mapping[str, str],
    *,
    timeout_s: float = 5.0,
) -> dict[str, Any]:
    """Verify that running services expose the exact configured artifacts."""

    host = environment.get("MODEL_HEALTH_HOST", "127.0.0.1").strip()
    if not host:
        host = "127.0.0.1"
    definitions = (
        (
            "pi05",
            _require_port(environment, "PI05_PORT", 8101),
            {
                "checkpoint_sha": "PI05_CHECKPOINT_SHA",
                "norm_stats_sha": "PI05_NORM_STATS_SHA",
            },
        ),
        (
            "yolo",
            _require_port(environment, "YOLO_PORT", 8103),
            {
                "checkpoint_sha": "YOLO_CHECKPOINT_SHA",
                "class_map_sha": "YOLO_CLASS_MAP_SHA",
                "config_sha": "YOLO_CONFIG_SHA",
            },
        ),
    )
    results: list[dict[str, str]] = []
    for service, port, digest_fields in definitions:
        url = f"http://{host}:{port}/health"

        def check_service(
            service: str = service,
            url: str = url,
            digest_fields: Mapping[str, str] = digest_fields,
        ) -> str:
            payload = _health_payload(url, timeout_s)
            if payload.get("service") != service:
                raise PreflightError(
                    f"expected service={service!r}, got {payload.get('service')!r}"
                )
            if payload.get("status") != "ready":
                raise PreflightError(
                    f"{service} is not ready: {payload.get('status')!r}"
                )
            device = payload.get("device")
            if isinstance(device, dict) and device.get("mode") != "real":
                raise PreflightError(f"{service} health reports non-real mode")
            for response_field, environment_name in digest_fields.items():
                expected = _require_digest(environment, environment_name)
                actual = str(payload.get(response_field, "")).casefold()
                if actual != expected:
                    raise PreflightError(
                        f"{service}.{response_field} mismatch: "
                        f"expected={expected}, actual={actual or '<missing>'}"
                    )
            return url

        _record_check(results, f"health:{service}", check_service)
    status = "PASS" if all(item["status"] == "PASS" for item in results) else "FAIL"
    return {"status": status, "checks": results, "warnings": []}


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate production model assets and optional live health endpoints."
    )
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument(
        "--phase",
        choices=("assets", "services"),
        default="assets",
        help="services performs asset checks and then validates all live /health endpoints",
    )
    parser.add_argument("--health-timeout-s", type=float, default=5.0)
    parser.add_argument("--output", type=Path)
    return parser


def main() -> int:
    args = _argument_parser().parse_args()
    try:
        environment = load_env_file(args.env_file)
        assets = validate_environment(environment)
        report: dict[str, Any] = {
            "schema_version": "1.0",
            "phase": args.phase,
            "assets": assets,
        }
        if args.phase == "services" and assets["status"] == "PASS":
            report["services"] = validate_services(
                environment,
                timeout_s=args.health_timeout_s,
            )
        passed = assets["status"] == "PASS" and (
            args.phase != "services"
            or report.get("services", {}).get("status") == "PASS"
        )
        report["status"] = "PASS" if passed else "FAIL"
    except (OSError, ValueError, PreflightError) as exc:
        report = {
            "schema_version": "1.0",
            "phase": args.phase,
            "status": "FAIL",
            "error": str(exc),
        }
        passed = False
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        destination = args.output.expanduser().resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(rendered + "\n", encoding="utf-8")
    return 0 if passed else 2


if __name__ == "__main__":
    sys.exit(main())
