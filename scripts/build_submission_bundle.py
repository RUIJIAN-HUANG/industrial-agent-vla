"""Build and verify a self-contained competition submission directory.

The normal Git repository deliberately excludes model weights.  This script
creates a separate delivery directory containing a tracked source snapshot,
the complete pi0.5 checkpoint directory, norm statistics, the YOLO checkpoint,
runtime configuration, immutable digests, and launch helpers.

The output directory must not already exist.  Model files are copied byte for
byte; a final bundle therefore needs enough free space for every supplied
artifact.  Use ``verify`` again after moving or extracting the bundle.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from deploy.preflight import _pi05_directory_digest, _sha256_file  # noqa: E402


SCHEMA_VERSION = "1.0"
BUNDLE_STATUS = "CANDIDATE"
DEFAULT_PI05_IMAGE = "industrial-agent/pi05:submission"
DEFAULT_YOLO_IMAGE = "industrial-agent/yolo:submission"
_COPY_BUFFER_BYTES = 16 << 20
_MANDATORY_BUNDLE_CODE = (
    Path("deploy/compose.models.submission.yaml"),
    Path("scripts/build_submission_bundle.py"),
    Path("simulation/run_v2_competition_ui.py"),
    Path("simulation/v2_competition_controller.py"),
    Path("simulation/v2_competition_window.py"),
)


class BundleError(RuntimeError):
    """The requested submission bundle cannot be built or verified safely."""


@dataclass(frozen=True)
class ArtifactInfo:
    relative_path: str
    digest: str
    size_bytes: int
    file_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "sha256": self.digest,
            "size_bytes": self.size_bytes,
            "file_count": self.file_count,
        }


def _run_git(repo_root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise BundleError(f"git {' '.join(args)} failed: {detail}")
    return result.stdout


def _git_snapshot(repo_root: Path) -> dict[str, Any]:
    commit = _run_git(repo_root, "rev-parse", "HEAD").strip()
    branch = _run_git(repo_root, "branch", "--show-current").strip() or "DETACHED"
    status = _run_git(repo_root, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": commit,
        "branch": branch,
        "tracked_worktree_dirty": bool(status.strip()),
    }


def _tracked_files(repo_root: Path) -> tuple[Path, ...]:
    raw = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=False,
        capture_output=True,
    )
    if raw.returncode != 0:
        raise BundleError("git ls-files failed")
    paths: list[Path] = []
    for item in raw.stdout.split(b"\0"):
        if not item:
            continue
        relative = Path(os.fsdecode(item))
        source = repo_root / relative
        if not source.is_file():
            raise BundleError(f"tracked source file is missing: {relative}")
        paths.append(relative)
    for relative in _MANDATORY_BUNDLE_CODE:
        if (repo_root / relative).is_file() and relative not in paths:
            paths.append(relative)
    return tuple(sorted(paths, key=lambda item: item.as_posix()))


def _file_size(path: Path) -> int:
    if not path.is_file():
        raise BundleError(f"required file does not exist: {path}")
    return path.stat().st_size


def _directory_files(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        raise BundleError(f"required directory does not exist: {root}")
    files = tuple(
        sorted(
            (path for path in root.rglob("*") if path.is_file()),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )
    if not files:
        raise BundleError(f"required directory is empty: {root}")
    return files


def _directory_size(root: Path) -> tuple[int, int]:
    files = _directory_files(root)
    return sum(path.stat().st_size for path in files), len(files)


def _tree_digest(root: Path, namespace: bytes) -> str:
    files = _directory_files(root)
    digest = hashlib.sha256(namespace + b"\0")
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        content = bytes.fromhex(_sha256_file(path).removeprefix("sha256:"))
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(path.stat().st_size.to_bytes(8, "big"))
        digest.update(content)
    return "sha256:" + digest.hexdigest()


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("rb") as reader, destination.open("wb") as writer:
        shutil.copyfileobj(reader, writer, length=_COPY_BUFFER_BYTES)
    shutil.copystat(source, destination, follow_symlinks=True)


def _copy_directory(source: Path, destination: Path) -> None:
    for item in _directory_files(source):
        _copy_file(item, destination / item.relative_to(source))


def _copy_code_snapshot(
    repo_root: Path,
    destination: Path,
    *,
    repo_files: Sequence[Path] | None = None,
) -> tuple[int, int]:
    files = tuple(repo_files) if repo_files is not None else _tracked_files(repo_root)
    total = 0
    for relative in files:
        source = repo_root / relative
        if not source.is_file():
            raise BundleError(f"source snapshot file is missing: {relative}")
        _copy_file(source, destination / relative)
        total += source.stat().st_size
    return total, len(files)


def _available_space_parent(output: Path) -> Path:
    candidate = output.resolve(strict=False)
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise BundleError(f"cannot resolve output filesystem: {output}")
        candidate = parent
    return candidate


def _require_free_space(output: Path, required_bytes: int) -> None:
    parent = _available_space_parent(output)
    free = shutil.disk_usage(parent).free
    reserve = max(1 << 30, required_bytes // 20)
    if free < required_bytes + reserve:
        raise BundleError(
            "not enough free space for a self-contained bundle: "
            f"required={required_bytes + reserve} available={free} path={parent}"
        )


def _artifact_file(path: Path, bundle_root: Path) -> ArtifactInfo:
    return ArtifactInfo(
        relative_path=path.relative_to(bundle_root).as_posix(),
        digest=_sha256_file(path),
        size_bytes=path.stat().st_size,
        file_count=1,
    )


def _artifact_pi05(path: Path, bundle_root: Path) -> ArtifactInfo:
    size, count = _directory_size(path)
    return ArtifactInfo(
        relative_path=path.relative_to(bundle_root).as_posix(),
        digest=_pi05_directory_digest(path),
        size_bytes=size,
        file_count=count,
    )


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _runtime_env(bundle_root: Path, manifest: dict[str, Any]) -> str:
    artifacts = manifest["artifacts"]
    pi05 = artifacts["pi05_checkpoint"]
    norm = artifacts["pi05_norm_stats"]
    yolo = artifacts["yolo_checkpoint"]
    class_map = artifacts["yolo_class_map"]
    yolo_config = artifacts["yolo_config"]
    paths = manifest["runtime"]["paths"]

    def absolute(relative: str) -> str:
        return str((bundle_root / relative).resolve())

    values = {
        "MODEL_BIND_IP": "127.0.0.1",
        "MODEL_HEALTH_HOST": "127.0.0.1",
        "PI05_PORT": "8101",
        "YOLO_PORT": "8103",
        "PI05_IMAGE_REPOSITORY": manifest["runtime"]["pi05_image"],
        "PI05_IMAGE_DIGEST": manifest["runtime"]["pi05_image_digest"],
        "YOLO_IMAGE_REPOSITORY": manifest["runtime"]["yolo_image"],
        "YOLO_IMAGE_DIGEST": manifest["runtime"]["yolo_image_digest"],
        "PI05_GPU_IDS": manifest["runtime"]["pi05_gpu_ids"],
        "PI05_GPU_ID": manifest["runtime"]["pi05_gpu_ids"].split(",")[0],
        "YOLO_GPU_ID": manifest["runtime"]["yolo_gpu_id"],
        "PI05_GPU_MEMORY_FRACTION": "0.85",
        "SHARED_CAS_DIR": absolute(paths["shared_cas"]),
        "PI05_CACHE_DIR_HOST": absolute(paths["pi05_cache"]),
        "YOLO_CACHE_DIR_HOST": absolute(paths["yolo_cache"]),
        "PI05_CHECKPOINT_DIR_HOST": absolute(pi05["relative_path"]),
        "PI05_NORM_STATS_FILE_HOST": absolute(norm["relative_path"]),
        "PI05_CHECKPOINT_SHA": pi05["sha256"],
        "PI05_NORM_STATS_SHA": norm["sha256"],
        "YOLO_MODEL_FILE_HOST": absolute(yolo["relative_path"]),
        "YOLO_CHECKPOINT_SHA": yolo["sha256"],
        "YOLO_CLASS_MAP_SHA": class_map["sha256"],
        "YOLO_CONFIG_SHA": yolo_config["sha256"],
    }
    return "\n".join(f"{key}={value}" for key, value in values.items()) + "\n"


def _write_runtime_agent_config(
    bundle_root: Path,
    manifest: dict[str, Any],
) -> Path:
    """Materialize a movable V2 agent config using packaged artifact digests."""

    source = bundle_root / "code" / "configs" / "agent.default.json"
    if not source.is_file():
        raise BundleError(f"packaged V2 agent config is missing: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BundleError("packaged V2 agent config root must be an object")
    artifacts = manifest["artifacts"]
    executors = payload.get("executors")
    if not isinstance(executors, dict) or not isinstance(executors.get("pi05"), dict):
        raise BundleError("packaged V2 agent config has no executors.pi05 object")
    executors["pi05"].update(
        {
            "base_url": "http://127.0.0.1:8101",
            "checkpoint_sha": artifacts["pi05_checkpoint"]["sha256"],
            "norm_stats_sha": artifacts["pi05_norm_stats"]["sha256"],
        }
    )
    image_cas = payload.get("image_cas")
    telemetry = payload.get("telemetry")
    if not isinstance(image_cas, dict) or not isinstance(telemetry, dict):
        raise BundleError("packaged V2 agent config lacks image_cas or telemetry")
    image_cas["root"] = str((bundle_root / "runtime" / "cas").resolve())
    telemetry["event_jsonl_path"] = str(
        (bundle_root / "evidence" / "competition-ui" / "agent-events.jsonl").resolve()
    )
    destination = bundle_root / "runtime" / "agent.runtime.json"
    _write_json(destination, payload)
    return destination


def _write_launchers(bundle_root: Path) -> None:
    powershell = r"""$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$BundleRoot\code\scripts\build_submission_bundle.py" prepare-env --bundle-dir "$BundleRoot"
docker load -i "$BundleRoot\runtime\images\pi05-service.tar"
docker load -i "$BundleRoot\runtime\images\yolo-service.tar"
docker image inspect (Get-Content "$BundleRoot\runtime\pi05-image-name.txt" -Raw).Trim() | Out-Null
docker image inspect (Get-Content "$BundleRoot\runtime\yolo-image-name.txt" -Raw).Trim() | Out-Null
python "$BundleRoot\code\deploy\preflight.py" --env-file "$BundleRoot\runtime\.env.runtime" --phase assets --output "$BundleRoot\evidence\asset-preflight.json"
docker compose --env-file "$BundleRoot\runtime\.env.runtime" -f "$BundleRoot\code\deploy\compose.models.submission.yaml" up -d --wait
python "$BundleRoot\code\deploy\preflight.py" --env-file "$BundleRoot\runtime\.env.runtime" --phase services --output "$BundleRoot\evidence\service-preflight.json"
"""
    shell = r"""#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$BUNDLE_ROOT/code/scripts/build_submission_bundle.py" prepare-env --bundle-dir "$BUNDLE_ROOT"
docker load -i "$BUNDLE_ROOT/runtime/images/pi05-service.tar"
docker load -i "$BUNDLE_ROOT/runtime/images/yolo-service.tar"
docker image inspect "$(tr -d '\r\n' < "$BUNDLE_ROOT/runtime/pi05-image-name.txt")" >/dev/null
docker image inspect "$(tr -d '\r\n' < "$BUNDLE_ROOT/runtime/yolo-image-name.txt")" >/dev/null
python3 "$BUNDLE_ROOT/code/deploy/preflight.py" --env-file "$BUNDLE_ROOT/runtime/.env.runtime" --phase assets --output "$BUNDLE_ROOT/evidence/asset-preflight.json"
docker compose --env-file "$BUNDLE_ROOT/runtime/.env.runtime" -f "$BUNDLE_ROOT/code/deploy/compose.models.submission.yaml" up -d --wait
python3 "$BUNDLE_ROOT/code/deploy/preflight.py" --env-file "$BUNDLE_ROOT/runtime/.env.runtime" --phase services --output "$BUNDLE_ROOT/evidence/service-preflight.json"
"""
    verify_ps = r"""$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$BundleRoot\code\scripts\build_submission_bundle.py" verify --bundle-dir "$BundleRoot"
"""
    verify_sh = r"""#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$BUNDLE_ROOT/code/scripts/build_submission_bundle.py" verify --bundle-dir "$BUNDLE_ROOT"
"""
    demo_ps = r"""$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
python "$BundleRoot\code\scripts\build_submission_bundle.py" prepare-env --bundle-dir "$BundleRoot"
if ($env:ISAAC_PYTHON) {
    $IsaacPython = $env:ISAAC_PYTHON
} elseif ($env:ISAAC_SIM_ROOT) {
    $IsaacPython = Join-Path $env:ISAAC_SIM_ROOT "python.bat"
} else {
    throw "Set ISAAC_PYTHON to Isaac Sim python.bat, or set ISAAC_SIM_ROOT."
}
if (-not (Test-Path -LiteralPath $IsaacPython -PathType Leaf)) {
    throw "Isaac Sim Python was not found: $IsaacPython"
}
$DemoArgs = @(
    "$BundleRoot\code\simulation\run_v2_competition_ui.py",
    "--agent-config", "$BundleRoot\runtime\agent.runtime.json",
    "--scene-config", "$BundleRoot\code\simulation\configs\single_bin_scene_v2.json",
    "--artifact-root", "$BundleRoot\evidence\competition-ui"
)
if ($env:TASK_STATE_FACTORY) {
    $DemoArgs += @("--task-state-factory", $env:TASK_STATE_FACTORY, "--require-terminal")
}
& $IsaacPython @DemoArgs
exit $LASTEXITCODE
"""
    demo_sh = r"""#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python3 "$BUNDLE_ROOT/code/scripts/build_submission_bundle.py" prepare-env --bundle-dir "$BUNDLE_ROOT"
if [[ -n "${ISAAC_PYTHON:-}" ]]; then
  ISAAC_PYTHON_BIN="$ISAAC_PYTHON"
elif [[ -n "${ISAAC_SIM_ROOT:-}" ]]; then
  ISAAC_PYTHON_BIN="$ISAAC_SIM_ROOT/python.sh"
else
  echo "Set ISAAC_PYTHON to Isaac Sim python.sh, or set ISAAC_SIM_ROOT." >&2
  exit 2
fi
if [[ ! -x "$ISAAC_PYTHON_BIN" ]]; then
  echo "Isaac Sim Python is not executable: $ISAAC_PYTHON_BIN" >&2
  exit 2
fi
DEMO_ARGS=(
  "$BUNDLE_ROOT/code/simulation/run_v2_competition_ui.py"
  --agent-config "$BUNDLE_ROOT/runtime/agent.runtime.json"
  --scene-config "$BUNDLE_ROOT/code/simulation/configs/single_bin_scene_v2.json"
  --artifact-root "$BUNDLE_ROOT/evidence/competition-ui"
)
if [[ -n "${TASK_STATE_FACTORY:-}" ]]; then
  DEMO_ARGS+=(--task-state-factory "$TASK_STATE_FACTORY" --require-terminal)
fi
exec "$ISAAC_PYTHON_BIN" "${DEMO_ARGS[@]}"
"""
    stop_ps = r"""$ErrorActionPreference = "Stop"
$BundleRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
docker compose --env-file "$BundleRoot\runtime\.env.runtime" -f "$BundleRoot\code\deploy\compose.models.submission.yaml" down
"""
    stop_sh = r"""#!/usr/bin/env bash
set -euo pipefail
BUNDLE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
docker compose --env-file "$BUNDLE_ROOT/runtime/.env.runtime" -f "$BUNDLE_ROOT/code/deploy/compose.models.submission.yaml" down
"""
    (bundle_root / "start-models.ps1").write_text(powershell, encoding="utf-8")
    (bundle_root / "start-models.sh").write_text(shell, encoding="utf-8", newline="\n")
    (bundle_root / "verify.ps1").write_text(verify_ps, encoding="utf-8")
    (bundle_root / "verify.sh").write_text(verify_sh, encoding="utf-8", newline="\n")
    (bundle_root / "start-demo.ps1").write_text(demo_ps, encoding="utf-8")
    (bundle_root / "start-demo.sh").write_text(demo_sh, encoding="utf-8", newline="\n")
    (bundle_root / "stop-demo.ps1").write_text(stop_ps, encoding="utf-8")
    (bundle_root / "stop-demo.sh").write_text(stop_sh, encoding="utf-8", newline="\n")


def _write_start_here(bundle_root: Path, manifest: dict[str, Any]) -> None:
    total_gib = manifest["total_artifact_bytes"] / (1 << 30)
    text = f"""# XH-202607 Submission Candidate

This directory is a self-contained candidate bundle containing the tracked
source snapshot and complete model artifacts.  It is not a Git checkout.

Bundle status: `{manifest["bundle_status"]}`
Source commit: `{manifest["source"]["commit"]}`
Model artifact size: `{total_gib:.2f} GiB`

## Required host software

- NVIDIA driver compatible with the packaged CUDA/OpenPI runtime
- Docker Engine and Docker Compose v2
- NVIDIA Container Toolkit
- Isaac Sim 5.1 for the simulator process
- Python 3.10+ for verification and launch orchestration

## Verify after copying or extraction

PowerShell:

```powershell
.\\verify.ps1
```

Linux:

```bash
bash ./verify.sh
```

## Start model services and verify health

PowerShell:

```powershell
.\\start-models.ps1
```

Linux:

```bash
bash ./start-models.sh
```

The launcher regenerates absolute host paths for the current extraction
location, validates every model digest, starts pi0.5 and YOLO, and saves asset
and service health reports under `evidence/`.

The bundle includes both service images.  The launcher loads them without a
network pull, checks the expected local tags, validates every model digest,
starts the services, and verifies their `/health` identities.

## Start the visible competition window

Set the Isaac Sim installation and launch the in-app task console:

```powershell
$env:ISAAC_SIM_ROOT = "C:\\isaacsim"
.\\start-demo.ps1
```

```bash
export ISAAC_SIM_ROOT=/opt/isaacsim
bash ./start-demo.sh
```

The window accepts the three formal instructions P01-to-S11, W01-to-S14 and
BIN01-to-FINISHED01.  To enable formal terminal success, set
`TASK_STATE_FACTORY=module.path:factory` to the deployment-owned sensor
verifier before starting the demo. Without it the robot loop can run, but the
UI deliberately cannot claim that a task succeeded.

Close the competition window before stopping model services. Then run
`./stop-demo.ps1` or `bash ./stop-demo.sh`.
"""
    (bundle_root / "START_HERE.md").write_text(text, encoding="utf-8")


def _checksums_lines(bundle_root: Path) -> list[str]:
    included_roots = (bundle_root / "models", bundle_root / "runtime" / "images")
    files: list[Path] = []
    for root in included_roots:
        if root.is_dir():
            files.extend(path for path in root.rglob("*") if path.is_file())
    return [
        f"{_sha256_file(path).removeprefix('sha256:')}  "
        f"{path.relative_to(bundle_root).as_posix()}"
        for path in sorted(
            files, key=lambda item: item.relative_to(bundle_root).as_posix()
        )
    ]


def build_bundle(
    *,
    repo_root: Path,
    output_dir: Path,
    pi05_checkpoint_dir: Path,
    pi05_norm_stats: Path,
    yolo_checkpoint: Path,
    yolo_config: Path,
    pi05_image_tar: Path,
    yolo_image_tar: Path,
    pi05_image: str,
    pi05_image_digest: str,
    yolo_image: str,
    yolo_image_digest: str,
    pi05_gpu_ids: str,
    yolo_gpu_id: str,
    repo_files: Sequence[Path] | None = None,
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    output_dir = output_dir.resolve(strict=False)
    pi05_checkpoint_dir = pi05_checkpoint_dir.resolve()
    pi05_norm_stats = pi05_norm_stats.resolve()
    yolo_checkpoint = yolo_checkpoint.resolve()
    yolo_config = yolo_config.resolve()
    pi05_image_tar = pi05_image_tar.resolve()
    yolo_image_tar = yolo_image_tar.resolve()
    if output_dir.exists():
        raise BundleError(f"output directory already exists: {output_dir}")
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise BundleError("output directory must be outside the source repository")

    pi05_bytes, _ = _directory_size(pi05_checkpoint_dir)
    required = (
        pi05_bytes
        + _file_size(pi05_norm_stats)
        + _file_size(yolo_checkpoint)
        + _file_size(pi05_image_tar)
        + _file_size(yolo_image_tar)
    )
    _file_size(yolo_config)
    _require_free_space(output_dir, required)

    output_dir.mkdir(parents=True)
    code_bytes, code_files = _copy_code_snapshot(
        repo_root,
        output_dir / "code",
        repo_files=repo_files,
    )
    _copy_directory(pi05_checkpoint_dir, output_dir / "models" / "pi05" / "checkpoint")
    _copy_file(pi05_norm_stats, output_dir / "models" / "pi05" / "norm_stats.json")
    _copy_file(yolo_checkpoint, output_dir / "models" / "yolo" / "best.pt")
    _copy_file(pi05_image_tar, output_dir / "runtime" / "images" / "pi05-service.tar")
    _copy_file(yolo_image_tar, output_dir / "runtime" / "images" / "yolo-service.tar")
    for relative in (
        "runtime/cas",
        "runtime/cache/pi05",
        "runtime/cache/yolo",
        "runtime/images",
        "evidence",
    ):
        (output_dir / relative).mkdir(parents=True, exist_ok=True)

    class_map = (
        output_dir
        / "code"
        / "services"
        / "yolo"
        / "src"
        / "yolo_service"
        / "resources"
        / "class_map.single_bin_v2.json"
    )
    packaged_yolo_config = output_dir / "code" / yolo_config.relative_to(repo_root)
    if not class_map.is_file() or not packaged_yolo_config.is_file():
        raise BundleError("packaged YOLO class map or service config is missing")

    artifacts = {
        "pi05_checkpoint": _artifact_pi05(
            output_dir / "models" / "pi05" / "checkpoint", output_dir
        ).as_dict(),
        "pi05_norm_stats": _artifact_file(
            output_dir / "models" / "pi05" / "norm_stats.json", output_dir
        ).as_dict(),
        "yolo_checkpoint": _artifact_file(
            output_dir / "models" / "yolo" / "best.pt", output_dir
        ).as_dict(),
        "yolo_class_map": _artifact_file(class_map, output_dir).as_dict(),
        "yolo_config": _artifact_file(packaged_yolo_config, output_dir).as_dict(),
        "pi05_image_tar": _artifact_file(
            output_dir / "runtime" / "images" / "pi05-service.tar", output_dir
        ).as_dict(),
        "yolo_image_tar": _artifact_file(
            output_dir / "runtime" / "images" / "yolo-service.tar", output_dir
        ).as_dict(),
    }
    source = _git_snapshot(repo_root)
    source.update(
        {
            "code_tree_sha256": _tree_digest(
                output_dir / "code", b"industrial-agent-code-snapshot-v1"
            ),
            "size_bytes": code_bytes,
            "file_count": code_files,
        }
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "bundle_status": BUNDLE_STATUS,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source": source,
        "artifacts": artifacts,
        "runtime": {
            "pi05_image": pi05_image,
            "pi05_image_digest": pi05_image_digest,
            "yolo_image": yolo_image,
            "yolo_image_digest": yolo_image_digest,
            "pi05_gpu_ids": pi05_gpu_ids,
            "yolo_gpu_id": yolo_gpu_id,
            "paths": {
                "shared_cas": "runtime/cas",
                "pi05_cache": "runtime/cache/pi05",
                "yolo_cache": "runtime/cache/yolo",
            },
        },
        "total_artifact_bytes": sum(
            item["size_bytes"]
            for name, item in artifacts.items()
            if name
            in {
                "pi05_checkpoint",
                "pi05_norm_stats",
                "yolo_checkpoint",
                "pi05_image_tar",
                "yolo_image_tar",
            }
        ),
    }
    _write_json(output_dir / "manifests" / "bundle-manifest.json", manifest)
    checksums = _checksums_lines(output_dir)
    (output_dir / "manifests" / "SHA256SUMS.txt").write_text(
        "\n".join(checksums) + "\n", encoding="utf-8"
    )
    _write_launchers(output_dir)
    _write_start_here(output_dir, manifest)
    (output_dir / "runtime" / "pi05-image-name.txt").write_text(
        pi05_image + "\n", encoding="utf-8"
    )
    (output_dir / "runtime" / "yolo-image-name.txt").write_text(
        yolo_image + "\n", encoding="utf-8"
    )
    prepare_runtime_env(output_dir)
    return manifest


def _load_manifest(bundle_root: Path) -> dict[str, Any]:
    path = bundle_root / "manifests" / "bundle-manifest.json"
    if not path.is_file():
        raise BundleError(f"bundle manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != SCHEMA_VERSION:
        raise BundleError("unsupported or malformed bundle manifest")
    return payload


def prepare_runtime_env(bundle_dir: Path) -> Path:
    bundle_root = bundle_dir.resolve()
    manifest = _load_manifest(bundle_root)
    for relative in manifest["runtime"]["paths"].values():
        (bundle_root / relative).mkdir(parents=True, exist_ok=True)
    path = bundle_root / "runtime" / ".env.runtime"
    path.write_text(_runtime_env(bundle_root, manifest), encoding="utf-8")
    _write_runtime_agent_config(bundle_root, manifest)
    return path


def verify_bundle(bundle_dir: Path) -> dict[str, Any]:
    bundle_root = bundle_dir.resolve()
    manifest = _load_manifest(bundle_root)
    artifacts = manifest["artifacts"]
    checks: list[dict[str, str]] = []

    def record(name: str, actual: str, expected: str) -> None:
        checks.append(
            {
                "name": name,
                "status": "PASS" if actual == expected else "FAIL",
                "expected": expected,
                "actual": actual,
            }
        )

    pi05_path = bundle_root / artifacts["pi05_checkpoint"]["relative_path"]
    record(
        "pi05_checkpoint",
        _pi05_directory_digest(pi05_path),
        artifacts["pi05_checkpoint"]["sha256"],
    )
    for name in (
        "pi05_norm_stats",
        "yolo_checkpoint",
        "yolo_class_map",
        "yolo_config",
        "pi05_image_tar",
        "yolo_image_tar",
    ):
        item = artifacts[name]
        record(name, _sha256_file(bundle_root / item["relative_path"]), item["sha256"])
    record(
        "code_snapshot",
        _tree_digest(bundle_root / "code", b"industrial-agent-code-snapshot-v1"),
        manifest["source"]["code_tree_sha256"],
    )
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    report = {"schema_version": SCHEMA_VERSION, "status": status, "checks": checks}
    _write_json(bundle_root / "evidence" / "bundle-verification.json", report)
    return report


def _digest_argument(value: str) -> str:
    normalized = value.strip().casefold()
    if not normalized.startswith("sha256:") or len(normalized) != 71:
        raise argparse.ArgumentTypeError("digest must be sha256:<64 lowercase hex>")
    try:
        int(normalized.removeprefix("sha256:"), 16)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("digest contains non-hex characters") from exc
    if normalized == "sha256:" + "0" * 64:
        raise argparse.ArgumentTypeError("digest cannot be all zeroes")
    return normalized


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="copy all files into a new bundle")
    build.add_argument("--repo-root", type=Path, default=PROJECT_ROOT)
    build.add_argument("--output-dir", type=Path, required=True)
    build.add_argument("--pi05-checkpoint-dir", type=Path, required=True)
    build.add_argument("--pi05-norm-stats", type=Path, required=True)
    build.add_argument("--yolo-checkpoint", type=Path, required=True)
    build.add_argument("--pi05-image-tar", type=Path, required=True)
    build.add_argument("--yolo-image-tar", type=Path, required=True)
    build.add_argument(
        "--yolo-config",
        type=Path,
        default=PROJECT_ROOT / "configs" / "yolo.service-manual994.json",
    )
    build.add_argument("--pi05-image", default=DEFAULT_PI05_IMAGE)
    build.add_argument("--pi05-image-digest", type=_digest_argument, required=True)
    build.add_argument("--yolo-image", default=DEFAULT_YOLO_IMAGE)
    build.add_argument("--yolo-image-digest", type=_digest_argument, required=True)
    build.add_argument("--pi05-gpu-ids", default="0")
    build.add_argument("--yolo-gpu-id", default="1")

    verify = subparsers.add_parser("verify", help="rehash an existing bundle")
    verify.add_argument("--bundle-dir", type=Path, required=True)
    prepare = subparsers.add_parser(
        "prepare-env", help="regenerate absolute runtime paths after moving a bundle"
    )
    prepare.add_argument("--bundle-dir", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "build":
            manifest = build_bundle(
                repo_root=args.repo_root,
                output_dir=args.output_dir,
                pi05_checkpoint_dir=args.pi05_checkpoint_dir,
                pi05_norm_stats=args.pi05_norm_stats,
                yolo_checkpoint=args.yolo_checkpoint,
                yolo_config=args.yolo_config,
                pi05_image_tar=args.pi05_image_tar,
                yolo_image_tar=args.yolo_image_tar,
                pi05_image=args.pi05_image,
                pi05_image_digest=args.pi05_image_digest,
                yolo_image=args.yolo_image,
                yolo_image_digest=args.yolo_image_digest,
                pi05_gpu_ids=args.pi05_gpu_ids,
                yolo_gpu_id=args.yolo_gpu_id,
            )
            result: dict[str, Any] = {
                "status": "PASS",
                "bundle_dir": str(args.output_dir.resolve()),
                "manifest": manifest,
            }
        elif args.command == "verify":
            result = verify_bundle(args.bundle_dir)
        else:
            path = prepare_runtime_env(args.bundle_dir)
            result = {"status": "PASS", "runtime_env": str(path)}
    except (BundleError, OSError, ValueError, json.JSONDecodeError) as exc:
        result = {"status": "FAIL", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
