"""Generate and finalize deterministic Canonical V2 replay batches.

The planner is intentionally Isaac-free.  It validates one successful source
Episode, materializes one immutable JSON configuration per trajectory, and
writes the exact PowerShell command sequence needed to execute and finalize
the batch.  Finalization admits only successful, byte-for-byte planned and
non-duplicate action trajectories into the training-ready manifest.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = SCRIPT_DIR.parent
SOURCE_DIR = REPOSITORY_ROOT / "src"
for _import_path in (REPOSITORY_ROOT, SOURCE_DIR):
    if str(_import_path) not in sys.path:
        sys.path.insert(0, str(_import_path))

from industrial_agent.data.recorder_v2 import V2_TASK_INSTRUCTIONS
from scripts.pi05.canonical_v2 import CanonicalV2Reader
from simulation.run_v2_keyboard_collection import (
    _diversify_replay_actions,
    _replay_task_actions_from_rows,
)


BATCH_SCHEMA_VERSION = "1.0"
CONFIG_SCHEMA_VERSION = "1.0"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_SHA_FILENAME = "manifest.sha256"
COMMANDS_FILENAME = "commands.ps1"
DEFAULT_SCENE_CONFIG = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"


class ReplayBatchError(ValueError):
    """A fail-closed replay batch validation error."""


@dataclass(frozen=True)
class SourceEpisode:
    path: Path
    episode_id: str
    task_id: str
    instruction: str
    scene_config_sha256: str
    hdf5_sha256: str
    actions: tuple[Any, ...]
    arm_ids: tuple[str, ...]


@dataclass(frozen=True)
class VariantSpec:
    profile: str
    seed: int
    variant: int
    lift_mm: float | None
    final_y_offset_mm: float
    final_z_offset_mm: float


def _canonical_json(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return "sha256:" + sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: Any) -> str:
    encoded = _canonical_json(payload)
    _write_bytes_atomic(path, encoded)
    return _sha256_bytes(encoded)


def action_sha256(actions: Iterable[Any]) -> str:
    """Hash exact float32 actions and durations with an unambiguous envelope."""

    digest = sha256(b"industrial-agent-v2-replay-actions-v1\0")
    count = 0
    for action in actions:
        values = np.asarray(action.values, dtype="<f4")
        if values.shape != (7,) or not np.all(np.isfinite(values)):
            raise ReplayBatchError("trajectory actions must be finite 7-D values")
        duration_ms = int(action.duration_ms)
        if duration_ms <= 0:
            raise ReplayBatchError("trajectory action duration must be positive")
        digest.update(values.tobytes(order="C"))
        digest.update(duration_ms.to_bytes(8, "little", signed=False))
        count += 1
    if count == 0:
        raise ReplayBatchError("trajectory must contain at least one task action")
    digest.update(count.to_bytes(8, "little", signed=False))
    return "sha256:" + digest.hexdigest()


def _validate_source_metadata(metadata: dict[str, Any]) -> tuple[str, str]:
    if metadata.get("outcome") != "SUCCEEDED":
        raise ReplayBatchError("source episode must have outcome SUCCEEDED")
    task_id = metadata.get("task_id")
    instruction = metadata.get("instruction")
    if not isinstance(task_id, str) or V2_TASK_INSTRUCTIONS.get(task_id) != instruction:
        raise ReplayBatchError(
            "source episode has an unsupported V2 task/instruction pair"
        )
    return task_id, str(instruction)


def load_source_episode(
    episode_dir: str | Path,
    *,
    expected_scene_config_sha256: str,
) -> SourceEpisode:
    path = Path(episode_dir).expanduser().resolve()
    with CanonicalV2Reader(path) as reader:
        metadata = reader.manifest["metadata"]
        task_id, instruction = _validate_source_metadata(metadata)
        actual_scene_sha = str(metadata.get("scene_config_sha256"))
        if actual_scene_sha != expected_scene_config_sha256:
            raise ReplayBatchError(
                "source episode scene config SHA-256 does not match the batch scene config"
            )
        rows = list(reader.iter_action_7d())
        actions = tuple(_replay_task_actions_from_rows(rows))
        stored_arm_ids = tuple(
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in reader.h5["actions/arm_id"][:]
        )
        arm_ids = stored_arm_ids[: len(actions)]
        if len(stored_arm_ids) != len(rows):
            raise ReplayBatchError("source action identity count is inconsistent")
        return SourceEpisode(
            path=path,
            episode_id=reader.episode_id,
            task_id=task_id,
            instruction=instruction,
            scene_config_sha256=actual_scene_sha,
            hdf5_sha256=str(reader.manifest["storage"]["sha256"]),
            actions=actions,
            arm_ids=arm_ids,
        )


def load_existing_action_hashes(
    roots: Sequence[str | Path],
) -> dict[str, tuple[str, ...]]:
    """Hash every successful Canonical V2 episode below the supplied roots."""

    by_hash: dict[str, list[str]] = {}
    visited: set[Path] = set()
    for root_value in roots:
        root = Path(root_value).expanduser().resolve()
        if not root.is_dir():
            raise ReplayBatchError(f"deduplication root does not exist: {root}")
        for structure_path in sorted(root.rglob("structure.json")):
            episode_path = structure_path.parent.resolve()
            if episode_path in visited or not (episode_path / "episode.h5").is_file():
                continue
            visited.add(episode_path)
            try:
                with CanonicalV2Reader(episode_path) as reader:
                    metadata = reader.manifest["metadata"]
                    if metadata.get("outcome") != "SUCCEEDED":
                        continue
                    rows = list(reader.iter_action_7d())
                actions = _replay_task_actions_from_rows(rows)
                digest = action_sha256(actions)
            except (KeyError, OSError, TypeError, ValueError) as exc:
                raise ReplayBatchError(
                    f"invalid episode below deduplication root: {episode_path}: {exc}"
                ) from exc
            by_hash.setdefault(digest, []).append(str(episode_path))
    return {digest: tuple(paths) for digest, paths in sorted(by_hash.items())}


def _centered_offset(index: int, *, step_mm: float = 2.0) -> float:
    if index == 0:
        return 0.0
    magnitude = ((index + 1) // 2) * step_mm
    return -magnitude if index % 2 else magnitude


def build_variant_specs(
    *,
    base_seed: int,
    diverse_low_count: int,
    approach_curve_count: int,
) -> tuple[VariantSpec, ...]:
    """Return a deterministic, bounded variant schedule for both profiles."""

    if (
        isinstance(diverse_low_count, bool)
        or not isinstance(diverse_low_count, int)
        or not 0 <= diverse_low_count <= 21
    ):
        raise ReplayBatchError("diverse_low_count must be in [0, 21]")
    if (
        isinstance(approach_curve_count, bool)
        or not isinstance(approach_curve_count, int)
        or not 0 <= approach_curve_count <= 4
    ):
        raise ReplayBatchError("approach_curve_count must be in [0, 4]")
    if (
        isinstance(base_seed, bool)
        or not isinstance(base_seed, int)
        or base_seed < 0
        or base_seed + diverse_low_count + approach_curve_count > 2**32
    ):
        raise ReplayBatchError("base_seed range must fit unsigned 32-bit seeds")
    if diverse_low_count + approach_curve_count == 0:
        raise ReplayBatchError("batch must request at least one trajectory")

    specs: list[VariantSpec] = []
    for index in range(diverse_low_count):
        specs.append(
            VariantSpec(
                profile="diverse_low",
                seed=base_seed + index,
                variant=0,
                lift_mm=None,
                final_y_offset_mm=_centered_offset(index),
                final_z_offset_mm=0.0,
            )
        )
    for index in range(approach_curve_count):
        specs.append(
            VariantSpec(
                profile="approach_curve",
                seed=base_seed + diverse_low_count + index,
                variant=index + 1,
                lift_mm=None,
                final_y_offset_mm=0.0,
                final_z_offset_mm=0.0,
            )
        )
    return tuple(specs)


def _resolved_lift_mm(spec: VariantSpec) -> float | None:
    if spec.profile != "diverse_low":
        return None
    if spec.lift_mm is not None:
        return spec.lift_mm
    rng = np.random.default_rng(int(spec.seed))
    return float(rng.choice(np.asarray([0.2, 0.3, 0.5])))


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _command_line(argv: Sequence[str]) -> str:
    return "& " + " ".join(_powershell_quote(item) for item in argv)


def _episode_id(task_id: str, profile: str, seed: int, ordinal: int) -> str:
    task = task_id.lower().replace("_", "-")
    abbreviation = "dl" if profile == "diverse_low" else "ac"
    return f"{task}-{abbreviation}-s{seed:010d}-{ordinal:04d}"


def _command_argv(
    *,
    python_command: str,
    scene_config: Path,
    source: SourceEpisode,
    config: dict[str, Any],
    frozen_collection_sha: str | None,
    openpi_root: Path | None,
) -> list[str]:
    run = config["run"]
    trajectory = config["trajectory"]
    argv = [
        python_command,
        "simulation/run_v2_keyboard_collection.py",
        "--config",
        str(scene_config),
        "--episode-root",
        run["episode_root"],
        "--cas-root",
        run["cas_root"],
        "--artifact-dir",
        run["artifact_dir"],
        "--output-scene",
        run["output_scene"],
        "--episode-id",
        run["episode_id"],
        "--task-id",
        source.task_id,
        "--instruction",
        source.instruction,
        "--scene-seed",
        str(run["scene_seed"]),
        "--split",
        run["split"],
        "--replay-episode",
        str(source.path),
        "--trajectory-profile",
        trajectory["profile"],
        "--trajectory-seed",
        str(trajectory["seed"]),
    ]
    if trajectory["variant"]:
        argv.extend(["--trajectory-variant", str(trajectory["variant"])])
    if trajectory["lift_mm"] is not None:
        argv.extend(["--lift-mm", str(trajectory["lift_mm"])])
    if trajectory["final_offset_mm"]["y"]:
        argv.extend(["--final-y-offset-mm", str(trajectory["final_offset_mm"]["y"])])
    if trajectory["final_offset_mm"]["z"]:
        argv.extend(["--final-z-offset-mm", str(trajectory["final_offset_mm"]["z"])])
    if run["split"] in {"train", "validation"}:
        if frozen_collection_sha is None or openpi_root is None:
            raise ReplayBatchError(
                "train/validation batches require frozen_collection_sha and openpi_root"
            )
        argv.extend(
            [
                "--frozen-collection-sha",
                frozen_collection_sha,
                "--expected-scene-config-sha256",
                source.scene_config_sha256,
                "--openpi-root",
                str(openpi_root),
            ]
        )
    return argv


def generate_batch(
    *,
    source_episode: str | Path,
    output_dir: str | Path,
    episode_root: str | Path,
    cas_root: str | Path,
    artifact_root: str | Path,
    scene_output_root: str | Path,
    scene_config: str | Path = DEFAULT_SCENE_CONFIG,
    split: str = "practice",
    base_seed: int = 1000,
    diverse_low_count: int = 3,
    approach_curve_count: int = 4,
    python_command: str = "python",
    frozen_collection_sha: str | None = None,
    openpi_root: str | Path | None = None,
    reject_against_roots: Sequence[str | Path] = (),
) -> Path:
    """Materialize a deterministic replay plan and return its manifest path."""

    if split not in {"practice", "test", "train", "validation"}:
        raise ReplayBatchError("split must be practice, test, train, or validation")
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise ReplayBatchError(f"output directory must be empty: {destination}")
    scene_path = Path(scene_config).expanduser().resolve()
    if not scene_path.is_file():
        raise ReplayBatchError(f"scene config does not exist: {scene_path}")
    scene_sha = _sha256_file(scene_path)
    source = load_source_episode(
        source_episode,
        expected_scene_config_sha256=scene_sha,
    )
    specs = build_variant_specs(
        base_seed=base_seed,
        diverse_low_count=diverse_low_count,
        approach_curve_count=approach_curve_count,
    )
    resolved_episode_root = Path(episode_root).expanduser().resolve()
    resolved_cas_root = Path(cas_root).expanduser().resolve()
    resolved_artifact_root = Path(artifact_root).expanduser().resolve()
    resolved_scene_output_root = Path(scene_output_root).expanduser().resolve()
    resolved_openpi_root = (
        Path(openpi_root).expanduser().resolve() if openpi_root is not None else None
    )

    existing_hashes = load_existing_action_hashes(reject_against_roots)
    entries: list[dict[str, Any]] = []
    seen_hashes: set[str] = {action_sha256(source.actions), *existing_hashes}
    profile_ordinals = {"diverse_low": 0, "approach_curve": 0}
    for spec in specs:
        varied_actions = _diversify_replay_actions(
            list(source.actions),
            profile=spec.profile,
            seed=spec.seed,
            variant=spec.variant,
            lift_mm=spec.lift_mm,
            final_y_offset_mm=spec.final_y_offset_mm,
            final_z_offset_mm=spec.final_z_offset_mm,
            arm_ids=list(source.arm_ids),
        )
        varied_sha = action_sha256(varied_actions)
        if varied_sha in seen_hashes:
            existing_paths = existing_hashes.get(varied_sha, ())
            provenance = (
                f"; existing episodes: {', '.join(existing_paths)}"
                if existing_paths
                else ""
            )
            raise ReplayBatchError(
                f"duplicate trajectory rejected for {spec.profile} seed={spec.seed}"
                f"{provenance}"
            )
        seen_hashes.add(varied_sha)
        profile_ordinals[spec.profile] += 1
        episode_id = _episode_id(
            source.task_id,
            spec.profile,
            spec.seed,
            profile_ordinals[spec.profile],
        )
        config_name = f"{episode_id}.json"
        config_path = destination / "configs" / config_name
        config: dict[str, Any] = {
            "schema_version": CONFIG_SCHEMA_VERSION,
            "source": {
                "episode_id": source.episode_id,
                "episode_path": str(source.path),
                "hdf5_sha256": source.hdf5_sha256,
                "scene_config_sha256": source.scene_config_sha256,
            },
            "trajectory": {
                "profile": spec.profile,
                "seed": spec.seed,
                "variant": spec.variant,
                "lift_mm": spec.lift_mm,
                "resolved_lift_mm": _resolved_lift_mm(spec),
                "final_offset_mm": {
                    "y": spec.final_y_offset_mm,
                    "z": spec.final_z_offset_mm,
                },
                "planned_action_sha256": varied_sha,
            },
            "run": {
                "episode_id": episode_id,
                "task_id": source.task_id,
                "instruction": source.instruction,
                "scene_seed": spec.seed,
                "split": split,
                "episode_root": str(resolved_episode_root),
                "episode_dir": str(resolved_episode_root / episode_id),
                "cas_root": str(resolved_cas_root),
                "artifact_dir": str(resolved_artifact_root / episode_id),
                "output_scene": str(resolved_scene_output_root / f"{episode_id}.usda"),
            },
        }
        argv = _command_argv(
            python_command=python_command,
            scene_config=scene_path,
            source=source,
            config=config,
            frozen_collection_sha=frozen_collection_sha,
            openpi_root=resolved_openpi_root,
        )
        config["command_argv"] = argv
        config_sha = _write_json_atomic(config_path, config)
        entries.append(
            {
                "config_file": str(config_path.relative_to(destination)).replace(
                    "\\", "/"
                ),
                "config_sha256": config_sha,
                "episode_id": episode_id,
                "profile": spec.profile,
                "seed": spec.seed,
                "variant": spec.variant,
                "resolved_lift_mm": _resolved_lift_mm(spec),
                "final_offset_mm": {
                    "y": spec.final_y_offset_mm,
                    "z": spec.final_z_offset_mm,
                },
                "planned_action_sha256": varied_sha,
                "status": "PLANNED",
            }
        )

    manifest: dict[str, Any] = {
        "schema_version": BATCH_SCHEMA_VERSION,
        "status": "PLANNED",
        "training_ready": False,
        "source": {
            "episode_id": source.episode_id,
            "episode_path": str(source.path),
            "task_id": source.task_id,
            "instruction": source.instruction,
            "hdf5_sha256": source.hdf5_sha256,
            "scene_config_file": str(scene_path),
            "scene_config_sha256": source.scene_config_sha256,
        },
        "batch": {
            "split": split,
            "base_seed": base_seed,
            "diverse_low_count": diverse_low_count,
            "approach_curve_count": approach_curve_count,
        },
        "deduplication": {
            "reference_roots": [
                str(Path(root).expanduser().resolve()) for root in reject_against_roots
            ],
            "existing_unique_action_hashes": len(existing_hashes),
            "existing_episode_count": sum(
                len(paths) for paths in existing_hashes.values()
            ),
        },
        "counts": {
            "planned": len(entries),
            "accepted": 0,
            "rejected": 0,
        },
        "trajectories": entries,
    }
    manifest_path = destination / MANIFEST_FILENAME
    manifest_sha = _write_json_atomic(manifest_path, manifest)
    _write_bytes_atomic(
        destination / MANIFEST_SHA_FILENAME,
        f"{manifest_sha.removeprefix('sha256:')}  {MANIFEST_FILENAME}\n".encode(
            "ascii"
        ),
    )
    commands = [
        _command_line(
            json.loads(
                (destination / entry["config_file"]).read_text(encoding="utf-8")
            )["command_argv"]
        )
        for entry in entries
    ]
    commands.append(
        _command_line(
            [
                python_command,
                "simulation/generate_v2_replay_batch.py",
                "finalize",
                "--manifest",
                str(manifest_path),
            ]
        )
    )
    command_lines = ["$ErrorActionPreference = 'Stop'"]
    for command in commands:
        command_lines.extend(
            [command, "if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }"]
        )
    _write_bytes_atomic(
        destination / COMMANDS_FILENAME,
        ("\n\n".join(command_lines) + "\n").encode("utf-8"),
    )
    return manifest_path


def _read_verified_manifest(path: Path) -> dict[str, Any]:
    manifest_path = path.expanduser().resolve()
    checksum_path = manifest_path.with_name(MANIFEST_SHA_FILENAME)
    if not manifest_path.is_file() or not checksum_path.is_file():
        raise ReplayBatchError("manifest or manifest SHA-256 sidecar is missing")
    parts = checksum_path.read_text(encoding="ascii").strip().split()
    if len(parts) != 2 or parts[1].lstrip("*") != manifest_path.name:
        raise ReplayBatchError("manifest SHA-256 sidecar is malformed")
    actual = _sha256_file(manifest_path).removeprefix("sha256:")
    if parts[0].lower() != actual:
        raise ReplayBatchError("manifest SHA-256 mismatch")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        not isinstance(payload, dict)
        or payload.get("schema_version") != BATCH_SCHEMA_VERSION
    ):
        raise ReplayBatchError("unsupported replay batch manifest")
    return payload


def finalize_batch(manifest: str | Path) -> dict[str, Any]:
    """Verify outputs and exclude every failed, changed, or duplicate trajectory."""

    manifest_path = Path(manifest).expanduser().resolve()
    payload = _read_verified_manifest(manifest_path)
    root = manifest_path.parent
    accepted_hashes: set[str] = set()
    accepted = 0
    rejected = 0
    for entry in payload.get("trajectories", []):
        reasons: list[str] = []
        config_path = root / str(entry.get("config_file", ""))
        if not config_path.is_file():
            reasons.append("CONFIG_MISSING")
            config = None
        elif _sha256_file(config_path) != entry.get("config_sha256"):
            reasons.append("CONFIG_SHA256_MISMATCH")
            config = None
        else:
            config = json.loads(config_path.read_text(encoding="utf-8"))

        actual_action_sha: str | None = None
        hdf5_sha: str | None = None
        if config is not None:
            episode_dir = Path(config["run"]["episode_dir"])
            try:
                with CanonicalV2Reader(episode_dir) as reader:
                    metadata = reader.manifest["metadata"]
                    if metadata.get("outcome") != "SUCCEEDED":
                        reasons.append("EPISODE_FAILED")
                    if metadata.get("task_id") != payload["source"]["task_id"]:
                        reasons.append("TASK_ID_MISMATCH")
                    if (
                        metadata.get("scene_config_sha256")
                        != payload["source"]["scene_config_sha256"]
                    ):
                        reasons.append("SCENE_CONFIG_SHA256_MISMATCH")
                    rows = list(reader.iter_action_7d())
                    task_actions = _replay_task_actions_from_rows(rows)
                    actual_action_sha = action_sha256(task_actions)
                    hdf5_sha = str(reader.manifest["storage"]["sha256"])
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
                reasons.append(f"EPISODE_INVALID:{type(exc).__name__}")

        if actual_action_sha is not None:
            if actual_action_sha != entry.get("planned_action_sha256"):
                reasons.append("ACTION_SHA256_MISMATCH")
            if actual_action_sha in accepted_hashes:
                reasons.append("DUPLICATE_TRAJECTORY")
        if reasons:
            entry["status"] = "REJECTED"
            entry["rejection_reasons"] = reasons
            rejected += 1
        else:
            entry["status"] = "ACCEPTED"
            entry.pop("rejection_reasons", None)
            entry["episode_hdf5_sha256"] = hdf5_sha
            entry["actual_action_sha256"] = actual_action_sha
            accepted_hashes.add(str(actual_action_sha))
            accepted += 1

    planned = len(payload.get("trajectories", []))
    payload["counts"] = {
        "planned": planned,
        "accepted": accepted,
        "rejected": rejected,
    }
    payload["status"] = (
        "ACCEPTED" if planned > 0 and accepted == planned else "REJECTED"
    )
    payload["training_ready"] = bool(
        payload["status"] == "ACCEPTED" and payload["batch"]["split"] == "train"
    )
    manifest_sha = _write_json_atomic(manifest_path, payload)
    _write_bytes_atomic(
        manifest_path.with_name(MANIFEST_SHA_FILENAME),
        f"{manifest_sha.removeprefix('sha256:')}  {manifest_path.name}\n".encode(
            "ascii"
        ),
    )
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    plan = subparsers.add_parser(
        "plan", help="Generate configs, manifest, and commands"
    )
    plan.add_argument("--source-episode", type=Path, required=True)
    plan.add_argument("--output-dir", type=Path, required=True)
    plan.add_argument("--episode-root", type=Path, required=True)
    plan.add_argument("--cas-root", type=Path, required=True)
    plan.add_argument("--artifact-root", type=Path, required=True)
    plan.add_argument("--scene-output-root", type=Path, required=True)
    plan.add_argument("--scene-config", type=Path, default=DEFAULT_SCENE_CONFIG)
    plan.add_argument(
        "--split",
        choices=("practice", "test", "train", "validation"),
        default="practice",
    )
    plan.add_argument("--base-seed", type=int, default=1000)
    plan.add_argument("--diverse-low-count", type=int, default=3)
    plan.add_argument("--approach-curve-count", type=int, default=4)
    plan.add_argument("--python-command", default="python")
    plan.add_argument("--frozen-collection-sha")
    plan.add_argument("--openpi-root", type=Path)
    plan.add_argument(
        "--reject-against-root",
        action="append",
        default=[],
        type=Path,
        help="Reject planned actions already present below this episode root; repeatable",
    )

    finalize = subparsers.add_parser("finalize", help="Validate completed outputs")
    finalize.add_argument("--manifest", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "plan":
            manifest_path = generate_batch(
                source_episode=args.source_episode,
                output_dir=args.output_dir,
                episode_root=args.episode_root,
                cas_root=args.cas_root,
                artifact_root=args.artifact_root,
                scene_output_root=args.scene_output_root,
                scene_config=args.scene_config,
                split=args.split,
                base_seed=args.base_seed,
                diverse_low_count=args.diverse_low_count,
                approach_curve_count=args.approach_curve_count,
                python_command=args.python_command,
                frozen_collection_sha=args.frozen_collection_sha,
                openpi_root=args.openpi_root,
                reject_against_roots=args.reject_against_root,
            )
            print(json.dumps({"status": "PLANNED", "manifest": str(manifest_path)}))
            return 0
        result = finalize_batch(args.manifest)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "ACCEPTED" else 1
    except (ReplayBatchError, OSError, ValueError, KeyError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ReplayBatchError",
    "SourceEpisode",
    "VariantSpec",
    "action_sha256",
    "build_variant_specs",
    "finalize_batch",
    "generate_batch",
    "load_source_episode",
    "load_existing_action_hashes",
    "main",
]
