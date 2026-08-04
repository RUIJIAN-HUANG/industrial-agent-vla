"""Convert framework-validated Canonical HDF5 Episodes to LeRobot.

Input is exactly ``episode.h5 + structure.json`` plus a verified external
Split Registry.  Images remain raw uint8 1280x720 RGB; OpenPI owns
resize-with-pad.  Publication stays offline, staged, and atomic.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.pi05.train_config import OPENPI_COMMIT
from industrial_agent.data import SplitRegistry
from industrial_agent.sync_contract import MODEL_INFERENCE_HZ
from scripts.pi05.provenance_context import (
    LEROBOT_PROVENANCE_MANIFEST_TYPE,
    ProvenanceContext,
    resolve_provenance_context,
)

try:
    from scripts.pi05.canonical_v1 import (
        CanonicalEpisode,
        CanonicalStep,
        CanonicalV1Error,
        StateMapper,
        find_episode_dirs,
        load_rgb_image,
        load_split_registry,
        load_state_mapper,
        map_state,
        read_canonical_dataset,
        read_canonical_episode,
        require_state_mapper,
    )
    from scripts.pi05.smoke_lerobot_loader import (
        PROVENANCE_FILENAME,
        load_provenance,
        open_offline_dataset,
        validate_dataset_instance,
        validate_provenance_manifest,
        write_provenance_checksum,
    )
except ModuleNotFoundError:  # direct ``python scripts/pi05/...`` execution
    from canonical_v1 import (  # type: ignore
        CanonicalEpisode,
        CanonicalStep,
        CanonicalV1Error,
        StateMapper,
        find_episode_dirs,
        load_rgb_image,
        load_split_registry,
        load_state_mapper,
        map_state,
        read_canonical_dataset,
        read_canonical_episode,
        require_state_mapper,
    )
    from smoke_lerobot_loader import (  # type: ignore
        PROVENANCE_FILENAME,
        load_provenance,
        open_offline_dataset,
        validate_dataset_instance,
        validate_provenance_manifest,
        write_provenance_checksum,
    )

logger = logging.getLogger("convert_openpi")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter("[%(asctime)s][%(levelname)s][convert_openpi] %(message)s")
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

LeRobotDataset: Any = None
LEROBOT_AVAILABLE = False
LEROBOT_IMPORT_ERROR: str | None = None
try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    LEROBOT_AVAILABLE = True
except Exception as _e:  # pragma: no cover - dependency is optional in local CI
    LEROBOT_IMPORT_ERROR = str(_e)

ACTION_DIM = 7
DEFAULT_FPS: None = None
DEFAULT_STATE_DIM: None = None
DEFAULT_ROBOT_TYPE = "franka"
DEFAULT_REPO_ID = "your_team/industrial"
DEFAULT_IMAGE_HW = (720, 1280)


@dataclass(frozen=True)
class PreparedEpisode:
    episode: CanonicalEpisode
    steps: tuple[CanonicalStep, ...]
    states: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class ConversionResult:
    manifest: dict[str, Any]
    manifest_path: Path
    manifest_sha256: str
    manifest_checksum_path: Path


def load_image_as_array(
    path: Path, size: tuple[int, int] = DEFAULT_IMAGE_HW
) -> np.ndarray | None:
    """Decode an already-gated raw RGB image without resizing it."""

    if tuple(size) != DEFAULT_IMAGE_HW:
        raise ValueError(
            "conversion-stage resize is forbidden; expected raw image size (720, 1280)"
        )
    try:
        from PIL import Image

        with Image.open(path) as image:
            image.load()
            if image.mode != "RGB" or image.size != (1280, 720):
                return None
            array = np.asarray(image, dtype=np.uint8)
    except Exception:
        return None
    return np.ascontiguousarray(array) if array.shape == (720, 1280, 3) else None


def load_steps(
    episode_dir: Path,
    *,
    split_registry: SplitRegistry,
) -> tuple[CanonicalStep, ...]:
    """Compatibility entry point backed only by the shared Canonical v1 reader."""

    return read_canonical_episode(
        episode_dir,
        split_registry=split_registry,
    ).steps


def find_episodes(data_dir: Path) -> list[Path]:
    """Compatibility entry point backed by strict v1 enumeration."""

    return find_episode_dirs(data_dir)


def detect_state_dim(
    data_dir: Path,
    fallback: int | None = None,
    *,
    state_mapper: StateMapper | None = None,
    production: bool = True,
) -> int:
    """Return only an explicitly injected mapper dimension; never infer/default it."""

    del data_dir
    if fallback is not None:
        raise RuntimeError("state_dim fallback is forbidden until role A freezes state")
    mapper = require_state_mapper(state_mapper, production=production)
    return int(mapper.state_dim)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return _sha256_file(path)


def _prepare_episodes(
    episodes: Sequence[CanonicalEpisode],
    mapper: StateMapper,
) -> tuple[PreparedEpisode, ...]:
    prepared: list[PreparedEpisode] = []
    for episode in episodes:
        steps = episode.imitation_steps
        states = tuple(map_state(mapper, episode, step) for step in steps)
        prepared.append(PreparedEpisode(episode=episode, steps=steps, states=states))
    if not prepared:
        raise CanonicalV1Error(
            "dataset contains no eligible Arm_A training Episodes",
            episode_id="<dataset>",
            field="eligible_for_imitation",
        )
    return tuple(prepared)


def _validate_episode_timing(
    item: PreparedEpisode,
    *,
    fps: int,
    timestamp_tolerance_ns: int,
) -> None:
    expected_interval_ns = 1_000_000_000 / fps
    source_steps = item.episode.steps
    for previous, current in zip(source_steps, source_steps[1:]):
        actual_interval_ns = current.timestamp_ns - previous.timestamp_ns
        error_ns = abs(actual_interval_ns - expected_interval_ns)
        if error_ns > timestamp_tolerance_ns:
            raise CanonicalV1Error(
                "timestamp interval does not match the explicitly supplied FPS: "
                f"fps={fps} expected_interval_ns={expected_interval_ns:.9f} "
                f"actual_interval_ns={actual_interval_ns} "
                f"tolerance_ns={timestamp_tolerance_ns}",
                episode_id=item.episode.episode_id,
                step_index=current.step_index,
                field="timestamp_ns",
            )


def _create_dataset(
    *,
    repo_id: str,
    output_dir: Path,
    robot_type: str,
    fps: int,
    state_dim: int,
) -> Any:
    if not LEROBOT_AVAILABLE or LeRobotDataset is None:
        raise RuntimeError(f"LeRobot is unavailable: {LEROBOT_IMPORT_ERROR}")
    features = {
        "image": {
            "dtype": "image",
            "shape": (720, 1280, 3),
            "names": ["height", "width", "channel"],
        },
        "state": {
            "dtype": "float32",
            "shape": (state_dim,),
            "names": ["state"],
        },
        "actions": {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": ["actions"],
        },
    }
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=output_dir,
        robot_type=robot_type,
        fps=fps,
        features=features,
    )


def _clear_episode_buffer(dataset: Any) -> None:
    for name in ("clear_episode_buffer", "clear_episode"):
        method = getattr(dataset, name, None)
        if callable(method):
            method()
            return


def convert_canonical_to_lerobot(
    *,
    data_dir: Path,
    output_dir: Path,
    output_repo_id: str,
    fps: int,
    timestamp_tolerance_ns: int,
    state_mapper: StateMapper,
    split_registry: SplitRegistry,
    provenance_context: ProvenanceContext,
    robot_type: str = DEFAULT_ROBOT_TYPE,
    production: bool = True,
    dataset_factory: Any | None = None,
    dataset_opener: Any | None = None,
) -> ConversionResult:
    """Run the strict conversion and post-write loader/count/action gates."""

    if isinstance(fps, bool) or not isinstance(fps, int) or fps < 1:
        raise ValueError("fps must be an explicit positive integer")
    if production and fps != MODEL_INFERENCE_HZ:
        raise ValueError(
            f"production fps must equal frozen MODEL_INFERENCE_HZ={MODEL_INFERENCE_HZ}"
        )
    if (
        isinstance(timestamp_tolerance_ns, bool)
        or not isinstance(timestamp_tolerance_ns, int)
        or timestamp_tolerance_ns < 0
    ):
        raise ValueError(
            "timestamp_tolerance_ns must be an explicit non-negative integer"
        )
    if not isinstance(split_registry, SplitRegistry):
        raise TypeError("split_registry must be a verified SplitRegistry")
    if not isinstance(provenance_context, ProvenanceContext):
        raise TypeError("provenance_context must be a verified ProvenanceContext")
    mapper = require_state_mapper(state_mapper, production=production)
    episodes = read_canonical_dataset(
        data_dir,
        split_registry=split_registry,
    )
    prepared = _prepare_episodes(episodes, mapper)
    for item in prepared:
        _validate_episode_timing(
            item,
            fps=fps,
            timestamp_tolerance_ns=timestamp_tolerance_ns,
        )

    final_output_dir = output_dir.resolve()
    if final_output_dir.exists():
        raise FileExistsError(
            f"output directory already exists; refusing to overwrite: {final_output_dir}"
        )
    final_output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{final_output_dir.name}.staging-",
            dir=final_output_dir.parent,
        )
    )
    create = dataset_factory or _create_dataset
    try:
        dataset = create(
            repo_id=output_repo_id,
            output_dir=staging_dir,
            robot_type=robot_type,
            fps=fps,
            state_dim=int(mapper.state_dim),
        )

        manifest_episodes: list[dict[str, Any]] = []
        expected_actions: list[np.ndarray] = []
        expected_tasks: list[str] = []
        total_steps = 0
        for output_episode_index, item in enumerate(prepared):
            source = item.episode
            try:
                for step, state in zip(item.steps, item.states, strict=True):
                    image = load_rgb_image(step, episode_id=source.episode_id)
                    try:
                        dataset.add_frame(
                            {
                                "image": image,
                                "state": state,
                                "actions": step.action_7d.copy(),
                                "task": source.instruction,
                            }
                        )
                    except Exception as exc:
                        raise CanonicalV1Error(
                            f"LeRobot add_frame failed: {exc}",
                            episode_id=source.episode_id,
                            step_index=step.step_index,
                            field="lerobot.add_frame",
                        ) from exc
                    expected_actions.append(step.action_7d.copy())
                    expected_tasks.append(source.instruction)
                try:
                    dataset.save_episode()
                except Exception as exc:
                    raise CanonicalV1Error(
                        f"LeRobot save_episode failed: {exc}",
                        episode_id=source.episode_id,
                        field="lerobot.save_episode",
                    ) from exc
            except Exception:
                _clear_episode_buffer(dataset)
                raise

            total_steps += len(item.steps)
            manifest_episodes.append(
                {
                    "lerobot_episode_index": output_episode_index,
                    "canonical_episode_id": source.episode_id,
                    "canonical_split": source.split,
                    "robot_role": source.robot_role,
                    "instruction": source.instruction,
                    "instruction_sha256": hashlib.sha256(
                        source.instruction.encode("utf-8")
                    ).hexdigest(),
                    "source_structure_sha256": source.structure_sha256,
                    "source_hdf5_sha256": source.hdf5_sha256,
                    "source_split_registry_sha256": source.split_registry_sha256,
                    "source_recorder_git_sha": source.recorder_git_sha,
                    "source_action_sequence_ids": [
                        step.action_sequence_id for step in item.steps
                    ],
                    "source_action_timestamp_ns": [
                        step.timestamp_ns for step in item.steps
                    ],
                    "source_physics_ticks": [step.physics_tick for step in item.steps],
                    "source_camera_sequence_ids": [
                        step.camera_sequence_id for step in item.steps
                    ],
                    "source_camera_timestamp_ns": [
                        step.camera_timestamp_ns for step in item.steps
                    ],
                    "source_state_sequence_ids": [
                        step.state_sequence_id for step in item.steps
                    ],
                    "source_state_timestamp_ns": [
                        step.state_timestamp_ns for step in item.steps
                    ],
                    "source_image_datasets": [
                        step.cam_a_top_relative_path for step in item.steps
                    ],
                    "source_image_sha256": [
                        step.cam_a_top_sha256 for step in item.steps
                    ],
                    "source_action_duration_s": [
                        step.action_duration_s for step in item.steps
                    ],
                    "step_count": len(item.steps),
                }
            )

        try:
            stop_image_writer = getattr(dataset, "stop_image_writer")
            if not callable(stop_image_writer):
                raise TypeError("stop_image_writer is not callable")
            stop_image_writer()
        except Exception as exc:
            raise RuntimeError(f"LeRobot writer close failed: {exc}") from exc

        opener = dataset_opener or open_offline_dataset
        dataset = None
        try:
            reopened_dataset = opener(staging_dir, output_repo_id)
        except Exception as exc:
            raise RuntimeError(f"LeRobot offline reopen failed: {exc}") from exc

        smoke = validate_dataset_instance(
            reopened_dataset,
            expected_frames=total_steps,
            expected_episodes=len(prepared),
            expected_actions=expected_actions,
            expected_tasks=expected_tasks,
            expected_state_dim=int(mapper.state_dim),
            roundtrip_samples=10,
        )
        manifest = {
            "schema_version": "1.0",
            "manifest_type": LEROBOT_PROVENANCE_MANIFEST_TYPE,
            "source_format": "canonical_hdf5_v1",
            "producer": provenance_context.as_manifest(),
            "source_split_registry_sha256": split_registry.registry_sha256.split(
                ":", 1
            )[-1],
            "repo_id": output_repo_id,
            "robot_type": robot_type,
            "fps": fps,
            "timestamp_tolerance_ns": timestamp_tolerance_ns,
            "image": {
                "camera_id": "CAM_A_TOP",
                "dtype": "uint8",
                "shape": [720, 1280, 3],
                "preprocessed": False,
                "wrist_image": None,
            },
            "state_mapper": {
                "name": mapper.name,
                "state_dim": int(mapper.state_dim),
                "approved_for_production": bool(mapper.approved_for_production),
                "version": str(getattr(mapper, "version", "unspecified")),
            },
            "counts": {
                "episodes": len(prepared),
                "steps": total_steps,
                "images": total_steps,
                "instructions": len(prepared),
                "language_frames": total_steps,
                "states": total_steps,
                "actions": total_steps,
            },
            "roundtrip": smoke,
            "episodes": manifest_episodes,
        }
        staged_manifest_path = staging_dir / PROVENANCE_FILENAME
        manifest_sha = _write_json_atomic(staged_manifest_path, manifest)
        staged_checksum_path, checksum_sha = write_provenance_checksum(
            staged_manifest_path
        )
        if checksum_sha != manifest_sha:
            raise RuntimeError("provenance checksum publication mismatch")
        validated_manifest = load_provenance(staged_manifest_path)
        validate_provenance_manifest(
            validated_manifest,
            expected_repo_id=output_repo_id,
            expected_provenance_context=provenance_context,
        )
        if final_output_dir.exists():
            raise FileExistsError(
                "output directory appeared during conversion; refusing to overwrite: "
                f"{final_output_dir}"
            )
        staging_dir.replace(final_output_dir)
    except Exception as exc:
        try:
            if staging_dir.exists():
                shutil.rmtree(staging_dir)
        except Exception as cleanup_exc:
            logger.exception(
                "conversion failed and staging cleanup failed: staging=%s",
                staging_dir,
            )
            raise RuntimeError(
                "conversion failed and staging cleanup also failed: "
                f"original={exc!r} cleanup={cleanup_exc!r}"
            ) from exc
        raise

    return ConversionResult(
        manifest=manifest,
        manifest_path=final_output_dir / PROVENANCE_FILENAME,
        manifest_sha256=manifest_sha,
        manifest_checksum_path=final_output_dir / staged_checksum_path.name,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert strict Canonical v1 PI05 Episodes to LeRobot",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--split-registry", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output_repo_id", default=DEFAULT_REPO_ID)
    parser.add_argument("--fps", type=int, required=True)
    parser.add_argument("--timestamp-tolerance-ns", type=int, required=True)
    parser.add_argument(
        "--state-mapper",
        required=True,
        help="Explicit approved mapper in module:attribute form; no default exists",
    )
    parser.add_argument("--robot_type", default=DEFAULT_ROBOT_TYPE)
    parser.add_argument("--push_to_hub", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.push_to_hub:
        print(
            "ERROR: this data-gate batch is offline-only; --push_to_hub is disabled",
            file=sys.stderr,
        )
        return 2
    try:
        mapper = load_state_mapper(args.state_mapper, production=True)
        split_registry = load_split_registry(args.split_registry)
        provenance_context = resolve_provenance_context(
            repo_root=args.project_root,
            openpi_commit=args.openpi_commit,
            expected_openpi_commit=OPENPI_COMMIT,
        )
        result = convert_canonical_to_lerobot(
            data_dir=Path(args.data_dir),
            output_dir=Path(args.output_dir).resolve(),
            output_repo_id=args.output_repo_id,
            fps=args.fps,
            timestamp_tolerance_ns=args.timestamp_tolerance_ns,
            state_mapper=mapper,
            split_registry=split_registry,
            provenance_context=provenance_context,
            robot_type=args.robot_type,
            production=True,
        )
    except Exception as exc:
        logger.error("conversion failed: %s", exc)
        return 1
    print(
        json.dumps(
            {
                "status": "ok",
                "manifest": str(result.manifest_path),
                "manifest_sha256": result.manifest_sha256,
                "counts": result.manifest["counts"],
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
