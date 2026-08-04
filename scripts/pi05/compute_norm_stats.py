"""Compute PI05 norm stats from validated Train-only data.

Canonical input is parsed exclusively by ``canonical_v1``.  LeRobot input is
opened offline and filtered through the conversion provenance manifest.  No
production state statistics can be emitted without an explicitly injected,
role-A-approved StateMapper.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from configs.pi05.train_config import OPENPI_COMMIT
from industrial_agent.data import SplitRegistry
from scripts.pi05.provenance_context import (
    NORM_STATS_SOURCE_MANIFEST_TYPE,
    ProvenanceContext,
    resolve_provenance_context,
)

try:
    from scripts.pi05.canonical_v1 import (
        StateMapper,
        load_split_registry,
        load_state_mapper,
        map_state,
        read_canonical_dataset,
        require_state_mapper,
    )
    from scripts.pi05.smoke_lerobot_loader import (
        PROVENANCE_FILENAME,
        load_provenance,
        open_offline_dataset,
        validate_provenance_manifest,
    )
except ModuleNotFoundError:  # direct ``python scripts/pi05/...`` execution
    from canonical_v1 import (  # type: ignore
        StateMapper,
        load_split_registry,
        load_state_mapper,
        map_state,
        read_canonical_dataset,
        require_state_mapper,
    )
    from smoke_lerobot_loader import (  # type: ignore
        PROVENANCE_FILENAME,
        load_provenance,
        open_offline_dataset,
        validate_provenance_manifest,
    )

logger = logging.getLogger("compute_norm_stats")
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(
        logging.Formatter(
            "[%(asctime)s][%(levelname)s][compute_norm_stats] %(message)s"
        )
    )
    logger.addHandler(_h)
logger.setLevel(logging.INFO)

try:
    from openpi.shared import normalize as _normalize  # type: ignore

    OPENPI_NORMALIZE_AVAILABLE = True
except Exception:  # pragma: no cover - local CI does not include OpenPI
    _normalize = None
    OPENPI_NORMALIZE_AVAILABLE = False

ACTION_DIM = 7
NORM_STATS_KEYS = ("state", "actions")
NORM_STATS_FILENAME = "norm_stats.json"
NORM_STATS_SOURCE_MANIFEST_FILENAME = "norm_stats_source_manifest.json"
EPS = 1e-6
MOCK_SEED = 42


@dataclass(frozen=True)
class NormStats:
    mean: np.ndarray
    std: np.ndarray
    q01: np.ndarray
    q99: np.ndarray


@dataclass(frozen=True)
class LoadedDataset:
    state: np.ndarray
    actions: np.ndarray
    mask: np.ndarray | None
    source_manifest: dict[str, Any]

    def as_dict(self) -> dict[str, np.ndarray]:
        result = {"state": self.state, "actions": self.actions}
        if self.mask is not None:
            result["mask"] = self.mask
        return result


def compute_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_stats(
    arr: np.ndarray,
    mask: np.ndarray | None = None,
    key: str = "",
) -> dict[str, np.ndarray]:
    """Compute finite statistics over strict float64[N,D] input."""

    array = np.asarray(arr, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"[{key}] expected 2-D [N,D], got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"[{key}] contains NaN or Infinity")
    if mask is not None:
        valid = np.asarray(mask)
        if valid.ndim != 1 or valid.shape[0] != array.shape[0]:
            raise ValueError(
                f"[{key}] mask length/shape mismatch: mask={valid.shape} "
                f"samples={array.shape[0]}"
            )
        if valid.dtype != np.bool_:
            raise ValueError(f"[{key}] mask must have boolean dtype")
        array = array[valid]
    if array.shape[0] < 2:
        raise ValueError(f"[{key}] requires at least two valid samples")
    mean = array.mean(axis=0)
    std = np.maximum(array.std(axis=0, ddof=0), EPS)
    result = {
        "mean": mean,
        "std": std,
        "q01": np.quantile(array, 0.01, axis=0),
        "q99": np.quantile(array, 0.99, axis=0),
        "min": array.min(axis=0),
        "max": array.max(axis=0),
    }
    if not all(np.all(np.isfinite(value)) for value in result.values()):
        raise ValueError(f"[{key}] computed non-finite statistics")
    return result


def build_norm_stats(
    stats_by_key: Mapping[str, Mapping[str, np.ndarray]],
) -> dict[str, NormStats]:
    return {
        key: NormStats(
            mean=np.asarray(value["mean"], dtype=np.float64),
            std=np.asarray(value["std"], dtype=np.float64),
            q01=np.asarray(value["q01"], dtype=np.float64),
            q99=np.asarray(value["q99"], dtype=np.float64),
        )
        for key, value in stats_by_key.items()
    }


def serialize_norm_stats(norm_stats: Mapping[str, NormStats]) -> str:
    if OPENPI_NORMALIZE_AVAILABLE and _normalize is not None:
        official = {
            key: _normalize.NormStats(
                mean=value.mean,
                std=value.std,
                q01=value.q01,
                q99=value.q99,
            )
            for key, value in norm_stats.items()
        }
        return _normalize.serialize_json(official)
    payload = {
        "norm_stats": {
            key: {
                "mean": value.mean.tolist(),
                "std": value.std.tolist(),
                "q01": value.q01.tolist(),
                "q99": value.q99.tolist(),
            }
            for key, value in norm_stats.items()
        }
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)


def _validate_serialized_stats(
    serialized: str,
    *,
    state_dim: int,
    action_dim: int,
) -> None:
    payload = json.loads(serialized)
    values = payload.get("norm_stats")
    if not isinstance(values, dict) or set(values) != set(NORM_STATS_KEYS):
        raise ValueError("norm stats must contain exactly state and actions")
    for key, expected_dim in (("state", state_dim), ("actions", action_dim)):
        entry = values.get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"norm stats {key!r} must be an object")
        if set(entry) != {"mean", "std", "q01", "q99"}:
            raise ValueError(f"norm stats {key!r} has unexpected fields")
        for field in ("mean", "std", "q01", "q99"):
            array = np.asarray(entry[field], dtype=np.float64)
            if array.shape != (expected_dim,) or not np.all(np.isfinite(array)):
                raise ValueError(
                    f"norm stats {key}.{field} must be finite [{expected_dim}]"
                )
        if np.any(np.asarray(entry["std"], dtype=np.float64) <= 0):
            raise ValueError(f"norm stats {key}.std must be positive")


def save_norm_stats(output_path: Path, norm_stats: Mapping[str, NormStats]) -> None:
    """Compatibility writer that validates complete stats before atomic replace."""

    serialized = serialize_norm_stats(norm_stats)
    state_dim = len(norm_stats["state"].mean)
    action_dim = len(norm_stats["actions"].mean)
    _validate_serialized_stats(serialized, state_dim=state_dim, action_dim=action_dim)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(serialized + "\n", encoding="utf-8")
    temporary.replace(output_path)


def validate_dimensions(
    data: Mapping[str, np.ndarray],
    *,
    expected_state_dim: int | None = None,
    expected_action_dim: int = ACTION_DIM,
) -> None:
    state = np.asarray(data["state"])
    actions = np.asarray(data["actions"])
    if state.ndim != 2 or actions.ndim != 2:
        raise ValueError(
            f"state/actions must be 2-D; got state={state.shape} actions={actions.shape}"
        )
    if state.shape[0] != actions.shape[0]:
        raise ValueError(
            f"state/action row count mismatch: {state.shape[0]} != {actions.shape[0]}"
        )
    if state.shape[0] == 0:
        raise ValueError("state/actions contain no Train samples")
    if expected_state_dim is not None and state.shape[1] != expected_state_dim:
        raise ValueError(
            f"state dimension mismatch: {state.shape[1]} != {expected_state_dim}"
        )
    if actions.shape[1] != expected_action_dim:
        raise ValueError(
            f"action dimension mismatch: {actions.shape[1]} != {expected_action_dim}"
        )
    if not np.all(np.isfinite(state)) or not np.all(np.isfinite(actions)):
        raise ValueError("state/actions contain NaN or Infinity")
    mask = data.get("mask")
    if mask is not None:
        valid = np.asarray(mask)
        if valid.ndim != 1 or valid.shape[0] != state.shape[0]:
            raise ValueError(
                f"mask length/shape mismatch: mask={valid.shape} rows={state.shape[0]}"
            )
        if valid.dtype != np.bool_:
            raise ValueError("mask must have boolean dtype")


def _load_canonical(
    path: Path,
    *,
    mapper: StateMapper,
    split_registry: SplitRegistry,
) -> LoadedDataset:
    episodes = read_canonical_dataset(path, split_registry=split_registry)
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    sources: list[dict[str, Any]] = []
    excluded = {"non_train_episodes": 0, "ineligible_episodes": 0, "invalid_steps": 0}
    for episode in episodes:
        if episode.split != "train":
            excluded["non_train_episodes"] += 1
            continue
        if not episode.eligible_for_imitation:
            excluded["ineligible_episodes"] += 1
            continue
        selected = episode.training_steps
        excluded["invalid_steps"] += len(episode.steps) - len(selected)
        for step in selected:
            states.append(map_state(mapper, episode, step))
            actions.append(step.action_7d.copy())
        if selected:
            sources.append(
                {
                    "canonical_episode_id": episode.episode_id,
                    "split": episode.split,
                    "source_action_sequence_ids": [
                        step.action_sequence_id for step in selected
                    ],
                    "source_physics_ticks": [step.physics_tick for step in selected],
                    "source_structure_sha256": episode.structure_sha256,
                    "source_hdf5_sha256": episode.hdf5_sha256,
                    "source_split_registry_sha256": episode.split_registry_sha256,
                    "source_recorder_git_sha": episode.recorder_git_sha,
                }
            )
    if not states:
        raise ValueError("no eligible valid_for_training Train samples were selected")
    return LoadedDataset(
        state=np.stack(states).astype(np.float32, copy=False),
        actions=np.stack(actions).astype(np.float32, copy=False),
        mask=None,
        source_manifest={
            "input_format": "canonical_v1",
            "input_path": str(path.resolve()),
            "split": "train",
            "split_registry_sha256": split_registry.registry_sha256,
            "sources": sources,
            "excluded": excluded,
        },
    )


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _validate_conversion_provenance(
    provenance: Mapping[str, Any],
    *,
    repo_id: str,
    mapper: StateMapper,
    split_registry: SplitRegistry,
    provenance_context: ProvenanceContext,
) -> list[dict[str, Any]]:
    episodes = validate_provenance_manifest(
        provenance,
        expected_repo_id=repo_id,
        expected_provenance_context=provenance_context,
    )
    mapper_info = provenance.get("state_mapper")
    if not isinstance(mapper_info, dict):
        raise ValueError("LeRobot provenance is missing state_mapper")
    if (
        mapper_info.get("name") != mapper.name
        or mapper_info.get("state_dim") != mapper.state_dim
        or mapper_info.get("approved_for_production")
        is not mapper.approved_for_production
        or mapper_info.get("version") != str(getattr(mapper, "version", "unspecified"))
    ):
        raise ValueError(
            "LeRobot provenance StateMapper does not match the injected mapper"
        )
    expected_registry_sha = split_registry.registry_sha256.split(":", 1)[-1]
    if provenance.get("source_split_registry_sha256") != expected_registry_sha:
        raise ValueError(
            "LeRobot provenance Split Registry SHA does not match the supplied registry"
        )
    return episodes


def _load_lerobot(
    path: Path,
    *,
    repo_id: str,
    mapper: StateMapper,
    split_registry: SplitRegistry,
    provenance_context: ProvenanceContext,
    manifest_path: Path | None,
) -> LoadedDataset:
    provenance_path = manifest_path or path / PROVENANCE_FILENAME
    provenance = load_provenance(provenance_path)
    episodes = _validate_conversion_provenance(
        provenance,
        repo_id=repo_id,
        mapper=mapper,
        split_registry=split_registry,
        provenance_context=provenance_context,
    )
    dataset = open_offline_dataset(path, repo_id)
    expected_total = sum(int(item["step_count"]) for item in episodes)
    if len(dataset) != expected_total:
        raise ValueError(
            f"LeRobot/provenance frame count mismatch: {len(dataset)} != {expected_total}"
        )

    train_indices: list[int] = []
    sources: list[dict[str, Any]] = []
    offset = 0
    for item in episodes:
        if not isinstance(item, dict):
            raise ValueError("LeRobot provenance episode entry must be an object")
        count = int(item["step_count"])
        for frame_index in range(count):
            dataset_index = offset + frame_index
            frame = dataset[dataset_index]
            if not isinstance(frame, Mapping):
                raise TypeError(f"LeRobot frame {dataset_index} is not a mapping")
            if frame.get("task") != item["instruction"]:
                raise ValueError(
                    f"LeRobot frame {dataset_index} task does not match provenance"
                )
            for key, expected in (
                ("episode_index", int(item["lerobot_episode_index"])),
                ("frame_index", frame_index),
            ):
                if key not in frame:
                    raise ValueError(
                        f"LeRobot frame {dataset_index} is missing provenance key {key}"
                    )
                scalar = _to_numpy(frame[key])
                if scalar.size != 1 or int(scalar.reshape(-1)[0]) != expected:
                    raise ValueError(
                        f"LeRobot frame {dataset_index} {key} does not match provenance"
                    )
        if item.get("canonical_split") == "train":
            train_indices.extend(range(offset, offset + count))
            sources.append(item)
        offset += count
    if not train_indices:
        raise ValueError("LeRobot provenance contains no Train Split frames")

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    masks: list[bool] = []
    mask_presence: list[bool] = []
    for index in train_indices:
        frame = dataset[index]
        if not isinstance(frame, Mapping):
            raise TypeError(f"LeRobot frame {index} is not a mapping")
        state = _to_numpy(frame["state"])
        action = _to_numpy(frame["actions"])
        if state.dtype != np.float32:
            raise ValueError(
                f"LeRobot frame {index} state dtype must be float32, got {state.dtype}"
            )
        if action.dtype != np.float32:
            raise ValueError(
                f"LeRobot frame {index} action dtype must be float32, got {action.dtype}"
            )
        if state.shape != (int(mapper.state_dim),):
            raise ValueError(
                f"LeRobot frame {index} state shape must be ({int(mapper.state_dim)},), "
                f"got {state.shape}"
            )
        if action.shape != (ACTION_DIM,):
            raise ValueError(
                f"LeRobot frame {index} action shape must be ({ACTION_DIM},), "
                f"got {action.shape}"
            )
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(action)):
            raise ValueError(
                f"LeRobot frame {index} state/actions contain NaN or Infinity"
            )
        states.append(state)
        actions.append(action)
        has_mask = "mask" in frame
        mask_presence.append(has_mask)
        if has_mask:
            mask = _to_numpy(frame["mask"])
            if mask.size != 1:
                raise ValueError(f"LeRobot frame {index} mask must be scalar")
            if mask.dtype != np.bool_:
                raise ValueError(
                    f"LeRobot frame {index} mask dtype must be bool, got {mask.dtype}"
                )
            masks.append(bool(mask.reshape(-1)[0]))
    if any(mask_presence) and not all(mask_presence):
        raise ValueError("LeRobot mask is missing from a subset of Train frames")

    return LoadedDataset(
        state=np.stack(states),
        actions=np.stack(actions),
        mask=np.asarray(masks, dtype=bool) if masks else None,
        source_manifest={
            "input_format": "lerobot",
            "input_path": str(path.resolve()),
            "repo_id": repo_id,
            "split": "train",
            "split_registry_sha256": split_registry.registry_sha256,
            "conversion_manifest_path": str(provenance_path.resolve()),
            "conversion_manifest_sha256": compute_sha256(provenance_path),
            "conversion_producer": provenance["producer"],
            "sources": sources,
            "excluded": {
                "non_train_episodes": len(episodes) - len(sources),
            },
        },
    )


def load_dataset(
    path: Path,
    *,
    input_format: str,
    state_mapper: StateMapper,
    split_registry: SplitRegistry,
    provenance_context: ProvenanceContext,
    production: bool = True,
    repo_id: str | None = None,
    manifest_path: Path | None = None,
) -> LoadedDataset:
    """Load one explicit format without guessing or legacy fallback."""

    if not isinstance(split_registry, SplitRegistry):
        raise TypeError("split_registry must be a verified SplitRegistry")
    if not isinstance(provenance_context, ProvenanceContext):
        raise TypeError("provenance_context must be a verified ProvenanceContext")
    mapper = require_state_mapper(state_mapper, production=production)
    if input_format == "canonical-v1":
        return _load_canonical(
            path,
            mapper=mapper,
            split_registry=split_registry,
        )
    if input_format == "lerobot":
        if not repo_id:
            raise ValueError("repo_id is required for LeRobot input")
        return _load_lerobot(
            path,
            repo_id=repo_id,
            mapper=mapper,
            split_registry=split_registry,
            provenance_context=provenance_context,
            manifest_path=manifest_path,
        )
    raise ValueError(
        "input_format must be explicitly canonical-v1 or lerobot; legacy, npz, "
        "parquet and hdf5 auto-detection are forbidden"
    )


def calculate_norm_stats(
    loaded: LoadedDataset,
    *,
    state_dim: int,
) -> tuple[dict[str, NormStats], dict[str, dict[str, np.ndarray]]]:
    data = loaded.as_dict()
    validate_dimensions(data, expected_state_dim=state_dim)
    stats_by_key = {
        key: compute_stats(data[key], mask=data.get("mask"), key=key)
        for key in NORM_STATS_KEYS
    }
    norm_stats = build_norm_stats(stats_by_key)
    serialized = serialize_norm_stats(norm_stats)
    _validate_serialized_stats(serialized, state_dim=state_dim, action_dim=ACTION_DIM)
    return norm_stats, stats_by_key


def write_norm_stats_bundle(
    *,
    output_path: Path,
    norm_stats: Mapping[str, NormStats],
    loaded: LoadedDataset,
    mapper: StateMapper,
    provenance_context: ProvenanceContext,
) -> tuple[str, Path, str]:
    """QA first, then atomically publish stats and their source manifest."""

    validate_dimensions(loaded.as_dict(), expected_state_dim=int(mapper.state_dim))
    serialized = serialize_norm_stats(norm_stats) + "\n"
    _validate_serialized_stats(
        serialized, state_dim=int(mapper.state_dim), action_dim=ACTION_DIM
    )
    stats_sha = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    source_manifest = {
        "schema_version": "1.0",
        "manifest_type": NORM_STATS_SOURCE_MANIFEST_TYPE,
        "producer": provenance_context.as_manifest(),
        "state_mapper": {
            "name": mapper.name,
            "state_dim": int(mapper.state_dim),
            "approved_for_production": bool(mapper.approved_for_production),
            "version": str(getattr(mapper, "version", "unspecified")),
        },
        "counts": {
            "state_rows": int(loaded.state.shape[0]),
            "action_rows": int(loaded.actions.shape[0]),
        },
        "norm_stats_sha256": stats_sha,
        "source": loaded.source_manifest,
    }
    output_path = output_path.resolve()
    manifest_path = output_path.with_name(NORM_STATS_SOURCE_MANIFEST_FILENAME)
    if output_path == manifest_path:
        raise ValueError(
            "norm stats output path collides with the source manifest path"
        )
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError(
            "refusing to overwrite an existing norm stats artifact: "
            f"output={output_path} manifest={manifest_path}"
        )
    manifest_text = (
        json.dumps(source_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    manifest_sha = hashlib.sha256(manifest_text.encode("utf-8")).hexdigest()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    stats_tmp = output_path.with_name(f".{output_path.name}.{token}.tmp")
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{token}.tmp")
    published: list[Path] = []
    try:
        stats_tmp.write_text(serialized, encoding="utf-8")
        manifest_tmp.write_text(manifest_text, encoding="utf-8")
        stats_tmp.replace(output_path)
        published.append(output_path)
        manifest_tmp.replace(manifest_path)
        published.append(manifest_path)
    except Exception as exc:
        cleanup_errors: list[str] = []
        for temporary in (stats_tmp, manifest_tmp):
            try:
                temporary.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(f"temporary {temporary}: {cleanup_exc}")
        for artifact in reversed(published):
            try:
                artifact.unlink(missing_ok=True)
            except OSError as cleanup_exc:
                cleanup_errors.append(f"published {artifact}: {cleanup_exc}")
        if cleanup_errors:
            logger.error(
                "norm-stats publication and cleanup failed: original=%r cleanup=%s",
                exc,
                cleanup_errors,
            )
            raise RuntimeError(
                "norm-stats publication failed and cleanup failed: "
                f"original={exc!r} cleanup={cleanup_errors}"
            ) from exc
        raise
    return stats_sha, manifest_path, manifest_sha


def print_qa_report(
    stats_by_key: Mapping[str, Mapping[str, np.ndarray]], quiet: bool
) -> None:
    if quiet:
        return
    for key in NORM_STATS_KEYS:
        values = stats_by_key[key]
        print(f"[{key}] dim={values['mean'].shape[0]}")
        for index in range(values["mean"].shape[0]):
            print(
                f"  {index}: mean={values['mean'][index]:.6f} "
                f"std={values['std'][index]:.6f} "
                f"q01={values['q01'][index]:.6f} "
                f"q99={values['q99'][index]:.6f}"
            )


def generate_mock_data(state_dim: int, action_dim: int) -> dict[str, np.ndarray]:
    """Test helper only; it is not exposed by the production CLI."""

    rng = np.random.default_rng(MOCK_SEED)
    return {
        "state": rng.normal(size=(100, state_dim)).astype(np.float32),
        "actions": rng.normal(size=(100, action_dim)).astype(np.float32),
        "mask": np.ones(100, dtype=bool),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute Train-only PI05 norm stats with strict provenance"
    )
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument(
        "--input-format", required=True, choices=("canonical-v1", "lerobot")
    )
    parser.add_argument("--state-mapper", required=True)
    parser.add_argument("--repo-id", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--split-registry", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--openpi-commit", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.quiet:
        logger.setLevel(logging.WARNING)
    try:
        if not OPENPI_NORMALIZE_AVAILABLE or _normalize is None:
            raise RuntimeError(
                "openpi.shared.normalize is required for production norm-stats publication"
            )
        mapper = load_state_mapper(args.state_mapper, production=True)
        split_registry = load_split_registry(args.split_registry)
        provenance_context = resolve_provenance_context(
            repo_root=args.project_root,
            openpi_commit=args.openpi_commit,
            expected_openpi_commit=OPENPI_COMMIT,
        )
        loaded = load_dataset(
            Path(args.dataset_path),
            input_format=args.input_format,
            state_mapper=mapper,
            split_registry=split_registry,
            provenance_context=provenance_context,
            production=True,
            repo_id=args.repo_id,
            manifest_path=Path(args.manifest) if args.manifest else None,
        )
        norm_stats, stats_by_key = calculate_norm_stats(
            loaded, state_dim=int(mapper.state_dim)
        )
        stats_sha, manifest_path, manifest_sha = write_norm_stats_bundle(
            output_path=Path(args.output_path),
            norm_stats=norm_stats,
            loaded=loaded,
            mapper=mapper,
            provenance_context=provenance_context,
        )
    except Exception as exc:
        logger.error("norm stats blocked: %s", exc)
        return 1
    print_qa_report(stats_by_key, args.quiet)
    print(
        json.dumps(
            {
                "status": "ok",
                "split": "train",
                "norm_stats_sha256": stats_sha,
                "source_manifest": str(manifest_path),
                "source_manifest_sha256": manifest_sha,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
