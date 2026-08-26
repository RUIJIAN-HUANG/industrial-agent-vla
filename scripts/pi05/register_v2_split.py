"""Register one successful formal V2 collection in the immutable Split Registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from industrial_agent.data import DatasetSplit, SplitRegistry, SplitRegistryError


COLLECTION_TO_REGISTRY_SPLIT = {
    "train": DatasetSplit.TRAIN,
    "validation": DatasetSplit.VAL,
}


class SplitRegistrationError(ValueError):
    """A collection result that is unsafe to register."""


def _read_object(path: Path, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise SplitRegistrationError(f"{name} must be a real JSON file: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitRegistrationError(f"cannot read {name}: {exc}") from exc
    if not isinstance(value, dict):
        raise SplitRegistrationError(f"{name} must contain a JSON object")
    return value


def _validated_manifest(episode_dir: Path) -> dict[str, Any]:
    # Imported lazily so --help and lightweight unit tests do not require HDF5.
    from scripts.pi05.canonical_v2 import CanonicalV2Reader

    with CanonicalV2Reader(episode_dir) as reader:
        return dict(reader.manifest)


def _required_object(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SplitRegistrationError(f"{name} must be an object")
    return value


def register_collection_result(
    *,
    result_json: str | Path,
    registry_path: str | Path,
    split: str,
    scenario_group_id: str | None = None,
    parent_episode_id: str | None = None,
) -> dict[str, Any]:
    """Validate, add, and atomically save one immutable split assignment."""

    try:
        requested_split = DatasetSplit(split)
    except ValueError as exc:
        raise SplitRegistrationError("split must be train or val") from exc
    if requested_split not in {DatasetSplit.TRAIN, DatasetSplit.VAL}:
        raise SplitRegistrationError("formal mother trajectories must use train or val")
    if (scenario_group_id is None) == (parent_episode_id is None):
        raise SplitRegistrationError(
            "provide exactly one of scenario_group_id (mother) or "
            "parent_episode_id (derived Episode)"
        )

    result_path = Path(result_json).expanduser().resolve()
    result = _read_object(result_path, name="collection result")
    preflight = _required_object(result.get("preflight"), name="preflight")
    expected_collection_split = (
        "train" if requested_split is DatasetSplit.TRAIN else "validation"
    )
    checks = {
        "status": (result.get("status"), "PASS"),
        "outcome": (result.get("outcome"), "SUCCEEDED"),
        "preflight.split": (preflight.get("split"), expected_collection_split),
    }
    for field, (actual, expected) in checks.items():
        if actual != expected:
            raise SplitRegistrationError(
                f"{field} mismatch: expected {expected!r}, got {actual!r}"
            )

    episode_path_value = result.get("episode_path")
    if not isinstance(episode_path_value, str) or not episode_path_value:
        raise SplitRegistrationError("result.episode_path must be a non-empty string")
    episode_dir = Path(episode_path_value).expanduser().resolve()
    preflight_episode_dir = preflight.get("episode_dir")
    if not isinstance(preflight_episode_dir, str) or (
        Path(preflight_episode_dir).expanduser().resolve() != episode_dir
    ):
        raise SplitRegistrationError(
            "preflight.episode_dir does not match result.episode_path"
        )

    manifest = _validated_manifest(episode_dir)
    metadata = _required_object(manifest.get("metadata"), name="manifest.metadata")
    episode_id = metadata.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise SplitRegistrationError("manifest.metadata.episode_id is invalid")
    if episode_id != episode_dir.name or episode_id != preflight.get("episode_id"):
        raise SplitRegistrationError(
            "Episode ID mismatch between directory, result preflight, and manifest"
        )
    if metadata.get("outcome") != "SUCCEEDED":
        raise SplitRegistrationError("Canonical Episode outcome must be SUCCEEDED")
    if metadata.get("scene_seed") != preflight.get("scene_seed"):
        raise SplitRegistrationError(
            "scene_seed mismatch between result preflight and Canonical Episode"
        )

    registry_target = Path(registry_path).expanduser()
    registry = (
        SplitRegistry.load(registry_target)
        if registry_target.exists()
        else SplitRegistry()
    )
    scene_seed = metadata.get("scene_seed")
    if isinstance(scene_seed, bool) or not isinstance(scene_seed, int):
        raise SplitRegistrationError("manifest.metadata.scene_seed must be an integer")

    if parent_episode_id is not None:
        parent = registry.get_assignment(parent_episode_id)
        if parent.split is not requested_split:
            raise SplitRegistrationError(
                "derived Episode split must equal its registered parent split"
            )
        if scene_seed != parent.scene_seed:
            raise SplitRegistrationError(
                "derived Episode scene_seed must equal its registered parent scene_seed"
            )
        group = {
            "scenario_group_id": parent.scenario_group_id,
            "asset_variant": parent.asset_variant,
            "camera_seed": parent.camera_seed,
            "lighting_seed": parent.lighting_seed,
        }
    else:
        scene_id = metadata.get("scene_id")
        scene_sha = metadata.get("scene_config_sha256")
        if not isinstance(scene_id, str) or not isinstance(scene_sha, str):
            raise SplitRegistrationError("Canonical scene identity is incomplete")
        group = {
            "scenario_group_id": scenario_group_id,
            "asset_variant": f"{scene_id}@{scene_sha}",
            # The frozen scene has no separate camera/lighting randomizer; bind
            # both registry dimensions to the recorded scene seed.
            "camera_seed": scene_seed,
            "lighting_seed": scene_seed,
        }

    assignment = registry.assign_episode(
        episode_id,
        requested_split,
        scenario_group_id=str(group["scenario_group_id"]),
        scene_seed=scene_seed,
        asset_variant=str(group["asset_variant"]),
        camera_seed=int(group["camera_seed"]),
        lighting_seed=int(group["lighting_seed"]),
        parent_episode_id=parent_episode_id,
    )
    saved_path = registry.save(registry_target)
    return {
        "status": "PASS",
        "episode_id": episode_id,
        "episode_path": str(episode_dir),
        "split": assignment.split.value,
        "scenario_group_id": assignment.scenario_group_id,
        "parent_episode_id": assignment.parent_episode_id,
        "registry_path": str(saved_path),
        "registry_sha256": registry.registry_sha256,
        "assignment_count": len(registry),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate a successful formal collection result and add its Episode "
            "to the immutable Split Registry. Never edit the Registry JSON or SHA."
        )
    )
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--split", required=True, choices=("train", "val"))
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scenario-group-id")
    group.add_argument("--parent-episode-id")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        receipt = register_collection_result(
            result_json=args.result_json,
            registry_path=args.registry,
            split=args.split,
            scenario_group_id=args.scenario_group_id,
            parent_episode_id=args.parent_episode_id,
        )
    except (SplitRegistrationError, SplitRegistryError) as exc:
        raise SystemExit(f"split registration rejected: {exc}") from exc
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
