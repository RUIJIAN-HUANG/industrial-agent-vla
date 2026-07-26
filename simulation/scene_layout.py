"""Static contract and reach preflight for the frozen Isaac Sim MVP scene.

This module deliberately uses only the Python standard library.  It can run
before Isaac Sim is installed and catches accidental coordinate, recipe, and
handoff-protocol drift.  Passing this check is not an IK or collision proof.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


DEFAULT_CONFIG_PATH = (
    Path(__file__).resolve().parent / "configs" / "single_bin_scene_v1.json"
)

_TABLE = {
    "position_m": [0.0, 0.0, 0.725],
    "rpy_deg": [0.0, 0.0, 0.0],
    "size_m": [2.3, 1.1, 0.05],
    "surface_z_m": 0.75,
}
_ROBOTS = {
    "Arm_A": {
        "position_m": [-0.55, -0.3, 0.75],
        "rpy_deg": [0.0, 0.0, 90.0],
        "executor": "pi05",
        "role": "pack_and_handoff",
    },
    "Arm_B": {
        "position_m": [0.5, -0.3, 0.75],
        "rpy_deg": [0.0, 0.0, 90.0],
        "executor": "openvla_oft",
        "role": "bin_transport",
    },
}
_STATIONS = {
    "PACK_STATION": {
        "position_m": [-0.35, -0.15, 0.785],
        "rpy_deg": [0.0, 0.0, 0.0],
        "footprint_m": [0.22, 0.16],
    },
    "HANDOFF_CENTER": {
        "position_m": [0.0, 0.0, 0.785],
        "rpy_deg": [0.0, 0.0, 0.0],
        "footprint_m": [0.26, 0.2],
    },
    "FINISHED_01": {
        "position_m": [0.7, 0.1, 0.785],
        "rpy_deg": [0.0, 0.0, 0.0],
        "footprint_m": [0.26, 0.2],
    },
}
_ZONES = {
    "A": {
        "position_m": [-0.85, 0.2, 0.751],
        "footprint_m": [0.22, 0.16],
        "expected_part_count": 2,
    },
    "B": {
        "position_m": [-0.6, 0.2, 0.751],
        "footprint_m": [0.18, 0.16],
        "expected_part_count": 1,
    },
    "C": {
        "position_m": [-0.85, 0.0, 0.751],
        "footprint_m": [0.18, 0.15],
        "expected_part_count": 1,
    },
    "D": {
        "position_m": [-0.6, 0.0, 0.751],
        "footprint_m": [0.18, 0.15],
        "expected_part_count": 0,
    },
}
_PARTS = {
    "P01": {
        "zone_id": "A",
        "state": "upright",
        "position_m": [-0.9, 0.2, 0.772],
        "rpy_deg": [0.0, 0.0, 0.0],
    },
    "P02": {
        "zone_id": "A",
        "state": "inverted",
        "position_m": [-0.8, 0.2, 0.772],
        "rpy_deg": [180.0, 0.0, 0.0],
    },
    "P03": {
        "zone_id": "B",
        "state": "upright",
        "position_m": [-0.6, 0.2, 0.772],
        "rpy_deg": [0.0, 0.0, 0.0],
    },
    "P04": {
        "zone_id": "C",
        "state": "upright",
        "position_m": [-0.85, 0.0, 0.772],
        "rpy_deg": [0.0, 0.0, 0.0],
    },
}
_CAMERAS = {
    "CAM_A_TOP": {
        "position_m": [-0.65, -0.15, 1.5],
        "look_at_m": [-0.65, 0.12, 0.75],
        "consumers": ["pi05", "yolo"],
    },
    "CAM_HANDOFF": {
        "position_m": [0.0, -0.45, 1.18],
        "look_at_m": [0.0, 0.0, 0.785],
        "consumers": ["yolo", "verifier"],
    },
    "CAM_B_TOP": {
        "position_m": [0.6, -0.18, 1.45],
        "look_at_m": [0.48, 0.18, 0.75],
        "consumers": ["openvla_oft", "yolo"],
    },
}
_REACH_TARGETS = (
    ("Arm_A", "station", "PACK_STATION"),
    ("Arm_A", "part", "P01"),
    ("Arm_A", "part", "P02"),
    ("Arm_A", "part", "P03"),
    ("Arm_A", "part", "P04"),
    ("Arm_A", "station", "HANDOFF_CENTER"),
    ("Arm_B", "station", "HANDOFF_CENTER"),
    ("Arm_B", "station", "FINISHED_01"),
)


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Load a scene JSON object without requiring Isaac Sim."""

    config_path = Path(path)
    try:
        raw = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot load scene config {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("scene config root must be a JSON object")
    return raw


def _matches(actual: Any, expected: Any) -> bool:
    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        return (
            isinstance(actual, (int, float))
            and not isinstance(actual, bool)
            and math.isclose(float(actual), float(expected), abs_tol=1e-9)
        )
    if isinstance(expected, list):
        return (
            isinstance(actual, list)
            and len(actual) == len(expected)
            and all(_matches(a, e) for a, e in zip(actual, expected))
        )
    return actual == expected


def _expect(
    errors: list[str],
    label: str,
    actual: Any,
    expected: Any,
) -> None:
    if not _matches(actual, expected):
        errors.append(f"{label} must be {expected!r}; got {actual!r}")


def _id_map(
    config: Mapping[str, Any],
    key: str,
    errors: list[str],
) -> dict[str, Mapping[str, Any]]:
    raw_items = config.get(key)
    if not isinstance(raw_items, list):
        errors.append(f"{key} must be a list")
        return {}

    result: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(raw_items):
        if not isinstance(item, Mapping):
            errors.append(f"{key}[{index}] must be an object")
            continue
        item_id = item.get("id")
        if not isinstance(item_id, str) or not item_id:
            errors.append(f"{key}[{index}].id must be a non-empty string")
            continue
        if item_id in result:
            errors.append(f"{key} contains duplicate id {item_id!r}")
            continue
        result[item_id] = item
    return result


def _pose(
    item: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = item.get(key)
    return value if isinstance(value, Mapping) else {}


def _check_ids(
    errors: list[str],
    label: str,
    actual: Mapping[str, Any],
    expected: Mapping[str, Any],
) -> None:
    if set(actual) != set(expected):
        errors.append(
            f"{label} ids must be {sorted(expected)!r}; got {sorted(actual)!r}"
        )


def _planar_distance(
    first: Sequence[float],
    second: Sequence[float],
) -> float:
    return math.hypot(
        float(first[0]) - float(second[0]), float(first[1]) - float(second[1])
    )


def reach_report(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return frozen XY reach checks.

    The result is a fast layout preflight only.  It does not account for arm
    posture, joint limits, collision geometry, grasp orientation, or payload.
    """

    local_errors: list[str] = []
    robots = _id_map(config, "robots", local_errors)
    stations = _id_map(config, "stations", local_errors)
    parts = _id_map(config, "parts", local_errors)
    if local_errors:
        raise ValueError("; ".join(local_errors))

    report: list[dict[str, Any]] = []
    for robot_id, target_kind, target_id in _REACH_TARGETS:
        robot = robots.get(robot_id)
        targets = stations if target_kind == "station" else parts
        target = targets.get(target_id)
        if robot is None or target is None:
            raise ValueError(f"missing reach endpoint {robot_id} -> {target_id}")
        base_position = _pose(robot, "base_pose").get("position_m")
        target_position = _pose(target, "pose").get("position_m")
        soft_limit = robot.get("soft_work_radius_m")
        if not (
            isinstance(base_position, list)
            and len(base_position) == 3
            and isinstance(target_position, list)
            and len(target_position) == 3
            and isinstance(soft_limit, (int, float))
        ):
            raise ValueError(f"invalid reach endpoint {robot_id} -> {target_id}")
        distance = _planar_distance(base_position, target_position)
        report.append(
            {
                "robot_id": robot_id,
                "target_kind": target_kind,
                "target_id": target_id,
                "distance_m": distance,
                "soft_limit_m": float(soft_limit),
                "within_soft_limit": distance <= float(soft_limit) + 1e-9,
            }
        )
    return report


def _check_table_boundary(
    errors: list[str],
    table: Mapping[str, Any],
    label: str,
    position: Any,
    footprint: Any = (0.0, 0.0),
) -> None:
    table_pose = _pose(table, "pose").get("position_m")
    table_size = table.get("size_m")
    if not (
        isinstance(table_pose, list)
        and len(table_pose) == 3
        and isinstance(table_size, list)
        and len(table_size) == 3
        and isinstance(position, list)
        and len(position) == 3
        and isinstance(footprint, (list, tuple))
        and len(footprint) == 2
    ):
        errors.append(f"{label} cannot be checked against table boundary")
        return
    table_min_x = float(table_pose[0]) - float(table_size[0]) / 2
    table_max_x = float(table_pose[0]) + float(table_size[0]) / 2
    table_min_y = float(table_pose[1]) - float(table_size[1]) / 2
    table_max_y = float(table_pose[1]) + float(table_size[1]) / 2
    half_x = float(footprint[0]) / 2
    half_y = float(footprint[1]) / 2
    if not (
        table_min_x <= float(position[0]) - half_x
        and float(position[0]) + half_x <= table_max_x
        and table_min_y <= float(position[1]) - half_y
        and float(position[1]) + half_y <= table_max_y
    ):
        errors.append(f"{label} footprint is outside the table boundary")


def validate_scene_config(config: Mapping[str, Any]) -> list[str]:
    """Validate the frozen scene contract and return every detected error."""

    if not isinstance(config, Mapping):
        return ["scene config root must be an object"]

    errors: list[str] = []
    _expect(errors, "schema_version", config.get("schema_version"), "1.0")
    _expect(
        errors,
        "scene_id",
        config.get("scene_id"),
        "single_bin_static_handoff_v1",
    )
    _expect(
        errors,
        "units",
        config.get("units"),
        {"linear": "meter", "angular": "degree", "up_axis": "Z"},
    )

    table = config.get("table")
    if not isinstance(table, Mapping):
        errors.append("table must be an object")
        table = {}
    _expect(errors, "table.id", table.get("id"), "TABLE")
    table_pose = _pose(table, "pose")
    _expect(
        errors,
        "table.pose.position_m",
        table_pose.get("position_m"),
        _TABLE["position_m"],
    )
    _expect(
        errors,
        "table.pose.rpy_deg",
        table_pose.get("rpy_deg"),
        _TABLE["rpy_deg"],
    )
    _expect(errors, "table.size_m", table.get("size_m"), _TABLE["size_m"])
    _expect(
        errors,
        "table.surface_z_m",
        table.get("surface_z_m"),
        _TABLE["surface_z_m"],
    )

    robots = _id_map(config, "robots", errors)
    stations = _id_map(config, "stations", errors)
    zones = _id_map(config, "zones", errors)
    parts = _id_map(config, "parts", errors)
    cameras = _id_map(config, "cameras", errors)
    _check_ids(errors, "robots", robots, _ROBOTS)
    _check_ids(errors, "stations", stations, _STATIONS)
    _check_ids(errors, "zones", zones, _ZONES)
    _check_ids(errors, "parts", parts, _PARTS)
    _check_ids(errors, "cameras", cameras, _CAMERAS)

    for robot_id, expected in _ROBOTS.items():
        robot = robots.get(robot_id, {})
        pose = _pose(robot, "base_pose")
        _expect(
            errors,
            f"robots.{robot_id}.base_pose.position_m",
            pose.get("position_m"),
            expected["position_m"],
        )
        _expect(
            errors,
            f"robots.{robot_id}.base_pose.rpy_deg",
            pose.get("rpy_deg"),
            expected["rpy_deg"],
        )
        _expect(
            errors,
            f"robots.{robot_id}.executor",
            robot.get("executor"),
            expected["executor"],
        )
        _expect(
            errors,
            f"robots.{robot_id}.role",
            robot.get("role"),
            expected["role"],
        )
        _expect(
            errors,
            f"robots.{robot_id}.soft_work_radius_m",
            robot.get("soft_work_radius_m"),
            0.65,
        )

    for station_id, expected in _STATIONS.items():
        station = stations.get(station_id, {})
        pose = _pose(station, "pose")
        _expect(
            errors,
            f"stations.{station_id}.pose.position_m",
            pose.get("position_m"),
            expected["position_m"],
        )
        _expect(
            errors,
            f"stations.{station_id}.pose.rpy_deg",
            pose.get("rpy_deg"),
            expected["rpy_deg"],
        )
        _expect(
            errors,
            f"stations.{station_id}.footprint_m",
            station.get("footprint_m"),
            expected["footprint_m"],
        )

    for zone_id, expected in _ZONES.items():
        zone = zones.get(zone_id, {})
        pose = _pose(zone, "pose")
        _expect(
            errors,
            f"zones.{zone_id}.pose.position_m",
            pose.get("position_m"),
            expected["position_m"],
        )
        _expect(
            errors,
            f"zones.{zone_id}.footprint_m",
            zone.get("footprint_m"),
            expected["footprint_m"],
        )
        _expect(
            errors,
            f"zones.{zone_id}.expected_part_count",
            zone.get("expected_part_count"),
            expected["expected_part_count"],
        )

    for part_id, expected in _PARTS.items():
        part = parts.get(part_id, {})
        pose = _pose(part, "pose")
        _expect(
            errors,
            f"parts.{part_id}.zone_id",
            part.get("zone_id"),
            expected["zone_id"],
        )
        _expect(
            errors,
            f"parts.{part_id}.state",
            part.get("state"),
            expected["state"],
        )
        _expect(
            errors,
            f"parts.{part_id}.pose.position_m",
            pose.get("position_m"),
            expected["position_m"],
        )
        _expect(
            errors,
            f"parts.{part_id}.pose.rpy_deg",
            pose.get("rpy_deg"),
            expected["rpy_deg"],
        )
        geometry = part.get("geometry")
        if not isinstance(geometry, Mapping):
            errors.append(f"parts.{part_id}.geometry must be an object")
            geometry = {}
        for key, value in (
            ("type", "cylinder"),
            ("radius_m", 0.022),
            ("height_m", 0.044),
            ("mass_kg", 0.065),
        ):
            _expect(
                errors,
                f"parts.{part_id}.geometry.{key}",
                geometry.get(key),
                value,
            )

    zone_counts = Counter(
        part.get("zone_id")
        for part in parts.values()
        if isinstance(part.get("zone_id"), str)
    )
    for zone_id, expected in _ZONES.items():
        if zone_counts[zone_id] != expected["expected_part_count"]:
            errors.append(
                f"zone {zone_id} contains {zone_counts[zone_id]} parts; "
                f"expected {expected['expected_part_count']}"
            )

    bin_config = config.get("bin")
    if not isinstance(bin_config, Mapping):
        errors.append("bin must be an object")
        bin_config = {}
    _expect(errors, "bin.id", bin_config.get("id"), "Bin_01")
    _expect(
        errors,
        "bin.initial_station_id",
        bin_config.get("initial_station_id"),
        "PACK_STATION",
    )
    bin_pose = _pose(bin_config, "pose")
    _expect(
        errors,
        "bin.pose.position_m",
        bin_pose.get("position_m"),
        [-0.35, -0.15, 0.785],
    )
    _expect(
        errors,
        "bin.pose.rpy_deg",
        bin_pose.get("rpy_deg"),
        [0.0, 0.0, 0.0],
    )
    _expect(errors, "bin.size_m", bin_config.get("size_m"), [0.18, 0.12, 0.07])
    _expect(errors, "bin.grid", bin_config.get("grid"), {"rows": 2, "columns": 3})
    recipe = bin_config.get("recipe_part_ids")
    _expect(errors, "bin.recipe_part_ids", recipe, ["P01", "P02", "P03", "P04"])
    _expect(
        errors,
        "bin.empty_slots_after_pack",
        bin_config.get("empty_slots_after_pack"),
        2,
    )
    if isinstance(recipe, list):
        if len(recipe) != len(set(recipe)):
            errors.append("bin.recipe_part_ids must not contain duplicates")
        if set(recipe) != set(parts):
            errors.append("bin recipe must contain every and only declared part id")
    grid = bin_config.get("grid")
    empty_slots = bin_config.get("empty_slots_after_pack")
    if (
        isinstance(grid, Mapping)
        and isinstance(grid.get("rows"), int)
        and isinstance(grid.get("columns"), int)
        and isinstance(recipe, list)
        and isinstance(empty_slots, int)
        and grid["rows"] * grid["columns"] != len(recipe) + empty_slots
    ):
        errors.append("bin grid capacity must equal recipe size plus empty slots")

    for camera_id, expected in _CAMERAS.items():
        camera = cameras.get(camera_id, {})
        pose = _pose(camera, "pose")
        _expect(
            errors,
            f"cameras.{camera_id}.pose.position_m",
            pose.get("position_m"),
            expected["position_m"],
        )
        _expect(
            errors,
            f"cameras.{camera_id}.look_at_m",
            camera.get("look_at_m"),
            expected["look_at_m"],
        )
        _expect(
            errors,
            f"cameras.{camera_id}.resolution_px",
            camera.get("resolution_px"),
            [1280, 720],
        )
        _expect(
            errors,
            f"cameras.{camera_id}.horizontal_fov_deg",
            camera.get("horizontal_fov_deg"),
            68.0,
        )
        _expect(
            errors,
            f"cameras.{camera_id}.consumers",
            camera.get("consumers"),
            expected["consumers"],
        )

    workflow = config.get("workflow")
    if not isinstance(workflow, Mapping):
        errors.append("workflow must be an object")
        workflow = {}
    _expect(
        errors,
        "workflow.token_sequence",
        workflow.get("token_sequence"),
        ["A_ONLY", "HANDOFF_VERIFY", "B_ONLY", "NONE"],
    )
    _expect(
        errors,
        "workflow.handoff_ready_event",
        workflow.get("handoff_ready_event"),
        "handoff_ready",
    )
    _expect(
        errors,
        "workflow.handoff_owner_by_token",
        workflow.get("handoff_owner_by_token"),
        {
            "A_ONLY": "Arm_A",
            "HANDOFF_VERIFY": None,
            "B_ONLY": "Arm_B",
            "NONE": None,
        },
    )

    safety = config.get("safety")
    if not isinstance(safety, Mapping):
        errors.append("safety must be an object")
        safety = {}
    _expect(
        errors,
        "safety.handoff_zone",
        safety.get("handoff_zone"),
        {
            "x_m": [-0.16, 0.16],
            "y_m": [-0.12, 0.12],
            "z_m": [0.74, 1.15],
        },
    )

    all_entity_ids = ["TABLE", "Bin_01"]
    for collection in (robots, stations, zones, parts, cameras):
        all_entity_ids.extend(collection)
    duplicate_ids = sorted(
        entity_id for entity_id, count in Counter(all_entity_ids).items() if count > 1
    )
    if duplicate_ids:
        errors.append(f"entity ids must be globally unique: {duplicate_ids!r}")

    for robot_id, robot in robots.items():
        _check_table_boundary(
            errors,
            table,
            f"robots.{robot_id}",
            _pose(robot, "base_pose").get("position_m"),
        )
    for station_id, station in stations.items():
        _check_table_boundary(
            errors,
            table,
            f"stations.{station_id}",
            _pose(station, "pose").get("position_m"),
            station.get("footprint_m"),
        )
    for zone_id, zone in zones.items():
        _check_table_boundary(
            errors,
            table,
            f"zones.{zone_id}",
            _pose(zone, "pose").get("position_m"),
            zone.get("footprint_m"),
        )
    for part_id, part in parts.items():
        geometry = part.get("geometry", {})
        radius = geometry.get("radius_m", 0.0) if isinstance(geometry, Mapping) else 0.0
        _check_table_boundary(
            errors,
            table,
            f"parts.{part_id}",
            _pose(part, "pose").get("position_m"),
            [2 * radius, 2 * radius]
            if isinstance(radius, (int, float))
            else [0.0, 0.0],
        )
    _check_table_boundary(
        errors,
        table,
        "bin",
        bin_pose.get("position_m"),
        bin_config.get("size_m", [0.0, 0.0])[:2]
        if isinstance(bin_config.get("size_m"), list)
        else [0.0, 0.0],
    )

    try:
        reach = reach_report(config)
    except ValueError as exc:
        errors.append(f"reach report failed: {exc}")
    else:
        for item in reach:
            if not item["within_soft_limit"]:
                errors.append(
                    f"{item['robot_id']} -> {item['target_id']} planar distance "
                    f"{item['distance_m']:.3f} m exceeds "
                    f"{item['soft_limit_m']:.3f} m soft limit"
                )

    return errors


def _main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate the frozen dual-Franka single-bin scene contract."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="scene JSON path",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="emit a machine-readable report",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(f"[FAIL] {exc}")
        return 2

    errors = validate_scene_config(config)
    try:
        reach = reach_report(config)
    except ValueError:
        reach = []

    if args.json:
        print(
            json.dumps(
                {"valid": not errors, "errors": errors, "reach": reach},
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        for item in reach:
            status = "PASS" if item["within_soft_limit"] else "FAIL"
            print(
                f"[{status}] {item['robot_id']} -> {item['target_id']}: "
                f"{item['distance_m']:.3f} m / "
                f"{item['soft_limit_m']:.3f} m"
            )
        if errors:
            print(f"[FAIL] scene contract has {len(errors)} error(s):")
            for error in errors:
                print(f"  - {error}")
        else:
            print(
                "[PASS] frozen static contract is valid. "
                "Isaac Sim IK/collision validation is still required."
            )
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(_main())
