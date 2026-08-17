"""Pure-Python contract and mass-budget checks for the V2 manual scene."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG_PATH = SCRIPT_DIR / "configs" / "single_bin_scene_v2.json"

EXPECTED_PARTS = {
    "P01": ("shaft", "A", "upright", (-0.90, 0.20)),
    "P02": ("shaft", "A", "upright", (-0.80, 0.20)),
    "P03": ("shaft", "B", "inverted", (-0.65, 0.20)),
    "P04": ("shaft", "B", "inverted", (-0.55, 0.20)),
    "N01": ("nut", "C", "flat", (-0.90, 0.00)),
    "N02": ("nut", "C", "flat", (-0.80, 0.00)),
    "W01": ("wrench", "D", "flat_y", (-0.65, 0.00)),
    "W02": ("wrench", "D", "flat_y", (-0.55, 0.00)),
}

EXPECTED_SLOTS = {
    "S11": ("P01", (-0.1125, 0.0550), "shaft"),
    "S12": ("P03", (-0.0375, 0.0550), "shaft"),
    "S13": ("N01", (0.0375, 0.0550), "nut"),
    "S14": ("W01", (0.1125, 0.0550), "wrench_y"),
    "S21": ("P02", (-0.1125, -0.0550), "shaft"),
    "S22": ("P04", (-0.0375, -0.0550), "shaft"),
    "S23": ("N02", (0.0375, -0.0550), "nut"),
    "S24": ("W02", (0.1125, -0.0550), "wrench_y"),
}

EXPECTED_HOME = [
    0.01199996,
    -0.56927347,
    0.00000009,
    -2.81087494,
    0.00000669,
    3.03692675,
    0.741,
]

EXPECTED_STATIONS = {
    "PACK_STATION": (-0.35, 0.2, 0.785),
    "HANDOFF_CENTER": (0.0, 0.0, 0.785),
    "FINISHED_01": (0.70, 0.10, 0.785),
}

EXPECTED_CAMERAS = {
    "CAM_A_TOP": (-0.60, -0.02, 1.90),
    "CAM_HANDOFF": (0.0, -0.35, 1.60),
    "CAM_B_TOP": (0.45, -0.02, 1.90),
}

EXPECTED_CAMERA_LOOK_AT = {
    "CAM_A_TOP": (-0.55, 0.08, 0.78),
    "CAM_HANDOFF": (0.0, 0.03, 0.82),
    "CAM_B_TOP": (0.35, 0.08, 0.78),
}


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"V2 scene config does not exist: {source}") from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"V2 scene config is not valid JSON: {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("V2 scene config root must be an object")
    return payload


def _id_map(items: Any, label: str, errors: list[str]) -> dict[str, Mapping[str, Any]]:
    if not isinstance(items, list):
        errors.append(f"{label} must be a list")
        return {}
    output: dict[str, Mapping[str, Any]] = {}
    for index, item in enumerate(items):
        if not isinstance(item, Mapping):
            errors.append(f"{label}[{index}] must be an object")
            continue
        identifier = item.get("id")
        if not isinstance(identifier, str) or not identifier:
            errors.append(f"{label}[{index}].id must be a non-empty string")
            continue
        if identifier in output:
            errors.append(f"{label} contains duplicate id {identifier!r}")
        output[identifier] = item
    return output


def _numbers(value: Any, length: int) -> list[float] | None:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != length
    ):
        return None
    try:
        converted = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return converted if all(math.isfinite(item) for item in converted) else None


def _expect_close(
    errors: list[str], label: str, actual: Any, expected: Sequence[float]
) -> None:
    values = _numbers(actual, len(expected))
    if values is None or any(
        not math.isclose(value, target, abs_tol=1e-9)
        for value, target in zip(values, expected, strict=True)
    ):
        errors.append(f"{label} must equal {list(expected)!r}; got {actual!r}")


def part_mass_kg(part: Mapping[str, Any]) -> float:
    geometry = part.get("geometry", {})
    if not isinstance(geometry, Mapping):
        raise ValueError(f"{part.get('id', 'part')}.geometry must be an object")
    mass = float(geometry["mass_kg"])
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"{part.get('id', 'part')} mass must be positive")
    return mass


def mass_budget(config: Mapping[str, Any]) -> dict[str, Any]:
    parts = _id_map(config.get("parts"), "parts", [])
    slots = _id_map(config.get("bin", {}).get("slots"), "bin.slots", [])
    empty_mass = float(config["bin"]["mass_kg"])
    part_mass = sum(part_mass_kg(part) for part in parts.values())
    loaded_mass = empty_mass + part_mass
    moment_x = 0.0
    moment_y = 0.0
    for slot in slots.values():
        part = parts[str(slot["part_id"])]
        center = _numbers(slot["center_local_m"], 3)
        if center is None:
            raise ValueError(f"slot {slot['id']} has an invalid center")
        mass = part_mass_kg(part)
        moment_x += mass * center[0]
        moment_y += mass * center[1]
    loaded_com = [moment_x / loaded_mass, moment_y / loaded_mass, 0.0]
    projection_error = math.hypot(loaded_com[0], loaded_com[1])
    return {
        "empty_bin_mass_kg": empty_mass,
        "parts_mass_kg": part_mass,
        "planned_loaded_mass_kg": loaded_mass,
        "planned_loaded_com_local_m": loaded_com,
        "carry_tcp_projection_error_m": projection_error,
    }


def validate_config(config: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    if config.get("schema_version") != "2.0":
        errors.append("schema_version must equal '2.0'")
    if config.get("scene_id") != "single_bin_manual_industrial_v2":
        errors.append("scene_id must equal 'single_bin_manual_industrial_v2'")
    if config.get("units") != {
        "linear": "meter",
        "angular": "degree",
        "up_axis": "Z",
    }:
        errors.append("units must freeze meter/degree/Z-up")

    robots = _id_map(config.get("robots"), "robots", errors)
    if set(robots) != {"Arm_A", "Arm_B"}:
        errors.append("robots must contain exactly Arm_A and Arm_B")
    for arm_id, base in {
        "Arm_A": (-0.55, -0.30, 0.75),
        "Arm_B": (0.50, -0.30, 0.75),
    }.items():
        robot = robots.get(arm_id)
        if robot is None:
            continue
        _expect_close(
            errors,
            f"robots.{arm_id}.base_pose.position_m",
            robot.get("base_pose", {}).get("position_m"),
            base,
        )
        home = robot.get("home", {})
        _expect_close(
            errors,
            f"robots.{arm_id}.home.arm_joint_positions_rad",
            home.get("arm_joint_positions_rad"),
            EXPECTED_HOME,
        )
        _expect_close(
            errors,
            f"robots.{arm_id}.home.finger_joint_positions_m",
            home.get("finger_joint_positions_m"),
            (0.04, 0.04),
        )

    stations = _id_map(config.get("stations"), "stations", errors)
    if set(stations) != set(EXPECTED_STATIONS):
        errors.append("stations must preserve PACK_STATION/HANDOFF_CENTER/FINISHED_01")
    for station_id, expected in EXPECTED_STATIONS.items():
        station = stations.get(station_id)
        if station is not None:
            _expect_close(
                errors,
                f"stations.{station_id}.pose.position_m",
                station.get("pose", {}).get("position_m"),
                expected,
            )

    zones = _id_map(config.get("zones"), "zones", errors)
    if set(zones) != {"A", "B", "C", "D"}:
        errors.append("zones must contain exactly A/B/C/D")
    for zone_id in ("A", "B", "C", "D"):
        if zones.get(zone_id, {}).get("expected_part_count") != 2:
            errors.append(f"zone {zone_id} must expect two parts")

    parts = _id_map(config.get("parts"), "parts", errors)
    if set(parts) != set(EXPECTED_PARTS):
        errors.append("parts must contain exactly P01-P04, N01-N02, and W01-W02")
    for part_id, (part_type, zone_id, state, xy) in EXPECTED_PARTS.items():
        part = parts.get(part_id)
        if part is None:
            continue
        if part.get("part_type") != part_type:
            errors.append(f"parts.{part_id}.part_type must equal {part_type!r}")
        if part.get("zone_id") != zone_id:
            errors.append(f"parts.{part_id}.zone_id must equal {zone_id!r}")
        if part.get("initial_orientation_state") != state:
            errors.append(
                f"parts.{part_id}.initial_orientation_state must equal {state!r}"
            )
        position = _numbers(part.get("pose", {}).get("position_m"), 3)
        if position is None or not all(
            math.isclose(position[index], xy[index], abs_tol=1e-9) for index in (0, 1)
        ):
            errors.append(f"parts.{part_id}.pose X/Y must equal {list(xy)!r}")
        try:
            part_mass_kg(part)
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    for part_id in ("P03", "P04"):
        part = parts.get(part_id)
        if part is not None:
            _expect_close(
                errors,
                f"parts.{part_id}.pose.rpy_deg",
                part.get("pose", {}).get("rpy_deg"),
                (180.0, 0.0, 0.0),
            )

    bin_config = config.get("bin")
    if not isinstance(bin_config, Mapping):
        errors.append("bin must be an object")
        return errors
    if bin_config.get("id") != "Bin_01":
        errors.append("bin.id must equal 'Bin_01'")
    _expect_close(errors, "bin.size_m", bin_config.get("size_m"), (0.30, 0.22, 0.09))
    grid = bin_config.get("grid", {})
    if grid != {"rows": 2, "columns": 4}:
        errors.append("bin.grid must equal 2 rows x 4 columns")
    slots = _id_map(bin_config.get("slots"), "bin.slots", errors)
    if set(slots) != set(EXPECTED_SLOTS):
        errors.append("bin.slots must contain exactly S11-S14 and S21-S24")
    seen_parts: set[str] = set()
    for slot_id, (part_id, xy, profile) in EXPECTED_SLOTS.items():
        slot = slots.get(slot_id)
        if slot is None:
            continue
        if slot.get("part_id") != part_id:
            errors.append(f"bin.slots.{slot_id}.part_id must equal {part_id!r}")
        if slot.get("profile") != profile:
            errors.append(f"bin.slots.{slot_id}.profile must equal {profile!r}")
        center = _numbers(slot.get("center_local_m"), 3)
        if center is None or not all(
            math.isclose(center[index], xy[index], abs_tol=1e-9) for index in (0, 1)
        ):
            errors.append(f"bin.slots.{slot_id}.center_local_m X/Y is frozen")
        if isinstance(slot.get("part_id"), str):
            seen_parts.add(str(slot["part_id"]))
    if seen_parts != set(EXPECTED_PARTS):
        errors.append("each of the eight parts must map to exactly one slot")

    handle = bin_config.get("carry_handle", {})
    if handle.get("frame_id") != "BIN_CARRY_TCP":
        errors.append("bin.carry_handle.frame_id must equal 'BIN_CARRY_TCP'")
    if handle.get("link") != "Carry_Handle":
        errors.append("bin.carry_handle.link must equal 'Carry_Handle'")
    clear_length = handle.get("clear_grasp_length_m")
    if not isinstance(clear_length, (int, float)) or not 0.055 <= clear_length <= 0.065:
        errors.append("carry handle clear grasp length must be in [0.055, 0.065] m")

    try:
        budget = mass_budget(config)
        design_limit = float(bin_config["design_loaded_mass_limit_kg"])
        hard_limit = float(bin_config["acceptance_loaded_mass_limit_kg"])
        if budget["planned_loaded_mass_kg"] > design_limit:
            errors.append("planned loaded mass exceeds the 1.20 kg design limit")
        if design_limit > hard_limit or hard_limit > 1.5:
            errors.append("bin loaded-mass limits are invalid")
        if budget["carry_tcp_projection_error_m"] > 0.010:
            errors.append("planned loaded COM is more than 0.010 m from BIN_CARRY_TCP")
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as exc:
        errors.append(f"mass budget is invalid: {exc}")

    cameras = _id_map(config.get("cameras"), "cameras", errors)
    if set(cameras) != set(EXPECTED_CAMERAS):
        errors.append("cameras must contain exactly the three frozen IDs")
    for camera_id, expected in EXPECTED_CAMERAS.items():
        camera = cameras.get(camera_id)
        if camera is None:
            continue
        if camera.get("prim_path") != f"/World/Cameras/{camera_id}":
            errors.append(f"cameras.{camera_id}.prim_path is frozen")
        _expect_close(
            errors,
            f"cameras.{camera_id}.pose.position_m",
            camera.get("pose", {}).get("position_m"),
            expected,
        )
        _expect_close(
            errors,
            f"cameras.{camera_id}.look_at_m",
            camera.get("look_at_m"),
            EXPECTED_CAMERA_LOOK_AT[camera_id],
        )
        if camera.get("resolution_px") != [1280, 720]:
            errors.append(f"cameras.{camera_id}.resolution_px must equal 1280x720")
        if camera.get("horizontal_fov_deg") != 82.0:
            errors.append(f"cameras.{camera_id}.horizontal_fov_deg must equal 82.0")

    workflow = config.get("workflow", {})
    if workflow.get("token_sequence") != [
        "A_ONLY",
        "HANDOFF_VERIFY",
        "B_ONLY",
        "NONE",
    ]:
        errors.append("workflow.token_sequence is frozen")
    collection = config.get("collection", {})
    if collection.get("mode") != "manual_keyboard":
        errors.append("collection.mode must equal 'manual_keyboard'")
    if collection.get("effective_sample_hz") != 10:
        errors.append("collection.effective_sample_hz must equal 10")
    if collection.get("online_gt_allowed") is not False:
        errors.append("collection.online_gt_allowed must be false")
    physics = config.get("physics", {})
    expected_rates = {
        "control_frequency_hz": 60,
        "render_frequency_hz": 30,
        "model_inference_frequency_hz": 10,
    }
    for field, expected in expected_rates.items():
        if physics.get(field) != expected:
            errors.append(f"physics.{field} must equal {expected}")
    try:
        if not math.isclose(float(physics["physics_dt_s"]), 1.0 / 120.0):
            errors.append("physics.physics_dt_s must equal 1/120")
    except (KeyError, TypeError, ValueError):
        errors.append("physics.physics_dt_s must equal 1/120")
    return errors


def require_valid_config(config: Mapping[str, Any]) -> None:
    errors = validate_config(config)
    if errors:
        formatted = "\n".join(f"  - {error}" for error in errors)
        raise ValueError("V2 scene contract failed:\n" + formatted)
